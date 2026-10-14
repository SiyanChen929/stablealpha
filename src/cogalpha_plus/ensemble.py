"""Composite-alpha controls and optional LightGBM backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .metrics import daily_ic, summarize_ic


def normalize_factor_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank-normalize each factor using same-date information."""

    if not isinstance(features.index, pd.MultiIndex) or "date" not in features.index.names:
        raise ValueError("features must use MultiIndex(date, symbol)")
    output = pd.DataFrame(index=features.index)
    for name in features.columns:
        values = pd.to_numeric(features[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ranks = values.groupby(level="date", sort=False).rank(pct=True, method="average")
        output[name] = (ranks - 0.5) * 2.0
    return output


def equal_date_sample_weights(index: pd.MultiIndex) -> np.ndarray:
    dates = index.get_level_values("date")
    counts = pd.Series(1.0, index=index).groupby(level="date").transform("sum")
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights / weights.mean()


def _align_xy(features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    aligned = features.join(labels.rename("__label__"), how="inner")
    aligned = aligned.loc[aligned["__label__"].replace([np.inf, -np.inf], np.nan).notna()]
    y = aligned.pop("__label__").astype(float)
    x = normalize_factor_matrix(aligned).fillna(0.0)
    return x, y


def rank_ic_mean(prediction: pd.Series, labels: pd.Series, min_names: int = 20) -> float:
    ic, _ = daily_ic(prediction, labels, method="spearman", min_names=min_names)
    return float(ic.mean())


@dataclass(frozen=True)
class CompositeFitResult:
    backend: str
    parameter: float | str
    validation_rank_ic: float
    equal_weight_rank_ic: float
    incremental_rank_ic: float
    advances_over_control: bool


class CompositeAlphaModel:
    """Fit Ridge control or the paper-compatible LightGBM candidate.

    LightGBM is never silently replaced. A faithful run requesting it fails
    clearly when the optional dependency is absent. Ridge remains the mandatory
    control and can be used for synthetic/plumbing tests.
    """

    def __init__(
        self,
        *,
        backend: str = "ridge",
        ridge_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
        min_incremental_rank_ic: float = 0.0,
        random_state: int = 17,
    ) -> None:
        if backend not in {"ridge", "lightgbm"}:
            raise ValueError("backend must be ridge or lightgbm")
        if not ridge_grid or any(value <= 0 for value in ridge_grid):
            raise ValueError("ridge_grid must contain positive values")
        self.backend = backend
        self.ridge_grid = tuple(float(value) for value in ridge_grid)
        self.min_incremental_rank_ic = float(min_incremental_rank_ic)
        self.random_state = int(random_state)
        self.model: object | None = None
        self.columns: tuple[str, ...] = ()
        self.result: CompositeFitResult | None = None

    def _new_lightgbm(self) -> object:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is required for the faithful composite model; install the 'production' optional dependencies"
            ) from exc
        return LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def fit(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        validation_features: pd.DataFrame,
        validation_labels: pd.Series,
        *,
        refit_on_train_and_validation: bool = True,
    ) -> CompositeFitResult:
        if list(train_features.columns) != list(validation_features.columns):
            raise ValueError("train/validation factor columns must match exactly")
        self.columns = tuple(str(name) for name in train_features.columns)
        train_x, train_y = _align_xy(train_features, train_labels)
        valid_x, valid_y = _align_xy(validation_features, validation_labels)
        if len(train_x) < 100 or len(valid_x) < 50:
            raise ValueError("insufficient rows for composite model")
        weights = equal_date_sample_weights(train_x.index)
        equal_prediction = valid_x.mean(axis=1).rename("equal_weight")
        equal_ic = rank_ic_mean(equal_prediction, valid_y, min_names=min(20, max(5, valid_x.index.get_level_values("symbol").nunique() - 2)))

        if self.backend == "ridge":
            best_model: Ridge | None = None
            best_alpha = self.ridge_grid[0]
            best_ic = -np.inf
            for alpha in self.ridge_grid:
                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(train_x.to_numpy(), train_y.to_numpy(), sample_weight=weights)
                prediction = pd.Series(model.predict(valid_x.to_numpy()), index=valid_x.index, name="ridge")
                value = rank_ic_mean(prediction, valid_y, min_names=min(20, max(5, valid_x.index.get_level_values("symbol").nunique() - 2)))
                if np.isfinite(value) and value > best_ic:
                    best_model, best_alpha, best_ic = model, alpha, value
            if best_model is None:
                raise RuntimeError("no finite validation score in ridge grid")
            self.model = best_model
            parameter: float | str = best_alpha
        else:
            model = self._new_lightgbm()
            model.fit(train_x.to_numpy(), train_y.to_numpy(), sample_weight=weights)
            prediction = pd.Series(model.predict(valid_x.to_numpy()), index=valid_x.index, name="lightgbm")
            best_ic = rank_ic_mean(prediction, valid_y, min_names=min(20, max(5, valid_x.index.get_level_values("symbol").nunique() - 2)))
            self.model = model
            parameter = "frozen_v1"

        incremental = float(best_ic - equal_ic)
        self.result = CompositeFitResult(
            backend=self.backend,
            parameter=parameter,
            validation_rank_ic=float(best_ic),
            equal_weight_rank_ic=float(equal_ic),
            incremental_rank_ic=incremental,
            advances_over_control=bool(incremental >= self.min_incremental_rank_ic),
        )
        if refit_on_train_and_validation:
            combined_x = pd.concat([train_x, valid_x]).sort_index()
            combined_y = pd.concat([train_y, valid_y]).reindex(combined_x.index)
            combined_weights = equal_date_sample_weights(combined_x.index)
            if self.backend == "ridge":
                refit = Ridge(alpha=float(parameter), fit_intercept=True)
            else:
                refit = self._new_lightgbm()
            refit.fit(combined_x.to_numpy(), combined_y.to_numpy(), sample_weight=combined_weights)
            self.model = refit
        return self.result

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None or not self.columns:
            raise RuntimeError("model has not been fit")
        if tuple(features.columns) != self.columns:
            raise ValueError("prediction factor columns do not match fitted model")
        x = normalize_factor_matrix(features).fillna(0.0)
        output = pd.Series(self.model.predict(x.to_numpy()), index=x.index, name=f"composite_{self.backend}")
        return output


__all__ = [
    "CompositeAlphaModel",
    "CompositeFitResult",
    "equal_date_sample_weights",
    "normalize_factor_matrix",
    "rank_ic_mean",
]

