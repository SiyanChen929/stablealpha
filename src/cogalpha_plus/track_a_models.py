"""Frozen factor matrix and chronological Track A composite models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .dsl import evaluate_expression, expression_hash
from .ensemble import equal_date_sample_weights, normalize_factor_matrix
from .evolution import EvaluatedCandidate
from .metrics import daily_ic, forward_return, summarize_ic
from .terminal_labels import TerminalLabelBridge, apply_terminal_label_bridges
from .validation import (
    PRODUCTION_DSL_SPEC,
    SplitPlan,
    build_cross_sectional_mask,
    build_research_feature_frame,
)


@dataclass(frozen=True)
class ModelPredictionBundle:
    predictions: Mapping[str, pd.Series]
    validation_metrics: Mapping[str, Mapping[str, object]]
    rolling_audit: tuple[Mapping[str, object], ...]
    feature_identity: Mapping[str, object]


def _hash_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_track_a_factor_matrix(
    champions: Sequence[EvaluatedCandidate],
    *,
    eligible_panel: pd.DataFrame,
    feature_tape: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Recompute the frozen champion ASTs on the complete continuous tape."""

    if len(champions) != 20:
        raise ValueError("Track A requires exactly 20 frozen champion factors")
    feature_frame = build_research_feature_frame(feature_tape)
    mask = build_cross_sectional_mask(feature_tape, eligible_panel)
    columns: dict[str, pd.Series] = {}
    records: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    for evaluated in champions:
        candidate = evaluated.candidate
        digest = expression_hash(candidate.expression, PRODUCTION_DSL_SPEC)
        if digest in seen_hashes:
            raise ValueError("Track A champion family contains duplicate ASTs")
        seen_hashes.add(digest)
        direction = candidate.metadata.get("expected_direction")
        if direction not in {-1, 1}:
            raise ValueError("Track A champion lacks a predeclared direction")
        values = evaluate_expression(
            candidate.expression,
            feature_frame,
            PRODUCTION_DSL_SPEC,
            cross_sectional_mask=mask,
        )
        signed = (values * int(direction)).reindex(eligible_panel.index)
        columns[candidate.candidate_id] = signed.astype("float32")
        records.append(
            {
                "candidate_id": candidate.candidate_id,
                "expression_hash": digest,
                "direction": int(direction),
                "roles": list(candidate.metadata.get("role_lineage", (candidate.agent_id,))),
            }
        )
    raw = pd.DataFrame(columns, index=eligible_panel.index)
    ranked = normalize_factor_matrix(raw).astype("float32")
    identity_payload = {
        "schema": "track_a_factor_matrix_v1",
        "rows": int(len(ranked)),
        "columns": list(ranked.columns),
        "champions": records,
        "transform": "same_date_cross_sectional_pct_rank_to_minus1_plus1_then_zero_impute",
        "missing_before_impute": {
            name: int(ranked[name].isna().sum()) for name in ranked.columns
        },
    }
    ranked = ranked.fillna(0.0)
    identity = {**identity_payload, "sha256": _hash_json(identity_payload)}
    return ranked, identity


def build_track_a_open_labels(
    *,
    eligible_panel: pd.DataFrame,
    feature_tape: pd.DataFrame,
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    split: SplitPlan,
    terminal_label_bridges: Sequence[TerminalLabelBridge] = (),
    terminal_label_bridge_identity: Mapping[str, object] | None = None,
) -> pd.Series:
    feature_frame = build_research_feature_frame(feature_tape)
    labels = forward_return(
        feature_frame["open"],
        split.label_horizon,
        entry_delay=split.entry_delay,
        trading_calendar=trading_calendar,
    )
    labels, _ = apply_terminal_label_bridges(
        labels,
        feature_tape,
        eligible_panel.index,
        terminal_label_bridges,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
        endpoint_price_field="open",
        horizon=split.label_horizon,
        entry_delay=split.entry_delay,
        trading_calendar=trading_calendar,
    )
    labels = labels.reindex(eligible_panel.index)
    labels.name = "future_10d_open_to_open"
    return labels


def _aligned_training(
    features: pd.DataFrame,
    labels: pd.Series,
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    date_set = pd.DatetimeIndex(dates)
    mask = features.index.get_level_values("date").isin(date_set)
    frame = features.loc[mask].join(labels.rename("__label__"), how="inner")
    finite = pd.to_numeric(frame["__label__"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).notna()
    frame = frame.loc[finite]
    y = frame.pop("__label__").astype(float)
    if frame.empty or len(frame.columns) != 20:
        raise ValueError("Track A model training matrix is empty or not 20-dimensional")
    x = frame.astype("float32")
    weights = equal_date_sample_weights(x.index)
    return x, y, weights


def _new_model(
    backend: str,
    *,
    ridge_alpha: float,
    lightgbm_parameters: Mapping[str, object],
) -> object:
    if backend == "ridge":
        return Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    if backend != "lightgbm":
        raise ValueError("backend must be ridge or lightgbm")
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("Track A requires LightGBM; no fallback is permitted") from exc
    return LGBMRegressor(**dict(lightgbm_parameters), verbosity=-1)


def _model_hash(model: object, backend: str) -> str:
    if backend == "ridge":
        coef = np.asarray(getattr(model, "coef_"), dtype=np.float64)
        intercept = np.asarray([getattr(model, "intercept_")], dtype=np.float64)
        return hashlib.sha256(coef.tobytes() + intercept.tobytes()).hexdigest()
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise RuntimeError("fitted LightGBM model has no booster")
    return hashlib.sha256(booster.model_to_string().encode("utf-8")).hexdigest()


def _prediction_metrics(
    prediction: pd.Series,
    labels: pd.Series,
    *,
    min_names: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    aligned_labels = labels.reindex(prediction.index)
    ic, names = daily_ic(
        prediction,
        aligned_labels,
        method="spearman",
        min_names=min_names,
    )
    result = summarize_ic(
        ic,
        names,
        hac_lags=9,
        block_size=20,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    ).to_dict()
    result["daily_ic_dates"] = int(len(ic))
    result["prediction_rows"] = int(len(prediction))
    return result


def score_track_a_predictions(
    predictions: Mapping[str, pd.Series],
    labels: pd.Series,
    *,
    min_names: int = 20,
    bootstrap_samples: int = 500,
) -> dict[str, Mapping[str, object]]:
    """Score a frozen prediction family without choosing among its members."""

    return {
        name: _prediction_metrics(
            prediction,
            labels,
            min_names=min_names,
            bootstrap_samples=bootstrap_samples,
            seed=701 + position,
        )
        for position, (name, prediction) in enumerate(sorted(predictions.items()))
    }


def fit_validation_controls(
    features: pd.DataFrame,
    labels: pd.Series,
    split: SplitPlan,
    *,
    ridge_alpha: float,
    lightgbm_parameters: Mapping[str, object],
    bootstrap_samples: int = 500,
    min_names: int = 20,
) -> tuple[dict[str, pd.Series], dict[str, Mapping[str, object]]]:
    """Fit on 2011-2019 and score all fixed controls on 2020 only."""

    train_x, train_y, weights = _aligned_training(
        features, labels, split.train_dates
    )
    valid_mask = features.index.get_level_values("date").isin(split.validation_dates)
    valid_x = features.loc[valid_mask]
    predictions: dict[str, pd.Series] = {
        "equal_weight": valid_x.mean(axis=1).rename("equal_weight")
    }
    for backend in ("ridge", "lightgbm"):
        model = _new_model(
            backend,
            ridge_alpha=ridge_alpha,
            lightgbm_parameters=lightgbm_parameters,
        )
        model.fit(train_x.to_numpy(), train_y.to_numpy(), sample_weight=weights)
        predictions[backend] = pd.Series(
            model.predict(valid_x.to_numpy()),
            index=valid_x.index,
            name=backend,
        )
    metrics = {
        name: _prediction_metrics(
            prediction,
            labels,
            min_names=min_names,
            bootstrap_samples=bootstrap_samples,
            seed=101 + position,
        )
        for position, (name, prediction) in enumerate(predictions.items())
    }
    return predictions, metrics


def rolling_track_a_predictions(
    features: pd.DataFrame,
    labels: pd.Series,
    split: SplitPlan,
    *,
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    ridge_alpha: float,
    lightgbm_parameters: Mapping[str, object],
    refit_every: int = 126,
) -> tuple[dict[str, pd.Series], tuple[Mapping[str, object], ...]]:
    """Generate sealed predictions with expanding, information-safe refits."""

    if refit_every != 126:
        raise ValueError("Track A rolling refit interval must equal 126 sessions")
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize()
    test_dates = pd.DatetimeIndex(split.sealed_dates)
    calendar_positions = {pd.Timestamp(date): i for i, date in enumerate(calendar)}
    missing_dates = test_dates.difference(calendar)
    if len(missing_dates):
        raise ValueError("sealed signal dates are missing from the frozen calendar")
    blocks = [test_dates[start : start + refit_every] for start in range(0, len(test_dates), refit_every)]
    output: dict[str, list[pd.Series]] = {
        "equal_weight": [],
        "ridge": [],
        "lightgbm": [],
    }
    audits: list[Mapping[str, object]] = []
    known_lag = split.label_horizon + split.entry_delay
    research_start = pd.Timestamp(split.train_dates[0])
    for block_number, block_dates in enumerate(blocks, start=1):
        refit_date = pd.Timestamp(block_dates[0])
        refit_position = calendar_positions[refit_date]
        cutoff_position = refit_position - known_lag
        if cutoff_position < 0:
            raise ValueError("insufficient calendar history for an information-safe refit")
        label_cutoff = pd.Timestamp(calendar[cutoff_position])
        train_dates = calendar[(calendar >= research_start) & (calendar <= label_cutoff)]
        train_x, train_y, weights = _aligned_training(features, labels, train_dates)
        block_mask = features.index.get_level_values("date").isin(block_dates)
        block_x = features.loc[block_mask]
        if block_x.empty:
            raise ValueError("rolling prediction block has no eligible rows")
        output["equal_weight"].append(
            block_x.mean(axis=1).rename("equal_weight")
        )
        model_hashes: dict[str, str] = {}
        for backend in ("ridge", "lightgbm"):
            model = _new_model(
                backend,
                ridge_alpha=ridge_alpha,
                lightgbm_parameters=lightgbm_parameters,
            )
            model.fit(train_x.to_numpy(), train_y.to_numpy(), sample_weight=weights)
            prediction = pd.Series(
                model.predict(block_x.to_numpy()),
                index=block_x.index,
                name=backend,
            )
            output[backend].append(prediction)
            model_hashes[backend] = _model_hash(model, backend)
        audits.append(
            {
                "schema": "track_a_rolling_refit_v1",
                "block": block_number,
                "refit_signal_date": str(refit_date.date()),
                "first_prediction_date": str(pd.Timestamp(block_dates[0]).date()),
                "last_prediction_date": str(pd.Timestamp(block_dates[-1]).date()),
                "prediction_dates": int(len(block_dates)),
                "label_availability_lag_sessions": known_lag,
                "latest_training_signal_date": str(label_cutoff.date()),
                "training_rows": int(len(train_x)),
                "training_dates": int(train_x.index.get_level_values("date").nunique()),
                "model_sha256": model_hashes,
                "sealed_metrics_consumed_for_model_selection": False,
            }
        )
    predictions = {
        name: pd.concat(parts).sort_index() for name, parts in output.items()
    }
    expected_index = features.loc[
        features.index.get_level_values("date").isin(test_dates)
    ].index
    for name, prediction in predictions.items():
        if not prediction.index.equals(expected_index) or prediction.isna().any():
            raise RuntimeError(f"{name} rolling predictions do not cover the sealed index")
    return predictions, tuple(audits)


def run_track_a_models(
    champions: Sequence[EvaluatedCandidate],
    *,
    eligible_panel: pd.DataFrame,
    feature_tape: pd.DataFrame,
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    split: SplitPlan,
    ridge_alpha: float,
    lightgbm_parameters: Mapping[str, object],
    bootstrap_samples: int = 500,
    terminal_label_bridges: Sequence[TerminalLabelBridge] = (),
    terminal_label_bridge_identity: Mapping[str, object] | None = None,
) -> ModelPredictionBundle:
    features, feature_identity = build_track_a_factor_matrix(
        champions,
        eligible_panel=eligible_panel,
        feature_tape=feature_tape,
    )
    labels = build_track_a_open_labels(
        eligible_panel=eligible_panel,
        feature_tape=feature_tape,
        trading_calendar=trading_calendar,
        split=split,
        terminal_label_bridges=terminal_label_bridges,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
    )
    _, validation_metrics = fit_validation_controls(
        features,
        labels,
        split,
        ridge_alpha=ridge_alpha,
        lightgbm_parameters=lightgbm_parameters,
        bootstrap_samples=bootstrap_samples,
    )
    predictions, audit = rolling_track_a_predictions(
        features,
        labels,
        split,
        trading_calendar=trading_calendar,
        ridge_alpha=ridge_alpha,
        lightgbm_parameters=lightgbm_parameters,
    )
    return ModelPredictionBundle(
        predictions=predictions,
        validation_metrics=validation_metrics,
        rolling_audit=audit,
        feature_identity=feature_identity,
    )


__all__ = [
    "ModelPredictionBundle",
    "build_track_a_factor_matrix",
    "build_track_a_open_labels",
    "fit_validation_controls",
    "rolling_track_a_predictions",
    "run_track_a_models",
    "score_track_a_predictions",
]
