import builtins
import importlib.util

import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.ensemble import CompositeAlphaModel


def make_data(seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=80, name="date")
    symbols = pd.Index([f"S{i:03d}" for i in range(25)], name="symbol")
    index = pd.MultiIndex.from_product([dates, symbols])
    a = rng.normal(size=len(index))
    b = rng.normal(size=len(index))
    noise = rng.normal(scale=0.7, size=len(index))
    x = pd.DataFrame({"a": a, "b": b}, index=index)
    y = pd.Series(0.7 * a - 0.4 * b + noise, index=index)
    train_dates = dates[:55]
    train_mask = index.get_level_values("date").isin(train_dates)
    return x.loc[train_mask], y.loc[train_mask], x.loc[~train_mask], y.loc[~train_mask]


def test_ridge_composite_uses_validation_and_predicts():
    train_x, train_y, valid_x, valid_y = make_data()
    model = CompositeAlphaModel(backend="ridge", min_incremental_rank_ic=-1.0)
    result = model.fit(train_x, train_y, valid_x, valid_y, refit_on_train_and_validation=False)
    prediction = model.predict(valid_x)
    assert result.validation_rank_ic > 0.5
    assert result.advances_over_control
    assert prediction.index.equals(valid_x.index)


def test_lightgbm_fails_loud_when_dependency_is_missing(monkeypatch):
    train_x, train_y, valid_x, valid_y = make_data()
    model = CompositeAlphaModel(backend="lightgbm")

    original_import = builtins.__import__

    def import_without_lightgbm(name, *args, **kwargs):
        if name == "lightgbm" or name.startswith("lightgbm."):
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_lightgbm)
    with pytest.raises(RuntimeError, match="LightGBM"):
        model.fit(train_x, train_y, valid_x, valid_y)


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="optional LightGBM dependency is not installed",
)
def test_lightgbm_composite_fits_when_dependency_is_installed():
    train_x, train_y, valid_x, valid_y = make_data()
    model = CompositeAlphaModel(backend="lightgbm", min_incremental_rank_ic=-1.0)
    result = model.fit(
        train_x,
        train_y,
        valid_x,
        valid_y,
        refit_on_train_and_validation=False,
    )
    prediction = model.predict(valid_x)
    assert result.backend == "lightgbm"
    assert np.isfinite(result.validation_rank_ic)
    assert prediction.index.equals(valid_x.index)
    assert np.isfinite(prediction.to_numpy()).all()
