from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.track_a import (
    TRACK_A_LIGHTGBM,
    _validate_track_a_label,
    load_track_a_protocol,
    make_track_a_split,
    split_identity,
)
from cogalpha_plus.paths import resolve_data_path
from cogalpha_plus.track_a_models import rolling_track_a_predictions
from cogalpha_plus.validation import SplitPlan


PROJECT = Path(__file__).resolve().parents[1]
_PANEL_MANIFEST = resolve_data_path(PROJECT, "data/baostock_hs300/panel") / "manifest.json"
requires_frozen_panel = pytest.mark.skipif(
    not _PANEL_MANIFEST.is_file(),
    reason="requires the locally built BaoStock CSI300 panel (not redistributed)",
)


@requires_frozen_panel
def test_track_a_protocol_is_bound_to_fixed_budget_data_and_model_contract():
    protocol = load_track_a_protocol(PROJECT)
    assert protocol.raw["factor_search"]["roles"] == 21
    assert protocol.raw["factor_search"]["generations"] == 24
    assert protocol.raw["models"]["lightgbm"] == TRACK_A_LIGHTGBM
    assert protocol.raw["label"]["formula"].startswith("adjusted_open")
    assert len(protocol.config_sha256) == 64


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("formula", "adjusted_vwap_t_plus_11_div_adjusted_vwap_t_plus_1_minus_1"),
        ("signal_time", "t_open"),
        ("missing_endpoint_policy", "drop_missing"),
        ("terminal_bridge_use", "unbound_current_catalog"),
        ("horizon_trading_days", 10.0),
        ("entry_delay_trading_days", True),
    ],
)
def test_track_a_label_contract_rejects_semantic_drift(field, bad_value):
    label = {
        "horizon_trading_days": 10,
        "entry_delay_trading_days": 1,
        "signal_time": "t_close",
        "entry": "open_t_plus_1",
        "exit": "open_t_plus_11",
        "formula": "adjusted_open_t_plus_11_div_adjusted_open_t_plus_1_minus_1",
        "missing_endpoint_policy": "missing_no_forward_fill",
        "terminal_bridge_use": "announced_symbol_exchange_exact_endpoint_only",
    }
    label[field] = bad_value
    with pytest.raises(ValueError, match="exact 10-day open-to-open"):
        _validate_track_a_label(label)


def test_track_a_split_purges_every_open_to_open_outcome_boundary():
    calendar = pd.bdate_range("2011-01-03", "2025-01-20")
    split = make_track_a_split(calendar)
    assert split.label_horizon == 10
    assert split.entry_delay == 1
    assert len(split.purge_after_train) == 11
    assert len(split.purge_after_validation) == 11
    assert len(split.outcome_tail) == 11
    assert max(split.train_dates) < min(split.purge_after_train)
    assert max(split.validation_dates) < min(split.purge_after_validation)
    assert len(split_identity(split)["sha256"]) == 64


def test_track_a_rolling_predictions_refit_every_126_days_without_future_labels():
    rng = np.random.default_rng(19)
    calendar = pd.bdate_range("2018-01-02", periods=431)
    symbols = [f"s{i:02d}" for i in range(24)]
    index = pd.MultiIndex.from_product(
        [calendar, symbols], names=["date", "symbol"]
    )
    features = pd.DataFrame(
        rng.normal(size=(len(index), 20)).astype("float32"),
        index=index,
        columns=[f"f{i:02d}" for i in range(20)],
    )
    labels = pd.Series(
        0.01 * features["f00"] + rng.normal(scale=0.01, size=len(index)),
        index=index,
    )
    split = SplitPlan(
        train_dates=tuple(calendar[:160]),
        purge_after_train=tuple(calendar[160:171]),
        validation_dates=tuple(calendar[171:231]),
        purge_after_validation=tuple(calendar[231:242]),
        sealed_dates=tuple(calendar[242:410]),
        outcome_tail=tuple(calendar[410:421]),
        label_horizon=10,
        entry_delay=1,
    )
    lightgbm = {
        **TRACK_A_LIGHTGBM,
        "n_estimators": 5,
        "learning_rate": 0.05,
        "n_jobs": 1,
    }
    predictions, audit = rolling_track_a_predictions(
        features,
        labels,
        split,
        trading_calendar=calendar,
        ridge_alpha=10.0,
        lightgbm_parameters=lightgbm,
        refit_every=126,
    )
    assert set(predictions) == {"equal_weight", "ridge", "lightgbm"}
    assert len(audit) == 2
    assert audit[0]["latest_training_signal_date"] == str(
        calendar[242 - 11].date()
    )
    assert audit[1]["latest_training_signal_date"] == str(
        calendar[368 - 11].date()
    )
    assert all(
        item["sealed_metrics_consumed_for_model_selection"] is False
        for item in audit
    )
    expected_rows = 168 * len(symbols)
    assert all(len(values) == expected_rows for values in predictions.values())
