import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.metrics import forward_return
from cogalpha_plus.terminal_labels import (
    TerminalLabelBridge,
    apply_terminal_label_bridges,
    load_terminal_label_bridges,
)


def _event(**overrides):
    values = {
        "event_id": "exchange",
        "kind": "symbol_exchange",
        "symbol": "OLD",
        "destination_symbol": "NEW",
        "announcement_date": pd.Timestamp("2020-01-01"),
        "effective_date": pd.Timestamp("2020-01-07"),
        "settlement_date": pd.Timestamp("2020-01-09"),
        "share_ratio": 1.1,
        "net_cash_per_share": 0.0,
        "sources": ("https://example.test/terms",),
    }
    values.update(overrides)
    return TerminalLabelBridge(**values)


def _fixture():
    dates = pd.bdate_range("2020-01-02", periods=7)
    index = pd.MultiIndex.from_product([dates, ["NEW", "OLD"]], names=["date", "symbol"])
    raw = pd.Series(10.0, index=index, name="vwap")
    exit_date = dates[5]
    raw.loc[(exit_date, "NEW")] = 12.0
    raw = raw.drop((exit_date, "OLD"))
    adjusted = raw.copy()
    labels = forward_return(adjusted, 4, entry_delay=1, trading_calendar=dates)
    eligible = pd.MultiIndex.from_tuples([(dates[0], "OLD")], names=index.names)
    return dates, raw, labels, eligible


def test_symbol_exchange_bridges_only_missing_eligible_label():
    dates, raw, labels, eligible = _fixture()
    bridged, audit = apply_terminal_label_bridges(
        labels,
        raw,
        eligible,
        [_event()],
        horizon=4,
        entry_delay=1,
        trading_calendar=dates,
    )
    assert bridged.loc[(dates[0], "OLD")] == pytest.approx(1.1 * 12.0 / 10.0 - 1.0)
    assert len(audit["applications"]) == 1
    assert audit["applications"][0]["executable_claim"] is False


def test_symbol_exchange_rejects_available_source_endpoint():
    dates, raw, labels, eligible = _fixture()
    exit_key = (dates[5], "OLD")
    labels = labels.reindex(raw.index.union(pd.MultiIndex.from_tuples([exit_key], names=raw.index.names)))
    raw = raw.reindex(labels.index)
    raw.loc[exit_key] = 11.0
    labels.loc[(dates[0], "OLD")] = np.nan
    with pytest.raises(ValueError, match="overwrite an available source endpoint"):
        apply_terminal_label_bridges(
            labels,
            raw,
            eligible,
            [_event()],
            horizon=4,
            entry_delay=1,
            trading_calendar=dates,
        )


def test_symbol_exchange_announced_after_signal_is_not_used():
    dates, raw, labels, eligible = _fixture()
    bridged, audit = apply_terminal_label_bridges(
        labels,
        raw,
        eligible,
        [_event(announcement_date=pd.Timestamp("2020-01-03"))],
        horizon=4,
        entry_delay=1,
        trading_calendar=dates,
    )
    assert pd.isna(bridged.loc[(dates[0], "OLD")])
    assert audit["applications"] == []


def test_real_catalog_is_content_hashed_and_sourced():
    events, identity = load_terminal_label_bridges(
        "configs/hs300_terminal_label_bridges.yaml"
    )
    assert len(events) == 1
    assert events[0].symbol == "sh.601299"
    assert events[0].destination_symbol == "sh.601766"
    assert events[0].share_ratio == pytest.approx(1.10)
    assert len(identity["identity_hash"]) == 64
