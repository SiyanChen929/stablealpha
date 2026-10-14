from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.metrics import forward_return
from cogalpha_plus.terminal_labels import (
    TERMINAL_LABEL_VALUATION_POLICY,
    TerminalLabelBridge,
    apply_terminal_label_bridges,
    load_terminal_label_bridge_catalog_snapshot,
    load_terminal_label_bridges,
    terminal_label_bridge_identity,
    terminal_label_tape_snapshot_hash,
    validate_terminal_label_bridge_identity,
)
from cogalpha_plus.track_a_models import build_track_a_open_labels
from cogalpha_plus.validation import SplitPlan


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
    tape = pd.DataFrame(
        {
            "vwap": raw,
            "adj_vwap": raw * 1.2,
            "adj_factor": 1.2,
        },
        index=raw.index,
    )
    labels = forward_return(tape["adj_vwap"], 4, entry_delay=1, trading_calendar=dates)
    eligible = pd.MultiIndex.from_tuples([(dates[0], "OLD")], names=index.names)
    return dates, tape, labels, eligible


def test_symbol_exchange_bridges_only_missing_eligible_label():
    dates, tape, labels, eligible = _fixture()
    events = [_event()]
    bridged, audit = apply_terminal_label_bridges(
        labels,
        tape,
        eligible,
        events,
        terminal_label_bridge_identity=terminal_label_bridge_identity(events),
        endpoint_price_field="vwap",
        horizon=4,
        entry_delay=1,
        trading_calendar=dates,
    )
    assert bridged.loc[(dates[0], "OLD")] == pytest.approx(1.1 * 12.0 / 10.0 - 1.0)
    assert len(audit["applications"]) == 1
    assert audit["applications"][0]["executable_claim"] is False
    assert audit["valuation_policy"] == TERMINAL_LABEL_VALUATION_POLICY
    assert audit["endpoint_price_field"] == "vwap"
    assert audit["applications"][0]["entry_raw_vwap"] == 10.0
    assert audit["endpoint_price_basis"].startswith("declared_raw")
    assert audit["raw_price_authentication_boundary"] == (
        "upstream_frozen_tape_lineage_required"
    )
    assert audit["endpoint_price_tape_snapshot_hash"] == (
        terminal_label_tape_snapshot_hash(tape)
    )


def test_symbol_exchange_rejects_available_source_endpoint():
    dates, tape, labels, eligible = _fixture()
    events = [_event()]
    exit_key = (dates[5], "OLD")
    added = pd.MultiIndex.from_tuples([exit_key], names=tape.index.names)
    labels = labels.reindex(tape.index.union(added))
    tape = tape.reindex(labels.index)
    tape.loc[exit_key, ["vwap", "adj_factor", "adj_vwap"]] = [11.0, 1.2, 13.2]
    labels.loc[(dates[0], "OLD")] = np.nan
    with pytest.raises(ValueError, match="overwrite an available source endpoint"):
        apply_terminal_label_bridges(
            labels,
            tape,
            eligible,
            events,
            terminal_label_bridge_identity=terminal_label_bridge_identity(events),
            endpoint_price_field="vwap",
            horizon=4,
            entry_delay=1,
            trading_calendar=dates,
        )


def test_symbol_exchange_announced_after_signal_is_not_used():
    dates, tape, labels, eligible = _fixture()
    events = [_event(announcement_date=pd.Timestamp("2020-01-03"))]
    bridged, audit = apply_terminal_label_bridges(
        labels,
        tape,
        eligible,
        events,
        terminal_label_bridge_identity=terminal_label_bridge_identity(events),
        endpoint_price_field="vwap",
        horizon=4,
        entry_delay=1,
        trading_calendar=dates,
    )
    assert pd.isna(bridged.loc[(dates[0], "OLD")])
    assert audit["applications"] == []


def test_terminal_bridge_rejects_naked_or_adjusted_endpoint_prices():
    dates, tape, labels, eligible = _fixture()
    events = [_event()]
    research_frame = pd.DataFrame(
        {"vwap": tape["adj_vwap"]}, index=tape.index
    )
    with pytest.raises(TypeError, match="complete raw/adjusted endpoint price tape"):
        apply_terminal_label_bridges(
            labels,
            research_frame["vwap"],
            eligible,
            events,
            terminal_label_bridge_identity=terminal_label_bridge_identity(events),
            endpoint_price_field="vwap",
            horizon=4,
            entry_delay=1,
            trading_calendar=dates,
        )

    adjusted_as_raw = tape.copy()
    adjusted_as_raw["vwap"] = adjusted_as_raw["adj_vwap"]
    with pytest.raises(ValueError, match=r"adj_price == raw_price \* adj_factor"):
        apply_terminal_label_bridges(
            labels,
            adjusted_as_raw,
            eligible,
            events,
            terminal_label_bridge_identity=terminal_label_bridge_identity(events),
            endpoint_price_field="vwap",
            horizon=4,
            entry_delay=1,
            trading_calendar=dates,
        )


def test_track_a_cnr_csr_bridge_uses_raw_open_not_adjusted_open():
    dates = pd.bdate_range("2015-05-06", periods=12)
    source = "sh.601299"
    destination = "sh.601766"
    tuples = [
        (date, symbol)
        for date in dates
        for symbol in (source, destination)
        if not (date == dates[11] and symbol == source)
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["date", "symbol"])
    symbol_index = index.get_level_values("symbol")
    raw_open = pd.Series(
        np.where(symbol_index == source, 29.98, 29.45), index=index
    )
    adjustment_factor = pd.Series(
        np.where(
            symbol_index == source,
            34.6725 / 29.98,
            32.686643 / 29.45,
        ),
        index=index,
    )
    tape = pd.DataFrame(index=index)
    for field in ("open", "high", "low", "close", "vwap"):
        tape[field] = raw_open
        tape[f"adj_{field}"] = raw_open * adjustment_factor
    tape["adj_factor"] = adjustment_factor
    tape["volume"] = 1_000.0
    tape["amount"] = tape["vwap"] * tape["volume"]
    eligible = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(dates[0], source)], names=["date", "symbol"]
        )
    )
    split = SplitPlan(
        train_dates=(dates[0],),
        validation_dates=(),
        sealed_dates=(),
        purge_after_train=(),
        purge_after_validation=(),
        label_horizon=10,
        entry_delay=1,
    )
    event = _event(
        event_id="cnr_into_csr_20150520",
        symbol=source,
        destination_symbol=destination,
        announcement_date=pd.Timestamp("2014-12-30"),
        effective_date=pd.Timestamp("2015-05-20"),
        settlement_date=pd.Timestamp("2015-05-28"),
    )

    labels = build_track_a_open_labels(
        eligible_panel=eligible,
        feature_tape=tape,
        trading_calendar=dates,
        split=split,
        terminal_label_bridges=[event],
        terminal_label_bridge_identity=terminal_label_bridge_identity([event]),
    )

    expected_raw = 1.10 * 29.45 / 29.98 - 1.0
    wrong_adjusted = 1.10 * 32.686643 / 34.6725 - 1.0
    assert labels.loc[(dates[0], source)] == pytest.approx(expected_raw)
    assert labels.loc[(dates[0], source)] == pytest.approx(0.08055370246831)
    assert labels.loc[(dates[0], source)] != pytest.approx(wrong_adjusted)


def test_real_catalog_is_content_hashed_and_sourced():
    events, identity = load_terminal_label_bridges(
        "configs/hs300_terminal_label_bridges.yaml"
    )
    assert len(events) == 1
    assert events[0].symbol == "sh.601299"
    assert events[0].destination_symbol == "sh.601766"
    assert events[0].share_ratio == pytest.approx(1.10)
    assert identity["valuation_policy"] == TERMINAL_LABEL_VALUATION_POLICY
    assert len(identity["identity_hash"]) == 64


def test_terminal_bridge_rejects_empty_mismatched_or_extra_identity():
    dates, tape, labels, eligible = _fixture()
    event = _event()
    for invalid in (
        {},
        terminal_label_bridge_identity([_event(share_ratio=1.2)]),
        {**terminal_label_bridge_identity([event]), "extra": True},
    ):
        with pytest.raises(ValueError, match="identity"):
            apply_terminal_label_bridges(
                labels,
                tape,
                eligible,
                [event],
                terminal_label_bridge_identity=invalid,
                endpoint_price_field="vwap",
                horizon=4,
                entry_delay=1,
                trading_calendar=dates,
            )


def test_identity_validator_rebuilds_exact_event_catalog():
    event = _event()
    identity = terminal_label_bridge_identity([event])
    assert validate_terminal_label_bridge_identity([event], identity) == identity
    forged = dict(identity)
    forged["identity_hash"] = "0" * 64
    with pytest.raises(ValueError, match="identity hash is invalid"):
        validate_terminal_label_bridge_identity([event], forged)

    type_confused = dict(identity)
    type_confused["schema_version"] = True
    with pytest.raises(ValueError, match="identity hash is invalid"):
        validate_terminal_label_bridge_identity([event], type_confused)

    numeric_type_confused = {
        **identity,
        "events": [{**identity["events"][0], "share_ratio": 1.10}],
    }
    numeric_type_confused["events"][0]["share_ratio"] = 1
    with pytest.raises(ValueError, match="identity hash is invalid"):
        validate_terminal_label_bridge_identity([event], numeric_type_confused)


def test_catalog_loader_rejects_extra_keys_and_boolean_schema(tmp_path):
    source = tmp_path / "catalog.yaml"
    raw = Path("configs/hs300_terminal_label_bridges.yaml").read_text(
        encoding="utf-8"
    )
    source.write_text(raw + "unexpected: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid schema"):
        load_terminal_label_bridge_catalog_snapshot(source)

    source.write_text(raw.replace("schema_version: 1", "schema_version: true"), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid schema"):
        load_terminal_label_bridge_catalog_snapshot(source)


def test_consistently_relabeled_adjusted_prices_are_only_declared_raw():
    dates, tape, labels, eligible = _fixture()
    factors = pd.Series(
        np.where(
            tape.index.get_level_values("symbol") == "OLD", 1.2, 1.5
        ),
        index=tape.index,
    )
    relabeled = tape.copy()
    relabeled["vwap"] = tape["vwap"] * factors
    relabeled["adj_factor"] = factors
    relabeled["adj_vwap"] = relabeled["vwap"] * factors
    relabeled_labels = forward_return(
        relabeled["adj_vwap"], 4, entry_delay=1, trading_calendar=dates
    )
    event = _event()
    bridged, audit = apply_terminal_label_bridges(
        relabeled_labels,
        relabeled,
        eligible,
        [event],
        terminal_label_bridge_identity=terminal_label_bridge_identity([event]),
        endpoint_price_field="vwap",
        horizon=4,
        entry_delay=1,
        trading_calendar=dates,
    )
    assert bridged.loc[(dates[0], "OLD")] == pytest.approx(0.65)
    assert audit["endpoint_price_basis"] == (
        "declared_raw_unadjusted_from_upstream_frozen_tape"
    )
    assert audit["raw_price_authentication_boundary"] == (
        "upstream_frozen_tape_lineage_required"
    )
