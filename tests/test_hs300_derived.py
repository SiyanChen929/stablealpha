import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.data_quality import (
    AuditMode,
    DataQualityError,
    RESEARCH_PROXY_WATERMARK,
    audit_hs300_fields,
)
from cogalpha_plus.data_sources.hs300 import (
    attach_proxy_benchmark_weights,
    audit_backward_adjustment_causality,
    audit_vendor_adjustment_events_asof,
    derive_adv20_dollars,
    derive_hs300_daily_features,
    derive_implied_float_mcap,
    derive_listed_days,
    derive_rolling_beta,
    derive_vwap,
    normalize_baostock_daily_fields,
    reconstruct_backward_adjusted_view,
    separate_price_views,
    trim_leading_unadjusted_vendor_history,
)
from cogalpha_plus.data_sources.baostock import ADJUST_FACTOR_FIELDS


def price_fixtures(days=8):
    dates = pd.bdate_range("2020-01-02", periods=days, name="date")
    symbols = pd.Index(["A", "B"], name="symbol")
    index = pd.MultiIndex.from_product([dates, symbols])
    row = np.arange(len(index), dtype=float)
    close = 10.0 + 0.1 * row
    volume = np.full(len(index), 1_000.0)
    amount = volume * close
    volume[2] = 0.0
    amount[2] = 0.0
    raw = pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": volume,
            "amount": amount,
            "turn": np.where(volume > 0, "1.0", ""),
            "tradestatus": np.where(volume > 0, 1, 0),
            "isST": 0,
            "adjustflag": "3",
        },
        index=index,
    )
    raw["preclose"] = raw["close"].groupby(level="symbol", sort=False).shift()
    raw["preclose"] = raw["preclose"].fillna(raw["close"])
    adjusted = raw[["open", "high", "low", "close", "preclose"]].mul(1.2)
    adjusted["adjustflag"] = "1"
    return raw, adjusted


def vendor_factor_events(rows):
    """Build a complete BaoStock query_adjust_factor-shaped fixture."""

    return pd.DataFrame(
        [
            {
                "code": code,
                "dividOperateDate": event_date,
                "foreAdjustFactor": 1.0 / float(back_factor),
                "backAdjustFactor": float(back_factor),
                "adjustFactor": float(back_factor),
            }
            for code, event_date, back_factor in rows
        ],
        columns=ADJUST_FACTOR_FIELDS,
    )


def test_price_views_are_separate_and_flags_are_enforced():
    raw, adjusted = price_fixtures()
    views = separate_price_views(raw, adjusted)
    assert views.execution_price_basis == "raw_unadjusted"
    assert views.research_price_basis == "backward_adjusted"
    assert views.raw_execution is not raw
    assert views.backward_adjusted_research is not adjusted
    assert views.raw_execution.loc[views.raw_execution.index[0], "close"] != views.backward_adjusted_research.loc[
        views.backward_adjusted_research.index[0], "close"
    ]

    wrong = adjusted.copy()
    wrong["adjustflag"] = "2"
    with pytest.raises(ValueError, match="adjustflag=1"):
        separate_price_views(raw, wrong)
    with pytest.raises(ValueError, match="preclose"):
        separate_price_views(raw.drop(columns="preclose"), adjusted)


def test_leading_unadjusted_vendor_history_is_trimmed_only_before_membership():
    raw, adjusted = price_fixtures(days=6)
    symbol = "A"
    symbol_index = raw.xs(symbol, level="symbol", drop_level=False).index
    requested = adjusted.copy()
    leading = symbol_index[:2]
    requested.loc[leading, ["open", "high", "low", "close", "preclose"]] = raw.loc[
        leading, ["open", "high", "low", "close", "preclose"]
    ]
    requested.loc[leading, "adjustflag"] = "3"
    membership = pd.DataFrame(
        {
            "date": [symbol_index[3][0], symbol_index[0][0]],
            "code": ["A", "B"],
        }
    )

    clean_raw, clean_adjusted, boundaries = trim_leading_unadjusted_vendor_history(
        raw, requested, membership
    )

    assert clean_raw.index.equals(clean_adjusted.index)
    assert leading.intersection(clean_raw.index).empty
    assert boundaries[0].symbol == "A"
    assert boundaries[0].excluded_rows == 2
    assert boundaries[0].first_adjusted_date == str(symbol_index[2][0].date())
    assert set(clean_adjusted["adjustflag"].astype(str)) == {"1"}


def test_leading_unadjusted_vendor_history_rejects_member_overlap_and_flag_reversion():
    raw, adjusted = price_fixtures(days=5)
    symbol_index = raw.xs("A", level="symbol", drop_level=False).index
    requested = adjusted.copy()
    leading = symbol_index[:2]
    requested.loc[leading, ["open", "high", "low", "close", "preclose"]] = raw.loc[
        leading, ["open", "high", "low", "close", "preclose"]
    ]
    requested.loc[leading, "adjustflag"] = "3"
    overlapping = pd.DataFrame({"date": [symbol_index[1][0]], "code": ["A"]})
    with pytest.raises(ValueError, match="starts after point-in-time membership"):
        trim_leading_unadjusted_vendor_history(raw, requested, overlapping)

    reverting = requested.copy()
    reverting.loc[symbol_index[-1], "adjustflag"] = "3"
    safe_membership = pd.DataFrame({"date": [symbol_index[3][0]], "code": ["A"]})
    with pytest.raises(ValueError, match="revert or interleave"):
        trim_leading_unadjusted_vendor_history(raw, reverting, safe_membership)


def test_adjustment_audit_rejects_wrong_flags_and_nonfinite_prices():
    raw, adjusted = price_fixtures()
    wrong_flag = raw.copy()
    wrong_flag["adjustflag"] = "1"
    flag_audit = audit_backward_adjustment_causality(wrong_flag, adjusted)
    assert not flag_audit.valid
    assert flag_audit.errors == ("raw_adjustflag_must_be_3",)

    infinite = adjusted.copy()
    infinite.loc[infinite.index[0], ["open", "high", "low", "close", "preclose"]] = np.inf
    infinity_audit = audit_backward_adjustment_causality(raw, infinite)
    assert not infinity_audit.valid
    assert "nonfinite" in infinity_audit.errors[0]


def test_event_factors_reconstruct_backward_prices_with_strict_asof_semantics():
    dates = pd.bdate_range("2020-01-02", periods=4, name="date")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"])
    close = np.array([10.0, 11.0, 6.0, 6.6])
    raw = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            # The third row is an ex-date: 11 / 5.5 = 2.
            "preclose": [10.0, 10.0, 5.5, 6.0],
            "adjustflag": "3",
        },
        index=index,
    )
    events = pd.DataFrame(
        {
            "code": ["A", "A", "A"],
            "dividOperateDate": [dates[0], dates[2], dates[-1] + pd.Timedelta(days=30)],
            "backAdjustFactor": [1.0, 2.0, 7.0],
        }
    )
    adjusted = reconstruct_backward_adjusted_view(raw, events)
    assert adjusted["adjustflag"].eq("1").all()
    assert adjusted["close"].to_list() == pytest.approx([10.0, 11.0, 12.0, 13.2])
    # Appending a future factor event cannot rewrite the already built prefix.
    prefix_events = events.iloc[:2]
    prefix_adjusted = reconstruct_backward_adjusted_view(raw, prefix_events)
    pd.testing.assert_frame_equal(adjusted, prefix_adjusted)
    audit = audit_backward_adjustment_causality(raw, adjusted)
    assert audit.valid
    assert audit.corporate_action_events == 1
    # A different positive global anchor has the same causal return path.
    reanchored = adjusted.copy()
    reanchored[["open", "high", "low", "close", "preclose"]] *= 3.7
    assert audit_backward_adjustment_causality(raw, reanchored).valid

    rewritten = adjusted.copy()
    rewritten.loc[(dates[1], "A"), "close"] *= 1.01
    failed = audit_backward_adjustment_causality(raw, rewritten)
    assert not failed.valid
    assert any("causal_path_mismatch" in error for error in failed.errors)

    wrong_event = events.copy()
    wrong_event.loc[1, "backAdjustFactor"] = 3.0
    with pytest.raises(ValueError, match="causal_path_mismatch"):
        reconstruct_backward_adjusted_view(raw, wrong_event)

    infinite_event = events.copy()
    infinite_event.loc[1, "backAdjustFactor"] = np.inf
    with pytest.raises(ValueError, match="invalid symbol/date/factor"):
        reconstruct_backward_adjusted_view(raw, infinite_event)


def test_vendor_event_audit_enforces_common_scale_and_strict_backward_asof():
    dates = pd.bdate_range("2020-01-02", periods=4, name="date")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"])
    close = np.array([10.0, 11.0, 6.0, 6.6])
    raw = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "preclose": [10.0, 10.0, 5.5, 6.0],
            "adjustflag": "3",
        },
        index=index,
    )
    events = vendor_factor_events(
        [
            ("A", dates[0], 1.0),
            ("A", dates[2], 2.0),
            ("A", dates[-1] + pd.Timedelta(days=30), 19.0),
        ]
    )
    adjusted = reconstruct_backward_adjusted_view(raw, events)
    audit = audit_vendor_adjustment_events_asof(
        raw, adjusted, events, event_catalog_symbols={"A"}
    )
    assert audit.valid
    assert audit.factor_changes == 1
    assert audit.future_event_rows_ignored == 1
    assert audit.raw_preclose_recurrence_valid

    # A future event is present in the input but cannot rewrite the prefix.
    without_future = audit_vendor_adjustment_events_asof(
        raw, adjusted, events.iloc[:2], event_catalog_symbols={"A"}
    )
    assert without_future.valid
    assert without_future.maximum_event_path_relative_error == pytest.approx(
        audit.maximum_event_path_relative_error
    )
    leaked = adjusted.copy()
    leaked.loc[(dates[1], "A"), ["open", "high", "low", "close", "preclose"]] *= 19.0
    leaked_audit = audit_vendor_adjustment_events_asof(
        raw, leaked, events, event_catalog_symbols={"A"}
    )
    assert not leaked_audit.valid
    assert any("vendor_event_path_mismatch" in error for error in leaked_audit.errors)

    # Adjusted OHLC/preclose must all use the close-derived scale on each row.
    split_scale = adjusted.copy()
    split_scale.loc[(dates[1], "A"), "high"] *= 1.01
    scale_audit = audit_vendor_adjustment_events_asof(
        raw, split_scale, events, event_catalog_symbols={"A"}
    )
    assert not scale_audit.valid
    assert any("daily_common_scale_mismatch" in error for error in scale_audit.errors)

    wrong_event_path = events.copy()
    wrong_event_path.loc[1, "backAdjustFactor"] = 3.0
    wrong_audit = audit_vendor_adjustment_events_asof(
        raw, adjusted, wrong_event_path, event_catalog_symbols={"A"}
    )
    assert not wrong_audit.valid
    assert any("vendor_event_path_mismatch" in error for error in wrong_audit.errors)


def test_vendor_event_audit_allows_empty_table_only_for_constant_anchor():
    raw, constant_adjusted = price_fixtures()
    no_events = pd.DataFrame(columns=ADJUST_FACTOR_FIELDS)
    constant_audit = audit_vendor_adjustment_events_asof(
        raw, constant_adjusted, no_events, event_catalog_symbols={"A", "B"}
    )
    assert constant_audit.valid
    assert constant_audit.event_rows == 0

    # A successful zero-row response is a per-security catalog state.  It may
    # coexist with another security's non-empty event history.
    dates = raw.index.get_level_values("date").unique()
    only_a_events = vendor_factor_events([("A", dates[0], 1.0)])
    mixed_catalog = audit_vendor_adjustment_events_asof(
        raw,
        constant_adjusted,
        only_a_events,
        event_catalog_symbols={"A", "B"},
    )
    assert mixed_catalog.valid
    missing_b_request = audit_vendor_adjustment_events_asof(
        raw,
        constant_adjusted,
        only_a_events,
        event_catalog_symbols={"A"},
    )
    assert not missing_b_request.valid
    assert missing_b_request.errors[0].startswith("event_catalog_symbol_missing:B")

    changed_empty_symbol = constant_adjusted.copy()
    b_late = [
        (date, "B")
        for date in dates[3:]
    ]
    changed_empty_symbol.loc[
        b_late, ["open", "high", "low", "close", "preclose"]
    ] *= 2.0
    mixed_failure = audit_vendor_adjustment_events_asof(
        raw,
        changed_empty_symbol,
        only_a_events,
        event_catalog_symbols={"A", "B"},
    )
    assert not mixed_failure.valid
    assert any(
        "catalogued_empty_event_nonconstant_anchor:B" in error
        for error in mixed_failure.errors
    )

    dates = pd.bdate_range("2020-01-02", periods=3, name="date")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"])
    close = np.array([10.0, 11.0, 6.0])
    action_raw = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "preclose": [10.0, 10.0, 5.5],
            "adjustflag": "3",
        },
        index=index,
    )
    action_events = vendor_factor_events(
        [("A", dates[0], 1.0), ("A", dates[2], 2.0)]
    )
    changing_adjusted = reconstruct_backward_adjusted_view(action_raw, action_events)
    empty_audit = audit_vendor_adjustment_events_asof(
        action_raw,
        changing_adjusted,
        no_events,
        event_catalog_symbols={"A"},
    )
    assert not empty_audit.valid
    assert any(
        "catalogued_empty_event_nonconstant_anchor" in error
        for error in empty_audit.errors
    )


def test_vendor_event_audit_requires_complete_adjust_factor_schema():
    raw, adjusted = price_fixtures()
    complete = vendor_factor_events(
        [("A", raw.index[0][0], 1.0), ("B", raw.index[0][0], 1.0)]
    )
    no_catalog = audit_vendor_adjustment_events_asof(raw, adjusted, complete)
    assert not no_catalog.valid
    assert no_catalog.errors == ("event_catalog_symbols_required",)

    incomplete = pd.DataFrame(
        {
            "code": ["A", "B"],
            "dividOperateDate": [raw.index[0][0], raw.index[0][0]],
            "backAdjustFactor": [1.0, 1.0],
        }
    )
    audit = audit_vendor_adjustment_events_asof(
        raw, adjusted, incomplete, event_catalog_symbols={"A", "B"}
    )
    assert not audit.valid
    assert audit.errors[0].startswith("adjustment_events_missing:")


def test_strict_factor_reconstruction_requires_pre_start_coverage():
    dates = pd.bdate_range("2020-01-02", periods=2, name="date")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"])
    raw = pd.DataFrame(
        {
            "open": [10.0, 10.5],
            "high": [10.2, 10.7],
            "low": [9.8, 10.3],
            "close": [10.0, 10.5],
            "preclose": [10.0, 10.0],
            "adjustflag": "3",
        },
        index=index,
    )
    too_late = pd.DataFrame(
        {
            "code": ["A"],
            "dividOperateDate": [dates[-1] + pd.Timedelta(days=1)],
            "backAdjustFactor": [2.0],
        }
    )
    with pytest.raises(ValueError, match="no adjustment factor on/before"):
        reconstruct_backward_adjusted_view(raw, too_late)


def test_vwap_adv_and_implied_float_mcap_are_backward_looking():
    raw, _ = price_fixtures()
    vwap = derive_vwap(raw)
    suspended = raw.index[2]
    assert vwap.loc[suspended] == raw.loc[suspended, "close"]

    adv = derive_adv20_dollars(raw, window=3, min_periods=3)
    a_rows = raw.xs("A", level="symbol")
    expected = a_rows["amount"].iloc[:3].mean()
    assert adv.loc[(a_rows.index[2], "A")] == pytest.approx(expected)
    assert np.isnan(adv.loc[(a_rows.index[1], "A")])

    proxy = derive_implied_float_mcap(raw)
    first = raw.index[0]
    expected_shares = raw.loc[first, "volume"] * 100.0 / float(raw.loc[first, "turn"])
    assert proxy.loc[first, "implied_float_shares"] == pytest.approx(expected_shares)
    assert proxy.loc[first, "implied_float_mcap"] == pytest.approx(
        expected_shares * raw.loc[first, "close"]
    )
    assert proxy.loc[suspended, "implied_float_mcap_source"] == "prior_turnover_carry_forward"


def test_only_suspension_blanks_are_normalized_to_zero():
    raw, _ = price_fixtures()
    raw[["volume", "amount"]] = raw[["volume", "amount"]].astype(object)
    suspended = raw.index[2]
    raw.loc[suspended, ["volume", "amount"]] = ""
    normalized = normalize_baostock_daily_fields(raw)
    assert normalized.loc[suspended, "volume"] == 0.0
    assert normalized.loc[suspended, "amount"] == 0.0
    assert derive_vwap(raw).loc[suspended] == raw.loc[suspended, "close"]

    broken = raw.copy()
    tradable = broken.index[0]
    broken.loc[tradable, "amount"] = ""
    with pytest.raises(ValueError, match="tradable row has missing amount"):
        normalize_baostock_daily_fields(broken)
    with pytest.raises(ValueError, match="tradable row has missing amount"):
        separate_price_views(broken, price_fixtures()[1])

    for invalid in ("", np.nan, 2, -1, "unknown"):
        broken_st = raw.copy()
        broken_st["isST"] = broken_st["isST"].astype(object)
        broken_st.loc[broken_st.index[0], "isST"] = invalid
        with pytest.raises(ValueError, match="isST must be exactly 0 or 1"):
            normalize_baostock_daily_fields(broken_st)


def test_suspended_flat_zero_amount_stale_volume_is_normalized_conservatively():
    raw, _ = price_fixtures()
    suspended = raw.index[2]
    raw.loc[suspended, ["open", "high", "low", "close", "preclose"]] = 10.0
    raw.loc[suspended, "volume"] = 12_345.0
    raw.loc[suspended, "amount"] = 0.0
    raw.loc[suspended, "tradestatus"] = 0

    normalized = normalize_baostock_daily_fields(raw)

    assert normalized.loc[suspended, "volume"] == 0.0
    assert derive_vwap(raw).loc[suspended] == 10.0

    nonflat = raw.copy()
    nonflat.loc[suspended, "high"] = 10.1
    with pytest.raises(ValueError, match="not a flat carry-forward"):
        normalize_baostock_daily_fields(nonflat)


def test_coherent_positive_transaction_overrides_stale_suspension_status():
    raw, _ = price_fixtures()
    row = raw.index[0]
    raw.loc[row, "tradestatus"] = 0

    normalized = normalize_baostock_daily_fields(raw)

    assert normalized.loc[row, "tradestatus"] == 1
    assert normalized.loc[row, "volume"] > 0
    assert normalized.loc[row, "amount"] > 0

    incoherent = raw.copy()
    incoherent.loc[row, "amount"] = raw.loc[row, "volume"] * (
        raw.loc[row, "high"] + 1.0
    )
    with pytest.raises(ValueError, match="not coherent with OHLC"):
        normalize_baostock_daily_fields(incoherent)


def test_listed_days_and_rolling_beta():
    raw, _ = price_fixtures()
    basic = pd.DataFrame(
        {"code": ["A", "B"], "ipoDate": ["2019-01-01", "2018-01-01"]}
    )
    listed = derive_listed_days(raw.index, basic)
    assert listed.loc[(pd.Timestamp("2020-01-02"), "A")] == 366

    dates = pd.bdate_range("2020-01-02", periods=9, name="date")
    benchmark = pd.Series(
        [0.01, -0.005, 0.02, -0.01, 0.015, 0.004, -0.008, 0.012, -0.003],
        index=dates,
    )
    asset_returns = 2.0 * benchmark.to_numpy()
    prices = 100.0 * np.cumprod(1.0 + asset_returns)
    index = pd.MultiIndex.from_arrays(
        [dates, np.repeat("A", len(dates))], names=["date", "symbol"]
    )
    close = pd.Series(prices, index=index)
    beta = derive_rolling_beta(close, benchmark, window=4, min_periods=3)
    assert beta.dropna().iloc[-1] == pytest.approx(2.0, abs=1e-10)


def test_daily_feature_builder_uses_raw_execution_prices():
    raw, adjusted = price_fixtures()
    views = separate_price_views(raw, adjusted)
    basic = pd.DataFrame(
        {"code": ["A", "B"], "ipoDate": ["2019-01-01", "2018-01-01"]}
    )
    dates = raw.index.get_level_values("date").unique()
    benchmark = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates)
    features = derive_hs300_daily_features(
        views,
        basic,
        benchmark,
        adv_window=3,
        adv_min_periods=2,
        beta_window=4,
        beta_min_periods=3,
    )
    first = raw.index[0]
    assert features.loc[first, "close"] == raw.loc[first, "close"]
    assert features.loc[first, "adj_factor"] == pytest.approx(1.2)
    for field in ("open", "high", "low", "close", "vwap"):
        assert np.allclose(
            features[f"adj_{field}"],
            features[field] * features["adj_factor"],
            equal_nan=True,
        )
    assert not bool(features.loc[raw.index[2], "tradable"])
    assert {"vwap", "adv20_dollars", "beta", "log_mcap", "listed_days"}.issubset(features.columns)


def test_daily_feature_builder_fails_on_noncausal_adjusted_path():
    raw, adjusted = price_fixtures()
    adjusted = adjusted.copy()
    adjusted.loc[adjusted.index[3], "close"] *= 1.02
    views = separate_price_views(raw, adjusted)
    basic = pd.DataFrame(
        {"code": ["A", "B"], "ipoDate": ["2019-01-01", "2018-01-01"]}
    )
    dates = raw.index.get_level_values("date").unique()
    benchmark = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates)
    with pytest.raises(ValueError, match="backward-adjustment causality failed"):
        derive_hs300_daily_features(
            views,
            basic,
            benchmark,
            adv_window=3,
            adv_min_periods=2,
            beta_window=4,
            beta_min_periods=3,
        )


def test_daily_feature_builder_fail_closes_when_explicit_vendor_events_disagree():
    raw, adjusted = price_fixtures()
    views = separate_price_views(raw, adjusted)
    basic = pd.DataFrame(
        {"code": ["A", "B"], "ipoDate": ["2019-01-01", "2018-01-01"]}
    )
    dates = raw.index.get_level_values("date").unique()
    benchmark = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates)
    events = vendor_factor_events(
        [("A", dates[0], 1.0), ("B", dates[0], 1.0)]
    )
    features = derive_hs300_daily_features(
        views,
        basic,
        benchmark,
        adjustment_events=events,
        event_catalog_symbols={"A", "B"},
        adv_window=3,
        adv_min_periods=2,
        beta_window=4,
        beta_min_periods=3,
    )
    assert len(features) == len(raw)

    contradicted = pd.concat(
        [events, vendor_factor_events([("A", dates[3], 2.0)])],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="vendor adjustment-event as-of audit failed"):
        derive_hs300_daily_features(
            views,
            basic,
            benchmark,
            adjustment_events=contradicted,
            event_catalog_symbols={"A", "B"},
            adv_window=3,
            adv_min_periods=2,
            beta_window=4,
            beta_min_periods=3,
        )


def test_explicit_vendor_catalog_makes_raw_preclose_recurrence_diagnostic_only():
    raw, adjusted = price_fixtures()
    damaged_raw = raw.copy()
    row = damaged_raw.index[5]
    damaged_raw.loc[row, "preclose"] *= 0.91
    adjusted = adjusted.copy()
    adjusted.loc[row, "preclose"] = damaged_raw.loc[row, "preclose"] * 1.2
    no_events = pd.DataFrame(columns=ADJUST_FACTOR_FIELDS)

    audit = audit_vendor_adjustment_events_asof(
        damaged_raw,
        adjusted,
        no_events,
        event_catalog_symbols={"A", "B"},
    )
    assert audit.valid
    assert not audit.raw_preclose_recurrence_valid
    assert audit.raw_preclose_recurrence_errors

    views = separate_price_views(damaged_raw, adjusted)
    basic = pd.DataFrame(
        {"code": ["A", "B"], "ipoDate": ["2019-01-01", "2018-01-01"]}
    )
    dates = raw.index.get_level_values("date").unique()
    benchmark = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates)
    features = derive_hs300_daily_features(
        views,
        basic,
        benchmark,
        adjustment_events=no_events,
        event_catalog_symbols={"A", "B"},
        adv_window=3,
        adv_min_periods=2,
        beta_window=4,
        beta_min_periods=3,
    )
    assert len(features) == len(raw)

    # Without an independently committed catalog, retain the legacy strict
    # recurrence behavior for compatibility and fail closed.
    with pytest.raises(ValueError, match="backward-adjustment causality failed"):
        derive_hs300_daily_features(
            views,
            basic,
            benchmark,
            adv_window=3,
            adv_min_periods=2,
            beta_window=4,
            beta_min_periods=3,
        )


def test_strict_fails_for_missing_official_weights_proxy_is_watermarked():
    strict = audit_hs300_fields(AuditMode.STRICT)
    assert not strict.accepted
    assert "official_historical_benchmark_weight_unavailable" in strict.errors

    proxy_audit = audit_hs300_fields("research_proxy")
    assert proxy_audit.accepted
    assert proxy_audit.watermarks == (RESEARCH_PROXY_WATERMARK,)

    dates = pd.bdate_range("2020-01-02", periods=2, name="date")
    symbols = pd.Index(["A", "B"], name="symbol")
    index = pd.MultiIndex.from_product([dates, symbols])
    panel = pd.DataFrame(
        {
            "universe_member": True,
            "implied_float_mcap": [100.0, 300.0, np.nan, 300.0],
        },
        index=index,
    )
    weighted = attach_proxy_benchmark_weights(panel)
    first = weighted.xs(dates[0], level="date")
    second = weighted.xs(dates[1], level="date")
    assert first["benchmark_weight"].tolist() == pytest.approx([0.25, 0.75])
    assert second["benchmark_weight"].tolist() == pytest.approx([0.5, 0.5])
    assert set(weighted["benchmark_weight_quality"]) == {"research_proxy"}
    assert set(weighted["research_watermark"]) == {RESEARCH_PROXY_WATERMARK}
    assert weighted.attrs["dataset_tier"] == "research_proxy"
    assert weighted.attrs["field_quality"]["benchmark_weight"] == "research_proxy"
    assert weighted.attrs["field_quality"]["adj_close"] == "derived_from_official"
    with pytest.raises(DataQualityError, match="official_historical_benchmark_weight"):
        attach_proxy_benchmark_weights(panel, mode="strict")
