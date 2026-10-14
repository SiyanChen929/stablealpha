import numpy as np
import pandas as pd

from cogalpha_plus.data_contract import DataContract, validate_daily_panel


def valid_panel(days=30):
    dates = pd.bdate_range("2024-01-01", periods=days, name="date")
    names = pd.Index(["A", "B"], name="symbol")
    index = pd.MultiIndex.from_product([dates, names])
    n = len(index)
    return pd.DataFrame(
        {
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.1,
            "vwap": 10.0,
            "adj_open": 10.0,
            "adj_high": 10.5,
            "adj_low": 9.5,
            "adj_close": 10.1,
            "adj_vwap": 10.0,
            "volume": 1_000_000.0,
            "amount": 10_000_000.0,
            "adj_factor": 1.0,
            "tradable": True,
            "limit_up": False,
            "limit_down": False,
            "is_st": False,
            "listed_days": 500,
            "industry": ["I1", "I2"] * days,
            "log_mcap": 22.0,
            "beta": 1.0,
            "universe_member": True,
            "benchmark_weight": 0.5,
            "adv20_dollars": 20_000_000.0,
        },
        index=index,
    )


META = {
    "vendor": "fixture",
    "extracted_at": "2026-08-29T00:00:00Z",
    "corporate_action_policy": "point_in_time_adjustment_factors",
    "calendar": "fixture",
    "includes_delisted": True,
    "point_in_time_universe": True,
    "execution_price_basis": "raw_unadjusted",
}
META.update(
    {
        "dataset_tier": "strict",
        "benchmark_weight_quality": "official_historical",
        "benchmark_weight_source": "fixture_official_index_archive",
        "field_quality": {
            name: (
                "derived_from_official"
                if name.startswith("adj_") or name == "adv20_dollars"
                else "official_point_in_time"
            )
            for name in DataContract().required_columns
        },
    }
)


def test_valid_contract_and_fail_closed_metadata():
    panel = valid_panel()
    report = validate_daily_panel(panel, contract=DataContract(minimum_history_days=20), metadata=META)
    assert report.valid
    assert report.snapshot_hash
    bad = validate_daily_panel(panel, contract=DataContract(minimum_history_days=20), metadata={})
    assert not bad.valid
    assert "metadata_does_not_confirm_delisted_history" in bad.errors


def test_current_constituent_backfill_and_ohlc_errors_are_caught():
    panel = valid_panel()
    panel.iloc[0, panel.columns.get_loc("high")] = 8.0
    panel.iloc[1, panel.columns.get_loc("benchmark_weight")] = 0.2
    report = validate_daily_panel(panel, contract=DataContract(minimum_history_days=20), metadata=META)
    assert not report.valid
    assert "ohlc_high_below_open_or_close" in report.errors
    assert any(item.startswith("benchmark_weights_not_one_dates") for item in report.errors)


def test_dual_price_views_are_required_and_must_match_adjustment_factor():
    panel = valid_panel()
    missing = validate_daily_panel(
        panel.drop(columns="adj_vwap"),
        contract=DataContract(minimum_history_days=20),
        metadata=META,
    )
    assert not missing.valid
    assert "missing_required_columns:adj_vwap" in missing.errors

    panel.loc[panel.index[0], "adj_close"] = 11.0
    mismatched = validate_daily_panel(
        panel,
        contract=DataContract(minimum_history_days=20),
        metadata=META,
    )
    assert not mismatched.valid
    assert any(
        item.startswith("adjusted_prices_inconsistent_with_adj_factor:adj_close")
        for item in mismatched.errors
    )


def test_nonfinite_numeric_values_fail_closed():
    panel = valid_panel()
    panel.loc[panel.index[0], ["open", "high", "low", "close", "vwap"]] = np.inf
    panel.loc[
        panel.index[0],
        ["adj_open", "adj_high", "adj_low", "adj_close", "adj_vwap"],
    ] = np.inf
    report = validate_daily_panel(
        panel,
        contract=DataContract(minimum_history_days=20),
        metadata=META,
    )
    assert not report.valid
    assert "nonfinite:close" in report.errors
    assert "nonfinite:adj_close" in report.errors


def test_sparse_risk_inputs_are_allowed_only_on_research_ineligible_rows():
    panel = valid_panel()
    panel["research_eligible"] = True
    first = panel.index[0]
    panel.loc[first, ["log_mcap", "beta", "adv20_dollars"]] = np.nan
    panel.loc[first, "research_eligible"] = False

    report = validate_daily_panel(
        panel,
        contract=DataContract(minimum_history_days=20),
        metadata=META,
    )
    assert report.valid
    assert "nonfinite_on_research_ineligible:log_mcap:1" in report.warnings
    assert "nonfinite_on_research_ineligible:beta:1" in report.warnings
    assert "nonfinite_on_research_ineligible:adv20_dollars:1" in report.warnings

    panel.loc[first, "research_eligible"] = True
    invalid = validate_daily_panel(
        panel,
        contract=DataContract(minimum_history_days=20),
        metadata=META,
    )
    assert not invalid.valid
    assert "missing_or_non_numeric:beta" in invalid.errors
    assert "nonfinite:beta" in invalid.errors


def test_strict_rejects_proxies_and_research_proxy_passes_with_watermark_warnings():
    panel = valid_panel()
    strict_proxy = dict(META)
    strict_proxy["field_quality"] = dict(META["field_quality"])
    strict_proxy["field_quality"]["industry"] = "research_proxy"
    strict_report = validate_daily_panel(
        panel,
        contract=DataContract(minimum_history_days=20),
        metadata=strict_proxy,
    )
    assert not strict_report.valid
    assert "strict_required_fields_are_proxy:industry" in strict_report.errors

    nonofficial_weights = dict(META)
    nonofficial_weights["benchmark_weight_quality"] = "research_proxy"
    nonofficial_report = validate_daily_panel(
        panel,
        contract=DataContract(minimum_history_days=20),
        metadata=nonofficial_weights,
    )
    assert not nonofficial_report.valid
    assert any(
        item.startswith("strict_requires_official_historical_benchmark_weights")
        for item in nonofficial_report.errors
    )

    research_meta = dict(strict_proxy)
    research_meta["dataset_tier"] = "research_proxy"
    research_meta["benchmark_weight_quality"] = "research_proxy"
    research_report = validate_daily_panel(
        panel,
        contract=DataContract(minimum_history_days=20),
        metadata=research_meta,
    )
    assert research_report.valid
    assert "dataset_tier_research_proxy:not_portfolio_grade" in research_report.warnings
    assert any(
        item.startswith("benchmark_weights_not_official_historical")
        for item in research_report.warnings
    )
    assert "research_proxy_required_fields:industry" in research_report.warnings
