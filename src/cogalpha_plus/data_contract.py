"""Fail-closed point-in-time daily-panel contract for investable research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

RAW_PRICE_COLUMNS = ("open", "high", "low", "close", "vwap")
ADJUSTED_PRICE_COLUMNS = tuple(f"adj_{name}" for name in RAW_PRICE_COLUMNS)
RESEARCH_PRICE_MAP = dict(zip(RAW_PRICE_COLUMNS, ADJUSTED_PRICE_COLUMNS, strict=True))

# ``field_quality`` is deliberately a small vocabulary.  The aliases make the
# contract usable with both a direct vendor adapter and a deterministic derived
# field, without allowing an estimate to acquire portfolio-grade status merely
# because it has a non-empty description.
_AUTHORITATIVE_FIELD_QUALITIES = frozenset(
    {
        "official",
        "official_point_in_time",
        "official_historical",
        "verified",
        "direct",
        "direct_observation",
        "derived",
        "derived_from_official",
        "deterministic_derived",
    }
)
_NON_AUTHORITATIVE_FIELD_QUALITIES = frozenset(
    {
        "proxy",
        "research_proxy",
        "unknown",
        "unavailable",
        "unverified",
        "estimated",
        "implied",
        "reconstructed",
        "missing",
    }
)
_OFFICIAL_HISTORICAL_WEIGHT_QUALITIES = frozenset(
    {"official_historical", "official_point_in_time_historical"}
)


@dataclass(frozen=True)
class DataContract:
    required_columns: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_vwap",
        "volume",
        "amount",
        "adj_factor",
        "tradable",
        "limit_up",
        "limit_down",
        "is_st",
        "listed_days",
        "industry",
        "log_mcap",
        "beta",
        "universe_member",
        "benchmark_weight",
        "adv20_dollars",
    )
    minimum_history_days: int = 252
    benchmark_weight_tolerance: float = 1e-4
    adjusted_price_tolerance: float = 1e-8
    require_delisted_history: bool = True
    require_historical_universe: bool = True
    raw_execution_prices: bool = True


@dataclass(frozen=True)
class DataValidationReport:
    valid: bool
    rows: int
    symbols: int
    trading_days: int
    first_date: str
    last_date: str
    snapshot_hash: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _panel_snapshot_hash(panel: pd.DataFrame) -> str:
    # Import lazily: ``runner`` owns orchestration and imports ``validation``,
    # while validation imports the price-view constants from this module.
    from .runner import panel_snapshot_hash

    return panel_snapshot_hash(panel)


def _normalise_quality(value: object) -> str:
    """Return a conservative scalar quality label from supported metadata forms."""

    if isinstance(value, Mapping):
        if value.get("official") is False:
            return "research_proxy"
        if value.get("official") is True and not any(key in value for key in ("quality", "status", "tier")):
            return "official"
        value = value.get("quality", value.get("status", value.get("tier", "unknown")))
    if value is True:
        return "official"
    if value is False or value is None:
        return "unknown"
    return str(value).strip().lower().replace("-", "_").replace(" ", "_") or "unknown"


def _field_quality_findings(
    metadata: Mapping[str, object],
    required_columns: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return required fields that are non-authoritative or have unknown quality."""

    raw_quality = metadata.get("field_quality")
    if not isinstance(raw_quality, Mapping):
        return (), tuple(required_columns)
    proxy: list[str] = []
    unknown: list[str] = []
    for name in required_columns:
        if name not in raw_quality:
            unknown.append(name)
            continue
        quality = _normalise_quality(raw_quality[name])
        if quality in _NON_AUTHORITATIVE_FIELD_QUALITIES:
            if quality in {"proxy", "research_proxy", "estimated", "implied", "reconstructed"}:
                proxy.append(name)
            else:
                unknown.append(name)
        elif quality not in _AUTHORITATIVE_FIELD_QUALITIES:
            # An unrecognised label is not evidence of authoritative data.
            unknown.append(name)
    return tuple(proxy), tuple(unknown)


def _validate_ohlc_view(
    numeric: pd.DataFrame,
    *,
    prefix: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    open_name = f"{prefix}open"
    high_name = f"{prefix}high"
    low_name = f"{prefix}low"
    close_name = f"{prefix}close"
    vwap_name = f"{prefix}vwap"
    label = "adj_ohlc" if prefix else "ohlc"
    if {open_name, high_name, low_name, close_name}.issubset(numeric.columns):
        max_oc = numeric[[open_name, close_name]].max(axis=1)
        min_oc = numeric[[open_name, close_name]].min(axis=1)
        if (numeric[high_name] + 1e-12 < max_oc).any():
            errors.append(f"{label}_high_below_open_or_close")
        if (numeric[low_name] - 1e-12 > min_oc).any():
            errors.append(f"{label}_low_above_open_or_close")
        if (numeric[high_name] < numeric[low_name]).any():
            errors.append(f"{label}_high_below_low")
    if {low_name, high_name, vwap_name}.issubset(numeric.columns):
        outside = (numeric[vwap_name] < numeric[low_name] * 0.999) | (
            numeric[vwap_name] > numeric[high_name] * 1.001
        )
        if outside.any():
            warning_name = "adj_vwap" if prefix else "vwap"
            warnings.append(f"{warning_name}_outside_daily_range_rows:{int(outside.sum())}")


def validate_daily_panel(
    panel: pd.DataFrame,
    *,
    contract: DataContract | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DataValidationReport:
    spec = contract or DataContract()
    meta = dict(metadata or {})
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["date", "symbol"]:
        errors.append("index_must_be_exactly_MultiIndex_date_symbol")
        dates = pd.Index([], dtype="datetime64[ns]")
        symbols = pd.Index([], dtype=str)
    else:
        dates = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
        symbols = pd.Index(panel.index.get_level_values("symbol").unique())
        if not panel.index.is_unique:
            errors.append("duplicate_date_symbol_rows")
        if not panel.index.is_monotonic_increasing:
            errors.append("panel_not_sorted")
        if not pd.api.types.is_datetime64_any_dtype(panel.index.get_level_values("date")):
            errors.append("date_level_not_datetime")
    missing = sorted(set(spec.required_columns).difference(panel.columns))
    if missing:
        errors.append(f"missing_required_columns:{','.join(missing)}")

    numeric_fields = [
        name
        for name in (
            *RAW_PRICE_COLUMNS,
            *ADJUSTED_PRICE_COLUMNS,
            "volume",
            "amount",
            "adj_factor",
            "listed_days",
            "log_mcap",
            "beta",
            "benchmark_weight",
            "adv20_dollars",
        )
        if name in panel.columns
    ]
    numeric = panel[numeric_fields].apply(pd.to_numeric, errors="coerce") if numeric_fields else pd.DataFrame(index=panel.index)
    research_eligible: pd.Series | None = None
    if "research_eligible" in panel.columns:
        values = panel["research_eligible"]
        if values.isna().any() or not values.dropna().isin([True, False, 0, 1]).all():
            errors.append("boolean_field_invalid:research_eligible")
        else:
            research_eligible = values.astype(bool)
    sparse_when_ineligible = {"log_mcap", "beta", "adv20_dollars"}
    for name in numeric_fields:
        missing_values = numeric[name].isna()
        nonfinite_values = ~np.isfinite(numeric[name].to_numpy(dtype=float))
        if name in sparse_when_ineligible and research_eligible is not None:
            eligible = research_eligible.to_numpy(dtype=bool)
            missing_for_validation = missing_values.to_numpy(dtype=bool) & eligible
            nonfinite_for_validation = nonfinite_values & eligible
            ignored = nonfinite_values & ~eligible
            if ignored.any():
                warnings.append(
                    f"nonfinite_on_research_ineligible:{name}:{int(ignored.sum())}"
                )
        else:
            missing_for_validation = missing_values.to_numpy(dtype=bool)
            nonfinite_for_validation = nonfinite_values
        if missing_for_validation.any():
            errors.append(f"missing_or_non_numeric:{name}")
        if nonfinite_for_validation.any():
            errors.append(f"nonfinite:{name}")
    for name in (*RAW_PRICE_COLUMNS, *ADJUSTED_PRICE_COLUMNS, "adj_factor"):
        if name in numeric and (numeric[name] <= 0).any():
            errors.append(f"nonpositive:{name}")
    for name in ("volume", "amount", "benchmark_weight", "adv20_dollars"):
        if name in numeric and (numeric[name] < 0).any():
            errors.append(f"negative:{name}")
    _validate_ohlc_view(numeric, prefix="", errors=errors, warnings=warnings)
    _validate_ohlc_view(numeric, prefix="adj_", errors=errors, warnings=warnings)

    required_for_adjustment = {*RAW_PRICE_COLUMNS, *ADJUSTED_PRICE_COLUMNS, "adj_factor"}
    if required_for_adjustment.issubset(numeric.columns):
        inconsistent: list[str] = []
        factor = numeric["adj_factor"].to_numpy(dtype=float)
        for raw_name, adjusted_name in RESEARCH_PRICE_MAP.items():
            raw = numeric[raw_name].to_numpy(dtype=float)
            adjusted = numeric[adjusted_name].to_numpy(dtype=float)
            finite = np.isfinite(raw) & np.isfinite(adjusted) & np.isfinite(factor)
            matches = np.isclose(
                adjusted[finite],
                raw[finite] * factor[finite],
                rtol=spec.adjusted_price_tolerance,
                atol=spec.adjusted_price_tolerance,
            )
            if not matches.all():
                inconsistent.append(adjusted_name)
        if inconsistent:
            errors.append(f"adjusted_prices_inconsistent_with_adj_factor:{','.join(inconsistent)}")

    for name in ("tradable", "limit_up", "limit_down", "is_st", "universe_member"):
        if name in panel.columns:
            values = panel[name].dropna()
            if not values.isin([True, False, 0, 1]).all() or panel[name].isna().any():
                errors.append(f"boolean_field_invalid:{name}")
    if {"limit_up", "limit_down"}.issubset(panel.columns):
        both = panel["limit_up"].astype(bool) & panel["limit_down"].astype(bool)
        if both.any():
            errors.append("same_row_both_limit_up_and_down")
    if "industry" in panel.columns and panel["industry"].isna().any():
        errors.append("missing_industry")
    if len(dates) < spec.minimum_history_days:
        errors.append(f"history_shorter_than:{spec.minimum_history_days}")

    if {"benchmark_weight", "universe_member"}.issubset(panel.columns) and len(dates):
        member = panel["universe_member"].astype(bool)
        weights = pd.to_numeric(panel["benchmark_weight"], errors="coerce").where(member, 0.0)
        sums = weights.groupby(level="date").sum()
        nonempty = member.groupby(level="date").any()
        bad = (sums[nonempty] - 1.0).abs() > spec.benchmark_weight_tolerance
        if bad.any():
            errors.append(f"benchmark_weights_not_one_dates:{int(bad.sum())}")

    if spec.require_delisted_history and meta.get("includes_delisted") is not True:
        errors.append("metadata_does_not_confirm_delisted_history")
    if spec.require_historical_universe and meta.get("point_in_time_universe") is not True:
        errors.append("metadata_does_not_confirm_point_in_time_universe")
    if spec.raw_execution_prices and meta.get("execution_price_basis") != "raw_unadjusted":
        errors.append("execution_prices_must_be_raw_unadjusted")
    required_meta = (
        "vendor",
        "extracted_at",
        "corporate_action_policy",
        "calendar",
        "dataset_tier",
        "benchmark_weight_quality",
        "benchmark_weight_source",
        "field_quality",
    )
    for name in required_meta:
        value = meta.get(name)
        if name == "field_quality":
            missing_value = not isinstance(value, Mapping) or not value
        else:
            missing_value = not str(value or "").strip()
        if missing_value:
            errors.append(f"missing_metadata:{name}")

    dataset_tier = str(meta.get("dataset_tier", "")).strip().lower()
    if dataset_tier and dataset_tier not in {"strict", "research_proxy"}:
        errors.append(f"invalid_dataset_tier:{dataset_tier}")
    benchmark_quality = _normalise_quality(meta.get("benchmark_weight_quality"))
    proxy_fields, unknown_fields = _field_quality_findings(meta, spec.required_columns)
    proxy_summary = ",".join(proxy_fields)
    unknown_summary = ",".join(unknown_fields)
    official_historical_weights = benchmark_quality in _OFFICIAL_HISTORICAL_WEIGHT_QUALITIES
    if dataset_tier == "strict":
        if not official_historical_weights:
            errors.append(
                f"strict_requires_official_historical_benchmark_weights:{benchmark_quality}"
            )
        if proxy_fields:
            errors.append(f"strict_required_fields_are_proxy:{proxy_summary}")
        if unknown_fields:
            errors.append(f"strict_required_fields_are_unknown:{unknown_summary}")
    elif dataset_tier == "research_proxy":
        warnings.append("dataset_tier_research_proxy:not_portfolio_grade")
        if not official_historical_weights:
            warnings.append(f"benchmark_weights_not_official_historical:{benchmark_quality}")
        if proxy_fields:
            warnings.append(f"research_proxy_required_fields:{proxy_summary}")
        if unknown_fields:
            warnings.append(f"research_proxy_unknown_fields:{unknown_summary}")

    unique_errors = tuple(dict.fromkeys(errors))
    unique_warnings = tuple(dict.fromkeys(warnings))
    return DataValidationReport(
        valid=not unique_errors,
        rows=len(panel),
        symbols=len(symbols),
        trading_days=len(dates),
        first_date=str(pd.Timestamp(dates[0]).date()) if len(dates) else "",
        last_date=str(pd.Timestamp(dates[-1]).date()) if len(dates) else "",
        snapshot_hash=_panel_snapshot_hash(panel),
        errors=unique_errors,
        warnings=unique_warnings,
        metadata=meta,
    )


def require_valid_daily_panel(
    panel: pd.DataFrame,
    *,
    contract: DataContract | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DataValidationReport:
    report = validate_daily_panel(panel, contract=contract, metadata=metadata)
    if not report.valid:
        raise ValueError("daily panel contract failed: " + "; ".join(report.errors))
    return report


__all__ = [
    "ADJUSTED_PRICE_COLUMNS",
    "DataContract",
    "DataValidationReport",
    "RAW_PRICE_COLUMNS",
    "RESEARCH_PRICE_MAP",
    "require_valid_daily_panel",
    "validate_daily_panel",
]
