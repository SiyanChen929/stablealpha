"""Pure transformations for a no-cost, point-in-time HS300 research panel.

Network downloads deliberately live outside this module.  Functions here take
already captured BaoStock-shaped frames, enforce their temporal semantics, and
derive only backward-looking fields.  Raw cash execution prices and
backward-adjusted research prices remain separate objects by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..data_quality import (
    AuditMode,
    DataQualityError,
    RESEARCH_PROXY_WATERMARK,
    audit_hs300_fields,
    hs300_field_quality_map,
)
from .baostock import ADJUST_FACTOR_FIELDS


RAW_ADJUSTFLAG = "3"
BACKWARD_ADJUSTFLAG = "1"
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class MembershipSnapshotAudit:
    """Completeness report for weekly historical constituent snapshots."""

    valid: bool
    snapshots: int
    first_snapshot: str
    last_snapshot: str
    minimum_members: int
    maximum_members: int
    counts_by_snapshot: Mapping[str, int]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def require_valid(self) -> MembershipSnapshotAudit:
        if not self.valid:
            raise ValueError("HS300 membership snapshots failed: " + "; ".join(self.errors))
        return self


@dataclass(frozen=True)
class HS300PriceViews:
    """Physically separate cash-execution and research-price views."""

    raw_execution: pd.DataFrame
    backward_adjusted_research: pd.DataFrame
    execution_price_basis: str = "raw_unadjusted"
    research_price_basis: str = "backward_adjusted"


@dataclass(frozen=True)
class PriceHistoryBoundary:
    """Leading vendor history excluded before adjusted lineage becomes valid.

    BaoStock can return predecessor/IPO rows marked ``adjustflag=3`` inside a
    request for ``adjustflag=1``.  Those rows are usable as raw observations,
    but they do not have the requested adjusted-price lineage.  They may be
    excluded only when every affected row predates the security's first
    point-in-time index membership.
    """

    symbol: str
    first_source_date: str
    first_adjusted_date: str
    last_excluded_date: str
    excluded_rows: int
    first_member_date: str
    policy: str = "trim_leading_nonadjusted_rows_before_first_membership"


@dataclass(frozen=True)
class BackwardAdjustmentAudit:
    """Audit the time direction of a vendor backward-adjusted price path.

    The audit deliberately does not infer the absolute first-day anchor from
    the adjusted series.  It takes that first positive scale as given and then
    verifies that every subsequent scale change is explained by information
    present in the raw tape: previous raw close divided by today's raw
    ``preclose``.  Consequently a later corporate action cannot silently
    rewrite an earlier relative return path.
    """

    valid: bool
    symbols: int
    rows: int
    corporate_action_events: int
    maximum_absolute_error: float
    maximum_relative_error: float
    errors: tuple[str, ...]

    def require_valid(self) -> BackwardAdjustmentAudit:
        if not self.valid:
            raise ValueError(
                "backward-adjustment causality failed: " + "; ".join(self.errors)
            )
        return self


@dataclass(frozen=True)
class VendorAdjustmentEventAudit:
    """Audit an adjusted tape against point-in-time vendor factor events.

    ``backAdjustFactor`` is treated as a cumulative vendor state, not as a
    return known before ``dividOperateDate``.  The state applied to a market
    row is therefore the last event whose operate date is no later than that
    row.  A per-security positive level anchor is immaterial; changes in that
    level must nevertheless be exactly explained by the as-of vendor state.

    ``raw_preclose_recurrence_*`` retains the independent raw-tape diagnostic.
    It is deliberately reported separately so callers can see whether an
    event-table failure, a raw recurrence failure, or both occurred.
    """

    valid: bool
    symbols: int
    rows: int
    event_rows: int
    factor_changes: int
    future_event_rows_ignored: int
    maximum_common_scale_relative_error: float
    maximum_event_path_relative_error: float
    raw_preclose_recurrence_valid: bool
    raw_preclose_recurrence_errors: tuple[str, ...]
    errors: tuple[str, ...]

    def require_valid(self) -> VendorAdjustmentEventAudit:
        if not self.valid:
            raise ValueError(
                "vendor adjustment-event as-of audit failed: "
                + "; ".join(self.errors)
            )
        return self


def _coerce_dates(values: Iterable[object], *, name: str) -> pd.DatetimeIndex:
    converted = pd.to_datetime(pd.Index(values), errors="coerce").normalize()
    if converted.isna().any():
        raise ValueError(f"{name} contains invalid dates")
    return pd.DatetimeIndex(converted, name=name)


def _require_panel_index(frame: pd.DataFrame, *, label: str) -> None:
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.names != ["date", "symbol"]:
        raise ValueError(f"{label} must use MultiIndex(date, symbol)")
    if not frame.index.is_unique:
        raise ValueError(f"{label} contains duplicate date-symbol rows")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} must be sorted")
    if not pd.api.types.is_datetime64_any_dtype(frame.index.get_level_values("date")):
        raise ValueError(f"{label} date index must be datetime64")


def _normalise_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(snapshots, pd.DataFrame):
        raise TypeError("snapshots must be a pandas DataFrame")
    date_column = next(
        (
            name
            for name in ("snapshot_date", "vendor_update_date", "updateDate", "date")
            if name in snapshots.columns
        ),
        None,
    )
    symbol_column = next((name for name in ("symbol", "code") if name in snapshots.columns), None)
    if date_column is None or symbol_column is None:
        raise ValueError(
            "snapshots require vendor_update_date/updateDate/snapshot_date and code/symbol"
        )
    selected = snapshots.copy()
    selected = selected.rename(columns={date_column: "snapshot_date", symbol_column: "symbol"})
    selected["snapshot_date"] = pd.to_datetime(selected["snapshot_date"], errors="coerce").dt.normalize()
    selected["symbol"] = selected["symbol"].astype("string").str.strip()
    if selected["snapshot_date"].isna().any():
        raise ValueError("membership snapshot contains invalid updateDate")
    if selected["symbol"].isna().any() or selected["symbol"].eq("").any():
        raise ValueError("membership snapshot contains blank symbol")
    if "code_name" not in selected.columns:
        selected["code_name"] = ""
    selected["code_name"] = selected["code_name"].astype("string").fillna("")
    return selected[["snapshot_date", "symbol", "code_name"]].sort_values(
        ["snapshot_date", "symbol"], kind="stable"
    ).reset_index(drop=True)


def validate_membership_snapshots(
    snapshots: pd.DataFrame,
    *,
    expected_members: int = 300,
    trading_dates: Iterable[object] | None = None,
    max_snapshot_gap_days: int = 21,
    max_snapshot_age_days: int = 21,
) -> MembershipSnapshotAudit:
    """Validate size, uniqueness, cadence and requested-calendar coverage.

    BaoStock publishes one weekly snapshot.  A trading date is covered only by
    the latest snapshot whose ``updateDate`` is no later than that date.  This
    as-of rule prevents future index additions from leaking backwards.
    """

    if expected_members <= 0:
        raise ValueError("expected_members must be positive")
    if max_snapshot_gap_days <= 0 or max_snapshot_age_days < 0:
        raise ValueError("snapshot gap and age limits must be non-negative")
    try:
        normalised = _normalise_snapshots(snapshots)
    except (TypeError, ValueError) as exc:
        return MembershipSnapshotAudit(False, 0, "", "", 0, 0, {}, (str(exc),), ())
    if normalised.empty:
        return MembershipSnapshotAudit(False, 0, "", "", 0, 0, {}, ("no_membership_snapshots",), ())

    grouped = normalised.groupby("snapshot_date", sort=True)
    counts = grouped["symbol"].size()
    duplicates = grouped["symbol"].apply(lambda values: bool(values.duplicated().any()))
    dates = pd.DatetimeIndex(counts.index).sort_values()
    errors: list[str] = []
    warnings: list[str] = []
    bad_size = counts[counts != expected_members]
    if len(bad_size):
        encoded = ",".join(f"{pd.Timestamp(date).date()}={int(value)}" for date, value in bad_size.items())
        errors.append(f"snapshot_member_count_not_{expected_members}:{encoded}")
    duplicate_dates = duplicates[duplicates].index
    if len(duplicate_dates):
        errors.append(
            "duplicate_symbols_in_snapshots:"
            + ",".join(str(pd.Timestamp(date).date()) for date in duplicate_dates)
        )
    if len(dates) > 1:
        gaps = pd.Series(dates[1:] - dates[:-1], index=dates[1:]).dt.days
        bad_gaps = gaps[gaps > max_snapshot_gap_days]
        if len(bad_gaps):
            errors.append(
                "membership_snapshot_gap_too_large:"
                + ",".join(f"{date.date()}={int(days)}" for date, days in bad_gaps.items())
            )

    if trading_dates is not None:
        calendar = _coerce_dates(trading_dates, name="date").unique().sort_values()
        if len(calendar):
            locations = dates.searchsorted(calendar, side="right") - 1
            if (locations < 0).any():
                errors.append(f"calendar_precedes_first_snapshot:{calendar[locations < 0][0].date()}")
            covered = locations >= 0
            if covered.any():
                ages = (calendar[covered] - dates[locations[covered]]).days
                if (ages > max_snapshot_age_days).any():
                    first_bad = int(np.flatnonzero(ages > max_snapshot_age_days)[0])
                    errors.append(
                        "membership_snapshot_stale:"
                        f"{calendar[covered][first_bad].date()}={int(ages[first_bad])}_days"
                    )

    count_map = {str(pd.Timestamp(date).date()): int(value) for date, value in counts.items()}
    return MembershipSnapshotAudit(
        valid=not errors,
        snapshots=len(dates),
        first_snapshot=str(dates[0].date()),
        last_snapshot=str(dates[-1].date()),
        minimum_members=int(counts.min()),
        maximum_members=int(counts.max()),
        counts_by_snapshot=count_map,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def expand_membership_snapshots(
    snapshots: pd.DataFrame,
    trading_dates: Iterable[object],
    *,
    expected_members: int = 300,
    max_snapshot_gap_days: int = 21,
    max_snapshot_age_days: int = 21,
) -> pd.DataFrame:
    """Expand weekly snapshots into a members-only daily as-of table."""

    calendar = _coerce_dates(trading_dates, name="date").unique().sort_values()
    audit = validate_membership_snapshots(
        snapshots,
        expected_members=expected_members,
        trading_dates=calendar,
        max_snapshot_gap_days=max_snapshot_gap_days,
        max_snapshot_age_days=max_snapshot_age_days,
    ).require_valid()
    del audit
    normalised = _normalise_snapshots(snapshots)
    snapshot_dates = pd.DatetimeIndex(normalised["snapshot_date"].unique()).sort_values()
    grouped = {
        pd.Timestamp(date): group[["symbol", "code_name"]].reset_index(drop=True)
        for date, group in normalised.groupby("snapshot_date", sort=True)
    }
    frames: list[pd.DataFrame] = []
    locations = snapshot_dates.searchsorted(calendar, side="right") - 1
    for date, location in zip(calendar, locations, strict=True):
        snapshot_date = pd.Timestamp(snapshot_dates[location])
        daily = grouped[snapshot_date].copy()
        daily.insert(0, "date", pd.Timestamp(date))
        daily["membership_snapshot_date"] = snapshot_date
        daily["universe_member"] = True
        frames.append(daily)
    if not frames:
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([], name="date"), pd.Index([], name="symbol")],
            names=["date", "symbol"],
        )
        return pd.DataFrame(
            columns=["code_name", "membership_snapshot_date", "universe_member"],
            index=empty_index,
        )
    expanded = pd.concat(frames, ignore_index=True)
    return expanded.set_index(["date", "symbol"]).sort_index()


def attach_membership(panel: pd.DataFrame, expanded_membership: pd.DataFrame) -> pd.DataFrame:
    """Attach daily member flags to any complete date-symbol market panel."""

    _require_panel_index(panel, label="panel")
    _require_panel_index(expanded_membership, label="expanded membership")
    output = panel.copy()
    output["universe_member"] = output.index.isin(expanded_membership.index)
    snapshot_dates = expanded_membership.get("membership_snapshot_date")
    if snapshot_dates is not None:
        output["membership_snapshot_date"] = snapshot_dates.reindex(output.index)
    return output


def normalize_baostock_daily_fields(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize only documented empty suspension liquidity fields.

    BaoStock may encode ``volume`` and ``amount`` as empty strings on a row
    whose ``tradestatus`` is zero.  Those two values are safely normalized to
    zero.  Empty or malformed liquidity on a tradable row is a hard failure;
    silently filling it would fabricate capacity and VWAP observations.
    """

    _require_panel_index(raw, label="raw price view")
    required = {"tradestatus", "volume", "amount", "isST"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("raw BaoStock view missing: " + ",".join(missing))
    output = raw.copy()
    status = pd.to_numeric(output["tradestatus"], errors="coerce")
    if status.isna().any() or not status.isin([0, 1]).all():
        raise ValueError("tradestatus must be exactly 0 or 1")
    is_st = pd.to_numeric(output["isST"], errors="coerce")
    if is_st.isna().any() or not is_st.isin([0, 1]).all():
        raise ValueError("isST must be exactly 0 or 1")
    output["isST"] = is_st.astype("int8")
    suspended = status.eq(0)
    parsed: dict[str, pd.Series] = {}
    for field in ("volume", "amount"):
        original = output[field]
        text = original.astype("string").str.strip()
        blank = original.isna() | text.eq("")
        numeric = pd.to_numeric(original, errors="coerce")
        malformed = numeric.isna() & ~blank
        if malformed.any():
            raise ValueError(f"{field} contains malformed non-empty values")
        if (blank & ~suspended).any():
            raise ValueError(f"tradable row has missing {field}")
        numeric = numeric.mask(blank & suspended, 0.0)
        if numeric.isna().any() or (numeric < 0).any():
            raise ValueError(f"{field} must be non-negative numeric")
        parsed[field] = numeric.astype(float)

    # A second provider edge case marks an otherwise coherent positive-value
    # transaction row as suspended.  The trade tape is stronger evidence than
    # that single status bit.  Correct it only when volume and amount are both
    # positive and the implied transaction-average price lies inside OHLC.
    stale_status = suspended & parsed["volume"].gt(0) & parsed["amount"].gt(0)
    if stale_status.any():
        prices = output[["low", "high"]].apply(pd.to_numeric, errors="coerce")
        implied = parsed["amount"].div(parsed["volume"].where(parsed["volume"].gt(0)))
        coherent = (
            prices.notna().all(axis=1)
            & prices["low"].gt(0)
            & prices["high"].ge(prices["low"])
            & implied.ge(prices["low"])
            & implied.le(prices["high"])
        )
        if not coherent.loc[stale_status].all():
            raise ValueError(
                "suspended positive-transaction row is not coherent with OHLC"
            )
        status = status.mask(stale_status, 1)
        suspended = status.eq(0)

    # BaoStock has a documented-in-cache edge case where the first suspended
    # row carries forward the prior day's positive volume/turn while amount is
    # zero and OHLC equals preclose.  Treating that stale value as capacity
    # would be optimistic.  Normalize it to zero only under the full
    # no-transaction signature; any non-flat or positive-amount case fails.
    stale_volume = suspended & parsed["volume"].gt(0) & parsed["amount"].eq(0)
    if stale_volume.any():
        flat_fields = output[["open", "high", "low", "close", "preclose"]].apply(
            pd.to_numeric, errors="coerce"
        )
        flat = flat_fields.notna().all(axis=1) & flat_fields.eq(
            flat_fields["close"], axis=0
        ).all(axis=1)
        if not flat.loc[stale_volume].all():
            raise ValueError(
                "suspended positive-volume zero-amount row is not a flat carry-forward"
            )
        parsed["volume"] = parsed["volume"].mask(stale_volume, 0.0)
    if (parsed["volume"].eq(0) & parsed["amount"].gt(0)).any():
        raise ValueError("positive transaction amount with zero volume")
    output["tradestatus"] = status.astype("int8")
    output["volume"] = parsed["volume"]
    output["amount"] = parsed["amount"]
    return output


def separate_price_views(raw: pd.DataFrame, backward_adjusted: pd.DataFrame) -> HS300PriceViews:
    """Validate and isolate BaoStock adjustflag=3 and adjustflag=1 views."""

    _require_panel_index(raw, label="raw price view")
    _require_panel_index(backward_adjusted, label="backward-adjusted price view")
    raw_required = set(PRICE_COLUMNS).union(
        {"preclose", "volume", "amount", "tradestatus", "isST", "adjustflag"}
    )
    research_required = set(PRICE_COLUMNS).union({"preclose", "adjustflag"})
    missing_raw = sorted(raw_required.difference(raw.columns))
    missing_research = sorted(research_required.difference(backward_adjusted.columns))
    if missing_raw:
        raise ValueError("raw price view missing: " + ",".join(missing_raw))
    if missing_research:
        raise ValueError("backward-adjusted price view missing: " + ",".join(missing_research))
    if not raw.index.equals(backward_adjusted.index):
        raise ValueError("raw and backward-adjusted views must have identical indexes")
    raw_flags = raw["adjustflag"].astype("string").str.strip().dropna().unique()
    research_flags = backward_adjusted["adjustflag"].astype("string").str.strip().dropna().unique()
    if len(raw_flags) != 1 or raw_flags[0] != RAW_ADJUSTFLAG:
        raise ValueError("raw execution view must contain only adjustflag=3")
    if len(research_flags) != 1 or research_flags[0] != BACKWARD_ADJUSTFLAG:
        raise ValueError("research view must contain only adjustflag=1 (backward-adjusted)")
    for label, frame in (("raw", raw), ("backward-adjusted", backward_adjusted)):
        numeric = frame[list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or (numeric <= 0).any().any():
            raise ValueError(f"{label} price view contains missing or nonpositive OHLC")
    normalized_raw = normalize_baostock_daily_fields(raw)
    return HS300PriceViews(normalized_raw, backward_adjusted.copy())


def trim_leading_unadjusted_vendor_history(
    raw: pd.DataFrame,
    requested_backward_adjusted: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[PriceHistoryBoundary, ...]]:
    """Exclude only provider-declared leading rows without adjusted lineage.

    The requested adjusted response must contain one monotone boundary:
    optional leading ``adjustflag=3`` rows followed by only ``adjustflag=1``
    rows.  Every leading row must exactly equal the independently committed raw
    view, and all excluded dates must precede the first point-in-time index
    membership.  Interior/reverting flags, altered leading values, or a member
    row before the adjusted boundary fail closed.
    """

    _require_panel_index(raw, label="raw price view")
    _require_panel_index(
        requested_backward_adjusted,
        label="requested backward-adjusted price view",
    )
    if not raw.index.equals(requested_backward_adjusted.index):
        raise ValueError("raw and requested adjusted views must have identical indexes")
    required = {*PRICE_COLUMNS, "preclose", "adjustflag"}
    missing_raw = sorted(required.difference(raw.columns))
    missing_adjusted = sorted(required.difference(requested_backward_adjusted.columns))
    if missing_raw or missing_adjusted:
        detail = []
        if missing_raw:
            detail.append("raw_missing:" + ",".join(missing_raw))
        if missing_adjusted:
            detail.append("adjusted_missing:" + ",".join(missing_adjusted))
        raise ValueError("mixed adjusted history schema invalid: " + ";".join(detail))
    raw_flags = raw["adjustflag"].astype("string").str.strip()
    if not raw_flags.eq(RAW_ADJUSTFLAG).all():
        raise ValueError("raw execution view must contain only adjustflag=3")
    if not {"date", "code"}.issubset(membership.columns):
        raise ValueError("membership requires date and code for price-boundary validation")
    member = membership[["date", "code"]].copy()
    member["date"] = pd.to_datetime(member["date"], errors="coerce").dt.normalize()
    member["code"] = member["code"].astype("string").str.strip()
    if member.isna().any().any() or member.duplicated(["date", "code"]).any():
        raise ValueError("membership is invalid for price-boundary validation")
    first_member = member.groupby("code", sort=False)["date"].min()

    keep = pd.Series(True, index=raw.index, dtype=bool)
    boundaries: list[PriceHistoryBoundary] = []
    comparison_fields = (*PRICE_COLUMNS, "preclose")
    for symbol, adjusted_symbol in requested_backward_adjusted.groupby(
        level="symbol", sort=False
    ):
        adjusted_symbol = adjusted_symbol.sort_index(level="date")
        flags = adjusted_symbol["adjustflag"].astype("string").str.strip()
        observed = set(flags.dropna().astype(str))
        if not observed or not observed.issubset({RAW_ADJUSTFLAG, BACKWARD_ADJUSTFLAG}):
            raise ValueError(f"requested adjusted payload flags are invalid for {symbol}")
        if observed == {BACKWARD_ADJUSTFLAG}:
            continue
        adjusted_positions = np.flatnonzero(
            flags.eq(BACKWARD_ADJUSTFLAG).to_numpy(dtype=bool)
        )
        if not len(adjusted_positions):
            raise ValueError(f"requested adjusted history never becomes adjusted for {symbol}")
        boundary_position = int(adjusted_positions[0])
        if boundary_position < 1:
            raise ValueError(f"mixed adjusted history boundary is malformed for {symbol}")
        if not flags.iloc[:boundary_position].eq(RAW_ADJUSTFLAG).all() or not flags.iloc[
            boundary_position:
        ].eq(BACKWARD_ADJUSTFLAG).all():
            raise ValueError(f"requested adjusted flags revert or interleave for {symbol}")

        leading_index = adjusted_symbol.index[:boundary_position]
        raw_leading = raw.loc[leading_index, list(comparison_fields)].apply(
            pd.to_numeric, errors="coerce"
        )
        adjusted_leading = adjusted_symbol.loc[
            leading_index, list(comparison_fields)
        ].apply(pd.to_numeric, errors="coerce")
        if (
            raw_leading.isna().any().any()
            or adjusted_leading.isna().any().any()
            or not np.array_equal(
                raw_leading.to_numpy(dtype=float),
                adjusted_leading.to_numpy(dtype=float),
            )
        ):
            raise ValueError(
                f"leading adjustflag=3 rows differ from committed raw history for {symbol}"
            )
        symbol_text = str(symbol)
        if symbol_text not in first_member.index:
            raise ValueError(f"membership has no observation for {symbol_text}")
        boundary_date = pd.Timestamp(
            adjusted_symbol.index[boundary_position][0]
        ).normalize()
        member_date = pd.Timestamp(first_member.loc[symbol_text]).normalize()
        excluded_dates = pd.DatetimeIndex(
            leading_index.get_level_values("date")
        ).normalize()
        if member_date < boundary_date or (excluded_dates >= member_date).any():
            raise ValueError(
                f"adjusted lineage starts after point-in-time membership for {symbol_text}"
            )
        keep.loc[leading_index] = False
        boundaries.append(
            PriceHistoryBoundary(
                symbol=symbol_text,
                first_source_date=str(excluded_dates.min().date()),
                first_adjusted_date=str(boundary_date.date()),
                last_excluded_date=str(excluded_dates.max().date()),
                excluded_rows=len(leading_index),
                first_member_date=str(member_date.date()),
            )
        )

    trimmed_raw = raw.loc[keep].copy().sort_index()
    trimmed_adjusted = requested_backward_adjusted.loc[keep].copy().sort_index()
    if not trimmed_raw.index.equals(trimmed_adjusted.index):
        raise AssertionError("trimmed raw and adjusted indexes differ")
    adjusted_flags = (
        trimmed_adjusted["adjustflag"].astype("string").str.strip().dropna().unique()
    )
    if len(adjusted_flags) != 1 or adjusted_flags[0] != BACKWARD_ADJUSTFLAG:
        raise ValueError("trimmed research view must contain only adjustflag=1")
    return trimmed_raw, trimmed_adjusted, tuple(sorted(boundaries, key=lambda item: item.symbol))


def audit_backward_adjustment_causality(
    raw: pd.DataFrame,
    backward_adjusted: pd.DataFrame,
    *,
    relative_tolerance: float = 2e-6,
    absolute_tolerance: float = 2e-5,
) -> BackwardAdjustmentAudit:
    """Verify BaoStock's causal, IPO-anchored adjustment recurrence.

    BaoStock documents the event multiplier as ``previous close / current
    preclose``.  Starting from the observed first-day factor, this function
    reconstructs the complete later factor path independently of the vendor's
    adjusted prices and checks raw ``OHLC/preclose * factor`` against them.

    The first-day constant is intentionally not treated as predictive
    information.  Price-level Factor DSL expressions are separately required
    to pass a per-security scale-invariance gate, so neither a different IPO
    anchor nor a vendor anchor revision can change an accepted factor.
    """

    if relative_tolerance <= 0 or absolute_tolerance < 0:
        raise ValueError("adjustment audit tolerances must be positive")
    _require_panel_index(raw, label="raw price view")
    _require_panel_index(backward_adjusted, label="backward-adjusted price view")
    if not raw.index.equals(backward_adjusted.index):
        return BackwardAdjustmentAudit(
            False, 0, 0, 0, float("inf"), float("inf"),
            ("raw_and_adjusted_indexes_differ",),
        )
    if "adjustflag" not in raw.columns or "adjustflag" not in backward_adjusted.columns:
        return BackwardAdjustmentAudit(
            False, 0, len(raw), 0, float("inf"), float("inf"),
            ("missing_adjustflag",),
        )
    raw_flags = raw["adjustflag"].astype("string").str.strip().dropna().unique()
    adjusted_flags = (
        backward_adjusted["adjustflag"].astype("string").str.strip().dropna().unique()
    )
    if len(raw_flags) != 1 or raw_flags[0] != RAW_ADJUSTFLAG:
        return BackwardAdjustmentAudit(
            False, 0, len(raw), 0, float("inf"), float("inf"),
            ("raw_adjustflag_must_be_3",),
        )
    if len(adjusted_flags) != 1 or adjusted_flags[0] != BACKWARD_ADJUSTFLAG:
        return BackwardAdjustmentAudit(
            False, 0, len(raw), 0, float("inf"), float("inf"),
            ("adjusted_adjustflag_must_be_1",),
        )
    fields = (*PRICE_COLUMNS, "preclose")
    missing_raw = sorted(set(fields).difference(raw.columns))
    missing_adjusted = sorted(set(fields).difference(backward_adjusted.columns))
    if missing_raw or missing_adjusted:
        errors = []
        if missing_raw:
            errors.append("raw_missing:" + ",".join(missing_raw))
        if missing_adjusted:
            errors.append("adjusted_missing:" + ",".join(missing_adjusted))
        return BackwardAdjustmentAudit(
            False, 0, len(raw), 0, float("inf"), float("inf"), tuple(errors)
        )

    raw_numeric = raw[list(fields)].apply(pd.to_numeric, errors="coerce")
    adjusted_numeric = backward_adjusted[list(fields)].apply(
        pd.to_numeric, errors="coerce"
    )
    raw_finite = pd.Series(
        np.isfinite(raw_numeric.to_numpy(dtype=float)).all(axis=1),
        index=raw_numeric.index,
    )
    adjusted_finite = pd.Series(
        np.isfinite(adjusted_numeric.to_numpy(dtype=float)).all(axis=1),
        index=adjusted_numeric.index,
    )
    invalid = (
        raw_numeric.isna().any(axis=1)
        | adjusted_numeric.isna().any(axis=1)
        | ~raw_finite
        | ~adjusted_finite
        | raw_numeric.le(0).any(axis=1)
        | adjusted_numeric.le(0).any(axis=1)
    )
    if invalid.any():
        first = invalid[invalid].index[0]
        return BackwardAdjustmentAudit(
            False,
            int(raw.index.get_level_values("symbol").nunique()),
            len(raw),
            0,
            float("inf"),
            float("inf"),
            (f"nonfinite_nonpositive_or_missing_price:{first}",),
        )

    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    corporate_action_events = 0
    errors: list[str] = []
    symbols: list[object] = []
    for symbol, raw_symbol in raw_numeric.groupby(level="symbol", sort=False):
        symbols.append(symbol)
        raw_symbol = raw_symbol.sort_index(level="date")
        adjusted_symbol = adjusted_numeric.loc[raw_symbol.index]
        first_factor = float(
            adjusted_symbol["close"].iloc[0] / raw_symbol["close"].iloc[0]
        )
        event_multiplier = raw_symbol["close"].shift(1).div(raw_symbol["preclose"])
        event_multiplier.iloc[0] = 1.0
        if (
            event_multiplier.isna().any()
            or not np.isfinite(event_multiplier.to_numpy(dtype=float)).all()
            or event_multiplier.le(0).any()
        ):
            errors.append(f"invalid_event_multiplier:{symbol}")
            continue
        corporate_action_events += int(
            (~np.isclose(
                event_multiplier.to_numpy(dtype=float),
                1.0,
                rtol=0.0,
                atol=5e-7,
            )).sum()
        )
        causal_factor = event_multiplier.cumprod().mul(first_factor)
        for field in fields:
            expected = raw_symbol[field].mul(causal_factor)
            observed = adjusted_symbol[field]
            absolute_error = expected.sub(observed).abs()
            relative_error = absolute_error.div(observed.abs().clip(lower=1e-12))
            field_absolute = float(absolute_error.max())
            field_relative = float(relative_error.max())
            maximum_absolute_error = max(maximum_absolute_error, field_absolute)
            maximum_relative_error = max(maximum_relative_error, field_relative)
            close_enough = np.isclose(
                expected.to_numpy(dtype=float),
                observed.to_numpy(dtype=float),
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            if not close_enough.all():
                position = int(np.flatnonzero(~close_enough)[0])
                date = raw_symbol.index[position][0]
                errors.append(
                    f"causal_path_mismatch:{symbol}:{pd.Timestamp(date).date()}:{field}"
                )
                break

    return BackwardAdjustmentAudit(
        valid=not errors,
        symbols=len(symbols),
        rows=len(raw),
        corporate_action_events=corporate_action_events,
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        errors=tuple(errors),
    )


def audit_vendor_adjustment_events_asof(
    raw: pd.DataFrame,
    backward_adjusted: pd.DataFrame,
    adjustment_events: pd.DataFrame,
    *,
    event_catalog_symbols: Iterable[object] | None = None,
    relative_tolerance: float = 2e-6,
    absolute_tolerance: float = 2e-5,
) -> VendorAdjustmentEventAudit:
    """Fail closed on vendor factor timing and adjusted-price consistency.

    The event input must have BaoStock's complete ``ADJUST_FACTOR_FIELDS``
    schema.  For every security, the adjusted/raw ratio must be a common daily
    scale across OHLC and ``preclose``.  If events are present, that scale may
    change only as ``anchor * last_asof(backAdjustFactor)``.  Thus a factor
    dated tomorrow cannot affect today's adjusted observations.  A completely
    catalogued zero-row response is accepted only when that security has one
    constant scale over its observed history.  ``event_catalog_symbols`` is
    mandatory because an empty successful response must not be confused with
    a request that was never made.

    The preclose recurrence audit remains an independent diagnostic and uses
    its original tolerances unchanged.  It does not determine this audit's
    validity because the independently committed event catalog is authoritative
    whenever it is supplied.
    """

    if relative_tolerance <= 0 or absolute_tolerance < 0:
        raise ValueError("adjustment audit tolerances must be positive")

    recurrence = audit_backward_adjustment_causality(
        raw,
        backward_adjusted,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    errors: list[str] = []
    symbols = 0
    rows = len(raw) if isinstance(raw, pd.DataFrame) else 0
    event_rows = (
        len(adjustment_events)
        if isinstance(adjustment_events, pd.DataFrame)
        else 0
    )
    factor_changes = 0
    future_event_rows_ignored = 0
    maximum_common_scale_relative_error = 0.0
    maximum_event_path_relative_error = 0.0

    def finish() -> VendorAdjustmentEventAudit:
        combined = list(dict.fromkeys(errors))
        return VendorAdjustmentEventAudit(
            valid=not combined,
            symbols=symbols,
            rows=rows,
            event_rows=event_rows,
            factor_changes=factor_changes,
            future_event_rows_ignored=future_event_rows_ignored,
            maximum_common_scale_relative_error=maximum_common_scale_relative_error,
            maximum_event_path_relative_error=maximum_event_path_relative_error,
            raw_preclose_recurrence_valid=recurrence.valid,
            raw_preclose_recurrence_errors=recurrence.errors,
            errors=tuple(combined),
        )

    if not isinstance(raw, pd.DataFrame) or not isinstance(
        backward_adjusted, pd.DataFrame
    ):
        errors.append("raw_and_adjusted_must_be_dataframes")
        return finish()
    try:
        _require_panel_index(raw, label="raw price view")
        _require_panel_index(backward_adjusted, label="backward-adjusted price view")
    except ValueError as exc:
        errors.append(str(exc))
        return finish()
    if not raw.index.equals(backward_adjusted.index):
        errors.append("raw_and_adjusted_indexes_differ")
        return finish()
    symbols = int(raw.index.get_level_values("symbol").nunique())

    fields = (*PRICE_COLUMNS, "preclose")
    missing_raw = sorted(set((*fields, "adjustflag")).difference(raw.columns))
    missing_adjusted = sorted(
        set((*fields, "adjustflag")).difference(backward_adjusted.columns)
    )
    if missing_raw:
        errors.append("raw_missing:" + ",".join(missing_raw))
    if missing_adjusted:
        errors.append("adjusted_missing:" + ",".join(missing_adjusted))
    if errors:
        return finish()
    raw_flags = raw["adjustflag"].astype("string").str.strip().dropna().unique()
    adjusted_flags = (
        backward_adjusted["adjustflag"]
        .astype("string")
        .str.strip()
        .dropna()
        .unique()
    )
    if len(raw_flags) != 1 or raw_flags[0] != RAW_ADJUSTFLAG:
        errors.append("raw_adjustflag_must_be_3")
    if len(adjusted_flags) != 1 or adjusted_flags[0] != BACKWARD_ADJUSTFLAG:
        errors.append("adjusted_adjustflag_must_be_1")
    if errors:
        return finish()

    raw_numeric = raw[list(fields)].apply(pd.to_numeric, errors="coerce")
    adjusted_numeric = backward_adjusted[list(fields)].apply(
        pd.to_numeric, errors="coerce"
    )
    finite_positive = (
        raw_numeric.notna().all(axis=1)
        & adjusted_numeric.notna().all(axis=1)
        & pd.Series(
            np.isfinite(raw_numeric.to_numpy(dtype=float)).all(axis=1),
            index=raw.index,
        )
        & pd.Series(
            np.isfinite(adjusted_numeric.to_numpy(dtype=float)).all(axis=1),
            index=raw.index,
        )
        & raw_numeric.gt(0).all(axis=1)
        & adjusted_numeric.gt(0).all(axis=1)
    )
    if not finite_positive.all():
        first_invalid = finite_positive[~finite_positive].index[0]
        errors.append(f"nonfinite_nonpositive_or_missing_price:{first_invalid}")
        return finish()

    if not isinstance(adjustment_events, pd.DataFrame):
        errors.append("adjustment_events_must_be_dataframe")
        return finish()
    if event_catalog_symbols is None:
        errors.append("event_catalog_symbols_required")
        return finish()
    catalog_values = pd.Series(list(event_catalog_symbols), dtype="string").str.strip()
    if (
        catalog_values.isna().any()
        or catalog_values.eq("").any()
        or catalog_values.duplicated().any()
    ):
        errors.append("event_catalog_symbols_invalid_or_duplicate")
        return finish()
    catalog_symbols = set(catalog_values.astype(str))
    raw_symbols = set(raw.index.get_level_values("symbol").astype(str))
    missing_catalog_symbols = sorted(raw_symbols.difference(catalog_symbols))
    if missing_catalog_symbols:
        errors.append(
            "event_catalog_symbol_missing:"
            + ",".join(missing_catalog_symbols[:10])
        )
        return finish()
    missing_event_fields = [
        field for field in ADJUST_FACTOR_FIELDS if field not in adjustment_events.columns
    ]
    if missing_event_fields:
        errors.append(
            "adjustment_events_missing:" + ",".join(missing_event_fields)
        )
        return finish()
    events = adjustment_events[list(ADJUST_FACTOR_FIELDS)].copy()
    events["code"] = events["code"].astype("string").str.strip()
    events["dividOperateDate"] = pd.to_datetime(
        events["dividOperateDate"], errors="coerce"
    ).dt.normalize()
    factor_fields = ("foreAdjustFactor", "backAdjustFactor", "adjustFactor")
    for field in factor_fields:
        events[field] = pd.to_numeric(events[field], errors="coerce")
    invalid_events = (
        events["code"].isna().any()
        or events["code"].eq("").any()
        or events["dividOperateDate"].isna().any()
        or events[list(factor_fields)].isna().any().any()
        or not np.isfinite(events[list(factor_fields)].to_numpy(dtype=float)).all()
        or events[list(factor_fields)].le(0).any().any()
    )
    if invalid_events:
        errors.append("adjustment_events_invalid_code_date_or_factor")
        return finish()
    if events.duplicated(["code", "dividOperateDate"]).any():
        errors.append("adjustment_events_duplicate_code_operate_date")
        return finish()
    uncatalogued_event_symbols = sorted(
        set(events["code"].dropna().astype(str)).difference(catalog_symbols)
    )
    if uncatalogued_event_symbols:
        errors.append(
            "adjustment_event_without_catalog_request:"
            + ",".join(uncatalogued_event_symbols[:10])
        )
        return finish()
    events = events.sort_values(["code", "dividOperateDate"], kind="stable")

    # First establish that there is one, and only one, adjusted/raw scale on
    # each market row.  The close-derived scale is merely an anchor for this
    # within-row identity; it is not used to establish its own event timing.
    observed_scale = adjusted_numeric["close"].div(raw_numeric["close"])
    for field in fields:
        expected = raw_numeric[field].mul(observed_scale)
        observed = adjusted_numeric[field]
        relative_error = expected.sub(observed).abs().div(
            observed.abs().clip(lower=1e-12)
        )
        maximum_common_scale_relative_error = max(
            maximum_common_scale_relative_error, float(relative_error.max())
        )
        close_enough = np.isclose(
            expected.to_numpy(dtype=float),
            observed.to_numpy(dtype=float),
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        if not close_enough.all():
            position = int(np.flatnonzero(~close_enough)[0])
            date, symbol = raw.index[position]
            errors.append(
                "daily_common_scale_mismatch:"
                f"{symbol}:{pd.Timestamp(date).date()}:{field}"
            )

    for symbol, raw_symbol in raw_numeric.groupby(level="symbol", sort=False):
        raw_symbol = raw_symbol.sort_index(level="date")
        adjusted_symbol = adjusted_numeric.loc[raw_symbol.index]
        scale_symbol = observed_scale.loc[raw_symbol.index]
        dates = pd.DatetimeIndex(raw_symbol.index.get_level_values("date")).normalize()
        symbol_events = events.loc[events["code"].eq(str(symbol))].sort_values(
            "dividOperateDate", kind="stable"
        )

        if symbol_events.empty:
            expected_scale = pd.Series(
                float(scale_symbol.iloc[0]), index=raw_symbol.index, dtype=float
            )
            mismatch_label = "catalogued_empty_event_nonconstant_anchor"
        else:
            future_event_rows_ignored += int(
                symbol_events["dividOperateDate"].gt(dates.max()).sum()
            )
            matched = pd.merge_asof(
                pd.DataFrame({"date": dates}),
                symbol_events[["dividOperateDate", "backAdjustFactor"]],
                left_on="date",
                right_on="dividOperateDate",
                direction="backward",
                allow_exact_matches=True,
            )
            if matched["backAdjustFactor"].isna().any():
                first_missing = int(
                    np.flatnonzero(matched["backAdjustFactor"].isna().to_numpy())[0]
                )
                errors.append(
                    "no_back_adjust_factor_on_or_before_price:"
                    f"{symbol}:{dates[first_missing].date()}"
                )
                continue
            matched_factor = matched["backAdjustFactor"].to_numpy(dtype=float)
            if len(matched_factor) > 1:
                factor_changes += int(
                    (~np.isclose(
                        matched_factor[1:],
                        matched_factor[:-1],
                        rtol=0.0,
                        atol=5e-12,
                    )).sum()
                )
            anchor = float(scale_symbol.iloc[0]) / float(matched_factor[0])
            expected_scale = pd.Series(
                anchor * matched_factor, index=raw_symbol.index, dtype=float
            )
            mismatch_label = "vendor_event_path_mismatch"

        for field in fields:
            expected = raw_symbol[field].mul(expected_scale)
            observed = adjusted_symbol[field]
            relative_error = expected.sub(observed).abs().div(
                observed.abs().clip(lower=1e-12)
            )
            maximum_event_path_relative_error = max(
                maximum_event_path_relative_error, float(relative_error.max())
            )
            close_enough = np.isclose(
                expected.to_numpy(dtype=float),
                observed.to_numpy(dtype=float),
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            if not close_enough.all():
                position = int(np.flatnonzero(~close_enough)[0])
                errors.append(
                    f"{mismatch_label}:{symbol}:{dates[position].date()}:{field}"
                )
                break

    return finish()


def reconstruct_backward_adjusted_view(
    raw: pd.DataFrame,
    adjustment_events: pd.DataFrame,
    *,
    require_initial_factor: bool = True,
) -> pd.DataFrame:
    """Rebuild an ``adjustflag=1`` view from raw bars and factor events.

    ``adjustment_events`` is the documented output of
    ``query_adjust_factor``: ``code``, ``dividOperateDate`` and
    ``backAdjustFactor``.  Each raw row receives only the last factor whose
    operate date is no later than the row date.  Future event rows may be in the
    table, but ``merge_asof(..., direction='backward')`` makes them
    inaccessible to earlier observations.

    For a strict reconstruction, query the event table from before the oldest
    IPO in the security union and keep ``require_initial_factor=True``.  Falling
    back to one is safe only when the caller has independently established that
    the first raw row is the listing anchor.
    """

    _require_panel_index(raw, label="raw price view")
    required_raw = {*PRICE_COLUMNS, "preclose", "adjustflag"}
    missing_raw = sorted(required_raw.difference(raw.columns))
    if missing_raw:
        raise ValueError("raw price view missing: " + ",".join(missing_raw))
    raw_flags = raw["adjustflag"].astype("string").str.strip().dropna().unique()
    if len(raw_flags) != 1 or raw_flags[0] != RAW_ADJUSTFLAG:
        raise ValueError("raw reconstruction input must contain only adjustflag=3")

    symbol_column = next(
        (name for name in ("symbol", "code") if name in adjustment_events.columns),
        None,
    )
    date_column = next(
        (
            name
            for name in ("dividOperateDate", "operate_date", "date")
            if name in adjustment_events.columns
        ),
        None,
    )
    factor_column = next(
        (
            name
            for name in ("backAdjustFactor", "back_adjust_factor")
            if name in adjustment_events.columns
        ),
        None,
    )
    if symbol_column is None or date_column is None or factor_column is None:
        raise ValueError(
            "adjustment_events require code/symbol, dividOperateDate and backAdjustFactor"
        )
    events = adjustment_events[[symbol_column, date_column, factor_column]].copy()
    events.columns = ["symbol", "event_date", "back_adjust_factor"]
    events["symbol"] = events["symbol"].astype("string").str.strip()
    events["event_date"] = pd.to_datetime(
        events["event_date"], errors="coerce"
    ).dt.normalize()
    events["back_adjust_factor"] = pd.to_numeric(
        events["back_adjust_factor"], errors="coerce"
    )
    if (
        events["symbol"].isna().any()
        or events["symbol"].eq("").any()
        or events["event_date"].isna().any()
        or events["back_adjust_factor"].isna().any()
        or not np.isfinite(
            events["back_adjust_factor"].to_numpy(dtype=float)
        ).all()
        or events["back_adjust_factor"].le(0).any()
    ):
        raise ValueError("adjustment_events contain invalid symbol/date/factor")
    if events.duplicated(["symbol", "event_date"]).any():
        raise ValueError("adjustment_events contain duplicate symbol-event dates")

    raw_rows = raw.reset_index().copy()
    raw_rows["symbol"] = raw_rows["symbol"].astype("string")
    factors = pd.Series(np.nan, index=raw.index, dtype=float, name="adj_factor")
    missing_initial: list[str] = []
    for symbol, rows in raw_rows.groupby("symbol", sort=False):
        ordered = rows.sort_values("date", kind="stable")
        symbol_events = events.loc[events["symbol"].eq(symbol)].sort_values(
            "event_date", kind="stable"
        )
        if symbol_events.empty:
            missing_initial.append(str(symbol))
            if require_initial_factor:
                continue
            values = np.ones(len(ordered), dtype=float)
        else:
            matched = pd.merge_asof(
                ordered[["date"]],
                symbol_events[["event_date", "back_adjust_factor"]],
                left_on="date",
                right_on="event_date",
                direction="backward",
                allow_exact_matches=True,
            )
            if matched["back_adjust_factor"].isna().any():
                missing_initial.append(str(symbol))
                if require_initial_factor:
                    continue
                values = matched["back_adjust_factor"].fillna(1.0).to_numpy(dtype=float)
            else:
                values = matched["back_adjust_factor"].to_numpy(dtype=float)
        row_index = pd.MultiIndex.from_frame(
            ordered[["date", "symbol"]], names=["date", "symbol"]
        )
        factors.loc[row_index] = values
    if require_initial_factor and missing_initial:
        raise ValueError(
            "no adjustment factor on/before first raw date for: "
            + ",".join(sorted(set(missing_initial))[:10])
        )
    if (
        factors.isna().any()
        or not np.isfinite(factors.to_numpy(dtype=float)).all()
        or factors.le(0).any()
    ):
        raise ValueError("reconstructed backward-adjustment factor is incomplete")

    output = raw.copy()
    for field in (*PRICE_COLUMNS, "preclose"):
        numeric = pd.to_numeric(output[field], errors="coerce")
        if (
            numeric.isna().any()
            or not np.isfinite(numeric.to_numpy(dtype=float)).all()
            or numeric.le(0).any()
        ):
            raise ValueError(f"raw {field} must be finite positive numeric")
        adjusted_values = numeric.mul(factors)
        if not np.isfinite(adjusted_values.to_numpy(dtype=float)).all():
            raise ValueError(f"reconstructed {field} is non-finite")
        output[field] = adjusted_values
    output["adjustflag"] = BACKWARD_ADJUSTFLAG
    # Do not trust an event table merely because its rows are dated.  Every
    # post-anchor factor transition must also agree with the independent raw
    # close/preclose recurrence.
    audit_backward_adjustment_causality(raw, output).require_valid()
    return output


def derive_vwap(raw: pd.DataFrame) -> pd.Series:
    """Derive daily transaction-average price from raw amount and volume.

    A suspended/no-volume row receives raw close as a non-tradable valuation
    placeholder.  Callers must still enforce ``tradable=False`` for that row.
    """

    _require_panel_index(raw, label="raw price view")
    required = {"amount", "volume", "close"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("cannot derive vwap; missing: " + ",".join(missing))
    clean = normalize_baostock_daily_fields(raw) if "tradestatus" in raw.columns else raw
    amount = pd.to_numeric(clean["amount"], errors="coerce")
    volume = pd.to_numeric(clean["volume"], errors="coerce")
    close = pd.to_numeric(clean["close"], errors="coerce")
    if amount.isna().any() or volume.isna().any() or close.isna().any():
        raise ValueError("amount, volume and close must be numeric")
    if (amount < 0).any() or (volume < 0).any() or (close <= 0).any():
        raise ValueError("amount/volume must be non-negative and close positive")
    inconsistent = (volume > 0) & (amount <= 0)
    if inconsistent.any():
        raise ValueError("positive volume with nonpositive amount")
    values = amount.div(volume.where(volume > 0)).where(volume > 0, close)
    return values.rename("vwap")


def derive_adv20_dollars(
    raw: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute trailing mean daily raw turnover without future observations."""

    _require_panel_index(raw, label="raw price view")
    if "amount" not in raw.columns:
        raise ValueError("cannot derive ADV; missing amount")
    if window <= 0:
        raise ValueError("ADV window must be positive")
    required = window if min_periods is None else int(min_periods)
    if required <= 0 or required > window:
        raise ValueError("ADV min_periods must be between 1 and window")
    clean = normalize_baostock_daily_fields(raw) if "tradestatus" in raw.columns else raw
    amount = pd.to_numeric(clean["amount"], errors="coerce")
    if amount.isna().any() or (amount < 0).any():
        raise ValueError("amount must be non-negative numeric")
    result = amount.groupby(level="symbol", sort=False).transform(
        lambda values: values.rolling(window, min_periods=required).mean()
    )
    return result.rename(f"adv{window}_dollars" if window != 20 else "adv20_dollars")


def derive_listed_days(panel_index: pd.MultiIndex, stock_basic: pd.DataFrame) -> pd.Series:
    """Calculate calendar days since IPO using only the stable IPO date."""

    fixture = pd.DataFrame(index=panel_index)
    _require_panel_index(fixture, label="panel index")
    symbol_column = next((name for name in ("symbol", "code") if name in stock_basic.columns), None)
    ipo_column = next((name for name in ("ipo_date", "ipoDate") if name in stock_basic.columns), None)
    if symbol_column is None or ipo_column is None:
        raise ValueError("stock_basic requires code/symbol and ipoDate/ipo_date")
    basic = stock_basic[[symbol_column, ipo_column]].copy()
    basic[symbol_column] = basic[symbol_column].astype("string").str.strip()
    if basic[symbol_column].duplicated().any():
        raise ValueError("stock_basic contains duplicate symbols")
    basic[ipo_column] = pd.to_datetime(basic[ipo_column], errors="coerce").dt.normalize()
    ipo_map = basic.set_index(symbol_column)[ipo_column]
    symbols = pd.Index(panel_index.get_level_values("symbol").astype("string"))
    ipo_values = pd.Series(symbols.map(ipo_map), index=panel_index)
    if ipo_values.isna().any():
        missing = sorted(set(symbols[ipo_values.isna().to_numpy()].astype(str)))
        raise ValueError("missing IPO dates for: " + ",".join(missing[:10]))
    dates = pd.Series(panel_index.get_level_values("date"), index=panel_index)
    listed = (dates - ipo_values).dt.days
    if (listed < 0).any():
        raise ValueError("panel contains observations before IPO date")
    return listed.astype("int64").rename("listed_days")


def derive_implied_float_mcap(
    raw: pd.DataFrame,
    *,
    turnover_column: str = "turn",
    forward_fill_shares: bool = True,
) -> pd.DataFrame:
    """Infer float shares and float market cap from BaoStock turnover percent.

    ``turn`` is percentage turnover, so ``float_shares = volume * 100 / turn``.
    Missing suspended-day shares may be carried forward, never backward.
    """

    _require_panel_index(raw, label="raw price view")
    required = {"volume", "close", turnover_column}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("cannot infer float market cap; missing: " + ",".join(missing))
    clean = normalize_baostock_daily_fields(raw) if "tradestatus" in raw.columns else raw
    volume = pd.to_numeric(clean["volume"], errors="coerce")
    close = pd.to_numeric(clean["close"], errors="coerce")
    turn = pd.to_numeric(clean[turnover_column].replace("", np.nan), errors="coerce")
    if volume.isna().any() or close.isna().any() or (volume < 0).any() or (close <= 0).any():
        raise ValueError("volume must be non-negative and close positive numeric")
    observed = (volume > 0) & (turn > 0)
    shares = volume.mul(100.0).div(turn).where(observed)
    source = pd.Series("unavailable", index=raw.index, dtype="string")
    source.loc[observed] = "turnover_implied"
    if forward_fill_shares:
        carried = shares.groupby(level="symbol", sort=False).ffill()
        carry_mask = shares.isna() & carried.notna()
        shares = carried
        source.loc[carry_mask] = "prior_turnover_carry_forward"
    mcap = shares.mul(close)
    log_mcap = np.log(mcap.where(mcap > 0))
    return pd.DataFrame(
        {
            "implied_float_shares": shares,
            "implied_float_mcap": mcap,
            "log_mcap": log_mcap,
            "implied_float_mcap_source": source,
        },
        index=raw.index,
    )


def derive_rolling_beta(
    backward_adjusted_close: pd.Series,
    benchmark_returns: pd.Series,
    *,
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    """Estimate trailing single-index beta with no centering lookahead."""

    if window <= 1 or min_periods <= 1 or min_periods > window:
        raise ValueError("beta window must exceed one and include valid min_periods")
    close_frame = backward_adjusted_close.to_frame("close")
    _require_panel_index(close_frame, label="backward-adjusted close")
    close = pd.to_numeric(backward_adjusted_close, errors="coerce")
    if close.isna().any() or (close <= 0).any():
        raise ValueError("backward-adjusted close must be positive numeric")
    benchmark = pd.to_numeric(benchmark_returns, errors="coerce").copy()
    benchmark.index = _coerce_dates(benchmark.index, name="date")
    if benchmark.index.has_duplicates:
        raise ValueError("benchmark_returns index must be unique")
    pieces: list[pd.Series] = []
    for symbol, prices in close.groupby(level="symbol", sort=False):
        dated = prices.droplevel("symbol")
        asset_return = dated.pct_change(fill_method=None)
        market_return = benchmark.reindex(dated.index)
        covariance = asset_return.rolling(window, min_periods=min_periods).cov(market_return)
        variance = market_return.rolling(window, min_periods=min_periods).var()
        beta = covariance.div(variance.where(variance > 0))
        index = pd.MultiIndex.from_arrays(
            [dated.index, np.repeat(str(symbol), len(dated))], names=["date", "symbol"]
        )
        pieces.append(pd.Series(beta.to_numpy(dtype=float), index=index))
    if not pieces:
        return pd.Series(index=close.index, dtype=float, name="beta")
    return pd.concat(pieces).sort_index().reindex(close.index).rename("beta")


def derive_hs300_daily_features(
    views: HS300PriceViews,
    stock_basic: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    adjustment_events: pd.DataFrame | None = None,
    event_catalog_symbols: Iterable[object] | None = None,
    adv_window: int = 20,
    adv_min_periods: int | None = None,
    beta_window: int = 252,
    beta_min_periods: int = 60,
) -> pd.DataFrame:
    """Build deterministic daily fields while retaining raw execution OHLC."""

    raw = normalize_baostock_daily_fields(views.raw_execution)
    adjusted = views.backward_adjusted_research
    _require_panel_index(raw, label="raw price view")
    _require_panel_index(adjusted, label="backward-adjusted price view")
    required = {"tradestatus", "isST", "turn"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("raw BaoStock view missing: " + ",".join(missing))
    if adjustment_events is None:
        if event_catalog_symbols is not None:
            raise ValueError(
                "event_catalog_symbols requires an explicit adjustment_events table"
            )
        audit_backward_adjustment_causality(raw, adjusted).require_valid()
    else:
        audit_vendor_adjustment_events_asof(
            raw,
            adjusted,
            adjustment_events,
            event_catalog_symbols=event_catalog_symbols,
        ).require_valid()
    raw["vwap"] = derive_vwap(raw)
    raw["vwap_is_no_trade_placeholder"] = pd.to_numeric(raw["volume"], errors="coerce").eq(0)
    raw["adv20_dollars"] = derive_adv20_dollars(
        raw, window=adv_window, min_periods=adv_min_periods
    )
    raw_close = pd.to_numeric(raw["close"], errors="coerce")
    adjusted_close = pd.to_numeric(adjusted["close"], errors="coerce")
    raw["adj_factor"] = adjusted_close.div(raw_close)
    if raw["adj_factor"].isna().any() or (raw["adj_factor"] <= 0).any():
        raise ValueError("backward-adjustment factor must be positive")
    # Materialize prefixed research prices from one common daily factor.  This
    # makes the contract identity adj_price == raw_price * adj_factor exact and
    # prevents vendor OHLC rounding differences from looking like returns.
    for field in PRICE_COLUMNS:
        raw[f"adj_{field}"] = pd.to_numeric(raw[field], errors="coerce").mul(raw["adj_factor"])
    raw["adj_vwap"] = raw["vwap"].mul(raw["adj_factor"])
    status = pd.to_numeric(raw["tradestatus"], errors="coerce")
    volume = pd.to_numeric(raw["volume"], errors="coerce")
    raw["tradable"] = status.eq(1) & volume.gt(0)
    raw["is_st"] = pd.to_numeric(raw["isST"], errors="coerce").eq(1)
    raw["listed_days"] = derive_listed_days(raw.index, stock_basic)
    raw = raw.join(derive_implied_float_mcap(raw))
    raw["beta"] = derive_rolling_beta(
        adjusted_close,
        benchmark_returns,
        window=beta_window,
        min_periods=beta_min_periods,
    )
    return raw


def attach_proxy_benchmark_weights(
    panel: pd.DataFrame,
    *,
    mode: AuditMode | str = AuditMode.RESEARCH_PROXY,
    member_column: str = "universe_member",
    mcap_column: str = "implied_float_mcap",
) -> pd.DataFrame:
    """Attach watermarked proxy weights; strict mode always fails closed."""

    _require_panel_index(panel, label="HS300 panel")
    selected = AuditMode.coerce(mode)
    audit = audit_hs300_fields(selected)
    if selected is AuditMode.STRICT:
        # Keep this explicit even if other proxy fields later become official.
        raise DataQualityError(
            "official_historical_benchmark_weight_unavailable; "
            "strict HS300 benchmark construction requires a licensed source"
        )
    audit.require_accepted()
    missing = sorted({member_column, mcap_column}.difference(panel.columns))
    if missing:
        raise ValueError("cannot construct proxy benchmark weights; missing: " + ",".join(missing))

    output = panel.copy()
    member = output[member_column].fillna(False).astype(bool)
    mcap = pd.to_numeric(output[mcap_column], errors="coerce")
    weights = pd.Series(0.0, index=output.index, name="benchmark_weight")
    source = pd.Series("not_member", index=output.index, dtype="string")
    dates = pd.Index(output.index.get_level_values("date").unique()).sort_values()
    date_values = output.index.get_level_values("date")
    for date in dates:
        date_index = output.index[date_values == date]
        date_member = member.reindex(date_index)
        member_index = date_index[date_member.to_numpy()]
        if len(member_index) == 0:
            raise ValueError(f"no HS300 members on {pd.Timestamp(date).date()}")
        date_mcap = mcap.reindex(member_index)
        if date_mcap.notna().all() and date_mcap.gt(0).all() and float(date_mcap.sum()) > 0:
            weights.loc[member_index] = date_mcap / float(date_mcap.sum())
            source.loc[member_index] = "research_proxy:implied_float_mcap"
        else:
            weights.loc[member_index] = 1.0 / len(member_index)
            source.loc[member_index] = "research_proxy:equal_weight_fallback"
    output["benchmark_weight"] = weights
    output["benchmark_weight_quality"] = "research_proxy"
    output["benchmark_weight_source"] = source
    output["research_watermark"] = RESEARCH_PROXY_WATERMARK
    output.attrs = {
        **panel.attrs,
        "dataset_tier": AuditMode.RESEARCH_PROXY.value,
        "benchmark_weight_quality": "research_proxy",
        "benchmark_weight_source": "implied_float_mcap_or_equal_weight_fallback",
        "research_watermark": RESEARCH_PROXY_WATERMARK,
        "data_quality_audit": audit.to_dict(),
        "field_quality": hs300_field_quality_map(),
    }
    return output


__all__ = [
    "BACKWARD_ADJUSTFLAG",
    "BackwardAdjustmentAudit",
    "HS300PriceViews",
    "MembershipSnapshotAudit",
    "PriceHistoryBoundary",
    "RAW_ADJUSTFLAG",
    "VendorAdjustmentEventAudit",
    "attach_membership",
    "attach_proxy_benchmark_weights",
    "audit_backward_adjustment_causality",
    "audit_vendor_adjustment_events_asof",
    "derive_adv20_dollars",
    "derive_hs300_daily_features",
    "derive_implied_float_mcap",
    "derive_listed_days",
    "derive_rolling_beta",
    "derive_vwap",
    "expand_membership_snapshots",
    "normalize_baostock_daily_fields",
    "reconstruct_backward_adjusted_view",
    "separate_price_views",
    "trim_leading_unadjusted_vendor_history",
    "validate_membership_snapshots",
]
