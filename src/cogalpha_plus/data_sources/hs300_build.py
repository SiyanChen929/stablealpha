"""Build a frozen, watermarked HS300 research panel from BaoStock caches.

This module never contacts BaoStock.  It consumes the normalized catalog and
content-addressed raw request cache, verifies every referenced object, derives
rolling fields on the complete downloaded history, and only then clips the
requested research period.  The resulting benchmark weights and price-limit
flags are explicit research proxies rather than official CSI index data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data_contract import DataContract, DataValidationReport, validate_daily_panel
from ..data_quality import RESEARCH_PROXY_WATERMARK, hs300_field_quality_map
from .baostock import (
    ADJUST_FACTOR_FIELDS,
    CacheIntegrityError,
    EQUITY_DAILY_FIELDS,
    HS300_FIELDS,
    INDUSTRY_FIELDS,
    INDEX_DAILY_FIELDS,
    STOCK_BASIC_FIELDS,
    TRADE_DATE_FIELDS,
    make_request_key,
)
from .hs300 import (
    attach_proxy_benchmark_weights,
    audit_backward_adjustment_causality,
    audit_vendor_adjustment_events_asof,
    derive_hs300_daily_features,
    separate_price_views,
    trim_leading_unadjusted_vendor_history,
    validate_membership_snapshots,
)
from .hs300_download import (
    _deduplicate_vendor_snapshots,
    _industry_request_dates,
    _materialize_industry_catalog,
    _materialize_security_master,
    _membership_observation_snapshots,
    _membership_request_dates,
    _validate_membership_snapshot,
    expand_membership_daily,
)


# Keep the continuous tape narrow: these are the complete production-DSL,
# label, eligibility and execution fields.  Wide vendor/proxy columns remain
# on the member panel, where they are needed for portfolio controls and audit.
_CONTINUOUS_FEATURE_TAPE_COLUMNS = (
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
    "preclose",
    "volume",
    "amount",
    "adj_factor",
    "tradable",
    "is_st",
    "listed_days",
    "log_mcap",
    "beta",
    "adv20_dollars",
    "vendor_universe_member",
    "universe_member",
    "research_eligible",
)

_CONTINUOUS_LIMIT_COLUMNS = (
    "limit_up",
    "limit_down",
    "limit_proxy_known",
    "limit_ratio_proxy",
    "research_close_limit_up_proxy",
    "research_close_limit_down_proxy",
    "research_close_limit_known_proxy",
)

_RAW_BUILD_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "turn",
    "tradestatus",
    "isST",
)

_ADJUSTED_BUILD_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "adjustflag",
)


@dataclass(frozen=True)
class HS300BuildConfig:
    cache_dir: Path | str
    normalized_dir: Path | str
    output_dir: Path | str
    research_start: str
    research_end: str | None = None
    expected_members: int = 300
    adv_window: int = 20
    adv_min_periods: int | None = None
    beta_window: int = 252
    beta_min_periods: int = 60
    contract_minimum_history_days: int = 252
    minimum_limit_rule_listed_days: int = 30
    maximum_industry_age_days: int = 35
    membership_source_start_date: str = "2010-01-01"
    membership_source_end_date: str = "2025-12-31"
    membership_frequency: str = "provider_weekly"
    maximum_membership_snapshot_gap_days: int = 21
    maximum_membership_snapshot_age_days: int = 21
    adjustment_event_start_date: str = "1990-01-01"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "normalized_dir", Path(self.normalized_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        start = pd.Timestamp(self.research_start).normalize()
        end = pd.Timestamp(self.research_end).normalize() if self.research_end else None
        if end is not None and start > end:
            raise ValueError("research_start is after research_end")
        if self.expected_members <= 0:
            raise ValueError("expected_members must be positive")
        if self.adv_window <= 0 or self.beta_window <= 1:
            raise ValueError("rolling windows must be positive")
        if self.beta_min_periods <= 1 or self.beta_min_periods > self.beta_window:
            raise ValueError("invalid beta_min_periods")
        if self.contract_minimum_history_days <= 0:
            raise ValueError("contract_minimum_history_days must be positive")
        if self.minimum_limit_rule_listed_days < 0 or self.maximum_industry_age_days <= 0:
            raise ValueError("limit listing age and industry staleness must be valid")
        source_start = pd.Timestamp(self.membership_source_start_date).normalize()
        source_end = pd.Timestamp(self.membership_source_end_date).normalize()
        adjustment_start = pd.Timestamp(self.adjustment_event_start_date).normalize()
        if source_start > source_end or adjustment_start > source_end:
            raise ValueError("membership/adjustment source date policy is invalid")
        if self.membership_frequency not in {"provider_weekly", "trading_day"}:
            raise ValueError("membership frequency policy is invalid")
        if (
            self.maximum_membership_snapshot_gap_days <= 0
            or self.maximum_membership_snapshot_age_days < 0
        ):
            raise ValueError("membership snapshot gap/age policy is invalid")


@dataclass(frozen=True)
class HS300BuildResult:
    panel: pd.DataFrame
    validation: DataValidationReport
    manifest_path: Path
    manifest_sha256_path: Path
    partition_paths: tuple[Path, ...]
    manifest: Mapping[str, object]
    feature_tape: pd.DataFrame | None = None
    feature_tape_partition_paths: tuple[Path, ...] = ()
    terminal_stale_membership_exceptions: pd.DataFrame | None = None
    terminal_stale_membership_exception_path: Path | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=True)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(text: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Mapping[str, object], target: Path) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n"
    _atomic_text(encoded, target)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheIntegrityError(f"JSON object expected at {path}")
    return value


def _safe_cache_object(cache_dir: Path, relative: object) -> Path:
    text = str(relative or "")
    candidate = Path(text)
    if not text or candidate.is_absolute():
        raise CacheIntegrityError("cache state has unsafe object path")
    root = cache_dir.resolve()
    resolved = (cache_dir / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise CacheIntegrityError("cache state object escapes cache directory")
    return resolved


def _read_committed_cache_object(
    cache_dir: Path,
    request_key: str,
    *,
    expected_endpoint: str,
    expected_fields: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    if not re.fullmatch(r"[0-9a-f]{64}", str(request_key)):
        raise CacheIntegrityError(f"invalid request key: {request_key!r}")
    state_path = cache_dir / "state" / f"{request_key}.json"
    if not state_path.is_file():
        raise CacheIntegrityError(f"missing cache state for {request_key}")
    state = _read_json(state_path)
    if state.get("state_schema_version") != 1:
        raise CacheIntegrityError(f"unsupported cache state schema for {request_key}")
    if state.get("request_key") != request_key:
        raise CacheIntegrityError(f"cache state key mismatch for {request_key}")
    if state.get("status") != "success":
        raise CacheIntegrityError(f"cache request is not committed-success: {request_key}")
    if state.get("endpoint") != expected_endpoint:
        raise CacheIntegrityError(
            f"cache endpoint mismatch for {request_key}: {state.get('endpoint')}"
        )
    expected = tuple(str(field) for field in expected_fields)
    if tuple(str(field) for field in state.get("fields", ())) != expected:
        raise CacheIntegrityError(f"cache state schema mismatch for {request_key}")
    args = state.get("args")
    provider_version = str(state.get("provider_version", "")).strip()
    if not isinstance(args, Mapping) or not provider_version:
        raise CacheIntegrityError(
            f"cache state lacks canonical args/provider version for {request_key}"
        )
    if make_request_key(
        expected_endpoint,
        args,
        provider_version=provider_version,
    ) != request_key:
        raise CacheIntegrityError(
            f"cache request key is not canonical for frozen args: {request_key}"
        )
    object_path = _safe_cache_object(cache_dir, state.get("object"))
    if not object_path.is_file():
        raise CacheIntegrityError(f"cache object missing for {request_key}")
    actual_hash = _sha256_file(object_path)
    if actual_hash != state.get("parquet_sha256"):
        raise CacheIntegrityError(f"cache object hash mismatch for {request_key}")
    try:
        frame = pd.read_parquet(object_path)
    except Exception as exc:
        raise CacheIntegrityError(f"cannot read cache object {request_key}: {exc}") from exc
    if tuple(frame.columns.astype(str)) != expected:
        raise CacheIntegrityError(f"cache Parquet schema mismatch for {request_key}")
    if len(frame) != int(state.get("rows", -1)):
        raise CacheIntegrityError(f"cache row count mismatch for {request_key}")
    return frame, state, object_path


def _normalised_input_paths(normalized_dir: Path) -> dict[str, Path]:
    return {
        "trade_calendar": normalized_dir / "trade_calendar.parquet",
        "membership_snapshots": normalized_dir / "membership_snapshots.parquet",
        "membership_daily": normalized_dir / "membership_daily.parquet",
        "security_union": normalized_dir / "security_union.parquet",
        "membership_request_index": normalized_dir / "membership_request_index.parquet",
        "security_master": normalized_dir / "security_master.parquet",
        "security_master_request_index": (
            normalized_dir / "security_master_request_index.parquet"
        ),
        "industry_snapshots": normalized_dir / "industry_snapshots.parquet",
        "industry_request_index": normalized_dir / "industry_request_index.parquet",
        "price_request_index": normalized_dir / "price_request_index.parquet",
        "adjustment_request_index": normalized_dir / "adjustment_request_index.parquet",
    }


def _declared_file_records(payload: object) -> Iterable[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        if "path" in payload and "sha256" in payload:
            yield payload
        for value in payload.values():
            yield from _declared_file_records(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _declared_file_records(value)


def _verify_normalized_manifest_hashes(normalized_dir: Path, paths: Iterable[Path]) -> None:
    targets = {path.resolve(): path for path in paths}
    by_name = {path.name: path.resolve() for path in paths}
    for manifest_path in (
        normalized_dir / "membership_manifest.json",
        normalized_dir / "price_manifest.json",
        normalized_dir / "adjustment_manifest.json",
        normalized_dir / "industry_manifest.json",
    ):
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        for record in _declared_file_records(manifest):
            raw_path = Path(str(record["path"]))
            resolved = raw_path.resolve() if raw_path.is_absolute() else (normalized_dir / raw_path).resolve()
            target = targets.get(resolved)
            if target is None and raw_path.name in by_name:
                target = Path(by_name[raw_path.name])
            if target is None:
                continue
            expected = str(record["sha256"])
            actual = _sha256_file(Path(target))
            if actual != expected:
                raise CacheIntegrityError(f"normalized manifest hash mismatch: {Path(target).name}")


def _read_normalized_inputs(config: HS300BuildConfig) -> dict[str, pd.DataFrame]:
    paths = _normalised_input_paths(config.normalized_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing normalized HS300 inputs: " + ",".join(missing))
    _verify_normalized_manifest_hashes(config.normalized_dir, paths.values())
    return {name: pd.read_parquet(path) for name, path in paths.items()}


def _blank_string(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _canonical_trade_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    required = [*TRADE_DATE_FIELDS, "date"]
    if tuple(frame.columns.astype(str)) != tuple(required):
        raise ValueError("trade_calendar must have the exact frozen schema")
    output = frame.loc[:, required].copy()
    output["calendar_date"] = output["calendar_date"].astype(str).str.strip()
    output["is_trading_day"] = output["is_trading_day"].astype(str).str.strip()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    parsed = pd.to_datetime(output["calendar_date"], errors="coerce").dt.normalize()
    if (
        output["date"].isna().any()
        or parsed.isna().any()
        or not output["date"].equals(parsed)
        or output["date"].duplicated().any()
        or not output["is_trading_day"].isin(["0", "1"]).all()
    ):
        raise ValueError("trade_calendar values are invalid")
    return output.sort_values("date").reset_index(drop=True)


def _canonical_membership_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["requested_date", "vendor_update_date", "code", "code_name"]
    if tuple(frame.columns.astype(str)) != tuple(required):
        raise ValueError("membership_snapshots must have the exact frozen schema")
    output = frame.loc[:, required].copy()
    for name in ("requested_date", "vendor_update_date"):
        output[name] = pd.to_datetime(output[name], errors="coerce").dt.normalize()
    output["code"] = output["code"].astype(str).str.strip()
    output["code_name"] = output["code_name"].astype(str)
    if output[["requested_date", "vendor_update_date"]].isna().any().any():
        raise ValueError("membership_snapshots contains invalid dates")
    return output.sort_values(
        ["vendor_update_date", "code"], kind="stable"
    ).reset_index(drop=True)


def _canonical_membership_daily(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "code", "code_name", "vendor_update_date"]
    if tuple(frame.columns.astype(str)) != tuple(required):
        raise ValueError("membership_daily must have the exact frozen schema")
    output = frame.loc[:, required].copy()
    for name in ("date", "vendor_update_date"):
        output[name] = pd.to_datetime(output[name], errors="coerce").dt.normalize()
    output["code"] = output["code"].astype(str).str.strip()
    output["code_name"] = output["code_name"].astype(str)
    if output[["date", "vendor_update_date"]].isna().any().any():
        raise ValueError("membership_daily contains invalid dates")
    return output.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def _canonical_security_union(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "code",
        "code_name",
        "first_member_date",
        "last_member_date",
        "member_sessions",
    ]
    if tuple(frame.columns.astype(str)) != tuple(required):
        raise ValueError("security_union must have the exact frozen schema")
    output = frame.loc[:, required].copy()
    output["code"] = output["code"].astype(str).str.strip()
    output["code_name"] = output["code_name"].astype(str)
    for name in ("first_member_date", "last_member_date"):
        output[name] = pd.to_datetime(output[name], errors="coerce").dt.normalize()
    output["member_sessions"] = pd.to_numeric(
        output["member_sessions"], errors="coerce"
    ).astype("Int64")
    if output.isna().any().any() or output["code"].duplicated().any():
        raise ValueError("security_union contains invalid values")
    return output.sort_values("code").reset_index(drop=True)


def _require_exact_frame(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
) -> None:
    if not actual.equals(expected):
        raise CacheIntegrityError(
            f"{label} differs from deterministic raw-cache reconstruction"
        )


def _load_and_rebuild_membership_lineage(
    config: HS300BuildConfig,
    inputs: Mapping[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, object]],
    Mapping[str, object],
]:
    """Replay raw calendar/snapshot requests into every normalized catalog."""

    request_index = inputs["membership_request_index"].copy()
    required_index = {
        "endpoint",
        "requested_date",
        "start_date",
        "end_date",
        "vendor_update_date",
        "rows",
        "request_key",
        "cache_hit",
    }
    missing = sorted(required_index.difference(request_index.columns))
    if missing:
        raise ValueError("membership_request_index missing: " + ",".join(missing))
    request_index["endpoint"] = request_index["endpoint"].astype(str).str.strip()
    if request_index["request_key"].astype(str).duplicated().any():
        raise ValueError("membership_request_index contains duplicate request keys")
    calendar_rows = request_index.loc[request_index["endpoint"].eq("trade_dates")]
    snapshot_rows = request_index.loc[
        request_index["endpoint"].eq("hs300_stocks")
    ].copy()
    if len(calendar_rows) != 1 or len(snapshot_rows) == 0 or len(
        request_index
    ) != 1 + len(snapshot_rows):
        raise ValueError("membership request index endpoint coverage is invalid")

    calendar_row = next(calendar_rows.itertuples(index=False))
    expected_start = str(pd.Timestamp(config.membership_source_start_date).date())
    expected_end = str(pd.Timestamp(config.membership_source_end_date).date())
    if (
        _blank_string(calendar_row.start_date) != expected_start
        or _blank_string(calendar_row.end_date) != expected_end
        or _blank_string(calendar_row.requested_date)
        or _blank_string(calendar_row.vendor_update_date)
    ):
        raise CacheIntegrityError("trade calendar request policy differs from build config")
    calendar_raw, calendar_state, calendar_object = _read_committed_cache_object(
        config.cache_dir,
        str(calendar_row.request_key),
        expected_endpoint="trade_dates",
        expected_fields=TRADE_DATE_FIELDS,
    )
    if calendar_state.get("args") != {
        "start_date": expected_start,
        "end_date": expected_end,
    } or int(calendar_row.rows) != len(calendar_raw):
        raise CacheIntegrityError("trade calendar request index/state mismatch")
    rebuilt_calendar = calendar_raw.copy()
    rebuilt_calendar["date"] = pd.to_datetime(
        rebuilt_calendar["calendar_date"], errors="coerce"
    ).dt.normalize()
    rebuilt_calendar = _canonical_trade_calendar(rebuilt_calendar)
    frozen_calendar = _canonical_trade_calendar(inputs["trade_calendar"])
    _require_exact_frame(
        frozen_calendar, rebuilt_calendar, label="trade_calendar"
    )
    expected_days = pd.date_range(expected_start, expected_end, freq="D")
    if not pd.DatetimeIndex(rebuilt_calendar["date"]).equals(expected_days):
        raise CacheIntegrityError("trade calendar does not cover the frozen date policy")
    trading_dates = pd.DatetimeIndex(
        rebuilt_calendar.loc[
            rebuilt_calendar["is_trading_day"].eq("1"), "date"
        ]
    )
    expected_requests = pd.DatetimeIndex(
        _membership_request_dates(
            trading_dates,
            expected_start,
            expected_end,
            config.membership_frequency,
        )
    )
    snapshot_rows["parsed_requested_date"] = pd.to_datetime(
        snapshot_rows["requested_date"], errors="coerce"
    ).dt.normalize()
    if (
        snapshot_rows["parsed_requested_date"].isna().any()
        or snapshot_rows["parsed_requested_date"].duplicated().any()
        or not pd.DatetimeIndex(
            snapshot_rows["parsed_requested_date"].sort_values()
        ).equals(expected_requests)
    ):
        raise CacheIntegrityError("membership snapshot request plan is incomplete or altered")

    validated_frames: list[pd.DataFrame] = []
    records: list[dict[str, object]] = [
        {
            "endpoint": "trade_dates",
            "requested_date": "",
            "args": dict(calendar_state["args"]),
            "request_key": str(calendar_row.request_key),
            "rows": len(calendar_raw),
            "parquet_sha256": calendar_state["parquet_sha256"],
            "object": str(calendar_object),
            "provider_version": str(calendar_state.get("provider_version", "")),
        }
    ]
    for row in snapshot_rows.sort_values("parsed_requested_date").itertuples(
        index=False
    ):
        requested = pd.Timestamp(row.parsed_requested_date).normalize()
        if (
            _blank_string(row.start_date)
            or _blank_string(row.end_date)
        ):
            raise CacheIntegrityError("snapshot request index has unexpected range args")
        raw, state, object_path = _read_committed_cache_object(
            config.cache_dir,
            str(row.request_key),
            expected_endpoint="hs300_stocks",
            expected_fields=HS300_FIELDS,
        )
        if state.get("args") != {"date": str(requested.date())} or int(
            row.rows
        ) != len(raw):
            raise CacheIntegrityError("membership snapshot request index/state mismatch")
        validated = _validate_membership_snapshot(
            raw,
            requested,
            expected_members=config.expected_members,
        )
        vendor_update = str(
            pd.Timestamp(validated["vendor_update_date"].iloc[0]).date()
        )
        if _blank_string(row.vendor_update_date) != vendor_update:
            raise CacheIntegrityError("membership request index updateDate mismatch")
        validated_frames.append(validated)
        records.append(
            {
                "endpoint": "hs300_stocks",
                "requested_date": str(requested.date()),
                "vendor_update_date": vendor_update,
                "args": dict(state["args"]),
                "request_key": str(row.request_key),
                "rows": len(raw),
                "parquet_sha256": state["parquet_sha256"],
                "object": str(object_path),
                "provider_version": str(state.get("provider_version", "")),
            }
        )
    rebuilt_snapshots = _canonical_membership_snapshots(
        _deduplicate_vendor_snapshots(validated_frames)
    )
    observation_snapshots = _membership_observation_snapshots(validated_frames)
    snapshot_audit = validate_membership_snapshots(
        observation_snapshots,
        expected_members=config.expected_members,
        trading_dates=trading_dates,
        max_snapshot_gap_days=config.maximum_membership_snapshot_gap_days,
        max_snapshot_age_days=config.maximum_membership_snapshot_age_days,
    ).require_valid()
    frozen_snapshots = _canonical_membership_snapshots(
        inputs["membership_snapshots"]
    )
    _require_exact_frame(
        frozen_snapshots,
        rebuilt_snapshots,
        label="membership_snapshots",
    )
    rebuilt_daily = _canonical_membership_daily(
        expand_membership_daily(
            rebuilt_snapshots,
            trading_dates,
            observation_snapshots=observation_snapshots,
            expected_members=config.expected_members,
            maximum_snapshot_gap_days=config.maximum_membership_snapshot_gap_days,
            maximum_snapshot_age_days=config.maximum_membership_snapshot_age_days,
        )
    )
    frozen_daily = _canonical_membership_daily(inputs["membership_daily"])
    _require_exact_frame(frozen_daily, rebuilt_daily, label="membership_daily")
    rebuilt_union = _canonical_security_union(
        rebuilt_daily.groupby("code", as_index=False)
        .agg(
            code_name=("code_name", "last"),
            first_member_date=("date", "min"),
            last_member_date=("date", "max"),
            member_sessions=("date", "size"),
        )
        .sort_values("code")
        .reset_index(drop=True)
    )
    frozen_union = _canonical_security_union(inputs["security_union"])
    _require_exact_frame(frozen_union, rebuilt_union, label="security_union")
    policy = {
        "schema_version": "1.0",
        "complete": True,
        "expected_members": config.expected_members,
        "source_start_date": expected_start,
        "source_end_date": expected_end,
        "membership_frequency": config.membership_frequency,
        "maximum_snapshot_gap_days": config.maximum_membership_snapshot_gap_days,
        "maximum_snapshot_age_days": config.maximum_membership_snapshot_age_days,
        "deterministic_daily_expansion": True,
        "freshness_clock": "successful_requested_observation_date",
        "membership_effective_clock": "vendor_update_date",
        "raw_cache_requests": len(records),
        "snapshot_audit": asdict(snapshot_audit),
    }
    return rebuilt_daily, rebuilt_snapshots, rebuilt_union, records, policy


def _canonical_security_master(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns.astype(str)) != STOCK_BASIC_FIELDS:
        raise ValueError("security_master must have the exact frozen schema")
    output = frame.loc[:, list(STOCK_BASIC_FIELDS)].copy()
    for name in STOCK_BASIC_FIELDS:
        output[name] = output[name].astype("string").fillna("").astype(str).str.strip()
    if output["code"].duplicated().any() or not output["code"].map(
        lambda value: bool(re.fullmatch(r"(sh|sz)\.\d{6}", value))
    ).all():
        raise ValueError("security_master contains duplicate or malformed codes")
    ipo = pd.to_datetime(output["ipoDate"], errors="coerce")
    raw_out = output["outDate"].replace({"": pd.NA, "0": pd.NA})
    out = pd.to_datetime(raw_out, errors="coerce")
    if ipo.isna().any() or (raw_out.notna() & out.isna()).any():
        raise ValueError("security_master contains invalid IPO/out dates")
    return output.sort_values("code").reset_index(drop=True)


def _canonical_industry_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    required = (
        "requested_date",
        "vendor_update_date",
        "code",
        "code_name",
        "industry",
        "industryClassification",
    )
    if tuple(frame.columns.astype(str)) != required:
        raise ValueError("industry_snapshots must have the exact frozen schema")
    output = frame.loc[:, list(required)].copy()
    for name in ("requested_date", "vendor_update_date"):
        output[name] = pd.to_datetime(output[name], errors="coerce").dt.normalize()
    for name in ("code", "code_name", "industry", "industryClassification"):
        output[name] = output[name].astype("string").fillna("").astype(str).str.strip()
    if (
        output[["requested_date", "vendor_update_date"]].isna().any().any()
        or output.duplicated(["requested_date", "code"]).any()
        or (output["vendor_update_date"] > output["requested_date"]).any()
    ):
        raise ValueError("industry_snapshots contains invalid keys or dates")
    return output.sort_values(
        ["requested_date", "code"], kind="stable"
    ).reset_index(drop=True)


def _request_index_semantics(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    sort_by: Sequence[str],
) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError("request index missing: " + ",".join(missing))
    output = frame.loc[:, list(columns)].copy()
    for name in columns:
        if name == "rows":
            output[name] = pd.to_numeric(output[name], errors="coerce").astype("Int64")
        else:
            output[name] = output[name].astype("string").fillna("").astype(str).str.strip()
    if output.isna().any().any():
        raise ValueError("request index contains invalid semantic values")
    return output.sort_values(list(sort_by), kind="stable").reset_index(drop=True)


def _cache_lineage_record(
    *,
    endpoint: str,
    state: Mapping[str, object],
    object_path: Path,
    rows: int,
    identity: Mapping[str, object],
) -> dict[str, object]:
    args = state.get("args")
    if not isinstance(args, Mapping):
        raise CacheIntegrityError("cache lineage record lacks canonical args")
    return {
        "endpoint": endpoint,
        **dict(identity),
        "args": dict(args),
        "request_key": str(state["request_key"]),
        "rows": int(rows),
        "parquet_sha256": str(state["parquet_sha256"]),
        "object": str(object_path),
        "provider_version": str(state.get("provider_version", "")),
    }


def _load_and_rebuild_security_lineage(
    config: HS300BuildConfig,
    inputs: Mapping[str, pd.DataFrame],
    union_codes: Iterable[str],
) -> tuple[pd.DataFrame, list[dict[str, object]], Mapping[str, object]]:
    """Replay the canonical global stock_basic query and exact union gaps."""

    index = inputs["security_master_request_index"].copy()
    semantic_columns = (
        "endpoint",
        "code",
        "code_name",
        "rows",
        "request_key",
    )
    frozen_semantics = _request_index_semantics(
        index, semantic_columns, sort_by=("code", "code_name")
    )
    if not frozen_semantics["endpoint"].eq("stock_basic").all():
        raise CacheIntegrityError("security request index endpoint is altered")
    global_rows = frozen_semantics.loc[
        frozen_semantics["code"].eq("") & frozen_semantics["code_name"].eq("")
    ]
    if len(global_rows) != 1:
        raise CacheIntegrityError("security request index lacks one global request")
    global_row = global_rows.iloc[0]
    global_raw, global_state, global_object = _read_committed_cache_object(
        config.cache_dir,
        str(global_row["request_key"]),
        expected_endpoint="stock_basic",
        expected_fields=STOCK_BASIC_FIELDS,
    )
    if global_state.get("args") != {"code": "", "code_name": ""} or int(
        global_row["rows"]
    ) != len(global_raw):
        raise CacheIntegrityError("global stock_basic request index/state mismatch")
    global_raw.attrs["baostock_cache"] = {
        "request_key": str(global_row["request_key"]),
        "cache_hit": True,
    }
    codes = sorted({str(code) for code in union_codes})
    covered = set(
        global_raw.loc[global_raw["code"].astype(str).isin(codes), "code"].astype(str)
    )
    missing = sorted(set(codes).difference(covered))
    supplement_rows = frozen_semantics.loc[~frozen_semantics.index.isin(global_rows.index)]
    if (
        supplement_rows["code_name"].ne("").any()
        or sorted(supplement_rows["code"].tolist()) != missing
    ):
        raise CacheIntegrityError("stock_basic supplement plan differs from global gaps")
    requests: list[tuple[str, str, pd.DataFrame]] = [("", "", global_raw)]
    records = [
        _cache_lineage_record(
            endpoint="stock_basic",
            state=global_state,
            object_path=global_object,
            rows=len(global_raw),
            identity={"code": "", "code_name": ""},
        )
    ]
    for row in supplement_rows.sort_values("code").itertuples(index=False):
        raw, state, object_path = _read_committed_cache_object(
            config.cache_dir,
            str(row.request_key),
            expected_endpoint="stock_basic",
            expected_fields=STOCK_BASIC_FIELDS,
        )
        expected_args = {"code": str(row.code), "code_name": ""}
        if state.get("args") != expected_args or int(row.rows) != len(raw):
            raise CacheIntegrityError("stock_basic supplement request/index mismatch")
        raw.attrs["baostock_cache"] = {
            "request_key": str(row.request_key),
            "cache_hit": True,
        }
        requests.append((str(row.code), "", raw))
        records.append(
            _cache_lineage_record(
                endpoint="stock_basic",
                state=state,
                object_path=object_path,
                rows=len(raw),
                identity={"code": str(row.code), "code_name": ""},
            )
        )
    rebuilt, rebuilt_index = _materialize_security_master(codes, requests)
    rebuilt = _canonical_security_master(rebuilt)
    frozen = _canonical_security_master(inputs["security_master"])
    _require_exact_frame(frozen, rebuilt, label="security_master")
    rebuilt_semantics = _request_index_semantics(
        rebuilt_index, semantic_columns, sort_by=("code", "code_name")
    )
    _require_exact_frame(
        frozen_semantics,
        rebuilt_semantics,
        label="security_master_request_index",
    )
    policy = {
        "schema_version": "1.0",
        "complete": True,
        "endpoint": "stock_basic",
        "global_query_args": {"code": "", "code_name": ""},
        "supplement_policy": "exact_union_gaps_by_code",
        "deterministic_union_filter": True,
        "historical_union_symbols": len(codes),
        "raw_cache_requests": len(records),
    }
    return rebuilt, records, policy


def _load_and_rebuild_industry_lineage(
    config: HS300BuildConfig,
    inputs: Mapping[str, pd.DataFrame],
    membership_snapshots: pd.DataFrame,
    union_codes: Iterable[str],
) -> tuple[pd.DataFrame, list[dict[str, object]], Mapping[str, object]]:
    """Replay every global PIT industry request into the normalized catalog."""

    index = inputs["industry_request_index"].copy()
    semantic_columns = (
        "endpoint",
        "requested_date",
        "vendor_update_date",
        "rows",
        "request_key",
    )
    frozen_semantics = _request_index_semantics(
        index, semantic_columns, sort_by=("requested_date",)
    )
    if not frozen_semantics["endpoint"].eq("stock_industry").all():
        raise CacheIntegrityError("industry request index endpoint is altered")
    requested = pd.to_datetime(
        frozen_semantics["requested_date"], errors="coerce"
    ).dt.normalize()
    expected = _industry_request_dates(
        membership_snapshots, inputs["membership_request_index"]
    )
    if (
        requested.isna().any()
        or requested.duplicated().any()
        or not pd.DatetimeIndex(requested).equals(expected)
    ):
        raise CacheIntegrityError("industry request plan differs from membership dates")
    requests: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    records: list[dict[str, object]] = []
    for row, day in zip(
        frozen_semantics.itertuples(index=False), requested, strict=True
    ):
        raw, state, object_path = _read_committed_cache_object(
            config.cache_dir,
            str(row.request_key),
            expected_endpoint="stock_industry",
            expected_fields=INDUSTRY_FIELDS,
        )
        expected_args = {"code": "", "date": str(pd.Timestamp(day).date())}
        if state.get("args") != expected_args or int(row.rows) != len(raw):
            raise CacheIntegrityError("industry request index/state mismatch")
        update = pd.to_datetime(raw["updateDate"], errors="coerce").dt.normalize()
        if (
            update.isna().any()
            or update.nunique() != 1
            or str(pd.Timestamp(update.iloc[0]).date()) != str(row.vendor_update_date)
        ):
            raise CacheIntegrityError("industry request index updateDate mismatch")
        raw.attrs["baostock_cache"] = {
            "request_key": str(row.request_key),
            "cache_hit": True,
        }
        requests.append((pd.Timestamp(day), raw))
        records.append(
            _cache_lineage_record(
                endpoint="stock_industry",
                state=state,
                object_path=object_path,
                rows=len(raw),
                identity={
                    "requested_date": str(pd.Timestamp(day).date()),
                    "vendor_update_date": str(pd.Timestamp(update.iloc[0]).date()),
                },
            )
        )
    rebuilt, rebuilt_index = _materialize_industry_catalog(
        membership_snapshots,
        union_codes,
        requests,
        expected_request_dates=expected,
    )
    rebuilt = _canonical_industry_snapshots(rebuilt)
    frozen = _canonical_industry_snapshots(inputs["industry_snapshots"])
    _require_exact_frame(frozen, rebuilt, label="industry_snapshots")
    rebuilt_semantics = _request_index_semantics(
        rebuilt_index, semantic_columns, sort_by=("requested_date",)
    )
    _require_exact_frame(
        frozen_semantics, rebuilt_semantics, label="industry_request_index"
    )
    policy = {
        "schema_version": "1.0",
        "complete": True,
        "endpoint": "stock_industry",
        "query_code": "",
        "query_dates": (
            "membership_requested_observation_dates_union_vendor_effective_dates"
        ),
        "freshness_clock": "successful_requested_observation_date",
        "vendor_update_policy": "nonfuture_effective_metadata",
        "deterministic_union_filter": True,
        "future_update_policy": "forbidden",
        "raw_cache_requests": len(records),
        "catalog_rows": len(rebuilt),
        "native_snapshots": int(rebuilt["vendor_update_date"].nunique()),
    }
    return rebuilt, records, policy


def _validate_membership_daily(
    membership: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    expected_members: int,
) -> pd.DataFrame:
    required = {"date", "code", "vendor_update_date"}
    missing = sorted(required.difference(membership.columns))
    if missing:
        raise ValueError("membership_daily missing: " + ",".join(missing))
    output = membership.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    output["vendor_update_date"] = pd.to_datetime(
        output["vendor_update_date"], errors="coerce"
    ).dt.normalize()
    output["code"] = output["code"].astype("string").str.strip()
    if output[["date", "vendor_update_date"]].isna().any().any():
        raise ValueError("membership_daily contains invalid dates")
    if (output["vendor_update_date"] > output["date"]).any():
        raise ValueError("membership_daily contains future vendor snapshots")
    if output.duplicated(["date", "code"]).any():
        raise ValueError("membership_daily contains duplicate date-symbol rows")
    counts = output.groupby("date")["code"].nunique()
    if not counts.eq(expected_members).all():
        raise ValueError(f"membership_daily must contain exactly {expected_members} names per date")
    if not {"calendar_date", "is_trading_day"}.issubset(calendar.columns):
        raise ValueError("trade_calendar schema is incomplete")
    trading = pd.DatetimeIndex(
        pd.to_datetime(
            calendar.loc[calendar["is_trading_day"].astype(str) == "1", "calendar_date"],
            errors="coerce",
        ).dropna()
    ).normalize()
    trading_set = set(trading)
    member_dates = set(output["date"])
    if not member_dates.issubset(trading_set):
        raise ValueError("membership_daily contains non-trading dates")
    bounded = {date for date in trading_set if output["date"].min() <= date <= output["date"].max()}
    if member_dates != bounded:
        raise ValueError("membership_daily does not cover every trading session")
    return output.sort_values(["date", "code"]).reset_index(drop=True)


def _load_price_views(
    cache_dir: Path,
    request_index: pd.DataFrame,
    expected_codes: set[str],
    membership: pd.DataFrame,
    *,
    expected_start_date: str,
    expected_end_date: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    required = {"code", "adjustflag", "rows", "request_key"}
    missing = sorted(required.difference(request_index.columns))
    if missing:
        raise ValueError("price_request_index missing: " + ",".join(missing))
    index = request_index.copy()
    index["code"] = index["code"].astype(str)
    index["adjustflag"] = index["adjustflag"].astype(str)
    if index.duplicated(["code", "adjustflag"]).any():
        raise ValueError("price_request_index contains duplicate code-adjustflag rows")
    if set(index["code"]) != expected_codes:
        raise ValueError("price_request_index does not exactly cover the historical member union")
    for code, flags in index.groupby("code")["adjustflag"]:
        if set(flags) != {"1", "3"}:
            raise ValueError(f"price_request_index lacks raw/backward pair for {code}")

    frames: dict[str, list[pd.DataFrame]] = {"1": [], "3": []}
    records: list[dict[str, object]] = []
    for row in index.sort_values(["code", "adjustflag"]).itertuples(index=False):
        code = str(row.code)
        flag = str(row.adjustflag)
        request_key = str(row.request_key)
        frame, state, object_path = _read_committed_cache_object(
            cache_dir,
            request_key,
            expected_endpoint="history_k_data",
            expected_fields=EQUITY_DAILY_FIELDS,
        )
        args = state.get("args")
        if not isinstance(args, Mapping):
            raise CacheIntegrityError(f"cache state has no request args: {request_key}")
        if str(args.get("code")) != code or str(args.get("adjustflag")) != flag:
            raise CacheIntegrityError(f"price request index/state mismatch for {code} flag={flag}")
        if str(args.get("frequency")) != "d" or str(args.get("fields")) != ",".join(
            EQUITY_DAILY_FIELDS
        ):
            raise CacheIntegrityError(f"price request is not the registered daily schema: {request_key}")
        if (
            str(args.get("start_date")) != expected_start_date
            or str(args.get("end_date")) != expected_end_date
        ):
            raise CacheIntegrityError(
                f"price request range differs from frozen source policy: {request_key}"
            )
        if int(row.rows) != len(frame):
            raise CacheIntegrityError(f"price request index row count mismatch for {request_key}")
        if not frame["code"].astype(str).eq(code).all():
            raise CacheIntegrityError(f"price cache payload code mismatch for {request_key}")
        payload_flags = set(frame["adjustflag"].astype(str))
        if flag == "3" and payload_flags != {"3"}:
            raise CacheIntegrityError(f"price cache payload adjustflag mismatch for {request_key}")
        if flag == "1" and (
            not payload_flags or not payload_flags.issubset({"1", "3"})
        ):
            raise CacheIntegrityError(f"price cache payload adjustflag mismatch for {request_key}")
        frame_dates = pd.to_datetime(frame["date"], errors="coerce")
        if frame_dates.isna().any():
            raise CacheIntegrityError(f"price cache has invalid dates for {request_key}")
        if hasattr(row, "first_date") and str(pd.Timestamp(frame_dates.min()).date()) != str(
            pd.Timestamp(row.first_date).date()
        ):
            raise CacheIntegrityError(f"price request index first_date mismatch for {request_key}")
        if hasattr(row, "last_date") and str(pd.Timestamp(frame_dates.max()).date()) != str(
            pd.Timestamp(row.last_date).date()
        ):
            raise CacheIntegrityError(f"price request index last_date mismatch for {request_key}")
        selected_fields = _RAW_BUILD_FIELDS if flag == "3" else _ADJUSTED_BUILD_FIELDS
        selected = frame.loc[:, list(selected_fields)].copy()
        numeric_fields = [
            name
            for name in selected_fields
            if name not in {"date", "code", "adjustflag"}
        ]
        for name in numeric_fields:
            original = selected[name]
            text = original.astype("string").str.strip()
            blank = original.isna() | text.eq("")
            numeric = pd.to_numeric(original, errors="coerce")
            malformed = numeric.isna() & ~blank
            if malformed.any():
                raise CacheIntegrityError(
                    f"price cache contains malformed {name} for {request_key}"
                )
            selected[name] = numeric.astype(float)
        frames[flag].append(selected)
        records.append(
            {
                "code": code,
                "adjustflag": flag,
                "args": dict(args),
                "request_key": request_key,
                "rows": len(frame),
                "parquet_sha256": state["parquet_sha256"],
                "object": str(object_path),
                "provider_version": str(state.get("provider_version", "")),
                "payload_adjustflags": sorted(payload_flags),
            }
        )

    def assemble(items: list[pd.DataFrame]) -> pd.DataFrame:
        result = pd.concat(items, ignore_index=True)
        items.clear()
        result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
        if result["date"].isna().any():
            raise ValueError("price cache contains invalid dates")
        result.rename(columns={"code": "symbol"}, inplace=True)
        result.set_index(["date", "symbol"], inplace=True)
        result.sort_index(inplace=True)
        if not result.index.is_unique:
            raise ValueError("price cache contains duplicate date-symbol rows")
        return result

    raw_result = assemble(frames.pop("3"))
    adjusted_result = assemble(frames.pop("1"))
    raw_result, adjusted_result, boundaries = trim_leading_unadjusted_vendor_history(
        raw_result,
        adjusted_result,
        membership,
    )
    return (
        raw_result,
        adjusted_result,
        records,
        [asdict(boundary) for boundary in boundaries],
    )


def _load_adjustment_events(
    cache_dir: Path,
    request_index: pd.DataFrame,
    expected_codes: set[str],
    *,
    expected_start_date: str,
    expected_end_date: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Load the independently committed vendor corporate-action catalog."""

    required = {
        "code",
        "rows",
        "first_event_date",
        "last_event_date",
        "request_key",
    }
    missing = sorted(required.difference(request_index.columns))
    if missing:
        raise ValueError("adjustment_request_index missing: " + ",".join(missing))
    index = request_index.copy()
    index["code"] = index["code"].astype(str)
    if index["code"].duplicated().any():
        raise ValueError("adjustment_request_index contains duplicate securities")
    if set(index["code"]) != expected_codes:
        raise ValueError(
            "adjustment_request_index does not exactly cover the historical member union"
        )

    frames: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    for row in index.sort_values("code").itertuples(index=False):
        code = str(row.code)
        request_key = str(row.request_key)
        frame, state, object_path = _read_committed_cache_object(
            cache_dir,
            request_key,
            expected_endpoint="adjust_factor",
            expected_fields=ADJUST_FACTOR_FIELDS,
        )
        args = state.get("args")
        if not isinstance(args, Mapping) or str(args.get("code")) != code:
            raise CacheIntegrityError(
                f"adjustment request index/state mismatch for {code}"
            )
        start = pd.to_datetime(args.get("start_date"), errors="coerce")
        end = pd.to_datetime(args.get("end_date"), errors="coerce")
        if pd.isna(start) or pd.isna(end) or start > end:
            raise CacheIntegrityError(
                f"adjustment request has an invalid frozen date range for {code}"
            )
        if (
            str(pd.Timestamp(start).date()) != expected_start_date
            or str(pd.Timestamp(end).date()) != expected_end_date
        ):
            raise CacheIntegrityError(
                f"adjustment request range differs from frozen policy for {code}"
            )
        if int(row.rows) != len(frame):
            raise CacheIntegrityError(
                f"adjustment request index row count mismatch for {request_key}"
            )
        current = frame.copy()
        if len(current):
            if not current["code"].astype(str).eq(code).all():
                raise CacheIntegrityError(
                    f"adjustment cache payload code mismatch for {request_key}"
                )
            dates = pd.to_datetime(
                current["dividOperateDate"], errors="coerce"
            ).dt.normalize()
            if (
                dates.isna().any()
                or dates.duplicated().any()
                or (dates < pd.Timestamp(start).normalize()).any()
                or (dates > pd.Timestamp(end).normalize()).any()
            ):
                raise CacheIntegrityError(
                    f"adjustment cache has invalid event dates for {request_key}"
                )
            for field in (
                "foreAdjustFactor",
                "backAdjustFactor",
                "adjustFactor",
            ):
                values = pd.to_numeric(current[field], errors="coerce")
                if (
                    values.isna().any()
                    or not np.isfinite(values.to_numpy(dtype=float)).all()
                    or values.le(0).any()
                ):
                    raise CacheIntegrityError(
                        f"adjustment cache has invalid {field} for {request_key}"
                    )
            first_event = str(dates.min().date())
            last_event = str(dates.max().date())
            declared_first = (
                ""
                if pd.isna(row.first_event_date)
                else str(row.first_event_date).strip()
            )
            declared_last = (
                ""
                if pd.isna(row.last_event_date)
                else str(row.last_event_date).strip()
            )
            if declared_first != first_event or declared_last != last_event:
                raise CacheIntegrityError(
                    f"adjustment request index event bounds mismatch for {request_key}"
                )
            current["dividOperateDate"] = dates
            frames.append(current)
        else:
            declared_first = (
                ""
                if pd.isna(row.first_event_date)
                else str(row.first_event_date).strip()
            )
            declared_last = (
                ""
                if pd.isna(row.last_event_date)
                else str(row.last_event_date).strip()
            )
            if declared_first or declared_last:
                raise CacheIntegrityError(
                    f"empty adjustment request declares event bounds for {request_key}"
                )
        records.append(
            {
                "code": code,
                "request_key": request_key,
                "rows": len(frame),
                "start_date": str(pd.Timestamp(start).date()),
                "end_date": str(pd.Timestamp(end).date()),
                "args": dict(args),
                "parquet_sha256": state["parquet_sha256"],
                "object": str(object_path),
                "provider_version": str(state.get("provider_version", "")),
            }
        )
    if frames:
        events = pd.concat(frames, ignore_index=True).sort_values(
            ["code", "dividOperateDate"]
        ).reset_index(drop=True)
    else:
        events = pd.DataFrame(columns=list(ADJUST_FACTOR_FIELDS), dtype=object)
    return events, records


_TERMINAL_STALE_REASON = "terminal_stale_membership_after_verified_out_date"


def _classify_terminal_stale_membership_exceptions(
    membership: pd.DataFrame,
    market_index: pd.MultiIndex,
    security_master: pd.DataFrame,
) -> pd.DataFrame:
    """Prove the only permitted member-without-bar exception.

    No prices are synthesized.  A missing member row is representable only
    on/after the security master's parseable ``outDate`` and only for an
    inactive security.  The last cached bar must land either on that date or
    on the immediately preceding market session when the out-date itself has
    no security bar.  Every other member/history mismatch remains a hard build
    failure.
    """

    member_index = pd.MultiIndex.from_frame(
        membership[["date", "code"]].rename(columns={"code": "symbol"}),
        names=["date", "symbol"],
    )
    missing = member_index.difference(market_index)
    columns = (
        "vendor_universe_member",
        "universe_member",
        "research_eligible",
        "reason",
        "stock_basic_out_date",
        "last_raw_bar_date",
        "membership_snapshot_date",
        "security_status",
        "price_fill_policy",
        "terminal_return_policy",
    )
    if not len(missing):
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([], name="date"), pd.Index([], name="symbol")],
            names=["date", "symbol"],
        )
        return pd.DataFrame(columns=list(columns), index=empty_index)

    required = {"code", "outDate", "status"}
    absent = sorted(required.difference(security_master.columns))
    if absent:
        raise ValueError(
            "security_master cannot prove terminal membership exceptions: "
            + ",".join(absent)
        )
    basic = security_master[["code", "outDate", "status"]].copy()
    basic["code"] = basic["code"].astype(str)
    if basic["code"].duplicated().any():
        raise ValueError("security_master contains duplicate codes")
    raw_out = basic["outDate"].astype("string").str.strip().replace("", pd.NA)
    basic["out_date"] = pd.to_datetime(raw_out, errors="coerce").dt.normalize()
    if (raw_out.notna() & basic["out_date"].isna()).any():
        raise ValueError("security_master contains an invalid nonblank outDate")
    basic = basic.set_index("code")
    last_bar = pd.Series(
        market_index.get_level_values("date"), index=market_index
    ).groupby(level="symbol").max()
    market_dates = pd.DatetimeIndex(
        market_index.get_level_values("date").unique()
    ).normalize().sort_values()
    membership_lookup = membership.rename(columns={"code": "symbol"}).set_index(
        ["date", "symbol"]
    )
    records: list[dict[str, object]] = []
    rejected: list[str] = []
    for date, symbol_value in missing:
        date = pd.Timestamp(date).normalize()
        symbol = str(symbol_value)
        if symbol not in basic.index or symbol not in last_bar.index:
            rejected.append(f"{date.date()}|{symbol}:no-security-or-bar-history")
            continue
        out_date = basic.loc[symbol, "out_date"]
        status = str(basic.loc[symbol, "status"]).strip()
        observed_last = pd.Timestamp(last_bar.loc[symbol]).normalize()
        previous_sessions = (
            market_dates[market_dates < pd.Timestamp(out_date)]
            if not pd.isna(out_date)
            else pd.DatetimeIndex([])
        )
        last_is_verified = bool(
            not pd.isna(out_date)
            and (
                observed_last == pd.Timestamp(out_date)
                or (
                    len(previous_sessions)
                    and observed_last == previous_sessions[-1]
                )
            )
        )
        if (
            pd.isna(out_date)
            or status != "0"
            or date < pd.Timestamp(out_date)
            or not last_is_verified
            or (
                observed_last == pd.Timestamp(out_date)
                and date <= observed_last
            )
        ):
            rejected.append(
                f"{date.date()}|{symbol}:out={out_date},last={observed_last.date()},status={status}"
            )
            continue
        membership_snapshot = pd.Timestamp(
            membership_lookup.loc[(date, symbol), "vendor_update_date"]
        ).normalize()
        records.append(
            {
                "date": date,
                "symbol": symbol,
                "vendor_universe_member": True,
                "universe_member": True,
                "research_eligible": False,
                "reason": _TERMINAL_STALE_REASON,
                "stock_basic_out_date": pd.Timestamp(out_date),
                "last_raw_bar_date": observed_last,
                "membership_snapshot_date": membership_snapshot,
                "security_status": status,
                "price_fill_policy": "none",
                "terminal_return_policy": "unresolved_no_imputation",
            }
        )
    if rejected:
        raise ValueError(
            f"market history misses {len(missing)} member rows; "
            "non-terminal-stale examples: " + "; ".join(rejected[:10])
        )
    output = pd.DataFrame.from_records(records).set_index(["date", "symbol"])
    output.index = output.index.set_names(["date", "symbol"])
    return output.loc[:, list(columns)].sort_index()


def _find_index_request_key(normalized_dir: Path, cache_dir: Path) -> str:
    price_manifest = normalized_dir / "price_manifest.json"
    if price_manifest.is_file():
        manifest = _read_json(price_manifest)
        index_request = manifest.get("index_request")
        if isinstance(index_request, Mapping) and str(index_request.get("request_key", "")):
            return str(index_request["request_key"])
    matches: list[str] = []
    for state_path in sorted((cache_dir / "state").glob("*.json")):
        state = _read_json(state_path)
        args = state.get("args")
        if (
            state.get("status") == "success"
            and state.get("endpoint") == "index_history"
            and isinstance(args, Mapping)
            and str(args.get("code")) == "sh.000300"
        ):
            matches.append(str(state.get("request_key", "")))
    if len(matches) != 1:
        raise CacheIntegrityError(
            f"expected exactly one committed sh.000300 index request, found {len(matches)}"
        )
    return matches[0]


def _load_benchmark_returns(
    cache_dir: Path,
    request_key: str,
    *,
    expected_start_date: str,
    expected_end_date: str,
) -> tuple[pd.Series, dict[str, object]]:
    frame, state, object_path = _read_committed_cache_object(
        cache_dir,
        request_key,
        expected_endpoint="index_history",
        expected_fields=INDEX_DAILY_FIELDS,
    )
    args = state.get("args")
    if not isinstance(args, Mapping) or args != {
        "code": "sh.000300",
        "fields": ",".join(INDEX_DAILY_FIELDS),
        "start_date": expected_start_date,
        "end_date": expected_end_date,
        "frequency": "d",
        "adjustflag": "3",
    }:
        raise CacheIntegrityError("HS300 index request args differ from frozen policy")
    date = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    close = pd.to_numeric(frame["close"], errors="coerce")
    if date.isna().any() or close.isna().any() or (close <= 0).any():
        raise ValueError("HS300 index cache contains invalid date/close")
    series = pd.Series(close.to_numpy(dtype=float), index=pd.DatetimeIndex(date, name="date"))
    if series.index.has_duplicates:
        raise ValueError("HS300 index cache contains duplicate dates")
    returns = series.sort_index().pct_change(fill_method=None).rename("benchmark_return")
    record = {
        "request_key": request_key,
        "args": dict(args),
        "rows": len(frame),
        "parquet_sha256": state["parquet_sha256"],
        "object": str(object_path),
        "provider_version": str(state.get("provider_version", "")),
    }
    return returns, record


def _industry_asof(
    member_index: pd.MultiIndex,
    industry_snapshots: pd.DataFrame,
    *,
    maximum_age_days: int,
) -> pd.DataFrame:
    required = {"requested_date", "vendor_update_date", "code", "industry"}
    missing = sorted(required.difference(industry_snapshots.columns))
    if missing:
        raise ValueError("industry_snapshots missing: " + ",".join(missing))
    industry = industry_snapshots.copy()
    for name in ("requested_date", "vendor_update_date"):
        industry[name] = pd.to_datetime(
            industry[name], errors="coerce"
        ).dt.normalize()
    industry["code"] = industry["code"].astype("string").str.strip()
    industry["industry"] = industry["industry"].astype("string").str.strip()
    if industry[["requested_date", "vendor_update_date"]].isna().any().any():
        raise ValueError("industry snapshots contain invalid dates")
    if (industry["vendor_update_date"] > industry["requested_date"]).any():
        raise ValueError("industry snapshots contain future vendor metadata")
    if industry.duplicated(["requested_date", "code"]).any():
        raise ValueError("industry snapshots contain duplicate date-symbol rows")

    result = pd.DataFrame(index=member_index, columns=["industry", "industry_snapshot_date"])
    date_values = member_index.get_level_values("date")
    symbol_values = member_index.get_level_values("symbol").astype(str)
    for symbol in pd.Index(symbol_values.unique()).sort_values():
        target_index = member_index[symbol_values == symbol]
        target_dates = pd.DatetimeIndex(target_index.get_level_values("date"))
        history = industry.loc[industry["code"].astype(str) == symbol].sort_values(
            "requested_date"
        )
        if history.empty:
            raise ValueError(f"industry history missing for {symbol}")
        source_dates = pd.DatetimeIndex(history["requested_date"])
        locations = source_dates.searchsorted(target_dates, side="right") - 1
        if (locations < 0).any():
            raise ValueError(f"industry history starts after membership for {symbol}")
        chosen_dates = source_dates[locations]
        ages = (target_dates - chosen_dates).days
        if (ages > maximum_age_days).any():
            first = int(np.flatnonzero(ages > maximum_age_days)[0])
            raise ValueError(
                f"industry snapshot stale for {symbol} on {target_dates[first].date()}: "
                f"{int(ages[first])} days"
            )
        values = history.iloc[locations]["industry"].astype("string").to_numpy()
        if pd.isna(values).any() or any(not str(value).strip() for value in values):
            raise ValueError(f"blank industry classification for {symbol}")
        result.loc[target_index, "industry"] = values
        result.loc[target_index, "industry_snapshot_date"] = chosen_dates.to_numpy()
    result["industry"] = result["industry"].astype("string")
    result["industry_snapshot_date"] = pd.to_datetime(result["industry_snapshot_date"])
    if (result["industry_snapshot_date"].array > date_values).any():
        raise AssertionError("industry as-of join leaked a future snapshot")
    return result


def _round_half_up_to_tick(values: pd.Series, tick: float = 0.01) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.floor(numeric / tick + 0.5 + 1e-10) * tick


def derive_conservative_one_price_limits(
    panel: pd.DataFrame,
    *,
    minimum_listed_days: int = 30,
) -> pd.DataFrame:
    """Derive separate execution-lock and close-at-limit research proxies.

    Both paths require a recognized board/rule era, a mature listing, an
    exchange-adjusted ``preclose`` and a real tradable row.  ``limit_up/down``
    additionally require identical OHLC and remain the one-price execution
    lock proxy.  ``research_close_limit_*_proxy`` only require raw close to hit
    the theoretical rounded limit and implement the report's cross-sectional
    exclusion.  Unknown rows remain false and are ineligible fail-closed.
    """

    if minimum_listed_days < 0:
        raise ValueError("minimum_listed_days cannot be negative")
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["date", "symbol"]:
        raise ValueError("limit proxy panel must use MultiIndex(date, symbol)")
    required = {
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "is_st",
        "listed_days",
        "tradable",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError("cannot derive limit proxy; missing: " + ",".join(missing))
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    symbols = pd.Index(panel.index.get_level_values("symbol").astype(str))
    is_st = panel["is_st"].fillna(False).astype(bool).to_numpy()
    listed = pd.to_numeric(panel["listed_days"], errors="coerce").to_numpy(dtype=float)
    tradable = panel["tradable"].fillna(False).astype(bool).to_numpy()
    prices = panel[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    preclose = pd.to_numeric(panel["preclose"], errors="coerce")

    main = symbols.str.match(r"^(sh\.(600|601|603|605)|sz\.(000|001|002|003))")
    star = symbols.str.match(r"^sh\.(688|689)")
    gem = symbols.str.match(r"^sz\.(300|301)")
    recognized = np.asarray(main | star | gem, dtype=bool)
    ratio = np.full(len(panel), np.nan, dtype=float)
    ratio[np.asarray(main)] = np.where(is_st[np.asarray(main)], 0.05, 0.10)
    ratio[np.asarray(star)] = 0.20
    gem_mask = np.asarray(gem)
    before_reform = dates < pd.Timestamp("2020-08-24")
    ratio[gem_mask & before_reform] = np.where(
        is_st[gem_mask & before_reform], 0.05, 0.10
    )
    ratio[gem_mask & ~before_reform] = 0.20

    price_valid = prices.notna().all(axis=1).to_numpy() & preclose.gt(0).to_numpy()
    known = recognized & (listed >= minimum_listed_days) & tradable & price_valid
    one_price = prices.max(axis=1).sub(prices.min(axis=1)).abs().le(1e-8).to_numpy()
    up_price = _round_half_up_to_tick(preclose * (1.0 + ratio))
    down_price = _round_half_up_to_tick(preclose * (1.0 - ratio))
    close = prices["close"]
    close_up = known & close.sub(up_price).abs().le(1e-8).to_numpy()
    close_down = known & close.sub(down_price).abs().le(1e-8).to_numpy()
    up = close_up & one_price
    down = close_down & one_price
    if np.any(up & down):
        raise AssertionError("one row cannot be both an up-limit and down-limit")
    return pd.DataFrame(
        {
            "limit_up": up,
            "limit_down": down,
            "limit_proxy_known": known,
            "limit_ratio_proxy": ratio,
            "limit_up_quality": "research_proxy",
            "limit_down_quality": "research_proxy",
            "limit_proxy_method": "one_price_ohlc_and_rounded_preclose_rule",
            "research_close_limit_up_proxy": close_up,
            "research_close_limit_down_proxy": close_down,
            "research_close_limit_known_proxy": known,
            "research_close_limit_up_proxy_quality": "research_proxy",
            "research_close_limit_down_proxy_quality": "research_proxy",
            "research_close_limit_known_proxy_quality": "research_proxy",
            "research_close_limit_proxy_method": (
                "raw_close_and_rounded_preclose_board_rule"
            ),
        },
        index=panel.index,
    )


def _derive_research_eligibility(
    features: pd.DataFrame,
    limits: pd.DataFrame,
    vendor_member_index: pd.MultiIndex,
) -> pd.Series:
    """Return the frozen, current-row-only research selection mask.

    This mask is intentionally distinct from provider membership.  It never
    asks whether a future label or exit price will exist.  It only applies the
    report's ex-ante stock filters at the current close: a real tradable bar,
    non-ST status, at least one calendar year since IPO, finite current risk
    and capacity inputs, and no recognized one-price limit lock.  Unknown
    limit status or unavailable risk/capacity data fails closed.
    """

    if not features.index.equals(limits.index):
        raise ValueError("feature and limit indexes differ for research eligibility")
    required_features = {
        "tradable",
        "is_st",
        "listed_days",
        "log_mcap",
        "beta",
        "adv20_dollars",
    }
    required_limits = {
        "research_close_limit_up_proxy",
        "research_close_limit_down_proxy",
        "research_close_limit_known_proxy",
    }
    missing_features = sorted(required_features.difference(features.columns))
    missing_limits = sorted(required_limits.difference(limits.columns))
    if missing_features or missing_limits:
        raise ValueError(
            "research eligibility inputs are incomplete: "
            + ",".join([*missing_features, *missing_limits])
        )
    member = pd.Series(
        features.index.isin(vendor_member_index),
        index=features.index,
        dtype=bool,
    )
    listed_days = pd.to_numeric(features["listed_days"], errors="coerce")
    log_mcap = pd.to_numeric(features["log_mcap"], errors="coerce")
    beta = pd.to_numeric(features["beta"], errors="coerce")
    adv20_dollars = pd.to_numeric(features["adv20_dollars"], errors="coerce")
    risk_and_capacity_available = pd.Series(
        np.isfinite(log_mcap.to_numpy(dtype=float))
        & np.isfinite(beta.to_numpy(dtype=float))
        & np.isfinite(adv20_dollars.to_numpy(dtype=float))
        & adv20_dollars.gt(0).to_numpy(dtype=bool),
        index=features.index,
        dtype=bool,
    )
    eligible = (
        member
        & features["tradable"].fillna(False).astype(bool)
        & ~features["is_st"].fillna(True).astype(bool)
        & listed_days.ge(365).fillna(False)
        & risk_and_capacity_available
        & limits["research_close_limit_known_proxy"].fillna(False).astype(bool)
        & ~limits["research_close_limit_up_proxy"].fillna(True).astype(bool)
        & ~limits["research_close_limit_down_proxy"].fillna(True).astype(bool)
    )
    return eligible.astype(bool).rename("research_eligible")


def _build_metadata(
    config: HS300BuildConfig,
    input_hashes: Mapping[str, str],
) -> dict[str, object]:
    return {
        "vendor": "BaoStock 0.9.x free public API",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "corporate_action_policy": "raw_execution_plus_backward_adjusted_research_view",
        "calendar": f"BaoStock trade_dates sha256={input_hashes['trade_calendar']}",
        "includes_delisted": True,
        "point_in_time_universe": True,
        "execution_price_basis": "raw_unadjusted",
        "research_price_basis": "backward_adjusted",
        "dataset_tier": "research_proxy",
        "benchmark_weight_quality": "research_proxy",
        "benchmark_weight_source": "implied_float_mcap_or_equal_weight_fallback",
        "field_quality": hs300_field_quality_map(),
        "research_watermark": RESEARCH_PROXY_WATERMARK,
        "research_start": str(pd.Timestamp(config.research_start).date()),
        "research_end": (
            str(pd.Timestamp(config.research_end).date()) if config.research_end else "latest_cached"
        ),
    }


def _write_snapshot(
    panel: pd.DataFrame,
    feature_tape: pd.DataFrame,
    terminal_exceptions: pd.DataFrame,
    report: DataValidationReport,
    config: HS300BuildConfig,
    *,
    input_hashes: Mapping[str, str],
    membership_cache_records: Sequence[Mapping[str, object]],
    membership_lineage_policy: Mapping[str, object],
    security_cache_records: Sequence[Mapping[str, object]],
    security_lineage_policy: Mapping[str, object],
    industry_cache_records: Sequence[Mapping[str, object]],
    industry_lineage_policy: Mapping[str, object],
    price_cache_records: Sequence[Mapping[str, object]],
    price_history_boundaries: Sequence[Mapping[str, object]],
    adjustment_cache_records: Sequence[Mapping[str, object]],
    adjustment_event_audit: Mapping[str, object],
    raw_preclose_adjustment_diagnostic: Mapping[str, object],
    index_cache_record: Mapping[str, object],
) -> tuple[Path, Path, tuple[Path, ...], tuple[Path, ...], dict[str, object]]:
    feature_hashed = pd.util.hash_pandas_object(
        feature_tape, index=True, categorize=True
    ).to_numpy(dtype=np.uint64)
    feature_snapshot_hash = hashlib.sha256(feature_hashed.tobytes()).hexdigest()
    del feature_hashed
    price_cache_contract_hash = hashlib.sha256(
        json.dumps(
            list(price_cache_records),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    price_history_boundary_hash = hashlib.sha256(
        json.dumps(
            list(price_history_boundaries),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    membership_cache_contract_hash = hashlib.sha256(
        json.dumps(
            list(membership_cache_records),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    security_cache_contract_hash = hashlib.sha256(
        json.dumps(
            list(security_cache_records),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    industry_cache_contract_hash = hashlib.sha256(
        json.dumps(
            list(industry_cache_records),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    lineage_domains: dict[str, dict[str, object]] = {}
    for name, policy, cache_hash, normalized_names in (
        (
            "membership",
            membership_lineage_policy,
            membership_cache_contract_hash,
            (
                "trade_calendar",
                "membership_snapshots",
                "membership_daily",
                "security_union",
                "membership_request_index",
            ),
        ),
        (
            "security",
            security_lineage_policy,
            security_cache_contract_hash,
            ("security_master", "security_master_request_index"),
        ),
        (
            "industry",
            industry_lineage_policy,
            industry_cache_contract_hash,
            ("industry_snapshots", "industry_request_index"),
        ),
    ):
        identity: dict[str, object] = {
            "policy": dict(policy),
            "cache_objects_sha256": cache_hash,
            "normalized_input_sha256": {
                key: input_hashes[key] for key in normalized_names
            },
        }
        lineage_domains[name] = {
            **identity,
            "identity_sha256": hashlib.sha256(
                json.dumps(
                    identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
        }
    source_lineage_root_hash = hashlib.sha256(
        json.dumps(
            lineage_domains,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    adjustment_cache_contract_hash = hashlib.sha256(
        json.dumps(
            list(adjustment_cache_records),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    terminal_exception_hashed = pd.util.hash_pandas_object(
        terminal_exceptions, index=True, categorize=True
    ).to_numpy(dtype=np.uint64)
    terminal_exception_snapshot_hash = hashlib.sha256(
        terminal_exception_hashed.tobytes()
    ).hexdigest()
    del terminal_exception_hashed
    eligibility_columns = (
        "vendor_universe_member",
        "universe_member",
        "research_eligible",
    )
    missing_eligibility = sorted(
        set(eligibility_columns).difference(feature_tape.columns)
    )
    if missing_eligibility:
        raise ValueError(
            "feature tape lacks frozen membership/eligibility fields: "
            + ",".join(missing_eligibility)
        )
    eligibility_hashed = pd.util.hash_pandas_object(
        feature_tape.loc[:, list(eligibility_columns)],
        index=True,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    eligibility_snapshot_hash = hashlib.sha256(
        eligibility_hashed.tobytes()
    ).hexdigest()
    del eligibility_hashed
    execution_limit_columns = ("limit_up", "limit_down", "limit_proxy_known")
    research_limit_columns = (
        "research_close_limit_up_proxy",
        "research_close_limit_down_proxy",
        "research_close_limit_known_proxy",
    )
    execution_limit_hashed = pd.util.hash_pandas_object(
        feature_tape.loc[:, list(execution_limit_columns)],
        index=True,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    execution_limit_snapshot_hash = hashlib.sha256(
        execution_limit_hashed.tobytes()
    ).hexdigest()
    del execution_limit_hashed
    research_limit_hashed = pd.util.hash_pandas_object(
        feature_tape.loc[:, list(research_limit_columns)],
        index=True,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    research_limit_snapshot_hash = hashlib.sha256(
        research_limit_hashed.tobytes()
    ).hexdigest()
    del research_limit_hashed
    tape_vendor_member = feature_tape["vendor_universe_member"].astype(bool)
    tape_research_eligible = feature_tape["research_eligible"].astype(bool)
    if (tape_research_eligible & ~tape_vendor_member).any():
        raise ValueError("research eligibility contains a non-member row")
    eligible_daily = (
        tape_research_eligible.loc[tape_vendor_member]
        .groupby(level="date")
        .sum()
    )
    index_cache_contract_hash = hashlib.sha256(
        json.dumps(
            dict(index_cache_record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    storage_contract = {
        "panel_snapshot_hash": report.snapshot_hash,
        "feature_snapshot_hash": feature_snapshot_hash,
        "normalized_input_sha256": dict(input_hashes),
        "price_cache_objects_sha256": price_cache_contract_hash,
        "price_history_boundaries_sha256": price_history_boundary_hash,
        "membership_cache_objects_sha256": membership_cache_contract_hash,
        "security_cache_objects_sha256": security_cache_contract_hash,
        "industry_cache_objects_sha256": industry_cache_contract_hash,
        "source_lineage_root_sha256": source_lineage_root_hash,
        "adjustment_cache_objects_sha256": adjustment_cache_contract_hash,
        "terminal_stale_membership_exceptions_sha256": (
            terminal_exception_snapshot_hash
        ),
        "research_eligibility_sha256": eligibility_snapshot_hash,
        "execution_one_price_limit_proxy_sha256": (
            execution_limit_snapshot_hash
        ),
        "research_close_limit_proxy_sha256": research_limit_snapshot_hash,
        "index_cache_object_sha256": index_cache_contract_hash,
    }
    storage_snapshot_id = hashlib.sha256(
        json.dumps(
            storage_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    snapshot_dir = config.output_dir / "snapshots" / storage_snapshot_id
    terminal_exception_path = (
        snapshot_dir / "terminal_stale_membership_exceptions.parquet"
    )
    _atomic_parquet(terminal_exceptions, terminal_exception_path)
    restored_terminal_exceptions = pd.read_parquet(terminal_exception_path)
    if (
        not isinstance(restored_terminal_exceptions.index, pd.MultiIndex)
        or restored_terminal_exceptions.index.names != ["date", "symbol"]
        or not restored_terminal_exceptions.index.equals(terminal_exceptions.index)
        or tuple(restored_terminal_exceptions.columns)
        != tuple(terminal_exceptions.columns)
    ):
        raise CacheIntegrityError(
            "written terminal-stale membership sidecar verification failed"
        )
    partitions: list[Path] = []
    partition_records: list[dict[str, object]] = []
    years = pd.Index(panel.index.get_level_values("date").year.unique()).sort_values()
    for year in years:
        mask = panel.index.get_level_values("date").year == int(year)
        block = panel.loc[mask]
        target = snapshot_dir / f"year={int(year)}" / "part-000.parquet"
        _atomic_parquet(block, target)
        restored = pd.read_parquet(target)
        same_index = (
            isinstance(restored.index, pd.MultiIndex)
            and restored.index.names == block.index.names
            and np.array_equal(
                pd.DatetimeIndex(restored.index.get_level_values("date")),
                pd.DatetimeIndex(block.index.get_level_values("date")),
            )
            and np.array_equal(
                restored.index.get_level_values("symbol").astype(str),
                block.index.get_level_values("symbol").astype(str),
            )
        )
        if len(restored) != len(block) or not same_index:
            raise CacheIntegrityError(f"written partition verification failed for {year}")
        partitions.append(target)
        partition_records.append(
            {
                "year": int(year),
                "path": str(target),
                "rows": len(block),
                "sha256": _sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )
    feature_partitions: list[Path] = []
    feature_partition_records: list[dict[str, object]] = []
    feature_columns: tuple[str, ...] | None = None
    feature_dtypes: dict[str, str] | None = None
    feature_years = pd.Index(
        feature_tape.index.get_level_values("date").year.unique()
    ).sort_values()
    for year in feature_years:
        mask = feature_tape.index.get_level_values("date").year == int(year)
        block = feature_tape.loc[mask]
        target = snapshot_dir / "feature_tape" / f"year={int(year)}" / "part-000.parquet"
        _atomic_parquet(block, target)
        restored = pd.read_parquet(target)
        same_index = (
            isinstance(restored.index, pd.MultiIndex)
            and restored.index.names == block.index.names
            and restored.index.equals(block.index)
        )
        if len(restored) != len(block) or not same_index:
            raise CacheIntegrityError(f"written feature-tape partition verification failed for {year}")
        restored_columns = tuple(str(name) for name in restored.columns)
        restored_dtypes = {str(name): str(dtype) for name, dtype in restored.dtypes.items()}
        if feature_columns is None:
            feature_columns = restored_columns
            feature_dtypes = restored_dtypes
        elif restored_columns != feature_columns or restored_dtypes != feature_dtypes:
            raise CacheIntegrityError(
                f"written feature-tape partition schema differs for {year}"
            )
        feature_partitions.append(target)
        feature_partition_records.append(
            {
                "year": int(year),
                "path": str(target),
                "rows": len(block),
                "sha256": _sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_hash": report.snapshot_hash,
        "storage_snapshot_id": storage_snapshot_id,
        "rows": report.rows,
        "symbols": report.symbols,
        "trading_days": report.trading_days,
        "first_date": report.first_date,
        "last_date": report.last_date,
        "research_proxy_only": True,
        "research_watermark": RESEARCH_PROXY_WATERMARK,
        "config": {
            **asdict(config),
            "cache_dir": str(config.cache_dir),
            "normalized_dir": str(config.normalized_dir),
            "output_dir": str(config.output_dir),
        },
        "validation": report.to_dict(),
        "normalized_input_sha256": dict(input_hashes),
        "membership_cache_objects": list(membership_cache_records),
        "membership_lineage": dict(membership_lineage_policy),
        "security_cache_objects": list(security_cache_records),
        "security_lineage": dict(security_lineage_policy),
        "industry_cache_objects": list(industry_cache_records),
        "industry_lineage": dict(industry_lineage_policy),
        "source_lineage": {
            "schema_version": "1.0",
            "complete": True,
            "domains": lineage_domains,
            "root_sha256": source_lineage_root_hash,
        },
        "price_cache_objects": list(price_cache_records),
        "price_history_boundaries": {
            "schema_version": "1.0",
            "complete": True,
            "policy": "trim_only_leading_adjustflag_3_rows_before_first_membership",
            "rows": len(price_history_boundaries),
            "sha256": price_history_boundary_hash,
            "records": list(price_history_boundaries),
        },
        "adjustment_cache_objects": list(adjustment_cache_records),
        "adjustment_event_audit": dict(adjustment_event_audit),
        "raw_preclose_adjustment_diagnostic": dict(
            raw_preclose_adjustment_diagnostic
        ),
        "index_cache_object": dict(index_cache_record),
        "terminal_stale_membership_exceptions": {
            "schema_version": "1.0",
            "complete": True,
            "policy": _TERMINAL_STALE_REASON,
            "price_fill_policy": "none",
            "terminal_return_policy": "unresolved_no_imputation",
            "rows": len(terminal_exceptions),
            "symbols": int(
                terminal_exceptions.index.get_level_values("symbol").nunique()
            ),
            "trading_days": int(
                terminal_exceptions.index.get_level_values("date").nunique()
            ),
            "snapshot_hash": terminal_exception_snapshot_hash,
            "file": {
                "path": str(terminal_exception_path),
                "rows": len(terminal_exceptions),
                "sha256": _sha256_file(terminal_exception_path),
                "bytes": terminal_exception_path.stat().st_size,
            },
        },
        "research_eligibility": {
            "schema_version": "1.0",
            "complete": True,
            "provider_membership_column": "vendor_universe_member",
            "legacy_membership_alias": "universe_member",
            "selection_column": "research_eligible",
            "decision_time": "t_close",
            "current_time_only": True,
            "future_endpoint_availability_filter": False,
            "minimum_listed_calendar_days": 365,
            "requires_real_cached_bar": True,
            "requires_tradable_status_and_positive_volume": True,
            "excludes_st": True,
            "required_finite_fields": ["log_mcap", "beta", "adv20_dollars"],
            "requires_positive_adv20_dollars": True,
            "missing_current_risk_or_capacity_policy": "ineligible_fail_closed",
            "limit_policy": "known_and_not_raw_close_at_theoretical_limit",
            "limit_source_columns": list(research_limit_columns),
            "unknown_research_close_limit_policy": "ineligible_fail_closed",
            "terminal_stale_membership_policy": "ineligible_sidecar_no_bar_fill",
            "vendor_member_rows_with_bars": int(tape_vendor_member.sum()),
            "eligible_rows": int(tape_research_eligible.sum()),
            "eligible_daily_min": (
                int(eligible_daily.min()) if len(eligible_daily) else 0
            ),
            "eligible_daily_max": (
                int(eligible_daily.max()) if len(eligible_daily) else 0
            ),
            "snapshot_hash": eligibility_snapshot_hash,
        },
        "execution_one_price_limit_proxy": {
            "schema_version": "1.0",
            "complete": True,
            "purpose": "execution_lock_diagnostic_and_legacy_limit_fields",
            "decision_time": "t_close",
            "columns": list(execution_limit_columns),
            "method": "one_price_ohlc_and_rounded_preclose_board_rule",
            "unknown_policy": "not_asserted_as_locked",
            "quality": "research_proxy",
            "up_rows": int(feature_tape["limit_up"].astype(bool).sum()),
            "down_rows": int(feature_tape["limit_down"].astype(bool).sum()),
            "known_rows": int(
                feature_tape["limit_proxy_known"].astype(bool).sum()
            ),
            "snapshot_hash": execution_limit_snapshot_hash,
        },
        "research_close_limit_proxy": {
            "schema_version": "1.0",
            "complete": True,
            "purpose": "research_cross_sectional_eligibility_exclusion",
            "decision_time": "t_close",
            "columns": list(research_limit_columns),
            "method": "raw_close_and_rounded_preclose_board_rule",
            "unknown_policy": "ineligible_fail_closed",
            "quality": "research_proxy",
            "up_rows": int(
                feature_tape["research_close_limit_up_proxy"].astype(bool).sum()
            ),
            "down_rows": int(
                feature_tape["research_close_limit_down_proxy"].astype(bool).sum()
            ),
            "known_rows": int(
                feature_tape["research_close_limit_known_proxy"].astype(bool).sum()
            ),
            "snapshot_hash": research_limit_snapshot_hash,
        },
        "partitions": partition_records,
        "feature_tape": {
            "schema_version": "1.0",
            "complete": True,
            "purpose": "continuous_union_history_for_time_series_features_labels_and_execution",
            "index_policy": (
                "raw_cache_union_after_strict_pre_membership_adjusted_lineage_boundaries"
            ),
            "provider_members_per_day": config.expected_members,
            "member_bar_exception_policy": _TERMINAL_STALE_REASON,
            "storage_snapshot_id": storage_snapshot_id,
            "columns": list(feature_columns or ()),
            "dtypes": dict(feature_dtypes or {}),
            "source_contract": {
                "raw_adjustflag": "3",
                "research_adjustflag": "1",
                "price_cache_pairs": len(price_cache_records) // 2,
                "price_cache_objects_sha256": price_cache_contract_hash,
                "price_history_boundaries_sha256": price_history_boundary_hash,
                "membership_cache_objects_sha256": (
                    membership_cache_contract_hash
                ),
                "security_cache_objects_sha256": security_cache_contract_hash,
                "industry_cache_objects_sha256": industry_cache_contract_hash,
                "source_lineage_root_sha256": source_lineage_root_hash,
                "membership_snapshots_sha256": input_hashes[
                    "membership_snapshots"
                ],
                "membership_request_index_sha256": input_hashes[
                    "membership_request_index"
                ],
                "security_union_sha256": input_hashes["security_union"],
                "security_master_request_index_sha256": input_hashes[
                    "security_master_request_index"
                ],
                "industry_request_index_sha256": input_hashes[
                    "industry_request_index"
                ],
                "industry_snapshots_sha256": input_hashes["industry_snapshots"],
                "adjustment_cache_objects_sha256": (
                    adjustment_cache_contract_hash
                ),
                "adjustment_request_index_sha256": input_hashes[
                    "adjustment_request_index"
                ],
                "terminal_stale_membership_exceptions_sha256": (
                    terminal_exception_snapshot_hash
                ),
                "execution_one_price_limit_proxy_sha256": (
                    execution_limit_snapshot_hash
                ),
                "research_close_limit_proxy_sha256": (
                    research_limit_snapshot_hash
                ),
                "index_cache_object_sha256": index_cache_contract_hash,
                "membership_daily_sha256": input_hashes["membership_daily"],
                "trade_calendar_sha256": input_hashes["trade_calendar"],
                "security_master_sha256": input_hashes["security_master"],
            },
            "rows": len(feature_tape),
            "symbols": int(feature_tape.index.get_level_values("symbol").nunique()),
            "trading_days": int(feature_tape.index.get_level_values("date").nunique()),
            "first_date": str(feature_tape.index.get_level_values("date").min().date()),
            "last_date": str(feature_tape.index.get_level_values("date").max().date()),
            "snapshot_hash": feature_snapshot_hash,
            "partitions": feature_partition_records,
        },
    }
    manifest_path = config.output_dir / "manifest.json"
    _atomic_json(manifest, manifest_path)
    manifest_hash = _sha256_file(manifest_path)
    manifest_sha_path = config.output_dir / "manifest.sha256"
    _atomic_text(f"{manifest_hash}  {manifest_path.name}\n", manifest_sha_path)
    return (
        manifest_path,
        manifest_sha_path,
        tuple(partitions),
        tuple(feature_partitions),
        manifest,
    )


def build_hs300_research_panel(config: HS300BuildConfig) -> HS300BuildResult:
    """Build, validate and atomically freeze the free-data HS300 panel."""

    inputs = _read_normalized_inputs(config)
    paths = _normalised_input_paths(config.normalized_dir)
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    (
        rebuilt_membership,
        rebuilt_snapshots,
        rebuilt_union,
        membership_cache_records,
        membership_lineage_policy,
    ) = _load_and_rebuild_membership_lineage(config, inputs)
    membership = _validate_membership_daily(
        rebuilt_membership,
        inputs["trade_calendar"],
        expected_members=config.expected_members,
    )
    member_codes = set(membership["code"].astype(str))
    if set(rebuilt_union["code"].astype(str)) != member_codes:
        raise AssertionError("deterministic security union differs from membership union")

    security, security_cache_records, security_lineage_policy = (
        _load_and_rebuild_security_lineage(config, inputs, member_codes)
    )
    industry, industry_cache_records, industry_lineage_policy = (
        _load_and_rebuild_industry_lineage(
            config,
            inputs,
            rebuilt_snapshots,
            member_codes,
        )
    )
    del rebuilt_snapshots

    raw, adjusted, price_cache_records, price_history_boundaries = _load_price_views(
        config.cache_dir,
        inputs["price_request_index"],
        member_codes,
        membership,
        expected_start_date=str(
            pd.Timestamp(config.membership_source_start_date).date()
        ),
        expected_end_date=str(pd.Timestamp(config.membership_source_end_date).date()),
    )
    adjustment_events, adjustment_cache_records = _load_adjustment_events(
        config.cache_dir,
        inputs["adjustment_request_index"],
        member_codes,
        expected_start_date=str(
            pd.Timestamp(config.adjustment_event_start_date).date()
        ),
        expected_end_date=str(pd.Timestamp(config.membership_source_end_date).date()),
    )
    views = separate_price_views(raw, adjusted)
    del raw, adjusted
    adjustment_event_audit = audit_vendor_adjustment_events_asof(
        views.raw_execution,
        views.backward_adjusted_research,
        adjustment_events,
        event_catalog_symbols=member_codes,
    ).require_valid()
    raw_preclose_adjustment_diagnostic = {
        "schema_version": "1.0",
        "authoritative_for_build": False,
        "purpose": "independent_diagnostic_only",
        "valid": adjustment_event_audit.raw_preclose_recurrence_valid,
        "errors": list(adjustment_event_audit.raw_preclose_recurrence_errors),
    }
    index_request_key = _find_index_request_key(config.normalized_dir, config.cache_dir)
    benchmark_returns, index_cache_record = _load_benchmark_returns(
        config.cache_dir,
        index_request_key,
        expected_start_date=str(
            pd.Timestamp(config.membership_source_start_date).date()
        ),
        expected_end_date=str(pd.Timestamp(config.membership_source_end_date).date()),
    )
    provider_versions = {
        str(record.get("provider_version", ""))
        for record in [
            *price_cache_records,
            *adjustment_cache_records,
            *membership_cache_records,
            *security_cache_records,
            *industry_cache_records,
            index_cache_record,
        ]
    }
    if "" in provider_versions or len(provider_versions) != 1:
        raise CacheIntegrityError(
            "research snapshot cannot mix missing or different BaoStock provider versions"
        )

    # All rolling values are computed before any research-period clipping.
    full_features = derive_hs300_daily_features(
        views,
        security,
        benchmark_returns,
        adjustment_events=adjustment_events,
        event_catalog_symbols=member_codes,
        adv_window=config.adv_window,
        adv_min_periods=config.adv_min_periods,
        beta_window=config.beta_window,
        beta_min_periods=config.beta_min_periods,
    )
    del views, benchmark_returns, adjustment_events
    tape_dates_all = full_features.index.get_level_values("date")
    tape_end = (
        pd.Timestamp(config.research_end).normalize()
        if config.research_end
        else pd.Timestamp(tape_dates_all.max())
    )
    membership = membership.loc[membership["date"] <= tape_end].copy()
    if membership.empty:
        raise ValueError("no frozen provider membership exists through research_end")
    membership_index = pd.MultiIndex.from_frame(
        membership[["date", "code"]].rename(columns={"code": "symbol"}),
        names=["date", "symbol"],
    )
    terminal_exceptions = _classify_terminal_stale_membership_exceptions(
        membership,
        full_features.index,
        security,
    )
    available_member_index = membership_index.difference(terminal_exceptions.index)
    missing_market = available_member_index.difference(full_features.index)
    if len(missing_market):
        raise AssertionError(
            "terminal exception classifier left missing provider-member bars"
        )
    full_limits = derive_conservative_one_price_limits(
        full_features,
        minimum_listed_days=config.minimum_limit_rule_listed_days,
    )
    full_features["vendor_universe_member"] = full_features.index.isin(
        available_member_index
    )
    # Compatibility alias.  New consumers should use the explicit provider
    # membership name, then apply research_eligible for cross-sectional work.
    full_features["universe_member"] = full_features["vendor_universe_member"]
    full_features["research_eligible"] = _derive_research_eligibility(
        full_features,
        full_limits,
        available_member_index,
    )
    feature_tape = full_features.loc[:, _CONTINUOUS_FEATURE_TAPE_COLUMNS].join(
        full_limits.loc[:, _CONTINUOUS_LIMIT_COLUMNS]
    ).sort_index()
    tape_dates = feature_tape.index.get_level_values("date")
    feature_tape = feature_tape.loc[tape_dates <= tape_end].copy()

    member_panel = full_features.reindex(available_member_index).sort_index()
    del full_features
    membership_lookup = membership.rename(columns={"code": "symbol"}).set_index(["date", "symbol"])
    member_panel["membership_snapshot_date"] = pd.to_datetime(
        membership_lookup["vendor_update_date"].reindex(member_panel.index)
    )
    member_panel = member_panel.join(
        _industry_asof(
            member_panel.index,
            industry,
            maximum_age_days=config.maximum_industry_age_days,
        )
    )

    start = pd.Timestamp(config.research_start).normalize()
    end = (
        pd.Timestamp(config.research_end).normalize()
        if config.research_end
        else pd.Timestamp(member_panel.index.get_level_values("date").max())
    )
    date_values = member_panel.index.get_level_values("date")
    panel = member_panel.loc[(date_values >= start) & (date_values <= end)].copy()
    if panel.empty:
        raise ValueError("research period contains no cached HS300 rows")
    expected_panel_index = available_member_index[
        (available_member_index.get_level_values("date") >= start)
        & (available_member_index.get_level_values("date") <= end)
    ].sort_values()
    if not panel.index.equals(expected_panel_index):
        raise ValueError(
            "research panel differs from provider membership after exact terminal exceptions"
        )

    panel = panel.join(full_limits.reindex(panel.index))
    del full_limits, member_panel
    panel.attrs.update(_build_metadata(config, input_hashes))
    panel = attach_proxy_benchmark_weights(panel, mode="research_proxy")
    metadata = dict(panel.attrs)
    contract = DataContract(minimum_history_days=config.contract_minimum_history_days)
    report = validate_daily_panel(panel, contract=contract, metadata=metadata)
    if not report.valid:
        raise ValueError("research_proxy DataContract failed: " + "; ".join(report.errors))
    (
        manifest_path,
        manifest_sha_path,
        partitions,
        feature_partitions,
        manifest,
    ) = _write_snapshot(
        panel,
        feature_tape,
        terminal_exceptions,
        report,
        config,
        input_hashes=input_hashes,
        membership_cache_records=membership_cache_records,
        membership_lineage_policy=membership_lineage_policy,
        security_cache_records=security_cache_records,
        security_lineage_policy=security_lineage_policy,
        industry_cache_records=industry_cache_records,
        industry_lineage_policy=industry_lineage_policy,
        price_cache_records=price_cache_records,
        price_history_boundaries=price_history_boundaries,
        adjustment_cache_records=adjustment_cache_records,
        adjustment_event_audit=asdict(adjustment_event_audit),
        raw_preclose_adjustment_diagnostic=raw_preclose_adjustment_diagnostic,
        index_cache_record=index_cache_record,
    )
    return HS300BuildResult(
        panel=panel,
        validation=report,
        manifest_path=manifest_path,
        manifest_sha256_path=manifest_sha_path,
        partition_paths=partitions,
        manifest=manifest,
        feature_tape=feature_tape,
        feature_tape_partition_paths=feature_partitions,
        terminal_stale_membership_exceptions=terminal_exceptions,
        terminal_stale_membership_exception_path=Path(
            manifest["terminal_stale_membership_exceptions"]["file"]["path"]
        ),
    )


__all__ = [
    "HS300BuildConfig",
    "HS300BuildResult",
    "build_hs300_research_panel",
    "derive_conservative_one_price_limits",
]
