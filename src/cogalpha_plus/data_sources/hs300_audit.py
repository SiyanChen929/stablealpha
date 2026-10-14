"""Offline, fail-closed audit of a frozen BaoStock HS300 research panel.

The builder validates inputs while it constructs a snapshot.  This module is a
separate post-build control: it starts from ``panel/manifest.json``, verifies
the frozen bytes again, reconstructs the panel, and reports structural,
temporal, coverage and economic-plausibility findings without contacting a
data provider.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data_contract import ADJUSTED_PRICE_COLUMNS, RAW_PRICE_COLUMNS, RESEARCH_PRICE_MAP
from ..runner import panel_snapshot_hash
from .baostock import (
    ADJUST_FACTOR_FIELDS,
    EQUITY_DAILY_FIELDS,
    INDEX_DAILY_FIELDS,
    make_request_key,
)
from .hs300 import (
    audit_vendor_adjustment_events_asof,
    derive_rolling_beta,
    normalize_baostock_daily_fields,
    trim_leading_unadjusted_vendor_history,
)
from .hs300_build import (
    HS300BuildConfig,
    _classify_terminal_stale_membership_exceptions,
    _load_and_rebuild_industry_lineage,
    _load_and_rebuild_membership_lineage,
    _load_and_rebuild_security_lineage,
    derive_conservative_one_price_limits,
)


class AuditSeverity(str, Enum):
    FAIL = "fail"
    WARN = "warn"


@dataclass(frozen=True)
class AuditFinding:
    severity: AuditSeverity
    code: str
    message: str
    count: int | None = None
    rate: float | None = None
    samples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


@dataclass(frozen=True)
class HS300AuditConfig:
    panel_dir: Path | str
    expected_members: int = 300
    normalized_dir: Path | str | None = None
    cache_dir: Path | str | None = None
    adjusted_price_tolerance: float = 1e-8
    adjusted_return_warning_threshold: float = 0.30
    adjusted_return_fail_threshold: float = 1.00
    suspension_daily_warning_rate: float = 0.10
    minimum_proxy_mcap_coverage: float = 1.00
    maximum_samples: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_dir", Path(self.panel_dir))
        if self.normalized_dir is not None:
            object.__setattr__(self, "normalized_dir", Path(self.normalized_dir))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.expected_members <= 0:
            raise ValueError("expected_members must be positive")
        if not np.isfinite(self.adjusted_price_tolerance) or self.adjusted_price_tolerance <= 0:
            raise ValueError("adjusted_price_tolerance must be positive")
        if self.adjusted_price_tolerance > 1e-8:
            raise ValueError("adjusted_price_tolerance cannot exceed the fail-closed ceiling 1e-8")
        if not (
            np.isfinite(self.adjusted_return_warning_threshold)
            and np.isfinite(self.adjusted_return_fail_threshold)
            and
            0 < self.adjusted_return_warning_threshold
            < self.adjusted_return_fail_threshold
        ):
            raise ValueError("adjusted return thresholds must be positive and ordered")
        if self.adjusted_return_warning_threshold > 0.30:
            raise ValueError("adjusted return warning threshold may only tighten the frozen 0.30 policy")
        if self.adjusted_return_fail_threshold > 1.00:
            raise ValueError("adjusted return fail threshold may only tighten the frozen 1.00 policy")
        if not np.isfinite(self.suspension_daily_warning_rate) or not 0 <= self.suspension_daily_warning_rate <= 1:
            raise ValueError("suspension_daily_warning_rate must be in [0, 1]")
        if self.suspension_daily_warning_rate > 0.10:
            raise ValueError("suspension warning rate may only tighten the frozen 0.10 policy")
        if not np.isfinite(self.minimum_proxy_mcap_coverage) or not 0 <= self.minimum_proxy_mcap_coverage <= 1:
            raise ValueError("minimum_proxy_mcap_coverage must be in [0, 1]")
        if self.minimum_proxy_mcap_coverage < 1.0:
            raise ValueError("proxy mcap coverage may not weaken the frozen 100% policy")
        if self.maximum_samples < 0:
            raise ValueError("maximum_samples cannot be negative")


@dataclass(frozen=True)
class HS300AuditReport:
    valid: bool
    status: str
    panel_dir: str
    manifest_path: str
    rows: int
    symbols: int
    trading_days: int
    first_date: str
    last_date: str
    expected_snapshot_hash: str
    actual_snapshot_hash: str
    research_valid: bool
    portfolio_grade: bool
    capital_deployment_allowed: bool
    findings: tuple[AuditFinding, ...]
    metrics: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "status": self.status,
            "panel_dir": self.panel_dir,
            "manifest_path": self.manifest_path,
            "rows": self.rows,
            "symbols": self.symbols,
            "trading_days": self.trading_days,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "expected_snapshot_hash": self.expected_snapshot_hash,
            "actual_snapshot_hash": self.actual_snapshot_hash,
            "research_valid": self.research_valid,
            "portfolio_grade": self.portfolio_grade,
            "capital_deployment_allowed": self.capital_deployment_allowed,
            "finding_counts": {
                severity.value: sum(
                    item.severity is severity for item in self.findings
                )
                for severity in AuditSeverity
            },
            "findings": [item.to_dict() for item in self.findings],
            "metrics": _json_safe(self.metrics),
        }


_NORMALIZED_FILENAMES = {
    "trade_calendar": "trade_calendar.parquet",
    "membership_snapshots": "membership_snapshots.parquet",
    "membership_daily": "membership_daily.parquet",
    "security_union": "security_union.parquet",
    "membership_request_index": "membership_request_index.parquet",
    "security_master": "security_master.parquet",
    "security_master_request_index": "security_master_request_index.parquet",
    "industry_snapshots": "industry_snapshots.parquet",
    "industry_request_index": "industry_request_index.parquet",
    "price_request_index": "price_request_index.parquet",
    "adjustment_request_index": "adjustment_request_index.parquet",
}

_AUDIT_REQUIRED_COLUMNS = tuple(
    dict.fromkeys(
        (
            *RAW_PRICE_COLUMNS,
            *ADJUSTED_PRICE_COLUMNS,
            "volume",
            "amount",
            "adj_factor",
            "tradable",
            "limit_up",
            "limit_down",
            "limit_proxy_known",
            "research_close_limit_up_proxy",
            "research_close_limit_down_proxy",
            "research_close_limit_known_proxy",
            "is_st",
            "listed_days",
            "industry",
            "log_mcap",
            "implied_float_mcap",
            "beta",
            "universe_member",
            "vendor_universe_member",
            "research_eligible",
            "benchmark_weight",
            "adv20_dollars",
            "membership_snapshot_date",
            "industry_snapshot_date",
            "research_watermark",
        )
    )
)

_AUDIT_RETAIN_COLUMNS = set(_AUDIT_REQUIRED_COLUMNS).union(
    {
        "preclose",
        "turn",
        "tradestatus",
        "isST",
        "implied_float_shares",
        "implied_float_mcap_source",
        "benchmark_weight_source",
        "benchmark_weight_quality",
        "limit_ratio_proxy",
        "limit_proxy_method",
        "vwap_is_no_trade_placeholder",
    }
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _add(
    findings: list[AuditFinding],
    severity: AuditSeverity,
    code: str,
    message: str,
    *,
    count: int | None = None,
    rate: float | None = None,
    samples: Sequence[object] = (),
    maximum_samples: int = 10,
) -> None:
    findings.append(
        AuditFinding(
            severity=severity,
            code=code,
            message=message,
            count=count,
            rate=rate,
            samples=tuple(str(item) for item in samples[:maximum_samples]),
        )
    )


def _resolve_manifest_directory(
    override: Path | None,
    manifest: Mapping[str, object],
    key: str,
    panel_dir: Path,
) -> Path | None:
    if override is not None:
        return override.resolve()
    raw_config = manifest.get("config")
    if not isinstance(raw_config, Mapping) or not str(raw_config.get(key, "")).strip():
        return None
    declared = Path(str(raw_config[key]))
    if declared.is_absolute():
        return declared.resolve()
    # Direct API use may have persisted relative paths.  Resolve them relative
    # to both the working snapshot and cwd, preferring an existing directory.
    candidates = [
        (panel_dir.parent / declared).resolve(),
        declared.resolve(),
        (panel_dir.parent / declared.name).resolve(),
    ]
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def _resolve_partition_path(panel_dir: Path, declared: object) -> Path | None:
    raw = Path(str(declared or ""))
    if not str(raw):
        return None
    root = panel_dir.resolve()
    candidates = [raw.resolve()] if raw.is_absolute() else [
        (root / raw).resolve(),
        raw.resolve(),
    ]
    if "snapshots" in raw.parts:
        suffix = Path(*raw.parts[raw.parts.index("snapshots") :])
        candidates.append((root / suffix).resolve())
    return next(
        (candidate for candidate in candidates if candidate.is_relative_to(root) and candidate.is_file()),
        next((candidate for candidate in candidates if candidate.is_relative_to(root)), None),
    )


def _resolve_cache_object(
    cache_dir: Path,
    declared: object,
) -> Path | None:
    raw = Path(str(declared or ""))
    if not str(raw):
        return None
    root = cache_dir.resolve()
    if raw.is_absolute() and raw.resolve().is_relative_to(root):
        return raw.resolve()
    if not raw.is_absolute():
        direct = (root / raw).resolve()
        if direct.is_relative_to(root):
            return direct
    # Permit an explicitly overridden, relocated cache while retaining the
    # content-addressed suffix beginning at ``objects/``.
    parts = raw.parts
    if "objects" in parts:
        suffix = Path(*parts[parts.index("objects") :])
        relocated = (root / suffix).resolve()
        if relocated.is_relative_to(root):
            return relocated
    return None


def _blank_mask(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype("string").str.strip().eq("")


def _sample_index(index: pd.Index, maximum: int) -> list[str]:
    output: list[str] = []
    for value in index[:maximum]:
        if isinstance(value, tuple) and len(value) == 2:
            output.append(f"{pd.Timestamp(value[0]).date()}|{value[1]}")
        else:
            output.append(str(value))
    return output


def _coverage(panel: pd.DataFrame, column: str) -> dict[str, object]:
    total = len(panel)
    if column not in panel:
        return {"present": False, "non_missing": 0, "missing": total, "rate": 0.0}
    values = panel[column]
    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.notna() & np.isfinite(numeric)
    elif pd.api.types.is_datetime64_any_dtype(values.dtype):
        valid = pd.to_datetime(values, errors="coerce").notna()
    else:
        valid = ~_blank_mask(values)
    non_missing = int(valid.sum())
    return {
        "present": True,
        "non_missing": non_missing,
        "missing": total - non_missing,
        "rate": non_missing / total if total else 0.0,
    }


def _manifest_and_partitions(
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> tuple[dict[str, object] | None, pd.DataFrame | None]:
    panel_dir = config.panel_dir.resolve()
    manifest_path = panel_dir / "manifest.json"
    sha_path = panel_dir / "manifest.sha256"
    if not manifest_path.is_file():
        _add(
            findings,
            AuditSeverity.FAIL,
            "manifest_missing",
            f"frozen manifest is missing: {manifest_path}",
        )
        return None, None
    try:
        manifest = _read_json(manifest_path)
    except Exception as exc:
        _add(
            findings,
            AuditSeverity.FAIL,
            "manifest_unreadable",
            f"cannot parse frozen manifest: {type(exc).__name__}: {exc}",
        )
        return None, None

    actual_manifest_hash = _sha256_file(manifest_path)
    sha_verified = False
    if not sha_path.is_file():
        _add(findings, AuditSeverity.FAIL, "manifest_sha_missing", "manifest.sha256 is missing")
    else:
        match = re.fullmatch(
            r"([0-9a-f]{64})\s+\*?manifest\.json\s*",
            sha_path.read_text(encoding="utf-8"),
        )
        if match is None:
            _add(
                findings,
                AuditSeverity.FAIL,
                "manifest_sha_malformed",
                "manifest.sha256 is not a canonical SHA-256 record",
            )
        elif match.group(1) != actual_manifest_hash:
            _add(
                findings,
                AuditSeverity.FAIL,
                "manifest_sha_mismatch",
                "manifest bytes do not match manifest.sha256",
            )
        else:
            sha_verified = True
    metrics["integrity"] = {
        "manifest_sha256": actual_manifest_hash,
        "manifest_sha_verified": sha_verified,
    }
    if manifest.get("complete") is not True:
        _add(findings, AuditSeverity.FAIL, "manifest_incomplete", "manifest is not marked complete")

    raw_partitions = manifest.get("partitions")
    if not isinstance(raw_partitions, list) or not raw_partitions:
        _add(
            findings,
            AuditSeverity.FAIL,
            "partitions_missing",
            "manifest has no frozen partition records",
        )
        return manifest, None

    frames: list[pd.DataFrame] = []
    logical_hasher = hashlib.sha256()
    paths_seen: set[Path] = set()
    years_seen: set[int] = set()
    verified = 0
    total_bytes = 0
    for number, record in enumerate(raw_partitions):
        if not isinstance(record, Mapping):
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_record_invalid",
                f"partition record {number} is not an object",
            )
            continue
        path = _resolve_partition_path(panel_dir, record.get("path"))
        if path is None:
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_path_unsafe",
                f"partition {number} is outside the frozen panel directory",
            )
            continue
        if path in paths_seen:
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_path_duplicate",
                f"partition path is declared more than once: {path}",
            )
            continue
        paths_seen.add(path)
        try:
            year = int(record.get("year"))
        except (TypeError, ValueError):
            year = -1
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_year_invalid",
                f"partition {number} has no valid year",
            )
        if year in years_seen:
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_year_duplicate",
                f"year {year} is split or duplicated without an explicit shard contract",
            )
        years_seen.add(year)
        if not path.is_file():
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_file_missing",
                f"partition file is missing: {path}",
            )
            continue
        total_bytes += path.stat().st_size
        actual_hash = _sha256_file(path)
        expected_hash = str(record.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or actual_hash != expected_hash:
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_sha_mismatch",
                f"partition hash mismatch: {path.name} ({year})",
            )
        else:
            verified += 1
        if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_byte_count_mismatch",
                f"partition byte count mismatch: {path.name} ({year})",
            )
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_unreadable",
                f"cannot read {path}: {type(exc).__name__}: {exc}",
            )
            continue
        if len(frame) != int(record.get("rows", -1)):
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_row_count_mismatch",
                f"partition row count disagrees with manifest for {year}",
            )
        if not isinstance(frame.index, pd.MultiIndex) or frame.index.names != ["date", "symbol"]:
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_index_invalid",
                f"partition {year} lacks MultiIndex(date, symbol)",
            )
            continue
        logical_hashes = pd.util.hash_pandas_object(
            frame, index=True, categorize=True
        ).to_numpy(dtype=np.uint64)
        logical_hasher.update(logical_hashes.tobytes())
        date_values = pd.to_datetime(frame.index.get_level_values("date"), errors="coerce")
        if date_values.isna().any() or (date_values.year != year).any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "partition_year_content_mismatch",
                f"partition {year} contains invalid or out-of-year dates",
            )
        retained = [name for name in frame.columns if name in _AUDIT_RETAIN_COLUMNS]
        frames.append(frame[retained])

    metrics["integrity"].update(
        {
            "partitions_declared": len(raw_partitions),
            "partitions_loaded": len(frames),
            "partitions_sha_verified": verified,
            "partition_bytes": total_bytes,
        }
    )
    if not frames:
        return manifest, None
    panel = pd.concat(frames, copy=False)
    del frames
    if not panel.index.is_monotonic_increasing:
        _add(
            findings,
            AuditSeverity.FAIL,
            "partition_order_not_monotonic",
            "manifest partition order does not reconstruct a sorted panel",
        )
        panel = panel.sort_index()
    metrics["integrity"]["actual_snapshot_hash"] = logical_hasher.hexdigest()
    return manifest, panel


_FEATURE_TAPE_REQUIRED_COLUMNS = (
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
    "limit_up",
    "limit_down",
    "limit_proxy_known",
    "limit_ratio_proxy",
    "research_close_limit_up_proxy",
    "research_close_limit_down_proxy",
    "research_close_limit_known_proxy",
    "is_st",
    "listed_days",
    "log_mcap",
    "beta",
    "adv20_dollars",
    "universe_member",
    "vendor_universe_member",
    "research_eligible",
)

_FEATURE_TAPE_BOOLEAN_COLUMNS = (
    "tradable",
    "limit_up",
    "limit_down",
    "limit_proxy_known",
    "research_close_limit_up_proxy",
    "research_close_limit_down_proxy",
    "research_close_limit_known_proxy",
    "is_st",
    "universe_member",
    "vendor_universe_member",
    "research_eligible",
)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_TERMINAL_STALE_REASON = "terminal_stale_membership_after_verified_out_date"
_TERMINAL_EXCEPTION_COLUMNS = (
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


def _validated_terminal_exception_sidecar(
    manifest: Mapping[str, object],
    panel_dir: Path,
) -> pd.DataFrame:
    """Load the exact, immutable member-without-bar exception contract."""

    section = manifest.get("terminal_stale_membership_exceptions")
    if not isinstance(section, Mapping):
        raise ValueError("terminal-stale membership exception contract is missing")
    if (
        section.get("schema_version") != "1.0"
        or section.get("complete") is not True
        or section.get("policy") != _TERMINAL_STALE_REASON
        or section.get("price_fill_policy") != "none"
        or section.get("terminal_return_policy") != "unresolved_no_imputation"
    ):
        raise ValueError("terminal-stale membership exception policy is invalid")
    record = section.get("file")
    if not isinstance(record, Mapping):
        raise ValueError("terminal-stale membership exception file is missing")
    path = _resolve_partition_path(panel_dir.resolve(), record.get("path"))
    storage_snapshot_id = str(manifest.get("storage_snapshot_id", ""))
    expected_path = (
        panel_dir.resolve()
        / "snapshots"
        / storage_snapshot_id
        / "terminal_stale_membership_exceptions.parquet"
    ).resolve()
    if path is None or path != expected_path or not path.is_file():
        raise ValueError("terminal-stale sidecar is not bound to this snapshot")
    if _sha256_file(path) != str(record.get("sha256", "")):
        raise ValueError("terminal-stale sidecar file hash mismatch")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise ValueError("terminal-stale sidecar byte count mismatch")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"terminal-stale sidecar is unreadable: {exc}") from exc
    if (
        not isinstance(frame.index, pd.MultiIndex)
        or frame.index.names != ["date", "symbol"]
        or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing
        or tuple(frame.columns) != _TERMINAL_EXCEPTION_COLUMNS
    ):
        raise ValueError("terminal-stale sidecar index/schema is invalid")
    dates = pd.to_datetime(
        frame.index.get_level_values("date"), errors="coerce"
    ).normalize()
    symbols = pd.Index(frame.index.get_level_values("symbol").astype(str))
    if dates.isna().any() or (symbols.str.strip() == "").any():
        raise ValueError("terminal-stale sidecar has invalid keys")
    if len(frame):
        true_member = (
            frame["vendor_universe_member"].eq(True)
            & frame["universe_member"].eq(True)
        )
        ineligible = frame["research_eligible"].eq(False)
        out_date = pd.to_datetime(
            frame["stock_basic_out_date"], errors="coerce"
        ).dt.normalize()
        last_bar = pd.to_datetime(
            frame["last_raw_bar_date"], errors="coerce"
        ).dt.normalize()
        snapshot_date = pd.to_datetime(
            frame["membership_snapshot_date"], errors="coerce"
        ).dt.normalize()
        terminal_gap_days = (out_date - last_bar).dt.days
        exception_dates = pd.Series(dates, index=frame.index)
        valid = (
            true_member
            & ineligible
            & frame["reason"].eq(_TERMINAL_STALE_REASON)
            & out_date.notna()
            & last_bar.le(out_date)
            & terminal_gap_days.between(0, 7)
            & exception_dates.ge(out_date.to_numpy())
            & exception_dates.gt(last_bar.to_numpy())
            & snapshot_date.notna()
            & pd.Series(snapshot_date.to_numpy(), index=frame.index).le(
                pd.Series(dates, index=frame.index)
            )
            & frame["security_status"].astype(str).eq("0")
            & frame["price_fill_policy"].eq("none")
            & frame["terminal_return_policy"].eq("unresolved_no_imputation")
        )
        if not valid.all():
            raise ValueError("terminal-stale sidecar contains an unproved exception")
    semantic_hash = panel_snapshot_hash(frame)
    if semantic_hash != str(section.get("snapshot_hash", "")):
        raise ValueError("terminal-stale sidecar semantic hash mismatch")
    if int(section.get("rows", -1)) != len(frame) or int(
        record.get("rows", -1)
    ) != len(frame):
        raise ValueError("terminal-stale sidecar row count mismatch")
    if int(section.get("symbols", -1)) != int(symbols.nunique()) or int(
        section.get("trading_days", -1)
    ) != int(pd.DatetimeIndex(dates).nunique()):
        raise ValueError("terminal-stale sidecar summary mismatch")
    return frame


def _audit_research_eligibility_contract(
    tape: pd.DataFrame,
    manifest: Mapping[str, object],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    """Recompute the t-close mask without consulting any future endpoint."""

    section = manifest.get("research_eligibility")
    expected_policy = {
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
        "limit_source_columns": [
            "research_close_limit_up_proxy",
            "research_close_limit_down_proxy",
            "research_close_limit_known_proxy",
        ],
        "unknown_research_close_limit_policy": "ineligible_fail_closed",
        "terminal_stale_membership_policy": "ineligible_sidecar_no_bar_fill",
    }
    if not isinstance(section, Mapping):
        _add(
            findings,
            AuditSeverity.FAIL,
            "research_eligibility_manifest_missing",
            "manifest lacks the frozen current-time eligibility policy",
        )
        return
    policy_mismatch = [
        key for key, value in expected_policy.items() if section.get(key) != value
    ]
    if policy_mismatch:
        _add(
            findings,
            AuditSeverity.FAIL,
            "research_eligibility_policy_mismatch",
            "research eligibility policy differs from the frozen t-close contract",
            count=len(policy_mismatch),
            samples=policy_mismatch,
            maximum_samples=config.maximum_samples,
        )
    required = {
        "vendor_universe_member",
        "universe_member",
        "research_eligible",
        "tradable",
        "is_st",
        "listed_days",
        "log_mcap",
        "beta",
        "adv20_dollars",
        "research_close_limit_up_proxy",
        "research_close_limit_down_proxy",
        "research_close_limit_known_proxy",
    }
    missing = sorted(required.difference(tape.columns))
    if missing:
        _add(
            findings,
            AuditSeverity.FAIL,
            "research_eligibility_inputs_missing",
            "feature tape cannot independently reproduce research eligibility",
            count=len(missing),
            samples=missing,
            maximum_samples=config.maximum_samples,
        )
        return
    vendor = tape["vendor_universe_member"].eq(True)
    legacy = tape["universe_member"].eq(True)
    actual = tape["research_eligible"].eq(True)
    listed = pd.to_numeric(tape["listed_days"], errors="coerce")
    log_mcap = pd.to_numeric(tape["log_mcap"], errors="coerce")
    beta = pd.to_numeric(tape["beta"], errors="coerce")
    adv20_dollars = pd.to_numeric(tape["adv20_dollars"], errors="coerce")
    risk_and_capacity_available = pd.Series(
        np.isfinite(log_mcap.to_numpy(dtype=float))
        & np.isfinite(beta.to_numpy(dtype=float))
        & np.isfinite(adv20_dollars.to_numpy(dtype=float))
        & adv20_dollars.gt(0).to_numpy(dtype=bool),
        index=tape.index,
        dtype=bool,
    )
    expected = (
        vendor
        & tape["tradable"].eq(True)
        & ~tape["is_st"].eq(True)
        & listed.ge(365).fillna(False)
        & risk_and_capacity_available
        & tape["research_close_limit_known_proxy"].eq(True)
        & ~tape["research_close_limit_up_proxy"].eq(True)
        & ~tape["research_close_limit_down_proxy"].eq(True)
    )
    mismatch = actual.ne(expected) | vendor.ne(legacy)
    if mismatch.any():
        _add(
            findings,
            AuditSeverity.FAIL,
            "research_eligibility_value_mismatch",
            "frozen eligibility is not the exact current-row-only policy",
            count=int(mismatch.sum()),
            samples=_sample_index(tape.index[mismatch], config.maximum_samples),
            maximum_samples=config.maximum_samples,
        )
    semantic_hash = panel_snapshot_hash(
        tape[
            [
                "vendor_universe_member",
                "universe_member",
                "research_eligible",
            ]
        ]
    )
    member_daily = vendor.groupby(level="date").sum()
    eligible_daily = actual.loc[vendor].groupby(level="date").sum()
    summary = {
        "vendor_member_rows_with_bars": int(vendor.sum()),
        "eligible_rows": int(actual.sum()),
        "eligible_daily_min": int(eligible_daily.min()) if len(eligible_daily) else 0,
        "eligible_daily_max": int(eligible_daily.max()) if len(eligible_daily) else 0,
        "snapshot_hash": semantic_hash,
        "provider_member_daily_min": int(member_daily.min()) if len(member_daily) else 0,
        "provider_member_daily_max": int(member_daily.max()) if len(member_daily) else 0,
    }
    metrics["research_eligibility"] = summary
    for key in (
        "vendor_member_rows_with_bars",
        "eligible_rows",
        "eligible_daily_min",
        "eligible_daily_max",
        "snapshot_hash",
    ):
        if str(section.get(key)) != str(summary[key]):
            _add(
                findings,
                AuditSeverity.FAIL,
                "research_eligibility_manifest_summary_mismatch",
                f"research eligibility manifest disagrees on {key}",
            )


def _audit_separate_limit_proxy_contracts(
    tape: pd.DataFrame,
    manifest: Mapping[str, object],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    contracts = (
        (
            "execution_one_price_limit_proxy",
            ("limit_up", "limit_down", "limit_proxy_known"),
            "execution_lock_diagnostic_and_legacy_limit_fields",
            "one_price_ohlc_and_rounded_preclose_board_rule",
            "not_asserted_as_locked",
        ),
        (
            "research_close_limit_proxy",
            (
                "research_close_limit_up_proxy",
                "research_close_limit_down_proxy",
                "research_close_limit_known_proxy",
            ),
            "research_cross_sectional_eligibility_exclusion",
            "raw_close_and_rounded_preclose_board_rule",
            "ineligible_fail_closed",
        ),
    )
    contract_metrics: dict[str, object] = {}
    for name, columns, purpose, method, unknown_policy in contracts:
        section = manifest.get(name)
        if not isinstance(section, Mapping):
            _add(
                findings,
                AuditSeverity.FAIL,
                f"{name}_manifest_missing",
                f"manifest lacks independent {name} provenance",
            )
            continue
        expected_policy = {
            "schema_version": "1.0",
            "complete": True,
            "purpose": purpose,
            "decision_time": "t_close",
            "columns": list(columns),
            "method": method,
            "unknown_policy": unknown_policy,
            "quality": "research_proxy",
        }
        mismatch_policy = [
            key for key, value in expected_policy.items() if section.get(key) != value
        ]
        if mismatch_policy:
            _add(
                findings,
                AuditSeverity.FAIL,
                f"{name}_policy_mismatch",
                f"{name} policy differs from the frozen proxy rule",
                count=len(mismatch_policy),
                samples=mismatch_policy,
                maximum_samples=config.maximum_samples,
            )
        if not set(columns).issubset(tape.columns):
            continue
        frame = tape.loc[:, list(columns)]
        summary = {
            "up_rows": int(frame.iloc[:, 0].astype(bool).sum()),
            "down_rows": int(frame.iloc[:, 1].astype(bool).sum()),
            "known_rows": int(frame.iloc[:, 2].astype(bool).sum()),
            "snapshot_hash": panel_snapshot_hash(frame),
        }
        contract_metrics[name] = summary
        for key, value in summary.items():
            if str(section.get(key)) != str(value):
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    f"{name}_manifest_summary_mismatch",
                    f"{name} manifest disagrees on {key}",
                )
    metrics["separate_limit_proxies"] = contract_metrics
    required = {
        "limit_up",
        "limit_down",
        "limit_proxy_known",
        "research_close_limit_up_proxy",
        "research_close_limit_down_proxy",
        "research_close_limit_known_proxy",
    }
    if required.issubset(tape.columns):
        invalid_relation = (
            (tape["limit_up"].eq(True) & ~tape["research_close_limit_up_proxy"].eq(True))
            | (
                tape["limit_down"].eq(True)
                & ~tape["research_close_limit_down_proxy"].eq(True)
            )
            | tape["limit_proxy_known"].eq(True).ne(
                tape["research_close_limit_known_proxy"].eq(True)
            )
        )
        if invalid_relation.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "separate_limit_proxy_relation_invalid",
                "one-price locks must be a subset of same-direction close limits with common knownness",
                count=int(invalid_relation.sum()),
                samples=_sample_index(
                    tape.index[invalid_relation], config.maximum_samples
                ),
                maximum_samples=config.maximum_samples,
            )


def _series_value_mismatch(actual: pd.Series, expected: pd.Series) -> np.ndarray:
    """Compare one frozen common field without materialising a wide frame."""

    left = actual.reset_index(drop=True)
    right = expected.reset_index(drop=True)
    if len(left) != len(right):
        return np.ones(max(len(left), len(right)), dtype=bool)
    if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(
        right.dtype
    ):
        return ~np.isclose(
            pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    if pd.api.types.is_datetime64_any_dtype(left.dtype) or pd.api.types.is_datetime64_any_dtype(
        right.dtype
    ):
        left_date = pd.to_datetime(left, errors="coerce", format="mixed")
        right_date = pd.to_datetime(right, errors="coerce", format="mixed")
        return left_date.to_numpy(dtype="datetime64[ns]") != right_date.to_numpy(
            dtype="datetime64[ns]"
        )
    both_missing = left.isna().to_numpy() & right.isna().to_numpy()
    equal = left.eq(right).fillna(False).to_numpy(dtype=bool)
    return ~(equal | both_missing)


def _load_feature_tape_normalized_sources(
    manifest: Mapping[str, object],
    section: Mapping[str, object],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    tape_metrics: dict[str, object],
) -> dict[str, pd.DataFrame]:
    """Load only the three frozen catalogs that define continuous-tape semantics."""

    normalized_dir = _resolve_manifest_directory(
        config.normalized_dir,
        manifest,
        "normalized_dir",
        config.panel_dir.resolve(),
    )
    declared_hashes = manifest.get("normalized_input_sha256")
    source_contract = section.get("source_contract")
    if (
        normalized_dir is None
        or not isinstance(declared_hashes, Mapping)
        or not isinstance(source_contract, Mapping)
    ):
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_normalized_provenance_unavailable",
            "feature-tape manifest cannot locate its frozen membership, calendar and security catalogs",
        )
        return {}

    tape_metrics["normalized_dir"] = str(normalized_dir)
    frames: dict[str, pd.DataFrame] = {}
    verified = 0
    for key in ("membership_daily", "trade_calendar", "security_master"):
        expected = str(declared_hashes.get(key, ""))
        contract_hash = str(source_contract.get(f"{key}_sha256", ""))
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or contract_hash != expected
        ):
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_source_contract_hash_mismatch",
                f"feature-tape source contract is not bound to normalized {key}",
            )
            continue
        path = normalized_dir / _NORMALIZED_FILENAMES[key]
        if not path.is_file():
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_normalized_source_missing",
                f"feature-tape normalized source is missing: {path}",
            )
            continue
        if _sha256_file(path) != expected:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_normalized_source_sha_mismatch",
                f"feature-tape normalized source hash mismatch: {path.name}",
            )
            continue
        try:
            frames[key] = pd.read_parquet(path)
            verified += 1
        except Exception as exc:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_normalized_source_unreadable",
                f"cannot read {path.name}: {type(exc).__name__}: {exc}",
            )
    tape_metrics["normalized_sources_verified"] = verified
    return frames


def _audit_feature_tape_source_lineage(
    tape: pd.DataFrame,
    manifest: Mapping[str, object],
    normalized_frames: Mapping[str, pd.DataFrame],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    tape_metrics: dict[str, object],
) -> None:
    """Stream cache pairs by security and reproduce the tape's research core."""

    raw_records = manifest.get("price_cache_objects")
    cache_dir = _resolve_manifest_directory(
        config.cache_dir,
        manifest,
        "cache_dir",
        config.panel_dir.resolve(),
    )
    if not isinstance(raw_records, list) or cache_dir is None:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_cache_provenance_unavailable",
            "feature-tape cannot resolve manifest-declared raw/backward-adjusted cache pairs",
        )
        return

    build_config = manifest.get("config")
    declared_end: pd.Timestamp | None = None
    if isinstance(build_config, Mapping) and str(build_config.get("research_end") or "").strip():
        declared_end = pd.Timestamp(build_config["research_end"]).normalize()
    minimum_days = (
        int(build_config.get("minimum_limit_rule_listed_days", 30))
        if isinstance(build_config, Mapping)
        else 30
    )

    records_by_code: dict[str, dict[str, Mapping[str, object]]] = {}
    duplicate_records: list[str] = []
    for record in raw_records:
        if not isinstance(record, Mapping):
            duplicate_records.append("non-object")
            continue
        code = str(record.get("code", ""))
        flag = str(record.get("adjustflag", ""))
        if flag in records_by_code.setdefault(code, {}):
            duplicate_records.append(f"{code}/{flag}")
        records_by_code[code][flag] = record
    incomplete = [
        code for code, pair in sorted(records_by_code.items()) if set(pair) != {"1", "3"}
    ]
    tape_symbols = set(tape.index.get_level_values("symbol").astype(str))
    record_symbols = set(records_by_code)
    # A historical-union security can legitimately IPO after a custom
    # research_end and therefore have no row at or before the frozen tape end.
    # Inspect only record-only names; all tape names are streamed in full below.
    inactive_after_end: set[str] = set()
    missing: list[str] = []
    for code in sorted(record_symbols - tape_symbols):
        pair = records_by_code.get(code, {})
        active = set(pair) != {"1", "3"}
        if not active:
            for flag in ("3", "1"):
                path = _resolve_cache_object(cache_dir, pair[flag].get("object"))
                if path is None or not path.is_file():
                    active = True
                    break
                try:
                    source_dates = pd.to_datetime(
                        pd.read_parquet(path, columns=["date"])["date"],
                        errors="coerce",
                    ).dropna()
                except Exception:
                    active = True
                    break
                if declared_end is None:
                    active = len(source_dates) > 0
                else:
                    active = bool((source_dates.dt.normalize() <= declared_end).any())
                if active:
                    break
        if active:
            missing.append(code)
        else:
            inactive_after_end.add(code)
    extra = sorted(tape_symbols - record_symbols)
    tape_metrics["source_symbols_inactive_through_tape_end"] = len(
        inactive_after_end
    )
    if duplicate_records or incomplete or missing or extra:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_source_symbol_or_pair_mismatch",
            "feature-tape symbols and active manifest raw/adjustflag=1 cache pairs are not one-to-one",
            count=len(duplicate_records) + len(incomplete) + len(missing) + len(extra),
            samples=[
                *[f"duplicate:{item}" for item in duplicate_records],
                *[f"incomplete:{item}" for item in incomplete],
                *[f"missing-tape:{item}" for item in missing],
                *[f"extra-tape:{item}" for item in extra],
            ],
            maximum_samples=config.maximum_samples,
        )

    security = normalized_frames.get("security_master")
    ipo_map = pd.Series(dtype="datetime64[ns]")
    if security is None or not {"code", "ipoDate"}.issubset(security.columns):
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_security_master_schema_invalid",
            "security master cannot reproduce feature-tape listed_days",
        )
    else:
        basic = security[["code", "ipoDate"]].copy()
        basic["code"] = basic["code"].astype(str)
        if basic["code"].duplicated().any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_security_master_duplicate_code",
                "security master contains duplicate securities",
            )
        else:
            ipo_map = pd.to_datetime(
                basic.set_index("code")["ipoDate"], errors="coerce"
            ).dt.normalize()

    index_mismatches: dict[str, int] = {}
    value_mismatches: dict[str, int] = {}
    samples: list[str] = []
    audited_symbols = 0
    source_union_dates: set[pd.Timestamp] = set()
    observed_price_boundaries: list[dict[str, object]] = []
    boundary_membership = normalized_frames.get("membership_daily")

    def index_record(name: str, count: int, sample: str) -> None:
        if count <= 0:
            return
        index_mismatches[name] = index_mismatches.get(name, 0) + count
        if len(samples) < config.maximum_samples:
            samples.append(sample)

    def value_record(
        name: str, mask: np.ndarray, dates: pd.DatetimeIndex, symbol: str
    ) -> None:
        bad = np.asarray(mask, dtype=bool)
        count = int(bad.sum())
        if not count:
            return
        value_mismatches[name] = value_mismatches.get(name, 0) + count
        if len(samples) < config.maximum_samples:
            location = int(np.flatnonzero(bad)[0])
            samples.append(f"{dates[location].date()}|{symbol}|{name}")

    for symbol, block in tape.groupby(level="symbol", sort=False):
        symbol = str(symbol)
        pair = records_by_code.get(symbol, {})
        raw_record = pair.get("3")
        adjusted_record = pair.get("1")
        if raw_record is None or adjusted_record is None:
            index_record("missing_cache_pair", len(block), symbol)
            continue
        raw_path = _resolve_cache_object(cache_dir, raw_record.get("object"))
        adjusted_path = _resolve_cache_object(cache_dir, adjusted_record.get("object"))
        if (
            raw_path is None
            or adjusted_path is None
            or not raw_path.is_file()
            or not adjusted_path.is_file()
        ):
            index_record("missing_cache_object", len(block), symbol)
            continue
        try:
            raw_frame = pd.read_parquet(raw_path)
            adjusted_frame = pd.read_parquet(adjusted_path)
        except Exception as exc:
            index_record("unreadable_cache_object", len(block), f"{symbol}:{type(exc).__name__}")
            continue
        if tuple(raw_frame.columns) != tuple(EQUITY_DAILY_FIELDS) or tuple(
            adjusted_frame.columns
        ) != tuple(EQUITY_DAILY_FIELDS):
            index_record("cache_schema", 1, symbol)
            continue
        raw_dates = pd.to_datetime(raw_frame["date"], errors="coerce").dt.normalize()
        adjusted_dates = pd.to_datetime(
            adjusted_frame["date"], errors="coerce"
        ).dt.normalize()
        adjusted_flags = set(adjusted_frame["adjustflag"].astype(str))
        payload_invalid = (
            raw_dates.isna().any()
            or adjusted_dates.isna().any()
            or not raw_frame["code"].astype(str).eq(symbol).all()
            or not adjusted_frame["code"].astype(str).eq(symbol).all()
            or not raw_frame["adjustflag"].astype(str).eq("3").all()
            or not adjusted_flags
            or not adjusted_flags.issubset({"1", "3"})
        )
        if payload_invalid:
            index_record("cache_payload_contract", 1, symbol)
            continue
        raw_frame = raw_frame.copy()
        adjusted_frame = adjusted_frame.copy()
        raw_frame["date"] = raw_dates
        adjusted_frame["date"] = adjusted_dates
        raw_view = raw_frame.rename(columns={"code": "symbol"}).set_index(
            ["date", "symbol"]
        ).sort_index()
        adjusted_view = adjusted_frame.rename(columns={"code": "symbol"}).set_index(
            ["date", "symbol"]
        ).sort_index()
        if (
            not raw_view.index.is_unique
            or not adjusted_view.index.is_unique
            or not raw_view.index.equals(adjusted_view.index)
        ):
            symmetric = raw_view.index.symmetric_difference(adjusted_view.index)
            index_record("raw_adjusted_source_index", max(1, len(symmetric)), symbol)
            continue
        if boundary_membership is None:
            index_record("price_boundary_membership_unavailable", 1, symbol)
            continue
        try:
            raw_view, adjusted_view, boundaries = (
                trim_leading_unadjusted_vendor_history(
                    raw_view,
                    adjusted_view,
                    boundary_membership.loc[
                        boundary_membership["code"].astype(str).eq(symbol)
                    ],
                )
            )
            observed_price_boundaries.extend(asdict(item) for item in boundaries)
        except Exception as exc:
            index_record(
                "price_adjusted_lineage_boundary",
                1,
                f"{symbol}:{type(exc).__name__}",
            )
            continue
        if declared_end is not None:
            keep = raw_view.index.get_level_values("date") <= declared_end
            raw_view = raw_view.loc[keep]
            adjusted_view = adjusted_view.loc[keep]
        source_union_dates.update(
            pd.DatetimeIndex(raw_view.index.get_level_values("date")).tolist()
        )
        block = block.sort_index()
        if not block.index.equals(raw_view.index):
            symmetric = block.index.symmetric_difference(raw_view.index)
            index_record(
                "exact_raw_cache_union",
                max(1, len(symmetric)),
                f"{symbol}:rows={len(block)}/{len(raw_view)}",
            )
            continue
        try:
            source_raw = normalize_baostock_daily_fields(raw_view)
        except Exception as exc:
            index_record("raw_normalization", len(block), f"{symbol}:{type(exc).__name__}")
            continue
        target_dates = pd.DatetimeIndex(block.index.get_level_values("date"))
        for field in ("open", "high", "low", "close", "preclose", "volume", "amount"):
            value_record(
                f"{field}_source",
                _numeric_mismatch(block[field], source_raw[field]),
                target_dates,
                symbol,
            )

        raw_close = pd.to_numeric(source_raw["close"], errors="coerce")
        adjusted_close = pd.to_numeric(adjusted_view["close"], errors="coerce")
        factor = adjusted_close.div(raw_close)
        value_record(
            "adj_factor_source",
            _numeric_mismatch(block["adj_factor"], factor),
            target_dates,
            symbol,
        )
        volume = pd.to_numeric(source_raw["volume"], errors="coerce")
        amount = pd.to_numeric(source_raw["amount"], errors="coerce")
        expected_vwap = amount.div(volume.where(volume.gt(0))).where(
            volume.gt(0), raw_close
        )
        value_record(
            "vwap_source",
            _numeric_mismatch(block["vwap"], expected_vwap),
            target_dates,
            symbol,
        )
        for field in ("open", "high", "low", "close"):
            expected_adjusted = pd.to_numeric(source_raw[field], errors="coerce").mul(
                factor
            )
            value_record(
                f"adj_{field}_source",
                _numeric_mismatch(block[f"adj_{field}"], expected_adjusted),
                target_dates,
                symbol,
            )
        value_record(
            "adj_vwap_source",
            _numeric_mismatch(block["adj_vwap"], expected_vwap.mul(factor)),
            target_dates,
            symbol,
        )

        expected_tradable = (
            pd.to_numeric(source_raw["tradestatus"], errors="coerce").eq(1)
            & volume.gt(0)
        )
        expected_st = pd.to_numeric(source_raw["isST"], errors="coerce").eq(1)
        value_record(
            "tradable_source",
            block["tradable"].eq(True).to_numpy() != expected_tradable.to_numpy(),
            target_dates,
            symbol,
        )
        value_record(
            "is_st_source",
            block["is_st"].eq(True).to_numpy() != expected_st.to_numpy(),
            target_dates,
            symbol,
        )
        if symbol not in ipo_map.index or pd.isna(ipo_map.get(symbol)):
            value_mismatches["ipo_date_source"] = value_mismatches.get(
                "ipo_date_source", 0
            ) + len(block)
        else:
            expected_listed = pd.Series(
                (target_dates - pd.Timestamp(ipo_map.loc[symbol])).days,
                index=block.index,
            )
            value_record(
                "listed_days_source",
                _numeric_mismatch(block["listed_days"], expected_listed),
                target_dates,
                symbol,
            )
            limit_input = pd.DataFrame(
                {
                    "open": pd.to_numeric(source_raw["open"], errors="coerce"),
                    "high": pd.to_numeric(source_raw["high"], errors="coerce"),
                    "low": pd.to_numeric(source_raw["low"], errors="coerce"),
                    "close": raw_close,
                    "preclose": pd.to_numeric(source_raw["preclose"], errors="coerce"),
                    "is_st": expected_st.to_numpy(dtype=bool),
                    "listed_days": expected_listed.to_numpy(dtype=float),
                    "tradable": expected_tradable.to_numpy(dtype=bool),
                },
                index=block.index,
            )
            expected_limits = derive_conservative_one_price_limits(
                limit_input, minimum_listed_days=minimum_days
            )
            for field in (
                "limit_up",
                "limit_down",
                "limit_proxy_known",
                "research_close_limit_up_proxy",
                "research_close_limit_down_proxy",
                "research_close_limit_known_proxy",
            ):
                value_record(
                    f"{field}_source",
                    block[field].eq(True).to_numpy()
                    != expected_limits[field].to_numpy(dtype=bool),
                    target_dates,
                    symbol,
                )
            ratio_bad = ~np.isclose(
                pd.to_numeric(block["limit_ratio_proxy"], errors="coerce").to_numpy(
                    dtype=float
                ),
                pd.to_numeric(
                    expected_limits["limit_ratio_proxy"], errors="coerce"
                ).to_numpy(dtype=float),
                rtol=0,
                atol=1e-12,
                equal_nan=True,
            )
            value_record(
                "limit_ratio_proxy_source", ratio_bad, target_dates, symbol
            )
        audited_symbols += 1

    actual_union_dates = set(
        pd.DatetimeIndex(tape.index.get_level_values("date")).normalize()
    )
    missing_union_dates = sorted(source_union_dates - actual_union_dates)
    extra_union_dates = sorted(actual_union_dates - source_union_dates)
    if missing_union_dates or extra_union_dates:
        index_record(
            "source_union_dates",
            len(missing_union_dates) + len(extra_union_dates),
            "date-union",
        )
    declared_boundaries = manifest.get("price_history_boundaries")
    if observed_price_boundaries or isinstance(declared_boundaries, Mapping):
        observed_price_boundaries.sort(key=lambda item: str(item.get("symbol", "")))
        observed_hash = hashlib.sha256(
            json.dumps(
                observed_price_boundaries,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        declared_records = (
            declared_boundaries.get("records")
            if isinstance(declared_boundaries, Mapping)
            else None
        )
        boundary_valid = (
            isinstance(declared_boundaries, Mapping)
            and declared_boundaries.get("schema_version") == "1.0"
            and declared_boundaries.get("complete") is True
            and declared_boundaries.get("policy")
            == "trim_only_leading_adjustflag_3_rows_before_first_membership"
            and declared_records == observed_price_boundaries
            and int(declared_boundaries.get("rows", -1))
            == len(observed_price_boundaries)
            and str(declared_boundaries.get("sha256", "")) == observed_hash
        )
        if not boundary_valid:
            index_record(
                "price_history_boundary_manifest",
                1,
                "declared-boundaries-differ-from-source-replay",
            )
    tape_metrics.update(
        {
            "source_symbols_audited": audited_symbols,
            "source_index_mismatches": dict(sorted(index_mismatches.items())),
            "source_value_mismatches": dict(sorted(value_mismatches.items())),
            "price_history_boundaries": observed_price_boundaries,
        }
    )
    if index_mismatches:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_source_index_mismatch",
            "feature-tape index differs from replayed raw/adjusted lineage boundaries",
            count=sum(index_mismatches.values()),
            samples=[
                *samples,
                *[f"{key}={value}" for key, value in sorted(index_mismatches.items())],
            ],
            maximum_samples=config.maximum_samples,
        )
    if value_mismatches:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_source_lineage_mismatch",
            "feature-tape execution and research core fields differ from frozen cache sources",
            count=sum(value_mismatches.values()),
            samples=[
                *samples,
                *[f"{key}={value}" for key, value in sorted(value_mismatches.items())],
            ],
            maximum_samples=config.maximum_samples,
        )


def _audit_feature_tape(
    panel: pd.DataFrame,
    manifest: Mapping[str, object],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    """Independently rebuild the continuous union tape's index and core values."""

    section = manifest.get("feature_tape")
    if not isinstance(section, Mapping):
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_manifest_missing",
            "manifest lacks the continuous union feature-tape contract",
        )
        return
    records = section.get("partitions")
    if not isinstance(records, list) or not records:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_partitions_missing",
            "feature-tape manifest has no frozen partitions",
        )
        return

    contract_errors: list[str] = []
    if section.get("schema_version") != "1.0":
        contract_errors.append("schema_version")
    if section.get("complete") is not True:
        contract_errors.append("complete")
    if section.get("purpose") != (
        "continuous_union_history_for_time_series_features_labels_and_execution"
    ):
        contract_errors.append("purpose")
    if section.get("index_policy") != (
        "raw_cache_union_after_strict_pre_membership_adjusted_lineage_boundaries"
    ):
        contract_errors.append("index_policy")
    storage_snapshot_id = str(manifest.get("storage_snapshot_id", ""))
    if (
        re.fullmatch(r"[0-9a-f]{64}", storage_snapshot_id) is None
        or section.get("storage_snapshot_id") != storage_snapshot_id
    ):
        contract_errors.append("storage_snapshot_id")
    try:
        if int(section.get("provider_members_per_day", -1)) != config.expected_members:
            contract_errors.append("provider_members_per_day")
    except (TypeError, ValueError):
        contract_errors.append("provider_members_per_day")
    if section.get("member_bar_exception_policy") != _TERMINAL_STALE_REASON:
        contract_errors.append("member_bar_exception_policy")
    source_contract = section.get("source_contract")
    raw_records = manifest.get("price_cache_objects")
    membership_records = manifest.get("membership_cache_objects")
    security_records = manifest.get("security_cache_objects")
    industry_records = manifest.get("industry_cache_objects")
    adjustment_records = manifest.get("adjustment_cache_objects")
    price_boundary_section = manifest.get("price_history_boundaries")
    if (
        not isinstance(source_contract, Mapping)
        or not isinstance(raw_records, list)
        or not isinstance(membership_records, list)
        or not isinstance(security_records, list)
        or not isinstance(industry_records, list)
        or not isinstance(adjustment_records, list)
    ):
        contract_errors.append("source_contract")
    else:
        if source_contract.get("raw_adjustflag") != "3":
            contract_errors.append("raw_adjustflag")
        if source_contract.get("research_adjustflag") != "1":
            contract_errors.append("research_adjustflag")
        if source_contract.get("price_cache_objects_sha256") != _canonical_json_sha256(
            raw_records
        ):
            contract_errors.append("price_cache_objects_sha256")
        if (
            not isinstance(price_boundary_section, Mapping)
            or not isinstance(price_boundary_section.get("records"), list)
            or source_contract.get("price_history_boundaries_sha256")
            != _canonical_json_sha256(price_boundary_section["records"])
        ):
            contract_errors.append("price_history_boundaries_sha256")
        for name, records_value in (
            ("membership", membership_records),
            ("security", security_records),
            ("industry", industry_records),
        ):
            if source_contract.get(
                f"{name}_cache_objects_sha256"
            ) != _canonical_json_sha256(records_value):
                contract_errors.append(f"{name}_cache_objects_sha256")
        source_lineage = manifest.get("source_lineage")
        if (
            not isinstance(source_lineage, Mapping)
            or source_contract.get("source_lineage_root_sha256")
            != source_lineage.get("root_sha256")
        ):
            contract_errors.append("source_lineage_root_sha256")
        if source_contract.get(
            "adjustment_cache_objects_sha256"
        ) != _canonical_json_sha256(adjustment_records):
            contract_errors.append("adjustment_cache_objects_sha256")
        declared_normalized = manifest.get("normalized_input_sha256")
        if (
            not isinstance(declared_normalized, Mapping)
            or source_contract.get("adjustment_request_index_sha256")
            != declared_normalized.get("adjustment_request_index")
        ):
            contract_errors.append("adjustment_request_index_sha256")
        terminal_section = manifest.get("terminal_stale_membership_exceptions")
        if (
            not isinstance(terminal_section, Mapping)
            or source_contract.get(
                "terminal_stale_membership_exceptions_sha256"
            )
            != terminal_section.get("snapshot_hash")
        ):
            contract_errors.append(
                "terminal_stale_membership_exceptions_sha256"
            )
        execution_limit_section = manifest.get(
            "execution_one_price_limit_proxy"
        )
        if (
            not isinstance(execution_limit_section, Mapping)
            or source_contract.get("execution_one_price_limit_proxy_sha256")
            != execution_limit_section.get("snapshot_hash")
        ):
            contract_errors.append("execution_one_price_limit_proxy_sha256")
        research_limit_section = manifest.get("research_close_limit_proxy")
        if (
            not isinstance(research_limit_section, Mapping)
            or source_contract.get("research_close_limit_proxy_sha256")
            != research_limit_section.get("snapshot_hash")
        ):
            contract_errors.append("research_close_limit_proxy_sha256")
        index_record = manifest.get("index_cache_object")
        if (
            not isinstance(index_record, Mapping)
            or source_contract.get("index_cache_object_sha256")
            != _canonical_json_sha256(dict(index_record))
        ):
            contract_errors.append("index_cache_object_sha256")
        try:
            pair_count_matches = int(source_contract.get("price_cache_pairs", -1)) == (
                len(raw_records) // 2
            )
        except (TypeError, ValueError):
            pair_count_matches = False
        if not pair_count_matches:
            contract_errors.append("price_cache_pairs")
    if contract_errors:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_manifest_contract_invalid",
            "continuous feature-tape manifest contract is incomplete or inconsistent",
            count=len(contract_errors),
            samples=contract_errors,
            maximum_samples=config.maximum_samples,
        )

    manifest_columns = section.get("columns")
    manifest_dtypes = section.get("dtypes")
    schema_contract_valid = (
        isinstance(manifest_columns, list)
        and bool(manifest_columns)
        and len({str(name) for name in manifest_columns}) == len(manifest_columns)
        and isinstance(manifest_dtypes, Mapping)
        and set(str(name) for name in manifest_columns)
        == {str(name) for name in manifest_dtypes}
    )
    if not schema_contract_valid:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_schema_contract_invalid",
            "feature-tape manifest lacks one exact ordered column/dtype schema",
        )
        expected_columns: tuple[str, ...] = ()
        expected_dtypes: dict[str, str] = {}
    else:
        expected_columns = tuple(str(name) for name in manifest_columns)
        expected_dtypes = {
            str(name): str(dtype) for name, dtype in manifest_dtypes.items()
        }

    root = config.panel_dir.resolve()
    expected_feature_root = (
        root / "snapshots" / storage_snapshot_id / "feature_tape"
    ).resolve()
    frames: list[pd.DataFrame] = []
    logical_hasher = hashlib.sha256()
    paths_seen: set[Path] = set()
    years_seen: set[int] = set()
    verified = 0
    total_bytes = 0
    schema_mismatches = 0
    boolean_domain_counts: dict[str, int] = {}
    common_field_mismatches: dict[str, int] = {}
    common_field_samples: list[str] = []
    panel_dates_for_schema = pd.DatetimeIndex(
        panel.index.get_level_values("date")
    )
    for number, record in enumerate(records):
        if not isinstance(record, Mapping):
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_record_invalid",
                f"feature-tape partition record {number} is not an object",
            )
            continue
        path = _resolve_partition_path(root, record.get("path"))
        if path is None:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_path_unsafe",
                f"feature-tape partition {number} is outside the frozen snapshot",
            )
            continue
        if path in paths_seen:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_path_duplicate",
                f"feature-tape path is declared twice: {path}",
            )
            continue
        paths_seen.add(path)
        try:
            year = int(record.get("year"))
        except (TypeError, ValueError):
            year = -1
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_year_invalid",
                f"feature-tape partition {number} has no valid year",
            )
        if year in years_seen:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_year_duplicate",
                f"feature-tape year {year} is declared more than once",
            )
        years_seen.add(year)
        expected_year_root = (expected_feature_root / f"year={year}").resolve()
        if (
            not path.is_relative_to(expected_feature_root)
            or path.parent != expected_year_root
        ):
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_not_bound_to_snapshot",
                "feature-tape partition path is not under the current snapshot/feature_tape year",
                samples=[path],
                maximum_samples=config.maximum_samples,
            )
        if not path.is_file():
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_file_missing",
                f"feature-tape partition is missing: {path}",
            )
            continue
        total_bytes += path.stat().st_size
        actual_hash = _sha256_file(path)
        expected_hash = str(record.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or actual_hash != expected_hash:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_sha_mismatch",
                f"feature-tape hash mismatch: {path.name} ({year})",
            )
        else:
            verified += 1
        try:
            expected_bytes = int(record.get("bytes", -1))
        except (TypeError, ValueError):
            expected_bytes = -1
        if expected_bytes != path.stat().st_size:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_byte_count_mismatch",
                f"feature-tape byte count mismatch for {year}",
            )
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_unreadable",
                f"cannot read {path}: {type(exc).__name__}: {exc}",
            )
            continue
        try:
            expected_rows = int(record.get("rows", -1))
        except (TypeError, ValueError):
            expected_rows = -1
        if len(frame) != expected_rows:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_row_count_mismatch",
                f"feature-tape row count disagrees with manifest for {year}",
            )
        if not isinstance(frame.index, pd.MultiIndex) or frame.index.names != ["date", "symbol"]:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_index_invalid",
                f"feature-tape partition {year} lacks MultiIndex(date, symbol)",
            )
            continue
        actual_columns = tuple(str(name) for name in frame.columns)
        actual_dtypes = {str(name): str(dtype) for name, dtype in frame.dtypes.items()}
        if (
            not schema_contract_valid
            or actual_columns != expected_columns
            or actual_dtypes != expected_dtypes
        ):
            schema_mismatches += 1
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_schema_mismatch",
                f"feature-tape partition {year} differs from the frozen ordered schema",
            )
        for name in _FEATURE_TAPE_BOOLEAN_COLUMNS:
            if name not in frame:
                continue
            invalid = frame[name].isna() | ~frame[name].isin([True, False, 0, 1])
            invalid_count = int(invalid.sum())
            if not pd.api.types.is_bool_dtype(frame[name].dtype):
                # The builder emits physical booleans.  Re-sealing an integer
                # or string surrogate is not an equivalent storage contract.
                invalid_count = max(invalid_count, len(frame))
            if invalid_count:
                boolean_domain_counts[name] = (
                    boolean_domain_counts.get(name, 0) + invalid_count
                )

        # Compare every field shared by the member panel while this one year is
        # resident.  This avoids retaining a second wide copy of the union tape.
        panel_year_index = panel.index[panel_dates_for_schema.year == year]
        common_index = panel_year_index.intersection(frame.index, sort=False)
        if len(common_index):
            for name in sorted(set(frame.columns).intersection(panel.columns)):
                bad = _series_value_mismatch(
                    frame.loc[common_index, name], panel.loc[common_index, name]
                )
                count = int(bad.sum())
                if not count:
                    continue
                common_field_mismatches[name] = (
                    common_field_mismatches.get(name, 0) + count
                )
                if len(common_field_samples) < config.maximum_samples:
                    location = int(np.flatnonzero(bad)[0])
                    item = common_index[location]
                    common_field_samples.append(f"{item[0].date()}|{item[1]}|{name}")
        logical_hashes = pd.util.hash_pandas_object(
            frame, index=True, categorize=True
        ).to_numpy(dtype=np.uint64)
        logical_hasher.update(logical_hashes.tobytes())
        dates = pd.to_datetime(frame.index.get_level_values("date"), errors="coerce")
        if dates.isna().any() or (dates.year != year).any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_partition_year_content_mismatch",
                f"feature-tape partition {year} contains invalid or out-of-year dates",
            )
        retained = [
            name for name in _FEATURE_TAPE_REQUIRED_COLUMNS if name in frame.columns
        ]
        frames.append(frame[retained])

    tape_metrics: dict[str, object] = {
        "partitions_declared": len(records),
        "partitions_loaded": len(frames),
        "partitions_sha_verified": verified,
        "partition_bytes": total_bytes,
        "partition_schema_mismatches": schema_mismatches,
        "boolean_domain_mismatches": dict(sorted(boolean_domain_counts.items())),
        "member_panel_common_field_mismatches": dict(
            sorted(common_field_mismatches.items())
        ),
    }
    metrics["continuous_feature_tape"] = tape_metrics
    if boolean_domain_counts:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_boolean_domain_invalid",
            "feature-tape boolean fields contain nulls, invalid values or a non-boolean physical dtype",
            count=sum(boolean_domain_counts.values()),
            samples=[
                f"{name}={count}" for name, count in sorted(boolean_domain_counts.items())
            ],
            maximum_samples=config.maximum_samples,
        )
    if common_field_mismatches:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_member_panel_field_mismatch",
            "feature-tape member rows differ from the frozen panel on shared fields",
            count=sum(common_field_mismatches.values()),
            samples=[
                *common_field_samples,
                *[
                    f"{name}={count}"
                    for name, count in sorted(common_field_mismatches.items())
                ],
            ],
            maximum_samples=config.maximum_samples,
        )
    if not frames:
        return
    tape = pd.concat(frames, copy=False)
    del frames
    if not tape.index.is_monotonic_increasing:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_order_not_monotonic",
            "feature-tape partition order does not reconstruct a sorted tape",
        )
        tape = tape.sort_index()
    if not tape.index.is_unique:
        duplicate = tape.index[tape.index.duplicated(keep=False)]
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_duplicate_date_symbol",
            "continuous feature tape contains duplicate date-symbol rows",
            count=len(duplicate),
            samples=_sample_index(duplicate, config.maximum_samples),
            maximum_samples=config.maximum_samples,
        )

    missing = sorted(set(_FEATURE_TAPE_REQUIRED_COLUMNS).difference(tape.columns))
    if missing:
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_required_columns_missing",
            "continuous feature tape lacks required research/execution fields",
            count=len(missing),
            samples=missing,
            maximum_samples=config.maximum_samples,
        )
        return

    dates = pd.DatetimeIndex(tape.index.get_level_values("date"))
    symbols = pd.Index(tape.index.get_level_values("symbol").astype(str))
    actual_summary = {
        "rows": len(tape),
        "symbols": int(symbols.nunique()),
        "trading_days": int(dates.nunique()),
        "first_date": str(dates.min().date()),
        "last_date": str(dates.max().date()),
        "snapshot_hash": logical_hasher.hexdigest(),
    }
    tape_metrics.update(actual_summary)
    for name, actual in actual_summary.items():
        expected = section.get(name)
        if name in {"rows", "symbols", "trading_days"}:
            try:
                matches = int(expected) == int(actual)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = str(expected) == str(actual)
        if not matches:
            _add(
                findings,
                AuditSeverity.FAIL,
                f"feature_tape_manifest_{name}_mismatch",
                f"feature-tape manifest {name} disagrees with frozen partitions",
            )

    declared_normalized = manifest.get("normalized_input_sha256")
    declared_index_cache = manifest.get("index_cache_object")
    declared_adjustment_cache = manifest.get("adjustment_cache_objects")
    declared_membership_cache = manifest.get("membership_cache_objects")
    declared_security_cache = manifest.get("security_cache_objects")
    declared_industry_cache = manifest.get("industry_cache_objects")
    declared_terminal = manifest.get("terminal_stale_membership_exceptions")
    declared_eligibility = manifest.get("research_eligibility")
    declared_execution_limit = manifest.get("execution_one_price_limit_proxy")
    declared_research_limit = manifest.get("research_close_limit_proxy")
    declared_source_lineage = manifest.get("source_lineage")
    declared_price_boundaries = manifest.get("price_history_boundaries")
    if (
        isinstance(raw_records, list)
        and isinstance(declared_normalized, Mapping)
        and isinstance(declared_index_cache, Mapping)
        and isinstance(declared_adjustment_cache, list)
        and isinstance(declared_membership_cache, list)
        and isinstance(declared_security_cache, list)
        and isinstance(declared_industry_cache, list)
        and isinstance(declared_terminal, Mapping)
        and isinstance(declared_eligibility, Mapping)
        and isinstance(declared_execution_limit, Mapping)
        and isinstance(declared_research_limit, Mapping)
        and isinstance(declared_source_lineage, Mapping)
        and isinstance(declared_price_boundaries, Mapping)
        and isinstance(declared_price_boundaries.get("records"), list)
    ):
        expected_storage_snapshot_id = _canonical_json_sha256(
            {
                "panel_snapshot_hash": str(manifest.get("snapshot_hash", "")),
                "feature_snapshot_hash": actual_summary["snapshot_hash"],
                "normalized_input_sha256": dict(declared_normalized),
                "price_cache_objects_sha256": _canonical_json_sha256(raw_records),
                "price_history_boundaries_sha256": _canonical_json_sha256(
                    declared_price_boundaries["records"]
                ),
                "membership_cache_objects_sha256": _canonical_json_sha256(
                    declared_membership_cache
                ),
                "security_cache_objects_sha256": _canonical_json_sha256(
                    declared_security_cache
                ),
                "industry_cache_objects_sha256": _canonical_json_sha256(
                    declared_industry_cache
                ),
                "source_lineage_root_sha256": str(
                    declared_source_lineage.get("root_sha256", "")
                ),
                "adjustment_cache_objects_sha256": _canonical_json_sha256(
                    declared_adjustment_cache
                ),
                "terminal_stale_membership_exceptions_sha256": str(
                    declared_terminal.get("snapshot_hash", "")
                ),
                "research_eligibility_sha256": str(
                    declared_eligibility.get("snapshot_hash", "")
                ),
                "execution_one_price_limit_proxy_sha256": str(
                    declared_execution_limit.get("snapshot_hash", "")
                ),
                "research_close_limit_proxy_sha256": str(
                    declared_research_limit.get("snapshot_hash", "")
                ),
                "index_cache_object_sha256": _canonical_json_sha256(
                    dict(declared_index_cache)
                ),
            }
        )
        tape_metrics["expected_storage_snapshot_id"] = expected_storage_snapshot_id
        if expected_storage_snapshot_id != storage_snapshot_id:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_storage_snapshot_identity_mismatch",
                "snapshot directory identity is not bound to panel, feature tape and source contracts",
            )

    numeric_core = (
        "open", "high", "low", "close", "vwap",
        "adj_open", "adj_high", "adj_low", "adj_close", "adj_vwap",
        "preclose", "volume", "amount", "adj_factor", "listed_days",
    )
    for name in numeric_core:
        values = pd.to_numeric(tape[name], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_nonfinite_core_field",
                f"feature-tape field {name} contains non-finite values",
                count=int(bad.sum()),
            )

    try:
        terminal_exceptions = _validated_terminal_exception_sidecar(
            manifest, config.panel_dir
        )
        tape_metrics["terminal_stale_membership_exception_rows"] = len(
            terminal_exceptions
        )
    except Exception as exc:
        terminal_exceptions = pd.DataFrame()
        _add(
            findings,
            AuditSeverity.FAIL,
            "terminal_stale_membership_sidecar_invalid",
            f"terminal-stale membership exception audit failed: {exc}",
        )

    member_domain_invalid = tape["vendor_universe_member"].isna() | ~tape[
        "vendor_universe_member"
    ].isin([True, False, 0, 1])
    if member_domain_invalid.any():
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_member_mask_missing",
            "feature-tape provider membership mask contains missing or non-boolean values",
            count=int(member_domain_invalid.sum()),
        )
    # Never call bool("False"): invalid encodings remain non-members while the
    # domain finding above fails the audit closed.
    # Audit the legacy alias as an independently attackable copy as well as
    # checking exact equality to vendor_universe_member below.  This preserves
    # fail-closed behavior for older consumers that still read the alias.
    member_mask = tape["universe_member"].eq(True)
    panel_dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    in_panel_span = (dates >= panel_dates.min()) & (dates <= panel_dates.max())
    declared_members = tape.index[in_panel_span & member_mask.to_numpy(dtype=bool)]
    if not declared_members.equals(panel.index):
        missing_members = panel.index.difference(declared_members)
        extra_members = declared_members.difference(panel.index)
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_member_mask_mismatch",
            "feature-tape daily eligibility does not exactly reproduce the member panel",
            count=len(missing_members) + len(extra_members),
            samples=(
                [f"missing:{item}" for item in _sample_index(missing_members, config.maximum_samples)]
                + [f"extra:{item}" for item in _sample_index(extra_members, config.maximum_samples)]
            ),
            maximum_samples=config.maximum_samples,
        )
    daily_members = member_mask.groupby(level="date").sum()
    exception_daily = pd.Series(0, index=daily_members.index, dtype=int)
    if len(terminal_exceptions):
        counts = pd.Series(1, index=terminal_exceptions.index).groupby(
            level="date"
        ).sum()
        exception_daily = counts.reindex(daily_members.index, fill_value=0).astype(int)
    expected_daily = config.expected_members - exception_daily
    bad_daily = daily_members.loc[~daily_members.eq(expected_daily)]
    if len(bad_daily):
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_daily_member_count_not_expected",
            "feature-tape provider-member bars must equal membership minus exact terminal exceptions",
            count=len(bad_daily),
            samples=[f"{date.date()}={int(value)}" for date, value in bad_daily.items()],
            maximum_samples=config.maximum_samples,
        )
    _audit_research_eligibility_contract(
        tape, manifest, config, findings, metrics
    )
    _audit_separate_limit_proxy_contracts(
        tape, manifest, config, findings, metrics
    )

    # Suspensions are explicit BaoStock rows, so every security must have a
    # bar on each market session between its first and last cached observation.
    unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
    date_position = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    grouped = pd.DataFrame({"date": dates}, index=tape.index).groupby(level="symbol")["date"]
    first = grouped.min()
    last = grouped.max()
    observed = grouped.nunique()
    expected = last.map(date_position) - first.map(date_position) + 1
    gaps = (expected - observed).astype(int)
    broken = gaps.loc[gaps.ne(0)]
    tape_metrics["symbols_with_internal_date_gaps"] = len(broken)
    if len(broken):
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_internal_date_gaps",
            "one or more securities lose sessions inside their cached life span",
            count=len(broken),
            samples=[f"{symbol}={int(count)}" for symbol, count in broken.items()],
            maximum_samples=config.maximum_samples,
        )

    warmup_sessions = int((unique_dates < panel_dates.min()).sum())
    tape_metrics["warmup_sessions_before_research"] = warmup_sessions
    if warmup_sessions < 252:
        _add(
            findings,
            AuditSeverity.WARN,
            "feature_tape_warmup_shorter_than_max_dsl_window",
            "early research dates cannot support every permitted 252-session factor",
            count=warmup_sessions,
        )

    normalized_frames = _load_feature_tape_normalized_sources(
        manifest, section, config, findings, tape_metrics
    )
    calendar = normalized_frames.get("trade_calendar")
    if calendar is None or not {"calendar_date", "is_trading_day"}.issubset(
        calendar.columns
    ):
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_trade_calendar_schema_invalid",
            "trade calendar cannot prove feature-tape session continuity",
        )
    else:
        trading_rows = calendar["is_trading_day"].astype(str).eq("1")
        parsed_calendar = pd.to_datetime(
            calendar.loc[trading_rows, "calendar_date"], errors="coerce"
        )
        if parsed_calendar.isna().any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_trade_calendar_date_invalid",
                "trade calendar contains an invalid trading date",
                count=int(parsed_calendar.isna().sum()),
            )
        calendar_dates = pd.DatetimeIndex(parsed_calendar.dropna()).normalize()
        if calendar_dates.has_duplicates:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_trade_calendar_duplicate_date",
                "trade calendar contains duplicate trading sessions",
            )
        bounded_calendar = calendar_dates[
            (calendar_dates >= dates.min()) & (calendar_dates <= dates.max())
        ].unique().sort_values()
        missing_dates = bounded_calendar.difference(unique_dates)
        extra_dates = unique_dates.difference(bounded_calendar)
        tape_metrics["calendar_missing_dates"] = len(missing_dates)
        tape_metrics["calendar_extra_dates"] = len(extra_dates)
        if len(missing_dates) or len(extra_dates):
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_trade_calendar_mismatch",
                "feature-tape dates do not exactly cover the frozen trading-calendar interval",
                count=len(missing_dates) + len(extra_dates),
                samples=[
                    *[f"missing:{date.date()}" for date in missing_dates],
                    *[f"extra:{date.date()}" for date in extra_dates],
                ],
                maximum_samples=config.maximum_samples,
            )

    membership = normalized_frames.get("membership_daily")
    if membership is None or not {"date", "code"}.issubset(membership.columns):
        _add(
            findings,
            AuditSeverity.FAIL,
            "feature_tape_membership_source_schema_invalid",
            "membership_daily cannot prove the full-period feature-tape mask",
        )
    else:
        members = membership[["date", "code"]].copy()
        members["date"] = pd.to_datetime(members["date"], errors="coerce").dt.normalize()
        members["code"] = members["code"].astype("string").str.strip()
        invalid_members = members["date"].isna() | members["code"].isna() | members[
            "code"
        ].eq("")
        duplicate_members = members.duplicated(["date", "code"], keep=False)
        if invalid_members.any() or duplicate_members.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_membership_source_invalid",
                "membership_daily contains invalid or duplicate date-symbol keys",
                count=int(invalid_members.sum() + duplicate_members.sum()),
            )
        members = members.loc[
            ~invalid_members & members["date"].between(dates.min(), dates.max())
        ].sort_values(["date", "code"])
        expected_membership = pd.MultiIndex.from_frame(
            members.rename(columns={"code": "symbol"})[["date", "symbol"]],
            names=["date", "symbol"],
        )
        if len(terminal_exceptions):
            expected_membership = expected_membership.difference(
                terminal_exceptions.index
            )
        actual_membership = tape.index[member_mask.to_numpy(dtype=bool)]
        missing_members = expected_membership.difference(actual_membership)
        extra_members = actual_membership.difference(expected_membership)
        membership_exact = (
            len(expected_membership) == len(actual_membership)
            and not len(missing_members)
            and not len(extra_members)
        )
        tape_metrics["membership_mask_exact"] = membership_exact
        if not membership_exact:
            _add(
                findings,
                AuditSeverity.FAIL,
                "feature_tape_normalized_membership_mask_mismatch",
                "feature-tape provider mask differs from membership_daily minus exact terminal exceptions",
                count=len(missing_members) + len(extra_members),
                samples=[
                    *[
                        f"missing:{item}"
                        for item in _sample_index(
                            missing_members, config.maximum_samples
                        )
                    ],
                    *[
                        f"extra:{item}"
                        for item in _sample_index(extra_members, config.maximum_samples)
                    ],
                ],
                maximum_samples=config.maximum_samples,
            )

    _audit_feature_tape_source_lineage(
        tape,
        manifest,
        normalized_frames,
        config,
        findings,
        tape_metrics,
    )


def _audit_panel_structure(
    panel: pd.DataFrame,
    manifest: Mapping[str, object],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["date", "symbol"]:
        _add(findings, AuditSeverity.FAIL, "panel_index_invalid", "panel index is not date-symbol")
        return
    if not panel.index.is_unique:
        duplicate = panel.index[panel.index.duplicated(keep=False)]
        _add(
            findings,
            AuditSeverity.FAIL,
            "duplicate_date_symbol_rows",
            "frozen partitions contain duplicate date-symbol observations",
            count=int(panel.index.duplicated(keep=False).sum()),
            samples=_sample_index(duplicate, config.maximum_samples),
            maximum_samples=config.maximum_samples,
        )
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    symbols = pd.Index(panel.index.get_level_values("symbol").astype(str))
    if dates.isna().any():
        _add(findings, AuditSeverity.FAIL, "invalid_panel_dates", "panel contains invalid dates")
        return
    if (symbols.str.strip() == "").any():
        _add(findings, AuditSeverity.FAIL, "blank_symbols", "panel contains blank symbols")

    counts = pd.Series(1, index=panel.index).groupby(level="date").sum()
    build_config = manifest.get("config")
    frozen_expected = (
        int(build_config.get("expected_members", -1))
        if isinstance(build_config, Mapping)
        else -1
    )
    if frozen_expected != config.expected_members:
        _add(
            findings,
            AuditSeverity.FAIL,
            "audit_expected_members_differs_from_frozen_policy",
            f"audit requested {config.expected_members} members but build froze {frozen_expected}",
        )
    try:
        terminal_exceptions = _validated_terminal_exception_sidecar(
            manifest, config.panel_dir
        )
    except Exception as exc:
        terminal_exceptions = pd.DataFrame()
        _add(
            findings,
            AuditSeverity.FAIL,
            "terminal_stale_membership_sidecar_invalid",
            f"terminal-stale membership exception audit failed: {exc}",
        )
    exception_daily = pd.Series(0, index=counts.index, dtype=int)
    if len(terminal_exceptions):
        terminal_in_span = terminal_exceptions.loc[
            terminal_exceptions.index.get_level_values("date").isin(counts.index)
        ]
        terminal_counts = pd.Series(1, index=terminal_in_span.index).groupby(
            level="date"
        ).sum()
        exception_daily = terminal_counts.reindex(counts.index, fill_value=0).astype(int)
    expected_counts = config.expected_members - exception_daily
    bad_counts = counts.loc[~counts.eq(expected_counts)]
    metrics["membership"] = {
        "expected_members_per_day": config.expected_members,
        "terminal_exception_rows_in_panel_span": int(exception_daily.sum()),
        "minimum_members": int(counts.min()) if len(counts) else 0,
        "maximum_members": int(counts.max()) if len(counts) else 0,
        "bad_dates": len(bad_counts),
    }
    if len(bad_counts):
        _add(
            findings,
            AuditSeverity.FAIL,
            "daily_member_count_not_expected",
            "each date must contain provider members minus exact terminal-stale exceptions",
            count=len(bad_counts),
            rate=len(bad_counts) / len(counts),
            samples=[f"{date.date()}={int(value)}" for date, value in bad_counts.items()],
            maximum_samples=config.maximum_samples,
        )
    unique_counts = pd.Series(symbols, index=panel.index).groupby(level="date").nunique()
    duplicate_within_day = unique_counts.ne(counts)
    if duplicate_within_day.any():
        _add(
            findings,
            AuditSeverity.FAIL,
            "daily_symbols_not_unique",
            "one or more dates repeat a member symbol",
            count=int(duplicate_within_day.sum()),
        )
    if "vendor_universe_member" in panel and not panel[
        "vendor_universe_member"
    ].fillna(False).astype(bool).all():
        bad = panel.index[
            ~panel["vendor_universe_member"].fillna(False).astype(bool)
        ]
        _add(
            findings,
            AuditSeverity.FAIL,
            "nonmember_rows_in_frozen_panel",
            "frozen HS300 panel contains rows not marked as members",
            count=len(bad),
            samples=_sample_index(bad, config.maximum_samples),
            maximum_samples=config.maximum_samples,
        )

    for manifest_name, actual in (
        ("rows", len(panel)),
        ("symbols", symbols.nunique()),
        ("trading_days", dates.nunique()),
    ):
        try:
            expected = int(manifest.get(manifest_name, -1))
        except (TypeError, ValueError):
            expected = -1
        if expected != int(actual):
            _add(
                findings,
                AuditSeverity.FAIL,
                f"manifest_{manifest_name}_mismatch",
                f"manifest {manifest_name}={expected} but partitions contain {int(actual)}",
            )
    if len(panel):
        actual_first = str(dates.min().date())
        actual_last = str(dates.max().date())
        if str(manifest.get("first_date", "")) != actual_first:
            _add(findings, AuditSeverity.FAIL, "manifest_first_date_mismatch", "manifest first date disagrees with partitions")
        if str(manifest.get("last_date", "")) != actual_last:
            _add(findings, AuditSeverity.FAIL, "manifest_last_date_mismatch", "manifest last date disagrees with partitions")

    actual_hash = str(metrics.get("integrity", {}).get("actual_snapshot_hash", ""))
    if not actual_hash:
        # Compatibility fallback for a programmatic caller that bypasses the
        # manifest loader; normal audits use the streaming full-column hash.
        actual_hash = panel_snapshot_hash(panel)
    metrics["integrity"]["expected_snapshot_hash"] = str(manifest.get("snapshot_hash", ""))
    metrics["integrity"]["actual_snapshot_hash"] = actual_hash
    if actual_hash != str(manifest.get("snapshot_hash", "")):
        _add(
            findings,
            AuditSeverity.FAIL,
            "snapshot_hash_mismatch",
            "reconstructed panel hash disagrees with the build-time snapshot hash",
        )
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping) or str(validation.get("snapshot_hash", "")) != actual_hash:
        _add(
            findings,
            AuditSeverity.FAIL,
            "embedded_validation_hash_mismatch",
            "embedded DataContract validation hash disagrees with frozen partitions",
        )
    if not isinstance(validation, Mapping):
        return
    if validation.get("valid") is not True or list(validation.get("errors", [])):
        _add(
            findings,
            AuditSeverity.FAIL,
            "embedded_validation_failed",
            "build-time DataContract report is invalid or contains errors",
            samples=list(validation.get("errors", [])),
            maximum_samples=config.maximum_samples,
        )
    embedded_values = {
        "rows": len(panel),
        "symbols": int(symbols.nunique()),
        "trading_days": int(dates.nunique()),
        "first_date": str(dates.min().date()) if len(dates) else "",
        "last_date": str(dates.max().date()) if len(dates) else "",
    }
    mismatched = [
        name for name, actual in embedded_values.items() if validation.get(name) != actual
    ]
    if mismatched:
        _add(
            findings,
            AuditSeverity.FAIL,
            "embedded_validation_summary_mismatch",
            "build-time DataContract summary disagrees with frozen partitions",
            count=len(mismatched),
            samples=mismatched,
            maximum_samples=config.maximum_samples,
        )


def _audit_column_coverage(
    panel: pd.DataFrame,
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    coverage = {name: _coverage(panel, name) for name in _AUDIT_REQUIRED_COLUMNS}
    metrics["coverage"] = coverage
    missing_columns = [name for name, item in coverage.items() if not item["present"]]
    if missing_columns:
        _add(
            findings,
            AuditSeverity.FAIL,
            "required_columns_missing",
            "frozen panel omits audit-required fields",
            count=len(missing_columns),
            samples=missing_columns,
            maximum_samples=config.maximum_samples,
        )

    # Every core field except the explicitly imperfect capitalization proxy is
    # complete by contract.  Proxy coverage has its own threshold below.
    complete_fields = [name for name in _AUDIT_REQUIRED_COLUMNS if name != "implied_float_mcap"]
    eligible = (
        panel["research_eligible"].eq(True)
        if "research_eligible" in panel.columns
        else pd.Series(True, index=panel.index, dtype=bool)
    )
    sparse_when_ineligible = {"log_mcap", "beta", "adv20_dollars"}
    for name in complete_fields:
        item = coverage[name]
        if not item["present"]:
            continue
        missing = panel[name].isna()
        missing_for_validation = (
            missing & eligible if name in sparse_when_ineligible else missing
        )
        ignored = missing & ~eligible if name in sparse_when_ineligible else None
        if missing_for_validation.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                f"missing_values_{name}",
                f"required field {name} is incomplete",
                count=int(missing_for_validation.sum()),
                rate=float(missing_for_validation.mean()),
            )
        if ignored is not None and ignored.any():
            _add(
                findings,
                AuditSeverity.WARN,
                f"missing_values_on_research_ineligible_{name}",
                f"{name} is missing only where the row is excluded from research",
                count=int(ignored.sum()),
                rate=float(ignored.mean()),
            )
    proxy = coverage["implied_float_mcap"]
    if proxy["present"] and float(proxy["rate"]) < config.minimum_proxy_mcap_coverage:
        _add(
            findings,
            AuditSeverity.FAIL,
            "proxy_mcap_coverage_below_minimum",
            "implied float-cap proxy coverage is below the registered minimum",
            count=int(proxy["missing"]),
            rate=float(proxy["rate"]),
        )
    elif proxy["present"] and int(proxy["missing"]) > 0:
        _add(
            findings,
            AuditSeverity.WARN,
            "proxy_mcap_incomplete",
            "some rows use no observed implied float capitalization",
            count=int(proxy["missing"]),
            rate=1.0 - float(proxy["rate"]),
        )

    for name in (
        "tradable",
        "limit_up",
        "limit_down",
        "limit_proxy_known",
        "research_close_limit_up_proxy",
        "research_close_limit_down_proxy",
        "research_close_limit_known_proxy",
        "is_st",
        "universe_member",
        "vendor_universe_member",
        "research_eligible",
    ):
        if name not in panel:
            continue
        values = panel[name]
        invalid = values.isna() | ~values.isin([True, False, 0, 1])
        if invalid.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                f"boolean_domain_invalid_{name}",
                f"boolean field {name} contains null or non-boolean values",
                count=int(invalid.sum()),
                samples=_sample_index(panel.index[invalid], config.maximum_samples),
                maximum_samples=config.maximum_samples,
            )

    if {"benchmark_weight", "universe_member"}.issubset(panel.columns):
        weights = pd.to_numeric(panel["benchmark_weight"], errors="coerce")
        member = panel["universe_member"].fillna(False).astype(bool)
        bad_domain = ~np.isfinite(weights) | weights.lt(0) | (~member & weights.ne(0))
        if bad_domain.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "benchmark_weight_domain_invalid",
                "benchmark weights must be finite, non-negative and zero off-universe",
                count=int(bad_domain.sum()),
            )
        sums = weights.where(member, 0.0).groupby(level="date").sum()
        bad_sums = sums.sub(1.0).abs().gt(1e-4)
        if bad_sums.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "benchmark_weights_not_one",
                "daily member benchmark weights do not sum to one",
                count=int(bad_sums.sum()),
                samples=[f"{date.date()}={value:.8f}" for date, value in sums[bad_sums].items()],
                maximum_samples=config.maximum_samples,
            )

    for name in ("listed_days", "adv20_dollars"):
        if name in panel:
            numeric = pd.to_numeric(panel[name], errors="coerce")
            invalid = ~np.isfinite(numeric) | numeric.lt(0)
            if name == "adv20_dollars":
                invalid = invalid & eligible
            if invalid.any():
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    f"numeric_domain_invalid_{name}",
                    f"{name} must be finite and non-negative",
                    count=int(invalid.sum()),
                )


def _audit_raw_adjusted_contract(
    panel: pd.DataFrame,
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    needed = {*RAW_PRICE_COLUMNS, *ADJUSTED_PRICE_COLUMNS, "adj_factor", "volume", "amount"}
    if not needed.issubset(panel.columns):
        return
    numeric = panel[list(needed)].apply(pd.to_numeric, errors="coerce")
    invalid_counts: dict[str, int] = {}
    for name in (*RAW_PRICE_COLUMNS, *ADJUSTED_PRICE_COLUMNS, "adj_factor"):
        invalid = ~np.isfinite(numeric[name]) | numeric[name].le(0)
        if invalid.any():
            invalid_counts[name] = int(invalid.sum())
    for name in ("volume", "amount"):
        invalid = ~np.isfinite(numeric[name]) | numeric[name].lt(0)
        if invalid.any():
            invalid_counts[name] = int(invalid.sum())
    if invalid_counts:
        _add(
            findings,
            AuditSeverity.FAIL,
            "raw_adjusted_numeric_domain_invalid",
            "raw/adjusted prices, factor or liquidity fields violate their numeric domain",
            count=sum(invalid_counts.values()),
            samples=[f"{key}={value}" for key, value in sorted(invalid_counts.items())],
            maximum_samples=config.maximum_samples,
        )

    mismatch_by_field: dict[str, int] = {}
    factor = numeric["adj_factor"].to_numpy(dtype=float)
    for raw_name, adjusted_name in RESEARCH_PRICE_MAP.items():
        raw = numeric[raw_name].to_numpy(dtype=float)
        adjusted = numeric[adjusted_name].to_numpy(dtype=float)
        matches = np.isclose(
            adjusted,
            raw * factor,
            rtol=config.adjusted_price_tolerance,
            atol=config.adjusted_price_tolerance,
            equal_nan=False,
        )
        if not matches.all():
            mismatch_by_field[adjusted_name] = int((~matches).sum())
    metrics["raw_adjusted_contract"] = {
        "tolerance": config.adjusted_price_tolerance,
        "mismatch_by_field": mismatch_by_field,
    }
    if mismatch_by_field:
        _add(
            findings,
            AuditSeverity.FAIL,
            "adjusted_price_factor_mismatch",
            "adjusted prices are not raw prices multiplied by one common daily factor",
            count=sum(mismatch_by_field.values()),
            samples=[f"{key}={value}" for key, value in sorted(mismatch_by_field.items())],
            maximum_samples=config.maximum_samples,
        )

    for prefix in ("", "adj_"):
        high = numeric[f"{prefix}high"]
        low = numeric[f"{prefix}low"]
        open_ = numeric[f"{prefix}open"]
        close = numeric[f"{prefix}close"]
        invalid = high.lt(pd.concat([open_, close], axis=1).max(axis=1)) | low.gt(
            pd.concat([open_, close], axis=1).min(axis=1)
        ) | high.lt(low)
        if invalid.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                f"{prefix or 'raw_'}ohlc_inconsistent",
                f"{prefix or 'raw '}OHLC ordering is inconsistent",
                count=int(invalid.sum()),
                samples=_sample_index(panel.index[invalid], config.maximum_samples),
                maximum_samples=config.maximum_samples,
            )

    traded = numeric["volume"].gt(0)
    expected_vwap = numeric["amount"].div(numeric["volume"].where(traded)).where(
        traded, numeric["close"]
    )
    vwap_match = np.isclose(
        numeric["vwap"], expected_vwap, rtol=1e-9, atol=1e-9, equal_nan=False
    )
    if not vwap_match.all():
        _add(
            findings,
            AuditSeverity.FAIL,
            "raw_vwap_not_amount_over_volume",
            "raw VWAP is not amount/volume (or close on a no-trade placeholder)",
            count=int((~vwap_match).sum()),
            samples=_sample_index(panel.index[~vwap_match], config.maximum_samples),
            maximum_samples=config.maximum_samples,
        )


def _audit_temporal_and_proxy_fields(
    panel: pd.DataFrame,
    manifest: Mapping[str, object],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    _add(
        findings,
        AuditSeverity.WARN,
        "research_proxy_not_portfolio_grade",
        "free BaoStock snapshot remains research-only; capital deployment is not allowed",
    )
    validation = manifest.get("validation")
    validation_meta = validation.get("metadata") if isinstance(validation, Mapping) else None
    if manifest.get("research_proxy_only") is not True:
        _add(
            findings,
            AuditSeverity.FAIL,
            "research_proxy_declaration_missing",
            "free BaoStock snapshot is not explicitly marked research_proxy_only",
        )
    if not isinstance(validation_meta, Mapping) or validation_meta.get("dataset_tier") != "research_proxy":
        _add(
            findings,
            AuditSeverity.FAIL,
            "dataset_tier_provenance_invalid",
            "build validation metadata does not preserve the research_proxy tier",
        )
    watermark = str(manifest.get("research_watermark", "")).strip()
    if not watermark or "RESEARCH_PROXY_ONLY" not in watermark:
        _add(
            findings,
            AuditSeverity.FAIL,
            "research_watermark_missing",
            "manifest lost the permanent research-only watermark",
        )
    if "research_watermark" in panel:
        panel_watermarks = panel["research_watermark"].astype("string")
        wrong = panel_watermarks.isna() | ~panel_watermarks.str.contains(
            "RESEARCH_PROXY_ONLY", regex=False, na=False
        )
        if wrong.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "panel_research_watermark_missing",
                "one or more panel rows lost the research-only watermark",
                count=int(wrong.sum()),
            )
    else:
        _add(
            findings,
            AuditSeverity.FAIL,
            "panel_research_watermark_column_missing",
            "panel has no row-level research-only watermark",
        )
    for column, code in (
        ("membership_snapshot_date", "future_membership_snapshot"),
        ("industry_snapshot_date", "future_industry_snapshot"),
    ):
        if column not in panel:
            continue
        source = pd.to_datetime(panel[column], errors="coerce", format="mixed")
        invalid_source = source.isna()
        if invalid_source.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                f"invalid_{column}",
                f"{column} contains null or unparsable dates",
                count=int(invalid_source.sum()),
                samples=_sample_index(panel.index[invalid_source], config.maximum_samples),
                maximum_samples=config.maximum_samples,
            )
        future = source.to_numpy(dtype="datetime64[ns]") > dates.to_numpy(dtype="datetime64[ns]")
        if future.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                code,
                f"{column} contains future information",
                count=int(future.sum()),
                samples=_sample_index(panel.index[future], config.maximum_samples),
                maximum_samples=config.maximum_samples,
            )
    if "industry" in panel:
        blank = _blank_mask(panel["industry"])
        metrics["industry"] = {
            "coverage": 1.0 - float(blank.mean()) if len(blank) else 0.0,
            "distinct_industries": int(panel.loc[~blank, "industry"].nunique()),
        }
    if "industry_snapshot_date" in panel:
        source = pd.to_datetime(
            panel["industry_snapshot_date"], errors="coerce", format="mixed"
        )
        ages = (pd.Series(dates, index=panel.index) - source).dt.days
        raw_build_config = manifest.get("config")
        maximum_age = 35
        if isinstance(raw_build_config, Mapping):
            maximum_age = int(raw_build_config.get("maximum_industry_age_days", 35))
        stale = ages.gt(maximum_age)
        metrics.setdefault("industry", {})["maximum_snapshot_age_days"] = (
            int(ages.max()) if ages.notna().any() else None
        )
        if stale.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "industry_snapshot_stale",
                "industry as-of snapshot exceeds the build-time maximum age",
                count=int(stale.sum()),
                samples=_sample_index(panel.index[stale], config.maximum_samples),
                maximum_samples=config.maximum_samples,
            )

    if "implied_float_mcap" in panel:
        mcap = pd.to_numeric(panel["implied_float_mcap"], errors="coerce")
        positive = np.isfinite(mcap) & mcap.gt(0)
        metrics["mcap_proxy"] = {
            "positive_rows": int(positive.sum()),
            "coverage": float(positive.mean()) if len(positive) else 0.0,
        }
        invalid_observed = mcap.notna() & ~positive
        if invalid_observed.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "proxy_mcap_nonpositive",
                "observed implied float capitalization must be finite and positive",
                count=int(invalid_observed.sum()),
            )
        if "log_mcap" in panel:
            log_mcap = pd.to_numeric(panel["log_mcap"], errors="coerce")
            comparable = positive & np.isfinite(log_mcap)
            matches = np.isclose(
                log_mcap[comparable],
                np.log(mcap[comparable]),
                rtol=1e-10,
                atol=1e-10,
                equal_nan=False,
            )
            if not matches.all():
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "log_mcap_proxy_mismatch",
                    "log_mcap is inconsistent with the implied float-cap proxy",
                    count=int((~matches).sum()),
                )
    if "benchmark_weight_source" in panel:
        fallback = panel["benchmark_weight_source"].astype(str).str.contains(
            "equal_weight_fallback", regex=False
        )
        metrics.setdefault("mcap_proxy", {})["equal_weight_fallback_rows"] = int(fallback.sum())
        if fallback.any():
            _add(
                findings,
                AuditSeverity.WARN,
                "equal_weight_fallback_used",
                "proxy benchmark weights fell back to equal weights",
                count=int(fallback.sum()),
                rate=float(fallback.mean()),
            )

    if "tradable" in panel:
        tradable = panel["tradable"].fillna(False).astype(bool)
        suspended = ~tradable
        by_date = suspended.groupby(level="date").mean()
        metrics["suspensions"] = {
            "rows": int(suspended.sum()),
            "rate": float(suspended.mean()),
            "dates_with_suspensions": int(suspended.groupby(level="date").any().sum()),
            "maximum_daily_rate": float(by_date.max()) if len(by_date) else 0.0,
        }
        high = by_date.gt(config.suspension_daily_warning_rate)
        if high.any():
            _add(
                findings,
                AuditSeverity.WARN,
                "high_daily_suspension_rate",
                "one or more dates exceed the registered suspension-rate review threshold",
                count=int(high.sum()),
                rate=float(by_date.max()),
                samples=[f"{date.date()}={value:.3%}" for date, value in by_date[high].items()],
                maximum_samples=config.maximum_samples,
            )
    else:
        tradable = pd.Series(False, index=panel.index)
        suspended = ~tradable

    if "limit_proxy_known" in panel:
        known = panel["limit_proxy_known"].fillna(False).astype(bool)
        unknown = ~known
        unknown_tradable = unknown & tradable
        metrics["limit_proxy"] = {
            "known_rows": int(known.sum()),
            "unknown_rows": int(unknown.sum()),
            "unknown_rate": float(unknown.mean()),
            "unknown_tradable_rows": int(unknown_tradable.sum()),
            "unknown_suspended_rows": int((unknown & suspended).sum()),
        }
        if unknown_tradable.any():
            _add(
                findings,
                AuditSeverity.WARN,
                "limit_proxy_unknown_on_tradable_rows",
                "daily-bar price-limit status is unknown on otherwise tradable rows",
                count=int(unknown_tradable.sum()),
                rate=float(unknown_tradable.mean()),
                samples=_sample_index(panel.index[unknown_tradable], config.maximum_samples),
                maximum_samples=config.maximum_samples,
            )
        if {"limit_up", "limit_down"}.issubset(panel.columns):
            flagged_unknown = unknown & (
                panel["limit_up"].fillna(False).astype(bool)
                | panel["limit_down"].fillna(False).astype(bool)
            )
            if flagged_unknown.any():
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "unknown_limit_rows_flagged_as_locked",
                    "a limit lock was asserted where the proxy rule was unknown",
                    count=int(flagged_unknown.sum()),
                )


def _audit_adjusted_returns(
    panel: pd.DataFrame,
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    if "adj_close" not in panel:
        return
    unique_dates = pd.DatetimeIndex(
        panel.index.get_level_values("date").unique()
    ).sort_values()
    ordinal_map = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    work = panel[["adj_close"]].copy()
    work["date_ordinal"] = panel.index.get_level_values("date").map(ordinal_map)
    work["previous_ordinal"] = work["date_ordinal"].groupby(level="symbol").shift(1)
    work["return"] = pd.to_numeric(work["adj_close"], errors="coerce").groupby(
        level="symbol"
    ).pct_change(fill_method=None)
    consecutive = work["date_ordinal"].sub(work["previous_ordinal"]).eq(1)
    returns = work["return"].where(consecutive)
    absolute = returns.abs()
    fail = absolute.gt(config.adjusted_return_fail_threshold)
    warn = absolute.gt(config.adjusted_return_warning_threshold) & ~fail
    finite = returns.notna() & np.isfinite(returns)
    quantiles = absolute[finite].quantile([0.50, 0.95, 0.99, 0.999]) if finite.any() else pd.Series(dtype=float)
    metrics["adjusted_returns"] = {
        "consecutive_observations": int(finite.sum()),
        "warning_threshold": config.adjusted_return_warning_threshold,
        "fail_threshold": config.adjusted_return_fail_threshold,
        "warning_rows": int(warn.sum()),
        "fail_rows": int(fail.sum()),
        "maximum_absolute_return": float(absolute.max()) if finite.any() else None,
        "absolute_return_quantiles": {str(key): float(value) for key, value in quantiles.items()},
    }
    if fail.any():
        samples = [
            f"{pd.Timestamp(date).date()}|{symbol}={returns.loc[(date, symbol)]:.4f}"
            for date, symbol in panel.index[fail][: config.maximum_samples]
        ]
        _add(
            findings,
            AuditSeverity.FAIL,
            "extreme_adjusted_returns",
            "adjusted close returns exceed the hard data-integrity threshold",
            count=int(fail.sum()),
            samples=samples,
            maximum_samples=config.maximum_samples,
        )
    if warn.any():
        samples = [
            f"{pd.Timestamp(date).date()}|{symbol}={returns.loc[(date, symbol)]:.4f}"
            for date, symbol in panel.index[warn][: config.maximum_samples]
        ]
        _add(
            findings,
            AuditSeverity.WARN,
            "suspicious_adjusted_returns",
            "adjusted close returns exceed the manual-review threshold",
            count=int(warn.sum()),
            samples=samples,
            maximum_samples=config.maximum_samples,
        )


def _audit_membership_industry_lineage(
    panel: pd.DataFrame,
    manifest: Mapping[str, object],
    normalized_frames: Mapping[str, pd.DataFrame],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    """Verify that panel membership and industries are exact catalog as-of joins."""

    membership = normalized_frames.get("membership_daily")
    if membership is None or not {"date", "code", "vendor_update_date"}.issubset(
        membership.columns
    ):
        _add(
            findings,
            AuditSeverity.FAIL,
            "membership_lineage_unavailable",
            "membership_daily cannot prove the panel universe lineage",
        )
        return
    members = membership[["date", "code", "vendor_update_date"]].copy()
    members["date"] = pd.to_datetime(members["date"], errors="coerce").dt.normalize()
    members["vendor_update_date"] = pd.to_datetime(
        members["vendor_update_date"], errors="coerce"
    ).dt.normalize()
    members["code"] = members["code"].astype(str)
    first = panel.index.get_level_values("date").min()
    last = panel.index.get_level_values("date").max()
    members = members.loc[members["date"].between(first, last)].sort_values(["date", "code"])
    expected_index = pd.MultiIndex.from_frame(
        members[["date", "code"]].rename(columns={"code": "symbol"}),
        names=["date", "symbol"],
    )
    try:
        terminal_exceptions = _validated_terminal_exception_sidecar(
            manifest, config.panel_dir
        )
        expected_index = expected_index.difference(terminal_exceptions.index)
    except Exception as exc:
        _add(
            findings,
            AuditSeverity.FAIL,
            "terminal_stale_membership_sidecar_invalid",
            f"terminal-stale membership exception audit failed: {exc}",
        )
    exact_membership = expected_index.equals(panel.index)
    metrics["source_lineage"] = {
        "membership_exact": exact_membership,
        "membership_rows": len(members),
        "terminal_exception_rows": len(members) - len(expected_index),
    }
    if not exact_membership:
        missing = expected_index.difference(panel.index)
        extra = panel.index.difference(expected_index)
        _add(
            findings,
            AuditSeverity.FAIL,
            "panel_membership_lineage_mismatch",
            "panel rows do not equal membership_daily minus exact terminal exceptions",
            count=len(missing) + len(extra),
            samples=[
                *[f"missing:{item}" for item in _sample_index(missing, config.maximum_samples)],
                *[f"extra:{item}" for item in _sample_index(extra, config.maximum_samples)],
            ],
            maximum_samples=config.maximum_samples,
        )
    else:
        actual_snapshot = pd.to_datetime(
            panel["membership_snapshot_date"], errors="coerce", format="mixed"
        ).to_numpy(dtype="datetime64[ns]")
        expected_snapshot = (
            members.rename(columns={"code": "symbol"})
            .set_index(["date", "symbol"])["vendor_update_date"]
            .reindex(panel.index)
            .to_numpy(dtype="datetime64[ns]")
        )
        mismatch = actual_snapshot != expected_snapshot
        if mismatch.any():
            _add(
                findings,
                AuditSeverity.FAIL,
                "membership_snapshot_lineage_mismatch",
                "panel membership snapshot dates differ from membership_daily",
                count=int(mismatch.sum()),
                samples=_sample_index(panel.index[mismatch], config.maximum_samples),
                maximum_samples=config.maximum_samples,
            )

    industry = normalized_frames.get("industry_snapshots")
    required = {"requested_date", "vendor_update_date", "code", "industry"}
    if industry is None or not required.issubset(industry.columns):
        _add(
            findings,
            AuditSeverity.FAIL,
            "industry_lineage_unavailable",
            "industry_snapshots cannot prove the panel industry lineage",
        )
        return
    source = industry[list(required)].copy()
    for name in ("requested_date", "vendor_update_date"):
        source[name] = pd.to_datetime(
            source[name], errors="coerce"
        ).dt.normalize()
    source["code"] = source["code"].astype(str)
    source["industry"] = source["industry"].astype("string").str.strip()
    if (
        source[["requested_date", "vendor_update_date"]].isna().any().any()
        or (source["vendor_update_date"] > source["requested_date"]).any()
        or source[["requested_date", "code"]].duplicated().any()
    ):
        _add(
            findings,
            AuditSeverity.FAIL,
            "industry_catalog_duplicate_keys",
            "industry catalog contains invalid observation-symbol keys or future metadata",
        )
        return
    date_mismatch_count = 0
    value_mismatch_count = 0
    lineage_samples: list[str] = []
    for symbol, block in panel.groupby(level="symbol", sort=False):
        history = source.loc[source["code"].eq(str(symbol))].sort_values(
            "requested_date"
        )
        target_dates = pd.DatetimeIndex(block.index.get_level_values("date"))
        if history.empty:
            date_mismatch_count += len(block)
            lineage_samples.append(f"missing-history:{symbol}")
            continue
        source_dates = pd.DatetimeIndex(history["requested_date"])
        locations = source_dates.searchsorted(target_dates, side="right") - 1
        invalid = locations < 0
        safe_locations = np.maximum(locations, 0)
        expected_dates = source_dates[safe_locations]
        actual_dates = pd.to_datetime(
            block["industry_snapshot_date"], errors="coerce", format="mixed"
        ).to_numpy(dtype="datetime64[ns]")
        date_bad = invalid | (actual_dates != expected_dates.to_numpy(dtype="datetime64[ns]"))
        expected_values = history.iloc[safe_locations]["industry"].astype("string").to_numpy()
        actual_values = block["industry"].astype("string").str.strip().to_numpy()
        value_bad = invalid | (actual_values != expected_values)
        date_mismatch_count += int(date_bad.sum())
        value_mismatch_count += int(value_bad.sum())
        if (date_bad | value_bad).any() and len(lineage_samples) < config.maximum_samples:
            first_bad = int(np.flatnonzero(date_bad | value_bad)[0])
            lineage_samples.append(f"{target_dates[first_bad].date()}|{symbol}")
    metrics["source_lineage"].update(
        {
            "industry_snapshot_date_mismatches": date_mismatch_count,
            "industry_value_mismatches": value_mismatch_count,
        }
    )
    if date_mismatch_count or value_mismatch_count:
        _add(
            findings,
            AuditSeverity.FAIL,
            "panel_industry_lineage_mismatch",
            "panel industry values or snapshot dates differ from the latest frozen as-of catalog",
            count=date_mismatch_count + value_mismatch_count,
            samples=lineage_samples,
            maximum_samples=config.maximum_samples,
        )


def _audit_cache_states(
    cache_dir: Path,
    records: Sequence[Mapping[str, object]],
    resolved_paths: Mapping[tuple[str, str], Path],
    manifest: Mapping[str, object],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    failures = 0
    samples: list[str] = []
    verified = 0
    build_policy = manifest.get("config")
    if isinstance(build_policy, Mapping):
        source_start = str(
            pd.Timestamp(build_policy.get("membership_source_start_date")).date()
        )
        source_end = str(
            pd.Timestamp(build_policy.get("membership_source_end_date")).date()
        )
        adjustment_start = str(
            pd.Timestamp(build_policy.get("adjustment_event_start_date")).date()
        )
    else:
        source_start = source_end = adjustment_start = ""
    for record in records:
        explicit_endpoint = str(record.get("endpoint", "")).strip()
        if explicit_endpoint:
            key = str(record.get("request_key", ""))
            identity = (f"__{explicit_endpoint}__", key)
            path = resolved_paths.get(identity)
            expected_fields_by_endpoint = {
                "trade_dates": ("calendar_date", "is_trading_day"),
                "hs300_stocks": ("updateDate", "code", "code_name"),
                "stock_basic": (
                    "code",
                    "code_name",
                    "ipoDate",
                    "outDate",
                    "type",
                    "status",
                ),
                "stock_industry": (
                    "updateDate",
                    "code",
                    "code_name",
                    "industry",
                    "industryClassification",
                ),
            }
            expected_fields = expected_fields_by_endpoint.get(explicit_endpoint)
            state_path = cache_dir / "state" / f"{key}.json"
            try:
                state = _read_json(state_path)
                args = state.get("args")
                declared_args = record.get("args")
                state_object = _resolve_cache_object(cache_dir, state.get("object"))
                valid = (
                    expected_fields is not None
                    and re.fullmatch(r"[0-9a-f]{64}", key) is not None
                    and state.get("state_schema_version") == 1
                    and state.get("request_key") == key
                    and state.get("status") == "success"
                    and state.get("endpoint") == explicit_endpoint
                    and isinstance(args, Mapping)
                    and isinstance(declared_args, Mapping)
                    and dict(args) == dict(declared_args)
                    and make_request_key(
                        explicit_endpoint,
                        args,
                        provider_version=str(state.get("provider_version", "")),
                    )
                    == key
                    and tuple(state.get("fields", ())) == tuple(expected_fields)
                    and int(state.get("rows", -1)) == int(record.get("rows", -1))
                    and state.get("parquet_sha256") == record.get("parquet_sha256")
                    and state_object == path
                )
            except Exception:
                valid = False
            if valid:
                verified += 1
            else:
                failures += 1
                if len(samples) < config.maximum_samples:
                    samples.append(f"{explicit_endpoint}/{key}")
            continue
        is_adjustment = "adjustflag" not in record and "code" in record
        is_index = "adjustflag" not in record and not is_adjustment
        code = str(record.get("code", "sh.000300" if is_index else ""))
        flag = (
            "adjustment"
            if is_adjustment
            else ("index" if is_index else str(record.get("adjustflag", "")))
        )
        key = str(record.get("request_key", ""))
        path = resolved_paths.get((code, flag))
        expected_fields = (
            ADJUST_FACTOR_FIELDS
            if is_adjustment
            else (INDEX_DAILY_FIELDS if is_index else EQUITY_DAILY_FIELDS)
        )
        expected_endpoint = (
            "adjust_factor"
            if is_adjustment
            else ("index_history" if is_index else "history_k_data")
        )
        state_path = cache_dir / "state" / f"{key}.json"
        try:
            state = _read_json(state_path)
            args = state.get("args")
            declared_args = record.get("args")
            state_object = _resolve_cache_object(cache_dir, state.get("object"))
            valid = (
                re.fullmatch(r"[0-9a-f]{64}", key) is not None
                and state.get("state_schema_version") == 1
                and state.get("request_key") == key
                and state.get("status") == "success"
                and state.get("endpoint") == expected_endpoint
                and make_request_key(
                    expected_endpoint,
                    args if isinstance(args, Mapping) else {},
                    provider_version=str(state.get("provider_version", "")),
                )
                == key
                and tuple(state.get("fields", ())) == tuple(expected_fields)
                and int(state.get("rows", -1)) == int(record.get("rows", -1))
                and state.get("parquet_sha256") == record.get("parquet_sha256")
                and state_object == path
                and isinstance(args, Mapping)
                and isinstance(declared_args, Mapping)
                and dict(args) == dict(declared_args)
                and str(args.get("code")) == code
                and (
                    (
                        is_adjustment
                        and pd.notna(
                            pd.to_datetime(args.get("start_date"), errors="coerce")
                        )
                        and pd.notna(
                            pd.to_datetime(args.get("end_date"), errors="coerce")
                        )
                        and str(args.get("start_date")) == adjustment_start
                        and str(args.get("end_date")) == source_end
                    )
                    or (
                        not is_adjustment
                        and str(args.get("frequency")) == "d"
                        and str(args.get("fields")) == ",".join(expected_fields)
                        and (is_index or str(args.get("adjustflag")) == flag)
                        and str(args.get("start_date")) == source_start
                        and str(args.get("end_date")) == source_end
                    )
                )
            )
        except Exception:
            valid = False
        if valid:
            verified += 1
        else:
            failures += 1
            if len(samples) < config.maximum_samples:
                samples.append(f"{code}/{flag}/{key}")
    metrics.setdefault("source_lineage", {})["cache_states_verified"] = verified
    metrics["source_lineage"]["cache_state_failures"] = failures
    if failures:
        _add(
            findings,
            AuditSeverity.FAIL,
            "cache_state_contract_mismatch",
            "cache state, endpoint, args, schema, rows or object lineage is inconsistent",
            count=failures,
            samples=samples,
            maximum_samples=config.maximum_samples,
        )


def _audit_vendor_adjustment_event_lineage(
    manifest: Mapping[str, object],
    normalized_frames: Mapping[str, pd.DataFrame],
    resolved_paths: Mapping[tuple[str, str], Path],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    """Independently replay every catalogued vendor factor path, serially."""

    request_index = normalized_frames.get("adjustment_request_index")
    price_index = normalized_frames.get("price_request_index")
    membership = normalized_frames.get("membership_daily")
    manifest_records = manifest.get("adjustment_cache_objects")
    required = {
        "code",
        "rows",
        "first_event_date",
        "last_event_date",
        "request_key",
    }
    if (
        request_index is None
        or price_index is None
        or membership is None
        or not required.issubset(request_index.columns)
        or not {"code", "adjustflag"}.issubset(price_index.columns)
        or not isinstance(manifest_records, list)
    ):
        _add(
            findings,
            AuditSeverity.FAIL,
            "adjustment_event_catalog_lineage_unavailable",
            "frozen adjustment request catalog cannot be independently audited",
        )
        return
    catalog = request_index.copy()
    catalog["code"] = catalog["code"].astype(str)
    membership_boundaries = membership.loc[:, ["date", "code"]].copy()
    membership_boundaries["code"] = membership_boundaries["code"].astype(str)
    membership_boundaries["date"] = pd.to_datetime(
        membership_boundaries["date"], errors="coerce"
    ).dt.normalize()
    first_membership = membership_boundaries.groupby("code", sort=False)[
        "date"
    ].min()
    price_codes = set(price_index["code"].astype(str))
    if (
        catalog["code"].duplicated().any()
        or set(catalog["code"]) != price_codes
        or len(manifest_records) != len(catalog)
    ):
        _add(
            findings,
            AuditSeverity.FAIL,
            "adjustment_event_catalog_symbol_contract_mismatch",
            "adjustment catalog must exactly cover the historical price union once",
        )
        return
    by_code = {
        str(record.get("code")): record
        for record in manifest_records
        if isinstance(record, Mapping)
    }
    invalid: list[str] = []
    recurrence_diagnostic_failures = 0
    audited_rows = 0
    audited_events = 0
    factor_changes = 0
    future_events_ignored = 0
    maximum_common_error = 0.0
    maximum_event_error = 0.0
    for row in catalog.sort_values("code").itertuples(index=False):
        code = str(row.code)
        record = by_code.get(code)
        raw_path = resolved_paths.get((code, "3"))
        adjusted_path = resolved_paths.get((code, "1"))
        event_path = resolved_paths.get((code, "adjustment"))
        if record is None or raw_path is None or adjusted_path is None or event_path is None:
            invalid.append(f"{code}:missing_source")
            continue
        if code not in first_membership.index or pd.isna(first_membership.loc[code]):
            invalid.append(f"{code}:missing_membership_boundary")
            continue
        if (
            str(record.get("request_key", "")) != str(row.request_key)
            or int(record.get("rows", -1)) != int(row.rows)
        ):
            invalid.append(f"{code}:request_index_manifest_mismatch")
            continue
        try:
            raw_frame = pd.read_parquet(raw_path)
            adjusted_frame = pd.read_parquet(adjusted_path)
            events = pd.read_parquet(event_path)
            if (
                tuple(raw_frame.columns) != tuple(EQUITY_DAILY_FIELDS)
                or tuple(adjusted_frame.columns) != tuple(EQUITY_DAILY_FIELDS)
                or tuple(events.columns) != tuple(ADJUST_FACTOR_FIELDS)
                or len(events) != int(row.rows)
            ):
                raise ValueError("schema_or_row_count")
            raw_frame["date"] = pd.to_datetime(
                raw_frame["date"], errors="coerce"
            ).dt.normalize()
            adjusted_frame["date"] = pd.to_datetime(
                adjusted_frame["date"], errors="coerce"
            ).dt.normalize()
            raw_view = raw_frame.rename(columns={"code": "symbol"}).set_index(
                ["date", "symbol"]
            ).sort_index()
            adjusted_view = adjusted_frame.rename(
                columns={"code": "symbol"}
            ).set_index(["date", "symbol"]).sort_index()
            raw_view, adjusted_view, _ = trim_leading_unadjusted_vendor_history(
                raw_view,
                adjusted_view,
                pd.DataFrame(
                    {"date": [first_membership.loc[code]], "code": [code]}
                ),
            )
            audit = audit_vendor_adjustment_events_asof(
                raw_view,
                adjusted_view,
                events,
                event_catalog_symbols=[code],
            )
        except Exception as exc:
            invalid.append(f"{code}:{type(exc).__name__}")
            continue
        if not audit.valid:
            invalid.append(f"{code}:{'|'.join(audit.errors[:2])}")
            continue
        audited_rows += audit.rows
        audited_events += audit.event_rows
        factor_changes += audit.factor_changes
        future_events_ignored += audit.future_event_rows_ignored
        maximum_common_error = max(
            maximum_common_error, audit.maximum_common_scale_relative_error
        )
        maximum_event_error = max(
            maximum_event_error, audit.maximum_event_path_relative_error
        )
        recurrence_diagnostic_failures += int(
            not audit.raw_preclose_recurrence_valid
        )
    audit_metrics = {
        "catalog_symbols": len(catalog),
        "audited_symbols": len(catalog) - len(invalid),
        "rows": audited_rows,
        "event_rows": audited_events,
        "factor_changes": factor_changes,
        "future_event_rows_ignored": future_events_ignored,
        "maximum_common_scale_relative_error": maximum_common_error,
        "maximum_event_path_relative_error": maximum_event_error,
        "raw_preclose_recurrence_diagnostic_failures": (
            recurrence_diagnostic_failures
        ),
    }
    metrics["vendor_adjustment_event_lineage"] = audit_metrics
    if invalid:
        _add(
            findings,
            AuditSeverity.FAIL,
            "vendor_adjustment_event_path_invalid",
            "one or more flag1 paths fail the independently frozen vendor-event audit",
            count=len(invalid),
            samples=invalid,
            maximum_samples=config.maximum_samples,
        )
    declared = manifest.get("adjustment_event_audit")
    if (
        not isinstance(declared, Mapping)
        or declared.get("valid") is not True
        or tuple(declared.get("errors", ()))
        or int(declared.get("symbols", -1)) != len(catalog)
        or int(declared.get("rows", -1)) != audited_rows
        or int(declared.get("event_rows", -1)) != audited_events
        or int(declared.get("factor_changes", -1)) != factor_changes
        or int(declared.get("future_event_rows_ignored", -1))
        != future_events_ignored
        or not np.isclose(
            float(declared.get("maximum_common_scale_relative_error", np.inf)),
            maximum_common_error,
            rtol=1e-12,
            atol=1e-15,
        )
        or not np.isclose(
            float(declared.get("maximum_event_path_relative_error", np.inf)),
            maximum_event_error,
            rtol=1e-12,
            atol=1e-15,
        )
    ):
        _add(
            findings,
            AuditSeverity.FAIL,
            "vendor_adjustment_event_manifest_mismatch",
            "manifested event audit does not equal the independent cache replay",
        )


def _numeric_mismatch(
    actual: pd.Series,
    expected: pd.Series,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> np.ndarray:
    left = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
    return ~np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)


def _audit_price_and_derived_lineage(
    panel: pd.DataFrame,
    manifest: Mapping[str, object],
    normalized_frames: Mapping[str, pd.DataFrame],
    security: pd.DataFrame,
    resolved_paths: Mapping[tuple[str, str], Path],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    """Stream each symbol cache pair and reproduce registered panel fields."""

    build_config = manifest.get("config")
    if not isinstance(build_config, Mapping):
        _add(findings, AuditSeverity.FAIL, "build_policy_missing", "manifest has no build policy")
        return
    adv_window = int(build_config.get("adv_window", 20))
    raw_adv_min = build_config.get("adv_min_periods")
    adv_min = adv_window if raw_adv_min is None else int(raw_adv_min)
    beta_window = int(build_config.get("beta_window", 252))
    beta_min = int(build_config.get("beta_min_periods", 60))

    index_path = resolved_paths.get(("sh.000300", "index"))
    if index_path is None:
        _add(findings, AuditSeverity.FAIL, "benchmark_cache_missing", "HS300 index cache is unavailable")
        return
    index_frame = pd.read_parquet(index_path)
    index_dates = pd.to_datetime(index_frame["date"], errors="coerce").dt.normalize()
    index_close = pd.to_numeric(index_frame["close"], errors="coerce")
    benchmark_returns = pd.Series(
        index_close.to_numpy(dtype=float),
        index=pd.DatetimeIndex(index_dates, name="date"),
    ).sort_index().pct_change(fill_method=None)

    basic = security.copy()
    basic["code"] = basic["code"].astype(str)
    ipo_map = pd.to_datetime(
        basic.set_index("code")["ipoDate"], errors="coerce"
    ).dt.normalize()
    price_index = normalized_frames.get("price_request_index")
    if price_index is None:
        _add(
            findings,
            AuditSeverity.FAIL,
            "price_request_index_lineage_unavailable",
            "price_request_index is unavailable for semantic lineage",
        )
        return
    registered = price_index.copy()
    registered["code"] = registered["code"].astype(str)
    registered["adjustflag"] = registered["adjustflag"].astype(str)
    membership = normalized_frames.get("membership_daily")
    if membership is None:
        _add(
            findings,
            AuditSeverity.FAIL,
            "membership_lineage_unavailable_for_price_boundaries",
            "membership_daily is required to replay mixed adjusted-history boundaries",
        )
        return
    membership_boundaries = membership.loc[:, ["date", "code"]].copy()
    membership_boundaries["code"] = membership_boundaries["code"].astype(str)
    membership_boundaries["date"] = pd.to_datetime(
        membership_boundaries["date"], errors="coerce"
    ).dt.normalize()
    first_membership = membership_boundaries.groupby("code", sort=False)[
        "date"
    ].min()

    mismatches: dict[str, int] = {}
    samples: list[str] = []
    entry_return_failures = 0
    entry_return_warnings = 0
    entry_return_samples: list[str] = []
    panel_dates = pd.DatetimeIndex(
        panel.index.get_level_values("date").unique()
    ).sort_values()
    panel_date_ordinal = pd.Series(np.arange(len(panel_dates)), index=panel_dates)

    def record(field: str, mask: np.ndarray, index: pd.Index, symbol: str) -> None:
        count = int(np.asarray(mask, dtype=bool).sum())
        if not count:
            return
        mismatches[field] = mismatches.get(field, 0) + count
        if len(samples) < config.maximum_samples:
            location = int(np.flatnonzero(mask)[0])
            samples.append(f"{pd.Timestamp(index[location]).date()}|{symbol}|{field}")

    audited_symbols = 0
    for symbol, block in panel.groupby(level="symbol", sort=False):
        symbol = str(symbol)
        raw_path = resolved_paths.get((symbol, "3"))
        adjusted_path = resolved_paths.get((symbol, "1"))
        if raw_path is None or adjusted_path is None:
            mismatches["missing_cache_pair"] = mismatches.get("missing_cache_pair", 0) + len(block)
            continue
        if symbol not in first_membership.index or pd.isna(first_membership.loc[symbol]):
            mismatches["missing_membership_boundary"] = mismatches.get(
                "missing_membership_boundary", 0
            ) + len(block)
            continue
        raw_frame = pd.read_parquet(raw_path)
        adjusted_frame = pd.read_parquet(adjusted_path)
        if tuple(raw_frame.columns) != tuple(EQUITY_DAILY_FIELDS) or tuple(
            adjusted_frame.columns
        ) != tuple(EQUITY_DAILY_FIELDS):
            mismatches["cache_schema"] = mismatches.get("cache_schema", 0) + 1
            continue
        raw_frame["date"] = pd.to_datetime(raw_frame["date"], errors="coerce").dt.normalize()
        adjusted_frame["date"] = pd.to_datetime(
            adjusted_frame["date"], errors="coerce"
        ).dt.normalize()
        raw_view = raw_frame.rename(columns={"code": "symbol"}).set_index(
            ["date", "symbol"]
        ).sort_index()
        adjusted_view = adjusted_frame.rename(columns={"code": "symbol"}).set_index(
            ["date", "symbol"]
        ).sort_index()
        if not raw_view.index.equals(adjusted_view.index):
            mismatches["raw_adjusted_source_index"] = mismatches.get(
                "raw_adjusted_source_index", 0
            ) + 1
            continue
        raw_view, adjusted_view, _ = trim_leading_unadjusted_vendor_history(
            raw_view,
            adjusted_view,
            pd.DataFrame(
                {"date": [first_membership.loc[symbol]], "code": [symbol]}
            ),
        )
        raw_view = normalize_baostock_daily_fields(raw_view)
        target_dates = pd.DatetimeIndex(block.index.get_level_values("date"))
        source_dates = pd.DatetimeIndex(raw_view.index.get_level_values("date"))
        if not target_dates.isin(source_dates).all():
            missing_count = int((~target_dates.isin(source_dates)).sum())
            mismatches["member_dates_missing_raw_source"] = mismatches.get(
                "member_dates_missing_raw_source", 0
            ) + missing_count
            continue
        source_raw = raw_view.droplevel("symbol").reindex(target_dates)
        source_adjusted = adjusted_view.droplevel("symbol").reindex(target_dates)
        for field in ("open", "high", "low", "close", "preclose", "volume", "amount"):
            if field in block and field in source_raw:
                record(field, _numeric_mismatch(block[field], source_raw[field]), target_dates, symbol)

        raw_close_full = pd.to_numeric(raw_view["close"], errors="coerce")
        adjusted_close_full = pd.to_numeric(adjusted_view["close"], errors="coerce")
        factor_full = adjusted_close_full.div(raw_close_full)
        factor = factor_full.droplevel("symbol").reindex(target_dates)
        record("adj_factor_source", _numeric_mismatch(block["adj_factor"], factor), target_dates, symbol)
        full_adjusted_return = adjusted_close_full.groupby(
            level="symbol", sort=False
        ).pct_change(fill_method=None)
        target_adjusted_return = full_adjusted_return.droplevel("symbol").reindex(
            target_dates
        )
        ordinals = target_dates.map(panel_date_ordinal).to_numpy(dtype=float)
        entry_or_reentry = np.r_[True, np.diff(ordinals) != 1]
        entry_absolute = target_adjusted_return.abs().to_numpy(dtype=float)
        hard_entry = entry_or_reentry & np.isfinite(entry_absolute) & (
            entry_absolute > config.adjusted_return_fail_threshold
        )
        review_entry = entry_or_reentry & np.isfinite(entry_absolute) & (
            entry_absolute > config.adjusted_return_warning_threshold
        ) & ~hard_entry
        entry_return_failures += int(hard_entry.sum())
        entry_return_warnings += int(review_entry.sum())
        if (hard_entry | review_entry).any() and len(entry_return_samples) < config.maximum_samples:
            location = int(np.flatnonzero(hard_entry | review_entry)[0])
            entry_return_samples.append(
                f"{target_dates[location].date()}|{symbol}={target_adjusted_return.iloc[location]:.4f}"
            )

        source_volume = pd.to_numeric(source_raw["volume"], errors="coerce")
        source_amount = pd.to_numeric(source_raw["amount"], errors="coerce")
        source_close = pd.to_numeric(source_raw["close"], errors="coerce")
        expected_vwap = source_amount.div(source_volume.where(source_volume.gt(0))).where(
            source_volume.gt(0), source_close
        )
        record("vwap_source", _numeric_mismatch(block["vwap"], expected_vwap), target_dates, symbol)
        expected_tradable = (
            pd.to_numeric(source_raw["tradestatus"], errors="coerce").eq(1)
            & source_volume.gt(0)
        )
        record(
            "tradable_source",
            block["tradable"].astype(bool).to_numpy() != expected_tradable.to_numpy(),
            target_dates,
            symbol,
        )
        expected_st = pd.to_numeric(source_raw["isST"], errors="coerce").eq(1)
        record(
            "is_st_source",
            block["is_st"].astype(bool).to_numpy() != expected_st.to_numpy(),
            target_dates,
            symbol,
        )

        full_amount = pd.to_numeric(raw_view["amount"], errors="coerce")
        expected_adv_full = full_amount.groupby(level="symbol", sort=False).transform(
            lambda values: values.rolling(adv_window, min_periods=adv_min).mean()
        )
        expected_adv = expected_adv_full.droplevel("symbol").reindex(target_dates)
        record("adv_source", _numeric_mismatch(block["adv20_dollars"], expected_adv), target_dates, symbol)

        full_volume = pd.to_numeric(raw_view["volume"], errors="coerce")
        full_turn = pd.to_numeric(raw_view["turn"].replace("", np.nan), errors="coerce")
        observed = full_volume.gt(0) & full_turn.gt(0)
        shares = full_volume.mul(100.0).div(full_turn).where(observed)
        shares = shares.groupby(level="symbol", sort=False).ffill()
        mcap = shares.mul(raw_close_full)
        expected_shares = shares.droplevel("symbol").reindex(target_dates)
        expected_mcap = mcap.droplevel("symbol").reindex(target_dates)
        record(
            "implied_float_shares_source",
            _numeric_mismatch(block["implied_float_shares"], expected_shares),
            target_dates,
            symbol,
        )
        record(
            "implied_float_mcap_source_value",
            _numeric_mismatch(block["implied_float_mcap"], expected_mcap),
            target_dates,
            symbol,
        )
        if symbol not in ipo_map or pd.isna(ipo_map.loc[symbol]):
            mismatches["ipo_date_source"] = mismatches.get("ipo_date_source", 0) + len(block)
        else:
            expected_listed = (target_dates - pd.Timestamp(ipo_map.loc[symbol])).days
            record(
                "listed_days_source",
                pd.to_numeric(block["listed_days"], errors="coerce").to_numpy(dtype=float)
                != expected_listed,
                target_dates,
                symbol,
            )

        expected_beta_full = derive_rolling_beta(
            adjusted_close_full,
            benchmark_returns,
            window=beta_window,
            min_periods=beta_min,
        )
        expected_beta = expected_beta_full.droplevel("symbol").reindex(target_dates)
        record(
            "beta_source",
            _numeric_mismatch(block["beta"], expected_beta, rtol=1e-9, atol=1e-10),
            target_dates,
            symbol,
        )
        audited_symbols += 1

        request_pair = registered.loc[registered["code"].eq(symbol)]
        for flag, path_frame, object_path in (
            ("3", raw_frame, raw_path),
            ("1", adjusted_frame, adjusted_path),
        ):
            row = request_pair.loc[request_pair["adjustflag"].eq(flag)]
            if len(row) != 1:
                mismatches["price_request_index_pair"] = mismatches.get(
                    "price_request_index_pair", 0
                ) + 1
                continue
            request = row.iloc[0]
            first_date = str(pd.Timestamp(path_frame["date"].min()).date())
            last_date = str(pd.Timestamp(path_frame["date"].max()).date())
            if (
                int(request["rows"]) != len(path_frame)
                or str(request["request_key"]) != object_path.stem
                or str(pd.Timestamp(request["first_date"]).date()) != first_date
                or str(pd.Timestamp(request["last_date"]).date()) != last_date
            ):
                mismatches["price_request_index_metadata"] = mismatches.get(
                    "price_request_index_metadata", 0
                ) + 1

    metrics.setdefault("source_lineage", {}).update(
        {
            "price_symbols_audited": audited_symbols,
            "price_and_derived_mismatches": dict(sorted(mismatches.items())),
            "entry_or_reentry_adjusted_return_failures": entry_return_failures,
            "entry_or_reentry_adjusted_return_warnings": entry_return_warnings,
        }
    )
    if mismatches:
        _add(
            findings,
            AuditSeverity.FAIL,
            "panel_price_or_derived_lineage_mismatch",
            "panel raw prices or deterministic derived fields differ from frozen sources",
            count=sum(mismatches.values()),
            samples=[*samples, *[f"{key}={value}" for key, value in sorted(mismatches.items())]],
            maximum_samples=config.maximum_samples,
        )
    if entry_return_failures:
        _add(
            findings,
            AuditSeverity.FAIL,
            "extreme_adjusted_returns_at_entry_or_reentry",
            "adjusted returns at universe entry/re-entry exceed the hard integrity threshold",
            count=entry_return_failures,
            samples=entry_return_samples,
            maximum_samples=config.maximum_samples,
        )
    if entry_return_warnings:
        _add(
            findings,
            AuditSeverity.WARN,
            "suspicious_adjusted_returns_at_entry_or_reentry",
            "adjusted returns at universe entry/re-entry require manual review",
            count=entry_return_warnings,
            samples=entry_return_samples,
            maximum_samples=config.maximum_samples,
        )


def _audit_limit_weight_lineage(
    panel: pd.DataFrame,
    manifest: Mapping[str, object],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    build_config = manifest.get("config")
    minimum_days = (
        int(build_config.get("minimum_limit_rule_listed_days", 30))
        if isinstance(build_config, Mapping)
        else 30
    )
    expected_limits = derive_conservative_one_price_limits(
        panel, minimum_listed_days=minimum_days
    )
    limit_mismatches: dict[str, int] = {}
    for field in (
        "limit_up",
        "limit_down",
        "limit_proxy_known",
        "research_close_limit_up_proxy",
        "research_close_limit_down_proxy",
        "research_close_limit_known_proxy",
    ):
        mismatch = panel[field].astype(bool).to_numpy() != expected_limits[field].astype(bool).to_numpy()
        if mismatch.any():
            limit_mismatches[field] = int(mismatch.sum())
    ratio_match = np.isclose(
        pd.to_numeric(panel["limit_ratio_proxy"], errors="coerce"),
        pd.to_numeric(expected_limits["limit_ratio_proxy"], errors="coerce"),
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    )
    if not ratio_match.all():
        limit_mismatches["limit_ratio_proxy"] = int((~ratio_match).sum())
    if limit_mismatches:
        _add(
            findings,
            AuditSeverity.FAIL,
            "limit_proxy_lineage_mismatch",
            "stored price-limit proxy differs from the frozen deterministic rule",
            count=sum(limit_mismatches.values()),
            samples=[f"{key}={value}" for key, value in sorted(limit_mismatches.items())],
            maximum_samples=config.maximum_samples,
        )

    mcap = pd.to_numeric(panel["implied_float_mcap"], errors="coerce")
    valid_mcap = mcap.notna() & mcap.gt(0)
    complete_date = valid_mcap.groupby(level="date").transform("all")
    date_sum = mcap.where(valid_mcap).groupby(level="date").transform("sum")
    date_count = mcap.groupby(level="date").transform("size")
    use_mcap = complete_date & date_sum.gt(0)
    expected_weight = mcap.div(date_sum).where(use_mcap, 1.0 / date_count)
    expected_source = pd.Series(
        np.where(
            use_mcap,
            "research_proxy:implied_float_mcap",
            "research_proxy:equal_weight_fallback",
        ),
        index=panel.index,
        dtype="string",
    )
    weight_bad = _numeric_mismatch(
        panel["benchmark_weight"], expected_weight, rtol=1e-12, atol=1e-12
    )
    source_bad = (
        panel["benchmark_weight_source"].astype("string").to_numpy()
        != expected_source.to_numpy()
    )
    metrics.setdefault("source_lineage", {}).update(
        {
            "limit_mismatches": limit_mismatches,
            "benchmark_weight_mismatches": int(weight_bad.sum()),
            "benchmark_weight_source_mismatches": int(source_bad.sum()),
        }
    )
    if weight_bad.any() or source_bad.any():
        _add(
            findings,
            AuditSeverity.FAIL,
            "benchmark_weight_lineage_mismatch",
            "stored benchmark proxy weights differ from the frozen deterministic rule",
            count=int(weight_bad.sum() + source_bad.sum()),
            samples=_sample_index(
                panel.index[weight_bad | source_bad], config.maximum_samples
            ),
            maximum_samples=config.maximum_samples,
        )


def _audit_replay_normalized_source_lineage(
    manifest: Mapping[str, object],
    normalized_frames: dict[str, pd.DataFrame],
    normalized_dir: Path,
    cache_dir: Path,
    panel_dir: Path,
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> tuple[pd.DataFrame | None, list[Mapping[str, object]]]:
    """Rebuild membership, security and industry catalogs from raw cache."""

    raw_config = manifest.get("config")
    if not isinstance(raw_config, Mapping):
        _add(
            findings,
            AuditSeverity.FAIL,
            "source_lineage_build_policy_missing",
            "manifest lacks the frozen build policy needed for source replay",
        )
        return None, []
    payload: dict[str, object] = {}
    for name in HS300BuildConfig.__dataclass_fields__:
        if name in raw_config:
            payload[name] = raw_config[name]
    payload.update(
        {
            "cache_dir": cache_dir,
            "normalized_dir": normalized_dir,
            "output_dir": panel_dir,
        }
    )
    try:
        build_config = HS300BuildConfig(**payload)
        (
            rebuilt_membership,
            rebuilt_snapshots,
            rebuilt_union,
            membership_records,
            membership_policy,
        ) = _load_and_rebuild_membership_lineage(build_config, normalized_frames)
        member_codes = set(rebuilt_union["code"].astype(str))
        security, security_records, security_policy = (
            _load_and_rebuild_security_lineage(
                build_config, normalized_frames, member_codes
            )
        )
        industry, industry_records, industry_policy = (
            _load_and_rebuild_industry_lineage(
                build_config,
                normalized_frames,
                rebuilt_snapshots,
                member_codes,
            )
        )
    except Exception as exc:
        _add(
            findings,
            AuditSeverity.FAIL,
            "normalized_source_raw_replay_failed",
            "membership/security/industry normalized sources cannot be reproduced from canonical cache: "
            f"{type(exc).__name__}: {exc}",
        )
        return None, []

    declared = (
        ("membership", membership_records, membership_policy),
        ("security", security_records, security_policy),
        ("industry", industry_records, industry_policy),
    )
    mismatches: list[str] = []
    for name, records, policy in declared:
        manifest_records = manifest.get(f"{name}_cache_objects")
        manifest_policy = manifest.get(f"{name}_lineage")
        if (
            not isinstance(manifest_records, list)
            or _canonical_json_sha256(manifest_records)
            != _canonical_json_sha256(records)
        ):
            mismatches.append(f"{name}_cache_objects")
        if (
            not isinstance(manifest_policy, Mapping)
            or _canonical_json_sha256(dict(manifest_policy))
            != _canonical_json_sha256(dict(policy))
        ):
            mismatches.append(f"{name}_lineage")
    if mismatches:
        _add(
            findings,
            AuditSeverity.FAIL,
            "normalized_source_lineage_manifest_mismatch",
            "manifested source lineage does not equal independent raw-cache replay",
            count=len(mismatches),
            samples=mismatches,
            maximum_samples=config.maximum_samples,
        )
    normalized_hashes = manifest.get("normalized_input_sha256")
    source_lineage = manifest.get("source_lineage")
    lineage_names = {
        "membership": (
            "trade_calendar",
            "membership_snapshots",
            "membership_daily",
            "security_union",
            "membership_request_index",
        ),
        "security": ("security_master", "security_master_request_index"),
        "industry": ("industry_snapshots", "industry_request_index"),
    }
    expected_domains: dict[str, dict[str, object]] = {}
    if isinstance(normalized_hashes, Mapping):
        for name, records, policy in declared:
            identity: dict[str, object] = {
                "policy": dict(policy),
                "cache_objects_sha256": _canonical_json_sha256(records),
                "normalized_input_sha256": {
                    key: normalized_hashes.get(key) for key in lineage_names[name]
                },
            }
            expected_domains[name] = {
                **identity,
                "identity_sha256": _canonical_json_sha256(identity),
            }
    expected_root = _canonical_json_sha256(expected_domains)
    if (
        not isinstance(source_lineage, Mapping)
        or source_lineage.get("schema_version") != "1.0"
        or source_lineage.get("complete") is not True
        or _canonical_json_sha256(source_lineage.get("domains"))
        != _canonical_json_sha256(expected_domains)
        or str(source_lineage.get("root_sha256", "")) != expected_root
    ):
        mismatches.append("source_lineage_root")
        _add(
            findings,
            AuditSeverity.FAIL,
            "source_lineage_root_mismatch",
            "unified source-lineage root is not bound to policies, raw objects and normalized bytes",
        )
    normalized_frames["membership_daily"] = rebuilt_membership
    normalized_frames["membership_snapshots"] = rebuilt_snapshots
    normalized_frames["security_union"] = rebuilt_union
    normalized_frames["security_master"] = security
    normalized_frames["industry_snapshots"] = industry
    metrics["normalized_source_replay"] = {
        "valid": not mismatches,
        "membership_rows": len(rebuilt_membership),
        "membership_cache_requests": len(membership_records),
        "security_rows": len(security),
        "security_cache_requests": len(security_records),
        "industry_rows": len(industry),
        "industry_cache_requests": len(industry_records),
        "root_sha256": expected_root,
    }
    all_records: list[Mapping[str, object]] = [
        *membership_records,
        *security_records,
        *industry_records,
    ]
    return security, all_records


def _audit_terminal_exception_source_replay(
    manifest: Mapping[str, object],
    membership: pd.DataFrame,
    security: pd.DataFrame,
    raw_price_paths: Mapping[tuple[str, str], Path],
    panel_dir: Path,
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    """Prove the sidecar is exactly membership minus real raw-market rows."""

    try:
        sidecar = _validated_terminal_exception_sidecar(manifest, panel_dir)
        raw_indexes: list[pd.MultiIndex] = []
        for (code, flag), path in sorted(raw_price_paths.items()):
            if flag != "3":
                continue
            raw = pd.read_parquet(path, columns=["date", "code"])
            dates = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
            if dates.isna().any() or not raw["code"].astype(str).eq(code).all():
                raise ValueError(f"raw price identity is invalid for {code}")
            raw_indexes.append(
                pd.MultiIndex.from_arrays(
                    [dates, raw["code"].astype(str)], names=["date", "symbol"]
                )
            )
        if not raw_indexes:
            raise ValueError("no raw price indexes are available")
        raw_index = raw_indexes[0]
        for index in raw_indexes[1:]:
            raw_index = raw_index.append(index)
        raw_index = raw_index.sort_values()
        if not raw_index.is_unique:
            raise ValueError("raw price index is not unique")
        tape_section = manifest.get("feature_tape")
        if not isinstance(tape_section, Mapping):
            raise ValueError("feature tape span is unavailable")
        tape_end = pd.Timestamp(str(tape_section.get("last_date", ""))).normalize()
        bounded = membership.copy()
        bounded["date"] = pd.to_datetime(
            bounded["date"], errors="coerce"
        ).dt.normalize()
        bounded = bounded.loc[bounded["date"].le(tape_end)].copy()
        expected = _classify_terminal_stale_membership_exceptions(
            bounded,
            raw_index,
            security,
        )
        if len(sidecar.index.intersection(raw_index)):
            raise ValueError("sidecar contains a member row with a real raw bar")
        if not sidecar.equals(expected):
            raise ValueError(
                "sidecar is not the exact authoritative membership-minus-raw classification"
            )
    except Exception as exc:
        _add(
            findings,
            AuditSeverity.FAIL,
            "terminal_exception_source_replay_mismatch",
            "terminal membership exceptions do not equal frozen membership/raw/basic sources: "
            f"{type(exc).__name__}: {exc}",
        )
        return
    metrics["terminal_exception_source_replay"] = {
        "valid": True,
        "rows": len(sidecar),
        "raw_market_rows": len(raw_index),
    }


def _audit_sources_and_delisted(
    panel: pd.DataFrame,
    manifest: Mapping[str, object],
    config: HS300AuditConfig,
    findings: list[AuditFinding],
    metrics: dict[str, object],
) -> None:
    normalized_dir = _resolve_manifest_directory(
        config.normalized_dir,
        manifest,
        "normalized_dir",
        config.panel_dir.resolve(),
    )
    cache_dir = _resolve_manifest_directory(
        config.cache_dir,
        manifest,
        "cache_dir",
        config.panel_dir.resolve(),
    )
    source_metrics: dict[str, object] = {
        "normalized_dir": str(normalized_dir) if normalized_dir else None,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "normalized_hashes_verified": 0,
        "cache_hashes_verified": 0,
    }
    metrics["source_integrity"] = source_metrics
    declared_hashes = manifest.get("normalized_input_sha256")
    frames: dict[str, pd.DataFrame] = {}
    if normalized_dir is None or not isinstance(declared_hashes, Mapping):
        _add(
            findings,
            AuditSeverity.FAIL,
            "normalized_provenance_unavailable",
            "manifest cannot locate and hash its normalized source catalogs",
        )
        security = None
        calendar = None
    else:
        for key, filename in _NORMALIZED_FILENAMES.items():
            path = normalized_dir / filename
            expected = str(declared_hashes.get(key, ""))
            if not path.is_file():
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "normalized_input_missing",
                    f"normalized source is missing: {path}",
                )
                continue
            actual = _sha256_file(path)
            if not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "normalized_input_sha_mismatch",
                    f"normalized source hash mismatch: {filename}",
                )
                continue
            source_metrics["normalized_hashes_verified"] = int(
                source_metrics["normalized_hashes_verified"]
            ) + 1
            try:
                frames[key] = pd.read_parquet(path)
            except Exception as exc:
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "normalized_input_unreadable",
                    f"cannot read {filename}: {type(exc).__name__}: {exc}",
                )
        security = frames.get("security_master")
        calendar = frames.get("trade_calendar")

    replay_records: list[Mapping[str, object]] = []
    if (
        normalized_dir is not None
        and cache_dir is not None
        and len(frames) == len(_NORMALIZED_FILENAMES)
    ):
        replayed_security, replay_records = _audit_replay_normalized_source_lineage(
            manifest,
            frames,
            normalized_dir,
            cache_dir,
            config.panel_dir.resolve(),
            config,
            findings,
            metrics,
        )
        if replayed_security is not None:
            security = replayed_security
            calendar = frames.get("trade_calendar")

    if calendar is not None and len(panel):
        if {"calendar_date", "is_trading_day"}.issubset(calendar.columns):
            trading = pd.to_datetime(
                calendar.loc[
                    calendar["is_trading_day"].astype(str).eq("1"), "calendar_date"
                ],
                errors="coerce",
            ).dropna()
            first = panel.index.get_level_values("date").min()
            last = panel.index.get_level_values("date").max()
            expected_dates = set(pd.DatetimeIndex(trading).normalize())
            expected_dates = {date for date in expected_dates if first <= date <= last}
            actual_dates = set(pd.DatetimeIndex(panel.index.get_level_values("date")).normalize())
            missing_dates = sorted(expected_dates - actual_dates)
            extra_dates = sorted(actual_dates - expected_dates)
            source_metrics["calendar_missing_dates"] = len(missing_dates)
            source_metrics["calendar_extra_dates"] = len(extra_dates)
            if missing_dates or extra_dates:
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "trading_calendar_coverage_mismatch",
                    "panel dates do not exactly cover the frozen BaoStock trading calendar interval",
                    count=len(missing_dates) + len(extra_dates),
                    samples=[
                        *[f"missing:{date.date()}" for date in missing_dates],
                        *[f"extra:{date.date()}" for date in extra_dates],
                    ],
                    maximum_samples=config.maximum_samples,
                )
        else:
            _add(findings, AuditSeverity.FAIL, "trade_calendar_schema_invalid", "trade calendar lacks required columns")

    cache_records: list[Mapping[str, object]] = []
    raw_records = manifest.get("price_cache_objects")
    membership_records = manifest.get("membership_cache_objects")
    security_records = manifest.get("security_cache_objects")
    industry_records = manifest.get("industry_cache_objects")
    adjustment_records = manifest.get("adjustment_cache_objects")
    index_record = manifest.get("index_cache_object")
    if isinstance(raw_records, list):
        cache_records.extend(item for item in raw_records if isinstance(item, Mapping))
    for records in (membership_records, security_records, industry_records):
        if isinstance(records, list):
            cache_records.extend(
                item for item in records if isinstance(item, Mapping)
            )
    if isinstance(adjustment_records, list):
        cache_records.extend(
            item for item in adjustment_records if isinstance(item, Mapping)
        )
    if isinstance(index_record, Mapping):
        cache_records.append(index_record)
    resolved_cache_paths: dict[tuple[str, str], Path] = {}
    if cache_dir is None or not cache_records:
        _add(
            findings,
            AuditSeverity.FAIL,
            "cache_provenance_unavailable",
            "manifest cannot locate its content-addressed BaoStock objects",
        )
    else:
        keys_seen: set[tuple[str, str]] = set()
        provider_versions: set[str] = set()
        for record in cache_records:
            explicit_endpoint = str(record.get("endpoint", "")).strip()
            if explicit_endpoint:
                key = str(record.get("request_key", ""))
                identity = (f"__{explicit_endpoint}__", key)
                code = explicit_endpoint
                flag = key[:12]
            else:
                is_adjustment = "adjustflag" not in record and "code" in record
                is_index = "adjustflag" not in record and not is_adjustment
                code = str(
                    record.get("code", "sh.000300" if is_index else "")
                )
                flag = (
                    "adjustment"
                    if is_adjustment
                    else ("index" if is_index else str(record.get("adjustflag", "")))
                )
                identity = (code, flag)
            if identity in keys_seen:
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "cache_record_duplicate",
                    f"duplicate cache record {code}/{flag}",
                )
            keys_seen.add(identity)
            provider_versions.add(str(record.get("provider_version", "")))
            path = _resolve_cache_object(cache_dir, record.get("object"))
            if path is None or not path.is_file():
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "cache_object_missing",
                    f"cache object cannot be resolved for {code}/{flag}",
                )
                continue
            resolved_cache_paths[identity] = path
            actual = _sha256_file(path)
            if actual != str(record.get("parquet_sha256", "")):
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "cache_object_sha_mismatch",
                    f"cache object hash mismatch for {code}/{flag}",
                )
            else:
                source_metrics["cache_hashes_verified"] = int(
                    source_metrics["cache_hashes_verified"]
                ) + 1
        if "" in provider_versions or len(provider_versions) != 1:
            _add(
                findings,
                AuditSeverity.FAIL,
                "cache_provider_version_mixed",
                "snapshot cache objects mix absent or different provider versions",
            )

    _audit_membership_industry_lineage(
        panel, manifest, frames, config, findings, metrics
    )
    _audit_limit_weight_lineage(panel, manifest, config, findings, metrics)
    if cache_dir is not None and cache_records:
        _audit_cache_states(
            cache_dir,
            cache_records,
            resolved_cache_paths,
            manifest,
            config,
            findings,
            metrics,
        )
        _audit_vendor_adjustment_event_lineage(
            manifest,
            frames,
            resolved_cache_paths,
            config,
            findings,
            metrics,
        )
    if security is not None and resolved_cache_paths:
        membership_frame = frames.get("membership_daily")
        if membership_frame is not None:
            _audit_terminal_exception_source_replay(
                manifest,
                membership_frame,
                security,
                resolved_cache_paths,
                config.panel_dir.resolve(),
                config,
                findings,
                metrics,
            )
        _audit_price_and_derived_lineage(
            panel,
            manifest,
            frames,
            security,
            resolved_cache_paths,
            config,
            findings,
            metrics,
        )

    validation = manifest.get("validation")
    validation_meta = validation.get("metadata") if isinstance(validation, Mapping) else None
    includes_delisted = isinstance(validation_meta, Mapping) and validation_meta.get("includes_delisted") is True
    if not includes_delisted:
        _add(
            findings,
            AuditSeverity.FAIL,
            "manifest_does_not_confirm_delisted_history",
            "build metadata does not affirm delisted-history inclusion",
        )

    delisted_metrics: dict[str, object] = {
        "metadata_confirms_delisted_history": includes_delisted,
        "securities_with_out_date": 0,
        "raw_adjusted_pairs_covered": 0,
        "symbols_seen_in_research_panel": 0,
    }
    metrics["delisted_history"] = delisted_metrics
    if security is None or "code" not in security or "outDate" not in security:
        _add(
            findings,
            AuditSeverity.FAIL,
            "security_master_delisted_schema_invalid",
            "security master cannot support the delisted-history audit",
        )
        return
    basic_columns = ["code", "outDate", *( ["status"] if "status" in security else [] )]
    basic = security[basic_columns].copy()
    basic["code"] = basic["code"].astype(str)
    raw_out_dates = basic["outDate"].replace({"": pd.NA, "0": pd.NA})
    basic["out_date"] = pd.to_datetime(raw_out_dates, errors="coerce").dt.normalize()
    invalid_out_date = raw_out_dates.notna() & basic["out_date"].isna()
    if invalid_out_date.any():
        _add(
            findings,
            AuditSeverity.FAIL,
            "security_master_out_date_invalid",
            "security master contains a nonblank, unparsable out-date",
            count=int(invalid_out_date.sum()),
            samples=basic.loc[invalid_out_date, "code"].tolist(),
            maximum_samples=config.maximum_samples,
        )
    inactive = (
        basic["status"].astype(str).str.strip().eq("0")
        if "status" in basic
        else pd.Series(False, index=basic.index)
    )
    delisted = basic.loc[basic["out_date"].notna() | inactive].drop_duplicates("code")
    inactive_without_date = inactive & basic["out_date"].isna()
    if inactive_without_date.any():
        _add(
            findings,
            AuditSeverity.WARN,
            "inactive_security_missing_out_date",
            "inactive historical-union securities have no parseable out-date",
            count=int(inactive_without_date.sum()),
            samples=basic.loc[inactive_without_date, "code"].tolist(),
            maximum_samples=config.maximum_samples,
        )
    delisted_metrics["securities_with_out_date"] = len(delisted)
    panel_symbols = set(panel.index.get_level_values("symbol").astype(str))
    delisted_metrics["symbols_seen_in_research_panel"] = int(
        delisted["code"].isin(panel_symbols).sum()
    )
    records_by_code: dict[str, dict[str, Mapping[str, object]]] = {}
    if isinstance(raw_records, list):
        for record in raw_records:
            if isinstance(record, Mapping):
                records_by_code.setdefault(str(record.get("code", "")), {})[
                    str(record.get("adjustflag", ""))
                ] = record
    security_codes = set(basic["code"])
    incomplete_union = [
        code for code in sorted(security_codes) if set(records_by_code.get(code, {})) != {"1", "3"}
    ]
    extra_union = sorted(set(records_by_code) - security_codes)
    if incomplete_union or extra_union:
        _add(
            findings,
            AuditSeverity.FAIL,
            "historical_union_cache_pair_mismatch",
            "raw/adjusted cache pairs do not exactly cover the security-master union",
            count=len(incomplete_union) + len(extra_union),
            samples=[
                *[f"incomplete:{code}" for code in incomplete_union],
                *[f"extra:{code}" for code in extra_union],
            ],
            maximum_samples=config.maximum_samples,
        )
    uncovered: list[str] = []
    delisted_date_map = delisted.set_index("code")["out_date"]
    for code in delisted["code"]:
        pair = records_by_code.get(code, {})
        if set(pair) != {"1", "3"} or any(int(item.get("rows", 0)) <= 0 for item in pair.values()):
            uncovered.append(code)
        else:
            delisted_metrics["raw_adjusted_pairs_covered"] = int(
                delisted_metrics["raw_adjusted_pairs_covered"]
            ) + 1
            raw_path = resolved_cache_paths.get((code, "3"))
            adjusted_path = resolved_cache_paths.get((code, "1"))
            if raw_path is None or adjusted_path is None:
                continue
            try:
                raw_history = pd.read_parquet(raw_path, columns=["date", "code"])
                adjusted_history = pd.read_parquet(adjusted_path, columns=["date", "code"])
            except Exception as exc:
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "delisted_cache_object_unreadable",
                    f"cannot inspect delisted history for {code}: {type(exc).__name__}: {exc}",
                )
                continue
            raw_dates = pd.to_datetime(raw_history["date"], errors="coerce").dt.normalize()
            adjusted_dates = pd.to_datetime(
                adjusted_history["date"], errors="coerce"
            ).dt.normalize()
            invalid_payload = (
                raw_dates.isna().any()
                or adjusted_dates.isna().any()
                or not raw_history["code"].astype(str).eq(code).all()
                or not adjusted_history["code"].astype(str).eq(code).all()
                or len(raw_history) != int(pair["3"].get("rows", -1))
                or len(adjusted_history) != int(pair["1"].get("rows", -1))
            )
            if invalid_payload:
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "delisted_cache_payload_invalid",
                    f"delisted raw/adjusted payload metadata is invalid for {code}",
                )
                continue
            if not pd.Index(raw_dates).equals(pd.Index(adjusted_dates)):
                _add(
                    findings,
                    AuditSeverity.FAIL,
                    "delisted_raw_adjusted_dates_mismatch",
                    f"delisted raw/adjusted histories cover different sessions for {code}",
                )
            raw_out_date = delisted_date_map.loc[code]
            if pd.isna(raw_out_date):
                continue
            out_date = pd.Timestamp(raw_out_date)
            if len(raw_dates) and out_date <= panel.index.get_level_values("date").max():
                last_history = pd.Timestamp(raw_dates.max())
                if last_history > out_date + pd.Timedelta(days=7):
                    _add(
                        findings,
                        AuditSeverity.FAIL,
                        "delisted_history_after_out_date",
                        f"{code} has price history materially after its security-master out-date",
                    )
                elif (out_date - last_history).days > 180:
                    _add(
                        findings,
                        AuditSeverity.WARN,
                        "delisted_history_ends_well_before_out_date",
                        f"{code} last price precedes its out-date by more than 180 days",
                    )
    if uncovered:
        _add(
            findings,
            AuditSeverity.FAIL,
            "delisted_price_history_uncovered",
            "historical-union delisted securities lack non-empty raw/adjusted cache pairs",
            count=len(uncovered),
            samples=uncovered,
            maximum_samples=config.maximum_samples,
        )
    if len(delisted) == 0:
        span = (
            panel.index.get_level_values("date").max()
            - panel.index.get_level_values("date").min()
        ).days
        severity = AuditSeverity.FAIL if span >= 5 * 365 else AuditSeverity.WARN
        _add(
            findings,
            severity,
            "delisted_coverage_not_observed",
            "no security with an out-date exists in the historical member union",
        )


def _final_report(
    config: HS300AuditConfig,
    manifest: Mapping[str, object] | None,
    panel: pd.DataFrame | None,
    findings: list[AuditFinding],
    metrics: Mapping[str, object],
) -> HS300AuditReport:
    fail = any(item.severity is AuditSeverity.FAIL for item in findings)
    warn = any(item.severity is AuditSeverity.WARN for item in findings)
    if panel is not None and len(panel) and isinstance(panel.index, pd.MultiIndex):
        dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
        symbols = pd.Index(panel.index.get_level_values("symbol").astype(str))
        rows = len(panel)
        symbol_count = int(symbols.nunique())
        trading_days = int(dates.nunique())
        first = str(dates.min().date())
        last = str(dates.max().date())
        actual_hash = str(metrics.get("integrity", {}).get("actual_snapshot_hash", ""))
    else:
        rows = symbol_count = trading_days = 0
        first = last = actual_hash = ""
    expected_hash = str(manifest.get("snapshot_hash", "")) if manifest else ""
    return HS300AuditReport(
        valid=not fail,
        status="fail" if fail else ("warn" if warn else "pass"),
        panel_dir=str(config.panel_dir.resolve()),
        manifest_path=str((config.panel_dir.resolve() / "manifest.json")),
        rows=rows,
        symbols=symbol_count,
        trading_days=trading_days,
        first_date=first,
        last_date=last,
        expected_snapshot_hash=expected_hash,
        actual_snapshot_hash=actual_hash,
        research_valid=not fail,
        portfolio_grade=False,
        capital_deployment_allowed=False,
        findings=tuple(findings),
        metrics=metrics,
    )


def audit_hs300_research_panel(config: HS300AuditConfig) -> HS300AuditReport:
    """Audit a built snapshot without network access or provider calls."""

    findings: list[AuditFinding] = []
    metrics: dict[str, object] = {}
    try:
        manifest, panel = _manifest_and_partitions(config, findings, metrics)
    except Exception as exc:
        _add(
            findings,
            AuditSeverity.FAIL,
            "frozen_snapshot_read_error",
            f"audit could not safely reconstruct the snapshot: {type(exc).__name__}: {exc}",
        )
        return _final_report(config, None, None, findings, metrics)
    if manifest is None or panel is None:
        return _final_report(config, manifest, panel, findings, metrics)
    stages = (
        ("panel_structure", _audit_panel_structure),
        ("continuous_feature_tape", _audit_feature_tape),
        ("column_coverage", _audit_column_coverage),
        ("raw_adjusted_contract", _audit_raw_adjusted_contract),
        ("temporal_and_proxy", _audit_temporal_and_proxy_fields),
        ("adjusted_returns", _audit_adjusted_returns),
        ("sources_and_delisted", _audit_sources_and_delisted),
    )
    for stage_name, stage in stages:
        try:
            if stage_name in {
                "panel_structure",
                "continuous_feature_tape",
                "temporal_and_proxy",
                "sources_and_delisted",
            }:
                stage(panel, manifest, config, findings, metrics)
            else:
                stage(panel, config, findings, metrics)
        except Exception as exc:
            _add(
                findings,
                AuditSeverity.FAIL,
                f"audit_stage_error_{stage_name}",
                f"{stage_name} audit failed closed: {type(exc).__name__}: {exc}",
            )
    return _final_report(config, manifest, panel, findings, metrics)


__all__ = [
    "AuditFinding",
    "AuditSeverity",
    "HS300AuditConfig",
    "HS300AuditReport",
    "audit_hs300_research_panel",
]
