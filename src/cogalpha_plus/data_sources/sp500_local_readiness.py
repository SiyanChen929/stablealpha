"""Fail-closed, read-only readiness audit for a point-in-time S&P 500 panel.

The auditor deliberately does not infer capital-grade readiness from a useful
CSV or parquet file.  A domain is ready only when an explicit evidence
descriptor binds an authoritative source contract to immutable raw objects.
Current-constituent histories and current classifications remain useful for
secondary validation, but can never satisfy point-in-time requirements.

No network access is used.  Paths containing sealed/holdout/reveal components
are rejected before file content is opened.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from ..manifest import canonical_json, hash_json, utc_now
from .us_index_contract import US_INDEX_EVENT_DAY_ORDER_SHA256


SCHEMA_VERSION = 1
AUDITOR_ID = "cogalpha_sp500_local_readiness_v1"
EVIDENCE_ARTIFACT_TYPE = "cogalpha_sp500_readiness_evidence"

REQUIRED_DOMAINS = (
    "price_panel",
    "historical_membership",
    "stable_security_id",
    "terminal_delisting_payoff",
    "corporate_actions",
    "pit_sector",
    "benchmark_weights",
)

DOMAIN_LABELS = {
    "price_panel": "price_panel",
    "historical_membership": "historical_membership",
    "stable_security_id": "stable_security_id",
    "terminal_delisting_payoff": "terminal/delisting_payoff",
    "corporate_actions": "corporate_actions",
    "pit_sector": "PIT sector",
    "benchmark_weights": "benchmark_weights",
}

_FORBIDDEN_COMPONENT_TOKENS = (
    "sealed",
    "lockbox",
    "reveal",
    "holdout",
    "outer_test",
    "out_of_sample_results",
)
_FORBIDDEN_EXACT_COMPONENTS = {"runs", ".git", "__pycache__", ".venv"}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_PERMANENT_FALSE_READINESS_FIELDS = (
    "ready",
    "us_index_contract_validated",
    "data_contract_ready",
    "factor_search_allowed",
    "capital_grade",
    "capital_deployment_allowed",
    "production_certification",
    "economic_content_correctness_certified",
    "point_in_time_panel_ready",
)

_COMMON_REQUIRED_CONTRACT_FLAGS = (
    "source_lineage_complete",
    "raw_vendor_objects_immutable",
    "vendor_authoritative_for_domain",
)

_DOMAIN_REQUIRED_COLUMNS: Mapping[str, frozenset[str]] = {
    "price_panel": frozenset(
        {
            "security_id",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "available_at",
        }
    ),
    "historical_membership": frozenset(
        {
            "security_id",
            "event_type",
            "announcement_at",
            "known_at",
            "effective_at",
        }
    ),
    "stable_security_id": frozenset(
        {
            "security_id",
            "identifier",
            "identifier_type",
            "valid_from",
            "valid_to",
            "known_at",
        }
    ),
    "terminal_delisting_payoff": frozenset(
        {
            "event_id",
            "source_security_id",
            "event_type",
            "effective_at",
            "settlement_at",
            "leg_type",
            "currency",
            "amount",
            "destination_security_id",
            "ratio",
        }
    ),
    "corporate_actions": frozenset(
        {
            "event_id",
            "security_id",
            "action_type",
            "announcement_at",
            "known_at",
            "effective_at",
        }
    ),
    "pit_sector": frozenset(
        {
            "security_id",
            "classification",
            "valid_from",
            "valid_to",
            "known_at",
        }
    ),
    "benchmark_weights": frozenset(
        {"security_id", "effective_at", "known_at", "weight"}
    ),
}

_DOMAIN_REQUIRED_CONTRACT_FLAGS: Mapping[str, tuple[str, ...]] = {
    "price_panel": (
        "historical_union_coverage",
        "ohlcv_unadjusted",
        "missing_sessions_explicit",
        "stable_security_id_join",
        "corporate_action_linkage",
        "availability_timestamps",
    ),
    "historical_membership": (
        "initial_snapshot_present",
        "add_remove_events_complete",
        "replay_validated",
        "announcement_and_known_times",
        "stable_security_id_join",
    ),
    "stable_security_id": (
        "identifier_aliases_complete",
        "ticker_not_primary_key",
        "permanent_id_namespace_documented",
        "identifier_reuse_checked",
    ),
    "terminal_delisting_payoff": (
        "terminal_universe_complete",
        "cash_and_security_legs_supported",
        "multi_leg_supported",
        "fractional_cash_in_lieu_supported",
        "destination_security_id_supported",
        "delisting_returns_validated",
    ),
    "corporate_actions": (
        "action_universe_complete",
        "split_and_distribution_supported",
        "merger_legs_linked",
        "no_future_adjusted_history",
        "availability_timestamps",
    ),
    "pit_sector": (
        "classification_history_complete",
        "validity_intervals",
        "knowledge_times",
        "historical_union_coverage",
    ),
    "benchmark_weights": (
        "official_benchmark_source",
        "historical_weights_complete",
        "weight_sums_validated",
        "stable_security_id_join",
        "knowledge_times",
    ),
}

_DOMAIN_REQUIRED_OBJECT_ROLES: Mapping[str, frozenset[str]] = {
    "price_panel": frozenset({"raw_prices"}),
    "historical_membership": frozenset(
        {"initial_membership_snapshot", "membership_events"}
    ),
    "stable_security_id": frozenset(
        {"security_master", "identifier_aliases"}
    ),
    "terminal_delisting_payoff": frozenset(
        {"terminal_events", "terminal_payoff_legs"}
    ),
    "corporate_actions": frozenset(
        {"corporate_actions", "corporate_action_legs"}
    ),
    "pit_sector": frozenset({"classification_history"}),
    "benchmark_weights": frozenset({"benchmark_weights"}),
}

_CURRENT_ONLY_SCOPES = {
    "current",
    "current_only",
    "current_constituents",
    "current_constituents_history",
    "latest",
    "latest_snapshot",
}

_HISTORICAL_SCOPES: Mapping[str, frozenset[str]] = {
    "price_panel": frozenset({"historical_union", "historical_price_panel"}),
    "historical_membership": frozenset(
        {"historical_union", "historical_membership_events"}
    ),
    "stable_security_id": frozenset(
        {"historical_union", "historical_security_master"}
    ),
    "terminal_delisting_payoff": frozenset(
        {"historical_union", "historical_union_terminal_events"}
    ),
    "corporate_actions": frozenset(
        {"historical_union", "historical_corporate_actions"}
    ),
    "pit_sector": frozenset(
        {"historical_union", "historical_classification"}
    ),
    "benchmark_weights": frozenset(
        {"historical_union", "historical_benchmark_weights"}
    ),
}

_CROSS_SECTIONAL_DOMAINS = frozenset(
    {
        "price_panel",
        "historical_membership",
        "stable_security_id",
        "pit_sector",
        "benchmark_weights",
    }
)


@dataclass(frozen=True)
class SP500ReadinessConfig:
    """Configuration for a bounded local audit."""

    project_dir: Path
    asset_paths: tuple[Path, ...] = ()
    include_known_assets: bool = True
    required_start: str = "2008-01-01"
    required_end: str = "2026-08-31"
    maximum_discovered_files: int = 256
    maximum_hash_bytes: int = 1_073_741_824
    minimum_security_count: int = 500

    def __post_init__(self) -> None:
        start = _parse_iso_date(self.required_start, "required_start")
        end = _parse_iso_date(self.required_end, "required_end")
        if start > end:
            raise ValueError("required_start must not be after required_end")
        if self.maximum_discovered_files < 1:
            raise ValueError("maximum_discovered_files must be positive")
        if self.maximum_hash_bytes < 1:
            raise ValueError("maximum_hash_bytes must be positive")
        if self.minimum_security_count < 1:
            raise ValueError("minimum_security_count must be positive")
        object.__setattr__(self, "project_dir", Path(self.project_dir).resolve())
        object.__setattr__(
            self,
            "asset_paths",
            tuple(Path(path) for path in self.asset_paths),
        )


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, init=False)
class SP500ReadinessReport:
    """Deeply immutable JSON audit result."""

    _canonical_manifest: str

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        payload = dict(manifest)
        recorded_hash = payload.pop("manifest_hash", None)
        expected_hash = hash_json(payload)
        if recorded_hash is not None and recorded_hash != expected_hash:
            raise ValueError("readiness manifest_hash does not match manifest content")
        if payload.get("status") != "NOT_READY":
            raise ValueError("S&P 500 readiness status is fail-closed at NOT_READY")
        for field in _PERMANENT_FALSE_READINESS_FIELDS:
            if payload.get(field) is not False:
                raise ValueError(f"readiness invariant requires {field}=false")
        if not isinstance(payload.get("structural_evidence_contract_ready"), bool):
            raise ValueError("structural_evidence_contract_ready must be boolean")
        blockers = payload.get("readiness_blockers")
        if not isinstance(blockers, list) or "semantic_validation_not_implemented" not in blockers:
            raise ValueError(
                "readiness_blockers must retain semantic_validation_not_implemented"
            )
        domains = payload.get("domains")
        if not isinstance(domains, Mapping) or set(domains) != set(REQUIRED_DOMAINS):
            raise ValueError("readiness manifest must contain every required domain")
        for domain, assessment in domains.items():
            if not isinstance(assessment, Mapping) or assessment.get("ready") is not False:
                raise ValueError(f"readiness domain {domain} must remain fail-closed")
        payload["manifest_hash"] = expected_hash
        object.__setattr__(self, "_canonical_manifest", canonical_json(payload))

    @property
    def manifest(self) -> Mapping[str, Any]:
        return _deep_freeze_json(json.loads(self._canonical_manifest))

    @property
    def ready(self) -> bool:
        """Whether the local data evidence contract passed, not production readiness."""

        return bool(self.to_dict()["data_contract_ready"])

    @property
    def data_contract_ready(self) -> bool:
        return bool(self.to_dict()["data_contract_ready"])

    @property
    def status(self) -> str:
        return str(self.to_dict()["status"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_manifest)

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> Path:
        target = Path(path)
        safe_target, reason = _safe_resolve(target)
        if reason or safe_target is None:
            raise ValueError(f"refusing readiness manifest output path: {reason}")
        target = safe_target
        target.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT
        flags |= os.O_TRUNC if overwrite else os.O_EXCL
        descriptor = _open_absolute_without_symlink_components(
            target, flags, mode=0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            payload = self.to_dict()
            recorded_hash = payload.pop("manifest_hash")
            if recorded_hash != hash_json(payload):
                raise ValueError("readiness manifest changed after construction")
            payload["manifest_hash"] = recorded_hash
            handle.write(canonical_json(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return target


@dataclass(frozen=True)
class _KnownAssetSpec:
    asset_id: str
    relative_to_desktop: str
    asset_type: str
    scope: str
    supports: tuple[str, ...]
    provenance: str
    limitations: tuple[str, ...]


_KNOWN_ASSET_SPECS = (
    _KnownAssetSpec(
        "yahoo_current_sp500_adjusted_close_13y",
        (
            "UBS/AI_Quant_Trading/AI_PS_TRADING/outputs/"
            "pair_identity_10y_evaluation_2026-08-26/data/raw/"
            "yahoo_adjusted_close_13y_full_universe.csv"
        ),
        "wide_adjusted_close",
        "current_constituents_history",
        ("price_panel",),
        "Yahoo Finance; current S&P 500 columns frozen 2026-08-26",
        (
            "adjusted close only; no OHLCV or independent action tape",
            "current-member survivorship bias; removed securities absent",
            "starts in 2013, after the required 2008 research start",
        ),
    ),
    _KnownAssetSpec(
        "yahoo_current_sp500_adjusted_close_metadata",
        (
            "UBS/AI_Quant_Trading/AI_PS_TRADING/outputs/"
            "pair_identity_10y_evaluation_2026-08-26/data/raw/"
            "yahoo_adjusted_close_13y_full_universe.csv.meta.json"
        ),
        "source_metadata",
        "current_constituents_history",
        ("price_panel",),
        "Local Yahoo extraction metadata sidecar",
        ("binds a current-constituent price file; does not repair survivorship",),
    ),
    _KnownAssetSpec(
        "sp500_current_constituents",
        (
            "UBS/AI_Quant_Trading/AI_PS_TRADING/outputs/"
            "pair_identity_10y_evaluation_2026-08-26/data/raw/"
            "sp500_current_constituents.csv"
        ),
        "constituent_snapshot",
        "current_only",
        ("historical_membership", "stable_security_id", "pit_sector"),
        "Wikipedia current S&P 500 snapshot",
        (
            "current constituents only; no removals or replayable events",
            "CIK is issuer-level and does not form a complete security-ID spine",
            "GICS is current, not point-in-time",
        ),
    ),
    _KnownAssetSpec(
        "sp500_current_constituents_html",
        (
            "UBS/AI_Quant_Trading/outputs/research_2026-08-26/data/raw/"
            "sp500_constituents_source.html"
        ),
        "constituent_snapshot_source",
        "current_only",
        ("historical_membership", "pit_sector"),
        "Wikipedia current-constituent source HTML",
        (
            "Date added for survivors cannot reconstruct removed constituents",
            "current GICS only",
        ),
    ),
    _KnownAssetSpec(
        "sp500_current_market_cap_and_gics",
        (
            "UBS/AI_Quant_Trading/outputs/research_2026-08-26/data/processed/"
            "sp500_security_market_cap_rankings.csv"
        ),
        "current_security_ranking",
        "current_only",
        ("pit_sector", "benchmark_weights"),
        "Current security market-cap ranking and GICS snapshot",
        (
            "market-cap proxy is not official S&P benchmark weight",
            "classification and market cap are not point-in-time histories",
        ),
    ),
    _KnownAssetSpec(
        "ai_alpha_daily_ohlcv_subset",
        "UBS/AI_Alpha_Engine/artifacts/public_research/data/daily_prices.parquet",
        "long_ohlcv_with_actions_subset",
        "selected_current_us_equities",
        ("price_panel", "corporate_actions"),
        "Mixed public US-equity research extraction",
        (
            "55-ticker subset, not an S&P 500 historical union",
            "published quality report says survivorship-free=false",
            "no delisting returns or terminal merger payoff coverage",
        ),
    ),
    _KnownAssetSpec(
        "ai_alpha_company_metadata_subset",
        "UBS/AI_Alpha_Engine/artifacts/public_research/data/company_metadata.parquet",
        "issuer_metadata_subset",
        "selected_current_us_equities",
        ("stable_security_id",),
        "SEC/company metadata for a small US-equity subset",
        ("issuer metadata is not a complete security-level alias history",),
    ),
    _KnownAssetSpec(
        "ai_alpha_explicit_quality_limits",
        "UBS/AI_Alpha_Engine/artifacts/public_research/data_quality.json",
        "quality_report",
        "selected_current_us_equities",
        (
            "price_panel",
            "historical_membership",
            "terminal_delisting_payoff",
            "pit_sector",
        ),
        "Public-research data quality report",
        (
            "negative evidence: survivorship, delisting, and historical sector gaps",
        ),
    ),
    _KnownAssetSpec(
        "msci_us_standard_resolved_events",
        (
            "UBS/msci_rebalance_research/data/events/"
            "msci_us_standard_events_resolved.csv"
        ),
        "foreign_index_membership_events",
        "msci_us_standard_historical_events",
        ("historical_membership", "stable_security_id"),
        "MSCI USA Standard review-event research",
        (
            "wrong benchmark universe; cannot substitute for S&P 500 membership",
            "ticker resolution is not a permanent security-ID master",
        ),
    ),
    _KnownAssetSpec(
        "spy_marketdata_benchmark_prices",
        "optiontrader/data/cache/marketdata_SPY_1999-01-01_latest_1d.csv",
        "benchmark_etf_ohlcv",
        "spy_etf",
        ("price_panel", "benchmark_weights"),
        "MarketData SPY daily bars",
        (
            "ETF price is useful for benchmark-return checks, not constituent weights",
            "provider adjustment is split-only; distributions need separate treatment",
        ),
    ),
    _KnownAssetSpec(
        "sp500_index_1950_2018",
        "PSTAT74/pstat174_final_project_submission/sp500_1950_2018.csv",
        "index_ohlcv",
        "sp500_index_level",
        ("price_panel", "benchmark_weights"),
        "OpenIntro/Yahoo S&P 500 index history",
        (
            "index-level series only; no constituent panel or benchmark weights",
            "ends in 2018",
        ),
    ),
)


def _parse_iso_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD)") from exc


def _parse_iso_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ISO timestamp")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _forbidden_path_reason(path: Path) -> str | None:
    parts = [part.casefold() for part in path.parts]
    for part in parts:
        if part in _FORBIDDEN_EXACT_COMPONENTS:
            return f"forbidden path component: {part}"
        if any(token in part for token in _FORBIDDEN_COMPONENT_TOKENS):
            return f"forbidden sealed/holdout path component: {part}"
    return None


def _safe_resolve(path: Path) -> tuple[Path | None, str | None]:
    lexical_reason = _forbidden_path_reason(path)
    if lexical_reason:
        return None, lexical_reason
    try:
        if path.is_symlink():
            return None, "symbolic links are not read by the readiness auditor"
        resolved = path.resolve(strict=False)
    except OSError as exc:
        return None, f"could not resolve path: {type(exc).__name__}"
    resolved_reason = _forbidden_path_reason(resolved)
    if resolved_reason:
        return None, resolved_reason
    try:
        if resolved.is_file() and resolved.stat().st_nlink > 1:
            return None, "hard-linked files are refused by the sealed-data guard"
    except OSError as exc:
        return None, f"could not inspect path identity: {type(exc).__name__}"
    return resolved, None


def _stat_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_nlink,
    )


def _open_absolute_without_symlink_components(
    path: Path,
    flags: int,
    *,
    mode: int = 0o600,
) -> int:
    """Open an absolute path through no-follow directory descriptors."""

    capabilities = _secure_open_capabilities()
    if not all(capabilities.values()):
        missing = sorted(name for name, available in capabilities.items() if not available)
        raise RuntimeError(f"secure no-follow open capabilities unavailable: {missing}")
    if not path.is_absolute() or len(path.parts) < 2:
        raise ValueError("secure file open requires an absolute non-root path")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(path.anchor, directory_flags)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(
                component,
                directory_flags | nofollow,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        if flags & os.O_CREAT:
            return os.open(
                path.name,
                flags | nofollow,
                mode,
                dir_fd=directory_fd,
            )
        return os.open(path.name, flags | nofollow, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def _secure_open_capabilities() -> dict[str, bool]:
    return {
        "o_directory": bool(getattr(os, "O_DIRECTORY", 0)),
        "o_nofollow": bool(getattr(os, "O_NOFOLLOW", 0)),
        "open_dir_fd": os.open in getattr(os, "supports_dir_fd", set()),
    }


def _sha256_file(
    path: Path,
    *,
    maximum_bytes: int,
    expected_identity: tuple[int, int, int, int, int] | None = None,
) -> str:
    before = path.stat(follow_symlinks=False)
    if expected_identity is not None and _stat_identity(before) != expected_identity:
        raise ValueError("file identity changed before content hashing")
    size = before.st_size
    if size > maximum_bytes:
        raise ValueError(
            f"file exceeds maximum_hash_bytes ({size} > {maximum_bytes})"
        )
    descriptor = _open_absolute_without_symlink_components(path, os.O_RDONLY)
    digest = sha256()
    try:
        opened_before = os.fstat(descriptor)
        if opened_before.st_nlink > 1:
            raise ValueError("hard-linked files are refused by the sealed-data guard")
        if _stat_identity(opened_before) != _stat_identity(before):
            raise ValueError("file identity changed before content hashing")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    if not (
        _stat_identity(opened_before)
        == _stat_identity(opened_after)
        == _stat_identity(after)
    ):
        raise ValueError("file identity or content metadata changed during hashing")
    return digest.hexdigest()


@contextmanager
def _open_nofollow(
    path: Path,
    *,
    binary: bool,
    expected_identity: tuple[int, int, int, int, int] | None = None,
):
    """Open one stable inode and reject symlink/hardlink substitutions."""

    before = path.stat(follow_symlinks=False)
    if expected_identity is not None and _stat_identity(before) != expected_identity:
        raise ValueError("file identity changed before inspection open")
    descriptor = _open_absolute_without_symlink_components(path, os.O_RDONLY)
    opened = os.fstat(descriptor)
    if opened.st_nlink > 1:
        os.close(descriptor)
        raise ValueError("hard-linked files are refused by the sealed-data guard")
    if _stat_identity(opened) != _stat_identity(before):
        os.close(descriptor)
        raise ValueError("file identity changed before inspection open")
    mode = "rb" if binary else "r"
    kwargs = {} if binary else {"encoding": "utf-8-sig", "newline": ""}
    handle = os.fdopen(descriptor, mode, **kwargs)
    try:
        yield handle
        opened_after = os.fstat(handle.fileno())
    finally:
        handle.close()
    after = path.stat(follow_symlinks=False)
    if not (
        _stat_identity(opened)
        == _stat_identity(opened_after)
        == _stat_identity(after)
    ):
        raise ValueError("file identity or metadata changed during inspection")


def _try_date(value: str) -> str | None:
    candidate = value.strip()[:10]
    try:
        return date.fromisoformat(candidate).isoformat()
    except ValueError:
        return None


def _inspect_delimited(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int, int],
) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    rows = 0
    minimum_date: str | None = None
    maximum_date: str | None = None
    malformed_rows = 0
    security_ids: set[str] = set()
    column_date_bounds: dict[str, dict[str, str]] = {}
    with _open_nofollow(
        path, binary=False, expected_identity=expected_identity
    ) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            columns = next(reader)
        except StopIteration:
            return {"rows": 0, "columns": [], "date_range": None}
        normalized = [str(item).strip() for item in columns]
        lower = [item.casefold() for item in normalized]
        date_index = next(
            (
                index
                for index, name in enumerate(lower)
                if name
                in {
                    "date",
                    "effective_date",
                    "effective_at",
                    "timestamp",
                    "datetime",
                }
            ),
            0 if normalized and lower[0] in {"", "date"} else None,
        )
        security_index = next(
            (
                index
                for index, name in enumerate(lower)
                if name in {"security_id", "source_security_id"}
            ),
            None,
        )
        date_indices = {
            index: name
            for index, name in enumerate(lower)
            if name
            in {
                "date",
                "snapshot_date",
                "effective_date",
                "effective_at",
                "valid_from",
                "valid_to",
                "known_at",
                "announcement_at",
                "settlement_at",
                "available_at",
                "timestamp",
                "datetime",
            }
        }
        for record in reader:
            if not record or not any(item.strip() for item in record):
                continue
            rows += 1
            if len(record) != len(normalized):
                malformed_rows += 1
            if date_index is not None and date_index < len(record):
                parsed = _try_date(record[date_index])
                if parsed is not None:
                    minimum_date = parsed if minimum_date is None else min(minimum_date, parsed)
                    maximum_date = parsed if maximum_date is None else max(maximum_date, parsed)
            if security_index is not None and security_index < len(record):
                security_id = record[security_index].strip()
                if security_id:
                    security_ids.add(security_id)
            for index, name in date_indices.items():
                if index >= len(record):
                    continue
                parsed = _try_date(record[index])
                if parsed is None:
                    continue
                bounds = column_date_bounds.setdefault(
                    name, {"start": parsed, "end": parsed}
                )
                bounds["start"] = min(bounds["start"], parsed)
                bounds["end"] = max(bounds["end"], parsed)
    return {
        "rows": rows,
        "columns": normalized,
        "date_range": (
            {"start": minimum_date, "end": maximum_date}
            if minimum_date is not None
            else None
        ),
        "malformed_rows": malformed_rows,
        "_security_ids": security_ids,
        "_column_date_bounds": column_date_bounds,
    }


def _inspect_parquet(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int, int],
) -> dict[str, Any]:
    try:
        import pyarrow.compute as pc  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        return {
            "rows": None,
            "columns": None,
            "date_range": None,
            "inspection_warning": "pyarrow unavailable; parquet schema not inspected",
        }
    with _open_nofollow(
        path, binary=True, expected_identity=expected_identity
    ) as handle:
        parquet = pq.ParquetFile(handle)
        columns = list(parquet.schema_arrow.names)
        rows = int(parquet.metadata.num_rows)
        date_column = next(
            (
                name
                for name in ("date", "effective_at", "timestamp", "datetime")
                if name in columns
            ),
            None,
        )
        date_range = None
        date_names = [
            name
            for name in (
                "date",
                "snapshot_date",
                "effective_date",
                "effective_at",
                "valid_from",
                "valid_to",
                "known_at",
                "announcement_at",
                "settlement_at",
                "available_at",
                "timestamp",
                "datetime",
            )
            if name in columns
        ]
        column_date_bounds: dict[str, dict[str, str]] = {}
        if date_names and rows:
            date_table = parquet.read(columns=date_names)
            for name in date_names:
                values = date_table[name]
                minimum = pc.min(values).as_py()
                maximum = pc.max(values).as_py()
                if minimum is not None and maximum is not None:
                    column_date_bounds[name] = {
                        "start": str(minimum)[:10],
                        "end": str(maximum)[:10],
                    }
        if date_column is not None and date_column in column_date_bounds:
            date_range = dict(column_date_bounds[date_column])
        security_ids: set[str] = set()
        security_column = next(
            (
                name
                for name in ("security_id", "source_security_id")
                if name in columns
            ),
            None,
        )
        if security_column is not None and rows:
            values = parquet.read(columns=[security_column])[security_column]
            security_ids = {
                str(value.as_py())
                for value in pc.unique(values)
                if value.as_py() is not None
            }
    return {
        "rows": rows,
        "columns": columns,
        "date_range": date_range,
        "_security_ids": security_ids,
        "_column_date_bounds": column_date_bounds,
    }


def _inspect_regular_file(
    path: Path,
    *,
    maximum_hash_bytes: int,
) -> dict[str, Any]:
    stat = path.stat(follow_symlinks=False)
    frozen_identity = _stat_identity(stat)
    if stat.st_nlink > 1:
        raise ValueError("hard-linked files are refused by the sealed-data guard")
    initial_sha256 = _sha256_file(
        path,
        maximum_bytes=maximum_hash_bytes,
        expected_identity=frozen_identity,
    )
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": int(stat.st_size),
        "sha256": initial_sha256,
        "format": path.suffix.casefold().lstrip(".") or "unknown",
        "rows": None,
        "columns": None,
        "date_range": None,
    }
    suffix = path.suffix.casefold()
    if suffix in {".csv", ".tsv"}:
        result.update(_inspect_delimited(path, expected_identity=frozen_identity))
    elif suffix == ".parquet":
        result.update(_inspect_parquet(path, expected_identity=frozen_identity))
    elif suffix == ".json" and stat.st_size <= 4 * 1024 * 1024:
        with _open_nofollow(
            path, binary=False, expected_identity=frozen_identity
        ) as handle:
            payload = json.load(handle)
        result["json_type"] = type(payload).__name__
        result["_json_payload"] = payload
    final_sha256 = _sha256_file(
        path,
        maximum_bytes=maximum_hash_bytes,
        expected_identity=frozen_identity,
    )
    if final_sha256 != initial_sha256:
        raise ValueError("file content changed during inspection")
    return result


def _candidate_file(path: Path) -> bool:
    if path.suffix.casefold() not in {".csv", ".tsv", ".json", ".parquet", ".html"}:
        return False
    name = path.name.casefold()
    terms = (
        "sp500",
        "s&p500",
        "s&p_500",
        "membership",
        "constituent",
        "security_master",
        "security_id",
        "identifier_alias",
        "delist",
        "terminal",
        "corporate_action",
        "sector",
        "classification",
        "benchmark_weight",
        "weight",
        "price",
        "evidence",
        "readiness_evidence",
    )
    return any(term in name for term in terms)


def _discover_explicit_path(
    path: Path,
    *,
    maximum_files: int,
) -> tuple[list[Path], list[dict[str, str]]]:
    resolved, reason = _safe_resolve(path)
    if reason or resolved is None:
        return [], [{"path": str(path), "reason": str(reason)}]
    if not resolved.exists():
        return [], [{"path": str(resolved), "reason": "path does not exist"}]
    if resolved.is_file():
        return [resolved], []
    if not resolved.is_dir():
        return [], [{"path": str(resolved), "reason": "not a regular file or directory"}]

    files: list[Path] = []
    skipped: list[dict[str, str]] = []
    for root, directories, filenames in os.walk(resolved, followlinks=False):
        root_path = Path(root)
        allowed_directories: list[str] = []
        for name in sorted(directories):
            child = root_path / name
            _, child_reason = _safe_resolve(child)
            if child_reason:
                skipped.append({"path": str(child), "reason": child_reason})
            else:
                allowed_directories.append(name)
        directories[:] = allowed_directories
        for name in sorted(filenames):
            child = root_path / name
            if not _candidate_file(child):
                continue
            safe, child_reason = _safe_resolve(child)
            if child_reason or safe is None:
                skipped.append({"path": str(child), "reason": str(child_reason)})
                continue
            files.append(safe)
            if len(files) >= maximum_files:
                skipped.append(
                    {
                        "path": str(resolved),
                        "reason": f"discovery capped at {maximum_files} files",
                    }
                )
                return files, skipped
    return files, skipped


def _desktop_root(project_dir: Path) -> Path:
    for candidate in (project_dir, *project_dir.parents):
        if candidate.name.casefold() == "desktop":
            return candidate
    return Path.home() / "Desktop"


def _known_asset_paths(project_dir: Path) -> list[tuple[Path, _KnownAssetSpec]]:
    desktop = _desktop_root(project_dir)
    found: list[tuple[Path, _KnownAssetSpec]] = []
    for spec in _KNOWN_ASSET_SPECS:
        path = desktop / spec.relative_to_desktop
        safe, reason = _safe_resolve(path)
        if reason is None and safe is not None and safe.is_file():
            found.append((safe, spec))
    return found


def _generic_asset_classification(metadata: Mapping[str, Any]) -> dict[str, Any]:
    columns = {str(item).casefold() for item in metadata.get("columns") or []}
    name = Path(str(metadata["path"])).name.casefold()
    supports: set[str] = set()
    scope = "unknown_unverified"
    limitations = [
        "no admissible CogAlpha S&P 500 readiness evidence descriptor",
        "schema resemblance alone cannot establish point-in-time completeness",
    ]
    if {"open", "high", "low", "close", "volume"}.issubset(columns):
        supports.add("price_panel")
    if "event_type" in columns and {"effective_at", "security_id"}.issubset(columns):
        supports.add("historical_membership")
    if {"security_id", "identifier", "identifier_type"}.issubset(columns):
        supports.add("stable_security_id")
    if "action_type" in columns:
        supports.add("corporate_actions")
    if "classification" in columns or "sector" in columns or "gics sector" in columns:
        supports.add("pit_sector")
    if "weight" in columns:
        supports.add("benchmark_weights")
    if "delist" in name or "terminal" in name:
        supports.add("terminal_delisting_payoff")
    if "current" in name or "constituent" in name:
        scope = "current_only"
        limitations.append("filename or schema indicates a current-only snapshot")
    if "constituent" in name:
        supports.add("historical_membership")
        if "cik" in columns or "security_id" in columns:
            supports.add("stable_security_id")
    return {
        "asset_type": "unverified_local_asset",
        "scope": scope,
        "supports": sorted(supports),
        "provenance": "user-supplied local path; authority not established",
        "limitations": limitations,
    }


def _descriptor_schema(payload: Mapping[str, Any]) -> set[str]:
    value = payload.get("schema")
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, Mapping) and isinstance(value.get("columns"), list):
        return {str(item) for item in value["columns"]}
    return set()


def _descriptor_identity_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("replay_identity_hash", None)
    return hash_json(body)


def _validate_descriptor(
    payload: Mapping[str, Any],
    descriptor_path: Path,
    config: SP500ReadinessConfig,
    file_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    domain = str(payload.get("domain", ""))
    if payload.get("artifact_type") != EVIDENCE_ARTIFACT_TYPE:
        failures.append(f"artifact_type must be {EVIDENCE_ARTIFACT_TYPE!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        failures.append(f"schema_version must equal {SCHEMA_VERSION}")
    if domain not in REQUIRED_DOMAINS:
        failures.append(f"domain must be one of {list(REQUIRED_DOMAINS)}")
    if str(payload.get("universe_id", "")).casefold() != "sp500":
        failures.append("universe_id must be exactly 'sp500'")
    if payload.get("complete") is not True:
        failures.append("complete must be true")
    if payload.get("point_in_time") is not True:
        failures.append("point_in_time must be true")
    if str(payload.get("quality_tier", "")).casefold() not in {
        "primary",
        "capital_grade",
        "capital_grade_primary",
    }:
        failures.append("quality_tier must be primary/capital_grade")

    source = payload.get("source")
    if not isinstance(source, Mapping):
        failures.append("source must be an object")
        source = {}
    for key in ("vendor", "dataset", "version", "extracted_at"):
        if not isinstance(source.get(key), str) or not str(source.get(key)).strip():
            failures.append(f"source.{key} must be a non-empty string")
    try:
        extracted_at = _parse_iso_timestamp(
            source.get("extracted_at"), "source.extracted_at"
        )
        if extracted_at.date() < _parse_iso_date(config.required_end, "required_end"):
            failures.append(
                "source.extracted_at predates required_end; end-of-window completeness "
                "cannot be established"
            )
    except ValueError as exc:
        failures.append(str(exc))

    scope = str(payload.get("scope", "")).casefold()
    if scope in _CURRENT_ONLY_SCOPES:
        failures.append("current-only/latest scope is never point-in-time ready")
    elif domain in _HISTORICAL_SCOPES and scope not in _HISTORICAL_SCOPES[domain]:
        failures.append(
            f"scope {scope!r} is not an accepted historical scope for {domain}"
        )

    primary_key = payload.get("primary_key")
    required_key = (
        "source_security_id" if domain == "terminal_delisting_payoff" else "security_id"
    )
    if not isinstance(primary_key, list) or required_key not in primary_key:
        failures.append(f"primary_key must be a list containing {required_key}")
    if isinstance(primary_key, list) and any(
        str(item).casefold() in {"ticker", "symbol"} for item in primary_key
    ):
        failures.append("ticker/symbol cannot be a primary-key field")

    date_range = payload.get("date_range")
    if not isinstance(date_range, Mapping):
        failures.append("date_range must contain start and end coverage dates")
    else:
        try:
            coverage_start = _parse_iso_date(str(date_range.get("start", "")), "date_range.start")
            coverage_end = _parse_iso_date(str(date_range.get("end", "")), "date_range.end")
            if coverage_start > _parse_iso_date(config.required_start, "required_start"):
                failures.append(
                    f"coverage starts {coverage_start}, after required {config.required_start}"
                )
            if coverage_end < _parse_iso_date(config.required_end, "required_end"):
                failures.append(
                    f"coverage ends {coverage_end}, before required {config.required_end}"
                )
        except ValueError as exc:
            failures.append(str(exc))

    schema = _descriptor_schema(payload)
    if domain in _DOMAIN_REQUIRED_COLUMNS:
        missing_columns = sorted(_DOMAIN_REQUIRED_COLUMNS[domain] - schema)
        if missing_columns:
            failures.append(f"schema is missing required columns: {missing_columns}")

    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        failures.append("contract must be an object")
        contract = {}
    required_flags = (
        *_COMMON_REQUIRED_CONTRACT_FLAGS,
        *_DOMAIN_REQUIRED_CONTRACT_FLAGS.get(domain, ()),
    )
    false_flags = [flag for flag in required_flags if contract.get(flag) is not True]
    if false_flags:
        failures.append(f"contract flags must be true: {false_flags}")
    if domain == "benchmark_weights":
        tolerance = contract.get("weight_sum_tolerance")
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or tolerance < 0
            or tolerance > 1e-8
        ):
            failures.append("benchmark weight_sum_tolerance must be between 0 and 1e-8")

    expected_identity = _descriptor_identity_hash(payload)
    if payload.get("replay_identity_hash") != expected_identity:
        failures.append("replay_identity_hash does not match the canonical descriptor body")

    raw_objects = payload.get("raw_objects")
    verified_objects: list[dict[str, Any]] = []
    observed_roles: set[str] = set()
    observed_security_ids: set[str] = set()
    observed_columns: set[str] = set()
    role_date_bounds: dict[str, dict[str, dict[str, str]]] = {}
    observed_raw_paths: set[Path] = set()
    if not isinstance(raw_objects, list) or not raw_objects:
        failures.append("raw_objects must be a non-empty list")
        raw_objects = []
    for index, raw in enumerate(raw_objects):
        prefix = f"raw_objects[{index}]"
        if not isinstance(raw, Mapping):
            failures.append(f"{prefix} must be an object")
            continue
        raw_path_value = raw.get("path")
        if not isinstance(raw_path_value, str) or not raw_path_value.strip():
            failures.append(f"{prefix}.path must be non-empty")
            continue
        roles_value = raw.get("roles", raw.get("role"))
        roles = (
            [str(item) for item in roles_value]
            if isinstance(roles_value, list)
            else [str(roles_value)]
            if isinstance(roles_value, str)
            else []
        )
        if len(roles) != 1 or any(not role.strip() for role in roles):
            failures.append(
                f"{prefix} must declare exactly one non-empty raw object role"
            )
        observed_roles.update(role for role in roles if role)
        raw_path = Path(raw_path_value)
        if not raw_path.is_absolute():
            raw_path = descriptor_path.parent / raw_path
        safe, reason = _safe_resolve(raw_path)
        if reason or safe is None:
            failures.append(f"{prefix} refused: {reason}")
            continue
        if not safe.is_file():
            failures.append(f"{prefix} is not an existing regular file")
            continue
        if safe in observed_raw_paths:
            failures.append(f"{prefix} duplicates a raw path used by another role")
            continue
        observed_raw_paths.add(safe)
        recorded_sha = str(raw.get("sha256", "")).casefold()
        if not _HASH_RE.fullmatch(recorded_sha):
            failures.append(f"{prefix}.sha256 must be a lowercase SHA-256 digest")
            continue
        try:
            metadata = file_cache.get(safe)
            if metadata is None:
                metadata = _inspect_regular_file(
                    safe, maximum_hash_bytes=config.maximum_hash_bytes
                )
                file_cache[safe] = metadata
        except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
            failures.append(f"{prefix} could not be verified: {type(exc).__name__}: {exc}")
            continue
        if metadata["sha256"] != recorded_sha:
            failures.append(f"{prefix} SHA-256 mismatch")
        if metadata.get("rows") is None or metadata.get("columns") is None:
            failures.append(
                f"{prefix} format/schema could not be parsed; fail-closed evidence rejected"
            )
        if int(metadata.get("malformed_rows") or 0) > 0:
            failures.append(
                f"{prefix} contains {metadata['malformed_rows']} malformed row(s)"
            )
        if "bytes" in raw and raw.get("bytes") != metadata["bytes"]:
            failures.append(f"{prefix} byte count mismatch")
        if (
            "rows" not in raw
            or not isinstance(raw.get("rows"), int)
            or isinstance(raw.get("rows"), bool)
            or raw.get("rows", 0) < 1
        ):
            failures.append(f"{prefix}.rows must be recorded as a positive integer")
        elif metadata.get("rows") is not None and raw["rows"] != metadata["rows"]:
            failures.append(f"{prefix} row count mismatch")
        declared_columns = raw.get("columns")
        if not isinstance(declared_columns, list):
            failures.append(f"{prefix}.columns must be a list")
        elif metadata.get("columns") is not None and set(declared_columns) != set(
            metadata["columns"]
        ):
            failures.append(f"{prefix} column schema mismatch")
        observed_security_ids.update(metadata.get("_security_ids", set()))
        observed_columns.update(str(item) for item in metadata.get("columns") or [])
        for role in roles:
            by_column = role_date_bounds.setdefault(role, {})
            for column, bounds in metadata.get("_column_date_bounds", {}).items():
                combined = by_column.setdefault(column, dict(bounds))
                combined["start"] = min(combined["start"], bounds["start"])
                combined["end"] = max(combined["end"], bounds["end"])
        verified_objects.append(
            {
                "path": str(safe),
                "sha256": metadata["sha256"],
                "bytes": metadata["bytes"],
                "rows": metadata.get("rows"),
                "columns": metadata.get("columns"),
                "date_range": metadata.get("date_range"),
                "roles": roles,
                "distinct_security_ids": len(metadata.get("_security_ids", set())),
                "column_date_bounds": metadata.get("_column_date_bounds", {}),
            }
        )

    missing_roles = sorted(
        _DOMAIN_REQUIRED_OBJECT_ROLES.get(domain, frozenset()) - observed_roles
    )
    if missing_roles:
        failures.append(f"raw_objects are missing required roles: {missing_roles}")
    missing_actual_columns = sorted(schema - observed_columns)
    if missing_actual_columns:
        failures.append(
            "descriptor schema columns are absent from all parsed raw objects: "
            f"{missing_actual_columns}"
        )
    if (
        domain in _CROSS_SECTIONAL_DOMAINS
        and len(observed_security_ids) < config.minimum_security_count
    ):
        failures.append(
            "verified raw objects contain "
            f"{len(observed_security_ids)} distinct security IDs; "
            f"minimum_security_count is {config.minimum_security_count}"
        )

    required_start = _parse_iso_date(config.required_start, "required_start")
    required_end = _parse_iso_date(config.required_end, "required_end")

    def require_role_coverage(
        role: str,
        *,
        start_columns: tuple[str, ...],
        end_columns: tuple[str, ...],
        require_start: bool = True,
        require_end: bool = True,
    ) -> None:
        by_column = role_date_bounds.get(role, {})
        starts = [
            bounds["start"]
            for column in start_columns
            if (bounds := by_column.get(column)) is not None
        ]
        ends = [
            bounds["end"]
            for column in end_columns
            if (bounds := by_column.get(column)) is not None
        ]
        if require_start and (
            not starts or _parse_iso_date(min(starts), f"{role}.start") > required_start
        ):
            failures.append(
                f"raw role {role!r} does not verify coverage at required_start "
                f"{config.required_start}"
            )
        if require_end and (
            not ends or _parse_iso_date(max(ends), f"{role}.end") < required_end
        ):
            failures.append(
                f"raw role {role!r} does not verify coverage at required_end "
                f"{config.required_end}"
            )

    if domain == "price_panel":
        require_role_coverage(
            "raw_prices",
            start_columns=("date", "effective_date", "effective_at"),
            end_columns=("date", "effective_date", "effective_at"),
        )
    elif domain == "historical_membership":
        require_role_coverage(
            "initial_membership_snapshot",
            start_columns=("snapshot_date", "date", "effective_date", "effective_at"),
            end_columns=(),
            require_end=False,
        )
    elif domain == "stable_security_id":
        require_role_coverage(
            "security_master",
            start_columns=("valid_from",),
            end_columns=("valid_to",),
        )
    elif domain == "pit_sector":
        require_role_coverage(
            "classification_history",
            start_columns=("valid_from",),
            end_columns=("valid_to",),
        )
    elif domain == "benchmark_weights":
        require_role_coverage(
            "benchmark_weights",
            start_columns=("date", "effective_date", "effective_at"),
            end_columns=("date", "effective_date", "effective_at"),
        )

    return {
        "domain": domain or None,
        "descriptor_path": str(descriptor_path),
        "descriptor_sha256": file_cache[descriptor_path]["sha256"],
        "valid": not failures,
        "failures": failures,
        "source": dict(source),
        "quality_tier": payload.get("quality_tier"),
        "scope": payload.get("scope"),
        "date_range": dict(date_range) if isinstance(date_range, Mapping) else None,
        "schema": sorted(schema),
        "raw_objects": verified_objects,
        "replay_identity_hash": payload.get("replay_identity_hash"),
    }


def _load_possible_descriptor(
    path: Path,
    metadata: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if path.suffix.casefold() != ".json" or metadata.get("bytes", 0) > 4 * 1024 * 1024:
        return None
    payload = metadata.get("_json_payload")
    if not isinstance(payload, Mapping):
        return None
    if payload.get("artifact_type") != EVIDENCE_ARTIFACT_TYPE:
        return None
    return payload


def audit_sp500_local_readiness(
    config: SP500ReadinessConfig,
) -> SP500ReadinessReport:
    """Audit local assets without network access and return a fail-closed report."""

    file_cache: dict[Path, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    known_specs_by_path: dict[Path, _KnownAssetSpec] = {}
    discovered: list[Path] = []

    if config.include_known_assets:
        for path, spec in _known_asset_paths(config.project_dir):
            known_specs_by_path[path] = spec
            discovered.append(path)

    for requested in config.asset_paths:
        paths, path_skipped = _discover_explicit_path(
            requested,
            maximum_files=config.maximum_discovered_files,
        )
        discovered.extend(paths)
        skipped.extend(path_skipped)

    unique_paths = sorted(set(discovered), key=lambda item: str(item))
    assets: list[dict[str, Any]] = []
    descriptor_payloads: list[tuple[Path, Mapping[str, Any]]] = []

    for path in unique_paths:
        try:
            metadata = file_cache.get(path)
            if metadata is None:
                metadata = _inspect_regular_file(
                    path, maximum_hash_bytes=config.maximum_hash_bytes
                )
                file_cache[path] = metadata
            descriptor = _load_possible_descriptor(path, metadata)
        except (
            OSError,
            RuntimeError,
            UnicodeError,
            csv.Error,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            skipped.append(
                {
                    "path": str(path),
                    "reason": f"inspection failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        if descriptor is not None:
            descriptor_payloads.append((path, descriptor))
            classification = {
                "asset_type": "readiness_evidence_descriptor",
                "scope": descriptor.get("scope", "unknown"),
                "supports": [descriptor.get("domain")]
                if descriptor.get("domain") in REQUIRED_DOMAINS
                else [],
                "provenance": "explicit CogAlpha readiness evidence descriptor",
                "limitations": [
                    "descriptor integrity is not US PIT semantic validation",
                    "not admissible for factor search until us_index_contract replay passes",
                ],
            }
            asset_id = f"evidence_{metadata['sha256'][:16]}"
        elif path in known_specs_by_path:
            spec = known_specs_by_path[path]
            classification = {
                "asset_type": spec.asset_type,
                "scope": spec.scope,
                "supports": list(spec.supports),
                "provenance": spec.provenance,
                "limitations": list(spec.limitations),
            }
            asset_id = spec.asset_id
        else:
            classification = _generic_asset_classification(metadata)
            asset_id = f"local_{metadata['sha256'][:16]}"
        public_metadata = {
            key: value for key, value in metadata.items() if not key.startswith("_")
        }
        assets.append(
            {
                "asset_id": asset_id,
                **public_metadata,
                **classification,
                "quality_tier": (
                    "candidate_primary_evidence"
                    if descriptor is not None
                    else "secondary_validation"
                ),
                "descriptor_integrity_valid": False,
                "admissible_as_primary": False,
            }
        )

    descriptor_results = [
        _validate_descriptor(payload, path, config, file_cache)
        for path, payload in descriptor_payloads
    ]
    valid_descriptor_paths = {
        result["descriptor_path"] for result in descriptor_results if result["valid"]
    }
    for asset in assets:
        if asset["path"] in valid_descriptor_paths:
            asset["descriptor_integrity_valid"] = True
            asset["quality_tier"] = "descriptor_integrity_evidence"

    domains: dict[str, dict[str, Any]] = {}
    for domain in REQUIRED_DOMAINS:
        candidates = [result for result in descriptor_results if result["domain"] == domain]
        accepted = [result for result in candidates if result["valid"]]
        secondary = [
            {
                "asset_id": asset["asset_id"],
                "path": asset["path"],
                "scope": asset["scope"],
                "reason": "; ".join(asset["limitations"]),
            }
            for asset in assets
            if domain in asset["supports"] and not asset["admissible_as_primary"]
        ]
        failures = [
            failure
            for candidate in candidates
            for failure in candidate["failures"]
        ]
        if not candidates:
            failures.append("no explicit admissible readiness evidence descriptor found")
        if accepted:
            failures.append(
                "descriptor integrity passed, but us_index_contract semantic replay "
                "has not been executed; domain remains NOT_READY"
            )
        domains[domain] = {
            "label": DOMAIN_LABELS[domain],
            "status": "NOT_READY",
            "ready": False,
            "descriptor_integrity_valid": bool(accepted),
            "us_index_contract_semantics_valid": False,
            "accepted_evidence": accepted,
            "rejected_evidence": [result for result in candidates if not result["valid"]],
            "failures": failures,
            "secondary_validation": secondary,
        }

    evidence_contract_complete = all(
        domains[name]["descriptor_integrity_valid"] for name in REQUIRED_DOMAINS
    )
    data_contract_ready = False
    missing_required_domains = list(REQUIRED_DOMAINS)
    secondary_validation = [
        {
            "asset_id": asset["asset_id"],
            "path": asset["path"],
            "supports": asset["supports"],
            "scope": asset["scope"],
            "limitations": asset["limitations"],
        }
        for asset in assets
        if asset["supports"]
    ]
    implementation_sha256 = sha256(Path(__file__).read_bytes()).hexdigest()
    secure_open_capabilities = _secure_open_capabilities()
    audit_contract_hash = hash_json(
        {
            "required_domains": list(REQUIRED_DOMAINS),
            "required_columns": {
                name: sorted(columns)
                for name, columns in _DOMAIN_REQUIRED_COLUMNS.items()
            },
            "required_contract_flags": {
                name: list(flags)
                for name, flags in _DOMAIN_REQUIRED_CONTRACT_FLAGS.items()
            },
            "required_object_roles": {
                name: sorted(roles)
                for name, roles in _DOMAIN_REQUIRED_OBJECT_ROLES.items()
            },
            "current_only_scopes": sorted(_CURRENT_ONLY_SCOPES),
        }
    )
    identity_body = {
        "auditor": AUDITOR_ID,
        "auditor_implementation_sha256": implementation_sha256,
        "audit_contract_hash": audit_contract_hash,
        "required_us_index_event_day_order_sha256": US_INDEX_EVENT_DAY_ORDER_SHA256,
        "secure_open_capabilities": secure_open_capabilities,
        "schema_version": SCHEMA_VERSION,
        "universe_id": "sp500",
        "requirements": {
            "required_start": config.required_start,
            "required_end": config.required_end,
            "minimum_security_count": config.minimum_security_count,
            "required_domains": list(REQUIRED_DOMAINS),
        },
        "asset_objects": [
            {
                "path": asset["path"],
                "sha256": asset["sha256"],
                "bytes": asset["bytes"],
                "scope": asset["scope"],
            }
            for asset in assets
        ],
        "domain_evidence": {
            name: [
                {
                    "descriptor_sha256": item["descriptor_sha256"],
                    "valid": item["valid"],
                    "replay_identity_hash": item["replay_identity_hash"],
                    "raw_objects": [raw["sha256"] for raw in item["raw_objects"]],
                }
                for item in domains[name]["accepted_evidence"]
                + domains[name]["rejected_evidence"]
            ]
            for name in REQUIRED_DOMAINS
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "auditor": AUDITOR_ID,
        "auditor_implementation_sha256": implementation_sha256,
        "audit_contract_hash": audit_contract_hash,
        "created_at_utc": utc_now(),
        "universe_id": "sp500",
        "status": "NOT_READY",
        "readiness_scope": "inventory_and_descriptor_integrity_only",
        "ready": False,
        "evidence_contract_complete": evidence_contract_complete,
        "structural_evidence_contract_ready": evidence_contract_complete,
        "us_index_contract_validated": False,
        "required_us_index_event_day_order_sha256": US_INDEX_EVENT_DAY_ORDER_SHA256,
        "data_contract_ready": False,
        "factor_search_allowed": False,
        "capital_grade": False,
        "capital_deployment_allowed": False,
        "production_certification": False,
        "economic_content_correctness_certified": False,
        "point_in_time_panel_ready": False,
        "readiness_blockers": [
            "semantic_validation_not_implemented",
            "cross_domain_security_id_foreign_keys_not_validated",
            "daily_membership_replay_not_validated",
            "daily_benchmark_weight_sums_not_validated",
            "corporate_action_and_terminal_payoff_replay_not_validated",
            "cross_sectional_and_execution_tape_coverage_not_validated",
        ],
        "missing_required_domains": missing_required_domains,
        "primary_readiness_evidence_found": 0,
        "descriptor_integrity_evidence_found": sum(
            len(domains[name]["accepted_evidence"]) for name in REQUIRED_DOMAINS
        ),
        "secondary_validation_only": bool(secondary_validation),
        "evidence_contract_limitations": [
            "A passing descriptor proves local hash, parsed schema, bounded coverage checks, and required declarations only; source authority and economic truth remain unverified.",
            "Descriptor integrity cannot unlock factor search. Security aliases, membership replay, corporate-action legs, event ordering, and execution-tape retention must pass us_index_contract first.",
            "Even a future data-contract pass cannot certify a model, backtest, portfolio, execution path, or live-trading system.",
        ],
        "required_downstream_production_gates": [
            "model_validation",
            "purged_walk_forward_backtest",
            "transaction_cost_and_capacity_stress",
            "portfolio_risk_controls",
            "execution_integration",
            "paper_trading",
            "operational_monitoring",
        ],
        "requirements": identity_body["requirements"],
        "current_only_policy": (
            "Current/latest constituent, ticker, sector, or weight snapshots are "
            "secondary validation only and never satisfy point-in-time readiness."
        ),
        "domains": domains,
        "assets": assets,
        "secondary_validation": secondary_validation,
        "skipped_paths": skipped,
        "network_used": False,
        "forbidden_path_guard_assertion": (
            "all user-data opens are attempted only after lexical/resolved path "
            "screening and through no-follow directory descriptors"
        ),
        "forbidden_paths_skipped_count": sum(
            "forbidden" in item["reason"] or "hard-linked" in item["reason"]
            for item in skipped
        ),
        "sealed_path_guard": {
            "lexical_and_resolved_path_filter": True,
            "symlinks_rejected": True,
            "hardlinks_rejected": True,
            "secure_open_capabilities": secure_open_capabilities,
            "all_required_capabilities_available": all(
                secure_open_capabilities.values()
            ),
            "hash_open_uses_no_follow": secure_open_capabilities["o_nofollow"],
            "inode_and_metadata_rechecked": True,
            "scope_limit": "guards inspected filesystem paths; copied bytes have no intrinsic sealed provenance marker",
        },
        "identity_hash": hash_json(identity_body),
    }
    return SP500ReadinessReport(manifest=manifest)


__all__ = [
    "AUDITOR_ID",
    "DOMAIN_LABELS",
    "EVIDENCE_ARTIFACT_TYPE",
    "REQUIRED_DOMAINS",
    "SCHEMA_VERSION",
    "SP500ReadinessConfig",
    "SP500ReadinessReport",
    "audit_sp500_local_readiness",
]
