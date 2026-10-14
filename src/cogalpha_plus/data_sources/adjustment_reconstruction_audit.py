"""Offline gate for replacing direct adjusted histories with factor reconstruction.

The audit is intentionally read-only.  It verifies the committed BaoStock raw
cache and normalized request catalogs, reconstructs ``adjustflag=1`` prices one
security at a time from ``adjustflag=3`` bars plus ``query_adjust_factor``, and
compares every OHLC/preclose cell with the independently cached direct vendor
history.  A future downloader may omit the direct adjusted history only when
this report is valid for the complete historical constituent union.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .baostock import (
    ADJUST_FACTOR_FIELDS,
    EQUITY_DAILY_FIELDS,
    CacheIntegrityError,
    make_request_key,
)
from .hs300 import reconstruct_backward_adjusted_view


AUDIT_SCHEMA_VERSION = "1.0"
COMPARISON_FIELDS = ("open", "high", "low", "close", "preclose")
DEFAULT_RELATIVE_TOLERANCE = 2e-6
DEFAULT_ABSOLUTE_TOLERANCE = 2e-5
REQUIRED_COVERAGE_FRACTION = 1.0
_REQUEST_KEY = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AdjustmentReconstructionAuditConfig:
    """Frozen limits and local paths for the offline reconstruction gate.

    Threshold overrides may only tighten the registered defaults.  A looser
    tolerance would change the decision rule after seeing the data.
    """

    cache_dir: Path | str
    normalized_dir: Path | str
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE
    maximum_failure_samples: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir).resolve())
        object.__setattr__(
            self, "normalized_dir", Path(self.normalized_dir).resolve()
        )
        relative = float(self.relative_tolerance)
        absolute = float(self.absolute_tolerance)
        samples = int(self.maximum_failure_samples)
        if (
            not math.isfinite(relative)
            or relative <= 0
            or relative > DEFAULT_RELATIVE_TOLERANCE
        ):
            raise ValueError(
                "relative_tolerance must be positive and no looser than "
                f"{DEFAULT_RELATIVE_TOLERANCE:g}"
            )
        if (
            not math.isfinite(absolute)
            or absolute < 0
            or absolute > DEFAULT_ABSOLUTE_TOLERANCE
        ):
            raise ValueError(
                "absolute_tolerance must be non-negative and no looser than "
                f"{DEFAULT_ABSOLUTE_TOLERANCE:g}"
            )
        if samples < 1:
            raise ValueError("maximum_failure_samples must be positive")
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "absolute_tolerance", absolute)
        object.__setattr__(self, "maximum_failure_samples", samples)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CacheIntegrityError(f"JSON payload is not an object: {path}")
    return payload


def _verify_declared_file(path: Path, metadata: object, *, label: str) -> None:
    if not isinstance(metadata, Mapping):
        raise CacheIntegrityError(f"{label} file metadata is missing")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        declared_bytes = int(metadata.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise CacheIntegrityError(f"{label} byte count is invalid") from exc
    if declared_bytes != path.stat().st_size:
        raise CacheIntegrityError(f"{label} byte count differs from manifest")
    declared_hash = str(metadata.get("sha256", ""))
    if not _REQUEST_KEY.fullmatch(declared_hash):
        raise CacheIntegrityError(f"{label} manifest hash is invalid")
    if _sha256_file(path) != declared_hash:
        raise CacheIntegrityError(f"{label} hash differs from manifest")


def _validate_complete_manifests(
    normalized_dir: Path,
) -> tuple[
    list[str],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    dict[str, object],
    dict[str, str],
]:
    membership_manifest = _read_json_object(
        normalized_dir / "membership_manifest.json"
    )
    price_manifest = _read_json_object(normalized_dir / "price_manifest.json")
    adjustment_manifest = _read_json_object(
        normalized_dir / "adjustment_manifest.json"
    )
    required_phases = (
        (membership_manifest, "membership_catalog", "membership"),
        (price_manifest, "price_catalog", "price"),
        (
            adjustment_manifest,
            "adjustment_factor_catalog",
            "adjustment-factor",
        ),
    )
    for manifest, phase, label in required_phases:
        if manifest.get("phase") != phase or manifest.get("complete") is not True:
            raise CacheIntegrityError(f"{label} manifest is not complete")

    union_path = normalized_dir / "security_union.parquet"
    membership_files = membership_manifest.get("files")
    union_metadata = (
        membership_files.get("security_union")
        if isinstance(membership_files, Mapping)
        else None
    )
    _verify_declared_file(union_path, union_metadata, label="security_union")
    union = pd.read_parquet(union_path)
    if tuple(union.columns.astype(str))[:1] != ("code",):
        raise CacheIntegrityError("security_union must start with the code column")
    codes = union["code"].astype("string").str.strip()
    if codes.isna().any() or codes.eq("").any() or codes.duplicated().any():
        raise CacheIntegrityError("security_union codes are invalid or duplicate")
    symbols = sorted(codes.astype(str).tolist())

    price_index_path = normalized_dir / "price_request_index.parquet"
    price_files = price_manifest.get("files")
    price_metadata = (
        price_files.get("price_request_index")
        if isinstance(price_files, Mapping)
        else None
    )
    _verify_declared_file(
        price_index_path, price_metadata, label="price_request_index"
    )
    price_index = pd.read_parquet(price_index_path)

    adjustment_index_path = normalized_dir / "adjustment_request_index.parquet"
    _verify_declared_file(
        adjustment_index_path,
        adjustment_manifest.get("file"),
        label="adjustment_request_index",
    )
    adjustment_index = pd.read_parquet(adjustment_index_path)

    expected_symbols = len(symbols)
    declared_counts = (
        (membership_manifest, "historical_union_symbols"),
        (price_manifest, "historical_union_symbols"),
        (price_manifest, "downloaded_symbols"),
        (adjustment_manifest, "historical_union_symbols"),
        (adjustment_manifest, "downloaded_symbols"),
        (adjustment_manifest, "adjustment_factor_requests"),
    )
    for manifest, key in declared_counts:
        try:
            observed = int(manifest.get(key, -1))
        except (TypeError, ValueError) as exc:
            raise CacheIntegrityError(f"manifest count {key} is invalid") from exc
        if observed != expected_symbols:
            raise CacheIntegrityError(
                f"manifest count {key}={observed} differs from union={expected_symbols}"
            )
    if int(price_manifest.get("price_requests", -1)) != 2 * expected_symbols:
        raise CacheIntegrityError("price manifest does not declare two requests per symbol")

    price_start = str(price_manifest.get("start_date", ""))
    price_end = str(price_manifest.get("end_date", ""))
    adjustment_start = str(adjustment_manifest.get("start_date", ""))
    adjustment_end = str(adjustment_manifest.get("end_date", ""))
    try:
        price_start_date = pd.Timestamp(price_start).normalize()
        price_end_date = pd.Timestamp(price_end).normalize()
        adjustment_start_date = pd.Timestamp(adjustment_start).normalize()
        adjustment_end_date = pd.Timestamp(adjustment_end).normalize()
    except Exception as exc:
        raise CacheIntegrityError("source manifests contain invalid date bounds") from exc
    if (
        price_start_date > price_end_date
        or adjustment_start_date > price_start_date
        or adjustment_end_date != price_end_date
    ):
        raise CacheIntegrityError(
            "adjustment range must begin no later than prices and share the price end"
        )

    required_price_columns = {"code", "adjustflag", "rows", "request_key"}
    missing_price = sorted(required_price_columns.difference(price_index.columns))
    if missing_price:
        raise CacheIntegrityError(
            "price_request_index missing: " + ",".join(missing_price)
        )
    price_index = price_index.copy()
    price_index["code"] = price_index["code"].astype(str)
    price_index["adjustflag"] = price_index["adjustflag"].astype(str)
    if price_index.duplicated(["code", "adjustflag"]).any():
        raise CacheIntegrityError("price_request_index has duplicate code/flag rows")
    if set(price_index["code"]) != set(symbols):
        raise CacheIntegrityError("price_request_index symbol set differs from union")
    observed_flags = price_index.groupby("code")["adjustflag"].agg(set)
    if not observed_flags.map(lambda value: value == {"1", "3"}).all():
        raise CacheIntegrityError("every symbol requires flags 1 and 3 exactly once")

    required_adjustment_columns = {"code", "rows", "request_key"}
    missing_adjustment = sorted(
        required_adjustment_columns.difference(adjustment_index.columns)
    )
    if missing_adjustment:
        raise CacheIntegrityError(
            "adjustment_request_index missing: " + ",".join(missing_adjustment)
        )
    adjustment_index = adjustment_index.copy()
    adjustment_index["code"] = adjustment_index["code"].astype(str)
    if adjustment_index["code"].duplicated().any():
        raise CacheIntegrityError("adjustment_request_index has duplicate symbols")
    if set(adjustment_index["code"]) != set(symbols):
        raise CacheIntegrityError(
            "adjustment_request_index symbol set differs from union"
        )

    dates = {
        "price_start": str(price_start_date.date()),
        "price_end": str(price_end_date.date()),
        "adjustment_start": str(adjustment_start_date.date()),
        "adjustment_end": str(adjustment_end_date.date()),
    }
    return (
        symbols,
        price_index,
        adjustment_index,
        price_manifest,
        adjustment_manifest,
        dates,
    )


def _read_committed_object(
    cache_dir: Path,
    request_key: object,
    *,
    expected_endpoint: str,
    expected_args: Mapping[str, object],
    expected_fields: Sequence[str],
    expected_rows: object,
) -> tuple[pd.DataFrame, str, str]:
    key = str(request_key)
    if not _REQUEST_KEY.fullmatch(key):
        raise CacheIntegrityError("request index contains an invalid request key")
    state_path = cache_dir / "state" / f"{key}.json"
    state = _read_json_object(state_path)
    if state.get("status") != "success" or state.get("request_key") != key:
        raise CacheIntegrityError(f"request state is not a committed success: {key}")
    if state.get("endpoint") != expected_endpoint:
        raise CacheIntegrityError(f"request endpoint differs for {key}")
    args = state.get("args")
    if not isinstance(args, Mapping) or dict(args) != dict(expected_args):
        raise CacheIntegrityError(f"request arguments differ for {key}")
    provider_version = str(state.get("provider_version", "")).strip()
    if not provider_version:
        raise CacheIntegrityError(f"provider version is absent for {key}")
    recomputed = make_request_key(
        expected_endpoint, expected_args, provider_version=provider_version
    )
    if recomputed != key:
        raise CacheIntegrityError(f"canonical request key differs for {key}")
    expected_object = (
        cache_dir / "objects" / expected_endpoint / f"{key}.parquet"
    ).resolve()
    try:
        declared_object = (cache_dir / str(state.get("object", ""))).resolve()
    except Exception as exc:
        raise CacheIntegrityError(f"cache object path is invalid for {key}") from exc
    if declared_object != expected_object or not expected_object.is_file():
        raise CacheIntegrityError(f"cache object path differs or is missing for {key}")
    declared_hash = str(state.get("parquet_sha256", ""))
    if not _REQUEST_KEY.fullmatch(declared_hash):
        raise CacheIntegrityError(f"cache object hash is invalid for {key}")
    if _sha256_file(expected_object) != declared_hash:
        raise CacheIntegrityError(f"cache object hash differs for {key}")
    try:
        frame = pd.read_parquet(expected_object)
    except Exception as exc:
        raise CacheIntegrityError(f"cannot read cache object {key}: {exc}") from exc
    expected_schema = tuple(str(field) for field in expected_fields)
    if tuple(frame.columns.astype(str)) != expected_schema:
        raise CacheIntegrityError(f"cache object schema differs for {key}")
    try:
        registered_rows = int(expected_rows)
        state_rows = int(state.get("rows", -1))
    except (TypeError, ValueError) as exc:
        raise CacheIntegrityError(f"row count is invalid for {key}") from exc
    if len(frame) != registered_rows or len(frame) != state_rows:
        raise CacheIntegrityError(f"row count differs for {key}")
    return frame, provider_version, declared_hash


def _price_panel(frame: pd.DataFrame, code: str, flag: str) -> pd.DataFrame:
    if frame.empty:
        raise CacheIntegrityError(f"price history is empty for {code} flag={flag}")
    if not frame["code"].astype(str).eq(code).all():
        raise CacheIntegrityError(f"price payload contains another code for {code}")
    if not frame["adjustflag"].astype(str).eq(flag).all():
        raise CacheIntegrityError(f"price payload flag differs for {code} flag={flag}")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise CacheIntegrityError(f"price dates are invalid for {code} flag={flag}")
    selected = frame.loc[:, [*COMPARISON_FIELDS, "adjustflag"]].copy()
    for field in COMPARISON_FIELDS:
        selected[field] = pd.to_numeric(selected[field], errors="coerce")
    values = selected.loc[:, list(COMPARISON_FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise CacheIntegrityError(
            f"price payload has nonfinite or nonpositive values for {code} flag={flag}"
        )
    selected.index = pd.MultiIndex.from_arrays(
        [dates, pd.Index([code] * len(selected))], names=["date", "symbol"]
    )
    if not selected.index.is_unique:
        raise CacheIntegrityError(f"price index is duplicate for {code} flag={flag}")
    return selected


def _event_frame(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    if frame.empty:
        raise CacheIntegrityError(
            f"adjustment factor history is empty and cannot establish an anchor for {code}"
        )
    if not frame["code"].astype(str).eq(code).all():
        raise CacheIntegrityError(
            f"adjustment-factor payload contains another code for {code}"
        )
    events = frame.loc[:, list(ADJUST_FACTOR_FIELDS)].copy()
    dates = pd.to_datetime(events["dividOperateDate"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.duplicated().any():
        raise CacheIntegrityError(f"adjustment-factor dates are invalid for {code}")
    for field in ("foreAdjustFactor", "backAdjustFactor", "adjustFactor"):
        events[field] = pd.to_numeric(events[field], errors="coerce")
        values = events[field].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise CacheIntegrityError(
                f"adjustment-factor {field} is invalid for {code}"
            )
    events["dividOperateDate"] = dates
    return events.sort_values("dividOperateDate", kind="stable")


def _empty_report(config: AdjustmentReconstructionAuditConfig) -> dict[str, object]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "baostock_raw_adjust_factor_vs_direct_backward_adjusted",
        "valid": False,
        "decision": "retain_direct_adjusted_history",
        "read_only": True,
        "network_access": False,
        "contract": {
            "required_coverage_fraction": REQUIRED_COVERAGE_FRACTION,
            "comparison_fields": list(COMPARISON_FIELDS),
            "field_alignment": "same date-symbol index; reconstructed and direct columns compared one-for-one",
            "event_asof_rule": "last backAdjustFactor with dividOperateDate <= price date",
            "initial_factor_rule": "factor required on or before every symbol's first raw price date",
            "relative_tolerance": float(config.relative_tolerance),
            "absolute_tolerance": float(config.absolute_tolerance),
            "tolerance_formula": "abs(reconstructed-direct) <= absolute_tolerance + relative_tolerance*abs(direct)",
            "threshold_override_policy": "tighten_only",
        },
        "source": {
            "cache_dir": str(config.cache_dir),
            "normalized_dir": str(config.normalized_dir),
        },
        "coverage": {
            "expected_symbols": 0,
            "source_covered_symbols": 0,
            "within_tolerance_symbols": 0,
            "source_coverage_fraction": 0.0,
            "within_tolerance_fraction": 0.0,
            "aligned_rows": 0,
            "compared_cells": 0,
            "within_tolerance_cells": 0,
        },
        "metrics": {
            "maximum_absolute_error": 0.0,
            "maximum_relative_error": 0.0,
            "maximum_absolute_error_by_field": {
                field: 0.0 for field in COMPARISON_FIELDS
            },
            "maximum_relative_error_by_field": {
                field: 0.0 for field in COMPARISON_FIELDS
            },
            "outside_tolerance_cells_by_field": {
                field: 0 for field in COMPARISON_FIELDS
            },
        },
        "failure_count": 0,
        "failed_symbols": [],
        "failure_samples": [],
    }


def configuration_failure_report(
    *,
    cache_dir: Path | str,
    normalized_dir: Path | str,
    error: Exception,
) -> dict[str, object]:
    """Return JSON-safe output when audit configuration itself is invalid."""

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "baostock_raw_adjust_factor_vs_direct_backward_adjusted",
        "valid": False,
        "decision": "retain_direct_adjusted_history",
        "read_only": True,
        "network_access": False,
        "source": {
            "cache_dir": str(Path(cache_dir).resolve()),
            "normalized_dir": str(Path(normalized_dir).resolve()),
        },
        "failure_count": 1,
        "failed_symbols": [],
        "failure_samples": [
            {
                "scope": "configuration",
                "code": type(error).__name__,
                "detail": str(error),
            }
        ],
    }


def audit_adjustment_reconstruction(
    config: AdjustmentReconstructionAuditConfig,
) -> dict[str, object]:
    """Audit all cached symbols and return a JSON-serializable fail-closed report."""

    report = _empty_report(config)
    failures: list[dict[str, str]] = []
    failed_symbols: list[str] = []
    try:
        (
            symbols,
            price_index,
            adjustment_index,
            price_manifest,
            adjustment_manifest,
            dates,
        ) = _validate_complete_manifests(config.normalized_dir)
    except Exception as exc:
        failures.append(
            {
                "scope": "catalog",
                "code": type(exc).__name__,
                "detail": str(exc),
            }
        )
        report["failure_count"] = 1
        report["failure_samples"] = failures
        return report

    expected_symbols = len(symbols)
    coverage = report["coverage"]
    assert isinstance(coverage, dict)
    coverage["expected_symbols"] = expected_symbols
    source = report["source"]
    assert isinstance(source, dict)
    source.update(
        {
            "membership_manifest_sha256": _sha256_file(
                config.normalized_dir / "membership_manifest.json"
            ),
            "price_manifest_sha256": _sha256_file(
                config.normalized_dir / "price_manifest.json"
            ),
            "adjustment_manifest_sha256": _sha256_file(
                config.normalized_dir / "adjustment_manifest.json"
            ),
            "symbol_set_sha256": _sha256_strings(symbols),
            "price_start": dates["price_start"],
            "price_end": dates["price_end"],
            "adjustment_start": dates["adjustment_start"],
            "adjustment_end": dates["adjustment_end"],
            "price_requests": int(price_manifest["price_requests"]),
            "adjustment_factor_requests": int(
                adjustment_manifest["adjustment_factor_requests"]
            ),
        }
    )

    price_fields = ",".join(EQUITY_DAILY_FIELDS)
    price_lookup = price_index.set_index(["code", "adjustflag"], verify_integrity=True)
    adjustment_lookup = adjustment_index.set_index("code", verify_integrity=True)
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    maximum_absolute = metrics["maximum_absolute_error_by_field"]
    maximum_relative = metrics["maximum_relative_error_by_field"]
    outside_counts = metrics["outside_tolerance_cells_by_field"]
    assert isinstance(maximum_absolute, dict)
    assert isinstance(maximum_relative, dict)
    assert isinstance(outside_counts, dict)

    source_covered = 0
    within_tolerance_symbols = 0
    aligned_rows = 0
    compared_cells = 0
    within_tolerance_cells = 0
    provider_versions: set[str] = set()
    cache_identities: list[str] = []

    for code in symbols:
        try:
            frames: dict[str, pd.DataFrame] = {}
            for flag in ("3", "1"):
                registered = price_lookup.loc[(code, flag)]
                expected_args = {
                    "code": code,
                    "fields": price_fields,
                    "start_date": dates["price_start"],
                    "end_date": dates["price_end"],
                    "frequency": "d",
                    "adjustflag": flag,
                }
                frame, provider, object_hash = _read_committed_object(
                    config.cache_dir,
                    registered["request_key"],
                    expected_endpoint="history_k_data",
                    expected_args=expected_args,
                    expected_fields=EQUITY_DAILY_FIELDS,
                    expected_rows=registered["rows"],
                )
                provider_versions.add(provider)
                cache_identities.append(
                    "|".join(
                        (
                            code,
                            "history_k_data",
                            flag,
                            str(registered["request_key"]),
                            object_hash,
                            str(len(frame)),
                        )
                    )
                )
                frames[flag] = _price_panel(frame, code, flag)

            registered_event = adjustment_lookup.loc[code]
            event_args = {
                "code": code,
                "start_date": dates["adjustment_start"],
                "end_date": dates["adjustment_end"],
            }
            event_raw, provider, object_hash = _read_committed_object(
                config.cache_dir,
                registered_event["request_key"],
                expected_endpoint="adjust_factor",
                expected_args=event_args,
                expected_fields=ADJUST_FACTOR_FIELDS,
                expected_rows=registered_event["rows"],
            )
            provider_versions.add(provider)
            cache_identities.append(
                "|".join(
                    (
                        code,
                        "adjust_factor",
                        "event",
                        str(registered_event["request_key"]),
                        object_hash,
                        str(len(event_raw)),
                    )
                )
            )
            events = _event_frame(event_raw, code)
            raw = frames["3"]
            direct = frames["1"]
            if not raw.index.equals(direct.index):
                raise CacheIntegrityError(
                    f"raw and direct-adjusted indexes differ for {code}"
                )
            price_dates = pd.DatetimeIndex(
                raw.index.get_level_values("date")
            ).normalize()
            if (
                price_dates.min() < pd.Timestamp(dates["price_start"])
                or price_dates.max() > pd.Timestamp(dates["price_end"])
            ):
                raise CacheIntegrityError(
                    f"price payload falls outside the frozen range for {code}"
                )
            event_dates = pd.DatetimeIndex(events["dividOperateDate"])
            if (
                event_dates.min() < pd.Timestamp(dates["adjustment_start"])
                or event_dates.max() > pd.Timestamp(dates["adjustment_end"])
            ):
                raise CacheIntegrityError(
                    f"adjustment-factor payload falls outside the frozen range for {code}"
                )
            first_price_date = pd.Timestamp(
                price_dates.min()
            ).normalize()
            if events["dividOperateDate"].min() > first_price_date:
                raise CacheIntegrityError(
                    f"no factor exists on or before the first raw price for {code}"
                )

            reconstructed = reconstruct_backward_adjusted_view(
                raw, events, require_initial_factor=True
            )
            if not reconstructed.index.equals(direct.index):
                raise CacheIntegrityError(
                    f"reconstructed and direct-adjusted indexes differ for {code}"
                )
            source_covered += 1
            first_outside_detail: str | None = None
            rows = len(direct)
            aligned_rows += rows
            compared_cells += rows * len(COMPARISON_FIELDS)
            for field in COMPARISON_FIELDS:
                rebuilt_values = pd.to_numeric(
                    reconstructed[field], errors="coerce"
                ).to_numpy(dtype=float)
                direct_values = pd.to_numeric(
                    direct[field], errors="coerce"
                ).to_numpy(dtype=float)
                if (
                    not np.isfinite(rebuilt_values).all()
                    or not np.isfinite(direct_values).all()
                ):
                    raise CacheIntegrityError(
                        f"comparison values are nonfinite for {code}:{field}"
                    )
                absolute_error = np.abs(rebuilt_values - direct_values)
                relative_error = absolute_error / np.maximum(
                    np.abs(direct_values), 1e-12
                )
                maximum_absolute[field] = max(
                    float(maximum_absolute[field]), float(absolute_error.max())
                )
                maximum_relative[field] = max(
                    float(maximum_relative[field]), float(relative_error.max())
                )
                close = np.isclose(
                    rebuilt_values,
                    direct_values,
                    rtol=config.relative_tolerance,
                    atol=config.absolute_tolerance,
                    equal_nan=False,
                )
                outside = int((~close).sum())
                outside_counts[field] = int(outside_counts[field]) + outside
                within_tolerance_cells += int(close.sum())
                if outside and first_outside_detail is None:
                    first = int(np.flatnonzero(~close)[0])
                    date = direct.index[first][0]
                    first_outside_detail = (
                        "outside_tolerance:"
                        f"{field}:{pd.Timestamp(date).date()}:"
                        f"absolute_error={absolute_error[first]:.12g}:"
                        f"relative_error={relative_error[first]:.12g}"
                    )
            if first_outside_detail is None:
                within_tolerance_symbols += 1
            else:
                failed_symbols.append(code)
                failures.append(
                    {
                        "scope": "symbol",
                        "symbol": code,
                        "code": "OutsideTolerance",
                        "detail": first_outside_detail,
                    }
                )
        except Exception as exc:
            failed_symbols.append(code)
            failures.append(
                {
                    "scope": "symbol",
                    "symbol": code,
                    "code": type(exc).__name__,
                    "detail": str(exc),
                }
            )

    # Mixing provider versions would invalidate the proposed reconstruction
    # shortcut even if every numeric cell happened to agree.
    if len(provider_versions) != 1:
        failures.append(
            {
                "scope": "catalog",
                "code": "ProviderVersionMismatch",
                "detail": "all raw, adjusted and factor requests must share one provider version",
            }
        )

    source_fraction = (
        source_covered / expected_symbols if expected_symbols else 0.0
    )
    tolerance_fraction = (
        within_tolerance_symbols / expected_symbols if expected_symbols else 0.0
    )
    coverage.update(
        {
            "source_covered_symbols": source_covered,
            "within_tolerance_symbols": within_tolerance_symbols,
            "source_coverage_fraction": source_fraction,
            "within_tolerance_fraction": tolerance_fraction,
            "aligned_rows": aligned_rows,
            "compared_cells": compared_cells,
            "within_tolerance_cells": within_tolerance_cells,
        }
    )
    metrics["maximum_absolute_error"] = max(
        float(value) for value in maximum_absolute.values()
    )
    metrics["maximum_relative_error"] = max(
        float(value) for value in maximum_relative.values()
    )
    report["provider_versions"] = sorted(provider_versions)
    source["audited_cache_objects"] = len(cache_identities)
    source["audited_cache_identity_sha256"] = _sha256_strings(
        sorted(cache_identities)
    )
    report["failure_count"] = len(failures)
    report["failed_symbols"] = sorted(set(failed_symbols))
    report["failure_samples"] = failures[: config.maximum_failure_samples]
    valid = (
        not failures
        and expected_symbols > 0
        and source_fraction == REQUIRED_COVERAGE_FRACTION
        and tolerance_fraction == REQUIRED_COVERAGE_FRACTION
        and within_tolerance_cells == compared_cells
    )
    report["valid"] = valid
    report["decision"] = (
        "eligible_for_preregistered_raw_plus_adjust_factor_v2"
        if valid
        else "retain_direct_adjusted_history"
    )
    return report


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "COMPARISON_FIELDS",
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "REQUIRED_COVERAGE_FRACTION",
    "AdjustmentReconstructionAuditConfig",
    "audit_adjustment_reconstruction",
    "configuration_failure_report",
]
