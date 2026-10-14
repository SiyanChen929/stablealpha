"""High-level, resumable BaoStock download plans for the historical CSI 300 domain."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Callable, Iterable, Iterator, Mapping

import pandas as pd

from .baostock import (
    ADJUST_FACTOR_FIELDS,
    HS300_FIELDS,
    INDUSTRY_FIELDS,
    STOCK_BASIC_FIELDS,
    TRADE_DATE_FIELDS,
    BaoStockConfig,
    BaoStockSession,
    CacheIntegrityError,
    make_request_key,
)
from .hs300 import validate_membership_snapshots


_CODE = re.compile(r"^(sh|sz)\.\d{6}$")
_PRICE_WRITER_LOCK = Path(".locks/price_catalog_writer.lock")


class PriceCatalogAlreadyRunning(RuntimeError):
    """Another process owns the shared BaoStock raw-catalog writer lock."""


@contextmanager
def _exclusive_price_writer_lock(cache_dir: Path) -> Iterator[Path]:
    """Prevent concurrent network publication into one BaoStock raw cache.

    The historical name and path are retained because an already-running
    price downloader holds this exact advisory lock.  Adjustment-factor and
    industry downloads deliberately join the same lock domain, so a direct
    CLI invocation cannot race the supervised serial pipeline.
    """

    lock_path = cache_dir / _PRICE_WRITER_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PriceCatalogAlreadyRunning(
                f"BaoStock raw-catalog writer lock is already held: {lock_path}"
            ) from exc
        acquired = True
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "acquired_at": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        try:
            if acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@dataclass(frozen=True)
class HS300SourceConfig:
    cache_dir: Path | str
    normalized_dir: Path | str
    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    membership_frequency: str = "provider_weekly"
    maximum_membership_snapshot_gap_days: int = 21
    maximum_membership_snapshot_age_days: int = 21
    max_attempts: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "normalized_dir", Path(self.normalized_dir))
        start, end = pd.Timestamp(self.start_date), pd.Timestamp(self.end_date)
        if start > end:
            raise ValueError("start_date is after end_date")
        if self.membership_frequency not in {"provider_weekly", "trading_day"}:
            raise ValueError("membership_frequency must be provider_weekly or trading_day")
        if (
            self.maximum_membership_snapshot_gap_days <= 0
            or self.maximum_membership_snapshot_age_days < 0
        ):
            raise ValueError("membership snapshot gap/age limits are invalid")


def _sha256(path: Path) -> str:
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
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _freeze_parquet_exact(
    frame: pd.DataFrame,
    target: Path,
    *,
    nonsemantic_columns: Iterable[str] = (),
) -> None:
    """Write once, or retain byte-identical lineage when semantics already exist."""

    if not target.is_file():
        _atomic_parquet(frame, target)
        return
    try:
        existing = pd.read_parquet(target)
    except Exception as exc:
        raise CacheIntegrityError(f"cannot verify existing normalized {target}: {exc}") from exc
    ignored = set(nonsemantic_columns)
    if tuple(existing.columns.astype(str)) != tuple(frame.columns.astype(str)):
        raise CacheIntegrityError(
            f"existing normalized schema differs from raw reconstruction: {target.name}"
        )
    semantic_columns = [name for name in frame.columns if name not in ignored]
    if not existing.loc[:, semantic_columns].equals(frame.loc[:, semantic_columns]):
        raise CacheIntegrityError(
            f"existing normalized values differ from raw reconstruction: {target.name}"
        )


def _atomic_json(payload: dict[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _publish_membership_manifest(
    summary: dict[str, object],
    manifest_path: Path,
) -> dict[str, object]:
    """Retain byte identity when a cache rebuild has identical semantics."""

    if manifest_path.exists():
        try:
            existing = json.loads(
                _read_regular_file_bytes(
                    manifest_path,
                    maximum_bytes=4 * 1024 * 1024,
                ).decode("utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(
                f"cannot verify existing membership manifest: {exc}"
            ) from exc
        if isinstance(existing, dict):
            existing_semantics = {
                key: value for key, value in existing.items() if key != "created_at"
            }
            normalized_summary = json.loads(
                json.dumps(summary, ensure_ascii=False, sort_keys=True)
            )
            new_semantics = {
                key: value
                for key, value in normalized_summary.items()
                if key != "created_at"
            }
            if (
                existing_semantics == new_semantics
                and isinstance(existing.get("created_at"), str)
                and existing["created_at"]
            ):
                return existing
    _atomic_json(summary, manifest_path)
    return summary


def _membership_request_dates(
    trading_dates: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
    frequency: str,
) -> tuple[pd.Timestamp, ...]:
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if frequency == "trading_day":
        values = list(trading_dates)
    else:
        # BaoStock documents this dataset as a Monday-updated weekly snapshot.
        # Querying calendar Mondays preserves every native version, including a
        # Monday holiday whose snapshot becomes usable on the next session.
        values = list(pd.date_range(start.normalize(), end.normalize(), freq="W-MON"))
    values.extend([start.normalize(), end.normalize()])
    return tuple(sorted({pd.Timestamp(value).normalize() for value in values}))


def _validate_membership_snapshot(
    frame: pd.DataFrame,
    requested: pd.Timestamp,
    *,
    expected_members: int = 300,
) -> pd.DataFrame:
    if len(frame) != expected_members:
        raise ValueError(
            f"HS300 snapshot {requested.date()} has {len(frame)} rows, "
            f"expected {expected_members}"
        )
    if frame["code"].duplicated().any():
        raise ValueError(f"HS300 snapshot {requested.date()} contains duplicate symbols")
    if not frame["code"].astype(str).map(lambda value: bool(_CODE.fullmatch(value))).all():
        raise ValueError(f"HS300 snapshot {requested.date()} contains malformed symbols")
    update = pd.to_datetime(frame["updateDate"], errors="coerce")
    if update.isna().any() or update.nunique() != 1:
        raise ValueError(f"HS300 snapshot {requested.date()} has invalid updateDate")
    if update.iloc[0] > requested:
        raise ValueError(f"HS300 snapshot {requested.date()} leaks future updateDate")
    output = frame.copy()
    output.insert(0, "requested_date", requested)
    output["vendor_update_date"] = update
    return output[["requested_date", "vendor_update_date", "code", "code_name"]]


def _deduplicate_vendor_snapshots(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    by_update: dict[pd.Timestamp, pd.DataFrame] = {}
    for frame in frames:
        update = pd.Timestamp(frame["vendor_update_date"].iloc[0])
        existing = by_update.get(update)
        if existing is not None:
            left = existing.sort_values("code")[["code", "code_name"]].reset_index(drop=True)
            right = frame.sort_values("code")[["code", "code_name"]].reset_index(drop=True)
            if not left.equals(right):
                raise ValueError(f"BaoStock returned two different sets for updateDate {update.date()}")
            continue
        by_update[update] = frame
    if not by_update:
        raise ValueError("no membership snapshots were downloaded")
    return pd.concat([by_update[key] for key in sorted(by_update)], ignore_index=True)


def expand_membership_daily(
    snapshots: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    *,
    observation_snapshots: pd.DataFrame | None = None,
    expected_members: int = 300,
    maximum_snapshot_gap_days: int = 21,
    maximum_snapshot_age_days: int = 21,
) -> pd.DataFrame:
    """As-of expand native snapshots; gaps or incomplete days fail closed."""

    freshness = snapshots if observation_snapshots is None else observation_snapshots
    validate_membership_snapshots(
        freshness,
        expected_members=expected_members,
        trading_dates=trading_dates,
        max_snapshot_gap_days=maximum_snapshot_gap_days,
        max_snapshot_age_days=maximum_snapshot_age_days,
    ).require_valid()

    updates = pd.DatetimeIndex(snapshots["vendor_update_date"].drop_duplicates()).sort_values()
    records: list[pd.DataFrame] = []
    for day in trading_dates:
        position = updates.searchsorted(day, side="right") - 1
        if position < 0:
            raise ValueError(f"no historical membership known on {day.date()}")
        update = updates[position]
        current = snapshots.loc[snapshots["vendor_update_date"] == update, ["code", "code_name"]]
        if len(current) != expected_members or current["code"].duplicated().any():
            raise ValueError(f"invalid membership expansion on {day.date()}")
        block = current.copy()
        block.insert(0, "date", day)
        block["vendor_update_date"] = update
        records.append(block)
    output = pd.concat(records, ignore_index=True)
    counts = output.groupby("date")["code"].nunique()
    if not (counts == expected_members).all():
        raise ValueError(
            "daily HS300 membership does not contain exactly the expected unique symbols"
        )
    return output


def _membership_observation_snapshots(
    frames: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """Represent successful request observations for cadence/freshness audit.

    An unchanged index composition can legitimately retain the same vendor
    ``updateDate`` for months.  Freshness is instead proved by each successful
    scheduled query.  Economic as-of expansion continues to use the vendor
    update date, so a future constituent is never backfilled.
    """

    observations = pd.concat(list(frames), ignore_index=True)
    required = {"requested_date", "vendor_update_date", "code", "code_name"}
    if not required.issubset(observations.columns):
        raise ValueError("membership observations lack request/vendor dates")
    requested = pd.to_datetime(
        observations["requested_date"], errors="coerce"
    ).dt.normalize()
    vendor = pd.to_datetime(
        observations["vendor_update_date"], errors="coerce"
    ).dt.normalize()
    if requested.isna().any() or vendor.isna().any() or (vendor > requested).any():
        raise ValueError("membership observations contain future or invalid dates")
    return pd.DataFrame(
        {
            "updateDate": requested,
            "code": observations["code"].astype(str),
            "code_name": observations["code_name"].astype(str),
        }
    )


_MEMBERSHIP_REQUEST_INDEX_COLUMNS = (
    "endpoint",
    "requested_date",
    "start_date",
    "end_date",
    "vendor_update_date",
    "rows",
    "request_key",
    "cache_hit",
)

_SECURITY_REQUEST_INDEX_COLUMNS = (
    "endpoint",
    "code",
    "code_name",
    "rows",
    "request_key",
    "cache_hit",
)

_INDUSTRY_REQUEST_INDEX_COLUMNS = (
    "endpoint",
    "requested_date",
    "vendor_update_date",
    "rows",
    "request_key",
    "cache_hit",
)


def _membership_cache_metadata(
    frame: pd.DataFrame,
    *,
    endpoint: str,
    requested_date: str = "",
    start_date: str = "",
    end_date: str = "",
    vendor_update_date: str = "",
) -> dict[str, object]:
    cache = dict(frame.attrs.get("baostock_cache", {}))
    request_key = str(cache.get("request_key", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", request_key):
        raise ValueError(f"{endpoint} response lacks a committed request key")
    return {
        "endpoint": endpoint,
        "requested_date": requested_date,
        "start_date": start_date,
        "end_date": end_date,
        "vendor_update_date": vendor_update_date,
        "rows": len(frame),
        "request_key": request_key,
        "cache_hit": bool(cache.get("cache_hit", False)),
    }


def _freeze_membership_outputs(
    config: HS300SourceConfig,
    calendar_raw: pd.DataFrame,
    validated_frames: Iterable[pd.DataFrame],
    request_rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Deterministically materialize every normalized membership artifact."""

    calendar = calendar_raw.copy()
    if tuple(calendar.columns.astype(str)) != TRADE_DATE_FIELDS:
        raise ValueError("trade calendar schema drift during membership freeze")
    calendar["date"] = pd.to_datetime(calendar["calendar_date"], errors="coerce")
    if calendar["date"].isna().any() or calendar["date"].duplicated().any():
        raise ValueError("trade calendar contains invalid or duplicate dates")
    calendar = calendar.sort_values("date").reset_index(drop=True)
    expected_calendar = pd.date_range(
        pd.Timestamp(config.start_date).normalize(),
        pd.Timestamp(config.end_date).normalize(),
        freq="D",
    )
    if not pd.DatetimeIndex(calendar["date"]).equals(expected_calendar):
        raise ValueError("trade calendar does not exactly cover frozen calendar days")
    trading_dates = pd.DatetimeIndex(
        calendar.loc[calendar["is_trading_day"].astype(str) == "1", "date"]
    ).sort_values()
    if trading_dates.empty:
        raise ValueError("trade calendar contains no trading sessions")

    frames = list(validated_frames)
    snapshots = _deduplicate_vendor_snapshots(frames)
    observation_snapshots = _membership_observation_snapshots(frames)
    snapshot_audit = validate_membership_snapshots(
        observation_snapshots,
        expected_members=300,
        trading_dates=trading_dates,
        max_snapshot_gap_days=config.maximum_membership_snapshot_gap_days,
        max_snapshot_age_days=config.maximum_membership_snapshot_age_days,
    ).require_valid()
    daily = expand_membership_daily(
        snapshots,
        trading_dates,
        observation_snapshots=observation_snapshots,
        expected_members=300,
        maximum_snapshot_gap_days=config.maximum_membership_snapshot_gap_days,
        maximum_snapshot_age_days=config.maximum_membership_snapshot_age_days,
    )
    union = (
        daily.groupby("code", as_index=False)
        .agg(
            code_name=("code_name", "last"),
            first_member_date=("date", "min"),
            last_member_date=("date", "max"),
            member_sessions=("date", "size"),
        )
        .sort_values("code")
        .reset_index(drop=True)
    )
    request_index = pd.DataFrame.from_records(
        list(request_rows), columns=_MEMBERSHIP_REQUEST_INDEX_COLUMNS
    )
    if len(request_index) != len(frames) + 1:
        raise ValueError("membership request index must bind calendar plus every query")
    if request_index["request_key"].astype(str).duplicated().any():
        raise ValueError("membership request index contains duplicate request keys")
    expected_requested = _membership_request_dates(
        trading_dates,
        config.start_date,
        config.end_date,
        config.membership_frequency,
    )
    membership_rows = request_index.loc[
        request_index["endpoint"].astype(str).eq("hs300_stocks")
    ]
    actual_requested = pd.DatetimeIndex(
        pd.to_datetime(membership_rows["requested_date"], errors="coerce")
    ).sort_values()
    if not actual_requested.equals(pd.DatetimeIndex(expected_requested)):
        raise ValueError("membership request index differs from the frozen query plan")
    request_index = request_index.sort_values(
        ["endpoint", "requested_date"], kind="stable"
    ).reset_index(drop=True)

    normalized = config.normalized_dir
    paths = {
        "trade_calendar": normalized / "trade_calendar.parquet",
        "membership_snapshots": normalized / "membership_snapshots.parquet",
        "membership_daily": normalized / "membership_daily.parquet",
        "security_union": normalized / "security_union.parquet",
        "membership_request_index": normalized / "membership_request_index.parquet",
    }
    _freeze_parquet_exact(calendar, paths["trade_calendar"])
    _freeze_parquet_exact(snapshots, paths["membership_snapshots"])
    _freeze_parquet_exact(daily, paths["membership_daily"])
    _freeze_parquet_exact(union, paths["security_union"])
    _freeze_parquet_exact(
        request_index,
        paths["membership_request_index"],
        nonsemantic_columns=("cache_hit",),
    )
    summary: dict[str, object] = {
        "phase": "membership_catalog",
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "cache_dir": str(config.cache_dir),
            "normalized_dir": str(config.normalized_dir),
        },
        "source_contract": {
            "expected_members": 300,
            "membership_frequency": config.membership_frequency,
            "maximum_snapshot_gap_days": (
                config.maximum_membership_snapshot_gap_days
            ),
            "maximum_snapshot_age_days": (
                config.maximum_membership_snapshot_age_days
            ),
            "deterministic_daily_expansion": True,
            "freshness_clock": "successful_requested_observation_date",
            "membership_effective_clock": "vendor_update_date",
            "calendar_endpoint": "trade_dates",
            "snapshot_endpoint": "hs300_stocks",
        },
        "snapshot_audit": asdict(snapshot_audit),
        "trading_sessions": len(trading_dates),
        "native_snapshots": snapshots["vendor_update_date"].nunique(),
        "requested_snapshots": len(frames),
        "raw_cache_requests": len(request_index),
        "historical_union_symbols": len(union),
        "daily_rows": len(daily),
        "files": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    return _publish_membership_manifest(
        summary,
        normalized / "membership_manifest.json",
    )


def download_membership_catalog(
    config: HS300SourceConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Run membership acquisition as the catalog's sole active writer."""

    with _exclusive_price_writer_lock(Path(config.cache_dir)):
        return _download_membership_catalog_locked(config, progress=progress)


def _download_membership_catalog_locked(
    config: HS300SourceConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Download calendar and all native membership versions, then freeze tables."""

    bs_config = BaoStockConfig(config.cache_dir, max_attempts=config.max_attempts)
    with BaoStockSession(bs_config) as session:
        calendar = session.query_trade_dates(config.start_date, config.end_date)
        request_rows: list[dict[str, object]] = [
            _membership_cache_metadata(
                calendar,
                endpoint="trade_dates",
                start_date=config.start_date,
                end_date=config.end_date,
            )
        ]
        calendar_dates = pd.to_datetime(
            calendar["calendar_date"], errors="coerce"
        )
        if calendar_dates.isna().any():
            raise ValueError("trade calendar contains invalid dates")
        trading_dates = pd.DatetimeIndex(
            calendar_dates.loc[calendar["is_trading_day"].astype(str) == "1"]
        ).sort_values()
        if trading_dates.empty:
            raise ValueError("trade calendar contains no trading sessions")
        requested = _membership_request_dates(
            trading_dates,
            config.start_date,
            config.end_date,
            config.membership_frequency,
        )
        frames: list[pd.DataFrame] = []
        total = len(requested)
        for number, day in enumerate(requested, 1):
            frame = session.query_hs300(str(day.date()))
            validated = _validate_membership_snapshot(frame, day)
            frames.append(validated)
            request_rows.append(
                _membership_cache_metadata(
                    frame,
                    endpoint="hs300_stocks",
                    requested_date=str(day.date()),
                    vendor_update_date=str(
                        pd.Timestamp(validated["vendor_update_date"].iloc[0]).date()
                    ),
                )
            )
            if progress is not None:
                progress(number, total, str(day.date()))
    return _freeze_membership_outputs(config, calendar, frames, request_rows)


def _read_exact_cached_request(
    cache_dir: Path,
    *,
    request_key: str,
    endpoint: str,
    args: Mapping[str, object],
    expected_fields: tuple[str, ...],
) -> pd.DataFrame:
    """Read the exact versioned cache key selected by a frozen request index."""

    request_key = str(request_key)
    if not re.fullmatch(r"[0-9a-f]{64}", request_key):
        raise CacheIntegrityError("selected cache request key is malformed")
    state_path = cache_dir / "state" / f"{request_key}.json"
    try:
        state_bytes = _read_regular_file_bytes(state_path, maximum_bytes=1024 * 1024)
        state = json.loads(state_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError(f"cannot read selected cache state {state_path}: {exc}") from exc
    if not isinstance(state, dict):
        raise CacheIntegrityError("selected cache state is not an object")
    expected_state_keys = {
        "state_schema_version",
        "request_key",
        "provider",
        "provider_version",
        "endpoint",
        "args",
        "attempt_count",
        "status",
        "started_at",
        "completed_at",
        "rows",
        "fields",
        "object",
        "parquet_sha256",
    }
    if (
        set(state) != expected_state_keys
        or type(state.get("state_schema_version")) is not int
        or state.get("state_schema_version") != 1
        or type(state.get("attempt_count")) is not int
        or state.get("attempt_count", 0) < 1
        or not isinstance(state.get("started_at"), str)
        or not state.get("started_at")
        or not isinstance(state.get("completed_at"), str)
        or not state.get("completed_at")
        or type(state.get("rows")) is not int
        or state.get("rows", -1) < 0
        or not isinstance(state.get("parquet_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(state.get("parquet_sha256"))) is None
    ):
        raise CacheIntegrityError("selected cache success state schema is invalid")
    if state.get("request_key") != request_key:
        raise CacheIntegrityError("cache state filename/request key mismatch")
    if (
        state.get("status") != "success"
        or state.get("provider") != "baostock"
        or state.get("endpoint") != endpoint
        or state.get("args") != dict(args)
    ):
        raise CacheIntegrityError("selected cache state differs from frozen request")
    provider_version = str(state.get("provider_version", "")).strip()
    if (
        not provider_version
        or make_request_key(endpoint, args, provider_version=provider_version)
        != request_key
    ):
        raise CacheIntegrityError("cache state request key is not canonical")
    if tuple(state.get("fields", ())) != tuple(expected_fields):
        raise CacheIntegrityError("cache state field contract mismatch")
    relative = Path(str(state.get("object", "")))
    expected_relative = Path("objects") / endpoint / f"{request_key}.parquet"
    if relative != expected_relative or relative.is_absolute():
        raise CacheIntegrityError("cache state has unsafe object path")
    object_path = cache_dir / relative
    try:
        object_bytes = _read_regular_file_bytes(object_path)
    except OSError as exc:
        raise CacheIntegrityError(f"cannot securely read cache object: {exc}") from exc
    if hashlib.sha256(object_bytes).hexdigest() != str(state["parquet_sha256"]):
        raise CacheIntegrityError("cache object hash mismatch")
    try:
        frame = pd.read_parquet(io.BytesIO(object_bytes))
    except Exception as exc:
        raise CacheIntegrityError(f"cannot decode selected cache object: {exc}") from exc
    if tuple(frame.columns.astype(str)) != tuple(expected_fields):
        raise CacheIntegrityError("cache object schema mismatch")
    if len(frame) != int(state.get("rows", -1)):
        raise CacheIntegrityError("cache object row count mismatch")
    frame.attrs["baostock_cache"] = {
        "request_key": request_key,
        "provider_version": provider_version,
        "cache_hit": True,
    }
    return frame


def _read_regular_file_bytes(
    path: Path,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    """Read stable bytes from one regular final path without following a link."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("O_NOFOLLOW is required for cache replay")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError(f"not a regular file: {path}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise OSError(f"file exceeds bounded replay size: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(current):
            raise OSError(f"file changed during replay: {path}")
        return raw
    finally:
        os.close(descriptor)


def _load_frozen_request_index(
    path: Path,
    *,
    expected_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Load one pre-existing selection index without inferring cache versions."""

    if not path.is_file() or path.is_symlink():
        raise CacheIntegrityError(f"frozen request index is missing or unsafe: {path}")
    try:
        index = pd.read_parquet(path)
    except Exception as exc:
        raise CacheIntegrityError(f"cannot read frozen request index {path}: {exc}") from exc
    if tuple(index.columns.astype(str)) != expected_columns:
        raise CacheIntegrityError(f"frozen request index schema mismatch: {path}")
    keys = index["request_key"].astype(str)
    if (
        index.empty
        or keys.duplicated().any()
        or not keys.map(lambda value: re.fullmatch(r"[0-9a-f]{64}", value) is not None).all()
    ):
        raise CacheIntegrityError(f"frozen request index keys are invalid: {path}")
    return index


def _index_text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _strict_index_row_count(row: Mapping[str, object], frame: pd.DataFrame) -> None:
    declared = row.get("rows")
    if isinstance(declared, bool) or not pd.api.types.is_integer(declared):
        raise CacheIntegrityError("frozen request index row count is not an integer")
    if int(declared) != len(frame):
        raise CacheIntegrityError("frozen request index/cache row count mismatch")


def _select_one_index_row(
    index: pd.DataFrame,
    mask: pd.Series,
    *,
    label: str,
) -> dict[str, object]:
    rows = index.loc[mask]
    if len(rows) != 1:
        raise CacheIntegrityError(
            f"frozen request index must select exactly one {label}; found {len(rows)}"
        )
    return rows.iloc[0].to_dict()


def rebuild_membership_catalog_from_cache(
    config: HS300SourceConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Rebuild membership lineage as the sole normalized catalog writer."""

    with _exclusive_price_writer_lock(Path(config.cache_dir)):
        return _rebuild_membership_catalog_from_cache_locked(
            config,
            progress=progress,
        )


def _rebuild_membership_catalog_from_cache_locked(
    config: HS300SourceConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Re-freeze membership lineage entirely from already committed raw cache."""

    source_index = _load_frozen_request_index(
        config.normalized_dir / "membership_request_index.parquet",
        expected_columns=_MEMBERSHIP_REQUEST_INDEX_COLUMNS,
    )
    endpoints = source_index["endpoint"].astype(str)
    if not endpoints.isin({"trade_dates", "hs300_stocks"}).all():
        raise CacheIntegrityError("membership request index has an unknown endpoint")
    calendar_args = {
        "start_date": config.start_date,
        "end_date": config.end_date,
    }
    calendar_row = _select_one_index_row(
        source_index,
        endpoints.eq("trade_dates")
        & source_index["start_date"].map(_index_text).eq(config.start_date)
        & source_index["end_date"].map(_index_text).eq(config.end_date)
        & source_index["requested_date"].map(_index_text).eq("")
        & source_index["vendor_update_date"].map(_index_text).eq(""),
        label="trade calendar request",
    )
    calendar = _read_exact_cached_request(
        config.cache_dir,
        request_key=str(calendar_row["request_key"]),
        endpoint="trade_dates",
        args=calendar_args,
        expected_fields=TRADE_DATE_FIELDS,
    )
    _strict_index_row_count(calendar_row, calendar)
    request_rows: list[dict[str, object]] = [
        _membership_cache_metadata(
            calendar,
            endpoint="trade_dates",
            start_date=config.start_date,
            end_date=config.end_date,
        )
    ]
    calendar_dates = pd.to_datetime(calendar["calendar_date"], errors="coerce")
    if calendar_dates.isna().any():
        raise ValueError("cached trade calendar contains invalid dates")
    trading_dates = pd.DatetimeIndex(
        calendar_dates.loc[calendar["is_trading_day"].astype(str).eq("1")]
    ).sort_values()
    requested = _membership_request_dates(
        trading_dates,
        config.start_date,
        config.end_date,
        config.membership_frequency,
    )
    requested_text = tuple(str(day.date()) for day in requested)
    membership_rows = source_index.loc[endpoints.eq("hs300_stocks")].copy()
    actual_requested = tuple(
        sorted(membership_rows["requested_date"].map(_index_text).tolist())
    )
    if (
        len(source_index) != len(requested_text) + 1
        or actual_requested != tuple(sorted(requested_text))
        or not membership_rows["start_date"].map(_index_text).eq("").all()
        or not membership_rows["end_date"].map(_index_text).eq("").all()
    ):
        raise CacheIntegrityError(
            "membership request index differs from the frozen query plan"
        )
    frames: list[pd.DataFrame] = []
    for number, day in enumerate(requested, 1):
        requested_date = str(day.date())
        selected = _select_one_index_row(
            membership_rows,
            membership_rows["requested_date"].map(_index_text).eq(requested_date),
            label=f"membership request {requested_date}",
        )
        raw = _read_exact_cached_request(
            config.cache_dir,
            request_key=str(selected["request_key"]),
            endpoint="hs300_stocks",
            args={"date": requested_date},
            expected_fields=HS300_FIELDS,
        )
        _strict_index_row_count(selected, raw)
        validated = _validate_membership_snapshot(raw, day)
        if _index_text(selected["vendor_update_date"]) != str(
            pd.Timestamp(validated["vendor_update_date"].iloc[0]).date()
        ):
            raise CacheIntegrityError(
                f"membership request index vendor date mismatch for {requested_date}"
            )
        frames.append(validated)
        request_rows.append(
            _membership_cache_metadata(
                raw,
                endpoint="hs300_stocks",
                requested_date=str(day.date()),
                vendor_update_date=str(
                    pd.Timestamp(validated["vendor_update_date"].iloc[0]).date()
                ),
            )
        )
        if progress is not None:
            progress(number, len(requested), str(day.date()))
    return _freeze_membership_outputs(config, calendar, frames, request_rows)


def _catalog_cache_metadata(
    frame: pd.DataFrame,
    *,
    endpoint: str,
    **values: object,
) -> dict[str, object]:
    cache = dict(frame.attrs.get("baostock_cache", {}))
    request_key = str(cache.get("request_key", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", request_key):
        raise ValueError(f"{endpoint} response lacks a committed request key")
    return {
        "endpoint": endpoint,
        **values,
        "rows": len(frame),
        "request_key": request_key,
        "cache_hit": bool(cache.get("cache_hit", False)),
    }


def _materialize_security_master(
    union_codes: Iterable[str],
    requests: Iterable[tuple[str, str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the union security master from one global request plus exact gaps."""

    codes = tuple(sorted({str(code).strip() for code in union_codes}))
    if not codes or not all(_CODE.fullmatch(code) for code in codes):
        raise ValueError("security union contains no valid BaoStock equity codes")
    supplied = list(requests)
    global_rows = [item for item in supplied if item[0] == "" and item[1] == ""]
    if len(global_rows) != 1:
        raise ValueError("security lineage requires exactly one global stock_basic request")
    global_frame = global_rows[0][2]
    if tuple(global_frame.columns.astype(str)) != STOCK_BASIC_FIELDS:
        raise ValueError("global stock_basic schema drift")
    if global_frame["code"].astype(str).duplicated().any():
        raise ValueError("global stock_basic response contains duplicate codes")
    selected = global_frame.loc[
        global_frame["code"].astype(str).isin(codes)
    ].copy()
    missing = sorted(set(codes).difference(selected["code"].astype(str)))
    supplements = [item for item in supplied if not (item[0] == "" and item[1] == "")]
    if sorted(code for code, code_name, _ in supplements if code_name == "") != missing:
        raise ValueError(
            "stock_basic supplement requests do not exactly equal global-response gaps"
        )
    if any(code_name != "" for _, code_name, _ in supplements):
        raise ValueError("stock_basic lineage forbids code-name supplement queries")
    supplemental_frames: list[pd.DataFrame] = []
    for code, _code_name, frame in sorted(supplements, key=lambda item: item[0]):
        if tuple(frame.columns.astype(str)) != STOCK_BASIC_FIELDS:
            raise ValueError(f"stock_basic supplement schema drift for {code}")
        if len(frame) != 1 or not frame["code"].astype(str).eq(code).all():
            raise ValueError(f"stock_basic supplement is not an exact row for {code}")
        supplemental_frames.append(frame.copy())
    master = pd.concat([selected, *supplemental_frames], ignore_index=True)
    master = master.loc[:, list(STOCK_BASIC_FIELDS)].sort_values("code").reset_index(drop=True)
    if master["code"].astype(str).duplicated().any() or set(
        master["code"].astype(str)
    ) != set(codes):
        raise ValueError("security master does not exactly cover the historical union")
    ipo = pd.to_datetime(master["ipoDate"], errors="coerce")
    raw_out = master["outDate"].astype("string").str.strip().replace(
        {"": pd.NA, "0": pd.NA}
    )
    out = pd.to_datetime(raw_out, errors="coerce")
    if ipo.isna().any() or (raw_out.notna() & out.isna()).any():
        raise ValueError("security master contains invalid IPO/out dates")
    if not master["status"].astype(str).str.strip().isin(["0", "1"]).all():
        raise ValueError("security master contains an unknown status")
    request_index = pd.DataFrame.from_records(
        [
            _catalog_cache_metadata(
                frame,
                endpoint="stock_basic",
                code=code,
                code_name=code_name,
            )
            for code, code_name, frame in supplied
        ],
        columns=_SECURITY_REQUEST_INDEX_COLUMNS,
    ).sort_values(["code", "code_name"], kind="stable").reset_index(drop=True)
    if request_index["request_key"].astype(str).duplicated().any():
        raise ValueError("security request index contains duplicate request keys")
    return master, request_index


def _freeze_security_outputs(
    config: HS300SourceConfig,
    union_codes: Iterable[str],
    requests: Iterable[tuple[str, str, pd.DataFrame]],
) -> dict[str, object]:
    master, request_index = _materialize_security_master(union_codes, requests)
    paths = {
        "security_master": config.normalized_dir / "security_master.parquet",
        "security_master_request_index": (
            config.normalized_dir / "security_master_request_index.parquet"
        ),
    }
    _freeze_parquet_exact(master, paths["security_master"])
    _freeze_parquet_exact(
        request_index,
        paths["security_master_request_index"],
        nonsemantic_columns=("cache_hit",),
    )
    summary: dict[str, object] = {
        "phase": "security_master_catalog",
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "historical_union_symbols": len(master),
        "raw_cache_requests": len(request_index),
        "source_contract": {
            "schema_version": "1.0",
            "endpoint": "stock_basic",
            "global_query_args": {"code": "", "code_name": ""},
            "supplement_policy": "exact_union_gaps_by_code",
            "deterministic_union_filter": True,
        },
        "files": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    _atomic_json(summary, config.normalized_dir / "security_master_manifest.json")
    return summary


def rebuild_security_master_from_cache(
    config: HS300SourceConfig,
) -> dict[str, object]:
    """Rebuild security lineage as the sole normalized catalog writer."""

    with _exclusive_price_writer_lock(Path(config.cache_dir)):
        return _rebuild_security_master_from_cache_locked(config)


def _rebuild_security_master_from_cache_locked(
    config: HS300SourceConfig,
) -> dict[str, object]:
    """Re-freeze security-master lineage without contacting BaoStock."""

    union_path = config.normalized_dir / "security_union.parquet"
    if not union_path.is_file():
        raise FileNotFoundError("run membership-lineage before security-lineage")
    union_codes = sorted(pd.read_parquet(union_path)["code"].astype(str).unique())
    source_index = _load_frozen_request_index(
        config.normalized_dir / "security_master_request_index.parquet",
        expected_columns=_SECURITY_REQUEST_INDEX_COLUMNS,
    )
    if not source_index["endpoint"].astype(str).eq("stock_basic").all():
        raise CacheIntegrityError("security request index has an unknown endpoint")
    global_row = _select_one_index_row(
        source_index,
        source_index["code"].map(_index_text).eq("")
        & source_index["code_name"].map(_index_text).eq(""),
        label="global stock_basic request",
    )
    global_frame = _read_exact_cached_request(
        config.cache_dir,
        request_key=str(global_row["request_key"]),
        endpoint="stock_basic",
        args={"code": "", "code_name": ""},
        expected_fields=STOCK_BASIC_FIELDS,
    )
    _strict_index_row_count(global_row, global_frame)
    covered = set(
        global_frame.loc[
            global_frame["code"].astype(str).isin(union_codes), "code"
        ].astype(str)
    )
    requests: list[tuple[str, str, pd.DataFrame]] = [("", "", global_frame)]
    missing = sorted(set(union_codes).difference(covered))
    indexed_supplements = sorted(
        source_index.loc[
            ~(
                source_index["code"].map(_index_text).eq("")
                & source_index["code_name"].map(_index_text).eq("")
            ),
            "code",
        ].map(_index_text)
    )
    if (
        len(source_index) != len(missing) + 1
        or indexed_supplements != missing
        or not source_index.loc[
            source_index["code"].map(_index_text).ne(""), "code_name"
        ].map(_index_text).eq("").all()
    ):
        raise CacheIntegrityError(
            "security request index differs from the exact union-gap plan"
        )
    for code in missing:
        selected = _select_one_index_row(
            source_index,
            source_index["code"].map(_index_text).eq(code)
            & source_index["code_name"].map(_index_text).eq(""),
            label=f"stock_basic supplement {code}",
        )
        frame = _read_exact_cached_request(
            config.cache_dir,
            request_key=str(selected["request_key"]),
            endpoint="stock_basic",
            args={"code": code, "code_name": ""},
            expected_fields=STOCK_BASIC_FIELDS,
        )
        _strict_index_row_count(selected, frame)
        requests.append(
            (
                code,
                "",
                frame,
            )
        )
    return _freeze_security_outputs(config, union_codes, requests)


def download_price_catalog(
    config: HS300SourceConfig,
    *,
    limit_symbols: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Run the resumable price plan as the cache's sole active writer."""

    with _exclusive_price_writer_lock(Path(config.cache_dir)):
        return _download_price_catalog_locked(
            config,
            limit_symbols=limit_symbols,
            progress=progress,
        )


def _validate_downloaded_price_pair(
    code: str,
    raw: pd.DataFrame,
    requested_adjusted: pd.DataFrame,
) -> None:
    """Reject a provider response truncated in only one price view."""

    indexes: dict[str, pd.DatetimeIndex] = {}
    for label, frame in (
        ("raw", raw),
        ("requested_adjusted", requested_adjusted),
    ):
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any() or dates.duplicated().any():
            raise CacheIntegrityError(
                f"{code} {label} price response has invalid or duplicate dates"
            )
        indexes[label] = pd.DatetimeIndex(dates).sort_values()
    if not indexes["raw"].equals(indexes["requested_adjusted"]):
        raise CacheIntegrityError(
            f"{code} raw and requested adjusted price responses have different dates"
        )


def _download_price_catalog_locked(
    config: HS300SourceConfig,
    *,
    limit_symbols: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Download raw and backward-adjusted histories for every historical member."""

    union_path = config.normalized_dir / "security_union.parquet"
    if not union_path.is_file():
        raise FileNotFoundError("run membership download before downloading security histories")
    union = pd.read_parquet(union_path).sort_values("code")
    symbols = union["code"].astype(str).tolist()
    if limit_symbols is not None:
        if limit_symbols < 1:
            raise ValueError("limit_symbols must be positive")
        symbols = symbols[:limit_symbols]

    rows: list[dict[str, object]] = []
    bs_config = BaoStockConfig(config.cache_dir, max_attempts=config.max_attempts)
    with BaoStockSession(bs_config) as session:
        index = session.query_index(config.start_date, config.end_date)
        index_meta = dict(index.attrs.get("baostock_cache", {}))
        all_basic = session.query_basic(code="", code_name="")
        basic_requests: list[tuple[str, str, pd.DataFrame]] = [
            ("", "", all_basic)
        ]
        basic = all_basic.loc[all_basic["code"].astype(str).isin(symbols)].copy()
        missing_basic = sorted(set(symbols).difference(basic["code"].astype(str)))
        if missing_basic:
            supplements = []
            for code in missing_basic:
                supplement = session.query_basic(code=code, allow_empty=False)
                supplements.append(supplement)
                basic_requests.append((code, "", supplement))
            basic = pd.concat([basic, *supplements], ignore_index=True)
        if basic["code"].duplicated().any() or set(basic["code"].astype(str)) != set(symbols):
            raise ValueError("security master does not exactly cover the requested historical union")

        total = len(symbols)
        for number, code in enumerate(symbols, 1):
            pair: dict[str, pd.DataFrame] = {}
            for flag, basis in (("3", "raw_unadjusted"), ("1", "backward_adjusted")):
                frame = session.query_history(
                    code,
                    config.start_date,
                    config.end_date,
                    adjustflag=flag,
                    allow_empty=False,
                )
                pair[flag] = frame
                cache = dict(frame.attrs.get("baostock_cache", {}))
                rows.append(
                    {
                        "code": code,
                        "adjustflag": flag,
                        "basis": basis,
                        "rows": len(frame),
                        "first_date": str(frame["date"].min()),
                        "last_date": str(frame["date"].max()),
                        "request_key": cache.get("request_key", ""),
                        "cache_hit": bool(cache.get("cache_hit", False)),
                    }
                )
            _validate_downloaded_price_pair(code, pair["3"], pair["1"])
            if progress is not None:
                progress(number, total, code)

    security_summary = _freeze_security_outputs(config, symbols, basic_requests)
    request_index = pd.DataFrame.from_records(rows).sort_values(["code", "adjustflag"])
    paths = {
        "price_request_index": config.normalized_dir / "price_request_index.parquet",
    }
    _atomic_parquet(request_index, paths["price_request_index"])
    summary: dict[str, object] = {
        "phase": "price_catalog",
        "complete": len(symbols) == len(union),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_date": config.start_date,
        "end_date": config.end_date,
        "historical_union_symbols": len(union),
        "downloaded_symbols": len(symbols),
        "price_requests": len(request_index),
        "index_request": index_meta,
        "delisted_symbols": int((basic["status"].astype(str) == "0").sum()),
        "security_master_manifest": str(
            config.normalized_dir / "security_master_manifest.json"
        ),
        "security_master_request_index_sha256": security_summary["files"][
            "security_master_request_index"
        ]["sha256"],
        "files": {
            name: {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
    }
    _atomic_json(summary, config.normalized_dir / "price_manifest.json")
    return summary


def download_adjustment_factor_catalog(
    config: HS300SourceConfig,
    *,
    limit_symbols: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Run the resumable adjustment plan as the cache's sole active writer."""

    with _exclusive_price_writer_lock(Path(config.cache_dir)):
        return _download_adjustment_factor_catalog_locked(
            config,
            limit_symbols=limit_symbols,
            progress=progress,
        )


def _download_adjustment_factor_catalog_locked(
    config: HS300SourceConfig,
    *,
    limit_symbols: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Freeze one serial, content-addressed corporate-action request per name.

    The catalog is deliberately independent of the adjusted-price download.
    A later build can therefore verify ``adjustflag=1`` against BaoStock's
    dated ``backAdjustFactor`` events instead of treating rounded raw
    ``preclose`` values as the sole corporate-action authority.  Securities
    with no events are committed as successful zero-row responses; an absent
    request is never confused with an event-free history.
    """

    union_path = config.normalized_dir / "security_union.parquet"
    if not union_path.is_file():
        raise FileNotFoundError(
            "run membership download before downloading adjustment factors"
        )
    union = pd.read_parquet(union_path).sort_values("code")
    symbols = union["code"].astype(str).tolist()
    if limit_symbols is not None:
        if limit_symbols < 1:
            raise ValueError("limit_symbols must be positive")
        symbols = symbols[:limit_symbols]

    start = pd.Timestamp(config.start_date).normalize()
    end = pd.Timestamp(config.end_date).normalize()
    rows: list[dict[str, object]] = []
    bs_config = BaoStockConfig(config.cache_dir, max_attempts=config.max_attempts)
    with BaoStockSession(bs_config) as session:
        total = len(symbols)
        for number, code in enumerate(symbols, 1):
            frame = session.query_adjustment_factors(
                code,
                config.start_date,
                config.end_date,
                allow_empty=True,
            )
            if tuple(frame.columns.astype(str)) != ADJUST_FACTOR_FIELDS:
                raise ValueError(f"adjustment-factor schema drift for {code}")
            if len(frame):
                if not frame["code"].astype(str).eq(code).all():
                    raise ValueError(
                        f"adjustment-factor response contains another security for {code}"
                    )
                dates = pd.to_datetime(
                    frame["dividOperateDate"], errors="coerce"
                ).dt.normalize()
                if dates.isna().any() or (dates < start).any() or (dates > end).any():
                    raise ValueError(
                        f"adjustment-factor dates fall outside the frozen request for {code}"
                    )
                if dates.duplicated().any():
                    raise ValueError(
                        f"adjustment-factor response has duplicate event dates for {code}"
                    )
                for field in (
                    "foreAdjustFactor",
                    "backAdjustFactor",
                    "adjustFactor",
                ):
                    values = pd.to_numeric(frame[field], errors="coerce")
                    if values.isna().any() or not (values > 0).all():
                        raise ValueError(
                            f"adjustment-factor response has invalid {field} for {code}"
                        )
                first_event = str(dates.min().date())
                last_event = str(dates.max().date())
            else:
                first_event = ""
                last_event = ""
            cache = dict(frame.attrs.get("baostock_cache", {}))
            request_key = str(cache.get("request_key", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", request_key):
                raise ValueError(
                    f"adjustment-factor response lacks a committed request key for {code}"
                )
            rows.append(
                {
                    "code": code,
                    "rows": len(frame),
                    "first_event_date": first_event,
                    "last_event_date": last_event,
                    "request_key": request_key,
                    "cache_hit": bool(cache.get("cache_hit", False)),
                }
            )
            if progress is not None:
                progress(number, total, code)

    request_index = pd.DataFrame.from_records(
        rows,
        columns=(
            "code",
            "rows",
            "first_event_date",
            "last_event_date",
            "request_key",
            "cache_hit",
        ),
    ).sort_values("code")
    target = config.normalized_dir / "adjustment_request_index.parquet"
    _atomic_parquet(request_index, target)
    summary: dict[str, object] = {
        "phase": "adjustment_factor_catalog",
        "complete": len(symbols) == len(union),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_date": config.start_date,
        "end_date": config.end_date,
        "historical_union_symbols": len(union),
        "downloaded_symbols": len(symbols),
        "adjustment_factor_requests": len(request_index),
        "event_rows": int(request_index["rows"].sum()),
        "file": {
            "path": str(target),
            "sha256": _sha256(target),
            "bytes": target.stat().st_size,
        },
    }
    _atomic_json(summary, config.normalized_dir / "adjustment_manifest.json")
    return summary


def _materialize_industry_catalog(
    membership_snapshots: pd.DataFrame,
    union_codes: Iterable[str],
    requests: Iterable[tuple[pd.Timestamp, pd.DataFrame]],
    *,
    expected_request_dates: Iterable[object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact PIT industry catalog from canonical global requests."""

    required_membership = {
        "requested_date",
        "vendor_update_date",
        "code",
        "code_name",
    }
    if not required_membership.issubset(membership_snapshots.columns):
        raise ValueError("membership snapshots cannot define the industry query plan")
    membership = membership_snapshots.copy()
    membership["requested_date"] = pd.to_datetime(
        membership["requested_date"], errors="coerce"
    ).dt.normalize()
    membership["vendor_update_date"] = pd.to_datetime(
        membership["vendor_update_date"], errors="coerce"
    ).dt.normalize()
    if (
        membership[["requested_date", "vendor_update_date"]].isna().any().any()
        or (membership["vendor_update_date"] > membership["requested_date"]).any()
    ):
        raise ValueError("membership snapshots contain invalid vendor dates")
    union = {str(code).strip() for code in union_codes}
    requested_dates = pd.DatetimeIndex(
        pd.to_datetime(list(expected_request_dates), errors="coerce")
    ).normalize().unique().sort_values()
    if requested_dates.isna().any() or requested_dates.empty:
        raise ValueError("industry query plan contains no valid dates")
    supplied = sorted(
        [(pd.Timestamp(day).normalize(), frame) for day, frame in requests],
        key=lambda item: item[0],
    )
    supplied_dates = pd.DatetimeIndex([day for day, _ in supplied])
    if (
        supplied_dates.duplicated().any()
        or not supplied_dates.equals(requested_dates)
    ):
        raise ValueError("industry requests differ from the frozen membership query plan")
    membership_effective_dates = pd.DatetimeIndex(
        membership["vendor_update_date"].drop_duplicates().sort_values()
    )
    blocks: list[pd.DataFrame] = []
    request_rows: list[dict[str, object]] = []
    for requested, raw in supplied:
        if tuple(raw.columns.astype(str)) != INDUSTRY_FIELDS:
            raise ValueError(f"industry snapshot {requested.date()} schema drift")
        if raw["code"].astype(str).duplicated().any():
            raise ValueError(f"industry snapshot {requested.date()} has duplicate codes")
        update = pd.to_datetime(raw["updateDate"], errors="coerce").dt.normalize()
        if update.isna().any() or update.nunique() != 1 or update.iloc[0] > requested:
            raise ValueError(f"industry snapshot {requested.date()} has invalid updateDate")
        vendor_update = pd.Timestamp(update.iloc[0]).normalize()
        selected = raw.loc[raw["code"].astype(str).isin(union)].copy()
        effective_location = membership_effective_dates.searchsorted(
            requested, side="right"
        ) - 1
        if effective_location < 0:
            raise ValueError(
                f"industry request {requested.date()} precedes membership history"
            )
        effective_date = membership_effective_dates[effective_location]
        current_members = set(
            membership.loc[
                membership["vendor_update_date"].eq(effective_date), "code"
            ].astype(str)
        )
        # Membership observations can contain names that never reach a trading
        # session in the frozen research calendar (for example, a holiday
        # observation superseded before the next open).  ``union`` is derived
        # from the expanded daily membership and is therefore the exact price
        # and research universe.  Do not require or retain industry rows for
        # observation-only names outside that universe.
        current_members.intersection_update(union)
        missing = sorted(current_members.difference(selected["code"].astype(str)))
        if missing:
            raise ValueError(
                f"industry snapshot {requested.date()} misses current members: {missing[:10]}"
            )
        selected.insert(0, "requested_date", requested)
        selected.insert(1, "vendor_update_date", vendor_update)
        block = selected[
            [
                "requested_date",
                "vendor_update_date",
                "code",
                "code_name",
                "industry",
                "industryClassification",
            ]
        ].sort_values("code").reset_index(drop=True)
        blocks.append(block)
        request_rows.append(
            _catalog_cache_metadata(
                raw,
                endpoint="stock_industry",
                requested_date=str(requested.date()),
                vendor_update_date=str(vendor_update.date()),
            )
        )
    if not blocks:
        raise ValueError("industry catalog has no committed snapshots")
    industry = pd.concat(blocks, ignore_index=True).sort_values(
        ["requested_date", "code"], kind="stable"
    ).reset_index(drop=True)
    if industry.duplicated(["requested_date", "code"]).any():
        raise ValueError("industry catalog has duplicate date-symbol rows")
    request_index = pd.DataFrame.from_records(
        request_rows, columns=_INDUSTRY_REQUEST_INDEX_COLUMNS
    ).sort_values("requested_date", kind="stable").reset_index(drop=True)
    if request_index["request_key"].astype(str).duplicated().any():
        raise ValueError("industry request index contains duplicate request keys")
    return industry, request_index


def _industry_request_dates(
    membership_snapshots: pd.DataFrame,
    membership_request_index: pd.DataFrame,
) -> pd.DatetimeIndex:
    required = {"endpoint", "requested_date"}
    if not required.issubset(membership_request_index.columns):
        raise ValueError("membership request index cannot define industry cadence")
    observed = pd.to_datetime(
        membership_request_index.loc[
            membership_request_index["endpoint"].astype(str).eq("hs300_stocks"),
            "requested_date",
        ],
        errors="coerce",
    ).dt.normalize()
    effective = pd.to_datetime(
        membership_snapshots["vendor_update_date"], errors="coerce"
    ).dt.normalize()
    if observed.isna().any() or effective.isna().any() or observed.empty:
        raise ValueError("membership lineage contains invalid industry query dates")
    return pd.DatetimeIndex([*observed, *effective]).unique().sort_values()


def _freeze_industry_outputs(
    config: HS300SourceConfig,
    membership_snapshots: pd.DataFrame,
    membership_request_index: pd.DataFrame,
    union_codes: Iterable[str],
    requests: Iterable[tuple[pd.Timestamp, pd.DataFrame]],
) -> dict[str, object]:
    industry, request_index = _materialize_industry_catalog(
        membership_snapshots,
        union_codes,
        requests,
        expected_request_dates=_industry_request_dates(
            membership_snapshots, membership_request_index
        ),
    )
    paths = {
        "industry_snapshots": config.normalized_dir / "industry_snapshots.parquet",
        "industry_request_index": config.normalized_dir / "industry_request_index.parquet",
    }
    _freeze_parquet_exact(industry, paths["industry_snapshots"])
    _freeze_parquet_exact(
        request_index,
        paths["industry_request_index"],
        nonsemantic_columns=("cache_hit",),
    )
    summary: dict[str, object] = {
        "phase": "industry_catalog",
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "native_snapshots": int(industry["vendor_update_date"].nunique()),
        "requested_snapshots": len(request_index),
        "historical_union_symbols": len(set(str(code) for code in union_codes)),
        "rows": len(industry),
        "source_contract": {
            "schema_version": "1.0",
            "endpoint": "stock_industry",
            "query_code": "",
            "query_dates": (
                "membership_requested_observation_dates_union_vendor_effective_dates"
            ),
            "freshness_clock": "successful_requested_observation_date",
            "vendor_update_policy": "nonfuture_effective_metadata",
            "deterministic_union_filter": True,
            "future_update_policy": "forbidden",
        },
        "files": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    _atomic_json(summary, config.normalized_dir / "industry_manifest.json")
    return summary


def download_industry_catalog(
    config: HS300SourceConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Run the resumable industry plan as the cache's sole active writer."""

    with _exclusive_price_writer_lock(Path(config.cache_dir)):
        return _download_industry_catalog_locked(config, progress=progress)


def _download_industry_catalog_locked(
    config: HS300SourceConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Download and freeze each raw industry request and its exact replay."""

    snapshot_path = config.normalized_dir / "membership_snapshots.parquet"
    request_path = config.normalized_dir / "membership_request_index.parquet"
    union_path = config.normalized_dir / "security_union.parquet"
    if not snapshot_path.is_file() or not request_path.is_file() or not union_path.is_file():
        raise FileNotFoundError("run membership-lineage before downloading industry history")
    membership = pd.read_parquet(snapshot_path)
    membership_requests = pd.read_parquet(request_path)
    union = set(pd.read_parquet(union_path)["code"].astype(str))
    dates = _industry_request_dates(membership, membership_requests)
    requests: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    bs_config = BaoStockConfig(config.cache_dir, max_attempts=config.max_attempts)
    with BaoStockSession(bs_config) as session:
        total = len(dates)
        for number, requested in enumerate(dates, 1):
            frame = session.query_industry(code="", date=str(requested.date()))
            requests.append((requested, frame))
            if progress is not None:
                progress(number, total, str(requested.date()))
    return _freeze_industry_outputs(
        config, membership, membership_requests, union, requests
    )


def rebuild_industry_catalog_from_cache(
    config: HS300SourceConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Rebuild industry lineage as the sole normalized catalog writer."""

    with _exclusive_price_writer_lock(Path(config.cache_dir)):
        return _rebuild_industry_catalog_from_cache_locked(
            config,
            progress=progress,
        )


def _rebuild_industry_catalog_from_cache_locked(
    config: HS300SourceConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Re-freeze industry lineage entirely from committed raw requests."""

    snapshot_path = config.normalized_dir / "membership_snapshots.parquet"
    request_path = config.normalized_dir / "membership_request_index.parquet"
    union_path = config.normalized_dir / "security_union.parquet"
    industry_index_path = config.normalized_dir / "industry_request_index.parquet"
    if (
        not snapshot_path.is_file()
        or not request_path.is_file()
        or not union_path.is_file()
        or not industry_index_path.is_file()
    ):
        raise FileNotFoundError("run membership-lineage before industry-lineage")
    membership = pd.read_parquet(snapshot_path)
    membership_requests = pd.read_parquet(request_path)
    union = set(pd.read_parquet(union_path)["code"].astype(str))
    dates = _industry_request_dates(membership, membership_requests)
    source_index = _load_frozen_request_index(
        industry_index_path,
        expected_columns=_INDUSTRY_REQUEST_INDEX_COLUMNS,
    )
    expected_dates = tuple(str(day.date()) for day in dates)
    if (
        len(source_index) != len(expected_dates)
        or not source_index["endpoint"].astype(str).eq("stock_industry").all()
        or tuple(sorted(source_index["requested_date"].map(_index_text)))
        != tuple(sorted(expected_dates))
    ):
        raise CacheIntegrityError(
            "industry request index differs from the membership query plan"
        )
    requests: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for number, requested in enumerate(dates, 1):
        requested_date = str(requested.date())
        selected = _select_one_index_row(
            source_index,
            source_index["requested_date"].map(_index_text).eq(requested_date),
            label=f"industry request {requested_date}",
        )
        frame = _read_exact_cached_request(
            config.cache_dir,
            request_key=str(selected["request_key"]),
            endpoint="stock_industry",
            args={"code": "", "date": requested_date},
            expected_fields=INDUSTRY_FIELDS,
        )
        _strict_index_row_count(selected, frame)
        update = pd.to_datetime(frame["updateDate"], errors="coerce").dt.normalize()
        if (
            update.isna().any()
            or update.nunique() != 1
            or _index_text(selected["vendor_update_date"])
            != str(pd.Timestamp(update.iloc[0]).date())
        ):
            raise CacheIntegrityError(
                f"industry request index vendor date mismatch for {requested_date}"
            )
        requests.append((requested, frame))
        if progress is not None:
            progress(number, len(dates), str(requested.date()))
    return _freeze_industry_outputs(
        config, membership, membership_requests, union, requests
    )


def raw_cache_status(cache_dir: Path | str) -> dict[str, object]:
    state_dir = Path(cache_dir) / "state"
    states: list[dict[str, object]] = []
    for path in sorted(state_dir.glob("*.json")) if state_dir.is_dir() else []:
        try:
            states.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            states.append({"status": "corrupt", "path": str(path)})
    by_status: dict[str, int] = {}
    by_endpoint: dict[str, int] = {}
    for state in states:
        status = str(state.get("status", "unknown"))
        endpoint = str(state.get("endpoint", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        by_endpoint[endpoint] = by_endpoint.get(endpoint, 0) + 1
    return {
        "requests": len(states),
        "by_status": dict(sorted(by_status.items())),
        "by_endpoint": dict(sorted(by_endpoint.items())),
        "successful_rows": sum(int(item.get("rows", 0)) for item in states if item.get("status") == "success"),
    }


__all__ = [
    "HS300SourceConfig",
    "PriceCatalogAlreadyRunning",
    "download_adjustment_factor_catalog",
    "download_membership_catalog",
    "rebuild_membership_catalog_from_cache",
    "rebuild_security_master_from_cache",
    "download_industry_catalog",
    "rebuild_industry_catalog_from_cache",
    "download_price_catalog",
    "expand_membership_daily",
    "raw_cache_status",
]
