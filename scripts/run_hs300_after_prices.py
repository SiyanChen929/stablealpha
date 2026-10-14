#!/usr/bin/env python3
"""Resume the quality-gated HS300 pipeline after the serial price phase.

The BaoStock price downloader intentionally owns the provider's only active
session.  This watcher waits for its complete, frozen price manifest before it
opens a new session for industry history.  Every later phase is chained with
``check=True``; a failed download, build, independent audit or development run
stops the pipeline.  Historical replay reveal is intentionally absent.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import tempfile
from typing import BinaryIO, Iterator, Mapping, Sequence

import pandas as pd


FROZEN_PRICE_START_DATE = "2010-01-01"
FROZEN_ADJUSTMENT_START_DATE = "1990-01-01"
FROZEN_END_DATE = "2025-12-31"
DEFAULT_NETWORK_INACTIVITY_TIMEOUT_SECONDS = 600.0
MINIMUM_NETWORK_INACTIVITY_TIMEOUT_SECONDS = 30.0
MAXIMUM_NETWORK_INACTIVITY_TIMEOUT_SECONDS = 86_400.0
DEFAULT_NETWORK_ATTEMPTS = 8
DEFAULT_NETWORK_INITIAL_BACKOFF_SECONDS = 30.0
DEFAULT_NETWORK_MAXIMUM_BACKOFF_SECONDS = 600.0
NETWORK_WATCHDOG_POLL_SECONDS = 5.0
NETWORK_CHILD_TERMINATION_GRACE_SECONDS = 5.0
DEFAULT_SECURE_JSON_MAXIMUM_BYTES = 1024 * 1024
PRICE_HANDOFF_MAXIMUM_BYTES = 8 * 1024 * 1024
_REQUEST_KEY_RE = re.compile(r"[0-9a-f]{64}")
_CODE_RE = re.compile(r"(?:sh|sz)\.\d{6}")
IMPLEMENTATION_MANIFEST_RELATIVE_PATH = Path(
    "data/baostock_hs300/normalized/post_price_implementation_manifest.json"
)
_IMPLEMENTATION_DISTRIBUTIONS = (
    "baostock",
    "lightgbm",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "PyYAML",
)


class NetworkInactivityTimeout(subprocess.TimeoutExpired):
    """A post-price network child stopped making registered progress."""

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float,
        *,
        action: str,
        successful_requests: int,
    ) -> None:
        super().__init__(list(command), timeout_seconds)
        self.action = action
        self.successful_requests = successful_requests


@dataclass(frozen=True)
class _NetworkPhaseContract:
    action: str
    endpoint: str
    fields: tuple[str, ...]
    progress_items: tuple[str, ...]
    expected_args: Mapping[str, Mapping[str, str]]
    state_dir: Path
    bound_inputs: Mapping[Path, str]
    progress_every: int = 20
    union_size: int = 0
    provider_version: str | None = None

    def __post_init__(self) -> None:
        if self.action not in {"adjustments", "industry"}:
            raise ValueError("unsupported post-price network action")
        if not self.progress_items or len(set(self.progress_items)) != len(
            self.progress_items
        ):
            raise ValueError("network progress plan must be nonempty and unique")
        if set(self.expected_args) != set(self.progress_items):
            raise ValueError("network request plan differs from progress plan")
        if self.progress_every < 1:
            raise ValueError("network progress interval must be positive")
        if self.union_size < 1:
            raise ValueError("network contract requires a positive security union")
        if self.provider_version is not None and (
            not self.provider_version
            or self.provider_version != self.provider_version.strip()
        ):
            raise ValueError("network contract provider version is invalid")


def _event(kind: str, **payload: object) -> None:
    print(
        json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "event": kind,
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


@contextmanager
def _shared_catalog_read_lock(raw_dir: Path) -> Iterator[Path]:
    """Block all catalog EX writers for one long read-only phase."""

    lock_path = raw_dir / ".locks" / "price_catalog_writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"catalog writer is active; refusing concurrent reader: {lock_path}"
            ) from exc
        acquired = True
        yield lock_path
    finally:
        try:
            if acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _implementation_files(project: Path) -> tuple[Path, ...]:
    groups = {
        "source": sorted((project / "src" / "cogalpha_plus").rglob("*.py")),
        "scripts": [
            project / "scripts" / "run_hs300_after_prices.py",
            project / "scripts" / "supervise_hs300_pipeline.py",
        ],
        "configs": sorted(path for path in (project / "configs").rglob("*") if path.is_file()),
        "factor_cards": sorted(
            path for path in (project / "proposals").rglob("*") if path.is_file()
        ),
        "semantic_reviews": sorted(
            path for path in (project / "reviews").rglob("*") if path.is_file()
        ),
    }
    if any(not paths for paths in groups.values()):
        missing = sorted(name for name, paths in groups.items() if not paths)
        raise RuntimeError(f"implementation freeze has empty required groups: {missing}")
    files = tuple(sorted({path for paths in groups.values() for path in paths}))
    project_root = project.resolve()
    for path in files:
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(project_root)
        ):
            raise RuntimeError(f"implementation input is unsafe: {path}")
    return files


def _runtime_versions() -> dict[str, object]:
    distributions: dict[str, str] = {}
    for name in _IMPLEMENTATION_DISTRIBUTIONS:
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = "not-installed"
    provider = importlib.import_module("baostock")
    provider_version = str(getattr(provider, "__version__", "")).strip()
    if not provider_version:
        raise RuntimeError("cannot freeze BaoStock client version")
    return {
        "python_version": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "baostock_client_version": provider_version,
        "distributions": distributions,
    }


def _implementation_snapshot(project: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in _implementation_files(project):
        metadata = _secure_file_metadata(path)
        if metadata is None:
            raise RuntimeError(f"cannot securely hash implementation input: {path}")
        digest, size = metadata
        files.append(
            {
                "path": str(path.relative_to(project)),
                "sha256": digest,
                "bytes": size,
            }
        )
    runtime = _runtime_versions()
    body = {
        "schema_version": 1,
        "files": files,
        "runtime": runtime,
    }
    aggregate = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        **body,
        "file_count": len(files),
        "aggregate_sha256": aggregate,
    }


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _freeze_post_price_implementation(project: Path) -> dict[str, object]:
    snapshot = _implementation_snapshot(project)
    target = project / IMPLEMENTATION_MANIFEST_RELATIVE_PATH
    if target.exists():
        existing = _load_regular_json(target)
        if existing != snapshot:
            raise RuntimeError(
                "post-price implementation differs from the already frozen manifest"
            )
    else:
        _atomic_json_write(target, snapshot)
        if _load_regular_json(target) != snapshot:
            raise RuntimeError("post-price implementation manifest publication failed")
    return snapshot


def _assert_post_price_implementation(
    project: Path,
    frozen: Mapping[str, object],
) -> None:
    target = project / IMPLEMENTATION_MANIFEST_RELATIVE_PATH
    if _load_regular_json(target) != dict(frozen):
        raise RuntimeError("post-price implementation manifest changed")
    if _implementation_snapshot(project) != dict(frozen):
        raise RuntimeError("post-price implementation/configuration changed after freeze")


def _bound_input_hashes(paths: Sequence[Path]) -> dict[Path, str]:
    hashes: dict[Path, str] = {}
    for path in paths:
        digest = _secure_file_sha256(path)
        if digest is None:
            raise FileNotFoundError(f"network phase input is not a regular file: {path}")
        hashes[path] = digest
    return hashes


def _assert_bound_inputs_unchanged(contract: _NetworkPhaseContract) -> None:
    for path, expected_hash in contract.bound_inputs.items():
        if _secure_file_sha256(path) != expected_hash:
            raise RuntimeError(
                f"{contract.action} request-plan input changed during supervision: {path}"
            )


def _load_unique_union(normalized: Path) -> tuple[str, ...]:
    path = normalized / "security_union.parquet"
    frame = pd.read_parquet(path, columns=["code"])
    codes = tuple(sorted(frame["code"].astype(str)))
    if (
        not codes
        or len(set(codes)) != len(codes)
        or any(_CODE_RE.fullmatch(code) is None for code in codes)
    ):
        raise ValueError("security union is empty, duplicated, or malformed")
    return codes


def _network_phase_contract(project: Path, action: str) -> _NetworkPhaseContract:
    """Bind a network child to the exact frozen canonical request plan."""

    source_path = str(project / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from cogalpha_plus.data_sources.baostock import (
        ADJUST_FACTOR_FIELDS,
        INDUSTRY_FIELDS,
    )
    from cogalpha_plus.data_sources.hs300_download import _industry_request_dates

    provider = importlib.import_module("baostock")
    provider_version = str(getattr(provider, "__version__", "")).strip()
    if not provider_version:
        raise RuntimeError("cannot bind BaoStock client version")

    root = project / "data" / "baostock_hs300"
    normalized = root / "normalized"
    union_path = normalized / "security_union.parquet"
    codes = _load_unique_union(normalized)
    if action == "adjustments":
        items = codes
        expected_args = {
            code: {
                "code": code,
                "start_date": FROZEN_ADJUSTMENT_START_DATE,
                "end_date": FROZEN_END_DATE,
            }
            for code in codes
        }
        endpoint = "adjust_factor"
        fields = tuple(ADJUST_FACTOR_FIELDS)
        inputs = [union_path]
    elif action == "industry":
        membership_path = normalized / "membership_snapshots.parquet"
        request_index_path = normalized / "membership_request_index.parquet"
        membership = pd.read_parquet(membership_path)
        request_index = pd.read_parquet(request_index_path)
        dates = _industry_request_dates(membership, request_index)
        items = tuple(str(pd.Timestamp(value).date()) for value in dates)
        expected_args = {
            item: {"code": "", "date": item}
            for item in items
        }
        endpoint = "stock_industry"
        fields = tuple(INDUSTRY_FIELDS)
        inputs = [union_path, membership_path, request_index_path]
    else:
        raise ValueError(f"unsupported post-price network action: {action}")
    return _NetworkPhaseContract(
        action=action,
        endpoint=endpoint,
        fields=fields,
        progress_items=items,
        expected_args=expected_args,
        state_dir=root / "raw" / "state",
        bound_inputs=_bound_input_hashes(inputs),
        union_size=len(codes),
        provider_version=provider_version,
    )


def _canonical_request_key(
    endpoint: str,
    args: Mapping[str, str],
    provider_version: str,
) -> str:
    payload = {
        "provider": "baostock",
        "provider_version": provider_version,
        "endpoint": endpoint,
        "args": dict(args),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _secure_file_metadata(path: Path) -> tuple[str, int] | None:
    if not hasattr(os, "O_NOFOLLOW"):
        return None
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, OSError):
        return None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return None
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        try:
            current = path.stat(follow_symlinks=False)
        except (FileNotFoundError, OSError):
            return None
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(metadata) != identity(after) or identity(after) != identity(current):
            return None
        return digest.hexdigest(), metadata.st_size
    finally:
        os.close(descriptor)


def _secure_file_sha256(path: Path) -> str | None:
    metadata = _secure_file_metadata(path)
    return None if metadata is None else metadata[0]


def _secure_json_state_with_metadata(
    path: Path,
    *,
    maximum_bytes: int = DEFAULT_SECURE_JSON_MAXIMUM_BYTES,
) -> tuple[object, tuple[int, int, int, int], str, int] | None:
    """Read one bounded regular JSON state without following a swapped link."""

    if not hasattr(os, "O_NOFOLLOW"):
        return None
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, OSError):
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            maximum_bytes <= 0
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > maximum_bytes
        ):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(maximum_bytes + 1)
        if len(raw) > maximum_bytes:
            return None
        after = os.fstat(descriptor)
        try:
            current = path.stat(follow_symlinks=False)
        except (FileNotFoundError, OSError):
            return None
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(metadata) != identity(after) or identity(after) != identity(current):
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        signature = (
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        return payload, signature, hashlib.sha256(raw).hexdigest(), len(raw)
    finally:
        os.close(descriptor)


def _secure_json_state(
    path: Path,
    *,
    maximum_bytes: int = DEFAULT_SECURE_JSON_MAXIMUM_BYTES,
) -> tuple[object, tuple[int, int, int, int]] | None:
    observed = _secure_json_state_with_metadata(
        path,
        maximum_bytes=maximum_bytes,
    )
    if observed is None:
        return None
    payload, signature, _digest, _size = observed
    return payload, signature


def _is_canonical_network_state(
    payload: object,
    *,
    request_key: str,
    contract: _NetworkPhaseContract,
) -> bool:
    """Accept only states reachable from the bound network-phase command."""

    if not isinstance(payload, dict):
        return False
    status = payload.get("status")
    attempt_count = payload.get("attempt_count")
    provider_version = payload.get("provider_version")
    args = payload.get("args")
    base_keys = {
        "state_schema_version",
        "request_key",
        "provider",
        "provider_version",
        "endpoint",
        "args",
        "attempt_count",
        "status",
        "started_at",
    }
    expected_keys = {
        "running": base_keys,
        "success": base_keys
        | {"completed_at", "rows", "fields", "object", "parquet_sha256"},
        "failed": base_keys | {"completed_at", "error_type", "error"},
    }
    if (
        status not in expected_keys
        or set(payload) != expected_keys[status]
        or payload.get("state_schema_version") != 1
        or type(payload.get("state_schema_version")) is not int
        or payload.get("request_key") != request_key
        or payload.get("provider") != "baostock"
        or payload.get("endpoint") != contract.endpoint
        or type(attempt_count) is not int
        or attempt_count <= 0
        or not isinstance(provider_version, str)
        or provider_version != provider_version.strip()
        or not provider_version
        or (
            contract.provider_version is not None
            and provider_version != contract.provider_version
        )
        or not isinstance(args, dict)
        or not isinstance(payload.get("started_at"), str)
        or not payload.get("started_at")
    ):
        return False
    item = (
        str(args.get("code", ""))
        if contract.action == "adjustments"
        else str(args.get("date", ""))
    )
    expected_args = contract.expected_args.get(item)
    if expected_args is None or args != dict(expected_args):
        return False
    if _canonical_request_key(
        contract.endpoint,
        expected_args,
        provider_version,
    ) != request_key:
        return False
    if status == "running":
        return True
    if (
        not isinstance(payload.get("completed_at"), str)
        or not payload.get("completed_at")
    ):
        return False
    if status == "failed":
        return isinstance(payload.get("error_type"), str) and isinstance(
            payload.get("error"), str
        )
    rows = payload.get("rows")
    object_relative = Path("objects") / contract.endpoint / f"{request_key}.parquet"
    declared_hash = payload.get("parquet_sha256")
    if (
        type(rows) is not int
        or rows < 0
        or (contract.action == "industry" and rows == 0)
        or payload.get("fields") != list(contract.fields)
        or payload.get("object") != str(object_relative)
        or not isinstance(declared_hash, str)
        or _REQUEST_KEY_RE.fullmatch(declared_hash) is None
    ):
        return False
    object_path = contract.state_dir.parent / object_relative
    try:
        frame = _secure_parquet(
            object_path,
            expected_sha256=declared_hash,
        )
    except RuntimeError:
        return False
    return (
        tuple(frame.columns.astype(str)) == contract.fields
        and len(frame) == rows
    )


class _CanonicalNetworkStateTracker:
    """Track only valid cache-state transitions in the bound request plan."""

    def __init__(self, contract: _NetworkPhaseContract) -> None:
        self.contract = contract
        self._known: dict[
            Path,
            tuple[tuple[int, int, int, int], bool, str, int, str],
        ] = {}
        self._initialized = False

    @staticmethod
    def _advanced(
        previous_status: str,
        previous_attempt: int,
        status: str,
        attempt: int,
    ) -> bool:
        if attempt > previous_attempt:
            return True
        return (
            attempt == previous_attempt
            and previous_status == "running"
            and status in {"success", "failed"}
        )

    def observe(self) -> tuple[bool, int]:
        current_paths: set[Path] = set()
        changed = False
        state_dir = self.contract.state_dir
        if state_dir.is_dir():
            with os.scandir(state_dir) as entries:
                for entry in entries:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    path = Path(entry.path)
                    if path.suffix != ".json" or _REQUEST_KEY_RE.fullmatch(
                        path.stem
                    ) is None:
                        continue
                    current_paths.add(path)
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    signature = (
                        metadata.st_ino,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                    )
                    previous = self._known.get(path)
                    if previous is not None and previous[0] == signature:
                        continue
                    observed = _secure_json_state(path)
                    if observed is None:
                        continue
                    payload, signature = observed
                    registered = _is_canonical_network_state(
                        payload,
                        request_key=path.stem,
                        contract=self.contract,
                    )
                    status = str(payload.get("status", ""))
                    raw_attempt = payload.get("attempt_count")
                    attempt = raw_attempt if type(raw_attempt) is int else -1
                    args = payload.get("args")
                    item = ""
                    if registered and isinstance(args, dict):
                        item = (
                            str(args.get("code", ""))
                            if self.contract.action == "adjustments"
                            else str(args.get("date", ""))
                        )
                    self._known[path] = (
                        signature,
                        registered,
                        status,
                        attempt,
                        item,
                    )
                    if self._initialized and registered:
                        if previous is None or not previous[1]:
                            changed = True
                        elif self._advanced(
                            previous[2], previous[3], status, attempt
                        ):
                            changed = True
        for path in set(self._known).difference(current_paths):
            self._known.pop(path)
        successful = sum(
            registered and status == "success"
            for _signature, registered, status, _attempt, _item in self._known.values()
        )
        self._initialized = True
        return changed, successful

    def successful_items(self) -> frozenset[str]:
        return frozenset(
            item
            for _signature, registered, status, _attempt, item in self._known.values()
            if registered and status == "success"
        )


def _secure_parquet(
    path: Path,
    *,
    expected_sha256: str,
) -> pd.DataFrame:
    """Hash and decode the same regular-file bytes without path reopening."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for canonical Parquet validation")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeError(f"cannot securely open canonical Parquet {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"canonical Parquet is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        try:
            current = path.stat(follow_symlinks=False)
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(f"canonical Parquet path changed: {path}") from exc
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(metadata) != identity(after) or identity(after) != identity(current):
            raise RuntimeError(f"canonical Parquet changed while reading: {path}")
    finally:
        os.close(descriptor)
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_sha256:
        raise RuntimeError(f"canonical Parquet hash mismatch: {path}")
    try:
        return pd.read_parquet(io.BytesIO(raw))
    except Exception as exc:
        raise RuntimeError(f"cannot decode canonical Parquet {path}: {exc}") from exc


def _canonical_success_states(
    contract: _NetworkPhaseContract,
) -> dict[str, tuple[str, dict[str, object], pd.DataFrame]]:
    """Decode every canonical success, keyed by versioned request key.

    Multiple provider versions for one logical item are legitimate immutable
    history.  The normalized request index later selects the exact version
    used by this run.
    """

    successes: dict[str, tuple[str, dict[str, object], pd.DataFrame]] = {}
    state_dir = contract.state_dir
    for path in sorted(state_dir.glob("*.json")) if state_dir.is_dir() else []:
        if _REQUEST_KEY_RE.fullmatch(path.stem) is None:
            continue
        observed = _secure_json_state(path)
        if observed is None:
            continue
        payload, _signature = observed
        if (
            not isinstance(payload, dict)
            or payload.get("status") != "success"
            or not _is_canonical_network_state(
                payload,
                request_key=path.stem,
                contract=contract,
            )
        ):
            continue
        args = payload["args"]
        item = (
            str(args["code"])
            if contract.action == "adjustments"
            else str(args["date"])
        )
        object_path = contract.state_dir.parent / str(payload["object"])
        frame = _secure_parquet(
            object_path,
            expected_sha256=str(payload["parquet_sha256"]),
        )
        if tuple(frame.columns.astype(str)) != contract.fields:
            raise RuntimeError(
                f"{contract.action} canonical object schema mismatch for {item}"
            )
        if len(frame) != payload["rows"]:
            raise RuntimeError(
                f"{contract.action} canonical object row mismatch for {item}"
            )
        successes[path.stem] = (item, payload, frame)
    return successes


def _load_regular_json(
    path: Path,
    *,
    maximum_bytes: int = DEFAULT_SECURE_JSON_MAXIMUM_BYTES,
) -> dict[str, object]:
    observed = _secure_json_state(path, maximum_bytes=maximum_bytes)
    if observed is None or not isinstance(observed[0], dict):
        raise RuntimeError(f"cannot securely decode manifest: {path}")
    return observed[0]


def _validate_file_declaration(
    declaration: object,
    path: Path,
) -> str:
    if not isinstance(declaration, dict) or set(declaration) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise RuntimeError(f"invalid file declaration for {path}")
    declared_hash = declaration.get("sha256")
    declared_bytes = declaration.get("bytes")
    actual_metadata = _secure_file_metadata(path)
    if (
        declaration.get("path") != str(path)
        or not isinstance(declared_hash, str)
        or _REQUEST_KEY_RE.fullmatch(declared_hash) is None
        or type(declared_bytes) is not int
        or declared_bytes <= 0
        or actual_metadata != (declared_hash, declared_bytes)
    ):
        raise RuntimeError(f"file declaration does not bind {path}")
    return declared_hash


def _validate_adjustment_phase_complete(
    contract: _NetworkPhaseContract,
    successes: Mapping[str, tuple[str, dict[str, object], pd.DataFrame]],
) -> None:
    normalized = contract.state_dir.parent.parent / "normalized"
    index_path = normalized / "adjustment_request_index.parquet"
    manifest_path = normalized / "adjustment_manifest.json"
    manifest = _load_regular_json(manifest_path)
    expected_manifest_keys = {
        "phase",
        "complete",
        "created_at",
        "start_date",
        "end_date",
        "historical_union_symbols",
        "downloaded_symbols",
        "adjustment_factor_requests",
        "event_rows",
        "file",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("phase") != "adjustment_factor_catalog"
        or manifest.get("complete") is not True
        or manifest.get("start_date") != FROZEN_ADJUSTMENT_START_DATE
        or manifest.get("end_date") != FROZEN_END_DATE
        or type(manifest.get("historical_union_symbols")) is not int
        or manifest.get("historical_union_symbols") != contract.union_size
        or type(manifest.get("downloaded_symbols")) is not int
        or manifest.get("downloaded_symbols") != contract.union_size
        or type(manifest.get("adjustment_factor_requests")) is not int
        or manifest.get("adjustment_factor_requests") != len(contract.progress_items)
        or type(manifest.get("event_rows")) is not int
    ):
        raise RuntimeError("adjustment manifest differs from the frozen contract")
    index_hash = _validate_file_declaration(manifest.get("file"), index_path)
    index = _secure_parquet(index_path, expected_sha256=index_hash)
    expected_columns = (
        "code",
        "rows",
        "first_event_date",
        "last_event_date",
        "request_key",
        "cache_hit",
    )
    if (
        tuple(index.columns.astype(str)) != expected_columns
        or tuple(index["code"].astype(str)) != contract.progress_items
        or len(index) != len(contract.progress_items)
        or not pd.api.types.is_integer_dtype(index["rows"].dtype)
        or pd.api.types.is_bool_dtype(index["rows"].dtype)
    ):
        raise RuntimeError("adjustment request index differs from the frozen plan")
    expected_keys = index["request_key"].astype(str).tolist()
    if len(set(expected_keys)) != len(expected_keys):
        raise RuntimeError("adjustment request index contains duplicate request keys")
    selected: dict[str, tuple[dict[str, object], pd.DataFrame]] = {}
    for item, request_key in zip(contract.progress_items, expected_keys, strict=True):
        record = successes.get(request_key)
        if record is None or record[0] != item:
            raise RuntimeError(
                f"adjustment request index selects no canonical success for {item}"
            )
        selected[item] = (record[1], record[2])
    provider_versions = {
        str(payload["provider_version"]) for payload, _frame in selected.values()
    }
    if len(provider_versions) != 1:
        raise RuntimeError("adjustment request index mixes provider versions")
    if (
        contract.provider_version is not None
        and provider_versions != {contract.provider_version}
    ):
        raise RuntimeError("adjustment request index selects another provider version")
    expected_rows = [int(selected[item][0]["rows"]) for item in contract.progress_items]
    if (
        index["rows"].astype(int).tolist() != expected_rows
        or manifest.get("event_rows") != sum(expected_rows)
    ):
        raise RuntimeError("adjustment request index/cache lineage mismatch")
    for position, item in enumerate(contract.progress_items):
        frame = selected[item][1]
        if len(frame):
            dates = pd.to_datetime(
                frame["dividOperateDate"],
                errors="coerce",
            ).dt.normalize()
            if dates.isna().any() or dates.duplicated().any():
                raise RuntimeError(f"adjustment event dates are invalid for {item}")
            first, last = str(dates.min().date()), str(dates.max().date())
        else:
            first = last = ""
        if (
            str(index.iloc[position]["first_event_date"]) != first
            or str(index.iloc[position]["last_event_date"]) != last
        ):
            raise RuntimeError(f"adjustment event bounds mismatch for {item}")


def _validate_industry_phase_complete(
    contract: _NetworkPhaseContract,
    successes: Mapping[str, tuple[str, dict[str, object], pd.DataFrame]],
) -> None:
    normalized = contract.state_dir.parent.parent / "normalized"
    index_path = normalized / "industry_request_index.parquet"
    snapshots_path = normalized / "industry_snapshots.parquet"
    manifest_path = normalized / "industry_manifest.json"
    manifest = _load_regular_json(manifest_path)
    expected_manifest_keys = {
        "phase",
        "complete",
        "created_at",
        "native_snapshots",
        "requested_snapshots",
        "historical_union_symbols",
        "rows",
        "source_contract",
        "files",
    }
    expected_source_contract = {
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
    }
    files = manifest.get("files")
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("phase") != "industry_catalog"
        or manifest.get("complete") is not True
        or type(manifest.get("requested_snapshots")) is not int
        or manifest.get("requested_snapshots") != len(contract.progress_items)
        or type(manifest.get("historical_union_symbols")) is not int
        or manifest.get("historical_union_symbols") != contract.union_size
        or type(manifest.get("native_snapshots")) is not int
        or manifest.get("native_snapshots", 0) < 1
        or type(manifest.get("rows")) is not int
        or manifest.get("rows", 0) < 1
        or manifest.get("source_contract") != expected_source_contract
        or not isinstance(files, dict)
        or set(files) != {"industry_snapshots", "industry_request_index"}
    ):
        raise RuntimeError("industry manifest differs from the frozen contract")
    index_hash = _validate_file_declaration(
        files["industry_request_index"],
        index_path,
    )
    snapshots_hash = _validate_file_declaration(
        files["industry_snapshots"],
        snapshots_path,
    )
    index = _secure_parquet(index_path, expected_sha256=index_hash)
    snapshots = _secure_parquet(snapshots_path, expected_sha256=snapshots_hash)
    expected_index_columns = (
        "endpoint",
        "requested_date",
        "vendor_update_date",
        "rows",
        "request_key",
        "cache_hit",
    )
    actual_dates = tuple(
        str(pd.Timestamp(value).date()) for value in index["requested_date"]
    )
    if (
        tuple(index.columns.astype(str)) != expected_index_columns
        or len(index) != len(contract.progress_items)
        or actual_dates != contract.progress_items
        or not index["endpoint"].astype(str).eq("stock_industry").all()
        or not pd.api.types.is_integer_dtype(index["rows"].dtype)
        or pd.api.types.is_bool_dtype(index["rows"].dtype)
    ):
        raise RuntimeError("industry request index differs from the frozen plan")
    expected_keys = index["request_key"].astype(str).tolist()
    if len(set(expected_keys)) != len(expected_keys):
        raise RuntimeError("industry request index contains duplicate request keys")
    selected: dict[str, tuple[dict[str, object], pd.DataFrame]] = {}
    for item, request_key in zip(contract.progress_items, expected_keys, strict=True):
        record = successes.get(request_key)
        if record is None or record[0] != item:
            raise RuntimeError(
                f"industry request index selects no canonical success for {item}"
            )
        selected[item] = (record[1], record[2])
    provider_versions = {
        str(payload["provider_version"]) for payload, _frame in selected.values()
    }
    if len(provider_versions) != 1:
        raise RuntimeError("industry request index mixes provider versions")
    if (
        contract.provider_version is not None
        and provider_versions != {contract.provider_version}
    ):
        raise RuntimeError("industry request index selects another provider version")
    expected_rows = [int(selected[item][0]["rows"]) for item in contract.progress_items]
    if (
        index["rows"].astype(int).tolist() != expected_rows
    ):
        raise RuntimeError("industry request index/cache lineage mismatch")
    expected_snapshot_columns = (
        "requested_date",
        "vendor_update_date",
        "code",
        "code_name",
        "industry",
        "industryClassification",
    )
    snapshot_dates = {
        str(pd.Timestamp(value).date()) for value in snapshots["requested_date"]
    }
    if (
        tuple(snapshots.columns.astype(str)) != expected_snapshot_columns
        or len(snapshots) != manifest.get("rows")
        or snapshot_dates != set(contract.progress_items)
        or snapshots.duplicated(["requested_date", "code"]).any()
    ):
        raise RuntimeError("industry snapshots differ from the frozen contract")
    source_path = str(Path(__file__).resolve().parents[1] / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from cogalpha_plus.data_sources.hs300_download import (
        _materialize_industry_catalog,
    )

    inputs = {path.name: (path, digest) for path, digest in contract.bound_inputs.items()}
    required_inputs = {
        "membership_snapshots.parquet",
        "membership_request_index.parquet",
        "security_union.parquet",
    }
    if set(inputs) != required_inputs:
        raise RuntimeError("industry watchdog inputs differ from the frozen contract")
    membership = _secure_parquet(
        inputs["membership_snapshots.parquet"][0],
        expected_sha256=inputs["membership_snapshots.parquet"][1],
    )
    membership_requests = _secure_parquet(
        inputs["membership_request_index.parquet"][0],
        expected_sha256=inputs["membership_request_index.parquet"][1],
    )
    union_frame = _secure_parquet(
        inputs["security_union.parquet"][0],
        expected_sha256=inputs["security_union.parquet"][1],
    )
    raw_requests: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for item in contract.progress_items:
        payload, frame = selected[item]
        replay = frame.copy()
        replay.attrs["baostock_cache"] = {
            "request_key": payload["request_key"],
            "cache_hit": True,
        }
        raw_requests.append((pd.Timestamp(item), replay))
    rebuilt_snapshots, rebuilt_index = _materialize_industry_catalog(
        membership,
        set(union_frame["code"].astype(str)),
        raw_requests,
        expected_request_dates=pd.to_datetime(contract.progress_items),
    )

    def canonical_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
        canonical = frame.copy()
        for column in ("requested_date", "vendor_update_date"):
            dates = pd.to_datetime(canonical[column], errors="coerce").dt.normalize()
            if dates.isna().any():
                raise RuntimeError("industry snapshots contain invalid dates")
            canonical[column] = dates.astype("datetime64[ns]")
        for column in expected_snapshot_columns[2:]:
            canonical[column] = canonical[column].astype("string")
        return canonical.sort_values(
            ["requested_date", "code"], kind="stable"
        ).reset_index(drop=True)

    if not canonical_snapshots(snapshots).equals(
        canonical_snapshots(rebuilt_snapshots)
    ):
        raise RuntimeError("industry snapshots do not reproduce from canonical raw cache")
    semantic_index_columns = [
        column for column in expected_index_columns if column != "cache_hit"
    ]

    def canonical_request_index(frame: pd.DataFrame) -> pd.DataFrame:
        canonical = frame[semantic_index_columns].copy()
        for column in ("requested_date", "vendor_update_date"):
            dates = pd.to_datetime(canonical[column], errors="coerce").dt.normalize()
            if dates.isna().any():
                raise RuntimeError("industry request index contains invalid dates")
            canonical[column] = dates.astype("datetime64[ns]")
        for column in ("endpoint", "request_key"):
            canonical[column] = canonical[column].astype("string")
        canonical["rows"] = canonical["rows"].astype("int64")
        return canonical.sort_values("requested_date", kind="stable").reset_index(
            drop=True
        )

    if not canonical_request_index(index).equals(
        canonical_request_index(rebuilt_index)
    ):
        raise RuntimeError("industry request index does not reproduce from canonical raw cache")


def _network_contract_attestation(
    contract: _NetworkPhaseContract,
) -> dict[str, object]:
    request_plan = [
        {
            "item": item,
            "args": dict(contract.expected_args[item]),
        }
        for item in contract.progress_items
    ]
    request_plan_sha256 = hashlib.sha256(
        json.dumps(
            request_plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "action": contract.action,
        "endpoint": contract.endpoint,
        "provider_version": contract.provider_version,
        "fields": list(contract.fields),
        "progress_every": contract.progress_every,
        "request_count": len(contract.progress_items),
        "union_size": contract.union_size,
        "request_plan": request_plan,
        "request_plan_sha256": request_plan_sha256,
        "bound_inputs": [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(
                contract.bound_inputs.items(),
                key=lambda pair: str(pair[0]),
            )
        ],
    }


def _network_contract_attestation_path(contract: _NetworkPhaseContract) -> Path:
    normalized = contract.state_dir.parent.parent / "normalized"
    return normalized / f"{contract.action}_watchdog_contract.json"


def _freeze_network_contract_attestation(
    contract: _NetworkPhaseContract,
) -> dict[str, object]:
    payload = _network_contract_attestation(contract)
    path = _network_contract_attestation_path(contract)
    if path.exists():
        if _load_regular_json(path) != payload:
            raise RuntimeError(
                f"{contract.action} watchdog contract differs from frozen attestation"
            )
    else:
        _atomic_json_write(path, payload)
        if _load_regular_json(path) != payload:
            raise RuntimeError(
                f"{contract.action} watchdog contract publication failed"
            )
    return payload


def _assert_network_contract_attestation(
    contract: _NetworkPhaseContract,
    frozen: Mapping[str, object],
) -> None:
    if _network_contract_attestation(contract) != dict(frozen):
        raise RuntimeError(f"{contract.action} in-memory watchdog contract changed")
    path = _network_contract_attestation_path(contract)
    if _load_regular_json(path) != dict(frozen):
        raise RuntimeError(f"{contract.action} watchdog contract file changed")
    _assert_bound_inputs_unchanged(contract)


def _validate_network_phase_complete(contract: _NetworkPhaseContract) -> None:
    successes = _canonical_success_states(contract)
    if contract.action == "adjustments":
        _validate_adjustment_phase_complete(contract, successes)
    else:
        _validate_industry_phase_complete(contract, successes)


class _StrictCLIProgressTracker:
    """Accept only monotonically advancing lines from the bound CLI plan."""

    def __init__(self, items: Sequence[str], *, every: int) -> None:
        self.items = tuple(items)
        positions = {1, len(self.items)}
        positions.update(range(every, len(self.items) + 1, every))
        self.expected_positions = tuple(sorted(positions))
        self.next_position = 0

    def observe(
        self,
        payload: object,
        *,
        committed_items: frozenset[str],
    ) -> bool:
        if not isinstance(payload, dict) or set(payload) != {
            "progress",
            "total",
            "item",
        }:
            return False
        done = payload.get("progress")
        total = payload.get("total")
        item = payload.get("item")
        if (
            type(done) is not int
            or type(total) is not int
            or total != len(self.items)
            or not 1 <= done <= total
            or type(item) is not str
            or item != self.items[done - 1]
            or item not in committed_items
            or self.next_position >= len(self.expected_positions)
            or done != self.expected_positions[self.next_position]
        ):
            return False
        self.next_position += 1
        return True


def _relay_child_output(
    stream: BinaryIO,
    activity: "queue.SimpleQueue[object]",
) -> None:
    """Forward child logs and queue only JSON objects for strict validation."""

    for raw_line in iter(stream.readline, b""):
        try:
            payload = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            activity.put(payload)
        try:
            target = getattr(sys.stdout, "buffer", None)
            if target is not None:
                target.write(raw_line)
                target.flush()
            else:
                sys.stdout.write(raw_line.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        except (OSError, ValueError):
            continue


def _drain_strict_cli_progress(
    activity: "queue.SimpleQueue[object]",
    tracker: _StrictCLIProgressTracker,
    *,
    committed_items: frozenset[str],
) -> bool:
    advanced = False
    while True:
        try:
            payload = activity.get_nowait()
        except queue.Empty:
            return advanced
        advanced = tracker.observe(
            payload,
            committed_items=committed_items,
        ) or advanced


def _terminate_child_only(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = NETWORK_CHILD_TERMINATION_GRACE_SECONDS,
) -> None:
    """Terminate and reap only the directly supervised network child."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("post-price network child survived SIGKILL") from exc


def _raise_controlled_signal_exit(signum: int, _frame: object) -> None:
    """Unwind through the child-cleanup path on supervisor termination."""

    raise SystemExit(128 + signum)


def _defer_controlled_signal(
    pending: list[int],
    signum: int,
    _frame: object,
) -> None:
    """Record a signal until Popen has returned a child that can be reaped."""

    pending.append(signum)


def _run_network_phase_once(
    command: Sequence[str],
    *,
    project: Path,
    environment: Mapping[str, str],
    contract: _NetworkPhaseContract,
    inactivity_timeout_seconds: float,
    poll_seconds: float = NETWORK_WATCHDOG_POLL_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    """Run one child until exit or genuine canonical inactivity."""

    if (
        not math.isfinite(inactivity_timeout_seconds)
        or inactivity_timeout_seconds <= 0
    ):
        raise ValueError("network inactivity timeout must be finite and positive")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("network watchdog poll must be finite and positive")
    _assert_bound_inputs_unchanged(contract)
    state_tracker = _CanonicalNetworkStateTracker(contract)
    _changed, successful_requests = state_tracker.observe()
    cli_tracker = _StrictCLIProgressTracker(
        contract.progress_items,
        every=contract.progress_every,
    )
    activity: queue.SimpleQueue[object] = queue.SimpleQueue()
    handled_signals = (signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in handled_signals
    }
    pending_signals: list[int] = []
    for signum in handled_signals:
        signal.signal(
            signum,
            lambda received, frame, pending=pending_signals: (
                _defer_controlled_signal(pending, received, frame)
            ),
        )
    try:
        process = subprocess.Popen(
            list(command),
            cwd=project,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        relay: threading.Thread | None = None
        try:
            for signum in handled_signals:
                signal.signal(signum, _raise_controlled_signal_exit)
            if pending_signals:
                raise SystemExit(128 + pending_signals[0])
            if process.stdout is None:  # pragma: no cover - Popen contract defense.
                raise RuntimeError("network child stdout pipe was not created")
            relay = threading.Thread(
                target=_relay_child_output,
                args=(process.stdout, activity),
                name=f"hs300-{contract.action}-output-relay",
                daemon=True,
            )
            relay.start()
            last_activity = time.monotonic()
            while True:
                cache_changed, successful_requests = state_tracker.observe()
                cli_changed = _drain_strict_cli_progress(
                    activity,
                    cli_tracker,
                    committed_items=state_tracker.successful_items(),
                )
                now = time.monotonic()
                if cache_changed or cli_changed:
                    last_activity = now
                return_code = process.poll()
                if return_code is not None:
                    return subprocess.CompletedProcess(list(command), return_code)
                inactive_for = now - last_activity
                if inactive_for >= inactivity_timeout_seconds:
                    cache_changed, successful_requests = state_tracker.observe()
                    cli_changed = _drain_strict_cli_progress(
                        activity,
                        cli_tracker,
                        committed_items=state_tracker.successful_items(),
                    )
                    if cache_changed or cli_changed:
                        last_activity = time.monotonic()
                        continue
                    if process.poll() is not None:
                        continue
                    _assert_bound_inputs_unchanged(contract)
                    _event(
                        "post_price_network_child_inactivity_timeout",
                        action=contract.action,
                        seconds=inactivity_timeout_seconds,
                        successful_requests=successful_requests,
                    )
                    _terminate_child_only(process)
                    raise NetworkInactivityTimeout(
                        command,
                        inactivity_timeout_seconds,
                        action=contract.action,
                        successful_requests=successful_requests,
                    )
                time.sleep(
                    min(
                        poll_seconds,
                        max(0.01, inactivity_timeout_seconds - inactive_for),
                    )
                )
        except BaseException:
            if process.poll() is None:
                _terminate_child_only(process)
            raise
        finally:
            if relay is not None and relay.ident is not None:
                relay.join(timeout=NETWORK_CHILD_TERMINATION_GRACE_SECONDS)
            if process.stdout is not None:
                process.stdout.close()
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _run_network_phase(
    command: Sequence[str],
    *,
    project: Path,
    environment: Mapping[str, str],
    contract: _NetworkPhaseContract,
    inactivity_timeout_seconds: float,
    attempts: int,
    initial_backoff_seconds: float,
    maximum_backoff_seconds: float,
) -> subprocess.CompletedProcess[bytes]:
    """Retry only inactivity, preserving cache and the exact frozen command."""

    if attempts < 1:
        raise ValueError("network attempts must be positive")
    frozen_command = tuple(str(token) for token in command)
    attestation = _freeze_network_contract_attestation(contract)
    for attempt in range(1, attempts + 1):
        _assert_network_contract_attestation(contract, attestation)
        _event(
            "post_price_network_attempt_start",
            action=contract.action,
            attempt=attempt,
            maximum_attempts=attempts,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
            command=list(frozen_command),
        )
        try:
            result = _run_network_phase_once(
                frozen_command,
                project=project,
                environment=environment,
                contract=contract,
                inactivity_timeout_seconds=inactivity_timeout_seconds,
            )
        except NetworkInactivityTimeout as exc:
            if attempt == attempts:
                raise RuntimeError(
                    f"{contract.action} made no canonical progress for "
                    f"{inactivity_timeout_seconds:g} seconds after {attempts} "
                    f"attempt(s); successful_requests={exc.successful_requests}"
                ) from exc
            delay = min(
                maximum_backoff_seconds,
                initial_backoff_seconds * (2 ** (attempt - 1)),
            )
            _event(
                "post_price_network_retry_backoff",
                action=contract.action,
                attempt=attempt,
                seconds=delay,
            )
            time.sleep(delay)
            continue
        with _shared_catalog_read_lock(contract.state_dir.parent):
            _assert_network_contract_attestation(contract, attestation)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    list(frozen_command),
                )
            _validate_network_phase_complete(contract)
            _assert_network_contract_attestation(contract, attestation)
        _event(
            "post_price_network_phase_validated",
            action=contract.action,
            requests=len(contract.progress_items),
        )
        return result
    raise AssertionError("unreachable network retry state")


def _network_action(command: Sequence[str]) -> str | None:
    tokens = [str(token) for token in command]
    if len(tokens) >= 5 and tokens[1:4] == ["-m", "cogalpha_plus.cli", "data"]:
        if tokens[4] in {"adjustments", "industry"}:
            return tokens[4]
    return None


def _requires_catalog_read_lock(command: Sequence[str]) -> bool:
    tokens = [str(token) for token in command]
    if len(tokens) < 5 or tokens[1:3] != ["-m", "cogalpha_plus.cli"]:
        return False
    return tokens[3:5] in (
        ["data", "build"],
        ["data", "audit"],
        ["research", "hs300"],
    )


def _canonical_price_handoff(project: Path) -> tuple[dict[str, object] | None, str]:
    """Reuse the supervisor's complete cache/object handoff validator."""

    path = project / "scripts" / "supervise_hs300_pipeline.py"
    if not path.is_file() or path.is_symlink():
        return None, "canonical_price_supervisor_missing_or_unsafe"
    module_name = "hs300_price_handoff_validator"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None, "canonical_price_supervisor_import_failed"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        config = module.SupervisorConfig(
            project=project,
            output=project / "runs" / ".price-handoff-validation-only",
        )
        config.validate()
        return module._complete_price_manifest(config)
    except Exception as exc:
        return None, f"canonical_price_handoff_error:{type(exc).__name__}:{exc}"
    finally:
        sys.modules.pop(module_name, None)


def _price_handoff_identity(
    project: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Bind the validated manifest, indexes, union, and selected cache states."""

    root = project / "data" / "baostock_hs300"
    normalized = root / "normalized"
    raw = root / "raw"
    required_artifact_paths = (
        normalized / "price_manifest.json",
        normalized / "price_request_index.parquet",
        normalized / "security_union.parquet",
        normalized / "membership_manifest.json",
        normalized / "membership_daily.parquet",
        normalized / "membership_snapshots.parquet",
        normalized / "trade_calendar.parquet",
        normalized / "security_master_request_index.parquet",
    )
    optional_membership_index = normalized / "membership_request_index.parquet"
    artifact_paths = (
        *required_artifact_paths,
        *((optional_membership_index,) if optional_membership_index.is_file() else ()),
    )
    artifacts: list[dict[str, object]] = []
    for path in artifact_paths:
        metadata = _secure_file_metadata(path)
        if metadata is None:
            raise RuntimeError(f"price handoff artifact is unsafe or missing: {path}")
        digest, size = metadata
        artifacts.append(
            {
                "path": str(path.relative_to(project)),
                "sha256": digest,
                "bytes": size,
            }
        )
    price_index_path = normalized / "price_request_index.parquet"
    price_index_hash = next(
        item["sha256"]
        for item in artifacts
        if item["path"] == str(price_index_path.relative_to(project))
    )
    price_index = _secure_parquet(
        price_index_path,
        expected_sha256=str(price_index_hash),
    )
    security_index_path = normalized / "security_master_request_index.parquet"
    security_index_hash = next(
        item["sha256"]
        for item in artifacts
        if item["path"] == str(security_index_path.relative_to(project))
    )
    security_index = _secure_parquet(
        security_index_path,
        expected_sha256=str(security_index_hash),
    )
    membership_index: pd.DataFrame | None = None
    if optional_membership_index.is_file():
        membership_index_hash = next(
            item["sha256"]
            for item in artifacts
            if item["path"]
            == str(optional_membership_index.relative_to(project))
        )
        membership_index = _secure_parquet(
            optional_membership_index,
            expected_sha256=str(membership_index_hash),
        )
    indexed_frames = [price_index, security_index]
    if membership_index is not None:
        indexed_frames.append(membership_index)
    if any("request_key" not in frame for frame in indexed_frames):
        raise RuntimeError("price handoff indexes lack request keys")
    benchmark = manifest.get("index_request")
    benchmark_key = benchmark.get("request_key") if isinstance(benchmark, dict) else None
    keys = [
        *price_index["request_key"].astype(str).tolist(),
        *security_index["request_key"].astype(str).tolist(),
        *(
            membership_index["request_key"].astype(str).tolist()
            if membership_index is not None
            else []
        ),
        str(benchmark_key),
    ]
    if (
        any(_REQUEST_KEY_RE.fullmatch(key) is None for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise RuntimeError("price handoff request-key set is invalid or duplicated")
    requests: list[dict[str, object]] = []
    for request_key in sorted(keys):
        state_path = raw / "state" / f"{request_key}.json"
        observed = _secure_json_state_with_metadata(state_path)
        if observed is None or not isinstance(observed[0], dict):
            raise RuntimeError(f"price handoff state is unsafe: {request_key}")
        state, _signature, state_hash, state_size = observed
        if (
            state.get("request_key") != request_key
            or state.get("status") != "success"
            or not isinstance(state.get("object"), str)
            or not isinstance(state.get("parquet_sha256"), str)
        ):
            raise RuntimeError(f"price handoff state is incomplete: {request_key}")
        requests.append(
            {
                "request_key": request_key,
                "state_sha256": state_hash,
                "state_bytes": state_size,
                "endpoint": state.get("endpoint"),
                "object": state.get("object"),
                "object_sha256": state.get("parquet_sha256"),
            }
        )
    body = {
        "schema_version": 1,
        "manifest": dict(manifest),
        "artifacts": artifacts,
        "membership_request_index_present": membership_index is not None,
        "requests": requests,
    }
    aggregate = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {**body, "aggregate_sha256": aggregate}


def _freeze_price_handoff_identity(
    project: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    identity = _price_handoff_identity(project, manifest)
    target = (
        project
        / "data"
        / "baostock_hs300"
        / "normalized"
        / "price_handoff_attestation.json"
    )
    if target.exists():
        if (
            _load_regular_json(
                target,
                maximum_bytes=PRICE_HANDOFF_MAXIMUM_BYTES,
            )
            != identity
        ):
            raise RuntimeError("price handoff differs from frozen attestation")
    else:
        _atomic_json_write(target, identity)
        if (
            _load_regular_json(
                target,
                maximum_bytes=PRICE_HANDOFF_MAXIMUM_BYTES,
            )
            != identity
        ):
            raise RuntimeError("price handoff attestation publication failed")
    return identity


def _assert_price_handoff_identity(
    project: Path,
    frozen: Mapping[str, object],
) -> None:
    manifest, reason = _canonical_price_handoff(project)
    if manifest is None:
        raise RuntimeError(f"price handoff no longer canonical: {reason}")
    current = _price_handoff_identity(project, manifest)
    target = (
        project
        / "data"
        / "baostock_hs300"
        / "normalized"
        / "price_handoff_attestation.json"
    )
    if current != dict(frozen) or _load_regular_json(
        target,
        maximum_bytes=PRICE_HANDOFF_MAXIMUM_BYTES,
    ) != dict(frozen):
        raise RuntimeError("price handoff identity changed after canonical validation")


def _wait_for_prices(
    project: Path,
    manifest_path: Path,
    *,
    poll_seconds: float,
    timeout_hours: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_hours * 3600.0
    last_heartbeat = 0.0
    while time.monotonic() < deadline:
        if manifest_path.is_file():
            payload, reason = _canonical_price_handoff(project)
            if payload is None:
                raise RuntimeError(
                    "price manifest failed canonical handoff validation: "
                    f"{reason}"
                )
            _event(
                "price_manifest_complete",
                historical_union_symbols=payload["historical_union_symbols"],
                price_requests=payload["price_requests"],
            )
            return payload
        now = time.monotonic()
        if now - last_heartbeat >= 900.0:
            _event("waiting_for_price_manifest", path=str(manifest_path))
            last_heartbeat = now
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"price phase did not freeze {manifest_path} within {timeout_hours:g} hours"
    )


def _commands(project: Path, output: Path) -> list[list[str]]:
    python = sys.executable
    module = [python, "-m", "cogalpha_plus.cli"]
    return [
        module + ["check-config", "--project", str(project)],
        module
        + [
            "data",
            "membership-lineage",
            "--start",
            FROZEN_PRICE_START_DATE,
            "--end",
            FROZEN_END_DATE,
            "--membership-frequency",
            "provider_weekly",
            "--progress-every",
            "25",
        ],
        module
        + [
            "data",
            "security-lineage",
            "--start",
            FROZEN_PRICE_START_DATE,
            "--end",
            FROZEN_END_DATE,
        ],
        module
        + [
            "data",
            "adjustments",
            "--start",
            FROZEN_ADJUSTMENT_START_DATE,
            "--end",
            FROZEN_END_DATE,
            "--progress-every",
            "20",
        ],
        module
        + [
            "data",
            "industry",
            "--start",
            FROZEN_PRICE_START_DATE,
            "--end",
            FROZEN_END_DATE,
            "--progress-every",
            "20",
        ],
        module
        + [
            "data",
            "build",
            "--research-start",
            "2011-01-01",
            "--research-end",
            FROZEN_END_DATE,
        ],
        module + ["data", "audit"],
        module
        + [
            "research",
            "hs300",
            "--output",
            str(output),
        ],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/hs300_plus_full_v1"),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=36.0)
    parser.add_argument(
        "--network-inactivity-timeout-seconds",
        type=float,
        default=DEFAULT_NETWORK_INACTIVITY_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--network-attempts",
        type=int,
        default=DEFAULT_NETWORK_ATTEMPTS,
    )
    parser.add_argument(
        "--network-initial-backoff-seconds",
        type=float,
        default=DEFAULT_NETWORK_INITIAL_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--network-maximum-backoff-seconds",
        type=float,
        default=DEFAULT_NETWORK_MAXIMUM_BACKOFF_SECONDS,
    )
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 5.0 or args.timeout_hours <= 0.0:
        parser.error("poll-seconds must be >=5 and timeout-hours must be positive")
    if (
        not math.isfinite(args.network_inactivity_timeout_seconds)
        or not (
            MINIMUM_NETWORK_INACTIVITY_TIMEOUT_SECONDS
            <= args.network_inactivity_timeout_seconds
            <= MAXIMUM_NETWORK_INACTIVITY_TIMEOUT_SECONDS
        )
    ):
        parser.error(
            "network-inactivity-timeout-seconds must be between "
            f"{MINIMUM_NETWORK_INACTIVITY_TIMEOUT_SECONDS:g} and "
            f"{MAXIMUM_NETWORK_INACTIVITY_TIMEOUT_SECONDS:g}"
        )
    if args.network_attempts < 1:
        parser.error("network-attempts must be positive")
    if (
        not math.isfinite(args.network_initial_backoff_seconds)
        or not math.isfinite(args.network_maximum_backoff_seconds)
        or args.network_initial_backoff_seconds < 0
        or args.network_maximum_backoff_seconds
        < args.network_initial_backoff_seconds
    ):
        parser.error("network backoff bounds are invalid")

    project = args.project.resolve()
    output = args.output if args.output.is_absolute() else project / args.output
    output = output.resolve()
    commands = _commands(project, output)
    if args.print_plan:
        print(json.dumps(commands, ensure_ascii=False, indent=2))
        return 0

    implementation = _freeze_post_price_implementation(project)
    _event(
        "post_price_implementation_frozen",
        aggregate_sha256=implementation["aggregate_sha256"],
        file_count=implementation["file_count"],
    )

    cache_dir = project / "data" / "baostock_hs300" / "raw"
    price_manifest = (
        project
        / "data"
        / "baostock_hs300"
        / "normalized"
        / "price_manifest.json"
    )
    price_payload = _wait_for_prices(
        project,
        price_manifest,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    price_handoff = _freeze_price_handoff_identity(project, price_payload)
    _event(
        "price_handoff_identity_frozen",
        aggregate_sha256=price_handoff["aggregate_sha256"],
        request_count=len(price_handoff["requests"]),
    )
    _assert_post_price_implementation(project, implementation)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"development output must be new or empty: {output}")

    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(project / "src")
        if not existing
        else str(project / "src") + os.pathsep + existing
    )
    for number, command in enumerate(commands, 1):
        _assert_post_price_implementation(project, implementation)
        _event("phase_start", phase=number, command=command)
        action = _network_action(command)
        if action is None:
            if _requires_catalog_read_lock(command):
                with _shared_catalog_read_lock(cache_dir):
                    _assert_price_handoff_identity(project, price_handoff)
                    subprocess.run(
                        command,
                        cwd=project,
                        env=environment,
                        check=True,
                    )
                    _assert_price_handoff_identity(project, price_handoff)
            else:
                _assert_price_handoff_identity(project, price_handoff)
                subprocess.run(command, cwd=project, env=environment, check=True)
                _assert_price_handoff_identity(project, price_handoff)
        else:
            _assert_price_handoff_identity(project, price_handoff)
            contract = _network_phase_contract(project, action)
            _event(
                "post_price_network_contract_bound",
                action=action,
                endpoint=contract.endpoint,
                requests=len(contract.progress_items),
                input_sha256={
                    str(path): digest
                    for path, digest in contract.bound_inputs.items()
                },
            )
            _run_network_phase(
                command,
                project=project,
                environment=environment,
                contract=contract,
                inactivity_timeout_seconds=(
                    args.network_inactivity_timeout_seconds
                ),
                attempts=args.network_attempts,
                initial_backoff_seconds=args.network_initial_backoff_seconds,
                maximum_backoff_seconds=args.network_maximum_backoff_seconds,
            )
            _assert_price_handoff_identity(project, price_handoff)
        _assert_post_price_implementation(project, implementation)
        _event("phase_complete", phase=number)
    _assert_post_price_implementation(project, implementation)
    _assert_price_handoff_identity(project, price_handoff)
    _event(
        "development_complete_replay_still_sealed",
        reveal_manifest=str(output / "sealed_reveal_manifest.json"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
