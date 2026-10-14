#!/usr/bin/env python3
"""Supervise the serial HS300 price download and development pipeline.

The price command is deliberately restarted with identical request arguments.
BaoStock's content-addressed cache therefore serves every committed request
locally and only retries missing or failed requests.  A complete, internally
consistent price manifest is required before the existing post-price runner is
started.  The child has no total-duration deadline; a conservative watchdog
retries it only after registered cache and structured CLI progress are both
silent for the predeclared inactivity interval.

One advisory project lock covers both phases.  Historical replay reveal and
trading commands are not present in this supervisor and are rejected if they
ever appear in the delegated runner's advertised plan.

The lock coordinates instances of this supervisor.  A legacy standalone
``data prices`` process does not acquire it and must be stopped before the
supervisor is started.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from typing import BinaryIO, Iterator, Sequence

import pandas as pd


LOCK_RELATIVE_PATH = Path(".locks/hs300_pipeline_supervisor.lock")
PRICE_MANIFEST_RELATIVE_PATH = Path(
    "data/baostock_hs300/normalized/price_manifest.json"
)
PRICE_INDEX_RELATIVE_PATH = Path(
    "data/baostock_hs300/normalized/price_request_index.parquet"
)
POST_PRICE_RUNNER_RELATIVE_PATH = Path("scripts/run_hs300_after_prices.py")
FROZEN_START_DATE = "2010-01-01"
FROZEN_END_DATE = "2025-12-31"
DEFAULT_PRICE_INACTIVITY_TIMEOUT_SECONDS = 600.0
MINIMUM_PRICE_INACTIVITY_TIMEOUT_SECONDS = 30.0
MAXIMUM_PRICE_INACTIVITY_TIMEOUT_SECONDS = 86_400.0
PRICE_WATCHDOG_POLL_SECONDS = 5.0
PRICE_CHILD_TERMINATION_GRACE_SECONDS = 5.0
_REQUEST_KEY_RE = re.compile(r"[0-9a-f]{64}")


class SupervisorAlreadyRunning(RuntimeError):
    """Another process currently owns the project supervisor lock."""


class PriceInactivityTimeout(subprocess.TimeoutExpired):
    """The price child made no registered cache or CLI progress in time."""

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float,
        *,
        successful_requests: int,
    ) -> None:
        super().__init__(list(command), timeout_seconds)
        self.successful_requests = successful_requests


@dataclass(frozen=True)
class SupervisorConfig:
    project: Path
    output: Path
    start_date: str = FROZEN_START_DATE
    end_date: str = FROZEN_END_DATE
    price_attempts: int = 8
    provider_attempts: int = 5
    initial_backoff_seconds: float = 30.0
    maximum_backoff_seconds: float = 600.0
    price_inactivity_timeout_seconds: float = (
        DEFAULT_PRICE_INACTIVITY_TIMEOUT_SECONDS
    )
    progress_every: int = 10

    def validate(self) -> None:
        if self.price_attempts < 1:
            raise ValueError("price_attempts must be positive")
        if self.provider_attempts < 1:
            raise ValueError("provider_attempts must be positive")
        if not math.isfinite(self.initial_backoff_seconds):
            raise ValueError("initial_backoff_seconds must be finite")
        if not math.isfinite(self.maximum_backoff_seconds):
            raise ValueError("maximum_backoff_seconds must be finite")
        if not math.isfinite(self.price_inactivity_timeout_seconds):
            raise ValueError("price_inactivity_timeout_seconds must be finite")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds cannot be negative")
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "maximum_backoff_seconds must be at least initial_backoff_seconds"
            )
        if not (
            MINIMUM_PRICE_INACTIVITY_TIMEOUT_SECONDS
            <= self.price_inactivity_timeout_seconds
            <= MAXIMUM_PRICE_INACTIVITY_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "price_inactivity_timeout_seconds must be between "
                f"{MINIMUM_PRICE_INACTIVITY_TIMEOUT_SECONDS:g} and "
                f"{MAXIMUM_PRICE_INACTIVITY_TIMEOUT_SECONDS:g}"
            )
        if self.progress_every < 1:
            raise ValueError("progress_every must be positive")
        if self.start_date >= self.end_date:
            raise ValueError("start_date must precede end_date")
        if (self.start_date, self.end_date) != (
            FROZEN_START_DATE,
            FROZEN_END_DATE,
        ):
            raise ValueError(
                "price dates must match the frozen post-price source contract"
            )


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
def _exclusive_project_lock(project: Path) -> Iterator[Path]:
    """Hold a non-blocking OS lock; stale lock-file contents are harmless."""

    lock_path = project / LOCK_RELATIVE_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorAlreadyRunning(
                f"HS300 supervisor lock is already held: {lock_path}"
            ) from exc
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
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _validate_price_cache_contract(
    config: SupervisorConfig,
    request_index: pd.DataFrame,
    *,
    expected_union: int,
    index_request: object,
) -> str | None:
    """Reconcile every normalized row to one committed immutable cache object."""

    source_path = str(config.project / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    try:
        from cogalpha_plus.data_sources.baostock import (
            EQUITY_DAILY_FIELDS,
            INDEX_DAILY_FIELDS,
        )
        from cogalpha_plus.data_sources.hs300_build import (
            _read_committed_cache_object,
        )
        from cogalpha_plus.data_sources.hs300 import (
            trim_leading_unadjusted_vendor_history,
        )
    except Exception as exc:  # pragma: no cover - installation failure.
        return f"price_cache_contract_import_failed:{type(exc).__name__}"

    required = {
        "code",
        "adjustflag",
        "basis",
        "rows",
        "first_date",
        "last_date",
        "request_key",
    }
    missing = sorted(required.difference(request_index.columns))
    if missing:
        return "price_request_index_columns_missing:" + ",".join(missing)
    index = request_index.copy()
    index["code"] = index["code"].astype(str)
    index["adjustflag"] = index["adjustflag"].astype(str)
    if index.duplicated(["code", "adjustflag"]).any():
        return "price_request_index_duplicate_pair"
    if len(index) != 2 * expected_union:
        return "price_request_index_row_count_mismatch"
    observed_flags = index.groupby("code")["adjustflag"].agg(lambda values: set(values))
    if len(observed_flags) != expected_union or not observed_flags.map(
        lambda flags: flags == {"1", "3"}
    ).all():
        return "price_request_index_pair_coverage_invalid"
    expected_basis = {"1": "backward_adjusted", "3": "raw_unadjusted"}
    if any(
        str(row.basis) != expected_basis.get(str(row.adjustflag))
        for row in index.itertuples(index=False)
    ):
        return "price_request_index_basis_invalid"

    union_path = (
        config.project
        / "data"
        / "baostock_hs300"
        / "normalized"
        / "security_union.parquet"
    )
    try:
        union = pd.read_parquet(union_path, columns=["code"])
    except Exception as exc:
        return f"security_union_unreadable:{type(exc).__name__}"
    expected_codes = set(union["code"].astype(str))
    if len(expected_codes) != expected_union or set(index["code"]) != expected_codes:
        return "price_request_index_union_mismatch"

    cache_dir = config.project / "data" / "baostock_hs300" / "raw"
    mixed_adjusted: dict[str, pd.DataFrame] = {}
    for row in index.sort_values(["code", "adjustflag"]).itertuples(index=False):
        code = str(row.code)
        flag = str(row.adjustflag)
        request_key = str(row.request_key)
        try:
            frame, state, _ = _read_committed_cache_object(
                cache_dir,
                request_key,
                expected_endpoint="history_k_data",
                expected_fields=EQUITY_DAILY_FIELDS,
            )
        except Exception as exc:
            return f"price_cache_object_invalid:{type(exc).__name__}"
        args = state.get("args")
        if not isinstance(args, dict):
            return "price_cache_args_missing"
        if (
            str(args.get("code")) != code
            or str(args.get("adjustflag")) != flag
            or str(args.get("start_date")) != config.start_date
            or str(args.get("end_date")) != config.end_date
            or str(args.get("frequency")) != "d"
            or str(args.get("fields")) != ",".join(EQUITY_DAILY_FIELDS)
        ):
            return "price_cache_args_mismatch"
        try:
            declared_rows = int(row.rows)
        except (TypeError, ValueError):
            return "price_request_index_rows_invalid"
        if declared_rows != len(frame):
            return "price_request_index_rows_mismatch"
        if not frame["code"].astype(str).eq(code).all():
            return "price_cache_code_mismatch"
        payload_flags = set(frame["adjustflag"].astype(str))
        if flag == "3" and payload_flags != {"3"}:
            return "price_cache_adjustflag_mismatch"
        if flag == "1":
            if not payload_flags or not payload_flags.issubset({"1", "3"}):
                return "price_cache_adjustflag_mismatch"
            if payload_flags != {"1"}:
                mixed_adjusted[code] = frame
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any():
            return "price_cache_dates_invalid"
        first_date = str(pd.Timestamp(dates.min()).date())
        last_date = str(pd.Timestamp(dates.max()).date())
        if str(row.first_date) != first_date or str(row.last_date) != last_date:
            return "price_request_index_date_range_mismatch"

    if mixed_adjusted:
        membership_path = (
            config.project
            / "data"
            / "baostock_hs300"
            / "normalized"
            / "membership_daily.parquet"
        )
        try:
            membership = pd.read_parquet(membership_path, columns=["date", "code"])
            lookup = index.set_index(["code", "adjustflag"], verify_integrity=True)

            def panel(frame: pd.DataFrame) -> pd.DataFrame:
                selected = frame.loc[
                    :, ["date", "code", "open", "high", "low", "close", "preclose", "adjustflag"]
                ].copy()
                selected["date"] = pd.to_datetime(
                    selected["date"], errors="coerce"
                ).dt.normalize()
                selected.rename(columns={"code": "symbol"}, inplace=True)
                return selected.set_index(["date", "symbol"]).sort_index()

            for code, adjusted_frame in sorted(mixed_adjusted.items()):
                raw_key = str(lookup.loc[(code, "3"), "request_key"])
                raw_frame, _, _ = _read_committed_cache_object(
                    cache_dir,
                    raw_key,
                    expected_endpoint="history_k_data",
                    expected_fields=EQUITY_DAILY_FIELDS,
                )
                trim_leading_unadjusted_vendor_history(
                    panel(raw_frame),
                    panel(adjusted_frame),
                    membership.loc[membership["code"].astype(str).eq(code)],
                )
        except Exception as exc:
            return f"price_cache_adjustflag_boundary_invalid:{type(exc).__name__}"

    if not isinstance(index_request, dict):
        return "benchmark_index_request_missing"
    benchmark_key = str(index_request.get("request_key", ""))
    if not benchmark_key:
        return "benchmark_index_request_key_missing"
    try:
        benchmark, benchmark_state, _ = _read_committed_cache_object(
            cache_dir,
            benchmark_key,
            expected_endpoint="index_history",
            expected_fields=INDEX_DAILY_FIELDS,
        )
    except Exception as exc:
        return f"benchmark_cache_object_invalid:{type(exc).__name__}"
    expected_benchmark_args = {
        "code": "sh.000300",
        "fields": ",".join(INDEX_DAILY_FIELDS),
        "start_date": config.start_date,
        "end_date": config.end_date,
        "frequency": "d",
        "adjustflag": "3",
    }
    if benchmark_state.get("args") != expected_benchmark_args:
        return "benchmark_cache_args_mismatch"
    if len(benchmark) < 1 or not benchmark["code"].astype(str).eq("sh.000300").all():
        return "benchmark_cache_payload_invalid"
    benchmark_dates = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark_close = pd.to_numeric(benchmark["close"], errors="coerce")
    if (
        benchmark_dates.isna().any()
        or benchmark_dates.duplicated().any()
        or benchmark_close.isna().any()
        or (benchmark_close <= 0).any()
    ):
        return "benchmark_cache_values_invalid"
    return None


def _complete_price_manifest(
    config: SupervisorConfig,
) -> tuple[dict[str, object] | None, str]:
    """Validate the frozen manifest and its bound request-index bytes."""

    manifest_path = config.project / PRICE_MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        return None, "price_manifest_missing"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"price_manifest_unreadable:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "price_manifest_not_an_object"
    if payload.get("complete") is not True:
        return None, "price_manifest_complete_flag_false"
    if payload.get("start_date") != config.start_date:
        return None, "price_manifest_start_date_mismatch"
    if payload.get("end_date") != config.end_date:
        return None, "price_manifest_end_date_mismatch"

    union = _strict_positive_int(payload.get("historical_union_symbols"))
    downloaded = _strict_positive_int(payload.get("downloaded_symbols"))
    request_count = _strict_positive_int(payload.get("price_requests"))
    if union is None or downloaded != union or request_count != 2 * union:
        return None, "price_manifest_catalog_counts_invalid"

    files = payload.get("files")
    metadata = files.get("price_request_index") if isinstance(files, dict) else None
    if not isinstance(metadata, dict):
        return None, "price_request_index_metadata_missing"
    index_path = config.project / PRICE_INDEX_RELATIVE_PATH
    declared_path = metadata.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        return None, "price_request_index_path_missing"
    declared = Path(declared_path)
    declared = (
        declared.resolve()
        if declared.is_absolute()
        else (config.project / declared).resolve()
    )
    if declared != index_path.resolve():
        return None, "price_request_index_path_mismatch"
    if not index_path.is_file():
        return None, "price_request_index_missing"
    declared_bytes = _strict_positive_int(metadata.get("bytes"))
    if declared_bytes != index_path.stat().st_size:
        return None, "price_request_index_size_mismatch"
    declared_hash = metadata.get("sha256")
    if not isinstance(declared_hash, str) or _sha256(index_path) != declared_hash:
        return None, "price_request_index_hash_mismatch"
    if payload.get("phase") != "price_catalog":
        return None, "price_manifest_phase_invalid"
    try:
        request_index = pd.read_parquet(index_path)
    except Exception as exc:
        return None, f"price_request_index_unreadable:{type(exc).__name__}"
    cache_reason = _validate_price_cache_contract(
        config,
        request_index,
        expected_union=union,
        index_request=payload.get("index_request"),
    )
    if cache_reason is not None:
        return None, cache_reason
    return payload, "complete"


def _python_environment(project: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source = str(project / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    return environment


def _price_command(config: SupervisorConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "cogalpha_plus.cli",
        "data",
        "prices",
        "--project",
        str(config.project),
        "--start",
        config.start_date,
        "--end",
        config.end_date,
        "--max-attempts",
        str(config.provider_attempts),
        "--progress-every",
        str(config.progress_every),
    ]


def _post_price_command(config: SupervisorConfig) -> list[str]:
    return [
        sys.executable,
        str(config.project / POST_PRICE_RUNNER_RELATIVE_PATH),
        "--project",
        str(config.project),
        "--output",
        str(config.output),
        "--poll-seconds",
        "5",
        "--timeout-hours",
        "1",
    ]


def _load_post_price_plan(config: SupervisorConfig) -> list[list[str]]:
    path = config.project / POST_PRICE_RUNNER_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"post-price runner is missing: {path}")
    spec = importlib.util.spec_from_file_location("hs300_post_price_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load post-price runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = module._commands(config.project, config.output)
    if not isinstance(commands, list):
        raise TypeError("post-price runner plan must be a list")
    return commands


def _expected_post_price_arguments(config: SupervisorConfig) -> list[list[str]]:
    return [
        ["check-config", "--project", str(config.project)],
        [
            "data",
            "membership-lineage",
            "--start",
            FROZEN_START_DATE,
            "--end",
            FROZEN_END_DATE,
            "--membership-frequency",
            "provider_weekly",
            "--progress-every",
            "25",
        ],
        [
            "data",
            "security-lineage",
            "--start",
            FROZEN_START_DATE,
            "--end",
            FROZEN_END_DATE,
        ],
        [
            "data",
            "adjustments",
            "--start",
            "1990-01-01",
            "--end",
            FROZEN_END_DATE,
            "--progress-every",
            "20",
        ],
        [
            "data",
            "industry",
            "--start",
            FROZEN_START_DATE,
            "--end",
            FROZEN_END_DATE,
            "--progress-every",
            "20",
        ],
        [
            "data",
            "build",
            "--research-start",
            "2011-01-01",
            "--research-end",
            FROZEN_END_DATE,
        ],
        ["data", "audit"],
        ["research", "hs300", "--output", str(config.output)],
    ]


def _assert_safe_post_price_plan(
    commands: Sequence[Sequence[str]],
    *,
    expected_arguments: Sequence[Sequence[str]] | None = None,
) -> None:
    """Allow only the predeclared data/development phases."""

    expected_actions = [
        ("check-config",),
        ("data", "membership-lineage"),
        ("data", "security-lineage"),
        ("data", "adjustments"),
        ("data", "industry"),
        ("data", "build"),
        ("data", "audit"),
        ("research", "hs300"),
    ]
    actions: list[tuple[str, ...]] = []
    all_arguments: list[list[str]] = []
    for command in commands:
        tokens = [str(token) for token in command]
        if (
            len(tokens) < 4
            or Path(tokens[0]).resolve() != Path(sys.executable).resolve()
            or tokens[1:3] != ["-m", "cogalpha_plus.cli"]
        ):
            raise RuntimeError("post-price plan contains a non-CogAlpha command")
        forbidden = {
            "hs300-reveal",
            "reveal",
            "trade",
            "trading",
            "live-trading",
        }
        if any(
            token.lower() in forbidden or token.lower().startswith("--reveal")
            for token in tokens[3:]
        ):
            raise RuntimeError("post-price plan contains a reveal or trading token")
        arguments = tokens[3:]
        all_arguments.append(arguments)
        if arguments[:1] == ["check-config"]:
            actions.append(("check-config",))
        elif arguments[:1] == ["data"] and len(arguments) >= 2:
            actions.append(("data", arguments[1]))
        elif arguments[:1] == ["research"] and len(arguments) >= 2:
            actions.append(("research", arguments[1]))
        else:
            raise RuntimeError(f"unrecognized post-price command: {tokens}")
    if actions != expected_actions:
        raise RuntimeError(
            "post-price plan differs from the data/audit/development allowlist: "
            f"{actions}"
        )
    if expected_arguments is not None:
        expected = [[str(token) for token in command] for command in expected_arguments]
        if all_arguments != expected:
            raise RuntimeError(
                "post-price command arguments differ from the frozen supervisor contract"
            )


def _is_registered_price_state(
    payload: object,
    *,
    request_key: str,
    start_date: str,
    end_date: str,
    expected_codes: frozenset[str],
    equity_fields: tuple[str, ...],
    index_fields: tuple[str, ...],
    stock_basic_fields: tuple[str, ...],
) -> bool:
    """Recognize only cache states reachable from the frozen price command."""

    if (
        not isinstance(payload, dict)
        or type(payload.get("state_schema_version")) is not int
        or payload.get("state_schema_version") != 1
        or payload.get("request_key") != request_key
        or payload.get("provider") != "baostock"
        or payload.get("status") not in {"running", "success", "failed"}
        or type(payload.get("attempt_count")) is not int
        or payload.get("attempt_count", 0) <= 0
    ):
        return False
    endpoint = payload.get("endpoint")
    args = payload.get("args")
    provider_version = payload.get("provider_version")
    if (
        not isinstance(args, dict)
        or not isinstance(provider_version, str)
        or not provider_version.strip()
        or provider_version != provider_version.strip()
    ):
        return False
    if endpoint == "stock_basic":
        code = args.get("code")
        valid_args = (
            set(args) == {"code", "code_name"}
            and args.get("code_name") == ""
            and (code == "" or code in expected_codes)
        )
        expected_fields = stock_basic_fields
    elif endpoint == "index_history":
        valid_args = args == {
            "code": "sh.000300",
            "fields": ",".join(index_fields),
            "start_date": start_date,
            "end_date": end_date,
            "frequency": "d",
            "adjustflag": "3",
        }
        expected_fields = index_fields
    elif endpoint == "history_k_data":
        valid_args = args == {
            "code": args.get("code"),
            "fields": ",".join(equity_fields),
            "start_date": start_date,
            "end_date": end_date,
            "frequency": "d",
            "adjustflag": args.get("adjustflag"),
        } and (
            args.get("code") in expected_codes
            and args.get("adjustflag") in {"1", "3"}
        )
        expected_fields = equity_fields
    else:
        return False
    if not valid_args:
        return False
    canonical = {
        "provider": "baostock",
        "provider_version": provider_version.strip(),
        "endpoint": endpoint,
        "args": args,
    }
    expected_key = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return expected_key == request_key and (
        payload.get("status") != "success"
        or payload.get("fields") == list(expected_fields)
    )


class _PriceStateProgressTracker:
    """Track atomic state transitions without rereading unchanged JSON files."""

    def __init__(
        self,
        state_dir: Path,
        *,
        start_date: str,
        end_date: str,
        expected_codes: frozenset[str],
        equity_fields: tuple[str, ...],
        index_fields: tuple[str, ...],
        stock_basic_fields: tuple[str, ...],
    ) -> None:
        self.state_dir = state_dir
        self.start_date = start_date
        self.end_date = end_date
        self.expected_codes = expected_codes
        self.equity_fields = equity_fields
        self.index_fields = index_fields
        self.stock_basic_fields = stock_basic_fields
        self._known: dict[
            Path,
            tuple[tuple[int, int, int, int], bool, str, int],
        ] = {}
        self._initialized = False

    def observe(self) -> tuple[bool, int]:
        current_paths: set[Path] = set()
        registered_change = False
        if self.state_dir.is_dir():
            with os.scandir(self.state_dir) as entries:
                for entry in entries:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    path = Path(entry.path)
                    if path.suffix != ".json" or not _REQUEST_KEY_RE.fullmatch(
                        path.stem
                    ):
                        continue
                    current_paths.add(path)
                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    signature = (
                        stat.st_ino,
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                    )
                    previous = self._known.get(path)
                    if previous is not None and previous[0] == signature:
                        continue
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        # State publication is atomic. A transient read race is
                        # ignored and retried on the next observation.
                        continue
                    registered = _is_registered_price_state(
                        payload,
                        request_key=path.stem,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        expected_codes=self.expected_codes,
                        equity_fields=self.equity_fields,
                        index_fields=self.index_fields,
                        stock_basic_fields=self.stock_basic_fields,
                    )
                    status = str(payload.get("status", ""))
                    attempt_count = int(payload.get("attempt_count", -1))
                    self._known[path] = (
                        signature,
                        registered,
                        status,
                        attempt_count,
                    )
                    if (
                        self._initialized
                        and registered
                        and (
                            previous is None
                            or not previous[1]
                            or previous[2:] != (status, attempt_count)
                        )
                    ):
                        registered_change = True

        removed = set(self._known).difference(current_paths)
        for path in removed:
            self._known.pop(path)

        successful = sum(
            registered and status == "success"
            for _signature, registered, status, _attempt_count in self._known.values()
        )
        self._initialized = True
        return registered_change, successful


def _structured_price_progress(
    raw_line: bytes,
) -> tuple[int, int, str] | None:
    try:
        payload = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    done = payload.get("progress")
    total = payload.get("total")
    item = payload.get("item")
    valid = (
        isinstance(done, int)
        and not isinstance(done, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and 1 <= done <= total
        and isinstance(item, str)
        and bool(item)
    )
    return (done, total, item) if valid else None


def _relay_child_output(
    stream: BinaryIO,
    activity: "queue.SimpleQueue[tuple[int, int, str]]",
) -> None:
    """Preserve streamed logs while recognizing the CLI's JSON progress lines."""

    for raw_line in iter(stream.readline, b""):
        progress = _structured_price_progress(raw_line)
        if progress is not None:
            activity.put(progress)
        try:
            target = getattr(sys.stdout, "buffer", None)
            if target is not None:
                target.write(raw_line)
                target.flush()
            else:  # pytest and other text-only stdout wrappers.
                sys.stdout.write(raw_line.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        except (OSError, ValueError):
            # Log forwarding is best effort. Keep draining the pipe so a
            # healthy child cannot block or lose watchdog progress.
            continue


def _drain_progress(
    activity: "queue.SimpleQueue[tuple[int, int, str]]",
    previous: tuple[int, int, str] | None,
) -> tuple[bool, tuple[int, int, str] | None]:
    advanced = False
    while True:
        try:
            current = activity.get_nowait()
        except queue.Empty:
            return advanced, previous
        if previous is None or (
            current[1] == previous[1] and current[0] > previous[0]
        ):
            previous = current
            advanced = True


def _terminate_child_only(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = PRICE_CHILD_TERMINATION_GRACE_SECONDS,
) -> None:
    """Stop only the supervised child, escalating after a short grace period."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "price child did not exit after SIGKILL"
            ) from exc


def _price_progress_contract(
    project: Path,
) -> tuple[
    frozenset[str],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    source_path = str(project / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from cogalpha_plus.data_sources.baostock import (
        EQUITY_DAILY_FIELDS,
        INDEX_DAILY_FIELDS,
        STOCK_BASIC_FIELDS,
    )

    union_path = (
        project
        / "data"
        / "baostock_hs300"
        / "normalized"
        / "security_union.parquet"
    )
    union = pd.read_parquet(union_path, columns=["code"])
    codes = frozenset(union["code"].astype(str))
    if not codes or len(codes) != len(union):
        raise ValueError("price watchdog requires a nonempty unique security union")
    return (
        codes,
        tuple(EQUITY_DAILY_FIELDS),
        tuple(INDEX_DAILY_FIELDS),
        tuple(STOCK_BASIC_FIELDS),
    )


def _raise_controlled_signal_exit(signum: int, _frame: object) -> None:
    """Turn termination into stack unwinding so the price child is reaped."""

    raise SystemExit(128 + signum)


def _run_with_price_inactivity_watchdog(
    command: Sequence[str],
    *,
    project: Path,
    environment: dict[str, str],
    inactivity_timeout_seconds: float,
    state_dir: Path,
    start_date: str,
    end_date: str,
    poll_seconds: float = PRICE_WATCHDOG_POLL_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    """Run a price child while progress, not total duration, resets its clock."""

    if not math.isfinite(inactivity_timeout_seconds) or inactivity_timeout_seconds <= 0:
        raise ValueError("inactivity_timeout_seconds must be finite and positive")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be finite and positive")

    expected_codes, equity_fields, index_fields, stock_basic_fields = (
        _price_progress_contract(project)
    )
    tracker = _PriceStateProgressTracker(
        state_dir,
        start_date=start_date,
        end_date=end_date,
        expected_codes=expected_codes,
        equity_fields=equity_fields,
        index_fields=index_fields,
        stock_basic_fields=stock_basic_fields,
    )
    _changed, successful_requests = tracker.observe()
    activity: queue.SimpleQueue[tuple[int, int, str]] = queue.SimpleQueue()
    handled_signals = (signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in handled_signals
    }
    for signum in handled_signals:
        signal.signal(signum, _raise_controlled_signal_exit)
    try:
        process = subprocess.Popen(
            list(command),
            cwd=project,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        relay: threading.Thread | None = None
        try:
            if process.stdout is None:  # pragma: no cover - Popen contract defense.
                raise RuntimeError("price child stdout pipe was not created")
            relay = threading.Thread(
                target=_relay_child_output,
                args=(process.stdout, activity),
                name="hs300-price-output-relay",
                daemon=True,
            )
            relay.start()
            last_activity = time.monotonic()
            last_cli_progress: tuple[int, int, str] | None = None
            while True:
                changed, successful_requests = tracker.observe()
                cli_progress, last_cli_progress = _drain_progress(
                    activity,
                    last_cli_progress,
                )
                now = time.monotonic()
                if changed or cli_progress:
                    last_activity = now

                return_code = process.poll()
                if return_code is not None:
                    return subprocess.CompletedProcess(list(command), return_code)

                inactive_for = now - last_activity
                if inactive_for >= inactivity_timeout_seconds:
                    # Close the narrow race between the first observation and
                    # termination: honor one final registered-progress check.
                    changed, successful_requests = tracker.observe()
                    cli_progress, last_cli_progress = _drain_progress(
                        activity,
                        last_cli_progress,
                    )
                    changed = cli_progress or changed
                    if changed:
                        last_activity = time.monotonic()
                        continue
                    if process.poll() is not None:
                        continue
                    _event(
                        "price_child_inactivity_timeout",
                        seconds=inactivity_timeout_seconds,
                        successful_requests=successful_requests,
                    )
                    _terminate_child_only(process)
                    raise PriceInactivityTimeout(
                        command,
                        inactivity_timeout_seconds,
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
                relay.join(timeout=PRICE_CHILD_TERMINATION_GRACE_SECONDS)
            if process.stdout is not None:
                process.stdout.close()
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _run_streaming(
    command: Sequence[str],
    *,
    project: Path,
    environment: dict[str, str],
    timeout_seconds: float | None = None,
    inactivity_timeout_seconds: float | None = None,
    progress_state_dir: Path | None = None,
    progress_start_date: str = FROZEN_START_DATE,
    progress_end_date: str = FROZEN_END_DATE,
) -> subprocess.CompletedProcess[bytes]:
    """Inherit stdout and merge child stderr into it for redirectable logs."""

    if inactivity_timeout_seconds is not None:
        if timeout_seconds is not None:
            raise ValueError("total and inactivity timeouts cannot be combined")
        if progress_state_dir is None:
            raise ValueError("progress_state_dir is required for inactivity timeout")
        return _run_with_price_inactivity_watchdog(
            command,
            project=project,
            environment=environment,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
            state_dir=progress_state_dir,
            start_date=progress_start_date,
            end_date=progress_end_date,
        )

    return subprocess.run(
        list(command),
        cwd=project,
        env=environment,
        check=False,
        stdout=None,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
    )


def _run_price_phase(
    config: SupervisorConfig,
    *,
    environment: dict[str, str],
) -> dict[str, object]:
    existing, reason = _complete_price_manifest(config)
    if existing is not None:
        _event(
            "price_phase_already_complete",
            historical_union_symbols=existing["historical_union_symbols"],
            price_requests=existing["price_requests"],
        )
        return existing

    command = _price_command(config)
    _event("price_phase_resume_required", reason=reason)
    last_reason = reason
    for attempt in range(1, config.price_attempts + 1):
        _event(
            "price_attempt_start",
            attempt=attempt,
            maximum_attempts=config.price_attempts,
            inactivity_timeout_seconds=(
                config.price_inactivity_timeout_seconds
            ),
            command=command,
        )
        process_reason: str | None = None
        try:
            result = _run_streaming(
                command,
                project=config.project,
                environment=environment,
                timeout_seconds=None,
                inactivity_timeout_seconds=(
                    config.price_inactivity_timeout_seconds
                ),
                progress_state_dir=(
                    config.project / "data" / "baostock_hs300" / "raw" / "state"
                ),
                progress_start_date=config.start_date,
                progress_end_date=config.end_date,
            )
            return_code: int | None = result.returncode
        except PriceInactivityTimeout as exc:
            return_code = None
            process_reason = (
                "price_process_inactivity_timeout:"
                f"{config.price_inactivity_timeout_seconds:g}_seconds:"
                f"successful_requests={exc.successful_requests}"
            )
        except subprocess.TimeoutExpired:
            return_code = None
            process_reason = "price_process_unexpected_total_timeout"
        except OSError as exc:
            return_code = None
            process_reason = f"price_process_error:{type(exc).__name__}:{exc}"
        manifest, manifest_reason = _complete_price_manifest(config)
        if manifest is not None:
            _event(
                "price_phase_complete",
                attempt=attempt,
                return_code=return_code,
                historical_union_symbols=manifest["historical_union_symbols"],
                price_requests=manifest["price_requests"],
            )
            return manifest
        if process_reason is not None:
            last_reason = f"{process_reason}:{manifest_reason}"
        else:
            last_reason = (
                f"price_exit_{return_code}:{manifest_reason}"
                if return_code is not None
                else manifest_reason
            )
        _event(
            "price_attempt_failed",
            attempt=attempt,
            return_code=return_code,
            reason=last_reason,
        )
        if attempt == config.price_attempts:
            break
        delay = min(
            config.maximum_backoff_seconds,
            config.initial_backoff_seconds * (2 ** (attempt - 1)),
        )
        _event("price_retry_backoff", attempt=attempt, seconds=delay)
        time.sleep(delay)
    raise RuntimeError(
        f"price phase did not produce a complete manifest after "
        f"{config.price_attempts} attempt(s): {last_reason}"
    )


def supervise(config: SupervisorConfig) -> int:
    config.validate()
    environment = _python_environment(config.project)
    with _exclusive_project_lock(config.project) as lock_path:
        _event("supervisor_lock_acquired", lock=str(lock_path), pid=os.getpid())
        _run_price_phase(config, environment=environment)
        runner_path = config.project / POST_PRICE_RUNNER_RELATIVE_PATH
        runner_hash = _sha256(runner_path)
        plan = _load_post_price_plan(config)
        if _sha256(runner_path) != runner_hash:
            raise RuntimeError("post-price runner changed while loading its plan")
        _assert_safe_post_price_plan(
            plan,
            expected_arguments=_expected_post_price_arguments(config),
        )
        _event("post_price_plan_validated", phases=len(plan))
        final_manifest, final_reason = _complete_price_manifest(config)
        if final_manifest is None:
            raise RuntimeError(
                "price catalog changed after completion and before development: "
                f"{final_reason}"
            )
        if _sha256(runner_path) != runner_hash:
            raise RuntimeError("post-price runner changed before invocation")
        command = _post_price_command(config)
        _event("post_price_runner_start", command=command)
        result = _run_streaming(
            command,
            project=config.project,
            environment=environment,
            timeout_seconds=None,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)
        _event("supervisor_complete_replay_still_sealed", output=str(config.output))
    return 0


def _reasonable_inactivity_timeout(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if not math.isfinite(seconds) or not (
        MINIMUM_PRICE_INACTIVITY_TIMEOUT_SECONDS
        <= seconds
        <= MAXIMUM_PRICE_INACTIVITY_TIMEOUT_SECONDS
    ):
        raise argparse.ArgumentTypeError(
            "timeout must be between "
            f"{MINIMUM_PRICE_INACTIVITY_TIMEOUT_SECONDS:g} and "
            f"{MAXIMUM_PRICE_INACTIVITY_TIMEOUT_SECONDS:g} seconds"
        )
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume HS300 prices with bounded retries, then run development"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, default=Path("runs/hs300_plus_full_v1"))
    parser.add_argument("--start", default=FROZEN_START_DATE)
    parser.add_argument("--end", default=FROZEN_END_DATE)
    parser.add_argument("--price-attempts", type=int, default=8)
    parser.add_argument("--provider-attempts", type=int, default=5)
    parser.add_argument("--initial-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-backoff-seconds", type=float, default=600.0)
    parser.add_argument(
        "--price-inactivity-timeout-seconds",
        dest="price_inactivity_timeout_seconds",
        type=_reasonable_inactivity_timeout,
        default=DEFAULT_PRICE_INACTIVITY_TIMEOUT_SECONDS,
        help=(
            "terminate and retry only after this many seconds without a "
            "registered cache-state transition or structured CLI progress"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main() -> int:
    args = _parser().parse_args()
    project = args.project.resolve()
    output = args.output if args.output.is_absolute() else project / args.output
    config = SupervisorConfig(
        project=project,
        output=output.resolve(),
        start_date=args.start,
        end_date=args.end,
        price_attempts=args.price_attempts,
        provider_attempts=args.provider_attempts,
        initial_backoff_seconds=args.initial_backoff_seconds,
        maximum_backoff_seconds=args.maximum_backoff_seconds,
        price_inactivity_timeout_seconds=(
            args.price_inactivity_timeout_seconds
        ),
        progress_every=args.progress_every,
    )
    try:
        return supervise(config)
    except SupervisorAlreadyRunning as exc:
        _event("supervisor_refused_lock_held", reason=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
