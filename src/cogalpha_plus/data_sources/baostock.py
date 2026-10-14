"""Auditable, resumable BaoStock request and raw-cache layer.

Only transport concerns live here.  Values are intentionally kept as the
strings returned by BaoStock; economic type conversion, adjustments and
point-in-time joins belong to the downstream normalizer.

The cache key covers the endpoint, normalized arguments and BaoStock client
version.  A Parquet object becomes visible only after the complete response has
passed its schema check.  A separate atomic JSON state file is written last, so
an interrupted request cannot masquerade as a successful cache entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd


TRADE_DATE_FIELDS = ("calendar_date", "is_trading_day")
HS300_FIELDS = ("updateDate", "code", "code_name")
STOCK_BASIC_FIELDS = ("code", "code_name", "ipoDate", "outDate", "type", "status")
INDUSTRY_FIELDS = ("updateDate", "code", "code_name", "industry", "industryClassification")
ADJUST_FACTOR_FIELDS = (
    "code",
    "dividOperateDate",
    "foreAdjustFactor",
    "backAdjustFactor",
    "adjustFactor",
)
EQUITY_DAILY_FIELDS = (
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
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "isST",
)
INDEX_DAILY_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "pctChg",
)

_ENDPOINT_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class BaoStockError(RuntimeError):
    """Base exception for the source adapter."""


class BaoStockResponseError(BaoStockError):
    """BaoStock returned a non-success response or an invalid result object."""


class SchemaDriftError(BaoStockError):
    """A response schema differs from the pre-registered endpoint schema."""


class CacheIntegrityError(BaoStockError):
    """A committed cache entry is missing, altered or internally inconsistent."""


@runtime_checkable
class BaoStockResultProtocol(Protocol):
    error_code: str
    error_msg: str
    fields: Sequence[str]

    def next(self) -> bool: ...

    def get_row_data(self) -> Sequence[object]: ...


@runtime_checkable
class BaoStockClientProtocol(Protocol):
    """The small part of the BaoStock module used by this adapter."""

    __version__: str

    def login(self) -> object: ...

    def logout(self) -> object: ...

    def query_trade_dates(self, start_date: str, end_date: str) -> BaoStockResultProtocol: ...

    def query_hs300_stocks(self, date: str = "") -> BaoStockResultProtocol: ...

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
    ) -> BaoStockResultProtocol: ...

    def query_stock_basic(self, code: str = "", code_name: str = "") -> BaoStockResultProtocol: ...

    def query_stock_industry(self, code: str = "", date: str = "") -> BaoStockResultProtocol: ...

    def query_adjust_factor(
        self,
        code: str,
        start_date: str,
        end_date: str,
    ) -> BaoStockResultProtocol: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_json_value(value: object) -> object:
    """Return a deterministic, JSON-safe representation or fail visibly."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("request arguments cannot contain NaN or infinity")
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        timestamp = pd.Timestamp(value)
        if isinstance(value, date) and not isinstance(value, datetime):
            return str(timestamp.date())
        return timestamp.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_normalized_json_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalized_json_value(item) for item in value]
    raise TypeError(f"request argument type is not deterministic JSON: {type(value).__name__}")


def normalized_request_payload(
    endpoint: str,
    args: Mapping[str, object],
    *,
    provider_version: str,
) -> dict[str, object]:
    if not _ENDPOINT_RE.fullmatch(endpoint):
        raise ValueError(f"invalid endpoint name: {endpoint!r}")
    version = str(provider_version).strip()
    if not version:
        raise ValueError("a non-empty BaoStock provider version is required")
    normalized_args = _normalized_json_value(args)
    if not isinstance(normalized_args, dict):  # pragma: no cover - defensive.
        raise TypeError("request args must normalize to an object")
    return {
        "provider": "baostock",
        "provider_version": version,
        "endpoint": endpoint,
        "args": normalized_args,
    }


def make_request_key(
    endpoint: str,
    args: Mapping[str, object],
    *,
    provider_version: str,
) -> str:
    payload = normalized_request_payload(endpoint, args, provider_version=provider_version)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _response_error(result: object) -> tuple[str, str]:
    if result is None:
        return "missing_result", "BaoStock returned None"
    code = str(getattr(result, "error_code", "missing_error_code"))
    message = str(getattr(result, "error_msg", ""))
    return code, message


def require_success_response(result: object, *, operation: str) -> None:
    code, message = _response_error(result)
    if code != "0":
        raise BaoStockResponseError(f"{operation} failed with error_code={code}: {message}")


def result_to_frame(
    result: BaoStockResultProtocol,
    *,
    expected_fields: Sequence[str],
    operation: str,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Exhaust a BaoStock result set and validate its exact field order."""

    require_success_response(result, operation=operation)
    actual_fields = tuple(str(name) for name in getattr(result, "fields", ()))
    expected = tuple(str(name) for name in expected_fields)
    if actual_fields != expected:
        raise SchemaDriftError(
            f"{operation} schema drift: expected {list(expected)}, received {list(actual_fields)}"
        )
    rows: list[list[object]] = []
    while True:
        try:
            has_next = bool(result.next())
        except Exception as exc:
            raise BaoStockResponseError(f"{operation} failed while paging: {type(exc).__name__}: {exc}") from exc
        if not has_next:
            break
        require_success_response(result, operation=operation)
        row = list(result.get_row_data())
        if len(row) != len(expected):
            raise SchemaDriftError(
                f"{operation} row width drift: expected {len(expected)}, received {len(row)}"
            )
        rows.append(row)
    require_success_response(result, operation=operation)
    if not rows and not allow_empty:
        raise BaoStockResponseError(f"{operation} returned an empty successful response")
    # object dtype preserves the provider's original decimal/date strings.
    return pd.DataFrame(rows, columns=list(expected), dtype=object)


@dataclass(frozen=True)
class BaoStockConfig:
    cache_dir: Path | str
    provider_version: str | None = None
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    maximum_backoff_seconds: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.provider_version is not None and not str(self.provider_version).strip():
            raise ValueError("provider_version cannot be blank")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.initial_backoff_seconds < 0 or self.maximum_backoff_seconds < 0:
            raise ValueError("backoff durations cannot be negative")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least one")


@dataclass(frozen=True)
class BaoStockRequest:
    endpoint: str
    args: Mapping[str, object] = field(default_factory=dict)
    allow_empty: bool = False

    def __post_init__(self) -> None:
        if not _ENDPOINT_RE.fullmatch(self.endpoint):
            raise ValueError(f"invalid endpoint name: {self.endpoint!r}")
        object.__setattr__(self, "args", dict(self.args))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RawRequestCache:
    """Content-addressed Parquet response cache with atomic state commits."""

    STATE_SCHEMA_VERSION = 1

    def __init__(self, root: Path | str, *, provider_version: str) -> None:
        self.root = Path(root)
        self.provider_version = str(provider_version).strip()
        if not self.provider_version:
            raise ValueError("provider_version is required for a raw request cache")
        self.objects_dir = self.root / "objects"
        self.states_dir = self.root / "state"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)

    def key(self, endpoint: str, args: Mapping[str, object]) -> str:
        return make_request_key(endpoint, args, provider_version=self.provider_version)

    def object_path(self, endpoint: str, request_key: str) -> Path:
        if not _ENDPOINT_RE.fullmatch(endpoint) or not re.fullmatch(r"[0-9a-f]{64}", request_key):
            raise ValueError("unsafe cache path components")
        return self.objects_dir / endpoint / f"{request_key}.parquet"

    def state_path(self, request_key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", request_key):
            raise ValueError("invalid request key")
        return self.states_dir / f"{request_key}.json"

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def read_state(self, request_key: str) -> dict[str, object] | None:
        path = self.state_path(request_key)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(f"cannot read state for {request_key}: {exc}") from exc
        if payload.get("request_key") != request_key:
            raise CacheIntegrityError(f"state request key mismatch for {request_key}")
        return payload

    def _base_state(self, endpoint: str, args: Mapping[str, object], request_key: str) -> dict[str, object]:
        previous = self.read_state(request_key)
        attempts = int(previous.get("attempt_count", 0)) + 1 if previous else 1
        payload = normalized_request_payload(endpoint, args, provider_version=self.provider_version)
        return {
            "state_schema_version": self.STATE_SCHEMA_VERSION,
            "request_key": request_key,
            **payload,
            "attempt_count": attempts,
        }

    def _load_success(
        self,
        endpoint: str,
        request_key: str,
        expected_fields: Sequence[str],
    ) -> pd.DataFrame | None:
        state = self.read_state(request_key)
        if state is None or state.get("status") != "success":
            return None
        path = self.object_path(endpoint, request_key)
        if not path.is_file():
            raise CacheIntegrityError(f"successful state has no Parquet object: {request_key}")
        actual_hash = _sha256_file(path)
        if actual_hash != state.get("parquet_sha256"):
            raise CacheIntegrityError(f"Parquet hash mismatch for {request_key}")
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise CacheIntegrityError(f"cannot read Parquet object {request_key}: {exc}") from exc
        expected = tuple(str(name) for name in expected_fields)
        if tuple(frame.columns.astype(str)) != expected:
            raise CacheIntegrityError(f"cached schema mismatch for {request_key}")
        if len(frame) != int(state.get("rows", -1)):
            raise CacheIntegrityError(f"cached row-count mismatch for {request_key}")
        frame.attrs["baostock_cache"] = {"request_key": request_key, "cache_hit": True}
        return frame

    def get_or_fetch(
        self,
        endpoint: str,
        args: Mapping[str, object],
        *,
        expected_fields: Sequence[str],
        fetcher: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        request_key = self.key(endpoint, args)
        cached = self._load_success(endpoint, request_key, expected_fields)
        if cached is not None:
            return cached

        base = self._base_state(endpoint, args, request_key)
        started = {**base, "status": "running", "started_at": _utc_now()}
        self._atomic_json(self.state_path(request_key), started)
        object_path = self.object_path(endpoint, request_key)
        object_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            frame = fetcher()
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("raw fetcher must return a pandas DataFrame")
            expected = tuple(str(name) for name in expected_fields)
            if tuple(frame.columns.astype(str)) != expected:
                raise SchemaDriftError(
                    f"{endpoint} normalized schema drift: expected {list(expected)}, "
                    f"received {list(frame.columns.astype(str))}"
                )
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{request_key}.", suffix=".parquet.tmp", dir=object_path.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            frame.to_parquet(temporary, index=False)
            # Sync the completed Parquet payload before publishing its name.
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            object_hash = _sha256_file(temporary)
            os.replace(temporary, object_path)
            temporary = None
            completed = {
                **base,
                "status": "success",
                "started_at": started["started_at"],
                "completed_at": _utc_now(),
                "rows": len(frame),
                "fields": list(expected),
                "object": str(object_path.relative_to(self.root)),
                "parquet_sha256": object_hash,
            }
            self._atomic_json(self.state_path(request_key), completed)
            output = frame.copy()
            output.attrs["baostock_cache"] = {"request_key": request_key, "cache_hit": False}
            return output
        except Exception as exc:
            failed = {
                **base,
                "status": "failed",
                "started_at": started["started_at"],
                "completed_at": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self._atomic_json(self.state_path(request_key), failed)
            raise
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


class BaoStockSession:
    """Logged-in BaoStock session backed by :class:`RawRequestCache`.

    The official Python client stores its connection in module-global state, so
    calls are deliberately serialized even if a caller accidentally shares this
    object across threads.
    """

    _EXPECTED_BY_ENDPOINT: Mapping[str, tuple[str, ...]] = {
        "trade_dates": TRADE_DATE_FIELDS,
        "hs300_stocks": HS300_FIELDS,
        "stock_basic": STOCK_BASIC_FIELDS,
        "stock_industry": INDUSTRY_FIELDS,
        "history_k_data": EQUITY_DAILY_FIELDS,
        "index_history": INDEX_DAILY_FIELDS,
    }

    def __init__(
        self,
        config: BaoStockConfig,
        *,
        client: BaoStockClientProtocol | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        if client is None:
            try:
                client = importlib.import_module("baostock")  # type: ignore[assignment]
            except ImportError as exc:
                raise ImportError(
                    "BaoStock is optional; install the project's BaoStock data extra before downloading"
                ) from exc
        self.client = client
        resolved_version = config.provider_version or str(getattr(client, "__version__", "")).strip()
        if not resolved_version:
            raise ValueError("cannot determine BaoStock version; set BaoStockConfig.provider_version")
        self.provider_version = resolved_version
        self.cache = RawRequestCache(config.cache_dir, provider_version=resolved_version)
        self._sleeper = sleeper
        self._logged_in = False
        self._call_lock = threading.RLock()

    def __enter__(self) -> "BaoStockSession":
        if self._logged_in:
            raise RuntimeError("BaoStock session is already open")
        result = self._retry_call("login", self.client.login)
        require_success_response(result, operation="login")
        self._logged_in = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._logged_in:
            try:
                result = self.client.logout()
                require_success_response(result, operation="logout")
            except Exception:
                if exc is None:
                    raise
            finally:
                self._logged_in = False
        return False

    def _ensure_open(self) -> None:
        if not self._logged_in:
            raise RuntimeError("use BaoStockSession as a context manager before querying")

    def _retry_call(self, operation: str, call: Callable[[], object]) -> object:
        delay = self.config.initial_backoff_seconds
        last: Exception | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                with self._call_lock:
                    result = call()
                require_success_response(result, operation=operation)
                return result
            except (BaoStockResponseError, ConnectionError, TimeoutError, OSError) as exc:
                last = exc
                if attempt == self.config.max_attempts:
                    break
                # BaoStock may invalidate an anonymous server session during a
                # long sequential plan.  Repeating the query on the same dead
                # module-global socket only returns "用户未登录" forever, so a
                # bounded retry must re-establish the session first.  The raw
                # cache still prevents a successfully committed request from
                # being issued twice after process restart.
                if self._logged_in and operation not in {"login", "logout"}:
                    try:
                        with self._call_lock:
                            self.client.logout()
                    except Exception:
                        pass
                    with self._call_lock:
                        login_result = self.client.login()
                    require_success_response(login_result, operation="relogin")
                if delay:
                    self._sleeper(delay)
                delay = min(delay * self.config.backoff_multiplier, self.config.maximum_backoff_seconds)
        assert last is not None
        raise last

    def _fetch(
        self,
        endpoint: str,
        args: Mapping[str, object],
        *,
        expected_fields: Sequence[str],
        allow_empty: bool,
        call: Callable[[], object],
    ) -> pd.DataFrame:
        self._ensure_open()

        def fetcher() -> pd.DataFrame:
            result = self._retry_call(endpoint, call)
            return result_to_frame(
                result,  # type: ignore[arg-type]
                expected_fields=expected_fields,
                operation=endpoint,
                allow_empty=allow_empty,
            )

        return self.cache.get_or_fetch(
            endpoint,
            args,
            expected_fields=expected_fields,
            fetcher=fetcher,
        )

    def query_trade_dates(self, start_date: str, end_date: str) -> pd.DataFrame:
        args = {"start_date": start_date, "end_date": end_date}
        return self._fetch(
            "trade_dates",
            args,
            expected_fields=TRADE_DATE_FIELDS,
            allow_empty=False,
            call=lambda: self.client.query_trade_dates(**args),
        )

    def query_hs300(self, date: str) -> pd.DataFrame:
        args = {"date": date}
        return self._fetch(
            "hs300_stocks",
            args,
            expected_fields=HS300_FIELDS,
            allow_empty=False,
            call=lambda: self.client.query_hs300_stocks(**args),
        )

    def query_history(
        self,
        code: str,
        start_date: str,
        end_date: str,
        *,
        adjustflag: str = "3",
        frequency: str = "d",
        fields: Sequence[str] = EQUITY_DAILY_FIELDS,
        allow_empty: bool = True,
    ) -> pd.DataFrame:
        expected = tuple(str(name) for name in fields)
        args = {
            "code": code,
            "fields": ",".join(expected),
            "start_date": start_date,
            "end_date": end_date,
            "frequency": frequency,
            "adjustflag": str(adjustflag),
        }
        return self._fetch(
            "history_k_data",
            args,
            expected_fields=expected,
            allow_empty=allow_empty,
            call=lambda: self.client.query_history_k_data_plus(**args),
        )

    def query_basic(self, *, code: str = "", code_name: str = "", allow_empty: bool = False) -> pd.DataFrame:
        args = {"code": code, "code_name": code_name}
        return self._fetch(
            "stock_basic",
            args,
            expected_fields=STOCK_BASIC_FIELDS,
            allow_empty=allow_empty,
            call=lambda: self.client.query_stock_basic(**args),
        )

    def query_industry(self, *, code: str = "", date: str, allow_empty: bool = False) -> pd.DataFrame:
        args = {"code": code, "date": date}
        return self._fetch(
            "stock_industry",
            args,
            expected_fields=INDUSTRY_FIELDS,
            allow_empty=allow_empty,
            call=lambda: self.client.query_stock_industry(**args),
        )

    def query_adjustment_factors(
        self,
        code: str,
        start_date: str,
        end_date: str,
        *,
        allow_empty: bool = False,
    ) -> pd.DataFrame:
        """Capture the small corporate-action factor event table for one code."""

        args = {"code": code, "start_date": start_date, "end_date": end_date}
        return self._fetch(
            "adjust_factor",
            args,
            expected_fields=ADJUST_FACTOR_FIELDS,
            allow_empty=allow_empty,
            call=lambda: self.client.query_adjust_factor(**args),
        )

    def query_index(
        self,
        start_date: str,
        end_date: str,
        *,
        code: str = "sh.000300",
        allow_empty: bool = False,
    ) -> pd.DataFrame:
        expected = INDEX_DAILY_FIELDS
        args = {
            "code": code,
            "fields": ",".join(expected),
            "start_date": start_date,
            "end_date": end_date,
            "frequency": "d",
            "adjustflag": "3",
        }
        return self._fetch(
            "index_history",
            args,
            expected_fields=expected,
            allow_empty=allow_empty,
            call=lambda: self.client.query_history_k_data_plus(**args),
        )

    def execute_request(self, request: BaoStockRequest) -> pd.DataFrame:
        args = dict(request.args)
        if request.endpoint == "trade_dates":
            return self.query_trade_dates(str(args["start_date"]), str(args["end_date"]))
        if request.endpoint == "hs300_stocks":
            return self.query_hs300(str(args["date"]))
        if request.endpoint == "history_k_data":
            fields_value = args.get("fields", EQUITY_DAILY_FIELDS)
            fields = tuple(str(fields_value).split(",")) if isinstance(fields_value, str) else tuple(fields_value)  # type: ignore[arg-type]
            return self.query_history(
                str(args["code"]),
                str(args["start_date"]),
                str(args["end_date"]),
                adjustflag=str(args.get("adjustflag", "3")),
                frequency=str(args.get("frequency", "d")),
                fields=fields,
                allow_empty=request.allow_empty,
            )
        if request.endpoint == "stock_basic":
            return self.query_basic(
                code=str(args.get("code", "")),
                code_name=str(args.get("code_name", "")),
                allow_empty=request.allow_empty,
            )
        if request.endpoint == "stock_industry":
            return self.query_industry(
                code=str(args.get("code", "")),
                date=str(args["date"]),
                allow_empty=request.allow_empty,
            )
        if request.endpoint == "adjust_factor":
            return self.query_adjustment_factors(
                str(args["code"]),
                str(args["start_date"]),
                str(args["end_date"]),
                allow_empty=request.allow_empty,
            )
        if request.endpoint == "index_history":
            return self.query_index(
                str(args["start_date"]),
                str(args["end_date"]),
                code=str(args.get("code", "sh.000300")),
                allow_empty=request.allow_empty,
            )
        raise ValueError(f"unsupported BaoStock request endpoint: {request.endpoint!r}")

    def resume(
        self,
        requests: Iterable[BaoStockRequest],
        *,
        continue_on_error: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Execute a plan; already-committed requests are served from cache."""

        output: dict[str, pd.DataFrame] = {}
        failures: list[Exception] = []
        for request in requests:
            try:
                frame = self.execute_request(request)
                key = str(frame.attrs["baostock_cache"]["request_key"])
                output[key] = frame
            except Exception as exc:
                if not continue_on_error:
                    raise
                failures.append(exc)
        if failures:
            # Results remain available to the caller through committed cache
            # entries; the raised summary makes a partial plan impossible to
            # mistake for a complete download.
            raise BaoStockError(
                f"resume completed with {len(failures)} failed request(s); first: "
                f"{type(failures[0]).__name__}: {failures[0]}"
            )
        return output


__all__ = [
    "ADJUST_FACTOR_FIELDS",
    "BaoStockClientProtocol",
    "BaoStockConfig",
    "BaoStockError",
    "BaoStockRequest",
    "BaoStockResponseError",
    "BaoStockSession",
    "CacheIntegrityError",
    "EQUITY_DAILY_FIELDS",
    "HS300_FIELDS",
    "INDEX_DAILY_FIELDS",
    "INDUSTRY_FIELDS",
    "RawRequestCache",
    "STOCK_BASIC_FIELDS",
    "SchemaDriftError",
    "TRADE_DATE_FIELDS",
    "make_request_key",
    "normalized_request_payload",
    "result_to_frame",
]
