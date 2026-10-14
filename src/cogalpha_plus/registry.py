"""Content-addressed factor records and append-only experiment ledgers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

try:  # ``fcntl`` exists on the Linux/macOS environments used by this project.
    import fcntl
except ImportError:  # pragma: no cover - fallback for non-POSIX development.
    fcntl = None  # type: ignore[assignment]

from .manifest import canonical_json, hash_json, utc_now


FACTOR_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64
VALID_STATUSES = frozenset({"success", "failed"})
VALID_OPERATIONS = frozenset(
    {"seed", "mutation", "crossover", "injection", "elite", "manual"}
)


class LedgerIntegrityError(RuntimeError):
    """Raised when an append-only trial ledger fails hash-chain verification."""


class RegistryIntegrityError(RuntimeError):
    """Raised when stored factor records fail content-address verification."""


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized[key] = _freeze_json(item)
        return MappingProxyType(dict(sorted(normalized.items())))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        canonical_json(value)  # Reject NaN and infinity.
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def formula_hash(formula_ast: Any) -> str:
    """Hash a JSON AST after key-order canonicalization."""

    return hash_json(formula_ast)


def factor_content_id(
    formula_ast_hash: str, data_version: str, config_version: str
) -> str:
    """Derive a stable identity for a formula evaluated in one data context."""

    digest = hash_json(
        {
            "formula_ast_hash": formula_ast_hash,
            "data_version": data_version,
            "config_version": config_version,
        }
    )
    return f"fac_{digest}"


@dataclass(frozen=True)
class FactorLineage:
    """How a candidate was produced in the evolutionary search."""

    parent_factor_ids: tuple[str, ...]
    operation: str
    operator: str
    agent_id: str
    generation: int

    def __post_init__(self) -> None:
        parents = tuple(str(parent) for parent in self.parent_factor_ids)
        if len(parents) != len(set(parents)):
            raise ValueError("parent_factor_ids must be unique")
        if any(not parent.strip() for parent in parents):
            raise ValueError("parent_factor_ids must not contain empty values")
        if self.operation not in VALID_OPERATIONS:
            raise ValueError(
                f"unsupported operation {self.operation!r}; "
                f"expected one of {sorted(VALID_OPERATIONS)}"
            )
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be empty")
        if not isinstance(self.generation, int) or isinstance(self.generation, bool):
            raise TypeError("generation must be an integer")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if self.operation == "mutation" and len(parents) < 1:
            raise ValueError("mutation lineage requires at least one parent")
        if self.operation == "crossover" and len(parents) < 2:
            raise ValueError("crossover lineage requires at least two parents")
        if self.operation in {"mutation", "crossover"} and not self.operator.strip():
            raise ValueError(f"{self.operation} lineage requires a named operator")
        object.__setattr__(self, "parent_factor_ids", parents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_factor_ids": list(self.parent_factor_ids),
            "operation": self.operation,
            "operator": self.operator,
            "agent_id": self.agent_id,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorLineage":
        return cls(
            parent_factor_ids=tuple(payload.get("parent_factor_ids", ())),
            operation=str(payload["operation"]),
            operator=str(payload.get("operator", "")),
            agent_id=str(payload["agent_id"]),
            generation=payload["generation"],
        )


@dataclass(frozen=True)
class FactorRecord:
    """An immutable success or failure record for one unique factor candidate."""

    factor_id: str
    name: str
    formula_ast: Any
    formula_ast_hash: str
    status: str
    lineage: FactorLineage
    run_id: str
    data_version: str
    config_version: str
    model_version: str
    prompt_version: str
    metrics: Mapping[str, Any]
    error: str = ""
    metadata: Mapping[str, Any] | None = None
    created_at: str = ""
    schema_version: int = FACTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FACTOR_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported factor schema {self.schema_version}; "
                f"expected {FACTOR_SCHEMA_VERSION}"
            )
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        for field_name in (
            "run_id",
            "data_version",
            "config_version",
            "model_version",
            "prompt_version",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.status == "failed" and not self.error.strip():
            raise ValueError("failed records require an error or rejection reason")
        if self.status == "success" and self.error:
            raise ValueError("successful records must not carry an error")

        # Freeze only after hashing; callers can supply ordinary dict/list ASTs.
        actual_formula_hash = formula_hash(self.formula_ast)
        if self.formula_ast_hash != actual_formula_hash:
            raise ValueError("formula_ast_hash does not match formula_ast")
        expected_id = factor_content_id(
            actual_formula_hash, self.data_version, self.config_version
        )
        if self.factor_id != expected_id:
            raise ValueError("factor_id does not match content-addressed identity")

        object.__setattr__(self, "formula_ast", _freeze_json(self.formula_ast))
        object.__setattr__(self, "metrics", _freeze_json(dict(self.metrics)))
        object.__setattr__(
            self, "metadata", _freeze_json(dict(self.metadata or {}))
        )
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now())

    @classmethod
    def create(
        cls,
        *,
        name: str,
        formula_ast: Any,
        status: str,
        lineage: FactorLineage,
        run_id: str,
        data_version: str,
        config_version: str,
        model_version: str,
        prompt_version: str,
        metrics: Mapping[str, Any] | None = None,
        error: str = "",
        metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> "FactorRecord":
        ast_hash = formula_hash(formula_ast)
        return cls(
            factor_id=factor_content_id(ast_hash, data_version, config_version),
            name=name,
            formula_ast=formula_ast,
            formula_ast_hash=ast_hash,
            status=status,
            lineage=lineage,
            run_id=run_id,
            data_version=data_version,
            config_version=config_version,
            model_version=model_version,
            prompt_version=prompt_version,
            metrics=dict(metrics or {}),
            error=error,
            metadata=dict(metadata or {}),
            created_at=created_at or utc_now(),
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "factor_id": self.factor_id,
            "name": self.name,
            "formula_ast": _thaw_json(self.formula_ast),
            "formula_ast_hash": self.formula_ast_hash,
            "status": self.status,
            "lineage": self.lineage.to_dict(),
            "run_id": self.run_id,
            "data_version": self.data_version,
            "config_version": self.config_version,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "metrics": _thaw_json(self.metrics),
            "error": self.error,
            "metadata": _thaw_json(self.metadata),
            "created_at": self.created_at,
        }

    @property
    def record_hash(self) -> str:
        return hash_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self._content_dict()
        payload["record_hash"] = self.record_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorRecord":
        record = cls(
            factor_id=str(payload["factor_id"]),
            name=str(payload.get("name", "")),
            formula_ast=payload["formula_ast"],
            formula_ast_hash=str(payload["formula_ast_hash"]),
            status=str(payload["status"]),
            lineage=FactorLineage.from_dict(payload["lineage"]),
            run_id=str(payload["run_id"]),
            data_version=str(payload["data_version"]),
            config_version=str(payload["config_version"]),
            model_version=str(payload["model_version"]),
            prompt_version=str(payload["prompt_version"]),
            metrics=payload.get("metrics", {}),
            error=str(payload.get("error", "")),
            metadata=payload.get("metadata", {}),
            created_at=str(payload["created_at"]),
            schema_version=int(payload.get("schema_version", FACTOR_SCHEMA_VERSION)),
        )
        recorded_hash = str(payload.get("record_hash", ""))
        if recorded_hash and recorded_hash != record.record_hash:
            raise ValueError("factor record hash mismatch")
        return record


@dataclass(frozen=True)
class RegistrationResult:
    inserted: bool
    record: FactorRecord
    duplicate_of: str | None = None


@contextmanager
def _locked(path: Path, *, exclusive: bool) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", newline="\n") as handle:
        if fcntl is not None:
            lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle.fileno(), lock_mode)
        try:
            yield handle
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class FactorRegistry:
    """A file-backed, append-only library of unique factor records.

    Deduplication is exact and deterministic: two records with the same
    canonical formula AST, data version, and configuration version share the
    same ``factor_id``.  Statistical-correlation deduplication belongs in the
    validator and can be recorded as a failed factor with its rejection reason.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        raw = Path(path)
        self.path = raw if raw.suffix == ".jsonl" else raw / "factors.jsonl"
        self._thread_lock = threading.RLock()

    @staticmethod
    def _read_handle(handle: Any) -> tuple[list[FactorRecord], dict[str, FactorRecord]]:
        handle.seek(0)
        records: list[FactorRecord] = []
        by_id: dict[str, FactorRecord] = {}
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = FactorRecord.from_dict(payload)
            except Exception as exc:
                raise RegistryIntegrityError(
                    f"invalid factor registry entry at line {line_number}: {exc}"
                ) from exc
            if record.factor_id in by_id:
                raise RegistryIntegrityError(
                    f"duplicate factor_id at line {line_number}: {record.factor_id}"
                )
            records.append(record)
            by_id[record.factor_id] = record
        return records, by_id

    def register(self, record: FactorRecord) -> RegistrationResult:
        """Insert a unique record or return the already registered equivalent."""

        if not isinstance(record, FactorRecord):
            raise TypeError("record must be a FactorRecord")
        with self._thread_lock, _locked(self.path, exclusive=True) as handle:
            _, by_id = self._read_handle(handle)
            existing = by_id.get(record.factor_id)
            if existing is not None:
                return RegistrationResult(
                    inserted=False,
                    record=existing,
                    duplicate_of=existing.factor_id,
                )
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json(record.to_dict()))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            return RegistrationResult(inserted=True, record=record)

    def records(self, *, status: str | None = None) -> tuple[FactorRecord, ...]:
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        with self._thread_lock, _locked(self.path, exclusive=False) as handle:
            records, _ = self._read_handle(handle)
        if status is not None:
            records = [record for record in records if record.status == status]
        return tuple(records)

    def get(self, factor_id: str) -> FactorRecord | None:
        with self._thread_lock, _locked(self.path, exclusive=False) as handle:
            _, by_id = self._read_handle(handle)
        return by_id.get(factor_id)

    def contains_formula(
        self,
        formula_ast_or_hash: Any,
        *,
        data_version: str,
        config_version: str,
    ) -> bool:
        ast_hash = (
            formula_ast_or_hash
            if isinstance(formula_ast_or_hash, str)
            and len(formula_ast_or_hash) == 64
            else formula_hash(formula_ast_or_hash)
        )
        return self.get(factor_content_id(ast_hash, data_version, config_version)) is not None

    def stats(self) -> dict[str, int]:
        records = self.records()
        return {
            "total": len(records),
            "success": sum(record.status == "success" for record in records),
            "failed": sum(record.status == "failed" for record in records),
        }

    def __len__(self) -> int:
        return len(self.records())


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    timestamp: str
    event_type: str
    payload: Any
    prev_hash: str
    entry_hash: str
    schema_version: int = LEDGER_SCHEMA_VERSION

    def body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": _thaw_json(self.payload),
            "prev_hash": self.prev_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.body_dict()
        result["entry_hash"] = self.entry_hash
        return result

    @classmethod
    def build(
        cls,
        *,
        seq: int,
        timestamp: str,
        event_type: str,
        payload: Any,
        prev_hash: str,
    ) -> "LedgerEntry":
        if not event_type.strip():
            raise ValueError("event_type must not be empty")
        frozen_payload = _freeze_json(payload)
        body = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "seq": seq,
            "timestamp": timestamp,
            "event_type": event_type,
            "payload": _thaw_json(frozen_payload),
            "prev_hash": prev_hash,
        }
        return cls(
            seq=seq,
            timestamp=timestamp,
            event_type=event_type,
            payload=frozen_payload,
            prev_hash=prev_hash,
            entry_hash=hash_json(body),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LedgerEntry":
        return cls(
            seq=payload["seq"],
            timestamp=str(payload["timestamp"]),
            event_type=str(payload["event_type"]),
            payload=_freeze_json(payload["payload"]),
            prev_hash=str(payload["prev_hash"]),
            entry_hash=str(payload["entry_hash"]),
            schema_version=int(payload.get("schema_version", LEDGER_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class LedgerVerification:
    valid: bool
    entries: int
    head_hash: str
    error: str = ""
    bad_sequence: int | None = None

    def __bool__(self) -> bool:
        return self.valid


class TrialLedger:
    """An append-only JSONL event log protected by a SHA-256 hash chain."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        raw = Path(path)
        self.path = raw if raw.suffix == ".jsonl" else raw / "trials.jsonl"
        self._thread_lock = threading.RLock()

    @staticmethod
    def _parse_lines(lines: Sequence[str]) -> tuple[list[LedgerEntry], LedgerVerification]:
        entries: list[LedgerEntry] = []
        expected_prev = GENESIS_HASH
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            seq = len(entries)
            try:
                entry = LedgerEntry.from_dict(json.loads(line))
            except Exception as exc:
                return entries, LedgerVerification(
                    False,
                    len(entries),
                    expected_prev,
                    f"invalid JSON or schema at line {line_number}: {exc}",
                    seq,
                )
            if entry.schema_version != LEDGER_SCHEMA_VERSION:
                return entries, LedgerVerification(
                    False,
                    len(entries),
                    expected_prev,
                    f"unsupported schema at sequence {seq}",
                    seq,
                )
            if entry.seq != seq:
                return entries, LedgerVerification(
                    False,
                    len(entries),
                    expected_prev,
                    f"expected sequence {seq}, found {entry.seq}",
                    seq,
                )
            if entry.prev_hash != expected_prev:
                return entries, LedgerVerification(
                    False,
                    len(entries),
                    expected_prev,
                    f"previous hash mismatch at sequence {seq}",
                    seq,
                )
            actual_hash = hash_json(entry.body_dict())
            if entry.entry_hash != actual_hash:
                return entries, LedgerVerification(
                    False,
                    len(entries),
                    expected_prev,
                    f"entry hash mismatch at sequence {seq}",
                    seq,
                )
            entries.append(entry)
            expected_prev = entry.entry_hash
        return entries, LedgerVerification(
            True, len(entries), expected_prev if entries else GENESIS_HASH
        )

    def append(
        self,
        event_type: str,
        payload: Any,
        *,
        timestamp: str | None = None,
    ) -> LedgerEntry:
        """Atomically append an event, refusing to extend a corrupted chain."""

        with self._thread_lock, _locked(self.path, exclusive=True) as handle:
            handle.seek(0)
            lines = handle.readlines()
            entries, verification = self._parse_lines(lines)
            if not verification:
                raise LedgerIntegrityError(verification.error)
            entry = LedgerEntry.build(
                seq=len(entries),
                timestamp=timestamp or utc_now(),
                event_type=event_type,
                payload=payload,
                prev_hash=verification.head_hash,
            )
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json(entry.to_dict()))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            return entry

    def entries(self) -> tuple[LedgerEntry, ...]:
        with self._thread_lock, _locked(self.path, exclusive=False) as handle:
            handle.seek(0)
            entries, verification = self._parse_lines(handle.readlines())
        if not verification:
            raise LedgerIntegrityError(verification.error)
        return tuple(entries)

    def verify(
        self,
        *,
        expected_head: str | None = None,
        expected_entries: int | None = None,
    ) -> LedgerVerification:
        """Verify hashes, ordering, and optional externally anchored tail state."""

        with self._thread_lock, _locked(self.path, exclusive=False) as handle:
            handle.seek(0)
            _, verification = self._parse_lines(handle.readlines())
        if not verification:
            return verification
        if expected_entries is not None and verification.entries != expected_entries:
            return LedgerVerification(
                False,
                verification.entries,
                verification.head_hash,
                f"expected {expected_entries} entries, found {verification.entries}",
            )
        if expected_head is not None and verification.head_hash != expected_head:
            return LedgerVerification(
                False,
                verification.entries,
                verification.head_hash,
                "ledger head does not match external anchor",
            )
        return verification


__all__ = [
    "FACTOR_SCHEMA_VERSION",
    "GENESIS_HASH",
    "FactorLineage",
    "FactorRecord",
    "FactorRegistry",
    "LedgerEntry",
    "LedgerIntegrityError",
    "LedgerVerification",
    "RegistrationResult",
    "RegistryIntegrityError",
    "TrialLedger",
    "factor_content_id",
    "formula_hash",
]
