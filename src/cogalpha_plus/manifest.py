"""Immutable run manifests and canonical JSON hashing.

The manifest is deliberately independent of any model or backtest framework.  It
captures the inputs which make an alpha-mining run identifiable, so a trial can
always be tied back to an exact budget, seed, data snapshot, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return a stable, timezone-explicit timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> str:
    """Serialize JSON data in the single representation used for all hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def hash_json(value: Any) -> str:
    """Return a SHA-256 digest of canonical JSON data."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        # Validate floats (NaN and infinity are intentionally not permitted).
        canonical_json(value)
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class RunManifest:
    """A deeply immutable description of one search run.

    ``data_snapshot`` should contain immutable identifiers rather than a mutable
    location alone—for example vendor, universe, as-of timestamp, and a file or
    table checksum.  ``config_hash`` is the digest of the complete resolved
    configuration, not merely the path to a YAML file.
    """

    run_id: str
    created_at: str
    seed: int
    data_snapshot: Mapping[str, Any]
    config_hash: str
    search_budget: Mapping[str, int]
    model_version: str = ""
    prompt_version: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not self.config_hash.strip():
            raise ValueError("config_hash must not be empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )

        budget: dict[str, int] = {}
        for name, amount in self.search_budget.items():
            if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
                raise ValueError(f"search budget {name!r} must be a non-negative int")
            budget[str(name)] = amount
        if not budget:
            raise ValueError("search_budget must contain at least one limit")

        object.__setattr__(self, "data_snapshot", _freeze_json(self.data_snapshot))
        object.__setattr__(self, "search_budget", _freeze_json(budget))

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        data_snapshot: Mapping[str, Any] | str,
        search_budget: Mapping[str, int],
        config: Any | None = None,
        config_hash: str | None = None,
        model_version: str = "",
        prompt_version: str = "",
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> "RunManifest":
        """Build a manifest and hash a resolved configuration when supplied."""

        if config_hash is None:
            if config is None:
                raise ValueError("provide config or config_hash")
            config_hash = hash_json(config)
        elif config is not None and config_hash != hash_json(config):
            raise ValueError("config_hash does not match config")

        snapshot = (
            {"snapshot_id": data_snapshot}
            if isinstance(data_snapshot, str)
            else dict(data_snapshot)
        )
        return cls(
            run_id=run_id or f"run_{uuid4().hex}",
            created_at=created_at or utc_now(),
            seed=seed,
            data_snapshot=snapshot,
            config_hash=config_hash,
            search_budget=dict(search_budget),
            model_version=model_version,
            prompt_version=prompt_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "data_snapshot": _thaw_json(self.data_snapshot),
            "config_hash": self.config_hash,
            "search_budget": _thaw_json(self.search_budget),
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
        }

    @property
    def manifest_hash(self) -> str:
        return hash_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        return cls(
            run_id=str(payload["run_id"]),
            created_at=str(payload["created_at"]),
            seed=payload["seed"],
            data_snapshot=payload["data_snapshot"],
            config_hash=str(payload["config_hash"]),
            search_budget=payload["search_budget"],
            model_version=str(payload.get("model_version", "")),
            prompt_version=str(payload.get("prompt_version", "")),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        )

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Persist the manifest; accidental replacement is refused by default."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "x"
        envelope = {"manifest": self.to_dict(), "manifest_hash": self.manifest_hash}
        with target.open(mode, encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(envelope))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "RunManifest":
        target = Path(path)
        with target.open("r", encoding="utf-8") as handle:
            envelope = json.load(handle)
        manifest = cls.from_dict(envelope["manifest"])
        recorded_hash = str(envelope.get("manifest_hash", ""))
        if recorded_hash != manifest.manifest_hash:
            raise ValueError("manifest hash mismatch; file may have been modified")
        return manifest


__all__ = [
    "RunManifest",
    "SCHEMA_VERSION",
    "canonical_json",
    "hash_json",
    "utc_now",
]
