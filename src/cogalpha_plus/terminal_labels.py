"""Audited terminal-security bridges for exact-endpoint forecast labels.

These bridges are narrow valuation rules for a security which ceases to exist
inside a label window.  They never add market bars, alter research eligibility,
or claim that the successor shares were executable at the valuation endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .manifest import hash_json
from .metrics import normalize_trading_calendar


TERMINAL_LABEL_VALUATION_POLICY = (
    "exact_calendar_endpoint_successor_raw_endpoint_price"
)
_SUPPORTED_ENDPOINT_PRICE_FIELDS = frozenset({"open", "vwap"})
_CATALOG_IDENTITY_KEYS = frozenset(
    {"schema_version", "valuation_policy", "events", "identity_hash"}
)
_CATALOG_SOURCE_KEYS = frozenset({"schema_version", "valuation_policy", "events"})
_EVENT_IDENTITY_KEYS = frozenset(
    {
        "event_id",
        "kind",
        "symbol",
        "destination_symbol",
        "announcement_date",
        "effective_date",
        "settlement_date",
        "share_ratio",
        "net_cash_per_share",
        "sources",
    }
)
_EVENT_SOURCE_KEYS = _EVENT_IDENTITY_KEYS | {"note"}


@dataclass(frozen=True)
class TerminalLabelBridge:
    event_id: str
    kind: str
    symbol: str
    destination_symbol: str
    announcement_date: pd.Timestamp
    effective_date: pd.Timestamp
    settlement_date: pd.Timestamp
    share_ratio: float
    net_cash_per_share: float
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("event_id", "symbol", "destination_symbol"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"terminal label bridge {name} must be non-empty")
        if self.kind != "symbol_exchange":
            raise ValueError("only sourced symbol_exchange label bridges are supported")
        if self.symbol == self.destination_symbol:
            raise ValueError("terminal label bridge destination must differ from source")
        announcement = pd.Timestamp(self.announcement_date).normalize()
        effective = pd.Timestamp(self.effective_date).normalize()
        settlement = pd.Timestamp(self.settlement_date).normalize()
        if not announcement <= effective <= settlement:
            raise ValueError(
                "terminal label bridge requires announcement <= effective <= settlement"
            )
        ratio = float(self.share_ratio)
        cash = float(self.net_cash_per_share)
        if not np.isfinite(ratio) or ratio <= 0 or not np.isfinite(cash) or cash < 0:
            raise ValueError("terminal label bridge economics must be finite and non-negative")
        sources = tuple(str(item).strip() for item in self.sources)
        if not sources or any(not item.startswith("https://") for item in sources):
            raise ValueError("terminal label bridge requires HTTPS source provenance")
        object.__setattr__(self, "announcement_date", announcement)
        object.__setattr__(self, "effective_date", effective)
        object.__setattr__(self, "settlement_date", settlement)
        object.__setattr__(self, "share_ratio", ratio)
        object.__setattr__(self, "net_cash_per_share", cash)
        object.__setattr__(self, "sources", sources)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "symbol": self.symbol,
            "destination_symbol": self.destination_symbol,
            "announcement_date": str(self.announcement_date.date()),
            "effective_date": str(self.effective_date.date()),
            "settlement_date": str(self.settlement_date.date()),
            "share_ratio": self.share_ratio,
            "net_cash_per_share": self.net_cash_per_share,
            "sources": list(self.sources),
        }


def _canonical_catalog(
    events: Sequence[TerminalLabelBridge],
) -> dict[str, object]:
    normalized = tuple(events)
    if any(not isinstance(event, TerminalLabelBridge) for event in normalized):
        raise ValueError("terminal label bridge events must be TerminalLabelBridge objects")
    ids = [event.event_id for event in normalized]
    symbols = [event.symbol for event in normalized]
    if len(ids) != len(set(ids)) or len(symbols) != len(set(symbols)):
        raise ValueError("terminal label bridge ids and source symbols must be unique")
    return {
        "schema_version": 1,
        "valuation_policy": TERMINAL_LABEL_VALUATION_POLICY,
        "events": [event.to_dict() for event in normalized],
    }


def terminal_label_bridge_identity(
    events: Sequence[TerminalLabelBridge],
) -> dict[str, object]:
    """Return the exact, content-addressed identity for one event sequence."""

    canonical = _canonical_catalog(events)
    return {**canonical, "identity_hash": hash_json(canonical)}


def validate_terminal_label_bridge_identity(
    events: Sequence[TerminalLabelBridge],
    identity: Mapping[str, object] | None,
) -> dict[str, object]:
    """Require an exact, self-authenticating identity for ``events``."""

    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("terminal label bridges require a non-empty frozen catalog identity")
    supplied = dict(identity)
    if set(supplied) != _CATALOG_IDENTITY_KEYS:
        raise ValueError("terminal label bridge catalog identity keys are not exact")
    unsigned = {
        key: value for key, value in supplied.items() if key != "identity_hash"
    }
    if supplied["identity_hash"] != hash_json(unsigned):
        raise ValueError("terminal label bridge catalog identity hash is invalid")
    expected = terminal_label_bridge_identity(events)
    if supplied != expected:
        raise ValueError("terminal label bridge catalog identity does not match events")
    return expected


def load_terminal_label_bridge_catalog_snapshot(
    path: str | Path,
) -> tuple[tuple[TerminalLabelBridge, ...], dict[str, object], dict[str, object]]:
    """Parse and hash one immutable byte snapshot of a bridge catalog."""

    source_path = Path(path)
    raw_bytes = source_path.read_bytes()
    payload = yaml.safe_load(raw_bytes.decode("utf-8"))
    if (
        not isinstance(payload, Mapping)
        or set(payload) != _CATALOG_SOURCE_KEYS
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise ValueError("terminal label bridge catalog has an invalid schema")
    if payload.get("valuation_policy") != TERMINAL_LABEL_VALUATION_POLICY:
        raise ValueError("terminal label bridge catalog has an unsupported valuation policy")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("terminal label bridge events must be a list")
    events: list[TerminalLabelBridge] = []
    for raw in raw_events:
        if not isinstance(raw, Mapping) or set(raw) != _EVENT_SOURCE_KEYS:
            raise ValueError("terminal label bridge event keys are not exact")
        events.append(
            TerminalLabelBridge(
                event_id=str(raw.get("event_id", "")),
                kind=str(raw.get("kind", "")),
                symbol=str(raw.get("symbol", "")),
                destination_symbol=str(raw.get("destination_symbol", "")),
                announcement_date=pd.Timestamp(raw.get("announcement_date")),
                effective_date=pd.Timestamp(raw.get("effective_date")),
                settlement_date=pd.Timestamp(raw.get("settlement_date")),
                share_ratio=float(raw.get("share_ratio", np.nan)),
                net_cash_per_share=float(raw.get("net_cash_per_share", np.nan)),
                sources=tuple(raw.get("sources", ())),
            )
        )
    identity = terminal_label_bridge_identity(events)
    raw_identity = {
        "path": str(source_path.resolve()),
        "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "file_bytes": len(raw_bytes),
    }
    return tuple(events), identity, raw_identity


def load_terminal_label_bridges(
    path: str | Path,
) -> tuple[tuple[TerminalLabelBridge, ...], dict[str, object]]:
    events, identity, _ = load_terminal_label_bridge_catalog_snapshot(path)
    return events, identity


def terminal_label_tape_snapshot_hash(tape: pd.DataFrame) -> str:
    """Match the repository's semantic panel hash without importing runner."""

    hashed = pd.util.hash_pandas_object(
        tape, index=True, categorize=True
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def apply_terminal_label_bridges(
    labels: pd.Series,
    endpoint_price_tape: pd.DataFrame,
    eligible_index: pd.MultiIndex,
    events: Sequence[TerminalLabelBridge],
    *,
    terminal_label_bridge_identity: Mapping[str, object] | None = None,
    endpoint_price_field: str,
    horizon: int,
    entry_delay: int,
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, object]]:
    """Fill missing labels from a successor's raw endpoint price.

    A complete raw/adjusted tape is required instead of a naked price Series.
    The relation check detects internally inconsistent columns. Raw-price
    provenance remains an upstream frozen-tape lineage responsibility, so the
    exact tape snapshot is recorded in this audit.
    """

    events = tuple(events)
    if events:
        catalog_identity = validate_terminal_label_bridge_identity(
            events, terminal_label_bridge_identity
        )
    elif terminal_label_bridge_identity is None or terminal_label_bridge_identity == {}:
        catalog_identity = {}
    else:
        catalog_identity = validate_terminal_label_bridge_identity(
            events, terminal_label_bridge_identity
        )
    if endpoint_price_field not in _SUPPORTED_ENDPOINT_PRICE_FIELDS:
        raise ValueError("terminal label bridge endpoint price field must be open or vwap")
    if not isinstance(endpoint_price_tape, pd.DataFrame):
        raise TypeError(
            "terminal label bridge requires the complete raw/adjusted endpoint price tape"
        )
    if not labels.index.equals(endpoint_price_tape.index):
        raise ValueError("terminal label bridge labels and endpoint price tape index differ")
    adjusted_field = f"adj_{endpoint_price_field}"
    required = {endpoint_price_field, adjusted_field, "adj_factor"}
    missing = sorted(required.difference(endpoint_price_tape.columns))
    if missing:
        raise ValueError(
            "terminal label bridge endpoint price tape is missing raw/adjusted contract fields: "
            + ",".join(missing)
        )
    raw_endpoint_price = pd.to_numeric(
        endpoint_price_tape[endpoint_price_field], errors="coerce"
    )
    adjusted_endpoint_price = pd.to_numeric(
        endpoint_price_tape[adjusted_field], errors="coerce"
    )
    adjustment_factor = pd.to_numeric(
        endpoint_price_tape["adj_factor"], errors="coerce"
    )
    tape_snapshot_hash = terminal_label_tape_snapshot_hash(endpoint_price_tape)
    if events:
        numeric = pd.DataFrame(
            {
                "raw": raw_endpoint_price,
                "adjusted": adjusted_endpoint_price,
                "factor": adjustment_factor,
            },
            index=endpoint_price_tape.index,
        )
        values = numeric.to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (numeric["raw"] <= 0).any()
            or (numeric["adjusted"] <= 0).any()
            or (numeric["factor"] <= 0).any()
        ):
            raise ValueError(
                "terminal label bridge raw/adjusted endpoint price contract is non-finite or non-positive"
            )
        expected_adjusted = numeric["raw"].mul(numeric["factor"])
        if not np.allclose(
            numeric["adjusted"].to_numpy(dtype=float),
            expected_adjusted.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError(
                "terminal label bridge endpoint price tape violates adj_price == raw_price * adj_factor"
            )
    calendar = normalize_trading_calendar(
        trading_calendar, panel_index=endpoint_price_tape.index
    )
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    out = labels.copy()
    applications: list[dict[str, object]] = []
    for event in events:
        source_rows = eligible_index[
            eligible_index.get_level_values("symbol") == event.symbol
        ]
        for signal_date, symbol in source_rows:
            key = (pd.Timestamp(signal_date), str(symbol))
            if pd.notna(out.get(key, np.nan)):
                continue
            signal = pd.Timestamp(signal_date).normalize()
            if signal not in positions.index:
                raise ValueError("terminal label bridge signal is outside the calendar")
            exit_position = int(positions.loc[signal]) + entry_delay + horizon
            entry_position = int(positions.loc[signal]) + entry_delay
            if exit_position >= len(calendar):
                continue
            entry_date = calendar[entry_position]
            exit_date = calendar[exit_position]
            if not (event.announcement_date <= signal < event.effective_date <= exit_date):
                continue
            source_exit_key = (exit_date, event.symbol)
            if source_exit_key in raw_endpoint_price.index and pd.notna(
                raw_endpoint_price.loc[source_exit_key]
            ):
                raise ValueError("terminal label bridge would overwrite an available source endpoint")
            entry_key = (entry_date, event.symbol)
            destination_key = (exit_date, event.destination_symbol)
            entry_value = pd.to_numeric(
                pd.Series([raw_endpoint_price.get(entry_key)]), errors="coerce"
            ).iloc[0]
            destination_value = pd.to_numeric(
                pd.Series([raw_endpoint_price.get(destination_key)]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(entry_value) or entry_value <= 0:
                raise ValueError("terminal label bridge source raw entry price is unavailable")
            if not np.isfinite(destination_value) or destination_value <= 0:
                raise ValueError(
                    "terminal label bridge successor raw endpoint price is unavailable"
                )
            terminal_value = event.share_ratio * destination_value + event.net_cash_per_share
            bridged_return = float(terminal_value / entry_value - 1.0)
            out.loc[key] = bridged_return
            applications.append(
                {
                    "event_id": event.event_id,
                    "signal_date": str(signal.date()),
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(exit_date.date()),
                    "symbol": event.symbol,
                    "destination_symbol": event.destination_symbol,
                    "endpoint_price_field": endpoint_price_field,
                    "endpoint_price_basis": "declared_raw_unadjusted_from_upstream_frozen_tape",
                    "entry_raw_endpoint_price": float(entry_value),
                    "destination_exit_raw_endpoint_price": float(destination_value),
                    f"entry_raw_{endpoint_price_field}": float(entry_value),
                    f"destination_exit_raw_{endpoint_price_field}": float(
                        destination_value
                    ),
                    "share_ratio": event.share_ratio,
                    "net_cash_per_share": event.net_cash_per_share,
                    "forecast_return": bridged_return,
                    "executable_claim": False,
                }
            )
    audit_payload = {
        "schema_version": 3,
        "valuation_policy": TERMINAL_LABEL_VALUATION_POLICY,
        "terminal_label_bridge_identity": catalog_identity,
        "endpoint_price_field": endpoint_price_field,
        "endpoint_price_basis": "declared_raw_unadjusted_from_upstream_frozen_tape",
        "raw_price_authentication_boundary": "upstream_frozen_tape_lineage_required",
        "internal_relation_check": (
            "adj_endpoint_price_equals_declared_raw_endpoint_price_times_adj_factor"
        ),
        "endpoint_price_tape_snapshot_hash": tape_snapshot_hash,
        "applications": applications,
    }
    return out, {**audit_payload, "audit_hash": hash_json(audit_payload)}
