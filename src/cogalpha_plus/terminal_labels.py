"""Audited terminal-security bridges for exact-endpoint forecast labels.

These bridges are narrow valuation rules for a security which ceases to exist
inside a label window.  They never add market bars, alter research eligibility,
or claim that the successor shares were executable at the valuation endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .manifest import hash_json
from .metrics import normalize_trading_calendar


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


def load_terminal_label_bridges(
    path: str | Path,
) -> tuple[tuple[TerminalLabelBridge, ...], dict[str, object]]:
    source_path = Path(path)
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("terminal label bridge catalog has an invalid schema")
    if payload.get("valuation_policy") != "exact_calendar_endpoint_successor_raw_vwap":
        raise ValueError("terminal label bridge catalog has an unsupported valuation policy")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("terminal label bridge events must be a list")
    events: list[TerminalLabelBridge] = []
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            raise ValueError("terminal label bridge event must be an object")
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
    ids = [event.event_id for event in events]
    symbols = [event.symbol for event in events]
    if len(ids) != len(set(ids)) or len(symbols) != len(set(symbols)):
        raise ValueError("terminal label bridge ids and source symbols must be unique")
    canonical = {
        "schema_version": 1,
        "valuation_policy": payload["valuation_policy"],
        "events": [event.to_dict() for event in events],
    }
    identity = {**canonical, "identity_hash": hash_json(canonical)}
    return tuple(events), identity


def apply_terminal_label_bridges(
    labels: pd.Series,
    raw_vwap: pd.Series,
    eligible_index: pd.MultiIndex,
    events: Sequence[TerminalLabelBridge],
    *,
    horizon: int,
    entry_delay: int,
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, object]]:
    """Fill only otherwise-missing eligible labels with sourced successor value."""

    if not labels.index.equals(raw_vwap.index):
        raise ValueError("terminal label bridge labels and raw VWAP index differ")
    calendar = normalize_trading_calendar(trading_calendar, panel_index=raw_vwap.index)
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
            if source_exit_key in raw_vwap.index and pd.notna(raw_vwap.loc[source_exit_key]):
                raise ValueError("terminal label bridge would overwrite an available source endpoint")
            entry_key = (entry_date, event.symbol)
            destination_key = (exit_date, event.destination_symbol)
            entry_value = pd.to_numeric(pd.Series([raw_vwap.get(entry_key)]), errors="coerce").iloc[0]
            destination_value = pd.to_numeric(
                pd.Series([raw_vwap.get(destination_key)]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(entry_value) or entry_value <= 0:
                raise ValueError("terminal label bridge source entry VWAP is unavailable")
            if not np.isfinite(destination_value) or destination_value <= 0:
                raise ValueError("terminal label bridge successor endpoint VWAP is unavailable")
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
                    "entry_raw_vwap": float(entry_value),
                    "destination_exit_raw_vwap": float(destination_value),
                    "share_ratio": event.share_ratio,
                    "net_cash_per_share": event.net_cash_per_share,
                    "forecast_return": bridged_return,
                    "executable_claim": False,
                }
            )
    audit_payload = {
        "schema_version": 1,
        "valuation_policy": "exact_calendar_endpoint_successor_raw_vwap",
        "applications": applications,
    }
    return out, {**audit_payload, "audit_hash": hash_json(audit_payload)}
