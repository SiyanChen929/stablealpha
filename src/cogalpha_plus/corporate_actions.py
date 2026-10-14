"""Explicit corporate-action events and portfolio-ledger application.

The module deliberately never infers an economic event from an adjustment
factor.  An adjustment-factor step is only a coverage alarm: cash, share and
terminal-security accounting require a dated, typed event from an auditable
source.

Only events whose cash/share consequences are completely specified are
applied.  Rights offerings are represented in the schema so a data adapter can
surface them, but this research implementation fails closed because exercising
rights is a portfolio decision and cannot be reconstructed from a price factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Iterable

import pandas as pd

if TYPE_CHECKING:  # Avoid an execution <-> ledger import cycle at runtime.
    from .execution import PortfolioState


class CorporateActionError(RuntimeError):
    """Base class for corporate-action accounting failures."""


class CorporateActionCoverageError(CorporateActionError):
    """A held security changed adjustment basis without a supplied event."""


class UnsupportedCorporateActionError(CorporateActionError):
    """An event is known but cannot be booked without an investment decision."""


class CorporateActionKind(str, Enum):
    CASH_DIVIDEND = "cash_dividend"
    SHARE_DISTRIBUTION = "share_distribution"
    SYMBOL_EXCHANGE = "symbol_exchange"
    CASH_OUT = "cash_out"
    RIGHTS_OFFERING = "rights_offering"


@dataclass(frozen=True)
class CorporateActionEvent:
    """One fully dated corporate-action instruction.

    ``effective_date`` is the first session on which the economic entitlement
    is recognized (normally the ex-date).  ``settlement_date`` is the cash-pay
    or new-share tradability date and cannot precede it.

    Cash amounts must already be on the intended net/gross basis.  The ledger
    will not guess dividend tax.  ``share_ratio`` means *new shares received per
    old share* for a distribution, and *destination shares per old share* for a
    symbol exchange.
    """

    event_id: str
    kind: CorporateActionKind | str
    symbol: str
    effective_date: pd.Timestamp | str
    settlement_date: pd.Timestamp | str
    net_cash_per_share: float = 0.0
    share_ratio: float = 0.0
    destination_symbol: str | None = None
    subscription_price: float | None = None
    source: str = ""

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        symbol = str(self.symbol).strip()
        source = str(self.source).strip()
        try:
            kind = CorporateActionKind(self.kind)
        except ValueError as exc:
            raise ValueError(f"unsupported corporate-action kind: {self.kind!r}") from exc
        effective = pd.Timestamp(self.effective_date).normalize()
        settlement = pd.Timestamp(self.settlement_date).normalize()
        destination = str(self.destination_symbol).strip() if self.destination_symbol else None
        cash = float(self.net_cash_per_share)
        ratio = float(self.share_ratio)
        if not event_id or not symbol or not source:
            raise ValueError("event_id, symbol and source must be non-empty")
        if settlement < effective:
            raise ValueError("corporate-action settlement precedes effective date")
        if not math.isfinite(cash) or cash < 0 or not math.isfinite(ratio) or ratio < 0:
            raise ValueError("corporate-action cash and share ratios must be finite and non-negative")
        if kind is CorporateActionKind.CASH_DIVIDEND:
            if cash <= 0 or ratio != 0 or destination is not None:
                raise ValueError("cash dividend requires only positive net_cash_per_share")
        elif kind is CorporateActionKind.SHARE_DISTRIBUTION:
            if ratio <= 0 or cash != 0 or destination is not None:
                raise ValueError("share distribution requires only positive share_ratio")
        elif kind is CorporateActionKind.SYMBOL_EXCHANGE:
            if ratio <= 0 or not destination or destination == symbol:
                raise ValueError("symbol exchange requires a distinct destination and positive ratio")
        elif kind is CorporateActionKind.CASH_OUT:
            if cash <= 0 or ratio != 0 or destination is not None:
                raise ValueError("cash-out requires only positive net_cash_per_share")
        elif kind is CorporateActionKind.RIGHTS_OFFERING:
            price = self.subscription_price
            if ratio <= 0 or price is None or not math.isfinite(float(price)) or float(price) < 0:
                raise ValueError("rights offering requires positive ratio and non-negative subscription price")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "effective_date", effective)
        object.__setattr__(self, "settlement_date", settlement)
        object.__setattr__(self, "destination_symbol", destination)
        object.__setattr__(self, "net_cash_per_share", cash)
        object.__setattr__(self, "share_ratio", ratio)
        object.__setattr__(self, "source", source)
        if self.subscription_price is not None:
            object.__setattr__(self, "subscription_price", float(self.subscription_price))


@dataclass(frozen=True)
class CorporateActionApplication:
    date: str
    event_id: str
    symbol: str
    kind: str
    phase: str
    old_shares: int
    new_shares: int
    cash_delta: float
    receivable_delta: float
    detail: str


def _whole_shares(value: float, *, event_id: str) -> int:
    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-8):
        raise CorporateActionError(
            f"event {event_id!r} creates fractional shares; an explicit cash-in-lieu event is required"
        )
    if rounded < 0:
        raise CorporateActionError(f"event {event_id!r} creates negative shares")
    return int(rounded)


class CorporateActionLedger:
    """Apply an immutable set of explicit events to a :class:`PortfolioState`.

    Idempotency markers live in ``PortfolioState`` rather than this object so a
    checkpointed portfolio cannot accidentally receive an event twice.
    """

    def __init__(self, events: Iterable[CorporateActionEvent] = ()) -> None:
        ordered = tuple(
            sorted(
                events,
                key=lambda item: (
                    pd.Timestamp(item.effective_date),
                    pd.Timestamp(item.settlement_date),
                    item.symbol,
                    item.event_id,
                ),
            )
        )
        ids = [event.event_id for event in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("corporate-action event_id values must be unique")
        terminal = [
            event.symbol
            for event in ordered
            if event.kind in {CorporateActionKind.CASH_OUT, CorporateActionKind.SYMBOL_EXCHANGE}
        ]
        if len(terminal) != len(set(terminal)):
            raise ValueError("a source symbol cannot have multiple terminal corporate actions")
        share_changing_keys = [
            (event.symbol, event.effective_date)
            for event in ordered
            if event.kind
            in {
                CorporateActionKind.SHARE_DISTRIBUTION,
                CorporateActionKind.SYMBOL_EXCHANGE,
                CorporateActionKind.CASH_OUT,
                CorporateActionKind.RIGHTS_OFFERING,
            }
        ]
        if len(share_changing_keys) != len(set(share_changing_keys)):
            raise ValueError("multiple same-day share-changing events require one consolidated source event")
        self.events = ordered
        self._effective = self._group_by("effective_date")
        self._settlement = self._group_by("settlement_date")

    def _group_by(self, name: str) -> dict[pd.Timestamp, tuple[CorporateActionEvent, ...]]:
        grouped: dict[pd.Timestamp, list[CorporateActionEvent]] = {}
        for event in self.events:
            grouped.setdefault(pd.Timestamp(getattr(event, name)), []).append(event)
        return {date: tuple(values) for date, values in grouped.items()}

    def covers_adjustment(self, symbol: str, date: pd.Timestamp | str) -> bool:
        """Whether a sourced economic event explains an adjustment step."""

        day = pd.Timestamp(date).normalize()
        return any(event.symbol == str(symbol) for event in self._effective.get(day, ()))

    def terminal_dates(self) -> dict[str, pd.Timestamp]:
        """Return effective dates after which a source security ceases to exist."""

        terminal = {CorporateActionKind.CASH_OUT, CorporateActionKind.SYMBOL_EXCHANGE}
        return {
            event.symbol: pd.Timestamp(event.effective_date)
            for event in self.events
            if event.kind in terminal
        }

    def validate_calendar(self, dates: Iterable[pd.Timestamp | str]) -> None:
        """Fail if an in-range event would be silently skipped by the tape."""

        calendar = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
        if calendar.empty:
            raise ValueError("corporate-action calendar is empty")
        available = set(calendar)
        for event in self.events:
            for field_name in ("effective_date", "settlement_date"):
                day = pd.Timestamp(getattr(event, field_name))
                if calendar[0] <= day <= calendar[-1] and day not in available:
                    raise CorporateActionError(
                        f"event {event.event_id!r} {field_name} {day.date()} is absent from execution tape"
                    )

    def apply_opening(
        self,
        date: pd.Timestamp | str,
        state: PortfolioState,
    ) -> tuple[CorporateActionApplication, ...]:
        """Book events before that session's orders and raw-price valuation."""

        day = pd.Timestamp(date).normalize()
        applications: list[CorporateActionApplication] = []
        opening_shares = {
            event.symbol: state.total_shares(event.symbol)
            for event in self._effective.get(day, ())
        }

        # Economic recognition is first so same-day payments/tradability can be
        # settled immediately and still leave a complete audit trail.
        for event in self._effective.get(day, ()):
            marker = f"effective:{event.event_id}"
            if marker in state.applied_corporate_action_steps:
                continue
            if event.kind is CorporateActionKind.RIGHTS_OFFERING:
                raise UnsupportedCorporateActionError(
                    f"rights offering {event.event_id!r} requires an explicit subscribe/decline decision"
                )
            old = opening_shares[event.symbol]
            cash_delta = 0.0
            receivable_delta = 0.0
            new_shares = old
            detail = "no_entitlement"
            if old > 0 and event.kind is CorporateActionKind.CASH_DIVIDEND:
                receivable_delta = old * event.net_cash_per_share
                state.cash_receivables[event.event_id] = receivable_delta
                detail = "net_cash_receivable_recognized"
            elif old > 0 and event.kind is CorporateActionKind.SHARE_DISTRIBUTION:
                added = _whole_shares(old * event.share_ratio, event_id=event.event_id)
                state.pending_shares[event.event_id] = (event.symbol, added)
                new_shares = old + added
                detail = "share_entitlement_recognized"
            elif old > 0 and event.kind is CorporateActionKind.SYMBOL_EXCHANGE:
                destination = str(event.destination_symbol)
                exchanged = _whole_shares(old * event.share_ratio, event_id=event.event_id)
                conflicts = [
                    pending_id
                    for pending_id, (pending_symbol, _) in state.pending_shares.items()
                    if pending_symbol == event.symbol and pending_id != event.event_id
                ]
                if conflicts:
                    raise CorporateActionError(
                        f"cannot exchange {event.symbol!r} while pending entitlements exist: {conflicts}"
                    )
                state.shares.pop(event.symbol, None)
                state.locked_shares.pop(event.symbol, None)
                state.pending_shares[event.event_id] = (destination, exchanged)
                new_shares = 0
                detail = f"exchange_receivable_recognized:{destination}"
            elif old > 0 and event.kind is CorporateActionKind.CASH_OUT:
                conflicts = [
                    pending_id
                    for pending_id, (pending_symbol, _) in state.pending_shares.items()
                    if pending_symbol == event.symbol
                ]
                if conflicts:
                    raise CorporateActionError(
                        f"cannot cash out {event.symbol!r} while pending entitlements exist: {conflicts}"
                    )
                receivable_delta = old * event.net_cash_per_share
                state.shares.pop(event.symbol, None)
                state.locked_shares.pop(event.symbol, None)
                state.cash_receivables[event.event_id] = receivable_delta
                new_shares = 0
                detail = "terminal_cash_receivable_recognized"
            state.applied_corporate_action_steps.add(marker)
            applications.append(
                CorporateActionApplication(
                    str(day.date()),
                    event.event_id,
                    event.symbol,
                    event.kind.value,
                    "effective",
                    old,
                    new_shares,
                    cash_delta,
                    receivable_delta,
                    detail,
                )
            )

        for event in self._settlement.get(day, ()):
            marker = f"settlement:{event.event_id}"
            if marker in state.applied_corporate_action_steps:
                continue
            cash_delta = 0.0
            old = state.total_shares(event.symbol)
            new = old
            detail = "nothing_to_settle"
            if event.kind in {CorporateActionKind.CASH_DIVIDEND, CorporateActionKind.CASH_OUT}:
                cash_delta = float(state.cash_receivables.pop(event.event_id, 0.0))
                state.cash += cash_delta
                detail = "cash_receivable_settled"
            elif event.kind in {
                CorporateActionKind.SHARE_DISTRIBUTION,
                CorporateActionKind.SYMBOL_EXCHANGE,
            }:
                pending = state.pending_shares.pop(event.event_id, None)
                if pending is not None:
                    destination, quantity = pending
                    state.shares[destination] = int(state.shares.get(destination, 0)) + int(quantity)
                    new = state.total_shares(event.symbol)
                    detail = f"shares_became_tradable:{destination}:{quantity}"
            elif event.kind is CorporateActionKind.RIGHTS_OFFERING:
                # The effective phase above always fails before reaching this.
                raise UnsupportedCorporateActionError(
                    f"rights offering {event.event_id!r} has no subscription decision"
                )
            state.applied_corporate_action_steps.add(marker)
            applications.append(
                CorporateActionApplication(
                    str(day.date()),
                    event.event_id,
                    event.symbol,
                    event.kind.value,
                    "settlement",
                    old,
                    new,
                    cash_delta,
                    -cash_delta,
                    detail,
                )
            )
        return tuple(applications)


__all__ = [
    "CorporateActionApplication",
    "CorporateActionCoverageError",
    "CorporateActionError",
    "CorporateActionEvent",
    "CorporateActionKind",
    "CorporateActionLedger",
    "UnsupportedCorporateActionError",
]
