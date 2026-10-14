"""A-share target-to-fill simulator with explicit, date-versioned assumptions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeeRegime:
    """User-supplied historical fee schedule; no regime is called 'current'."""

    start: pd.Timestamp
    end: pd.Timestamp | None
    commission_rate: float
    minimum_commission: float
    stamp_tax_sell: float
    transfer_fee_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", pd.Timestamp(self.start))
        if self.end is not None:
            object.__setattr__(self, "end", pd.Timestamp(self.end))
            if self.end < self.start:
                raise ValueError("fee regime end precedes start")
        values = (self.commission_rate, self.minimum_commission, self.stamp_tax_sell, self.transfer_fee_rate)
        if any(value < 0 for value in values):
            raise ValueError("fees cannot be negative")

    def contains(self, date: pd.Timestamp) -> bool:
        day = pd.Timestamp(date)
        return day >= self.start and (self.end is None or day <= self.end)


@dataclass(frozen=True)
class ExecutionRules:
    lot_size: int = 100
    max_participation: float = 0.05
    slippage_bps: float = 5.0
    impact_coefficient: float = 0.001
    prohibit_st_buys: bool = True
    minimum_listed_days: int = 250

    def __post_init__(self) -> None:
        if self.lot_size < 1:
            raise ValueError("lot_size must be positive")
        if not (0 < self.max_participation <= 1):
            raise ValueError("max_participation must be in (0, 1]")
        if self.slippage_bps < 0 or self.impact_coefficient < 0:
            raise ValueError("slippage and impact cannot be negative")


@dataclass
class PortfolioState:
    cash: float
    shares: dict[str, int] = field(default_factory=dict)
    locked_shares: dict[str, int] = field(default_factory=dict)
    # Corporate-action assets are explicit balance-sheet items.  A receivable
    # remains part of NAV until paid; pending shares remain economic exposure
    # but cannot be sold until the sourced settlement/tradability date.
    cash_receivables: dict[str, float] = field(default_factory=dict)
    pending_shares: dict[str, tuple[str, int]] = field(default_factory=dict)
    applied_corporate_action_steps: set[str] = field(default_factory=set)
    last_execution_date: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.cash < 0:
            raise ValueError("cash cannot be negative")
        if any(value < 0 or int(value) != value for value in self.shares.values()):
            raise ValueError("shares must be non-negative integers")
        if any(value < 0 or int(value) != value for value in self.locked_shares.values()):
            raise ValueError("locked shares must be non-negative integers")
        if any(not np.isfinite(value) or value < 0 for value in self.cash_receivables.values()):
            raise ValueError("cash receivables must be finite and non-negative")
        if any(
            not str(symbol).strip() or quantity < 0 or int(quantity) != quantity
            for symbol, quantity in self.pending_shares.values()
        ):
            raise ValueError("pending shares must contain a symbol and non-negative integer quantity")

    @property
    def receivable_cash(self) -> float:
        return float(sum(self.cash_receivables.values()))

    def total_shares(self, symbol: str) -> int:
        name = str(symbol)
        pending = sum(
            int(quantity)
            for pending_symbol, quantity in self.pending_shares.values()
            if pending_symbol == name
        )
        return int(self.shares.get(name, 0)) + pending

    def share_symbols(self) -> set[str]:
        return set(self.shares) | {str(symbol) for symbol, _ in self.pending_shares.values()}

    def start_session(self, date: pd.Timestamp) -> None:
        day = pd.Timestamp(date)
        if self.last_execution_date is not None and day < self.last_execution_date:
            raise ValueError("execution dates must be non-decreasing")
        if self.last_execution_date is None or day > self.last_execution_date:
            self.locked_shares.clear()
        self.last_execution_date = day


@dataclass(frozen=True)
class Fill:
    date: str
    symbol: str
    side: str
    requested_shares: int
    filled_shares: int
    price: float | None
    notional: float
    fees: float
    status: str
    reason: str
    participation: float


@dataclass(frozen=True)
class ExecutionReport:
    date: str
    nav_before: float
    nav_after_at_execution_prices: float
    cash_after: float
    fills: tuple[Fill, ...]
    target_weights: Mapping[str, float]
    realized_weights: Mapping[str, float]
    total_fees: float
    total_notional: float


@dataclass(frozen=True)
class StateWeightSnapshot:
    """Weights reconstructed from booked assets, never from the prior target."""

    risky_weights: pd.Series
    cash_weight: float
    receivable_weight: float
    nav: float


class ChinaAExecutionSimulator:
    """Execute target weights using canonical raw prices and market frictions."""

    RAW_PRICE_FIELDS = frozenset({"open", "vwap", "close"})
    REQUIRED_MARKET_FIELDS = frozenset(
        {"vwap", "close", "volume", "tradable", "limit_up", "limit_down", "is_st", "listed_days"}
    )

    def __init__(
        self,
        fee_regimes: Sequence[FeeRegime],
        rules: ExecutionRules | None = None,
        *,
        execution_price_field: str = "vwap",
    ) -> None:
        if not fee_regimes:
            raise ValueError("at least one explicit historical fee regime is required")
        if execution_price_field not in {"open", "vwap"}:
            raise ValueError("execution_price_field must be open or vwap")
        self.fee_regimes = tuple(sorted(fee_regimes, key=lambda item: item.start))
        self.rules = rules or ExecutionRules()
        self.execution_price_field = execution_price_field

    def _fees(self, date: pd.Timestamp) -> FeeRegime:
        matches = [regime for regime in self.fee_regimes if regime.contains(date)]
        if len(matches) != 1:
            raise ValueError(f"execution date {pd.Timestamp(date).date()} matches {len(matches)} fee regimes")
        return matches[0]

    def _validate_market(self, market: pd.DataFrame) -> pd.DataFrame:
        required = self.REQUIRED_MARKET_FIELDS | {self.execution_price_field}
        missing = required.difference(market.columns)
        if missing:
            raise ValueError(f"market snapshot missing execution fields: {sorted(missing)}")
        if market.index.has_duplicates:
            raise ValueError("market symbol index must be unique")
        # Drop every research-only column at the execution boundary.  In
        # particular, adj_close/adj_vwap can never be selected downstream.
        output = market.loc[:, sorted(required)].copy()
        output.index = output.index.astype(str)
        numeric = tuple(
            dict.fromkeys(
                ("vwap", "close", self.execution_price_field, "volume", "listed_days")
            )
        )
        for name in numeric:
            output[name] = pd.to_numeric(output[name], errors="coerce")
        price_fields = list(dict.fromkeys(("vwap", "close", self.execution_price_field)))
        if output[[*price_fields, "volume"]].isna().any().any():
            raise ValueError("market prices/volume contain missing values")
        if (output[price_fields] <= 0).any().any() or (output["volume"] < 0).any():
            raise ValueError("market prices must be positive and volume non-negative")
        return output

    def _execution_price(self, base: float, side: str, participation: float) -> float:
        direction = 1.0 if side == "buy" else -1.0
        slippage = self.rules.slippage_bps / 10_000.0
        impact = self.rules.impact_coefficient * math.sqrt(max(participation, 0.0))
        return float(base * (1.0 + direction * (slippage + impact)))

    @staticmethod
    def _transaction_fees(regime: FeeRegime, side: str, notional: float) -> float:
        if notional <= 0:
            return 0.0
        commission = max(regime.minimum_commission, regime.commission_rate * notional)
        sell_tax = regime.stamp_tax_sell * notional if side == "sell" else 0.0
        transfer = regime.transfer_fee_rate * notional
        return float(commission + sell_tax + transfer)

    def mark_to_market(self, state: PortfolioState, market: pd.DataFrame, *, price_field: str = "close") -> float:
        if price_field not in self.RAW_PRICE_FIELDS:
            raise ValueError(
                f"execution only accepts canonical raw price fields: {sorted(self.RAW_PRICE_FIELDS)}"
            )
        snapshot = self._validate_market(market)
        value = state.cash + state.receivable_cash
        for symbol in state.share_symbols():
            if symbol not in snapshot.index:
                raise ValueError(f"held symbol {symbol!r} missing from market snapshot")
            value += state.total_shares(symbol) * float(snapshot.loc[symbol, price_field])
        return float(value)

    def state_weights(
        self,
        state: PortfolioState,
        market: pd.DataFrame,
        *,
        price_field: str = "close",
    ) -> StateWeightSnapshot:
        """Reconstruct risky and cash-like weights from the actual ledger."""

        if price_field not in self.RAW_PRICE_FIELDS:
            raise ValueError(
                f"execution only accepts canonical raw price fields: {sorted(self.RAW_PRICE_FIELDS)}"
            )
        snapshot = self._validate_market(market)
        nav = self.mark_to_market(state, snapshot, price_field=price_field)
        if nav <= 0:
            raise ValueError("portfolio NAV must be positive to compute state weights")
        values: dict[str, float] = {}
        for symbol in sorted(state.share_symbols()):
            if symbol not in snapshot.index:
                raise ValueError(f"held symbol {symbol!r} missing from market snapshot")
            values[symbol] = state.total_shares(symbol) * float(snapshot.loc[symbol, price_field])
        risky = pd.Series(values, dtype=float, name="previous_weight").div(nav)
        risky.index.name = "symbol"
        return StateWeightSnapshot(
            risky_weights=risky,
            cash_weight=float(state.cash / nav),
            receivable_weight=float(state.receivable_cash / nav),
            nav=float(nav),
        )

    def execute(
        self,
        date: pd.Timestamp | str,
        target_weights: pd.Series,
        market: pd.DataFrame,
        state: PortfolioState,
    ) -> ExecutionReport:
        day = pd.Timestamp(date)
        state.start_session(day)
        snapshot = self._validate_market(market)
        regime = self._fees(day)
        targets = pd.to_numeric(target_weights, errors="coerce").fillna(0.0)
        targets.index = targets.index.astype(str)
        if targets.index.has_duplicates or (targets < -1e-12).any() or targets.sum() > 1.000001:
            raise ValueError("target weights must be unique, non-negative and sum to at most one")
        universe = sorted(set(targets.index) | state.share_symbols())
        missing = set(universe).difference(snapshot.index)
        if missing:
            raise ValueError(f"target/held symbols missing from market snapshot: {sorted(missing)}")
        execution_field = self.execution_price_field
        nav_before = self.mark_to_market(state, snapshot, price_field=execution_field)
        desired: dict[str, int] = {}
        for symbol in universe:
            weight = float(targets.get(symbol, 0.0))
            raw = nav_before * weight / float(snapshot.loc[symbol, execution_field])
            desired[symbol] = int(max(0, math.floor(raw / self.rules.lot_size) * self.rules.lot_size))

        fills: list[Fill] = []

        def blocked(symbol: str, side: str) -> str | None:
            row = snapshot.loc[symbol]
            if not bool(row["tradable"]):
                return "suspended_or_not_tradable"
            if side == "buy" and bool(row["limit_up"]):
                return "limit_up_buy_blocked"
            if side == "sell" and bool(row["limit_down"]):
                return "limit_down_sell_blocked"
            if side == "buy" and self.rules.prohibit_st_buys and bool(row["is_st"]):
                return "st_buy_blocked"
            if side == "buy" and float(row["listed_days"]) < self.rules.minimum_listed_days:
                return "listing_age_buy_blocked"
            return None

        # Sells release cash first. T+1-locked shares and participation are hard caps.
        for symbol in universe:
            current = int(state.shares.get(symbol, 0))
            beneficial = state.total_shares(symbol)
            pending = beneficial - current
            request = max(0, beneficial - desired[symbol])
            if request == 0:
                continue
            reason = blocked(symbol, "sell")
            if reason:
                fills.append(Fill(str(day.date()), symbol, "sell", request, 0, None, 0.0, 0.0, "blocked", reason, 0.0))
                continue
            locked = int(state.locked_shares.get(symbol, 0))
            available = max(0, current - locked)
            volume_cap = int(math.floor(float(snapshot.loc[symbol, "volume"]) * self.rules.max_participation / self.rules.lot_size) * self.rules.lot_size)
            cap = min(request, available, volume_cap)
            # Buys are always board-lot rounded.  On sells, an existing odd-lot
            # tail may be liquidated in one order, but a partial odd-lot sale is
            # never invented.  Participation remains a hard cap.
            if request == current and cap >= current:
                quantity = current
            else:
                quantity = int(math.floor(cap / self.rules.lot_size) * self.rules.lot_size)
            if quantity <= 0:
                if pending > 0 and current == 0:
                    reason = "corporate_action_shares_pending"
                elif locked > 0 and available < min(request, current):
                    reason = "t_plus_one_locked"
                elif 0 < cap < self.rules.lot_size and request != current:
                    reason = "odd_lot_partial_sell_not_permitted"
                else:
                    reason = "participation_cap"
                fills.append(Fill(str(day.date()), symbol, "sell", request, 0, None, 0.0, 0.0, "blocked", reason, 0.0))
                continue
            participation = quantity / max(float(snapshot.loc[symbol, "volume"]), 1.0)
            price = self._execution_price(float(snapshot.loc[symbol, execution_field]), "sell", participation)
            notional = quantity * price
            fees = self._transaction_fees(regime, "sell", notional)
            state.cash += notional - fees
            state.shares[symbol] = current - quantity
            if state.shares[symbol] == 0:
                state.shares.pop(symbol)
            status = "filled" if quantity == request else "partial"
            parts = []
            if quantity < min(request, available):
                parts.append("participation_cap")
            if locked > 0 and available < min(request, current):
                parts.append("t_plus_one_locked")
            if pending > 0 and request > current:
                parts.append("corporate_action_shares_pending")
            fills.append(Fill(str(day.date()), symbol, "sell", request, quantity, price, notional, fees, status, "+".join(parts), participation))

        # Buys use remaining cash and become unavailable to sell until a later date.
        buy_requests = []
        for symbol in universe:
            beneficial = state.total_shares(symbol)
            request = max(0, desired[symbol] - beneficial)
            if request:
                buy_requests.append((float(targets.get(symbol, 0.0)), symbol, request))
        for _, symbol, request in sorted(buy_requests, reverse=True):
            reason = blocked(symbol, "buy")
            if reason:
                fills.append(Fill(str(day.date()), symbol, "buy", request, 0, None, 0.0, 0.0, "blocked", reason, 0.0))
                continue
            volume_cap = int(math.floor(float(snapshot.loc[symbol, "volume"]) * self.rules.max_participation / self.rules.lot_size) * self.rules.lot_size)
            quantity = min(request, volume_cap)
            quantity = int(math.floor(quantity / self.rules.lot_size) * self.rules.lot_size)
            if quantity <= 0:
                fills.append(Fill(str(day.date()), symbol, "buy", request, 0, None, 0.0, 0.0, "blocked", "participation_cap", 0.0))
                continue
            # Impact depends on final quantity. If cash is short, recompute after
            # reducing to affordable lots; never allow negative cash.
            participation = quantity / max(float(snapshot.loc[symbol, "volume"]), 1.0)
            price = self._execution_price(float(snapshot.loc[symbol, execution_field]), "buy", participation)
            affordable = int(
                math.floor(max(state.cash - regime.minimum_commission, 0.0) / price / self.rules.lot_size)
                * self.rules.lot_size
            )
            quantity = min(quantity, affordable)
            while quantity > 0:
                participation = quantity / max(float(snapshot.loc[symbol, "volume"]), 1.0)
                price = self._execution_price(float(snapshot.loc[symbol, execution_field]), "buy", participation)
                notional = quantity * price
                fees = self._transaction_fees(regime, "buy", notional)
                if notional + fees <= state.cash + 1e-9:
                    break
                quantity -= self.rules.lot_size
            if quantity <= 0:
                fills.append(Fill(str(day.date()), symbol, "buy", request, 0, None, 0.0, 0.0, "blocked", "insufficient_cash", 0.0))
                continue
            participation = quantity / max(float(snapshot.loc[symbol, "volume"]), 1.0)
            price = self._execution_price(float(snapshot.loc[symbol, execution_field]), "buy", participation)
            notional = quantity * price
            fees = self._transaction_fees(regime, "buy", notional)
            state.cash -= notional + fees
            state.shares[symbol] = int(state.shares.get(symbol, 0)) + quantity
            state.locked_shares[symbol] = int(state.locked_shares.get(symbol, 0)) + quantity
            parts = []
            if quantity < min(request, volume_cap):
                parts.append("cash_cap")
            if volume_cap < request:
                parts.append("participation_cap")
            fills.append(
                Fill(
                    str(day.date()),
                    symbol,
                    "buy",
                    request,
                    quantity,
                    price,
                    notional,
                    fees,
                    "filled" if quantity == request else "partial",
                    "+".join(parts),
                    participation,
                )
            )

        if state.cash < -1e-6:
            raise RuntimeError("execution produced negative cash")
        nav_after = state.cash + state.receivable_cash + sum(
            state.total_shares(symbol) * float(snapshot.loc[symbol, execution_field])
            for symbol in state.share_symbols()
        )
        realized = {
            symbol: state.total_shares(symbol) * float(snapshot.loc[symbol, execution_field]) / nav_after
            for symbol in state.share_symbols()
        }
        return ExecutionReport(
            date=str(day.date()),
            nav_before=nav_before,
            nav_after_at_execution_prices=float(nav_after),
            cash_after=float(state.cash),
            fills=tuple(fills),
            target_weights={str(symbol): float(value) for symbol, value in targets.items()},
            realized_weights=realized,
            total_fees=float(sum(fill.fees for fill in fills)),
            total_notional=float(sum(fill.notional for fill in fills)),
        )


__all__ = [
    "ChinaAExecutionSimulator",
    "ExecutionReport",
    "ExecutionRules",
    "FeeRegime",
    "Fill",
    "PortfolioState",
    "StateWeightSnapshot",
]
