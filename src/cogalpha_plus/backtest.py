"""Event-driven portfolio accounting around next-session execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .corporate_actions import (
    CorporateActionApplication,
    CorporateActionCoverageError,
    CorporateActionLedger,
)
from .execution import ChinaAExecutionSimulator, ExecutionReport, Fill, PortfolioState
from .execution_tape import UnionExecutionTape


@dataclass(frozen=True)
class BacktestStatistics:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_zero_rate: float
    max_drawdown: float
    total_fees: float
    total_notional: float
    average_daily_turnover: float
    blocked_order_rate: float


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    executions: tuple[ExecutionReport, ...]
    fills: tuple[Fill, ...]
    signal_to_execution: Mapping[str, str]
    final_state: PortfolioState
    statistics: BacktestStatistics
    corporate_action_applications: tuple[CorporateActionApplication, ...] = ()


class EventDrivenBacktester:
    """Signals observed on t can only produce orders on the next trading date."""

    def __init__(
        self,
        execution_simulator: ChinaAExecutionSimulator,
        *,
        corporate_action_ledger: CorporateActionLedger | None = None,
        adjustment_factor_tolerance: float = 5e-4,
    ) -> None:
        if adjustment_factor_tolerance <= 0:
            raise ValueError("adjustment_factor_tolerance must be positive")
        self.execution_simulator = execution_simulator
        self.corporate_action_ledger = corporate_action_ledger or CorporateActionLedger()
        self.adjustment_factor_tolerance = float(adjustment_factor_tolerance)

    def run(
        self,
        market_panel: pd.DataFrame,
        target_weights_by_signal_date: Mapping[pd.Timestamp | str, pd.Series],
        *,
        initial_cash: float,
        research_membership: pd.DataFrame | pd.Series | None = None,
    ) -> BacktestResult:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        tape = UnionExecutionTape.build(
            market_panel,
            research_membership=research_membership,
            terminal_dates=self.corporate_action_ledger.terminal_dates(),
        )
        dates = tape.dates
        self.corporate_action_ledger.validate_calendar(dates)
        position = {pd.Timestamp(date): i for i, date in enumerate(dates)}
        scheduled: dict[pd.Timestamp, pd.Series] = {}
        mapping: dict[str, str] = {}
        for raw_date, target in target_weights_by_signal_date.items():
            signal_date = pd.Timestamp(raw_date)
            if signal_date not in position:
                raise ValueError(f"signal date {signal_date.date()} is not a market trading date")
            index = position[signal_date]
            if index + 1 >= len(dates):
                raise ValueError(f"signal date {signal_date.date()} has no next trading session")
            execution_date = pd.Timestamp(dates[index + 1])
            if execution_date in scheduled:
                raise ValueError(f"multiple target portfolios scheduled for {execution_date.date()}")
            scheduled[execution_date] = target.copy()
            mapping[str(signal_date.date())] = str(execution_date.date())

        state = PortfolioState(cash=float(initial_cash))
        executions: list[ExecutionReport] = []
        corporate_applications: list[CorporateActionApplication] = []
        records: list[dict[str, object]] = []
        previous_nav = float(initial_cash)
        previous_adjustment_factors: dict[str, float] = {}
        for raw_date in dates:
            date = pd.Timestamp(raw_date)
            # Retrieve first without a coverage requirement: a sourced terminal
            # event can remove an old symbol before today's valuation.
            snapshot = tape.snapshot(date)
            state.start_session(date)
            held_before_actions = state.share_symbols()
            if "adj_factor" in snapshot.columns:
                factors = pd.to_numeric(snapshot["adj_factor"], errors="coerce")
                for symbol in sorted(held_before_actions):
                    if symbol not in factors.index:
                        continue
                    current = float(factors.loc[symbol])
                    prior = previous_adjustment_factors.get(symbol)
                    if not np.isfinite(current) or current <= 0:
                        raise CorporateActionCoverageError(
                            f"held symbol {symbol!r} has invalid adj_factor on {date.date()}"
                        )
                    changed = prior is not None and not np.isclose(
                        current,
                        prior,
                        rtol=self.adjustment_factor_tolerance,
                        atol=self.adjustment_factor_tolerance,
                    )
                    if changed and not self.corporate_action_ledger.covers_adjustment(symbol, date):
                        raise CorporateActionCoverageError(
                            f"held symbol {symbol!r} adj_factor changed on {date.date()} "
                            "without a sourced corporate-action event"
                        )
                for symbol, factor in factors.items():
                    value = float(factor)
                    if np.isfinite(value) and value > 0:
                        previous_adjustment_factors[str(symbol)] = value

            applications = self.corporate_action_ledger.apply_opening(date, state)
            corporate_applications.extend(applications)
            required = state.share_symbols()
            if date in scheduled:
                required |= set(scheduled[date].index.astype(str))
            snapshot = tape.snapshot(date, required_symbols=required)
            report: ExecutionReport | None = None
            if date in scheduled:
                report = self.execution_simulator.execute(date, scheduled[date], snapshot, state)
                executions.append(report)
            nav_close = self.execution_simulator.mark_to_market(state, snapshot, price_field="close")
            daily_return = nav_close / previous_nav - 1.0 if previous_nav > 0 else np.nan
            records.append(
                {
                    "date": date,
                    "nav": nav_close,
                    "cash": state.cash,
                    "cash_receivables": state.receivable_cash,
                    "pending_shares": sum(quantity for _, quantity in state.pending_shares.values()),
                    "positions": len(state.share_symbols()),
                    "daily_return": daily_return,
                    "trade_notional": report.total_notional if report else 0.0,
                    "fees": report.total_fees if report else 0.0,
                    "orders": len(report.fills) if report else 0,
                    "blocked_orders": sum(fill.status == "blocked" for fill in report.fills) if report else 0,
                    "corporate_action_steps": len(applications),
                }
            )
            previous_nav = nav_close
        daily = pd.DataFrame.from_records(records).set_index("date")
        returns = daily["daily_return"].iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()
        total_return = float(daily["nav"].iloc[-1] / initial_cash - 1.0)
        years = max((len(daily) - 1) / 252.0, 1 / 252.0)
        annualized = float((daily["nav"].iloc[-1] / initial_cash) ** (1.0 / years) - 1.0)
        volatility = float(returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else np.nan
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 and returns.std(ddof=1) > 0 else np.nan
        running_max = daily["nav"].cummax()
        drawdown = daily["nav"].div(running_max).sub(1.0)
        fills = tuple(fill for report in executions for fill in report.fills)
        total_orders = len(fills)
        blocked = sum(fill.status == "blocked" for fill in fills)
        stats = BacktestStatistics(
            total_return=total_return,
            annualized_return=annualized,
            annualized_volatility=volatility,
            sharpe_zero_rate=sharpe,
            max_drawdown=float(drawdown.min()),
            total_fees=float(daily["fees"].sum()),
            total_notional=float(daily["trade_notional"].sum()),
            average_daily_turnover=float((daily["trade_notional"] / daily["nav"].replace(0, np.nan)).mean()),
            blocked_order_rate=float(blocked / total_orders) if total_orders else 0.0,
        )
        # Return an isolated state snapshot so later simulations cannot mutate
        # already-reported results.
        final_state = PortfolioState(
            cash=state.cash,
            shares=dict(state.shares),
            locked_shares=dict(state.locked_shares),
            cash_receivables=dict(state.cash_receivables),
            pending_shares=dict(state.pending_shares),
            applied_corporate_action_steps=set(state.applied_corporate_action_steps),
            last_execution_date=state.last_execution_date,
        )
        return BacktestResult(
            daily,
            tuple(executions),
            fills,
            mapping,
            final_state,
            stats,
            tuple(corporate_applications),
        )


__all__ = ["BacktestResult", "BacktestStatistics", "EventDrivenBacktester"]
