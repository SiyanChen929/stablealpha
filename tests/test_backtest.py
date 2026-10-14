import pandas as pd

from cogalpha_plus.backtest import EventDrivenBacktester
from cogalpha_plus.execution import ChinaAExecutionSimulator, ExecutionRules, FeeRegime


def panel(limit_on_execution=False):
    dates = pd.bdate_range("2024-01-02", periods=6, name="date")
    names = pd.Index(["A", "B"], name="symbol")
    index = pd.MultiIndex.from_product([dates, names])
    a = [10.0, 10.0, 10.5, 11.0, 11.5, 12.0]
    b = [20.0] * 6
    close = [value for pair in zip(a, b) for value in pair]
    frame = pd.DataFrame(
        {
            "vwap": close,
            "close": close,
            "volume": 1_000_000.0,
            "tradable": True,
            "limit_up": False,
            "limit_down": False,
            "is_st": False,
            "listed_days": 500,
        },
        index=index,
    )
    if limit_on_execution:
        frame.loc[(dates[1], "A"), "limit_up"] = True
    return frame, dates


def simulator():
    return ChinaAExecutionSimulator(
        [FeeRegime(pd.Timestamp("2020-01-01"), None, 0.0003, 5.0, 0.001)],
        ExecutionRules(slippage_bps=0.0, impact_coefficient=0.0),
    )


def test_signal_executes_next_day_and_accounting_includes_cost():
    market, dates = panel()
    result = EventDrivenBacktester(simulator()).run(
        market,
        {dates[0]: pd.Series({"A": 0.95})},
        initial_cash=100_000.0,
    )
    assert result.signal_to_execution[str(dates[0].date())] == str(dates[1].date())
    assert result.daily.loc[dates[0], "trade_notional"] == 0.0
    assert result.daily.loc[dates[1], "trade_notional"] > 0.0
    assert result.statistics.total_fees > 0.0
    assert result.statistics.total_return > 0.0


def test_unfilled_limit_order_remains_visible_in_results():
    market, dates = panel(limit_on_execution=True)
    result = EventDrivenBacktester(simulator()).run(
        market,
        {dates[0]: pd.Series({"A": 0.95})},
        initial_cash=100_000.0,
    )
    assert result.statistics.blocked_order_rate == 1.0
    assert result.fills[0].reason == "limit_up_buy_blocked"
    assert result.final_state.shares == {}

