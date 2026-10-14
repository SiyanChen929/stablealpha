import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.execution import (
    ChinaAExecutionSimulator,
    ExecutionRules,
    FeeRegime,
    PortfolioState,
)


def market():
    names = pd.Index(["A", "B", "C"], name="symbol")
    return pd.DataFrame(
        {
            "vwap": [10.0, 20.0, 30.0],
            "close": [10.1, 19.8, 30.2],
            "volume": [1_000_000, 1_000_000, 1_000_000],
            "tradable": [True, True, True],
            "limit_up": [False, False, False],
            "limit_down": [False, False, False],
            "is_st": [False, False, False],
            "listed_days": [500, 500, 500],
        },
        index=names,
    )


def simulator(participation=0.05):
    return ChinaAExecutionSimulator(
        [FeeRegime(pd.Timestamp("2020-01-01"), None, 0.0003, 5.0, 0.001, 0.00001)],
        ExecutionRules(max_participation=participation, slippage_bps=0.0, impact_coefficient=0.0),
    )


def test_buys_are_lot_rounded_fee_charged_and_t_plus_one_locked():
    state = PortfolioState(cash=100_000.0)
    sim = simulator()
    first = sim.execute("2024-01-02", pd.Series({"A": 0.5, "B": 0.5}), market(), state)
    assert all(fill.filled_shares % 100 == 0 for fill in first.fills)
    assert first.total_fees >= 10.0
    assert state.locked_shares
    second = sim.execute("2024-01-02", pd.Series({"A": 0.0, "B": 0.0}), market(), state)
    assert all(fill.filled_shares == 0 for fill in second.fills)
    assert all(fill.reason == "t_plus_one_locked" for fill in second.fills)
    third = sim.execute("2024-01-03", pd.Series({"A": 0.0, "B": 0.0}), market(), state)
    assert sum(fill.filled_shares for fill in third.fills) > 0
    assert all(fill.fees > 5.0 for fill in third.fills)


def test_limits_suspension_and_participation_are_not_excluded_after_fact():
    snapshot = market()
    snapshot.loc["A", "limit_up"] = True
    snapshot.loc["B", "tradable"] = False
    snapshot.loc["C", "volume"] = 1_000
    state = PortfolioState(cash=100_000.0)
    report = simulator(participation=0.05).execute(
        "2024-01-02",
        pd.Series({"A": 0.3, "B": 0.3, "C": 0.3}),
        snapshot,
        state,
    )
    by_symbol = {fill.symbol: fill for fill in report.fills}
    assert by_symbol["A"].reason == "limit_up_buy_blocked"
    assert by_symbol["B"].reason == "suspended_or_not_tradable"
    # 5% of 1,000 shares is below one 100-share board lot.
    assert by_symbol["C"].reason == "participation_cap"


def test_execution_ignores_adjusted_prices_and_rejects_adjusted_marking():
    snapshot = market()
    snapshot["adj_vwap"] = snapshot["vwap"] * 100.0
    snapshot["adj_close"] = snapshot["close"] * 100.0
    sim = simulator()
    state = PortfolioState(cash=100_000.0)
    report = sim.execute("2024-01-02", pd.Series({"A": 0.5}), snapshot, state)
    fill = next(item for item in report.fills if item.symbol == "A")
    assert fill.price == 10.0

    marked_state = PortfolioState(cash=0.0, shares={"A": 100})
    assert sim.mark_to_market(marked_state, snapshot, price_field="close") == 1_010.0
    with pytest.raises(ValueError, match="canonical raw price fields"):
        sim.mark_to_market(marked_state, snapshot, price_field="adj_close")


def test_track_a_can_execute_at_raw_open_without_using_vwap():
    snapshot = market()
    snapshot["open"] = [11.0, 21.0, 31.0]
    sim = ChinaAExecutionSimulator(
        [FeeRegime(pd.Timestamp("2020-01-01"), None, 0.0005, 5.0, 0.001)],
        ExecutionRules(
            max_participation=1.0,
            slippage_bps=0.0,
            impact_coefficient=0.0,
        ),
        execution_price_field="open",
    )
    report = sim.execute(
        "2024-01-02", pd.Series({"A": 0.5}), snapshot, PortfolioState(100_000.0)
    )
    fill = next(item for item in report.fills if item.symbol == "A")
    assert fill.price == 11.0
    assert fill.fees == max(5.0, fill.notional * 0.0005)


def test_existing_odd_lot_can_be_fully_liquidated_but_buys_remain_board_lots():
    state = PortfolioState(cash=0.0, shares={"A": 150})
    report = simulator().execute("2024-01-02", pd.Series({"A": 0.0}), market(), state)
    fill = next(item for item in report.fills if item.symbol == "A")
    assert fill.filled_shares == 150
    assert state.shares == {}


def test_partial_odd_lot_sale_is_not_invented():
    state = PortfolioState(cash=0.0, shares={"A": 150})
    report = simulator().execute("2024-01-02", pd.Series({"A": 2 / 3}), market(), state)
    fill = next(item for item in report.fills if item.symbol == "A")
    assert fill.filled_shares == 0
    assert fill.reason == "odd_lot_partial_sell_not_permitted"
    assert state.shares["A"] == 150
