import pandas as pd
import pytest

from cogalpha_plus.backtest import EventDrivenBacktester
from cogalpha_plus.corporate_actions import (
    CorporateActionCoverageError,
    CorporateActionEvent,
    CorporateActionKind,
    CorporateActionLedger,
    UnsupportedCorporateActionError,
)
from cogalpha_plus.execution import (
    ChinaAExecutionSimulator,
    ExecutionRules,
    FeeRegime,
    PortfolioState,
)


def _simulator():
    return ChinaAExecutionSimulator(
        [FeeRegime(pd.Timestamp("2020-01-01"), None, 0.0, 0.0, 0.0)],
        ExecutionRules(slippage_bps=0.0, impact_coefficient=0.0, minimum_listed_days=0),
    )


def _panel(raw_close, adj_factor):
    dates = pd.bdate_range("2024-01-02", periods=len(raw_close), name="date")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"])
    frame = pd.DataFrame(
        {
            "vwap": raw_close,
            "close": raw_close,
            "volume": 1_000_000,
            "tradable": True,
            "limit_up": False,
            "limit_down": False,
            "is_st": False,
            "listed_days": 500,
            "adj_factor": adj_factor,
        },
        index=index,
    )
    return frame, dates


def test_held_adjustment_factor_step_without_event_fails_closed():
    panel, dates = _panel([10.0, 10.0, 9.0, 9.0], [0.8, 0.8, 1.0, 1.0])
    with pytest.raises(CorporateActionCoverageError, match="without a sourced"):
        EventDrivenBacktester(_simulator()).run(
            panel,
            {dates[0]: pd.Series({"A": 1.0})},
            initial_cash=100_000.0,
        )


def test_explicit_cash_dividend_books_receivable_then_cash_and_preserves_nav():
    panel, dates = _panel([10.0, 10.0, 9.0, 9.0], [0.8, 0.8, 1.0, 1.0])
    event = CorporateActionEvent(
        event_id="A-2024-div",
        kind=CorporateActionKind.CASH_DIVIDEND,
        symbol="A",
        effective_date=dates[2],
        settlement_date=dates[3],
        net_cash_per_share=1.0,
        source="unit-test-explicit-event",
    )
    result = EventDrivenBacktester(
        _simulator(), corporate_action_ledger=CorporateActionLedger([event])
    ).run(
        panel,
        {dates[0]: pd.Series({"A": 1.0})},
        initial_cash=100_000.0,
    )
    assert result.daily.loc[dates[2], "nav"] == pytest.approx(100_000.0)
    assert result.daily.loc[dates[2], "cash_receivables"] == pytest.approx(10_000.0)
    assert result.daily.loc[dates[3], "cash_receivables"] == 0.0
    assert result.final_state.cash == pytest.approx(10_000.0)
    assert [item.phase for item in result.corporate_action_applications] == [
        "effective",
        "settlement",
    ]


def test_explicit_share_distribution_changes_beneficial_and_tradable_shares():
    panel, dates = _panel([10.0, 10.0, 5.0, 5.0], [0.5, 0.5, 1.0, 1.0])
    event = CorporateActionEvent(
        event_id="A-2024-bonus",
        kind="share_distribution",
        symbol="A",
        effective_date=dates[2],
        settlement_date=dates[2],
        share_ratio=1.0,
        source="unit-test-explicit-event",
    )
    result = EventDrivenBacktester(
        _simulator(), corporate_action_ledger=CorporateActionLedger([event])
    ).run(
        panel,
        {dates[0]: pd.Series({"A": 1.0})},
        initial_cash=100_000.0,
    )
    assert result.daily.loc[dates[2], "nav"] == pytest.approx(100_000.0)
    assert result.final_state.shares["A"] == 20_000
    assert result.final_state.pending_shares == {}


def test_rights_offering_is_surfaced_but_never_auto_exercised():
    event = CorporateActionEvent(
        event_id="A-rights",
        kind="rights_offering",
        symbol="A",
        effective_date="2024-01-03",
        settlement_date="2024-01-10",
        share_ratio=0.3,
        subscription_price=4.2,
        source="unit-test-explicit-event",
    )
    state = PortfolioState(cash=100.0, shares={"A": 100})
    with pytest.raises(UnsupportedCorporateActionError, match="subscribe/decline"):
        CorporateActionLedger([event]).apply_opening("2024-01-03", state)


def test_state_weights_include_pending_shares_and_cash_receivables():
    market = pd.DataFrame(
        {
            "vwap": [10.0, 20.0],
            "close": [10.0, 20.0],
            "volume": [1_000_000, 1_000_000],
            "tradable": [True, True],
            "limit_up": [False, False],
            "limit_down": [False, False],
            "is_st": [False, False],
            "listed_days": [500, 500],
        },
        index=pd.Index(["A", "B"], name="symbol"),
    )
    state = PortfolioState(
        cash=100.0,
        shares={"A": 10},
        cash_receivables={"div": 50.0},
        pending_shares={"bonus": ("B", 5)},
    )
    weights = _simulator().state_weights(state, market)
    assert weights.nav == pytest.approx(350.0)
    assert weights.risky_weights.sum() == pytest.approx(200 / 350)
    assert weights.cash_weight == pytest.approx(100 / 350)
    assert weights.receivable_weight == pytest.approx(50 / 350)
