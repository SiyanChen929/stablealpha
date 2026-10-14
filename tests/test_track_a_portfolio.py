import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.track_a_portfolio import (
    run_paper_top50_drop5,
    top50_drop5_targets,
)
from cogalpha_plus.track_a_report import _inception_anchored_chart_data


def _prediction(days=8, names=60):
    dates = pd.bdate_range("2021-01-04", periods=days)
    symbols = [f"s{i:02d}" for i in range(names)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    values = []
    for day in range(days):
        values.extend(np.arange(names, dtype=float) + day * np.arange(names) / names)
    return pd.Series(values, index=index, name="alpha"), dates, symbols


def test_top50_drop5_changes_no_more_than_five_names_per_day():
    prediction, dates, symbols = _prediction()
    # Force a complete rank reversal after the first signal date.
    second_day = prediction.index.get_level_values("date") == dates[1]
    prediction.iloc[np.flatnonzero(second_day)] = -prediction.iloc[
        np.flatnonzero(second_day)
    ].to_numpy()
    targets = top50_drop5_targets(prediction)
    previous = set(targets[dates[0]])
    current = set(targets[dates[1]])
    assert len(previous) == len(current) == 50
    assert len(previous - current) == 5
    assert len(current - previous) == 5
    assert all(len(names) == 50 for names in targets.values())


def test_paper_portfolio_uses_next_open_costs_and_csi300_benchmark():
    prediction, signal_dates, symbols = _prediction(days=6)
    calendar = pd.bdate_range(signal_dates[0], periods=8)
    index = pd.MultiIndex.from_product([calendar, symbols], names=["date", "symbol"])
    opens = pd.Series(
        [100.0 * (1.001 ** day) for day in range(len(calendar)) for _ in symbols],
        index=index,
        name="adj_open",
    )
    benchmark = pd.Series(
        [4000.0 * (1.0005 ** day) for day in range(len(calendar))],
        index=calendar,
    )
    result = run_paper_top50_drop5(
        prediction,
        adjusted_open=opens,
        trading_calendar=calendar,
        benchmark_open=benchmark,
    )
    assert result.daily.index[0] == calendar[1]
    assert result.daily.iloc[0]["fees"] == 500.0
    assert (result.daily["positions"] == 50).all()
    assert result.statistics["top_k"] == 50
    assert result.statistics["drop_n"] == 5
    assert result.statistics["annualized_excess_return"] > 0
    assert result.statistics["accounting_scope"].endswith("research_proxy")
    assert np.prod(1.0 + result.daily["net_return"]) == pytest.approx(
        result.statistics["final_nav"] / result.statistics["initial_capital"],
        abs=1e-14,
    )
    assert np.prod(1.0 + result.daily["excess_return"]) == pytest.approx(
        result.daily["excess_nav"].iloc[-1], abs=1e-14
    )


def test_paper_portfolio_does_not_rebalance_survivors_when_targets_are_unchanged():
    prediction, signal_dates, symbols = _prediction(days=4)
    calendar = pd.bdate_range(signal_dates[0], periods=6)
    index = pd.MultiIndex.from_product([calendar, symbols], names=["date", "symbol"])
    opens = pd.Series(
        [
            100.0 * ((1.0 + 0.0002 * position) ** day)
            for day in range(len(calendar))
            for position in range(len(symbols))
        ],
        index=index,
        name="adj_open",
    )
    benchmark = pd.Series(4000.0, index=calendar)
    result = run_paper_top50_drop5(
        prediction,
        adjusted_open=opens,
        trading_calendar=calendar,
        benchmark_open=benchmark,
    )
    assert result.daily.iloc[0]["fees"] == 500.0
    assert (result.daily.iloc[1:]["fees"] == 0.0).all()
    assert (result.daily.iloc[1:]["two_way_turnover"] == 0.0).all()


def test_flat_portfolio_reconciles_inception_fee_across_returns_risk_and_chart():
    prediction, signal_dates, symbols = _prediction(days=6)
    calendar = pd.bdate_range(signal_dates[0], periods=8)
    index = pd.MultiIndex.from_product(
        [calendar, symbols], names=["date", "symbol"]
    )
    opens = pd.Series(100.0, index=index, name="adj_open")
    benchmark = pd.Series(4000.0, index=calendar)

    result = run_paper_top50_drop5(
        prediction,
        adjusted_open=opens,
        trading_calendar=calendar,
        benchmark_open=benchmark,
    )

    daily = result.daily
    stats = result.statistics
    expected_ratio = 999_500.0 / 1_000_000.0
    holding_periods = len(daily) - 1
    assert daily.iloc[0]["fees"] == 500.0
    assert np.isnan(daily.iloc[0]["net_return"])
    assert np.isnan(daily.iloc[0]["excess_return"])
    assert daily.iloc[1]["net_return"] == pytest.approx(-0.0005)
    assert daily.iloc[1]["excess_return"] == pytest.approx(-0.0005)
    assert (daily.iloc[2:]["net_return"] == 0.0).all()
    assert np.prod(1.0 + daily["net_return"].dropna()) == pytest.approx(
        expected_ratio
    )
    assert np.prod(1.0 + daily["excess_return"].dropna()) == pytest.approx(
        expected_ratio
    )
    assert stats["final_nav"] == pytest.approx(999_500.0)
    assert stats["total_return"] == pytest.approx(-0.0005)
    assert stats["annualized_return"] == pytest.approx(
        expected_ratio ** (252.0 / holding_periods) - 1.0
    )
    assert stats["annualized_excess_return"] == pytest.approx(
        expected_ratio ** (252.0 / holding_periods) - 1.0
    )
    assert stats["maximum_drawdown"] == pytest.approx(-0.0005)
    assert stats["maximum_excess_drawdown"] == pytest.approx(-0.0005)
    assert stats["return_observations"] == holding_periods
    assert stats["annualization_holding_periods"] == holding_periods
    assert stats["inception_cost_return"] == pytest.approx(-0.0005)
    periodic_excess = daily["excess_return"].dropna()
    expected_std = float(periodic_excess.std(ddof=1))
    assert stats["tracking_error"] == pytest.approx(expected_std * np.sqrt(252.0))
    assert stats["information_ratio"] == pytest.approx(
        float(periodic_excess.mean() / expected_std * np.sqrt(252.0))
    )
    assert stats["return_observation_scope"] == (
        "inception_cost_rolled_into_first_open_to_open_holding_period"
    )

    chart = _inception_anchored_chart_data(
        daily, initial_capital=float(stats["initial_capital"])
    )
    assert chart.iloc[0].to_dict() == {
        "nav": 1.0,
        "benchmark_nav": 1.0,
        "excess_nav": 1.0,
    }
    assert chart.iloc[1]["nav"] == pytest.approx(expected_ratio)
    assert chart.iloc[1]["excess_nav"] == pytest.approx(expected_ratio)
    assert chart.iloc[-1]["nav"] == pytest.approx(
        stats["final_nav"] / stats["initial_capital"]
    )


def test_paper_portfolio_rejects_invalid_benchmark_open_at_execution():
    prediction, signal_dates, symbols = _prediction(days=4)
    calendar = pd.bdate_range(signal_dates[0], periods=6)
    index = pd.MultiIndex.from_product(
        [calendar, symbols], names=["date", "symbol"]
    )
    opens = pd.Series(100.0, index=index, name="adj_open")
    benchmark = pd.Series(4000.0, index=calendar)
    benchmark.loc[calendar[2]] = np.nan

    with pytest.raises(ValueError, match="non-finite or non-positive"):
        run_paper_top50_drop5(
            prediction,
            adjusted_open=opens,
            trading_calendar=calendar,
            benchmark_open=benchmark,
        )
