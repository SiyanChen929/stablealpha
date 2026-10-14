"""Paper-compatible Top50/drop5 portfolio and benchmark accounting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PaperPortfolioResult:
    targets: Mapping[pd.Timestamp, tuple[str, ...]]
    daily: pd.DataFrame
    statistics: Mapping[str, float | int | str]


def top50_drop5_targets(
    prediction: pd.Series,
    *,
    top_k: int = 50,
    drop_n: int = 5,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    """Build deterministic Qlib-style TopkDropout target name sets."""

    if top_k < 1 or drop_n < 1 or drop_n > top_k:
        raise ValueError("invalid TopkDropout parameters")
    if not isinstance(prediction.index, pd.MultiIndex) or tuple(
        prediction.index.names
    ) != ("date", "symbol"):
        raise ValueError("prediction must use MultiIndex(date, symbol)")
    targets: dict[pd.Timestamp, tuple[str, ...]] = {}
    held: list[str] = []
    for raw_date, values in prediction.groupby(level="date", sort=True):
        day = pd.Timestamp(raw_date)
        scores = pd.to_numeric(values.droplevel("date"), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        ranked = sorted(scores.index.astype(str), key=lambda symbol: (-float(scores.loc[symbol]), symbol))
        if len(ranked) < top_k:
            raise ValueError(f"{day.date()} has fewer than {top_k} finite scores")
        if not held:
            held = ranked[:top_k]
        else:
            rank_position = {symbol: position for position, symbol in enumerate(ranked)}
            sellable = [
                symbol
                for symbol in held
                if rank_position.get(symbol, len(ranked)) >= top_k
            ]
            sellable.sort(
                key=lambda symbol: (
                    -rank_position.get(symbol, len(ranked)),
                    symbol,
                )
            )
            sold = set(sellable[:drop_n])
            retained = [symbol for symbol in held if symbol not in sold]
            slots = top_k - len(retained)
            additions = [symbol for symbol in ranked if symbol not in retained][:slots]
            held = retained + additions
            held.sort(key=lambda symbol: (rank_position.get(symbol, len(ranked)), symbol))
        if len(held) != top_k or len(set(held)) != top_k:
            raise RuntimeError("TopkDropout target does not contain exactly top_k names")
        targets[day] = tuple(held)
    return targets


def _maximum_drawdown(nav: pd.Series, *, initial_value: float) -> float:
    """Return drawdown after anchoring the path at its pre-trade value."""

    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("drawdown initial value must be finite and positive")
    anchored = pd.concat(
        [pd.Series([float(initial_value)], dtype=float), nav.reset_index(drop=True)]
    )
    drawdown = anchored.div(anchored.cummax()).sub(1.0)
    return float(drawdown.min())


def _holding_period_returns(values: pd.Series, *, initial_value: float) -> pd.Series:
    """Build open-to-open returns with inception cost in the first real period.

    The first valuation is immediately after the initial trade, so it is not a
    holding-period observation.  The next valuation is compared with the
    pre-trade initial value, which rolls the inception commission into the
    first actual open-to-open period without inventing a zero-duration day.
    """

    if len(values) < 2:
        raise ValueError("return path requires at least one holding period")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("return initial value must be finite and positive")
    result = values.pct_change()
    result.iloc[0] = np.nan
    result.iloc[1] = float(values.iloc[1] / initial_value - 1.0)
    return result.astype(float)


def _annualized_return(nav: pd.Series) -> float:
    if len(nav) < 2:
        return float("nan")
    years = (len(nav) - 1) / 252.0
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)


def _annualized_from_initial(final_value: float, initial_value: float, periods: int) -> float:
    if periods < 1 or min(final_value, initial_value) <= 0:
        return float("nan")
    return float((final_value / initial_value) ** (252.0 / periods) - 1.0)


def run_paper_top50_drop5(
    prediction: pd.Series,
    *,
    adjusted_open: pd.Series,
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    benchmark_open: pd.Series,
    initial_capital: float = 1_000_000.0,
    open_cost_rate: float = 0.0005,
    close_cost_rate: float = 0.0015,
    minimum_commission: float = 5.0,
    top_k: int = 50,
    drop_n: int = 5,
) -> PaperPortfolioResult:
    """Account at consecutive adjusted opens with explicit one-way costs.

    This is the paper-parity layer. Corporate actions are embedded in the
    frozen adjusted-open series. It does not claim raw-share or board-lot
    executability; the separate event-driven replay owns that stronger claim.
    """

    if initial_capital <= 0 or min(open_cost_rate, close_cost_rate, minimum_commission) < 0:
        raise ValueError("portfolio capital and costs are invalid")
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize()
    position = {pd.Timestamp(date): i for i, date in enumerate(calendar)}
    signal_targets = top50_drop5_targets(
        prediction, top_k=top_k, drop_n=drop_n
    )
    evaluation_end = max(signal_targets)
    execution_targets: dict[pd.Timestamp, tuple[str, ...]] = {}
    for signal_date, names in signal_targets.items():
        if signal_date not in position or position[signal_date] + 1 >= len(calendar):
            raise ValueError("signal date has no next-open execution session")
        execution_date = pd.Timestamp(calendar[position[signal_date] + 1])
        # The published test interval ends at the last signal date. A signal
        # generated on that close would trade outside the reported period, so
        # do not charge its rebalance cost without observing a holding return.
        if execution_date <= evaluation_end:
            execution_targets[execution_date] = names
    execution_dates = pd.DatetimeIndex(sorted(execution_targets))
    if len(execution_dates) < 2:
        raise ValueError("portfolio requires at least two execution dates")

    opens = pd.to_numeric(adjusted_open, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    benchmark = pd.to_numeric(benchmark_open, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).sort_index()
    benchmark_at_execution = benchmark.reindex(execution_dates)
    if benchmark_at_execution.isna().any() or (benchmark_at_execution <= 0).any():
        raise ValueError("CSI300 benchmark open is non-finite or non-positive")
    nav = float(initial_capital)
    benchmark_nav = float(initial_capital)
    holdings_value: dict[str, float] = {}
    cash = float(initial_capital)
    prior_date: pd.Timestamp | None = None
    total_fees = 0.0
    records: list[dict[str, object]] = []
    for execution_date in execution_dates:
        names = execution_targets[execution_date]
        gross_return = 0.0
        benchmark_return = 0.0
        if prior_date is not None:
            prior_nav = nav
            for symbol in tuple(holdings_value):
                try:
                    start = float(opens.loc[(prior_date, symbol)])
                    end = float(opens.loc[(execution_date, symbol)])
                except KeyError as exc:
                    raise ValueError(
                        f"held symbol {symbol} lacks adjusted open through {execution_date.date()}"
                    ) from exc
                if not np.isfinite(start) or not np.isfinite(end) or min(start, end) <= 0:
                    raise ValueError("held adjusted-open path is invalid")
                holdings_value[symbol] *= end / start
            pretrade_nav = float(cash + sum(holdings_value.values()))
            if not np.isfinite(pretrade_nav) or pretrade_nav <= 0:
                raise RuntimeError("portfolio open-to-open growth is invalid")
            gross_return = float(pretrade_nav / prior_nav - 1.0)
            if prior_date not in benchmark.index or execution_date not in benchmark.index:
                raise ValueError("CSI300 benchmark open is missing on an execution date")
            benchmark_return = float(
                benchmark.loc[execution_date] / benchmark.loc[prior_date] - 1.0
            )
            benchmark_nav *= 1.0 + benchmark_return
        else:
            pretrade_nav = nav

        target_set = set(names)
        held_set = set(holdings_value)
        sold_names = sorted(held_set - target_set)
        added_names = [symbol for symbol in names if symbol not in held_set]
        sold_notional = 0.0
        sell_fees = 0.0
        for symbol in sold_names:
            notional = float(holdings_value.pop(symbol))
            fee = max(minimum_commission, notional * close_cost_rate)
            sold_notional += notional
            sell_fees += fee
            cash += notional - fee

        bought_notional = 0.0
        buy_fees = 0.0
        if added_names:
            per_name_budget = cash / len(added_names)
            for symbol in added_names:
                try:
                    price = float(opens.loc[(execution_date, symbol)])
                except KeyError as exc:
                    raise ValueError(
                        f"new holding {symbol} lacks adjusted open on {execution_date.date()}"
                    ) from exc
                if not np.isfinite(price) or price <= 0:
                    raise ValueError("new holding adjusted open is invalid")
                fee = max(minimum_commission, per_name_budget * open_cost_rate)
                invested = per_name_budget - fee
                if invested <= 0:
                    raise RuntimeError("minimum commission exceeds per-name buy budget")
                holdings_value[symbol] = float(invested)
                bought_notional += float(invested)
                buy_fees += float(fee)
            cash = 0.0

        fees = float(buy_fees + sell_fees)
        nav = float(cash + sum(holdings_value.values()))
        if nav <= 0:
            raise RuntimeError("transaction costs exhausted portfolio NAV")
        if set(holdings_value) != target_set or len(holdings_value) != top_k:
            raise RuntimeError("post-trade holdings differ from TopkDropout target")
        total_fees += fees
        buy_turnover = bought_notional / pretrade_nav
        sell_turnover = sold_notional / pretrade_nav
        turnover = float(buy_turnover + sell_turnover)
        records.append(
            {
                "date": execution_date,
                "nav": nav,
                "benchmark_nav": benchmark_nav,
                "gross_return": gross_return,
                "benchmark_return": benchmark_return,
                "fees": fees,
                "one_way_buy_turnover": float(buy_turnover),
                "one_way_sell_turnover": float(sell_turnover),
                "two_way_turnover": turnover,
                "positions": len(names),
            }
        )
        prior_date = execution_date

    daily = pd.DataFrame.from_records(records).set_index("date")
    inception_cost_return = float(daily["nav"].iloc[0] / initial_capital - 1.0)
    # Row zero is an inception valuation rather than a time period. Roll its
    # transaction cost into the first genuine open-to-open observation so the
    # periodic series has the same frequency as IR/TE and still reconciles to
    # total wealth.
    daily["net_return"] = _holding_period_returns(
        daily["nav"], initial_value=initial_capital
    )
    daily["excess_nav"] = daily["nav"].div(daily["benchmark_nav"])
    daily["excess_return"] = _holding_period_returns(
        daily["excess_nav"], initial_value=1.0
    )
    compounded_net = float(np.prod(1.0 + daily["net_return"].dropna()))
    compounded_excess = float(np.prod(1.0 + daily["excess_return"].dropna()))
    terminal_net_ratio = float(daily["nav"].iloc[-1] / initial_capital)
    terminal_excess_ratio = float(daily["excess_nav"].iloc[-1])
    if not np.isclose(compounded_net, terminal_net_ratio, rtol=1e-12, atol=1e-12):
        raise RuntimeError("portfolio periodic returns do not reconcile to terminal NAV")
    if not np.isclose(
        compounded_excess,
        terminal_excess_ratio,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "portfolio periodic excess returns do not reconcile to terminal excess NAV"
        )
    periodic_excess = daily["excess_return"].dropna()
    excess_std = float(periodic_excess.std(ddof=1))
    tracking_error = excess_std * np.sqrt(252.0)
    periods = len(daily) - 1
    statistics: dict[str, float | int | str] = {
        "schema": "track_a_paper_top50_drop5_v1",
        "execution_sessions": int(len(daily)),
        "return_observations": int(periods),
        "annualization_holding_periods": int(periods),
        "inception_cost_return": inception_cost_return,
        "initial_capital": float(initial_capital),
        "final_nav": float(daily["nav"].iloc[-1]),
        "total_return": float(daily["nav"].iloc[-1] / initial_capital - 1.0),
        "annualized_return": _annualized_from_initial(
            float(daily["nav"].iloc[-1]), initial_capital, periods
        ),
        "benchmark_annualized_return": _annualized_return(daily["benchmark_nav"]),
        "annualized_excess_return": _annualized_from_initial(
            float(daily["excess_nav"].iloc[-1]), 1.0, periods
        ),
        "tracking_error": float(tracking_error),
        "information_ratio": (
            float(periodic_excess.mean() / excess_std * np.sqrt(252.0))
            if np.isfinite(excess_std) and excess_std > 0
            else float("nan")
        ),
        "maximum_drawdown": _maximum_drawdown(
            daily["nav"], initial_value=initial_capital
        ),
        "maximum_excess_drawdown": _maximum_drawdown(
            daily["excess_nav"], initial_value=1.0
        ),
        "total_fees": total_fees,
        "average_two_way_turnover": float(daily["two_way_turnover"].mean()),
        "top_k": top_k,
        "drop_n": drop_n,
        "open_cost_rate": open_cost_rate,
        "close_cost_rate": close_cost_rate,
        "minimum_commission": minimum_commission,
        "accounting_scope": "paper_parity_adjusted_open_research_proxy",
        "return_observation_scope": (
            "inception_cost_rolled_into_first_open_to_open_holding_period"
        ),
        "annualization_scope": "execution_sessions_minus_one_open_to_open_holding_periods",
        "evaluation_end": str(evaluation_end.date()),
    }
    return PaperPortfolioResult(signal_targets, daily, statistics)


def load_frozen_csi300_open(
    panel_manifest: Mapping[str, object],
) -> tuple[pd.Series, Mapping[str, object]]:
    record = panel_manifest.get("index_cache_object")
    if not isinstance(record, Mapping):
        raise ValueError("panel manifest lacks CSI300 index cache identity")
    path = Path(str(record.get("object", ""))).resolve()
    expected_hash = str(record.get("parquet_sha256", ""))
    if not path.is_file():
        raise FileNotFoundError("frozen CSI300 index history is missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError("frozen CSI300 index history hash mismatch")
    frame = pd.read_parquet(path)
    if "date" not in frame.columns or "open" not in frame.columns:
        raise ValueError("CSI300 index history lacks date/open")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"])).normalize()
    result = pd.Series(
        pd.to_numeric(frame["open"], errors="coerce").to_numpy(),
        index=dates,
        name="csi300_open",
    ).sort_index()
    if result.index.has_duplicates or result.isna().any() or (result <= 0).any():
        raise ValueError("CSI300 benchmark open series is invalid")
    return result, {
        "provider": "BaoStock",
        "code": "sh.000300",
        "price": "raw_open",
        "parquet_sha256": digest,
        "rows": int(len(result)),
        "first_date": str(result.index.min().date()),
        "last_date": str(result.index.max().date()),
    }


__all__ = [
    "PaperPortfolioResult",
    "load_frozen_csi300_open",
    "run_paper_top50_drop5",
    "top50_drop5_targets",
]
