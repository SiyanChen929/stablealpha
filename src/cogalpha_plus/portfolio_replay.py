"""Pre-registered, cost-aware HS300 research portfolio replay.

The module deliberately separates target construction from outcome scoring.
Target construction receives alpha and information available no later than each
signal close.  The scorer receives a frozen target book plus ten-session return
labels afterward.  Results are research-proxy performance, not cash-ledger or
official-index-enhancement claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .manifest import canonical_json, hash_json
from .portfolio import ConstrainedPortfolioOptimizer, PortfolioConstraints
from .validation import build_style_exposures


POLICY_KIND = "hs300_pre_registered_research_portfolio_replay"
REQUIRED_STYLE_COLUMNS = ("beta", "log_mcap", "momentum", "volatility", "liquidity")


@dataclass(frozen=True)
class PortfolioReplayPolicy:
    initial_nav_cny: float
    rebalance_every_sessions: int
    execution_delay_sessions: int
    holding_horizon_sessions: int
    benchmark: str
    primary_cost_bps_one_way: float
    cost_sensitivity_bps_one_way: tuple[float, ...]
    initialization: str
    risk_method: str
    risk_lookback_sessions: int
    risk_minimum_observations: int
    variance_floor: float
    constraints: PortfolioConstraints
    return_basis: str
    benchmark_costed: bool
    cash_execution_authorized: bool
    official_index_enhancement_claim_authorized: bool
    policy_hash: str


@dataclass(frozen=True)
class TargetBook:
    targets: pd.DataFrame
    benchmarks: pd.DataFrame
    diagnostics: pd.DataFrame
    signal_dates: tuple[pd.Timestamp, ...]
    execution_dates: tuple[pd.Timestamp, ...]
    exit_dates: tuple[pd.Timestamp, ...]
    policy_hash: str


@dataclass(frozen=True)
class PortfolioReplayResult:
    periods: pd.DataFrame
    cost_scenarios: pd.DataFrame
    summary: Mapping[str, object]


def _strict_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def load_portfolio_replay_policy(path: Path | str) -> PortfolioReplayPolicy:
    target = Path(path).resolve()
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    root = _strict_mapping(raw, label="portfolio replay policy")
    if set(root) != {
        "schema_version", "kind", "status", "alpha", "portfolio",
        "risk_model", "constraints", "accounting", "limitations",
    }:
        raise ValueError("portfolio replay policy root keys are not exact")
    if (
        root["schema_version"] != 1
        or root["kind"] != POLICY_KIND
        or root["status"] != "frozen_before_historical_replay"
    ):
        raise ValueError("portfolio replay policy identity is invalid")
    alpha = _strict_mapping(root["alpha"], label="alpha policy")
    if alpha != {
        "backend": "lightgbm",
        "use_post_model_industry_style_neutralization": True,
    }:
        raise ValueError("portfolio alpha policy is not the unique primary composite")
    portfolio = _strict_mapping(root["portfolio"], label="portfolio policy")
    risk = _strict_mapping(root["risk_model"], label="risk model")
    constraint_values = _strict_mapping(root["constraints"], label="constraints")
    style_bounds = _strict_mapping(
        constraint_values.pop("style_active_bounds"), label="style bounds"
    )
    constraints = PortfolioConstraints(
        **{key: value for key, value in constraint_values.items()},
        style_active_bounds={str(key): float(value) for key, value in style_bounds.items()},
    )
    accounting = _strict_mapping(root["accounting"], label="accounting policy")
    if set(style_bounds) != set(REQUIRED_STYLE_COLUMNS):
        raise ValueError("portfolio style-bound family is not exact")
    costs = tuple(float(value) for value in portfolio["cost_sensitivity_bps_one_way"])
    if costs != (10.0, 20.0, 40.0) or float(
        portfolio["primary_cost_bps_one_way"]
    ) != 20.0:
        raise ValueError("cost family or primary cost changed")
    if (
        portfolio["benchmark"] != "point_in_time_implied_float_mcap_proxy"
        or portfolio["initialization"]
        != "benchmark_endowment_for_optimizer_with_full_initial_funding_cost"
        or risk["method"] != "diagonal_adjusted_close_variance"
        or accounting["return_basis"]
        != "backward_adjusted_vwap_exact_calendar_endpoint"
        or accounting["benchmark_costed"] is not False
        or accounting["cash_execution_authorized"] is not False
        or accounting["official_index_enhancement_claim_authorized"] is not False
    ):
        raise ValueError("portfolio replay boundary declarations changed")
    payload_hash = hash_json(root)
    return PortfolioReplayPolicy(
        initial_nav_cny=float(portfolio["initial_nav_cny"]),
        rebalance_every_sessions=int(portfolio["rebalance_every_sessions"]),
        execution_delay_sessions=int(portfolio["execution_delay_sessions"]),
        holding_horizon_sessions=int(portfolio["holding_horizon_sessions"]),
        benchmark=str(portfolio["benchmark"]),
        primary_cost_bps_one_way=float(portfolio["primary_cost_bps_one_way"]),
        cost_sensitivity_bps_one_way=costs,
        initialization=str(portfolio["initialization"]),
        risk_method=str(risk["method"]),
        risk_lookback_sessions=int(risk["lookback_sessions"]),
        risk_minimum_observations=int(risk["minimum_observations"]),
        variance_floor=float(risk["variance_floor"]),
        constraints=constraints,
        return_basis=str(accounting["return_basis"]),
        benchmark_costed=bool(accounting["benchmark_costed"]),
        cash_execution_authorized=bool(accounting["cash_execution_authorized"]),
        official_index_enhancement_claim_authorized=bool(
            accounting["official_index_enhancement_claim_authorized"]
        ),
        policy_hash=payload_hash,
    )


def _panel_index(frame: pd.DataFrame | pd.Series, *, label: str) -> None:
    if not isinstance(frame.index, pd.MultiIndex) or tuple(frame.index.names) != (
        "date", "symbol"
    ):
        raise ValueError(f"{label} must use a (date, symbol) MultiIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} index must be unique and sorted")


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("cross-sectional input is not finite")
    centered = numeric - float(numeric.mean())
    scale = float(centered.std(ddof=0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("cross-sectional input is constant")
    return centered / scale


def _benchmark_weights(log_mcap: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(log_mcap, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("benchmark log_mcap is incomplete")
    scaled = np.exp(numeric - float(numeric.max()))
    weights = scaled / float(scaled.sum())
    weights.name = "benchmark_weight"
    return weights


def _diagonal_covariance(
    adjusted_close: pd.Series,
    *,
    as_of: pd.Timestamp,
    names: pd.Index,
    policy: PortfolioReplayPolicy,
) -> tuple[pd.DataFrame, int]:
    history = adjusted_close.loc[
        adjusted_close.index.get_level_values("date") <= as_of
    ]
    returns = history.groupby(level="symbol", sort=False).pct_change(fill_method=None)
    matrix = returns.unstack("symbol").reindex(columns=names).tail(
        policy.risk_lookback_sessions
    )
    observations = matrix.notna().sum()
    variances = matrix.var(ddof=0)
    eligible = observations >= policy.risk_minimum_observations
    usable = variances.loc[eligible].replace([np.inf, -np.inf], np.nan).dropna()
    if usable.empty:
        raise ValueError("risk window has no estimable variances")
    fallback = max(float(usable.median()), policy.variance_floor)
    variances = variances.where(eligible).fillna(fallback).clip(
        lower=policy.variance_floor
    )
    covariance = pd.DataFrame(
        np.diag(variances.to_numpy(dtype=float)), index=names, columns=names
    )
    return covariance, int((~eligible).sum())


def _style_snapshot(exposures: pd.DataFrame, date: pd.Timestamp, names: pd.Index) -> pd.DataFrame:
    block = exposures.xs(date, level="date").reindex(names)
    renamed = block.rename(
        columns={"momentum_20": "momentum", "volatility_20": "volatility"}
    )
    output = pd.DataFrame(index=names)
    for name in REQUIRED_STYLE_COLUMNS:
        output[name] = _zscore(renamed[name])
    return output


def _drift_to_signal_close(
    prior_target: pd.Series,
    *,
    prior_execution_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    adjusted_vwap: pd.Series,
    adjusted_close: pd.Series,
) -> pd.Series:
    entry = adjusted_vwap.xs(prior_execution_date, level="date").reindex(
        prior_target.index
    )
    mark = adjusted_close.xs(signal_date, level="date").reindex(prior_target.index)
    if entry.isna().any() or mark.isna().any() or (entry <= 0).any() or (mark <= 0).any():
        missing = prior_target.index[entry.isna() | mark.isna()].astype(str)
        raise ValueError(
            "cannot reconstruct pre-trade weights at signal close: "
            + ",".join(missing[:10])
        )
    values = prior_target * mark / entry
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("drifted portfolio value is invalid")
    return values / total


def prepare_target_book(
    alpha: pd.Series,
    eligible_panel: pd.DataFrame,
    feature_tape: pd.DataFrame,
    trading_calendar: Sequence[pd.Timestamp],
    sealed_dates: Sequence[pd.Timestamp],
    policy: PortfolioReplayPolicy,
) -> TargetBook:
    """Build all targets without receiving any forward-return outcome."""

    _panel_index(alpha, label="alpha")
    _panel_index(eligible_panel, label="eligible panel")
    _panel_index(feature_tape, label="feature tape")
    if policy.execution_delay_sessions != 1:
        raise ValueError("only next-session execution is pre-registered")
    if policy.holding_horizon_sessions != policy.rebalance_every_sessions:
        raise ValueError("non-overlapping replay requires horizon equal rebalance spacing")
    required = {"log_mcap", "industry", "adv20_dollars", "tradable"}
    missing = required.difference(eligible_panel.columns)
    if missing:
        raise ValueError("eligible panel omits portfolio fields: " + ",".join(sorted(missing)))
    if not {"adj_close", "adj_vwap"}.issubset(feature_tape.columns):
        raise ValueError("feature tape omits adjusted close/VWAP")
    calendar = pd.DatetimeIndex(trading_calendar).normalize().sort_values().unique()
    positions = {pd.Timestamp(date): index for index, date in enumerate(calendar)}
    signals = tuple(pd.DatetimeIndex(sealed_dates)[:: policy.rebalance_every_sessions])
    executions = tuple(calendar[positions[date] + 1] for date in signals)
    exits = tuple(
        calendar[positions[date] + 1 + policy.holding_horizon_sessions]
        for date in signals
    )
    exposures = build_style_exposures(eligible_panel, price_tape=feature_tape)
    adjusted_close = pd.to_numeric(feature_tape["adj_close"], errors="coerce")
    adjusted_vwap = pd.to_numeric(feature_tape["adj_vwap"], errors="coerce")
    optimizer = ConstrainedPortfolioOptimizer(policy.constraints)
    target_rows: list[pd.DataFrame] = []
    benchmark_rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    prior_target: pd.Series | None = None
    prior_execution: pd.Timestamp | None = None
    for signal_date, execution_date in zip(signals, executions):
        current = eligible_panel.xs(signal_date, level="date").sort_index()
        names = pd.Index(current.index.astype(str), name="symbol")
        signal_alpha = alpha.xs(signal_date, level="date").reindex(names)
        signal_alpha = _zscore(signal_alpha)
        benchmark = _benchmark_weights(current["log_mcap"].set_axis(names))
        if prior_target is None:
            previous = benchmark.copy()
        else:
            assert prior_execution is not None
            drifted = _drift_to_signal_close(
                prior_target,
                prior_execution_date=prior_execution,
                signal_date=signal_date,
                adjusted_vwap=adjusted_vwap,
                adjusted_close=adjusted_close,
            )
            previous = drifted.reindex(names).fillna(0.0)
        covariance, imputed_variances = _diagonal_covariance(
            adjusted_close, as_of=signal_date, names=names, policy=policy
        )
        styles = _style_snapshot(exposures, signal_date, names)
        industry = current["industry"].astype("string").set_axis(names)
        adv = pd.to_numeric(current["adv20_dollars"], errors="coerce").set_axis(names)
        tradable = current["tradable"].astype(bool).set_axis(names)
        result = optimizer.optimize(
            signal_alpha, benchmark, previous, covariance, industry,
            style_exposures=styles, adv20_dollars=adv,
            portfolio_value=policy.initial_nav_cny, tradable=tradable,
        )
        block = result.weights.rename("target_weight").to_frame()
        block["signal_date"] = signal_date
        block["execution_date"] = execution_date
        block = block.reset_index().set_index(["signal_date", "symbol"]).sort_index()
        target_rows.append(block)
        bench_block = benchmark.rename("benchmark_weight").to_frame()
        bench_block["signal_date"] = signal_date
        bench_block = bench_block.reset_index().set_index(["signal_date", "symbol"]).sort_index()
        benchmark_rows.append(bench_block)
        diagnostics.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "names": len(names),
                "solver_status": result.status,
                "planned_one_way_turnover": result.one_way_turnover,
                "annual_tracking_error": result.annual_tracking_error,
                "risk_variances_imputed": imputed_variances,
                "maximum_constraint_violation": max(
                    result.post_solve_constraint_violations.values(), default=0.0
                ),
            }
        )
        prior_target = result.weights.copy()
        prior_execution = pd.Timestamp(execution_date)
    targets = pd.concat(target_rows).sort_index()
    benchmarks = pd.concat(benchmark_rows).sort_index()
    diagnostic_frame = pd.DataFrame.from_records(diagnostics).set_index("signal_date")
    if targets.index.has_duplicates or benchmarks.index.has_duplicates:
        raise ValueError("target book contains duplicate date-symbol keys")
    return TargetBook(
        targets=targets,
        benchmarks=benchmarks,
        diagnostics=diagnostic_frame,
        signal_dates=signals,
        execution_dates=executions,
        exit_dates=exits,
        policy_hash=policy.policy_hash,
    )


def _union_turnover(current: pd.Series, previous: pd.Series | None) -> float:
    if previous is None:
        return 1.0
    names = current.index.union(previous.index)
    return float(0.5 * (current.reindex(names, fill_value=0.0) - previous.reindex(names, fill_value=0.0)).abs().sum())


def _performance_summary(periods: pd.DataFrame, *, spacing: int) -> dict[str, float]:
    portfolio = pd.to_numeric(periods["portfolio_net_return"], errors="coerce")
    benchmark = pd.to_numeric(periods["benchmark_gross_return"], errors="coerce")
    if not np.isfinite(portfolio).all() or not np.isfinite(benchmark).all():
        raise ValueError("performance periods contain non-finite returns")
    years = max(len(periods) * spacing / 252.0, spacing / 252.0)
    portfolio_nav = (1.0 + portfolio).cumprod()
    benchmark_nav = (1.0 + benchmark).cumprod()
    active = portfolio - benchmark
    active_std = float(active.std(ddof=1))
    return {
        "periods": int(len(periods)),
        "portfolio_cumulative_return": float(portfolio_nav.iloc[-1] - 1.0),
        "benchmark_cumulative_return": float(benchmark_nav.iloc[-1] - 1.0),
        "cumulative_excess_return": float(portfolio_nav.iloc[-1] / benchmark_nav.iloc[-1] - 1.0),
        "portfolio_annualized_return": float(portfolio_nav.iloc[-1] ** (1.0 / years) - 1.0),
        "benchmark_annualized_return": float(benchmark_nav.iloc[-1] ** (1.0 / years) - 1.0),
        "annualized_excess_return": float(portfolio_nav.iloc[-1] ** (1.0 / years) - benchmark_nav.iloc[-1] ** (1.0 / years)),
        "information_ratio": float(active.mean() / active_std * math.sqrt(252.0 / spacing)) if active_std > 0 else float("nan"),
        "portfolio_endpoint_max_drawdown": float((portfolio_nav / portfolio_nav.cummax() - 1.0).min()),
        "excess_win_rate": float((active > 0).mean()),
        "average_one_way_turnover": float(periods["one_way_turnover"].mean()),
        "minimum_immediate_executable_weight_coverage": float(periods["immediate_executable_weight_coverage"].min()),
    }


def score_target_book(
    book: TargetBook,
    forecast_returns: pd.Series,
    immediate_executable_returns: pd.Series,
    policy: PortfolioReplayPolicy,
) -> PortfolioReplayResult:
    """Score a previously frozen target book; no targets are changed here."""

    if book.policy_hash != policy.policy_hash:
        raise ValueError("target book policy hash mismatch")
    _panel_index(forecast_returns, label="forecast returns")
    _panel_index(immediate_executable_returns, label="executable returns")
    scenario_frames: list[pd.DataFrame] = []
    primary: pd.DataFrame | None = None
    previous_post_return_weights: pd.Series | None = None
    rows: list[dict[str, object]] = []
    for position, (signal_date, execution_date, exit_date) in enumerate(
        zip(book.signal_dates, book.execution_dates, book.exit_dates)
    ):
        target = book.targets.xs(signal_date, level="signal_date")["target_weight"]
        benchmark = book.benchmarks.xs(signal_date, level="signal_date")["benchmark_weight"]
        returns = forecast_returns.xs(signal_date, level="date")
        if target.index.difference(returns.index).size or benchmark.index.difference(returns.index).size:
            raise ValueError("return labels do not cover a target/benchmark universe")
        selected_returns = pd.to_numeric(returns.reindex(target.index), errors="coerce")
        benchmark_returns = pd.to_numeric(returns.reindex(benchmark.index), errors="coerce")
        if selected_returns.isna().any() or benchmark_returns.isna().any():
            raise ValueError("standard adjusted-VWAP portfolio labels are incomplete")
        turnover = _union_turnover(target, previous_post_return_weights)
        gross = float(target.dot(selected_returns))
        benchmark_gross = float(benchmark.dot(benchmark_returns))
        executable = immediate_executable_returns.xs(signal_date, level="date").reindex(target.index)
        executable_coverage = float(target.loc[executable.notna()].sum())
        rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "exit_date": exit_date,
                "portfolio_gross_return": gross,
                "benchmark_gross_return": benchmark_gross,
                "one_way_turnover": turnover,
                "immediate_executable_weight_coverage": executable_coverage,
            }
        )
        ending_values = target * (1.0 + selected_returns)
        ending_total = float(ending_values.sum())
        if not math.isfinite(ending_total) or ending_total <= 0:
            raise ValueError("portfolio ending value is invalid")
        previous_post_return_weights = ending_values / ending_total
    base = pd.DataFrame.from_records(rows).set_index("signal_date")
    summaries: dict[str, object] = {}
    for cost_bps in policy.cost_sensitivity_bps_one_way:
        frame = base.copy()
        cost = frame["one_way_turnover"] * cost_bps / 10_000.0
        frame["transaction_cost"] = cost
        frame["portfolio_net_return"] = (
            (1.0 - cost) * (1.0 + frame["portfolio_gross_return"]) - 1.0
        )
        frame["portfolio_nav"] = policy.initial_nav_cny * (
            1.0 + frame["portfolio_net_return"]
        ).cumprod()
        frame["benchmark_nav"] = policy.initial_nav_cny * (
            1.0 + frame["benchmark_gross_return"]
        ).cumprod()
        frame["cost_bps_one_way"] = cost_bps
        scenario_frames.append(frame.assign(cost_scenario_bps=cost_bps))
        summaries[f"cost_{int(cost_bps)}bps"] = _performance_summary(
            frame, spacing=policy.rebalance_every_sessions
        )
        if cost_bps == policy.primary_cost_bps_one_way:
            primary = frame
    if primary is None:
        raise ValueError("primary cost scenario is absent")
    scenario_table = pd.concat(scenario_frames, keys=policy.cost_sensitivity_bps_one_way, names=["cost_bps", "signal_date"])
    summary = {
        "schema_version": 1,
        "kind": "hs300_cost_aware_research_portfolio_replay_result",
        "policy_hash": policy.policy_hash,
        "primary_cost_bps_one_way": policy.primary_cost_bps_one_way,
        "primary": summaries[f"cost_{int(policy.primary_cost_bps_one_way)}bps"],
        "cost_sensitivity": summaries,
        "cash_execution_authorized": False,
        "official_index_enhancement_claim_authorized": False,
        "research_status": "NOT VALIDATED",
        "research_status_reason": (
            "official historical HS300 weights and a complete sourced cash "
            "corporate-action/execution ledger are unavailable"
        ),
    }
    return PortfolioReplayResult(primary, scenario_table, summary)


def target_book_hash(book: TargetBook) -> str:
    payload = {
        "policy_hash": book.policy_hash,
        "signal_dates": [str(date.date()) for date in book.signal_dates],
        "execution_dates": [str(date.date()) for date in book.execution_dates],
        "exit_dates": [str(date.date()) for date in book.exit_dates],
        "targets": json.loads(book.targets.reset_index().to_json(orient="records", date_format="iso")),
        "benchmarks": json.loads(book.benchmarks.reset_index().to_json(orient="records", date_format="iso")),
        "diagnostics": json.loads(book.diagnostics.reset_index().to_json(orient="records", date_format="iso")),
    }
    return hash_json(payload)


__all__ = [
    "PortfolioReplayPolicy", "PortfolioReplayResult", "TargetBook",
    "load_portfolio_replay_policy", "prepare_target_book",
    "score_target_book", "target_book_hash",
]
