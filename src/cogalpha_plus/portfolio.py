"""Integrated long-only benchmark-aware portfolio optimization.

The solver status is only a candidate result. Every hard constraint is
recomputed from the returned numeric weights before a solution is accepted.
This is especially important for ``optimal_inaccurate`` results: they receive
the same post-solve tolerance as an ``optimal`` result and fail closed when any
realized constraint is outside that tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import cvxpy as cp
import numpy as np
import pandas as pd


# A weight is a fraction of total NAV. One part per million is small enough to
# accommodate ordinary conic-solver round-off without silently granting a
# commercially meaningful relaxation. The same absolute tolerance is applied
# to OPTIMAL and OPTIMAL_INACCURATE candidates.
POST_SOLVE_ABSOLUTE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PortfolioConstraints:
    max_weight: float = 0.04
    max_active_weight: float = 0.02
    max_industry_active: float = 0.05
    max_one_way_turnover: float = 0.20
    max_annual_tracking_error: float = 0.08
    risk_aversion: float = 4.0
    turnover_penalty: float = 0.002
    target_investment: float = 1.0
    participation_rate: float = 0.05
    style_active_bounds: Mapping[str, float] = field(
        default_factory=lambda: {
            "beta": 0.10,
            "log_mcap": 0.20,
            "momentum": 0.20,
            "volatility": 0.20,
            "liquidity": 0.20,
        }
    )

    def __post_init__(self) -> None:
        positive_limits = (
            self.max_weight,
            self.max_active_weight,
            self.max_industry_active,
            self.max_one_way_turnover,
            self.max_annual_tracking_error,
            self.target_investment,
            self.participation_rate,
        )
        if any(not math.isfinite(float(value)) or value <= 0 for value in positive_limits):
            raise ValueError("portfolio limits must be positive")
        if self.target_investment > 1.0 or self.participation_rate > 1.0:
            raise ValueError("target investment and participation rate cannot exceed 1")
        if (
            not math.isfinite(float(self.risk_aversion))
            or not math.isfinite(float(self.turnover_penalty))
            or self.risk_aversion < 0
            or self.turnover_penalty < 0
        ):
            raise ValueError("risk aversion and turnover penalty must be finite and non-negative")
        if not self.style_active_bounds:
            raise ValueError("style active bounds cannot be empty")
        if any(
            not str(name).strip()
            or not math.isfinite(float(bound))
            or float(bound) <= 0
            for name, bound in self.style_active_bounds.items()
        ):
            raise ValueError("style active bounds must have non-blank names and positive finite values")


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    status: str
    objective_value: float
    one_way_turnover: float
    annual_tracking_error: float
    industry_active: Mapping[str, float]
    style_active: Mapping[str, float]
    binding_capacity_names: tuple[str, ...]
    post_solve_tolerance: float
    post_solve_constraint_violations: Mapping[str, float]


@dataclass(frozen=True)
class _AuditedSolution:
    weights: np.ndarray
    one_way_turnover: float
    annual_tracking_error: float
    industry_active: Mapping[str, float]
    style_active: Mapping[str, float]
    binding_capacity_names: tuple[str, ...]
    constraint_violations: Mapping[str, float]


def _nearest_psd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    return (vectors * values) @ vectors.T


def _positive_violation(value: float, limit: float) -> float:
    return max(float(value) - float(limit), 0.0)


def _audit_post_solve_solution(
    weights: np.ndarray,
    *,
    status: str,
    objective_value: float | None,
    names: pd.Index,
    benchmark: np.ndarray,
    previous: np.ndarray,
    sigma: np.ndarray,
    industry_values: pd.Series,
    style_vectors: Mapping[str, np.ndarray],
    max_weight_trade: np.ndarray | None,
    can_trade: np.ndarray | None,
    constraints: PortfolioConstraints,
    cash_leg_change: float,
    tolerance: float = POST_SOLVE_ABSOLUTE_TOLERANCE,
) -> _AuditedSolution:
    """Recompute and verify every hard constraint on a solver candidate."""

    candidate = np.asarray(weights, dtype=float).reshape(-1)
    if candidate.shape != (len(names),):
        raise RuntimeError(
            "post-solve constraint audit failed "
            f"(status={status}): weight_shape"
        )
    if not np.isfinite(candidate).all():
        raise RuntimeError(
            "post-solve constraint audit failed "
            f"(status={status}): finite_weights"
        )
    if objective_value is None or not math.isfinite(float(objective_value)):
        raise RuntimeError(
            "post-solve constraint audit failed "
            f"(status={status}): finite_objective"
        )

    # Remove only sub-machine-actionable signed noise. The cleaned vector is
    # the vector returned to the caller and is itself audited below; values near
    # the public tolerance are never projected into feasibility.
    candidate = candidate.copy()
    candidate[np.abs(candidate) < 1e-12] = 0.0
    active = candidate - benchmark
    trade = candidate - previous
    risky_weight = float(candidate.sum())
    cash_weight = float(1.0 - risky_weight)
    target_cash_weight = float(1.0 - constraints.target_investment)
    one_way = float((np.abs(trade).sum() + cash_leg_change) / 2.0)
    risk_variance = float(active @ sigma @ active)
    if not math.isfinite(risk_variance):
        raise RuntimeError(
            "post-solve constraint audit failed "
            f"(status={status}): finite_risk"
        )
    tracking = float(np.sqrt(max(risk_variance, 0.0) * 252.0))

    industry_active = {
        str(label): float(active[(industry_values == label).to_numpy()].sum())
        for label in sorted(industry_values.unique())
    }
    style_active = {
        name: float(vector @ active) for name, vector in style_vectors.items()
    }

    violations: dict[str, float] = {
        "investment": abs(risky_weight - constraints.target_investment),
        "cash": abs(cash_weight - target_cash_weight),
        "long_only": max(float(-candidate.min()), 0.0),
        "single_name_weight": _positive_violation(
            float(candidate.max()), constraints.max_weight
        ),
        "benchmark_active_weight": _positive_violation(
            float(np.abs(active).max()), constraints.max_active_weight
        ),
        "one_way_turnover": _positive_violation(
            one_way, constraints.max_one_way_turnover
        ),
        "annual_tracking_error": _positive_violation(
            tracking, constraints.max_annual_tracking_error
        ),
    }
    for label, realized in industry_active.items():
        violations[f"industry:{label}"] = _positive_violation(
            abs(realized), constraints.max_industry_active
        )
    for name, realized in style_active.items():
        violations[f"style:{name}"] = _positive_violation(
            abs(realized), float(constraints.style_active_bounds[name])
        )

    capacity_names: tuple[str, ...] = ()
    if max_weight_trade is not None:
        capacity_excess = np.abs(trade) - max_weight_trade
        violations["adv_participation"] = max(float(capacity_excess.max()), 0.0)
        usage = np.divide(
            np.abs(trade),
            max_weight_trade,
            out=np.full_like(trade, np.inf),
            where=max_weight_trade > 0,
        )
        capacity_names = tuple(names[usage >= 0.999].astype(str))
    if can_trade is not None and (~can_trade).any():
        frozen_drift = np.abs(trade[~can_trade])
        violations["untradable_freeze"] = float(frozen_drift.max())

    failed = {
        name: value
        for name, value in violations.items()
        if not math.isfinite(value) or value > tolerance
    }
    if failed:
        detail = ", ".join(
            f"{name}={value:.12g}" for name, value in sorted(failed.items())
        )
        raise RuntimeError(
            "post-solve constraint audit failed "
            f"(status={status}, tolerance={tolerance:.1e}): {detail}"
        )

    return _AuditedSolution(
        weights=candidate,
        one_way_turnover=one_way,
        annual_tracking_error=tracking,
        industry_active=industry_active,
        style_active=style_active,
        binding_capacity_names=capacity_names,
        constraint_violations=violations,
    )


class ConstrainedPortfolioOptimizer:
    """Optimize alpha, active risk, turnover, capacity and exposures together."""

    def __init__(self, constraints: PortfolioConstraints | None = None) -> None:
        self.constraints = constraints or PortfolioConstraints()

    def optimize(
        self,
        alpha: pd.Series,
        benchmark_weights: pd.Series,
        previous_weights: pd.Series,
        covariance: pd.DataFrame,
        industry: pd.Series,
        *,
        style_exposures: pd.DataFrame | None = None,
        adv20_dollars: pd.Series | None = None,
        portfolio_value: float | None = None,
        tradable: pd.Series | None = None,
    ) -> OptimizationResult:
        names = pd.Index(alpha.index.astype(str), name="symbol")
        if names.has_duplicates or len(names) < 2:
            raise ValueError("alpha index must contain unique symbols")
        if covariance.index.has_duplicates or covariance.columns.has_duplicates:
            raise ValueError("covariance axes must contain unique symbols")
        if set(covariance.index.astype(str)) != set(names) or set(covariance.columns.astype(str)) != set(names):
            raise ValueError("covariance must cover exactly the alpha universe")
        alpha_numeric = pd.to_numeric(alpha.reindex(names), errors="coerce")
        if not np.isfinite(alpha_numeric.to_numpy(dtype=float)).all():
            raise ValueError("alpha must be finite for every symbol")
        alpha_v = alpha_numeric.to_numpy(dtype=float)
        benchmark_input = benchmark_weights.copy()
        benchmark_input.index = benchmark_input.index.astype(str)
        if benchmark_input.index.has_duplicates or set(benchmark_input.index) != set(names):
            raise ValueError("benchmark weights must cover exactly the alpha universe")
        benchmark_numeric = pd.to_numeric(benchmark_input.reindex(names), errors="coerce")
        if not np.isfinite(benchmark_numeric.to_numpy(dtype=float)).all():
            raise ValueError("benchmark weights must be finite for every symbol")
        benchmark = benchmark_numeric.to_numpy(dtype=float)
        previous_input = previous_weights.copy()
        previous_input.index = previous_input.index.astype(str)
        if previous_input.index.has_duplicates:
            raise ValueError("previous weights index must contain unique symbols")
        previous_numeric = pd.to_numeric(previous_input, errors="coerce")
        if not np.isfinite(previous_numeric.to_numpy(dtype=float)).all():
            raise ValueError("previous weights must be finite")
        outside = previous_numeric.loc[~previous_numeric.index.isin(names)].fillna(0.0)
        if (outside.abs() > 1e-12).any():
            raise ValueError(
                "alpha universe omits actually held symbols: "
                + ",".join(sorted(outside.index[outside.abs() > 1e-12].astype(str)))
            )
        previous = previous_numeric.reindex(names).fillna(0.0).to_numpy(dtype=float)
        if np.any(benchmark < -1e-12) or np.any(previous < -1e-12):
            raise ValueError("benchmark and previous weights must be non-negative")
        if not np.isclose(benchmark.sum(), self.constraints.target_investment, atol=1e-6):
            raise ValueError("benchmark weights must sum to target investment")
        if previous.sum() > 1.000001:
            raise ValueError("previous risky weights cannot exceed total portfolio NAV")
        covariance_values = covariance.copy()
        covariance_values.index = covariance_values.index.astype(str)
        covariance_values.columns = covariance_values.columns.astype(str)
        covariance_array = covariance_values.loc[names, names].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(covariance_array).all():
            raise ValueError("covariance must be finite")
        if not np.allclose(covariance_array, covariance_array.T, rtol=1e-8, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        sigma = _nearest_psd(covariance_array)

        n = len(names)
        weight = cp.Variable(n)
        active = weight - benchmark
        trade = weight - previous
        # Previous weights are reconstructed from actual fills, so cash and
        # receivables can make risky weights sum below one.  Include the cash
        # leg in one-way turnover instead of renormalizing away that state.
        cash_leg_change = abs(float(previous.sum()) - self.constraints.target_investment)
        objective = cp.Maximize(
            alpha_v @ weight
            - self.constraints.risk_aversion * cp.quad_form(active, sigma)
            - self.constraints.turnover_penalty * cp.norm1(trade)
        )
        constraints: list[cp.Constraint] = [
            cp.sum(weight) == self.constraints.target_investment,
            weight >= 0.0,
            weight <= self.constraints.max_weight,
            active <= self.constraints.max_active_weight,
            active >= -self.constraints.max_active_weight,
            0.5 * (cp.norm1(trade) + cash_leg_change) <= self.constraints.max_one_way_turnover,
            cp.quad_form(active, sigma) <= (self.constraints.max_annual_tracking_error**2) / 252.0,
        ]

        industry_input = industry.copy()
        industry_input.index = industry_input.index.astype(str)
        if industry_input.index.has_duplicates or set(industry_input.index) != set(names):
            raise ValueError("industry must cover exactly the alpha universe")
        industry_values = industry_input.reindex(names).astype("string")
        if industry_values.isna().any() or (industry_values.str.strip() == "").any():
            raise ValueError("industry is missing for one or more symbols")
        for label in sorted(industry_values.unique()):
            mask = (industry_values == label).to_numpy(dtype=float)
            constraints.extend(
                [
                    mask @ active <= self.constraints.max_industry_active,
                    mask @ active >= -self.constraints.max_industry_active,
                ]
            )

        style_columns: list[str] = []
        style_vectors: dict[str, np.ndarray] = {}
        if style_exposures is not None:
            styles = style_exposures.copy()
            styles.index = styles.index.astype(str)
            if styles.index.has_duplicates or set(styles.index) != set(names):
                raise ValueError("style exposures must cover exactly the alpha universe")
            styles = styles.reindex(names)
            style_columns = [name for name in styles.columns if name in self.constraints.style_active_bounds]
            for name in style_columns:
                exposure = pd.to_numeric(styles[name], errors="coerce")
                if not np.isfinite(exposure.to_numpy(dtype=float)).all():
                    raise ValueError(f"style exposure {name!r} must be finite")
                vector = exposure.to_numpy(dtype=float)
                style_vectors[name] = vector
                bound = float(self.constraints.style_active_bounds[name])
                constraints.extend([vector @ active <= bound, vector @ active >= -bound])

        max_weight_trade: np.ndarray | None = None
        if adv20_dollars is not None or portfolio_value is not None:
            if (
                adv20_dollars is None
                or portfolio_value is None
                or not math.isfinite(float(portfolio_value))
                or portfolio_value <= 0
            ):
                raise ValueError("capacity constraints require both positive portfolio_value and ADV20")
            adv_input = adv20_dollars.copy()
            adv_input.index = adv_input.index.astype(str)
            if adv_input.index.has_duplicates or set(adv_input.index) != set(names):
                raise ValueError("ADV20 must cover exactly the alpha universe")
            adv = pd.to_numeric(adv_input.reindex(names), errors="coerce")
            if not np.isfinite(adv.to_numpy(dtype=float)).all() or (adv <= 0).any():
                raise ValueError("ADV20 must be positive for every symbol")
            max_weight_trade = self.constraints.participation_rate * adv.to_numpy(dtype=float) / portfolio_value
            constraints.extend([trade <= max_weight_trade, trade >= -max_weight_trade])

        can_trade: np.ndarray | None = None
        if tradable is not None:
            tradable_input = tradable.copy()
            tradable_input.index = tradable_input.index.astype(str)
            if tradable_input.index.has_duplicates or set(tradable_input.index) != set(names):
                raise ValueError("tradable must cover exactly the alpha universe")
            tradable_values = tradable_input.reindex(names)
            if tradable_values.isna().any() or not tradable_values.isin([True, False, 0, 1]).all():
                raise ValueError("tradable must contain strict boolean values")
            can_trade = tradable_values.astype(bool).to_numpy()
            for index in np.flatnonzero(~can_trade):
                constraints.append(weight[index] == previous[index])

        problem = cp.Problem(objective, constraints)
        audited: _AuditedSolution | None = None
        last_error: Exception | None = None
        audit_errors: list[str] = []
        for solver in ("CLARABEL", "SCS"):
            if solver not in cp.installed_solvers():
                continue
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                    if weight.value is None:
                        audit_errors.append(f"{solver}:missing_weights")
                        continue
                    try:
                        audited = _audit_post_solve_solution(
                            np.asarray(weight.value, dtype=float),
                            status=str(problem.status),
                            objective_value=problem.value,
                            names=names,
                            benchmark=benchmark,
                            previous=previous,
                            sigma=sigma,
                            industry_values=industry_values,
                            style_vectors=style_vectors,
                            max_weight_trade=max_weight_trade,
                            can_trade=can_trade,
                            constraints=self.constraints,
                            cash_leg_change=cash_leg_change,
                        )
                    except RuntimeError as exc:
                        audit_errors.append(f"{solver}:{exc}")
                        last_error = exc
                        continue
                    break
            except Exception as exc:  # pragma: no cover - solver-specific fallback.
                last_error = exc
        if audited is None:
            if audit_errors:
                suffix = ": " + " | ".join(audit_errors)
            else:
                suffix = f": {last_error}" if last_error else ""
            raise RuntimeError(f"portfolio optimization failed with status {problem.status}{suffix}")

        return OptimizationResult(
            weights=pd.Series(audited.weights, index=names, name="target_weight"),
            status=str(problem.status),
            objective_value=float(problem.value),
            one_way_turnover=audited.one_way_turnover,
            annual_tracking_error=audited.annual_tracking_error,
            industry_active=audited.industry_active,
            style_active=audited.style_active,
            binding_capacity_names=audited.binding_capacity_names,
            post_solve_tolerance=POST_SOLVE_ABSOLUTE_TOLERANCE,
            post_solve_constraint_violations=audited.constraint_violations,
        )


__all__ = [
    "ConstrainedPortfolioOptimizer",
    "OptimizationResult",
    "PortfolioConstraints",
    "POST_SOLVE_ABSOLUTE_TOLERANCE",
]
