import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.portfolio import (
    POST_SOLVE_ABSOLUTE_TOLERANCE,
    ConstrainedPortfolioOptimizer,
    PortfolioConstraints,
)


def test_optimizer_enforces_risk_turnover_industry_style_and_capacity():
    names = pd.Index([f"S{i:02d}" for i in range(30)], name="symbol")
    alpha = pd.Series(np.linspace(-0.02, 0.03, len(names)), index=names)
    benchmark = pd.Series(1 / len(names), index=names)
    previous = benchmark.copy()
    covariance = pd.DataFrame(np.eye(len(names)) * 0.0001, index=names, columns=names)
    industry = pd.Series([f"I{i % 5}" for i in range(len(names))], index=names)
    styles = pd.DataFrame(
        {
            "beta": np.linspace(0.8, 1.2, len(names)),
            "log_mcap": np.linspace(-1, 1, len(names)),
            "momentum": np.sin(np.arange(len(names))),
            "volatility": np.cos(np.arange(len(names))),
            "liquidity": np.linspace(1, -1, len(names)),
        },
        index=names,
    )
    adv = pd.Series(20_000_000.0, index=names)
    constraints = PortfolioConstraints(
        max_weight=0.05,
        max_active_weight=0.02,
        max_industry_active=0.02,
        max_one_way_turnover=0.10,
        max_annual_tracking_error=0.08,
        participation_rate=0.05,
    )
    result = ConstrainedPortfolioOptimizer(constraints).optimize(
        alpha,
        benchmark,
        previous,
        covariance,
        industry,
        style_exposures=styles,
        adv20_dollars=adv,
        portfolio_value=1_000_000.0,
    )
    assert np.isclose(result.weights.sum(), 1.0, atol=2e-5)
    assert result.weights.max() <= 0.05001
    assert result.one_way_turnover <= 0.1001
    assert max(abs(value) for value in result.industry_active.values()) <= 0.0201
    assert result.annual_tracking_error <= 0.0801
    assert result.post_solve_tolerance == POST_SOLVE_ABSOLUTE_TOLERANCE
    assert max(result.post_solve_constraint_violations.values()) <= result.post_solve_tolerance


def test_untradable_position_is_frozen():
    names = pd.Index([f"S{i:02d}" for i in range(25)], name="symbol")
    benchmark = pd.Series(1 / len(names), index=names)
    alpha = pd.Series(np.arange(len(names), dtype=float), index=names)
    covariance = pd.DataFrame(np.eye(len(names)) * 0.0001, index=names, columns=names)
    industry = pd.Series([f"I{i % 5}" for i in range(len(names))], index=names)
    tradable = pd.Series(True, index=names)
    tradable.iloc[-1] = False
    result = ConstrainedPortfolioOptimizer(
        PortfolioConstraints(max_weight=0.06, max_active_weight=0.03, max_one_way_turnover=0.2)
    ).optimize(alpha, benchmark, benchmark, covariance, industry, tradable=tradable)
    assert np.isclose(result.weights.iloc[-1], benchmark.iloc[-1], atol=1e-6)


def test_optimizer_accepts_actual_risky_weights_with_cash_and_counts_cash_leg_turnover():
    names = pd.Index([f"S{i:02d}" for i in range(25)], name="symbol")
    benchmark = pd.Series(1 / len(names), index=names)
    previous = benchmark * 0.9
    alpha = pd.Series(np.linspace(0.0, 1.0, len(names)), index=names)
    covariance = pd.DataFrame(np.eye(len(names)) * 0.0001, index=names, columns=names)
    industry = pd.Series([f"I{i % 5}" for i in range(len(names))], index=names)
    result = ConstrainedPortfolioOptimizer(
        PortfolioConstraints(max_weight=0.06, max_active_weight=0.03, max_one_way_turnover=0.2)
    ).optimize(alpha, benchmark, previous, covariance, industry)
    assert result.one_way_turnover >= 0.1 - 1e-6
    assert result.one_way_turnover <= 0.2001


def test_optimizer_fails_if_actual_holdings_are_omitted_from_alpha_universe():
    names = pd.Index([f"S{i:02d}" for i in range(25)], name="symbol")
    benchmark = pd.Series(1 / len(names), index=names)
    previous = benchmark.copy()
    previous.loc["REMOVED_BUT_HELD"] = 0.01
    alpha = pd.Series(0.0, index=names)
    covariance = pd.DataFrame(np.eye(len(names)) * 0.0001, index=names, columns=names)
    industry = pd.Series("I", index=names)
    with pytest.raises(ValueError, match="omits actually held"):
        ConstrainedPortfolioOptimizer().optimize(
            alpha,
            benchmark,
            previous,
            covariance,
            industry,
        )


def _mocked_solver_inputs() -> dict[str, object]:
    names = pd.Index(["A", "B", "C", "D"], name="symbol")
    return {
        "alpha": pd.Series(0.0, index=names),
        "benchmark": pd.Series(0.25, index=names),
        "previous": pd.Series(0.25, index=names),
        "covariance": pd.DataFrame(
            np.eye(len(names)) * 1e-6, index=names, columns=names
        ),
        "industry": pd.Series(["I0", "I0", "I1", "I1"], index=names),
        "styles": pd.DataFrame(
            {"beta": [1.0, 1.0, -1.0, -1.0]}, index=names
        ),
        "adv": pd.Series(1e9, index=names),
        "tradable": pd.Series(True, index=names),
    }


def _relaxed_constraints(**overrides: object) -> PortfolioConstraints:
    values: dict[str, object] = {
        "max_weight": 0.80,
        "max_active_weight": 0.80,
        "max_industry_active": 0.80,
        "max_one_way_turnover": 0.80,
        "max_annual_tracking_error": 10.0,
        "risk_aversion": 0.0,
        "turnover_penalty": 0.0,
        "participation_rate": 0.50,
        "style_active_bounds": {"beta": 0.80},
    }
    values.update(overrides)
    return PortfolioConstraints(**values)


def _run_mocked_solver_candidate(
    monkeypatch: pytest.MonkeyPatch,
    weights: np.ndarray,
    *,
    constraints: PortfolioConstraints | None = None,
    status: str = cp.OPTIMAL_INACCURATE,
    objective_value: float = 0.0,
    covariance: pd.DataFrame | None = None,
    adv: pd.Series | None = None,
    tradable: pd.Series | None = None,
):
    inputs = _mocked_solver_inputs()

    def fake_solve(problem, *, solver, verbose):
        del solver, verbose
        candidate = np.asarray(weights, dtype=float)
        if np.isfinite(candidate).all():
            problem.variables()[0].value = candidate
        else:
            # CVXPY's public setter rejects NaN before the optimizer can audit
            # a malicious/broken backend, so bypass only for this fault test.
            problem.variables()[0]._value = candidate
        problem._status = status
        problem._value = objective_value
        return objective_value

    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL"])
    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    return ConstrainedPortfolioOptimizer(constraints or _relaxed_constraints()).optimize(
        inputs["alpha"],
        inputs["benchmark"],
        inputs["previous"],
        covariance if covariance is not None else inputs["covariance"],
        inputs["industry"],
        style_exposures=inputs["styles"],
        adv20_dollars=adv if adv is not None else inputs["adv"],
        portfolio_value=1_000_000.0,
        tradable=tradable if tradable is not None else inputs["tradable"],
    )


@pytest.mark.parametrize("status", [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
def test_solver_status_is_accepted_only_after_full_post_solve_audit(monkeypatch, status):
    result = _run_mocked_solver_candidate(
        monkeypatch,
        np.full(4, 0.25),
        status=status,
    )
    assert result.status == status
    assert max(result.post_solve_constraint_violations.values()) == 0.0


@pytest.mark.parametrize("status", [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
def test_solver_status_never_excuses_a_material_constraint_breach(monkeypatch, status):
    with pytest.raises(RuntimeError, match="single_name_weight"):
        _run_mocked_solver_candidate(
            monkeypatch,
            np.array([0.31, 0.23, 0.23, 0.23]),
            constraints=_relaxed_constraints(max_weight=0.30),
            status=status,
        )


@pytest.mark.parametrize(
    ("weights", "constraint_overrides", "expected"),
    [
        (np.array([0.251, 0.25, 0.25, 0.25]), {}, "investment"),
        (np.array([0.251, 0.25, 0.25, 0.25]), {}, "cash"),
        (np.array([-0.01, 0.34, 0.34, 0.33]), {}, "long_only"),
        (
            np.array([0.31, 0.23, 0.23, 0.23]),
            {"max_weight": 0.30},
            "single_name_weight",
        ),
        (
            np.array([0.31, 0.23, 0.23, 0.23]),
            {"max_active_weight": 0.05},
            "benchmark_active_weight",
        ),
        (
            np.array([0.28, 0.22, 0.25, 0.25]),
            {"max_one_way_turnover": 0.02},
            "one_way_turnover",
        ),
        (
            np.array([0.26, 0.26, 0.24, 0.24]),
            {"max_industry_active": 0.01},
            "industry:I0",
        ),
        (
            np.array([0.26, 0.26, 0.24, 0.24]),
            {"style_active_bounds": {"beta": 0.01}},
            "style:beta",
        ),
    ],
)
def test_post_solve_audit_fails_closed_for_each_weight_and_exposure_constraint(
    monkeypatch,
    weights,
    constraint_overrides,
    expected,
):
    with pytest.raises(RuntimeError, match=expected):
        _run_mocked_solver_candidate(
            monkeypatch,
            weights,
            constraints=_relaxed_constraints(**constraint_overrides),
        )


def test_post_solve_audit_fails_closed_for_adv_participation(monkeypatch):
    inputs = _mocked_solver_inputs()
    adv = pd.Series(10_000.0, index=inputs["alpha"].index)
    with pytest.raises(RuntimeError, match="adv_participation"):
        _run_mocked_solver_candidate(
            monkeypatch,
            np.array([0.256, 0.244, 0.25, 0.25]),
            adv=adv,
        )


def test_post_solve_audit_fails_closed_for_untradable_freeze(monkeypatch):
    inputs = _mocked_solver_inputs()
    tradable = pd.Series(True, index=inputs["alpha"].index)
    tradable.iloc[0] = False
    with pytest.raises(RuntimeError, match="untradable_freeze"):
        _run_mocked_solver_candidate(
            monkeypatch,
            np.array([0.251, 0.249, 0.25, 0.25]),
            tradable=tradable,
        )


def test_post_solve_audit_fails_closed_for_tracking_risk(monkeypatch):
    inputs = _mocked_solver_inputs()
    covariance = pd.DataFrame(
        np.eye(4), index=inputs["alpha"].index, columns=inputs["alpha"].index
    )
    with pytest.raises(RuntimeError, match="annual_tracking_error"):
        _run_mocked_solver_candidate(
            monkeypatch,
            np.array([0.26, 0.24, 0.25, 0.25]),
            constraints=_relaxed_constraints(max_annual_tracking_error=0.10),
            covariance=covariance,
        )


@pytest.mark.parametrize(
    ("weights", "objective", "expected"),
    [
        (np.array([np.nan, 0.25, 0.25, 0.25]), 0.0, "finite_weights"),
        (np.full(4, 0.25), np.nan, "finite_objective"),
    ],
)
def test_post_solve_audit_rejects_non_finite_solver_output(
    monkeypatch, weights, objective, expected
):
    with pytest.raises(RuntimeError, match=expected):
        _run_mocked_solver_candidate(
            monkeypatch,
            weights,
            objective_value=objective,
        )


def test_post_solve_tolerance_is_explicit_and_not_a_material_relaxation(monkeypatch):
    within = _relaxed_constraints(
        max_weight=0.25 - POST_SOLVE_ABSOLUTE_TOLERANCE / 2.0
    )
    result = _run_mocked_solver_candidate(
        monkeypatch,
        np.full(4, 0.25),
        constraints=within,
    )
    assert result.post_solve_constraint_violations["single_name_weight"] == pytest.approx(
        POST_SOLVE_ABSOLUTE_TOLERANCE / 2.0
    )

    above = _relaxed_constraints(
        max_weight=0.25 - POST_SOLVE_ABSOLUTE_TOLERANCE * 2.0
    )
    with pytest.raises(RuntimeError, match="single_name_weight"):
        _run_mocked_solver_candidate(
            monkeypatch,
            np.full(4, 0.25),
            constraints=above,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_annual_tracking_error": np.nan},
        {"risk_aversion": -1.0},
        {"turnover_penalty": np.inf},
        {"style_active_bounds": {"beta": 0.0}},
    ],
)
def test_invalid_constraint_parameters_fail_before_solver(kwargs):
    with pytest.raises(ValueError):
        PortfolioConstraints(**kwargs)
