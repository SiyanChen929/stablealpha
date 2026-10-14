from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.dsl import evaluate_expression
from cogalpha_plus.proposals import FactorCard, load_factor_cards
from cogalpha_plus.validation import PRODUCTION_DSL_SPEC


PROJECT = Path(__file__).resolve().parents[1]
PROPOSAL_PATHS = (
    PROJECT / "proposals" / "codex_seed" / "l1_l2.jsonl",
    PROJECT / "proposals" / "codex_seed" / "l3_l4.jsonl",
    PROJECT / "proposals" / "codex_seed" / "l5_l7.jsonl",
    PROJECT / "proposals" / "codex_injection" / "l1_l4.jsonl",
    PROJECT / "proposals" / "codex_injection" / "l5_l7.jsonl",
)


@lru_cache(maxsize=1)
def _cards() -> dict[str, FactorCard]:
    cards = load_factor_cards(PROPOSAL_PATHS, expected_horizon=10, expected_count=176)
    return {card.candidate_key: card for card in cards}


def _panel(
    close_by_symbol: dict[str, np.ndarray | list[float]],
    *,
    open_by_symbol: dict[str, np.ndarray | list[float]] | None = None,
    vwap_by_symbol: dict[str, np.ndarray | list[float]] | None = None,
    high_by_symbol: dict[str, np.ndarray | list[float]] | None = None,
    low_by_symbol: dict[str, np.ndarray | list[float]] | None = None,
    volume_by_symbol: dict[str, np.ndarray | list[float]] | None = None,
    amount_by_symbol: dict[str, np.ndarray | list[float]] | None = None,
) -> pd.DataFrame:
    symbols = tuple(close_by_symbol)
    lengths = {len(values) for values in close_by_symbol.values()}
    assert len(lengths) == 1
    dates = pd.bdate_range("2024-01-02", periods=lengths.pop())
    records: list[dict[str, object]] = []
    for position, date in enumerate(dates):
        for symbol in symbols:
            close = float(close_by_symbol[symbol][position])
            open_price = float(
                close if open_by_symbol is None else open_by_symbol[symbol][position]
            )
            vwap = float(
                close if vwap_by_symbol is None else vwap_by_symbol[symbol][position]
            )
            high = float(
                max(open_price, close, vwap) * 1.01
                if high_by_symbol is None
                else high_by_symbol[symbol][position]
            )
            low = float(
                min(open_price, close, vwap) * 0.99
                if low_by_symbol is None
                else low_by_symbol[symbol][position]
            )
            volume = float(
                1_000.0
                if volume_by_symbol is None
                else volume_by_symbol[symbol][position]
            )
            amount = float(
                volume * vwap
                if amount_by_symbol is None
                else amount_by_symbol[symbol][position]
            )
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "vwap": vwap,
                    "volume": volume,
                    "amount": amount,
                }
            )
    return pd.DataFrame.from_records(records).set_index(["date", "symbol"])


def _final_values(expression: dict[str, object], panel: pd.DataFrame) -> pd.Series:
    values = evaluate_expression(expression, panel, PRODUCTION_DSL_SPEC)
    return values.xs(panel.index.get_level_values("date").max(), level="date")


def _find_half_gate(expression: dict[str, object], scale: float) -> dict[str, object]:
    if expression.get("op") == "div":
        numerator, denominator = expression["args"]
        if (
            numerator.get("op") == "const"
            and float(numerator["value"]) == pytest.approx(scale)
            and denominator.get("op") == "add"
            and any(
                child.get("op") == "const"
                and float(child["value"]) == pytest.approx(scale)
                for child in denominator["args"]
            )
        ):
            return expression
    for child in expression.get("args", []):
        try:
            return _find_half_gate(child, scale)
        except LookupError:
            pass
    raise LookupError(f"no {scale} half-saturation gate")


def _contains_pseudo_cross_sectional_residual(expression: dict[str, object]) -> bool:
    if expression.get("op") == "sub":
        left, right = expression["args"]
        if left.get("op") == right.get("op") == "cs_zscore":
            return True
    return any(
        _contains_pseudo_cross_sectional_residual(child)
        for child in expression.get("args", [])
    )


def test_gap_rejection_and_confirmation_have_distinct_four_quadrant_ordering():
    close = {
        "up_same": np.array([100.0, 100.0, 115.5]),
        "up_reject": np.array([100.0, 100.0, 104.5]),
        "down_same": np.array([100.0, 100.0, 85.5]),
        "down_reject": np.array([100.0, 100.0, 94.5]),
    }
    open_price = {
        "up_same": np.array([100.0, 100.0, 110.0]),
        "up_reject": np.array([100.0, 100.0, 110.0]),
        "down_same": np.array([100.0, 100.0, 90.0]),
        "down_reject": np.array([100.0, 100.0, 90.0]),
    }
    panel = _panel(close, open_by_symbol=open_price)

    for key in ("codex_inject_l4_reversal_a01", "codex_seed_l7_bar_shape_03"):
        values = _final_values(_cards()[key].expression, panel)
        assert values.idxmax() == "down_reject"
        assert values.idxmin() == "up_reject"
        assert values["down_reject"] > values["down_same"]
        assert values["up_reject"] < values["up_same"]

    confirmation = _final_values(
        _cards()["codex_inject_l7_bar_d01"].expression,
        panel,
    )
    assert confirmation.idxmax() == "up_same"
    assert confirmation.idxmin() == "down_same"
    assert confirmation["up_same"] > confirmation["up_reject"]
    assert confirmation["down_same"] < confirmation["down_reject"]


def test_tail_downside_halfwave_excludes_positive_returns():
    positive = np.empty(10, dtype=float)
    negative = np.empty(10, dtype=float)
    positive[0] = negative[0] = 100.0
    for position in range(1, 10):
        positive[position] = positive[position - 1] * 1.01
        negative[position] = negative[position - 1] * 0.99
    panel = _panel({"positive_only": positive, "negative_only": negative})
    card = _cards()["codex_inject_l2_tail_risk_a01"]
    values = _final_values(card.expression, panel)
    assert card.expected_direction == -1
    assert values["negative_only"] > values["positive_only"]


def test_drawdown_depth_gate_is_bounded_and_monotone_for_equal_recovery():
    targets = {"shallow": 95.0, "medium": 80.0, "deep": 60.0}
    paths: dict[str, np.ndarray] = {}
    for symbol, target in targets.items():
        values = np.full(60, 100.0, dtype=float)
        base = target / 1.06
        values[16:54] = np.linspace(100.0, base, 38)
        values[54:60] = np.linspace(base, target, 7)[1:]
        paths[symbol] = values
    panel = _panel(paths)
    values = _final_values(_cards()["codex_inject_l7_composite_e01"].expression, panel)
    assert 0.0 < values["shallow"] < values["medium"] < values["deep"] < 0.06


@pytest.mark.parametrize(
    ("key", "scale", "panel"),
    [
        (
            "codex_inject_l6_regime_a01",
            0.015,
            lambda: _panel(
                {
                    "asset": np.fromiter(
                        (
                            value
                            for value in np.multiply.accumulate(
                                [100.0]
                                + [1.015 if position % 2 else 0.985 for position in range(1, 40)]
                            )
                        ),
                        dtype=float,
                    )
                }
            ),
        ),
        (
            "codex_inject_l6_regime_e01",
            0.01,
            lambda: _panel(
                {"asset": np.full(40, 100.0)},
                open_by_symbol={"asset": np.r_[100.0, np.full(39, 101.0)]},
            ),
        ),
        (
            "codex_inject_l6_stability_b01",
            0.02,
            lambda: _stability_half_life_panel(),
        ),
        (
            "codex_inject_l7_composite_b01",
            0.005,
            lambda: _panel(
                {"asset": np.full(40, 100.0)},
                vwap_by_symbol={
                    "asset": np.array(
                        [100.0 / (1.005 if position % 2 else 0.995) for position in range(40)]
                    )
                },
            ),
        ),
    ],
)
def test_continuous_confidence_gates_reach_half_weight_at_declared_scale(
    key: str,
    scale: float,
    panel,
):
    resolved_panel = panel()
    gate = _find_half_gate(_cards()[key].expression, scale)
    value = _final_values(gate, resolved_panel)["asset"]
    assert value == pytest.approx(0.5, abs=1e-10)


def _stability_half_life_panel() -> pd.DataFrame:
    close = np.full(40, 100.0, dtype=float)
    for position in range(5, 40):
        return_5d = 0.02 if position % 2 else -0.02
        close[position] = close[position - 5] * (1.0 + return_5d)
    return _panel({"asset": close})


def test_order_imbalance_response_and_dual_signature_branches_are_comparable_scale():
    close_location_target = 0.5
    high = 102.0
    low = 98.0
    vwap = 102.0
    close = (high + low + close_location_target * (high - low + 0.0001 * vwap)) / 2.0
    b_panel = _panel(
        {"asset": np.r_[np.full(9, 100.0), close]},
        open_by_symbol={"asset": np.full(10, 100.0)},
        vwap_by_symbol={"asset": np.r_[np.full(9, 100.0), vwap]},
        high_by_symbol={"asset": np.r_[np.full(9, 101.0), high]},
        low_by_symbol={"asset": np.r_[np.full(9, 99.0), low]},
    )
    b_expression = _cards()["codex_inject_l3_order_imbalance_b01"].expression
    latent_sub = b_expression["args"][0]["args"][0]
    assert latent_sub["op"] == "sub"
    close_location, bounded_response = latent_sub["args"]
    close_value = _final_values(close_location, b_panel)["asset"]
    response_value = _final_values(bounded_response, b_panel)["asset"]
    assert close_value == pytest.approx(0.5, abs=1e-10)
    assert response_value == pytest.approx(0.5, abs=1e-10)

    denominator = 1.0 - 0.000025 / 1.005
    common_close = 101.0 / denominator
    common_vwap = common_close / 1.005
    c_panel = _panel(
        {"asset": np.full(40, common_close)},
        vwap_by_symbol={"asset": np.full(40, common_vwap)},
        high_by_symbol={"asset": np.full(40, 102.0)},
        low_by_symbol={"asset": np.full(40, 98.0)},
    )
    c_expression = _cards()["codex_inject_l3_order_imbalance_c01"].expression
    add = c_expression["args"][0]
    assert add["op"] == "add"
    branch_values = [
        _final_values(branch, c_panel)["asset"] for branch in add["args"]
    ]
    assert branch_values[0] == pytest.approx(0.5, abs=2e-4)
    assert branch_values[1] == pytest.approx(0.5, abs=2e-4)


def test_style_repair_cards_use_honest_formula_heuristics_and_evaluator_projection():
    style_cards = [
        card
        for card in _cards().values()
        if card.source_file.endswith("codex_injection/l1_l4.jsonl")
        and card.candidate_key.endswith("_d01")
    ]
    assert len(style_cards) == 13
    for card in style_cards:
        assert "repair_style_robustification" in card.mechanism_tags
        assert not _contains_pseudo_cross_sectional_residual(card.expression)
        narrative = " ".join(
            (
                card.hypothesis,
                card.economic_mechanism,
                card.falsification_test,
                card.parameters_rationale,
                " ".join(card.mechanism_tags),
            )
        )
        assert "Evaluator" in card.economic_mechanism
        assert "PreparedNeutralizer" in card.economic_mechanism
        assert not any(term in narrative for term in ("Beta", "β", "Barra", "残差"))


def test_seed_intraday_stress_absorption_only_activates_decline_plus_range_lead():
    daily_returns = {
        "down_range_lead": [-0.01] * 5,
        "down_range_lag": [-0.04, 0.02, -0.04, 0.02, -0.04],
        "up_range_lead": [0.01] * 5,
        "up_range_lag": [0.04, -0.02, 0.04, -0.02, 0.04],
    }
    close: dict[str, np.ndarray] = {}
    high: dict[str, np.ndarray] = {}
    low: dict[str, np.ndarray] = {}
    for symbol, returns in daily_returns.items():
        path = [100.0]
        for value in returns:
            path.append(path[-1] * (1.0 + value))
        close[symbol] = np.array(path)
        width = 0.02 if symbol.endswith("lead") else 0.001
        high[symbol] = close[symbol] * (1.0 + width / 2.0)
        low[symbol] = close[symbol] * (1.0 - width / 2.0)
    panel = _panel(close, high_by_symbol=high, low_by_symbol=low)
    values = _final_values(
        _cards()["L1VR_004_intraday_stress_absorption"].expression,
        panel,
    )
    assert values.idxmax() == "down_range_lead"
    inactive = values.drop("down_range_lead")
    assert inactive.max() == pytest.approx(inactive.min(), abs=1e-12)


def test_seed_gap_absorption_is_zero_for_same_sign_and_directed_for_rejection():
    cases = {
        "up_same": (0.10, 0.05),
        "up_reject": (0.10, -0.05),
        "down_same": (-0.10, -0.05),
        "down_reject": (-0.10, 0.05),
    }
    close: dict[str, np.ndarray] = {}
    open_price: dict[str, np.ndarray] = {}
    for symbol, (gap, body) in cases.items():
        closes = np.full(45, 100.0, dtype=float)
        opens = np.full(45, 100.0, dtype=float)
        for position in range(42, 45):
            opens[position] = closes[position - 1] * (1.0 + gap)
            closes[position] = opens[position] * (1.0 + body)
        close[symbol] = closes
        open_price[symbol] = opens
    panel = _panel(close, open_by_symbol=open_price)
    values = _final_values(
        _cards()["codex_l3_order_imbalance_02"].expression,
        panel,
    )
    assert values["down_reject"] > 0.0
    assert values["up_reject"] < 0.0
    assert values["down_same"] == pytest.approx(0.0, abs=1e-12)
    assert values["up_same"] == pytest.approx(0.0, abs=1e-12)


def test_seed_underreaction_interactions_cannot_become_negative_momentum_alone():
    close = np.fromiter((100.0 * 0.995**position for position in range(70)), dtype=float)
    open_price = np.r_[close[0], close[:-1]]
    high = close * 1.01
    low = close * 0.99
    panel = _panel(
        {"asset": close},
        open_by_symbol={"asset": open_price},
        vwap_by_symbol={"asset": close},
        high_by_symbol={"asset": high},
        low_by_symbol={"asset": low},
    )
    keys = (
        "codex_l3_order_imbalance_04",
        "codex_l4_lag_response_01",
        "codex_l4_lag_response_02",
        "codex_l4_lag_response_03",
    )
    for key in keys:
        card = _cards()[key]
        assert card.expression["op"] == "cs_rank"
        raw_interaction = card.expression["args"][0]
        assert raw_interaction["op"] == "mul"
        value = _final_values(raw_interaction, panel)["asset"]
        assert value == pytest.approx(0.0, abs=1e-12)
        narrative = " ".join(
            (card.hypothesis, card.economic_mechanism, " ".join(card.mechanism_tags))
        ).lower()
        assert "residual" not in narrative


def test_seed_disagreement_and_range_instability_gates_have_real_half_saturation():
    coherence_close = np.full(41, 100.0, dtype=float)
    coherence_close[-6] = 100.0 / 1.05
    coherence_panel = _panel({"asset": coherence_close})
    coherence_gate = _find_half_gate(
        _cards()["codex_l3_price_volume_coherence_04"].expression,
        0.05,
    )
    assert _final_values(coherence_gate, coherence_panel)["asset"] == pytest.approx(
        0.5,
        abs=1e-10,
    )

    range_close = np.full(25, 100.0, dtype=float)
    normalized_ranges = np.array(
        [0.01 if position % 2 else 0.03 for position in range(25)]
    )
    range_panel = _panel(
        {"asset": range_close},
        high_by_symbol={"asset": range_close * (1.0 + normalized_ranges / 2.0)},
        low_by_symbol={"asset": range_close * (1.0 - normalized_ranges / 2.0)},
    )
    range_gate = _find_half_gate(
        _cards()["codex_l4_range_vol_01"].expression,
        0.01,
    )
    assert _final_values(range_gate, range_panel)["asset"] == pytest.approx(
        0.5,
        abs=1e-10,
    )


def test_seed_reversal_requires_negative_shock_and_positive_stabilization():
    cases = {
        "down_positive_stabilization": (90.0, 0.005),
        "down_negative_stabilization": (90.0, -0.005),
        "up_positive_stabilization": (110.0, 0.005),
        "up_negative_stabilization": (110.0, -0.005),
    }
    close: dict[str, np.ndarray] = {}
    vwap: dict[str, np.ndarray] = {}
    for symbol, (target, deviation) in cases.items():
        path = np.full(15, 100.0, dtype=float)
        path[5:] = np.linspace(100.0, target, 10)
        close[symbol] = path
        vwap[symbol] = path / (1.0 + deviation)
    panel = _panel(close, vwap_by_symbol=vwap)
    values = _final_values(_cards()["codex_l4_reversal_02"].expression, panel)
    target = "down_positive_stabilization"
    assert values.idxmax() == target
    inactive = values.drop(target)
    assert inactive.max() == pytest.approx(inactive.min(), abs=1e-12)


def test_seed_bar_shape_uses_role_permitted_volume_not_amount():
    card = _cards()["codex_seed_l7_bar_shape_01"]

    def fields(node):
        found = {node["name"]} if node.get("op") == "field" else set()
        for child in node.get("args", []):
            found.update(fields(child))
        return found

    used = fields(card.expression)
    assert "volume" in used
    assert "amount" not in used
