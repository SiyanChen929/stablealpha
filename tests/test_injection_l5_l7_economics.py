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
INJECTION_PATH = PROJECT / "proposals" / "codex_injection" / "l5_l7.jsonl"
ALL_PROPOSAL_PATHS = (
    PROJECT / "proposals" / "codex_seed" / "l1_l2.jsonl",
    PROJECT / "proposals" / "codex_seed" / "l3_l4.jsonl",
    PROJECT / "proposals" / "codex_seed" / "l5_l7.jsonl",
    PROJECT / "proposals" / "codex_injection" / "l1_l4.jsonl",
    INJECTION_PATH,
)


@lru_cache(maxsize=1)
def _cards() -> dict[str, FactorCard]:
    cards = load_factor_cards(
        ALL_PROPOSAL_PATHS,
        expected_horizon=10,
        expected_count=176,
    )
    return {card.candidate_key: card for card in cards}


def _panel(
    close_by_symbol: dict[str, np.ndarray | list[float]],
    *,
    open_by_symbol: dict[str, np.ndarray | list[float]] | None = None,
    high_by_symbol: dict[str, np.ndarray | list[float]] | None = None,
    low_by_symbol: dict[str, np.ndarray | list[float]] | None = None,
    vwap_by_symbol: dict[str, np.ndarray | list[float]] | None = None,
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
            amount = float(
                1_000.0 * vwap
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
                    "volume": 1_000.0,
                    "amount": amount,
                }
            )
    return pd.DataFrame.from_records(records).set_index(["date", "symbol"])


def _final_values(expression: dict[str, object], panel: pd.DataFrame) -> pd.Series:
    values = evaluate_expression(expression, panel, PRODUCTION_DSL_SPEC)
    return values.xs(panel.index.get_level_values("date").max(), level="date")


def _find_node(expression: dict[str, object], predicate) -> dict[str, object]:
    if predicate(expression):
        return expression
    for child in expression.get("args", []):
        try:
            return _find_node(child, predicate)
        except LookupError:
            pass
    raise LookupError("requested expression node is absent")


def test_drawdown_persistence_uses_one_current_peak_anchor_for_both_paths():
    card = _cards()["codex_inject_l5_drawdown_b01"]
    fast, slow = card.expression["args"]
    assert fast["op"] == slow["op"] == "div"
    assert fast["args"][1] == slow["args"][1] == {
        "op": "ts_max",
        "args": [{"op": "field", "name": "close"}],
        "window": 40,
    }
    assert fast["args"][0]["window"] == 5
    assert slow["args"][0]["window"] == 20

    close = np.r_[np.linspace(80.0, 120.0, 20), np.linspace(119.0, 90.0, 30)]
    panel = _panel({"asset": close})
    actual = _final_values(card.expression, panel)["asset"]
    expected = (close[-5:].mean() - close[-20:].mean()) / close[-40:].max()
    assert actual == pytest.approx(expected, abs=1e-12)


def test_style_repairs_are_honest_self_or_relative_horizon_mechanisms():
    drawdown = _cards()["codex_inject_l5_drawdown_c01"]
    herding = _cards()["codex_inject_l5_herding_c01"]
    for card, forbidden_tag in (
        (drawdown, "style_residual"),
        (herding, "cross_sectional_residual"),
    ):
        assert forbidden_tag not in card.mechanism_tags
        assert "Evaluator" in card.economic_mechanism
        assert "PreparedCrossSectionalNeutralizer" in card.economic_mechanism

    close = np.r_[np.full(30, 100.0), np.linspace(100.0, 70.0, 50)]
    panel = _panel({"asset": close})
    value = _final_values(drawdown.expression, panel)["asset"]
    assert -1.0 < value < 0.0


def test_low_volume_anomaly_is_not_treated_as_crowding():
    length = 32
    close = {
        "low": np.linspace(100.0, 102.0, length),
        "normal": np.linspace(100.0, 104.0, length),
        "high": np.linspace(100.0, 106.0, length),
    }
    amount = {
        "low": np.r_[np.full(length - 1, 1_000.0), 100.0],
        "normal": np.full(length, 1_000.0),
        "high": np.r_[np.full(length - 1, 1_000.0), 2_000.0],
    }
    panel = _panel(close, amount_by_symbol=amount)
    gate = _find_node(
        _cards()["codex_inject_l5_herding_d01"].expression,
        lambda node: node.get("op") == "sqrt",
    )
    values = _final_values(gate, panel)
    assert values["low"] == pytest.approx(0.0, abs=1e-12)
    assert values["normal"] == pytest.approx(0.0, abs=1e-12)
    assert values["high"] > 0.0


def test_directed_gap_rejection_has_the_required_four_quadrants():
    close = {
        "down_recover": np.array([100.0, 100.0, 95.0]),
        "up_reject": np.array([100.0, 100.0, 105.0]),
        "down_same": np.array([100.0, 100.0, 85.0]),
        "up_same": np.array([100.0, 100.0, 115.0]),
    }
    open_price = {
        "down_recover": np.array([100.0, 100.0, 90.0]),
        "up_reject": np.array([100.0, 100.0, 110.0]),
        "down_same": np.array([100.0, 100.0, 90.0]),
        "up_same": np.array([100.0, 100.0, 110.0]),
    }
    panel = _panel(close, open_by_symbol=open_price)
    values = _final_values(
        _cards()["codex_inject_l7_bar_f01"].expression,
        panel,
    )
    assert values["down_recover"] > 0.0
    assert values["up_reject"] < 0.0
    assert values["down_same"] == pytest.approx(0.0, abs=1e-12)
    assert values["up_same"] == pytest.approx(0.0, abs=1e-12)


def _curvature_half_saturation_panel() -> pd.DataFrame:
    length = 32
    curvature = np.zeros(length, dtype=float)
    half_scale = 0.01
    # The production DSL deliberately uses population rolling variance.
    amplitude = half_scale
    curvature[-12:] = np.tile(np.array([-amplitude, amplitude]), 6)

    close = np.full(length, 100.0, dtype=float)
    for position in range(6, length):
        previous_ratio = close[position - 3] / close[position - 6]
        current_ratio = previous_ratio + curvature[position]
        assert current_ratio > 0.0
        close[position] = close[position - 3] * current_ratio
    return _panel({"asset": close})


def test_curvature_stability_gate_reaches_half_weight_at_one_percent():
    gate = _find_node(
        _cards()["codex_inject_l7_creative_b01"].expression,
        lambda node: node.get("op") == "div"
        and node.get("args", [{}])[0] == {"op": "const", "value": 0.01},
    )
    assert gate["op"] == "div"
    assert gate["args"][0] == {"op": "const", "value": 0.01}
    panel = _curvature_half_saturation_panel()
    value = _final_values(gate, panel)["asset"]
    assert value == pytest.approx(0.5, abs=1e-10)


def _pressure_correlation_panel(kind: str) -> pd.DataFrame:
    length = 32
    amount = np.r_[np.full(length - 7, 1_000.0), np.linspace(700.0, 1_500.0, 7)]
    rolling_mean = pd.Series(amount).rolling(20, min_periods=20).mean().to_numpy()
    participation = amount / (rolling_mean + 1.0)
    x = participation[-7:] - participation[-7:].mean()
    if kind == "positive":
        pressure = x
    elif kind == "negative":
        pressure = -x
    elif kind == "zero":
        base = np.array([1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 0.0])
        base -= base.mean()
        pressure = base - np.dot(base, x) / np.dot(x, x) * x
        pressure -= pressure.mean()
    else:  # pragma: no cover - test helper contract
        raise ValueError(kind)
    pressure *= 0.01 / np.max(np.abs(pressure))
    full_pressure = np.r_[np.zeros(length - 7), pressure]
    vwap = np.full(length, 100.0)
    close = vwap * (1.0 + full_pressure)
    return _panel(
        {"asset": close},
        vwap_by_symbol={"asset": vwap},
        amount_by_symbol={"asset": amount},
    )


def test_positive_correlation_gate_preserves_direction_and_closes_nonpositive_states():
    gate = _find_node(
        _cards()["codex_inject_l7_creative_e01"].expression,
        lambda node: node.get("op") == "div"
        and node.get("args", [{}, {}])[1] == {"op": "const", "value": 2},
    )
    positive = _final_values(gate, _pressure_correlation_panel("positive"))["asset"]
    zero = _final_values(gate, _pressure_correlation_panel("zero"))["asset"]
    negative = _final_values(gate, _pressure_correlation_panel("negative"))["asset"]
    assert positive == pytest.approx(1.0, abs=1e-10)
    assert zero == pytest.approx(0.0, abs=1e-10)
    assert negative == pytest.approx(0.0, abs=1e-10)


def test_repaired_library_keeps_full_count_and_structural_uniqueness():
    cards = load_factor_cards(
        ALL_PROPOSAL_PATHS,
        expected_horizon=10,
        expected_count=176,
    )
    assert len({card.candidate_key for card in cards}) == 176
    injection = [card for card in cards if "codex_injection" in card.source_file]
    assert len(injection) == 96
