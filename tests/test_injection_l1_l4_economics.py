from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from cogalpha_plus.dsl import evaluate_expression
from cogalpha_plus.proposals import FactorCard, load_factor_cards
from cogalpha_plus.validation import PRODUCTION_DSL_SPEC


PROJECT = Path(__file__).resolve().parents[1]
INJECTION_PATH = PROJECT / "proposals" / "codex_injection" / "l1_l4.jsonl"


@lru_cache(maxsize=1)
def _cards() -> dict[str, FactorCard]:
    cards = load_factor_cards(
        (INJECTION_PATH,), expected_horizon=10, expected_count=52
    )
    return {card.candidate_key: card for card in cards}


def _series(start: float, returns: list[float]) -> np.ndarray:
    values = [float(start)]
    for daily_return in returns:
        values.append(values[-1] * (1.0 + daily_return))
    return np.asarray(values, dtype=float)


def _panel(
    close_by_symbol: dict[str, np.ndarray],
    *,
    open_by_symbol: dict[str, np.ndarray] | None = None,
    vwap_by_symbol: dict[str, np.ndarray] | None = None,
    high_by_symbol: dict[str, np.ndarray] | None = None,
    low_by_symbol: dict[str, np.ndarray] | None = None,
    volume_by_symbol: dict[str, np.ndarray] | None = None,
    amount_by_symbol: dict[str, np.ndarray] | None = None,
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


def _final(key: str, panel: pd.DataFrame) -> pd.Series:
    values = evaluate_expression(
        _cards()[key].expression, panel, PRODUCTION_DSL_SPEC
    )
    return values.xs(panel.index.get_level_values("date").max(), level="date")


def test_market_cycle_acceleration_uses_nonoverlapping_dailyized_segments():
    flat_then_fast = _series(100.0, [0.0] * 4 + [0.02] * 3)
    steady = _series(100.0, [0.01] * 7)
    fast_then_flat = _series(100.0, [0.02] * 4 + [0.0] * 3)
    panel = _panel(
        {
            "accelerating": flat_then_fast,
            "steady": steady,
            "decelerating": fast_then_flat,
        }
    )

    values = _final("codex_inject_l1_market_cycle_a01", panel)
    assert values["accelerating"] > values["steady"] > values["decelerating"]


def _volatility_consensus_panel(*, body_geometry: bool) -> pd.DataFrame:
    symbols = ("both", "first_only", "second_only", "neither")
    length = 30
    close: dict[str, np.ndarray] = {}
    open_price: dict[str, np.ndarray] = {}
    high: dict[str, np.ndarray] = {}
    low: dict[str, np.ndarray] = {}
    for symbol in symbols:
        first_high = symbol in {"both", "first_only"}
        second_high = symbol in {"both", "second_only"}
        if body_geometry:
            close_path = np.full(length, 100.0)
            body_ratio = np.ones(length)
            if second_high:
                body_ratio[-7:] = np.array([0.94, 1.06, 0.95, 1.05, 0.93, 1.07, 0.96])
            open_path = close_path / body_ratio
            wide_range = first_high
        else:
            daily = [0.0] * (length - 8)
            daily += (
                [0.05, -0.04, 0.06, -0.05, 0.04, -0.03, 0.05]
                if first_high
                else [0.0] * 7
            )
            close_path = _series(100.0, daily)
            open_path = close_path.copy()
            wide_range = second_high
        width = np.full(length, 0.02)
        if wide_range:
            width[-7:] = 0.14
        close[symbol] = close_path
        open_price[symbol] = open_path
        if body_geometry:
            # Keep the two synthetic geometry branches orthogonal.  The DSL
            # consumes raw OHLC fields independently; this deliberately
            # adversarial fixture is not intended to emulate an exchange bar.
            high[symbol] = close_path * (1.0 + width / 2.0)
            low[symbol] = close_path * (1.0 - width / 2.0)
        else:
            high[symbol] = np.maximum(close_path, open_path) * (1.0 + width / 2.0)
            low[symbol] = np.minimum(close_path, open_path) * (1.0 - width / 2.0)
    return _panel(
        close,
        open_by_symbol=open_price,
        high_by_symbol=high,
        low_by_symbol=low,
    )


def test_volatility_regime_requires_both_estimators_to_expand():
    values = _final(
        "codex_inject_l1_volatility_regime_c01",
        _volatility_consensus_panel(body_geometry=False),
    )
    assert values["both"] > values["first_only"]
    assert values["both"] > values["second_only"]
    assert values["first_only"] == values["second_only"] == values["neither"]


def test_fragile_rally_only_scores_up_and_low_participation_quadrant():
    symbols = ("up_low", "up_high", "down_low", "down_high")
    close: dict[str, np.ndarray] = {}
    volume: dict[str, np.ndarray] = {}
    for symbol in symbols:
        is_up = symbol.startswith("up")
        is_low = symbol.endswith("low")
        close[symbol] = _series(100.0, [0.0] * 19 + [0.02 if is_up else -0.02] * 5)
        volume[symbol] = np.r_[
            np.full(22, 100.0), np.full(3, 20.0 if is_low else 300.0)
        ]
    values = _final(
        "codex_inject_l2_crash_predictor_a01",
        _panel(close, volume_by_symbol=volume),
    )
    assert values["up_low"] > values["up_high"]
    assert values["up_high"] == values["down_low"] == values["down_high"]


def test_tail_consensus_is_constrained_by_the_weaker_downside_branch():
    symbols = ("both", "close_only", "vwap_only", "neither")
    close: dict[str, np.ndarray] = {}
    vwap: dict[str, np.ndarray] = {}
    for symbol in symbols:
        close_high = symbol in {"both", "close_only"}
        vwap_high = symbol in {"both", "vwap_only"}
        close[symbol] = _series(100.0, [0.0] * 23 + ([-0.02] * 16 if close_high else [0.0] * 16))
        vwap[symbol] = _series(100.0, [0.0] * 23 + ([-0.02] * 16 if vwap_high else [0.0] * 16))
    values = _final(
        "codex_inject_l2_tail_risk_c01",
        _panel(close, vwap_by_symbol=vwap),
    )
    assert values["both"] > values["close_only"]
    assert values["both"] > values["vwap_only"]
    assert values["close_only"] == values["vwap_only"] == values["neither"]


def test_price_volume_confirmation_preserves_direction_and_closes_negative_corr():
    symbols = ("up_positive", "up_negative", "down_positive", "down_negative")
    shape = np.asarray([0.004, 0.011, 0.006, 0.014, 0.008, 0.012, 0.007])
    close: dict[str, np.ndarray] = {}
    volume: dict[str, np.ndarray] = {}
    for symbol in symbols:
        up = symbol.startswith("up")
        positive = symbol.endswith("positive")
        price_returns = shape if up else -shape
        volume_returns = price_returns if positive else -price_returns
        close[symbol] = _series(100.0, [0.0] * 22 + price_returns.tolist())
        volume[symbol] = _series(1_000.0, [0.0] * 22 + volume_returns.tolist())
    values = _final(
        "codex_inject_l3_price_volume_coherence_a01",
        _panel(close, volume_by_symbol=volume),
    )
    assert values["up_positive"] > values["up_negative"]
    assert values["down_positive"] < values["down_negative"]
    assert values["up_negative"] == values["down_negative"]


def test_volume_burst_only_scores_upward_price_pressure_intersection():
    symbols = ("up_burst", "up_quiet", "down_burst", "down_quiet")
    close: dict[str, np.ndarray] = {}
    volume: dict[str, np.ndarray] = {}
    for symbol in symbols:
        up = symbol.startswith("up")
        burst = symbol.endswith("burst")
        close[symbol] = _series(100.0, [0.0] * 21 + [0.025 if up else -0.025] * 3)
        volume[symbol] = np.r_[
            np.full(22, 100.0), np.full(3, 300.0 if burst else 30.0)
        ]
    values = _final(
        "codex_inject_l3_volume_structure_a01",
        _panel(close, volume_by_symbol=volume),
    )
    assert values["up_burst"] > values["up_quiet"]
    assert values["up_quiet"] == values["down_burst"] == values["down_quiet"]


def test_joint_volume_amount_change_has_strength_and_price_direction():
    symbols = ("joint_up", "joint_down", "unchanged", "volume_only", "opposed")
    length = 45
    close: dict[str, np.ndarray] = {}
    volume: dict[str, np.ndarray] = {}
    amount: dict[str, np.ndarray] = {}
    for symbol in symbols:
        direction = -0.01 if symbol == "joint_down" else 0.01
        close[symbol] = _series(100.0, [0.0] * 35 + [direction] * 9)
        volume[symbol] = np.full(length, 100.0)
        amount[symbol] = np.full(length, 100.0)
        if symbol in {"joint_up", "joint_down", "volume_only", "opposed"}:
            volume[symbol][-9:] = 300.0
        if symbol in {"joint_up", "joint_down"}:
            amount[symbol][-9:] = 300.0
        elif symbol == "opposed":
            amount[symbol][-9:] = 30.0
    values = _final(
        "codex_inject_l3_volume_structure_c01",
        _panel(close, volume_by_symbol=volume, amount_by_symbol=amount),
    )
    assert values["joint_up"] > values["unchanged"] > values["joint_down"]
    assert values["unchanged"] == values["volume_only"] == values["opposed"]


def test_daily_trend_explicitly_penalizes_one_jump_at_equal_net_return():
    smooth = np.geomspace(100.0, 110.0, 11)
    one_jump = np.r_[np.full(10, 100.0), 110.0]
    values = _final(
        "codex_inject_l4_daily_trend_d01",
        _panel({"smooth": smooth, "one_jump": one_jump}),
    )
    assert values["smooth"] > values["one_jump"]


def test_lagged_gap_body_rejection_has_directed_four_quadrant_ordering():
    close = {
        "up_confirm": np.array([100.0, 115.0, 115.0]),
        "up_reject": np.array([100.0, 100.0, 100.0]),
        "down_confirm": np.array([100.0, 85.0, 85.0]),
        "down_reject": np.array([100.0, 100.0, 100.0]),
    }
    open_price = {
        "up_confirm": np.array([100.0, 110.0, 115.0]),
        "up_reject": np.array([100.0, 110.0, 100.0]),
        "down_confirm": np.array([100.0, 90.0, 85.0]),
        "down_reject": np.array([100.0, 90.0, 100.0]),
    }
    values = _final(
        "codex_inject_l4_lag_response_c01",
        _panel(close, open_by_symbol=open_price),
    )
    assert values["down_reject"] > values["up_confirm"]
    assert values["up_reject"] < values["down_confirm"]
    assert values["up_confirm"] == values["down_confirm"]


def test_range_geometry_requires_both_expansions_not_an_additive_substitute():
    values = _final(
        "codex_inject_l4_range_vol_c01",
        _volatility_consensus_panel(body_geometry=True),
    )
    assert values["both"] > values["first_only"]
    assert values["both"] > values["second_only"]
    assert values["first_only"] == values["second_only"] == values["neither"]
