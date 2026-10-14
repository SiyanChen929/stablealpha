"""Seeded synthetic point-in-time market used only for integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_market(
    *,
    days: int = 520,
    symbols: int = 80,
    seed: int = 7,
    start: str = "2018-01-02",
) -> pd.DataFrame:
    """Create an OHLCV panel with weak, known nonlinear mechanisms.

    Tomorrow's idiosyncratic return contains both a short-reversal term and a
    volume-confirmed medium-horizon term. This is not a performance demo; it is
    a deterministic fixture that lets leakage, evolution and IC code fail loud.
    """

    if days < 80 or symbols < 20:
        raise ValueError("synthetic fixture needs at least 80 days and 20 symbols")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days, name="date")
    names = pd.Index([f"S{i:04d}" for i in range(symbols)], name="symbol")
    beta = rng.uniform(0.7, 1.3, symbols)
    size = rng.normal(22.5, 1.0, symbols)
    industry = np.array([f"I{i % 10:02d}" for i in range(symbols)])
    base_volume = np.exp(rng.normal(14.0, 0.65, symbols))

    returns = np.zeros((days, symbols), dtype=float)
    volume_shock = rng.normal(0.0, 1.0, (days, symbols))
    market = rng.normal(0.00015, 0.008, days)
    idio = rng.normal(0.0, 0.018, (days, symbols))
    for t in range(1, days):
        short_reversal = -(returns[t - 1] - returns[t - 1].mean())
        if t >= 6:
            medium = returns[t - 5 : t].sum(axis=0)
            medium -= medium.mean()
        else:
            medium = np.zeros(symbols)
        confirmation = medium * (volume_shock[t - 1] > 0.35)
        alpha = 0.10 * short_reversal + 0.035 * confirmation
        returns[t] = beta * market[t] + alpha + idio[t]
        returns[t] = np.clip(returns[t], -0.095, 0.095)

    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    overnight = rng.normal(0.0, 0.0025, (days, symbols))
    open_ = close * np.exp(overnight)
    spread = np.abs(rng.normal(0.007, 0.003, (days, symbols)))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * np.maximum(0.5, 1.0 - spread)
    volume = base_volume * np.exp(0.35 * volume_shock + 2.5 * np.abs(returns))
    amount = volume * (open_ + close) / 2.0

    index = pd.MultiIndex.from_product([dates, names], names=["date", "symbol"])
    frame = pd.DataFrame(
        {
            "open": open_.reshape(-1),
            "high": high.reshape(-1),
            "low": low.reshape(-1),
            "close": close.reshape(-1),
            "adj_open": open_.reshape(-1),
            "adj_high": high.reshape(-1),
            "adj_low": low.reshape(-1),
            "adj_close": close.reshape(-1),
            "volume": volume.reshape(-1),
            "amount": amount.reshape(-1),
            "vwap": (amount / volume).reshape(-1),
            "adj_vwap": (amount / volume).reshape(-1),
            "adj_factor": 1.0,
            "tradable": True,
            "vendor_universe_member": True,
            "universe_member": True,
            "research_eligible": True,
            "is_st": False,
            "listed_days": np.tile(np.arange(days), (symbols, 1)).T.reshape(-1) + 400,
            "industry": np.tile(industry, days),
            "log_mcap": np.tile(size, days) + np.repeat(np.cumsum(rng.normal(0, 0.003, (days, symbols)), axis=0), 1, axis=0).reshape(-1),
            "beta": np.tile(beta, days),
        },
        index=index,
    )
    return frame.sort_index()
