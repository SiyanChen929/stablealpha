"""Cross-sectional factor diagnostics with time-series-aware inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DATE_LEVEL = "date"
SYMBOL_LEVEL = "symbol"


@dataclass(frozen=True)
class ICResult:
    mean: float
    median: float
    std: float
    icir: float
    win_rate: float
    observations: int
    average_names: float
    hac_t: float
    bootstrap_low: float
    bootstrap_high: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def require_panel_index(obj: pd.Series | pd.DataFrame) -> None:
    if not isinstance(obj.index, pd.MultiIndex):
        raise ValueError("panel data must use a MultiIndex(date, symbol)")
    missing = {DATE_LEVEL, SYMBOL_LEVEL}.difference(obj.index.names)
    if missing:
        raise ValueError(f"panel index is missing levels: {sorted(missing)}")
    dates = obj.index.get_level_values(DATE_LEVEL)
    if obj.index.has_duplicates:
        raise ValueError("panel index must be unique")
    if not dates.is_monotonic_increasing:
        raise ValueError("panel must be sorted by date")


def normalize_trading_calendar(
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex | None,
    *,
    panel_index: pd.MultiIndex | None = None,
    allow_test_calendar_from_panel: bool = False,
) -> pd.DatetimeIndex:
    """Validate the session calendar used for every forward endpoint lookup.

    Production callers must pass the calendar frozen by the data manifest.  A
    panel-derived calendar is available only behind an explicit test flag; it
    is useful for self-contained synthetic fixtures but is not a valid real-
    data provenance source.
    """

    if trading_calendar is None:
        if not allow_test_calendar_from_panel or panel_index is None:
            raise ValueError(
                "forward labels require an explicit frozen trading calendar"
            )
        raw = pd.DatetimeIndex(
            panel_index.get_level_values(DATE_LEVEL)
        ).unique()
    else:
        raw = trading_calendar
    calendar = pd.DatetimeIndex(pd.to_datetime(raw, errors="coerce")).normalize()
    if (
        calendar.empty
        or calendar.hasnans
        or calendar.has_duplicates
        or not calendar.is_monotonic_increasing
    ):
        raise ValueError(
            "trading calendar must be valid, unique, and strictly increasing"
        )
    if panel_index is not None:
        dates = pd.DatetimeIndex(panel_index.get_level_values(DATE_LEVEL)).normalize()
        missing = dates.unique().difference(calendar)
        if len(missing):
            raise ValueError("panel contains dates outside the trading calendar")
    return calendar


def _calendar_endpoint_values(
    values: pd.Series,
    calendar: pd.DatetimeIndex,
    offset: int,
) -> pd.Series:
    """Look up ``(calendar[t + offset], symbol)`` without compressing gaps."""

    dates = pd.DatetimeIndex(values.index.get_level_values(DATE_LEVEL)).normalize()
    symbols = values.index.get_level_values(SYMBOL_LEVEL)
    positions = calendar.get_indexer(dates)
    if np.any(positions < 0):
        raise ValueError("panel contains dates outside the trading calendar")
    endpoints = positions + int(offset)
    valid = endpoints < len(calendar)
    requested_dates = np.full(len(values), np.datetime64("NaT"), dtype="datetime64[ns]")
    requested_dates[valid] = calendar.to_numpy(dtype="datetime64[ns]")[endpoints[valid]]
    requested = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(requested_dates), symbols],
        names=[DATE_LEVEL, SYMBOL_LEVEL],
    )
    looked_up = values.reindex(requested).to_numpy(dtype=float, na_value=np.nan)
    return pd.Series(looked_up, index=values.index, dtype=float)


def forward_return(
    price: pd.Series,
    horizon: int,
    *,
    entry_delay: int = 1,
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex | None = None,
    allow_test_calendar_from_panel: bool = False,
) -> pd.Series:
    """Return from ``t + entry_delay`` to ``t + entry_delay + horizon``.

    A signal computed after the close on t therefore uses ``entry_delay=1``.
    This definition intentionally leaves the last ``entry_delay+horizon`` rows
    per symbol missing, making label overlap visible at split boundaries.
    """

    require_panel_index(price)
    if horizon < 1 or entry_delay < 0:
        raise ValueError("horizon must be positive and entry_delay non-negative")
    calendar = normalize_trading_calendar(
        trading_calendar,
        panel_index=price.index,
        allow_test_calendar_from_panel=allow_test_calendar_from_panel,
    )
    entry = _calendar_endpoint_values(price, calendar, entry_delay)
    exit_ = _calendar_endpoint_values(price, calendar, entry_delay + horizon)
    out = exit_.div(entry).sub(1.0)
    out.name = f"fwd_{entry_delay}_{horizon}"
    return out


def _strict_boolean_series(values: pd.Series, *, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} contains missing values")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError(f"{name} must contain only boolean or numeric 0/1 values")
        return numeric.astype(bool)
    raise ValueError(f"{name} must use a boolean or numeric 0/1 dtype")


def endpoint_executable_forward_return(
    price: pd.Series,
    tradable: pd.Series,
    horizon: int,
    *,
    entry_delay: int = 1,
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex | None = None,
    allow_test_calendar_from_panel: bool = False,
    entry_blocked: pd.Series | None = None,
    exit_blocked: pd.Series | None = None,
    execution_status_known: pd.Series | None = None,
) -> pd.Series:
    """Immediate-endpoint diagnostic for a long-only VWAP label.

    This does not pretend to be a full order simulation: if the scheduled
    entry or exit cannot execute, the observation is explicitly missing.  The
    caller must report that no-fill rate rather than silently treating the
    valuation placeholder as a trade.  Delayed fills and frozen holdings belong
    to the event-driven portfolio simulator.
    """

    require_panel_index(price)
    if horizon < 1 or entry_delay < 0:
        raise ValueError("horizon must be positive and entry_delay non-negative")
    if not price.index.equals(tradable.index):
        raise ValueError("tradable must exactly align with price")
    can_trade = _strict_boolean_series(tradable, name="tradable")
    false = pd.Series(False, index=price.index, dtype=bool)
    true = pd.Series(True, index=price.index, dtype=bool)
    buy_block = (
        false
        if entry_blocked is None
        else _strict_boolean_series(entry_blocked, name="entry_blocked")
    )
    sell_block = (
        false
        if exit_blocked is None
        else _strict_boolean_series(exit_blocked, name="exit_blocked")
    )
    known = (
        true
        if execution_status_known is None
        else _strict_boolean_series(
            execution_status_known,
            name="execution_status_known",
        )
    )
    for name, values in (
        ("entry_blocked", buy_block),
        ("exit_blocked", sell_block),
        ("execution_status_known", known),
    ):
        if not values.index.equals(price.index):
            raise ValueError(f"{name} must exactly align with price")
    calendar = normalize_trading_calendar(
        trading_calendar,
        panel_index=price.index,
        allow_test_calendar_from_panel=allow_test_calendar_from_panel,
    )
    entry = _calendar_endpoint_values(price, calendar, entry_delay)
    exit_ = _calendar_endpoint_values(price, calendar, entry_delay + horizon)
    endpoint_status = (can_trade & ~buy_block & known).astype(float)
    entry_ok = _calendar_endpoint_values(
        endpoint_status, calendar, entry_delay
    ).eq(1.0)
    endpoint_status = (can_trade & ~sell_block & known).astype(float)
    exit_ok = _calendar_endpoint_values(
        endpoint_status, calendar, entry_delay + horizon
    ).eq(1.0)
    result = exit_.div(entry).sub(1.0).where(entry_ok & exit_ok)
    result.name = f"executable_endpoint_fwd_{entry_delay}_{horizon}"
    return result


def _safe_corr(x: pd.Series, y: pd.Series, method: str, min_names: int) -> tuple[float, int]:
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    if n < min_names or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan, n
    return float(pair["x"].corr(pair["y"], method=method)), n


def daily_ic(
    scores: pd.Series,
    labels: pd.Series,
    *,
    method: str = "spearman",
    min_names: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Produce one IC observation per date, the correct inference unit."""

    require_panel_index(scores)
    require_panel_index(labels)
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    frame = pd.concat([scores.rename("score"), labels.rename("label")], axis=1)
    # The original implementation materialized two Series plus a DataFrame for
    # every date, then asked pandas to rediscover the same finite-pair mask.
    # A full factor search invokes this routine millions of times through the
    # novelty archive.  Build the aligned numeric arrays once and call the same
    # correlation kernels used by ``Series.corr`` on each contiguous group.
    # In particular, retaining scipy.stats.spearmanr (instead of an algebraic
    # batch approximation) preserves the exact floating-point result and tie
    # handling of the reference path.
    score_values = frame["score"].replace(
        [np.inf, -np.inf], np.nan
    ).to_numpy(dtype=float, na_value=np.nan)
    label_values = frame["label"].replace(
        [np.inf, -np.inf], np.nan
    ).to_numpy(dtype=float, na_value=np.nan)
    dates = frame.index.get_level_values(DATE_LEVEL)
    codes, unique_dates = pd.factorize(dates, sort=True)
    included = np.flatnonzero(codes >= 0)
    ordered = included[np.argsort(codes[included], kind="stable")]
    ordered_codes = codes[ordered]
    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.diff(ordered_codes)).astype(np.int64) + 1,
            np.array([len(ordered_codes)], dtype=np.int64),
        )
    )
    values = np.full(len(unique_dates), np.nan, dtype=float)
    counts = np.zeros(len(unique_dates), dtype=float)
    rankdata = None
    if method == "spearman":
        # scipy.stats.spearmanr computes the statistic by applying rankdata to
        # the two-column input and passing that exact layout to np.corrcoef.
        # Reproduce those operations byte-for-byte, while omitting the unused
        # t-distribution p-value calculation that dominates small groups.
        from scipy.stats import rankdata as scipy_rankdata

        rankdata = scipy_rankdata
    for position in range(len(unique_dates)):
        locations = ordered[boundaries[position] : boundaries[position + 1]]
        left = score_values[locations]
        right = label_values[locations]
        finite = np.isfinite(left) & np.isfinite(right)
        count = int(finite.sum())
        counts[position] = count
        if count < min_names:
            continue
        left = left[finite]
        right = right[finite]
        if np.unique(left).size < 2 or np.unique(right).size < 2:
            continue
        if method == "spearman":
            assert rankdata is not None
            paired = np.column_stack((left, right))
            ranked = np.apply_along_axis(rankdata, 0, paired)
            values[position] = float(
                np.corrcoef(ranked, rowvar=0)[1, 0]
            )
        else:
            values[position] = float(np.corrcoef(left, right)[0, 1])
    return (
        pd.Series(values, index=unique_dates, name=f"{method}_ic", dtype=float),
        pd.Series(counts, index=unique_dates, name="names", dtype=float),
    )


def newey_west_mean_se(values: Sequence[float] | pd.Series, lags: int) -> float:
    """HAC standard error of a sample mean using Bartlett weights."""

    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = x.size
    if n < 2:
        return np.nan
    lag = max(0, min(int(lags), n - 1))
    centered = x - x.mean()
    long_run = float(centered @ centered / n)
    for k in range(1, lag + 1):
        gamma = float(centered[k:] @ centered[:-k] / n)
        long_run += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    return float(np.sqrt(max(long_run, 0.0) / n))


def moving_block_bootstrap_mean(
    values: Sequence[float] | pd.Series,
    *,
    block_size: int = 60,
    samples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> tuple[float, float]:
    """Confidence interval that retains local dependence between IC dates."""

    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = x.size
    if n < 2:
        return np.nan, np.nan
    block = max(1, min(int(block_size), n))
    rng = np.random.default_rng(seed)
    possible = np.arange(0, n - block + 1)
    draws = np.empty(samples, dtype=float)
    blocks_needed = int(np.ceil(n / block))
    for i in range(samples):
        starts = rng.choice(possible, size=blocks_needed, replace=True)
        sample = np.concatenate([x[s : s + block] for s in starts])[:n]
        draws[i] = sample.mean()
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def holm_adjusted_pvalues(values: Sequence[float]) -> np.ndarray:
    """Holm step-down FWER adjustment in the input hypothesis order."""

    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    if ((~np.isfinite(raw)) | (raw < 0.0) | (raw > 1.0)).any():
        raise ValueError("p-values must be finite values in [0, 1]")
    count = len(raw)
    if count == 0:
        return raw.copy()
    order = np.argsort(raw, kind="mergesort")
    adjusted_sorted = np.empty(count, dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (count - rank) * raw[position])
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def moving_block_max_t_pvalues(
    values: pd.DataFrame,
    *,
    hac_lags: int = 9,
    block_size: int = 20,
    samples: int = 1_000,
    seed: int = 17,
) -> pd.Series:
    """One-sided, family-wise adjusted p-values for a frozen factor batch.

    Rows are the daily inference units and columns are hypotheses.  Blocks are
    sampled with one shared row draw across all factors, retaining both local
    time dependence and contemporaneous cross-factor dependence.  Each null
    series is centered before sampling and the maximum studentized mean across
    the family supplies the adjusted reference distribution.
    """

    if not isinstance(values, pd.DataFrame) or values.shape[1] == 0:
        raise ValueError("values must be a non-empty DataFrame")
    if samples < 20:
        raise ValueError("max-T requires at least 20 bootstrap samples")
    numeric = values.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    common = numeric.dropna(axis=0, how="any")
    if len(common) < 2:
        return pd.Series(1.0, index=values.columns, name="block_max_t_p")
    x = common.to_numpy(dtype=float)
    means = x.mean(axis=0)
    standard_errors = np.asarray(
        [newey_west_mean_se(x[:, column], hac_lags) for column in range(x.shape[1])],
        dtype=float,
    )
    usable = np.isfinite(standard_errors) & (standard_errors > 0.0)
    observed = np.full(x.shape[1], -np.inf, dtype=float)
    observed[usable] = means[usable] / standard_errors[usable]
    centered = x - means
    n = len(centered)
    block = max(1, min(int(block_size), n))
    possible = np.arange(0, n - block + 1)
    blocks_needed = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    maxima = np.empty(samples, dtype=float)
    for draw in range(samples):
        starts = rng.choice(possible, size=blocks_needed, replace=True)
        positions = np.concatenate(
            [np.arange(start, start + block) for start in starts]
        )[:n]
        sampled_means = centered[positions].mean(axis=0)
        sampled_t = np.full(x.shape[1], -np.inf, dtype=float)
        sampled_t[usable] = sampled_means[usable] / standard_errors[usable]
        maxima[draw] = float(np.max(sampled_t))
    adjusted = np.ones(x.shape[1], dtype=float)
    for column in np.flatnonzero(usable):
        adjusted[column] = (
            1.0 + float(np.count_nonzero(maxima >= observed[column]))
        ) / (samples + 1.0)
    return pd.Series(adjusted, index=values.columns, name="block_max_t_p")


def one_sided_normal_pvalue(test_statistic: float) -> float:
    """Upper-tail standard-normal p-value without a SciPy dependency."""

    if not math.isfinite(test_statistic):
        return 1.0
    return float(0.5 * math.erfc(test_statistic / math.sqrt(2.0)))


def summarize_ic(
    values: pd.Series,
    names: pd.Series | None = None,
    *,
    hac_lags: int = 9,
    block_size: int = 60,
    bootstrap_samples: int = 1_000,
    seed: int = 17,
) -> ICResult:
    clean = pd.Series(values, dtype=float).dropna()
    if clean.empty:
        return ICResult(*(np.nan,) * 5, observations=0, average_names=np.nan, hac_t=np.nan,
                        bootstrap_low=np.nan, bootstrap_high=np.nan)
    std = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    mean = float(clean.mean())
    se = newey_west_mean_se(clean, hac_lags)
    low, high = moving_block_bootstrap_mean(
        clean,
        block_size=block_size,
        samples=bootstrap_samples,
        seed=seed,
    )
    avg_names = float(names.reindex(clean.index).mean()) if names is not None else np.nan
    return ICResult(
        mean=mean,
        median=float(clean.median()),
        std=std,
        icir=float(mean / std) if std and np.isfinite(std) else np.nan,
        win_rate=float((clean > 0).mean()),
        observations=int(len(clean)),
        average_names=avg_names,
        hac_t=float(mean / se) if se and np.isfinite(se) else np.nan,
        bootstrap_low=low,
        bootstrap_high=high,
    )


@dataclass(frozen=True)
class _NeutralizationDateDesign:
    start: int
    stop: int
    x: pd.DataFrame
    exposure_valid: pd.Series
    exposure_valid_values: np.ndarray
    complete_x: np.ndarray
    complete_gram: np.ndarray


class PreparedCrossSectionalNeutralizer:
    """Reusable point-in-time risk design for many candidate score vectors.

    Industry dummies and standardized continuous exposures depend only on the
    frozen exposure panel, but rebuilding them for every factor makes a full
    2,384-candidate search unnecessarily expensive.  This object caches only
    that fixed design. Candidate-specific missing-score masks, the ridge solve,
    and the residual arithmetic are identical to the reference calculation.
    """

    def __init__(
        self,
        exposures: pd.DataFrame,
        *,
        categorical: Iterable[str] = ("industry",),
        min_names: int = 20,
    ) -> None:
        require_panel_index(exposures)
        self.index = exposures.index.copy()
        self.min_names = int(min_names)
        cats = set(categorical)
        designs: list[_NeutralizationDateDesign] = []
        offset = 0
        for _, group in exposures.groupby(level=DATE_LEVEL, sort=True):
            columns: list[pd.DataFrame | pd.Series] = []
            for name in group.columns:
                col = group[name]
                if name in cats:
                    columns.append(
                        pd.get_dummies(
                            col.astype("string"),
                            prefix=name,
                            drop_first=True,
                            dtype=float,
                        )
                    )
                else:
                    numeric = pd.to_numeric(col, errors="coerce")
                    scale = numeric.std(ddof=0)
                    columns.append(
                        ((numeric - numeric.mean()) / scale)
                        if scale and np.isfinite(scale)
                        else numeric * 0.0
                    )
            x = (
                pd.concat(columns, axis=1)
                if columns
                else pd.DataFrame(index=group.index)
            )
            x = x.replace([np.inf, -np.inf], np.nan)
            exposure_valid = x.notna().all(axis=1)
            exposure_valid_values = exposure_valid.to_numpy(dtype=bool)
            complete_x = np.column_stack(
                [
                    np.ones(int(exposure_valid_values.sum())),
                    x.loc[exposure_valid].to_numpy(dtype=float),
                ]
            )
            penalty = np.eye(complete_x.shape[1]) * 1e-8
            penalty[0, 0] = 0.0
            complete_gram = complete_x.T @ complete_x + penalty
            stop = offset + len(group)
            if not group.index.equals(self.index[offset:stop]):
                raise ValueError(
                    "exposure rows must be contiguous within each date"
                )
            designs.append(
                _NeutralizationDateDesign(
                    start=offset,
                    stop=stop,
                    x=x,
                    exposure_valid=exposure_valid,
                    exposure_valid_values=exposure_valid_values,
                    complete_x=complete_x,
                    complete_gram=complete_gram,
                )
            )
            offset = stop
        self._designs = tuple(designs)

    def transform(self, scores: pd.Series) -> pd.Series:
        require_panel_index(scores)
        aligned = scores.reindex(self.index).astype(float)
        residual_values = np.full(len(aligned), np.nan, dtype=float)
        for design in self._designs:
            y = aligned.iloc[design.start : design.stop]
            valid = y.replace([np.inf, -np.inf], np.nan).notna()
            valid &= design.exposure_valid
            if int(valid.sum()) >= max(
                self.min_names, design.x.shape[1] + 2
            ):
                valid_values = valid.to_numpy(dtype=bool)
                if np.array_equal(valid_values, design.exposure_valid_values):
                    xv = design.complete_x
                    gram = design.complete_gram
                else:
                    xv = np.column_stack(
                        [
                            np.ones(int(valid.sum())),
                            design.x.loc[valid].to_numpy(dtype=float),
                        ]
                    )
                    penalty = np.eye(xv.shape[1]) * 1e-8
                    penalty[0, 0] = 0.0
                    gram = xv.T @ xv + penalty
                yv = y.loc[valid].to_numpy(dtype=float)
                beta = np.linalg.solve(gram, xv.T @ yv)
                day_residual = np.full(len(y), np.nan, dtype=float)
                day_residual[valid_values] = yv - xv @ beta
                residual_values[design.start : design.stop] = day_residual
        result = pd.Series(residual_values, index=self.index, dtype=float).sort_index()
        result.name = f"{scores.name or 'score'}_neutral"
        return result


def cross_sectional_neutralize(
    scores: pd.Series,
    exposures: pd.DataFrame,
    *,
    categorical: Iterable[str] = ("industry",),
    min_names: int = 20,
) -> pd.Series:
    """Residualize scores using only the contemporaneous cross-section.

    Continuous columns are z-scored inside each date. Categorical columns are
    one-hot encoded without looking at any other date. Ridge stabilization is
    used only for the small normal equation, not as a trained predictive model.
    """

    return PreparedCrossSectionalNeutralizer(
        exposures,
        categorical=categorical,
        min_names=min_names,
    ).transform(scores)


def rank_turnover(scores: pd.Series) -> float:
    """Average absolute day-to-day percentile-rank change."""

    require_panel_index(scores)
    ranks = scores.groupby(level=DATE_LEVEL).rank(pct=True)
    changes = ranks.groupby(level=SYMBOL_LEVEL).diff().abs()
    return float(changes.groupby(level=DATE_LEVEL).mean().mean())


def factor_correlation(a: pd.Series, b: pd.Series, *, min_names: int = 20) -> float:
    series, _ = daily_ic(a, b, method="spearman", min_names=min_names)
    return float(series.mean())
