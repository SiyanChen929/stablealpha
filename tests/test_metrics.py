import numpy as np
import pandas as pd

import cogalpha_plus.metrics as metrics_module
from cogalpha_plus.metrics import (
    PreparedCrossSectionalNeutralizer,
    cross_sectional_neutralize,
    daily_ic,
    endpoint_executable_forward_return,
    forward_return,
    holm_adjusted_pvalues,
    moving_block_bootstrap_mean,
    moving_block_max_t_pvalues,
    one_sided_normal_pvalue,
    summarize_ic,
)
from cogalpha_plus.synthetic import make_synthetic_market


def _calendar(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        panel.index.get_level_values("date").unique()
    ).sort_values()


def test_forward_return_respects_entry_delay():
    panel = make_synthetic_market(days=90, symbols=20)
    result = forward_return(
        panel["close"], horizon=5, entry_delay=1, trading_calendar=_calendar(panel)
    )
    one = panel.xs("S0000", level="symbol")["close"]
    date = one.index[10]
    expected = one.iloc[16] / one.iloc[11] - 1.0
    assert np.isclose(result.loc[(date, "S0000")], expected)


def test_forward_return_uses_exact_calendar_endpoint_and_never_compresses_gap():
    panel = make_synthetic_market(days=90, symbols=20)
    calendar = _calendar(panel)
    symbol = "S0000"
    signal_date = calendar[10]
    missing_endpoint = calendar[13]
    # entry=11, exit=13 for horizon=2. Removing the official endpoint must not
    # silently substitute this symbol's next observation on calendar[14].
    price = panel["close"].drop(index=(missing_endpoint, symbol))
    result = forward_return(
        price, horizon=2, entry_delay=1, trading_calendar=calendar
    )
    assert np.isnan(result.loc[(signal_date, symbol)])


def test_forward_return_rejects_duplicate_or_out_of_order_calendar():
    panel = make_synthetic_market(days=90, symbols=20)
    calendar = _calendar(panel)
    duplicate = calendar.insert(3, calendar[2])
    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        forward_return(panel["close"], 1, trading_calendar=duplicate)
    out_of_order = calendar.to_list()
    out_of_order[2], out_of_order[3] = out_of_order[3], out_of_order[2]
    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        forward_return(panel["close"], 1, trading_calendar=out_of_order)


def test_executable_endpoint_label_never_uses_no_trade_vwap_placeholder():
    panel = make_synthetic_market(days=90, symbols=20)
    tradable = panel["tradable"].copy()
    dates = panel.index.get_level_values("date").unique().sort_values()
    symbol = "S0000"
    signal_date = dates[10]
    entry_date = dates[11]
    tradable.loc[(entry_date, symbol)] = False
    result = endpoint_executable_forward_return(
        panel["adj_vwap"],
        tradable,
        horizon=5,
        entry_delay=1,
        trading_calendar=_calendar(panel),
    )
    mark_to_market = forward_return(
        panel["adj_vwap"], horizon=5, entry_delay=1, trading_calendar=_calendar(panel)
    )
    assert np.isfinite(mark_to_market.loc[(signal_date, symbol)])
    assert np.isnan(result.loc[(signal_date, symbol)])

    invalid = tradable.astype(object)
    invalid.iloc[0] = "False"
    with np.testing.assert_raises(ValueError):
        endpoint_executable_forward_return(
            panel["adj_vwap"], invalid, horizon=5, entry_delay=1,
            trading_calendar=_calendar(panel),
        )


def test_ic_summary_and_block_bootstrap_are_deterministic():
    panel = make_synthetic_market(days=100, symbols=25)
    labels = forward_return(panel["close"], 1, trading_calendar=_calendar(panel))
    scores = labels.copy()
    ic, names = daily_ic(scores, labels, min_names=20)
    summary = summarize_ic(ic, names, bootstrap_samples=100, block_size=10)
    assert summary.mean > 0.99
    assert summary.win_rate == 1.0
    assert moving_block_bootstrap_mean(ic, samples=50, seed=1) == moving_block_bootstrap_mean(ic, samples=50, seed=1)


def test_daily_ic_fast_grouping_is_exactly_equal_to_reference_path():
    panel = make_synthetic_market(days=80, symbols=25, seed=91)
    rng = np.random.default_rng(20260830)
    scores = pd.Series(
        rng.integers(-4, 5, len(panel)).astype(float),
        index=panel.index,
        name="candidate",
    )
    labels = pd.Series(
        rng.normal(size=len(panel)), index=panel.index, name="label"
    )
    scores.iloc[rng.choice(len(scores), 137, replace=False)] = np.nan
    labels.iloc[rng.choice(len(labels), 113, replace=False)] = np.nan
    scores.iloc[7] = np.inf
    dates = panel.index.get_level_values("date").unique().sort_values()
    scores.loc[dates[0]] = 1.0  # constant cross-section
    scores.loc[(dates[1], slice(None))] = np.nan  # below min_names

    def reference(method: str) -> tuple[pd.Series, pd.Series]:
        frame = pd.concat(
            [scores.rename("score"), labels.rename("label")], axis=1
        )
        values: dict[pd.Timestamp, float] = {}
        counts: dict[pd.Timestamp, int] = {}
        for date, group in frame.groupby(level="date", sort=True):
            value, count = metrics_module._safe_corr(
                group["score"], group["label"], method, 10
            )
            values[date] = value
            counts[date] = count
        return (
            pd.Series(values, name=f"{method}_ic", dtype=float),
            pd.Series(counts, name="names", dtype=float),
        )

    for method in ("pearson", "spearman"):
        expected_ic, expected_names = reference(method)
        actual_ic, actual_names = daily_ic(
            scores, labels, method=method, min_names=10
        )
        pd.testing.assert_series_equal(actual_ic, expected_ic, check_exact=True)
        pd.testing.assert_series_equal(
            actual_names, expected_names, check_exact=True
        )


def test_neutralization_removes_linear_size_exposure():
    panel = make_synthetic_market(days=90, symbols=25)
    score = panel["log_mcap"] * 2.0 + panel["beta"]
    residual = cross_sectional_neutralize(score, panel[["industry", "log_mcap", "beta"]], min_names=20)
    valid = residual.dropna()
    assert valid.abs().mean() < 1e-6


def test_prepared_neutralizer_is_exactly_equal_to_reference_path():
    panel = make_synthetic_market(days=80, symbols=25, seed=92)
    exposures = panel[["industry", "log_mcap", "beta"]].copy()
    rng = np.random.default_rng(20260831)
    scores = pd.Series(
        rng.normal(size=len(panel)), index=panel.index, name="factor"
    )
    scores.iloc[rng.choice(len(scores), 91, replace=False)] = np.nan
    scores.iloc[13] = np.inf
    exposures.iloc[27, exposures.columns.get_loc("beta")] = np.nan

    def reference(values: pd.Series) -> pd.Series:
        joined = exposures.copy()
        joined["__score__"] = values
        pieces: list[pd.Series] = []
        for _, group in joined.groupby(level="date", sort=True):
            y = group.pop("__score__").astype(float)
            valid = y.replace([np.inf, -np.inf], np.nan).notna()
            columns: list[pd.DataFrame | pd.Series] = []
            for name in group.columns:
                col = group[name]
                if name == "industry":
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
            x = pd.concat(columns, axis=1).replace(
                [np.inf, -np.inf], np.nan
            )
            valid &= x.notna().all(axis=1)
            residual = pd.Series(np.nan, index=group.index, dtype=float)
            if int(valid.sum()) >= max(10, x.shape[1] + 2):
                xv = np.column_stack(
                    [
                        np.ones(int(valid.sum())),
                        x.loc[valid].to_numpy(dtype=float),
                    ]
                )
                yv = y.loc[valid].to_numpy(dtype=float)
                penalty = np.eye(xv.shape[1]) * 1e-8
                penalty[0, 0] = 0.0
                beta = np.linalg.solve(
                    xv.T @ xv + penalty, xv.T @ yv
                )
                residual.loc[valid] = yv - xv @ beta
            pieces.append(residual)
        result = pd.concat(pieces).sort_index()
        result.name = "factor_neutral"
        return result

    prepared = PreparedCrossSectionalNeutralizer(
        exposures, categorical=("industry",), min_names=10
    )
    expected = reference(scores)
    actual = prepared.transform(scores)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)

    # Reuse must not leak any state from the first candidate, including its
    # missing-value pattern.
    second = scores.fillna(0.0).replace(np.inf, 0.0) * -0.37
    second.name = "factor"
    pd.testing.assert_series_equal(
        prepared.transform(second), reference(second), check_exact=True
    )


def test_holm_and_block_max_t_control_a_frozen_factor_family():
    np.testing.assert_allclose(
        holm_adjusted_pvalues([0.01, 0.04, 0.03]),
        [0.03, 0.06, 0.06],
    )
    assert one_sided_normal_pvalue(0.0) == 0.5

    rng = np.random.default_rng(73)
    noise = rng.normal(0.0, 0.02, size=(400, 3))
    values = pd.DataFrame(
        {
            "strong": noise[:, 0] + 0.03,
            "null": noise[:, 1],
            "negative": noise[:, 2] - 0.02,
        }
    )
    first = moving_block_max_t_pvalues(
        values, hac_lags=9, block_size=20, samples=200, seed=41
    )
    second = moving_block_max_t_pvalues(
        values, hac_lags=9, block_size=20, samples=200, seed=41
    )
    pd.testing.assert_series_equal(first, second)
    assert first["strong"] < 0.05
    assert first["negative"] > 0.95
