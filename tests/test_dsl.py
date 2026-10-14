from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cogalpha_plus.dsl as dsl_module
from cogalpha_plus.dsl import (
    DSLValidationError,
    ValidationSpec,
    canonicalize,
    evaluate_expression,
    expression_hash,
    required_lookback,
    validate_expression,
)


def field(name: str) -> dict:
    return {"op": "field", "name": name}


def make_panel(days: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    symbols = ["AAA", "BBB", "CCC"]
    index = pd.MultiIndex.from_product(
        [dates, symbols], names=["date", "symbol"]
    )
    day = np.repeat(np.arange(days, dtype=float), len(symbols))
    stock = np.tile(np.arange(len(symbols), dtype=float), days)
    close = 10.0 + day * (1.0 + stock * 0.1) + stock
    return pd.DataFrame(
        {
            "close": close,
            "open": close - 0.2 + stock * 0.05,
            "volume": 100.0 + day * 4.0 + stock * 10.0,
        },
        index=index,
    )


RAW_SPEC = ValidationSpec(
    allowed_fields=frozenset({"open", "close", "volume"}), max_window=20
)
PRODUCTION_SPEC = ValidationSpec(
    allowed_fields=frozenset({"open", "high", "low", "close", "volume"}),
    max_window=20,
    price_scale_fields=frozenset({"open", "high", "low", "close"}),
    require_price_scale_invariance=True,
    forbid_time_series_over_cross_sectional=True,
)


def test_static_report_and_nested_lookback() -> None:
    expression = {
        "op": "ts_mean",
        "args": [
            {
                "op": "return",
                "args": [field("close")],
                "periods": 2,
            }
        ],
        "window": 5,
    }
    report = validate_expression(expression, RAW_SPEC)
    assert report.lookback == 6
    assert required_lookback(expression, RAW_SPEC) == 6
    assert report.fields == ("close",)
    assert set(report.operators) == {"field", "return", "ts_mean"}


@pytest.mark.parametrize("periods", [-3, -1, 0])
def test_future_or_zero_reference_is_rejected(periods: int) -> None:
    expression = {"op": "lag", "args": [field("close")], "periods": periods}
    with pytest.raises(DSLValidationError, match="future/zero"):
        validate_expression(expression, RAW_SPEC)


def test_field_whitelist_and_label_guard() -> None:
    with pytest.raises(DSLValidationError, match="not whitelisted"):
        validate_expression(field("secret_feature"), RAW_SPEC)
    with pytest.raises(DSLValidationError, match="label-like"):
        validate_expression(field("future_return"))


def test_window_complexity_and_unknown_keys_are_rejected() -> None:
    too_long = {"op": "ts_std", "args": [field("close")], "window": 21}
    with pytest.raises(DSLValidationError, match="max_window"):
        validate_expression(too_long, RAW_SPEC)
    injected = {"op": "field", "name": "close", "python": "do_bad_things()"}
    with pytest.raises(DSLValidationError, match="unexpected keys"):
        validate_expression(injected, RAW_SPEC)


def test_production_price_scale_dimension_gate_is_fail_closed() -> None:
    invariant = (
        {
            "op": "return",
            "args": [field("close")],
            "periods": 5,
        },
        {
            "op": "div",
            "args": [
                {"op": "sub", "args": [field("close"), field("open")]},
                field("open"),
            ],
        },
        {
            "op": "cs_rank",
            "args": [
                {
                    "op": "return",
                    "args": [field("close")],
                    "periods": 2,
                }
            ],
        },
        {
            "op": "ts_corr",
            "args": [field("close"), field("volume")],
            "window": 5,
        },
    )
    for expression in invariant:
        validate_expression(expression, PRODUCTION_SPEC)

    anchor_dependent = (
        field("close"),
        {"op": "cs_rank", "args": [field("close")]},
        {
            "op": "add",
            "args": [field("close"), {"op": "const", "value": 1e-4}],
        },
        {"op": "log1p", "args": [field("close")]},
    )
    for expression in anchor_dependent:
        with pytest.raises(DSLValidationError, match="not provably invariant"):
            validate_expression(expression, PRODUCTION_SPEC)


def test_production_rejects_time_series_operator_above_any_cross_sectional_node() -> None:
    ranked_return = {
        "op": "cs_rank",
        "args": [
            {
                "op": "return",
                "args": [field("close")],
                "periods": 1,
            }
        ],
    }
    direct = {"op": "ts_mean", "args": [ranked_return], "window": 3}
    nested = {
        "op": "lag",
        "args": [
            {
                "op": "add",
                "args": [ranked_return, {"op": "const", "value": 1.0}],
            }
        ],
        "periods": 1,
    }
    for expression in (direct, nested):
        with pytest.raises(DSLValidationError, match="may not wrap"):
            validate_expression(expression, PRODUCTION_SPEC)

    cross_sectional_above_time_series = {
        "op": "cs_rank",
        "args": [
            {
                "op": "ts_mean",
                "args": [
                    {
                        "op": "return",
                        "args": [field("close")],
                        "periods": 1,
                    }
                ],
                "window": 3,
            }
        ],
    }
    validate_expression(cross_sectional_above_time_series, PRODUCTION_SPEC)


def test_canonical_hash_collapses_commutative_equivalents() -> None:
    left = {
        "op": "add",
        "args": [field("close"), {"op": "const", "value": 1.0}],
    }
    right = {
        "op": "add",
        "args": [{"op": "const", "value": 1}, field("close")],
    }
    assert canonicalize(left, RAW_SPEC) == canonicalize(right, RAW_SPEC)
    assert expression_hash(left, RAW_SPEC) == expression_hash(right, RAW_SPEC)


def test_time_series_evaluation_and_safe_domains() -> None:
    panel = make_panel()
    one_day_return = {
        "op": "return",
        "args": [field("close")],
        "periods": 1,
    }
    actual = evaluate_expression(one_day_return, panel, RAW_SPEC)
    for symbol in panel.index.get_level_values("symbol").unique():
        expected = panel.xs(symbol, level="symbol")["close"].pct_change(
            fill_method=None
        )
        np.testing.assert_allclose(
            actual.xs(symbol, level="symbol").to_numpy(),
            expected.to_numpy(),
            equal_nan=True,
        )

    safe_divide = {
        "op": "div",
        "args": [field("close"), {"op": "const", "value": 0}],
    }
    assert evaluate_expression(safe_divide, panel, RAW_SPEC).isna().all()

    invalid_log = {
        "op": "log1p",
        "args": [{"op": "const", "value": -2}],
    }
    assert evaluate_expression(invalid_log, panel, RAW_SPEC).isna().all()

    tiny = panel.copy()
    tiny[["open", "close"]] *= 1e-14
    tiny_return = evaluate_expression(one_day_return, tiny, RAW_SPEC)
    expected_tiny = tiny["close"].groupby(level="symbol", sort=False).pct_change(
        fill_method=None
    )
    np.testing.assert_allclose(
        tiny_return.to_numpy(),
        expected_tiny.to_numpy(),
        equal_nan=True,
    )


def test_cross_sectional_rank_and_zscore() -> None:
    panel = make_panel()
    rank_expression = {"op": "cs_rank", "args": [field("close")]}
    rank = evaluate_expression(rank_expression, panel, RAW_SPEC)
    first = rank.xs(panel.index.levels[0][0], level="date")
    np.testing.assert_allclose(first.to_numpy(), [1 / 3, 2 / 3, 1.0])

    z_expression = {"op": "cs_zscore", "args": [field("close")]}
    zscore = evaluate_expression(z_expression, panel, RAW_SPEC)
    by_date = zscore.groupby(level="date")
    np.testing.assert_allclose(by_date.mean().to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(by_date.std(ddof=0).to_numpy(), 1.0, atol=1e-12)


def test_cross_sectional_mask_excludes_nonmembers_without_removing_time_history() -> None:
    panel = make_panel()
    symbols = panel.index.get_level_values("symbol")
    excluded = panel.index.levels[1][-1]
    mask = pd.Series(symbols != excluded, index=panel.index, dtype=bool)
    expression = {
        "op": "cs_rank",
        "args": [
            {"op": "delta", "args": [field("close")], "periods": 1}
        ],
    }
    ranked = evaluate_expression(
        expression,
        panel,
        RAW_SPEC,
        cross_sectional_mask=mask,
    )
    second_date = panel.index.levels[0][1]
    cross_section = ranked.xs(second_date, level="date")
    np.testing.assert_allclose(cross_section.iloc[:2].to_numpy(), [0.5, 1.0])
    assert np.isnan(cross_section.loc[excluded])

    # The excluded symbol remains in the tape, so its lag path is still
    # continuous even though it is invisible to every cross-sectional node.
    returns = evaluate_expression(
        {"op": "return", "args": [field("close")], "periods": 1},
        panel,
        RAW_SPEC,
        cross_sectional_mask=mask,
    )
    assert np.isfinite(returns.xs(excluded, level="symbol").iloc[1:]).all()


@pytest.mark.parametrize("invalid", ["False", "0", 2, -1])
def test_cross_sectional_mask_rejects_truthy_non_boolean_values(invalid: object) -> None:
    panel = make_panel()
    raw = [True] * len(panel)
    raw[0] = invalid
    mask = pd.Series(raw, index=panel.index, dtype=object)
    expression = {"op": "cs_rank", "args": [field("close")]}
    with pytest.raises(ValueError, match="booleans or numeric 0/1"):
        evaluate_expression(
            expression,
            panel,
            RAW_SPEC,
            cross_sectional_mask=mask,
        )


def test_cross_sectional_mask_accepts_numeric_zero_one_and_rejects_missing() -> None:
    panel = make_panel()
    numeric = pd.Series(
        np.tile([1, 1, 0], len(panel) // 3),
        index=panel.index,
        dtype=int,
    )
    expression = {"op": "cs_rank", "args": [field("close")]}
    ranked = evaluate_expression(
        expression,
        panel,
        RAW_SPEC,
        cross_sectional_mask=numeric,
    )
    assert ranked.loc[numeric.eq(0)].isna().all()
    missing = numeric.astype(float)
    missing.iloc[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        evaluate_expression(
            expression,
            panel,
            RAW_SPEC,
            cross_sectional_mask=missing,
        )


def test_rolling_correlation() -> None:
    panel = make_panel()
    expression = {
        "op": "ts_corr",
        "args": [field("close"), field("volume")],
        "window": 4,
    }
    result = evaluate_expression(expression, panel, RAW_SPEC)
    assert result.groupby(level="symbol").head(3).isna().all()
    np.testing.assert_allclose(result.dropna().to_numpy(), 1.0, atol=1e-12)


def test_audited_trailing_extension_operators_are_causal_and_scale_invariant() -> None:
    panel = make_panel(days=24)
    returns = {"op": "return", "args": [field("close")], "periods": 1}
    expressions = [
        {"op": "ts_sign_entropy", "args": [returns], "window": 6},
        {"op": "ts_cusum", "args": [returns], "window": 6},
        {"op": "ts_spectral_ratio", "args": [returns], "window": 8},
        {"op": "ts_state_transition", "args": [returns], "window": 6},
    ]
    scaled = panel.copy()
    multipliers = scaled.index.get_level_values("symbol").map(
        {"AAA": 0.01, "BBB": 7.0, "CCC": 200.0}
    ).to_numpy(dtype=float)
    scaled["close"] *= multipliers
    scaled["open"] *= multipliers
    cutoff = panel.index.get_level_values("date").unique()[16]
    truncated = panel.loc[panel.index.get_level_values("date") <= cutoff]
    for expression in expressions:
        report = validate_expression(expression, PRODUCTION_SPEC)
        assert report.lookback >= 6
        full = evaluate_expression(expression, panel, PRODUCTION_SPEC)
        rescaled = evaluate_expression(expression, scaled, PRODUCTION_SPEC)
        pd.testing.assert_series_equal(
            full, rescaled, check_names=False, check_exact=False, rtol=1e-10, atol=1e-10
        )
        partial = evaluate_expression(expression, truncated, PRODUCTION_SPEC)
        pd.testing.assert_series_equal(
            full.loc[partial.index], partial, check_names=False, check_exact=True
        )


def test_extension_window_lower_bounds_are_rejected() -> None:
    returns = {"op": "return", "args": [field("close")], "periods": 1}
    for op, window in (
        ("ts_sign_entropy", 1),
        ("ts_cusum", 1),
        ("ts_state_transition", 1),
        ("ts_spectral_ratio", 3),
    ):
        with pytest.raises(DSLValidationError, match="requires a window"):
            validate_expression(
                {"op": op, "args": [returns], "window": window}, RAW_SPEC
            )


def test_cross_sectional_residualization_removes_linear_exposure() -> None:
    panel = make_panel(days=8)
    raw = {
        "op": "return",
        "args": [field("close")],
        "periods": 1,
    }
    exposure = {
        "op": "return",
        "args": [field("open")],
        "periods": 1,
    }
    expression = {"op": "cs_residual", "args": [raw, exposure]}
    validate_expression(expression, PRODUCTION_SPEC)
    residual = evaluate_expression(expression, panel, PRODUCTION_SPEC)
    for date, values in residual.groupby(level="date"):
        aligned = pd.DataFrame(
            {
                "residual": values.droplevel("date"),
                "exposure": evaluate_expression(
                    exposure, panel, PRODUCTION_SPEC
                ).xs(date, level="date"),
            }
        ).dropna()
        if len(aligned) >= 3:
            assert abs(float(aligned["residual"].mean())) < 1e-10
            assert abs(float(aligned.cov(ddof=0).iloc[0, 1])) < 1e-10


def test_truncation_invariance_proves_no_lookahead() -> None:
    panel = make_panel(days=16)
    expression = {
        "op": "cs_zscore",
        "args": [
            {
                "op": "sub",
                "args": [
                    {
                        "op": "ts_mean",
                        "args": [field("volume")],
                        "window": 4,
                    },
                    {"op": "lag", "args": [field("volume")], "periods": 2},
                ],
            }
        ],
    }
    full = evaluate_expression(expression, panel, RAW_SPEC)
    cutoff = panel.index.get_level_values("date").unique()[10]
    truncated_panel = panel.loc[
        panel.index.get_level_values("date") <= cutoff
    ].copy()
    truncated = evaluate_expression(expression, truncated_panel, RAW_SPEC)
    pd.testing.assert_series_equal(
        full.loc[truncated.index], truncated, check_names=False, check_exact=True
    )


def test_input_order_is_restored_after_internal_time_sort() -> None:
    panel = make_panel()
    shuffled = panel.sample(frac=1.0, random_state=7)
    expression = {"op": "lag", "args": [field("close")], "periods": 1}
    expected = evaluate_expression(expression, panel, RAW_SPEC).reindex(shuffled.index)
    actual = evaluate_expression(expression, shuffled, RAW_SPEC)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_positional_time_transforms_are_exactly_equal_to_reference_path() -> None:
    panel = make_panel(days=20).sample(frac=1.0, random_state=83)
    left = panel["close"].copy()
    right = panel["volume"].copy()
    left.iloc[[2, 17]] = np.nan
    right.iloc[[4, 21]] = np.nan

    def unary_reference(series: pd.Series, transform: object) -> pd.Series:
        result = pd.Series(np.nan, index=series.index, dtype=float)
        for _, group in series.groupby(level="symbol", sort=False):
            ordered = group.sort_index(level="date", kind="mergesort")
            values = pd.Series(ordered.to_numpy(dtype=float), dtype=float)
            transformed = transform(values)  # type: ignore[operator]
            result.loc[ordered.index] = np.asarray(transformed, dtype=float)
        return result

    transforms = (
        lambda values: values.shift(3),
        lambda values: values - values.shift(2),
        lambda values: values.rolling(5, min_periods=5).mean(),
        lambda values: values.rolling(5, min_periods=5).std(ddof=0),
    )
    for transform in transforms:
        expected = unary_reference(left, transform)
        actual = dsl_module._time_transform(
            left, "date", "symbol", transform
        )
        pd.testing.assert_series_equal(actual, expected, check_exact=True)

    def pair_reference() -> pd.Series:
        result = pd.Series(np.nan, index=left.index, dtype=float)
        paired = pd.DataFrame({"left": left, "right": right})
        for _, group in paired.groupby(level="symbol", sort=False):
            ordered = group.sort_index(level="date", kind="mergesort")
            left_values = pd.Series(
                ordered["left"].to_numpy(dtype=float), dtype=float
            )
            right_values = pd.Series(
                ordered["right"].to_numpy(dtype=float), dtype=float
            )
            transformed = left_values.rolling(
                5, min_periods=5
            ).corr(right_values)
            result.loc[ordered.index] = np.asarray(transformed, dtype=float)
        return result

    actual_pair = dsl_module._time_pair_transform(
        left,
        right,
        "date",
        "symbol",
        lambda a, b: a.rolling(5, min_periods=5).corr(b),
    )
    pd.testing.assert_series_equal(
        actual_pair, pair_reference(), check_exact=True
    )
