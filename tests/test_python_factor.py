from __future__ import annotations

import pandas as pd
import pytest

from cogalpha_plus.dsl import ValidationSpec, evaluate_expression
from cogalpha_plus.python_factor import (
    PythonFactorSandboxPolicy,
    PythonFactorValidationError,
    evaluate_python_factor,
    parse_python_factor,
    render_python_factor,
)


SPEC = ValidationSpec(
    allowed_fields=frozenset({"open", "close", "volume", "log_mcap"}),
    max_window=60,
    max_nodes=64,
    price_scale_fields=frozenset({"open", "close"}),
    require_price_scale_invariance=True,
    forbid_time_series_over_cross_sectional=True,
)


def panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=12, freq="B")
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB", "CCC"]], names=["date", "symbol"]
    )
    return pd.DataFrame(
        {
            "open": [10.0 + i * 0.1 for i in range(len(index))],
            "close": [10.2 + i * 0.11 for i in range(len(index))],
            "volume": [1000.0 + i * 3.0 for i in range(len(index))],
            "log_mcap": [8.0 + (i % 3) for i in range(len(index))],
        },
        index=index,
    )


def test_python_round_trip_and_evaluation_matches_dsl_exactly() -> None:
    expression = {
        "op": "cs_residual",
        "args": [
            {
                "op": "ts_sign_entropy",
                "args": [
                    {"op": "return", "args": [{"op": "field", "name": "close"}], "periods": 1}
                ],
                "window": 5,
            },
            {"op": "cs_zscore", "args": [{"op": "field", "name": "log_mcap"}]},
        ],
    }
    source = render_python_factor(expression, SPEC)
    parsed = parse_python_factor(source, SPEC)
    assert parsed.expression == expression
    assert parsed.sandbox_policy["network"] == "denied"
    assert parsed.sandbox_policy["execution_mode"] == "ast_lowering_without_exec_or_eval"
    expected = evaluate_expression(expression, panel(), SPEC)
    actual = evaluate_python_factor(source, panel(), SPEC)
    pd.testing.assert_series_equal(actual, expected, check_names=False, check_exact=True)


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef factor(ctx):\n    return ctx.close\n",
        "def factor(ctx):\n    open('/tmp/leak', 'w')\n    return ctx.close\n",
        "def factor(ctx):\n    return __import__('socket').socket()\n",
        "def factor(ctx):\n    return ctx.close.shift(-1)\n",
        "def factor(ctx):\n    while True:\n        pass\n",
        "def factor(data):\n    return data.close\n",
    ],
)
def test_python_factor_rejects_code_execution_surfaces(source: str) -> None:
    with pytest.raises(PythonFactorValidationError):
        parse_python_factor(source, SPEC)


def test_python_factor_source_and_ast_limits_fail_closed() -> None:
    tiny = PythonFactorSandboxPolicy(max_source_bytes=128, max_ast_nodes=8)
    with pytest.raises(PythonFactorValidationError, match="max_source_bytes"):
        parse_python_factor(
            "def factor(ctx):\n    return ctx.rank(ctx.ret(ctx.close, 1))\n" + " " * 200,
            SPEC,
            policy=tiny,
        )
    with pytest.raises(PythonFactorValidationError, match="max_ast_nodes"):
        parse_python_factor(
            "def factor(ctx):\n    return ctx.rank(ctx.ret(ctx.close, 1))\n",
            SPEC,
            policy=tiny,
        )
