"""Audited Python-shaped factor programs lowered into the safe factor DSL.

The public CogAlpha paper represents factors as Python code.  Executing arbitrary
model output would expose files, credentials and the network, so Track A accepts
one deliberately small Python surface: a single ``factor(ctx)`` function whose
body is one expression composed from audited ``ctx`` fields and operators.  The
source is parsed, never passed to ``exec`` or ``eval``, and is then calculated by
the deterministic DSL runtime over an in-memory, read-only panel view.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping

import pandas as pd

from .dsl import (
    JSONExpression,
    ValidationReport,
    ValidationSpec,
    canonicalize,
    evaluate_expression,
    validate_expression,
)


class PythonFactorValidationError(ValueError):
    """Raised when generated Python leaves the audited factor surface."""


@dataclass(frozen=True)
class PythonFactorSandboxPolicy:
    schema_version: int = 1
    max_source_bytes: int = 16_384
    max_ast_nodes: int = 256
    cpu_seconds_per_worker: int = 7_200
    memory_bytes_per_worker: int = 8 * 1024**3
    network: str = "denied"
    filesystem: str = "no_generated_code_access"
    input_mutability: str = "read_only_copy"
    execution_mode: str = "ast_lowering_without_exec_or_eval"

    def __post_init__(self) -> None:
        if self.max_source_bytes < 128 or self.max_ast_nodes < 8:
            raise ValueError("Python factor source limits are too small")
        if self.cpu_seconds_per_worker < 1 or self.memory_bytes_per_worker < 1:
            raise ValueError("Python factor resource limits must be positive")
        if self.network != "denied":
            raise ValueError("generated factor network access must be denied")


@dataclass(frozen=True)
class ParsedPythonFactor:
    expression: Mapping[str, Any]
    report: ValidationReport
    source_sha256: str
    sandbox_policy: Mapping[str, object]


_METHODS: dict[str, tuple[str, int, str | None]] = {
    "add": ("add", 2, None),
    "sub": ("sub", 2, None),
    "mul": ("mul", 2, None),
    "div": ("div", 2, None),
    "abs": ("abs", 1, None),
    "log1p": ("log1p", 1, None),
    "sqrt": ("sqrt", 1, None),
    "neg": ("neg", 1, None),
    "lag": ("lag", 2, "periods"),
    "delta": ("delta", 2, "periods"),
    "ret": ("return", 2, "periods"),
    "mean": ("ts_mean", 2, "window"),
    "std": ("ts_std", 2, "window"),
    "minimum": ("ts_min", 2, "window"),
    "maximum": ("ts_max", 2, "window"),
    "sum": ("ts_sum", 2, "window"),
    "corr": ("ts_corr", 3, "window"),
    "sign_entropy": ("ts_sign_entropy", 2, "window"),
    "cusum": ("ts_cusum", 2, "window"),
    "spectral_ratio": ("ts_spectral_ratio", 2, "window"),
    "state_transition": ("ts_state_transition", 2, "window"),
    "rank": ("cs_rank", 1, None),
    "zscore": ("cs_zscore", 1, None),
    "residual": ("cs_residual", 2, None),
}

_RENDER_METHOD = {
    "return": "ret",
    "ts_mean": "mean",
    "ts_std": "std",
    "ts_min": "minimum",
    "ts_max": "maximum",
    "ts_sum": "sum",
    "ts_corr": "corr",
    "ts_sign_entropy": "sign_entropy",
    "ts_cusum": "cusum",
    "ts_spectral_ratio": "spectral_ratio",
    "ts_state_transition": "state_transition",
    "cs_rank": "rank",
    "cs_zscore": "zscore",
    "cs_residual": "residual",
}


def _positive_integer(node: ast.AST, *, label: str) -> int:
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
        raise PythonFactorValidationError(f"{label} must be a positive integer literal")
    value = node.value
    if not isinstance(value, int) or value < 1:
        raise PythonFactorValidationError(f"{label} must be a positive integer literal")
    return value


def _lower(node: ast.AST, allowed_fields: frozenset[str]) -> dict[str, Any]:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PythonFactorValidationError("factor constants must be real numeric literals")
        return {"op": "const", "value": value}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _lower(node.operand, allowed_fields)
        if value.get("op") == "const":
            return {"op": "const", "value": -float(value["value"])}
        return {"op": "neg", "args": [value]}
    if isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name) or node.value.id != "ctx":
            raise PythonFactorValidationError("field access must use ctx.<field>")
        if node.attr not in allowed_fields:
            raise PythonFactorValidationError(f"field {node.attr!r} is not whitelisted")
        return {"op": "field", "name": node.attr}
    if not isinstance(node, ast.Call):
        raise PythonFactorValidationError(
            f"unsupported Python syntax in factor expression: {type(node).__name__}"
        )
    if node.keywords:
        raise PythonFactorValidationError("factor operators accept positional arguments only")
    function = node.func
    if (
        not isinstance(function, ast.Attribute)
        or not isinstance(function.value, ast.Name)
        or function.value.id != "ctx"
    ):
        raise PythonFactorValidationError("operator calls must use ctx.<operator>(...)")
    method = _METHODS.get(function.attr)
    if method is None:
        raise PythonFactorValidationError(f"operator {function.attr!r} is not audited")
    operator, arity, parameter_name = method
    if len(node.args) != arity:
        raise PythonFactorValidationError(
            f"ctx.{function.attr} expects exactly {arity} positional arguments"
        )
    expression_count = arity - (1 if parameter_name is not None else 0)
    expression = {
        "op": operator,
        "args": [_lower(item, allowed_fields) for item in node.args[:expression_count]],
    }
    if parameter_name is not None:
        expression[parameter_name] = _positive_integer(
            node.args[-1], label=f"ctx.{function.attr} window"
        )
    return expression


def parse_python_factor(
    source: str,
    spec: ValidationSpec,
    *,
    policy: PythonFactorSandboxPolicy | None = None,
) -> ParsedPythonFactor:
    """Validate one Python-shaped factor and lower it into canonical DSL."""

    resolved_policy = policy or PythonFactorSandboxPolicy()
    if not isinstance(source, str):
        raise TypeError("factor source must be text")
    encoded = source.encode("utf-8")
    if len(encoded) > resolved_policy.max_source_bytes:
        raise PythonFactorValidationError("factor source exceeds max_source_bytes")
    if "\x00" in source:
        raise PythonFactorValidationError("factor source contains a null byte")
    try:
        module = ast.parse(source, mode="exec")
    except (SyntaxError, MemoryError, RecursionError) as exc:
        raise PythonFactorValidationError(f"invalid factor source: {type(exc).__name__}") from exc
    nodes = tuple(ast.walk(module))
    if len(nodes) > resolved_policy.max_ast_nodes:
        raise PythonFactorValidationError("factor source exceeds max_ast_nodes")
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise PythonFactorValidationError("source must contain exactly one factor function")
    function = module.body[0]
    if (
        function.name != "factor"
        or function.decorator_list
        or function.returns is not None
        or function.type_comment is not None
        or function.args.posonlyargs
        or function.args.vararg is not None
        or function.args.kwonlyargs
        or function.args.kwarg is not None
        or function.args.defaults
        or function.args.kw_defaults
        or len(function.args.args) != 1
        or function.args.args[0].arg != "ctx"
        or function.args.args[0].annotation is not None
    ):
        raise PythonFactorValidationError("function signature must be exactly factor(ctx)")
    if len(function.body) != 1 or not isinstance(function.body[0], ast.Return):
        raise PythonFactorValidationError("factor(ctx) must contain exactly one return statement")
    if function.body[0].value is None:
        raise PythonFactorValidationError("factor(ctx) must return an expression")
    allowed_fields = spec.allowed_fields
    if allowed_fields is None:
        raise ValueError("Python factor parsing requires an explicit field whitelist")
    expression = canonicalize(_lower(function.body[0].value, allowed_fields), spec)
    report = validate_expression(expression, spec)
    return ParsedPythonFactor(
        expression=expression,
        report=report,
        source_sha256=hashlib.sha256(encoded).hexdigest(),
        sandbox_policy=asdict(resolved_policy),
    )


def render_python_factor(expression: JSONExpression, spec: ValidationSpec) -> str:
    """Render canonical DSL as the exact Python surface accepted above."""

    normalized = canonicalize(expression, spec)

    def render(node: Mapping[str, Any]) -> str:
        operator = str(node["op"])
        if operator == "field":
            return f"ctx.{node['name']}"
        if operator == "const":
            return repr(node["value"])
        method = _RENDER_METHOD.get(operator, operator)
        arguments = [render(item) for item in node["args"]]
        if "periods" in node:
            arguments.append(str(node["periods"]))
        if "window" in node:
            arguments.append(str(node["window"]))
        return f"ctx.{method}({', '.join(arguments)})"

    return "def factor(ctx):\n    return " + render(normalized) + "\n"


def evaluate_python_factor(
    source: str,
    panel: pd.DataFrame,
    spec: ValidationSpec,
    *,
    cross_sectional_mask: pd.Series | None = None,
    policy: PythonFactorSandboxPolicy | None = None,
) -> pd.Series:
    """Calculate generated source without granting it Python execution access."""

    parsed = parse_python_factor(source, spec, policy=policy)
    read_only_panel = panel.copy(deep=False)
    return evaluate_expression(
        parsed.expression,
        read_only_panel,
        spec,
        cross_sectional_mask=cross_sectional_mask,
    )


__all__ = [
    "ParsedPythonFactor",
    "PythonFactorSandboxPolicy",
    "PythonFactorValidationError",
    "evaluate_python_factor",
    "parse_python_factor",
    "render_python_factor",
]
