"""A small, deterministic and look-ahead-safe factor expression language.

Expressions are JSON-compatible mappings.  Leaves have one of these forms::

    {"op": "field", "name": "close"}
    {"op": "const", "value": 1.0}

Every other node has an ``args`` list.  Time-series operators additionally use
``window`` (rolling operators) or ``periods`` (lag/delta/return).  The module
deliberately does not accept arbitrary Python source or callable objects: an
LLM can propose expressions, but it cannot execute code through this layer.

The evaluator expects a two-level ``pandas.MultiIndex`` panel, conventionally
named ``(date, symbol)``.  All time-series operators sort observations inside
each symbol, use current/past values only, and finally restore the input order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields
from fractions import Fraction
import hashlib
import json
import math
import numbers
import re
from typing import Any

import numpy as np
import pandas as pd


JSONExpression = Mapping[str, Any]
_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DSLValidationError(ValueError):
    """Raised when an expression is outside the safe DSL."""

    def __init__(self, issues: Sequence[str]):
        self.issues = tuple(issues)
        super().__init__("invalid factor expression: " + "; ".join(self.issues))


@dataclass(frozen=True)
class ValidationSpec:
    """Static limits applied before an expression can be evaluated.

    ``allowed_fields=None`` permits syntactically safe, non-label-like field
    names.  Production callers should pass an explicit raw-data whitelist,
    identify adjusted-price fields, require price-scale invariance, and forbid
    time-series operators above cross-sectional nodes.
    """

    allowed_fields: frozenset[str] | None = None
    max_window: int = 252
    max_nodes: int = 128
    max_depth: int = 24
    price_scale_fields: frozenset[str] = frozenset(
        {"open", "high", "low", "close", "vwap"}
    )
    require_price_scale_invariance: bool = False
    forbid_time_series_over_cross_sectional: bool = False

    def __post_init__(self) -> None:
        if self.allowed_fields is not None:
            if isinstance(self.allowed_fields, (str, bytes)):
                raise TypeError("allowed_fields must be a collection of field names")
            if not isinstance(self.allowed_fields, frozenset):
                object.__setattr__(
                    self, "allowed_fields", frozenset(self.allowed_fields)
                )
        if isinstance(self.price_scale_fields, (str, bytes)):
            raise TypeError("price_scale_fields must be a collection of field names")
        if not isinstance(self.price_scale_fields, frozenset):
            object.__setattr__(
                self, "price_scale_fields", frozenset(self.price_scale_fields)
            )
        if self.max_window < 1:
            raise ValueError("max_window must be positive")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")
        for name in self.price_scale_fields:
            if not isinstance(name, str) or not _FIELD_RE.fullmatch(name):
                raise ValueError("price_scale_fields must contain valid field names")
        for name in (
            "require_price_scale_invariance",
            "forbid_time_series_over_cross_sectional",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")


@dataclass(frozen=True)
class ValidationReport:
    """Useful static properties of a validated expression."""

    node_count: int
    max_depth: int
    lookback: int
    fields: tuple[str, ...]
    operators: tuple[str, ...]


_BINARY = frozenset({"add", "sub", "mul", "div"})
_UNARY = frozenset({"abs", "log1p", "sqrt", "neg"})
_PERIODIC = frozenset({"lag", "delta", "return"})
_ROLLING_UNARY = frozenset(
    {"ts_mean", "ts_std", "ts_min", "ts_max", "ts_sum"}
)
_ROLLING_BINARY = frozenset({"ts_corr"})
_CROSS_SECTIONAL = frozenset({"cs_rank", "cs_zscore"})
_TIME_SERIES = _PERIODIC | _ROLLING_UNARY | _ROLLING_BINARY
_ALL_OPERATORS = frozenset(
    {"field", "const"}
    | _BINARY
    | _UNARY
    | _PERIODIC
    | _ROLLING_UNARY
    | _ROLLING_BINARY
    | _CROSS_SECTIONAL
)
_COMMUTATIVE = frozenset({"add", "mul"})
_LABEL_LIKE_RE = re.compile(
    r"(?:^|_)(?:future|forward|target|label|lead|next)(?:_|$)", re.IGNORECASE
)


def _coerce_spec(spec: ValidationSpec | Mapping[str, Any] | None) -> ValidationSpec:
    if spec is None:
        return ValidationSpec()
    if isinstance(spec, ValidationSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a ValidationSpec, mapping, or None")
    valid_names = {field.name for field in dataclass_fields(ValidationSpec)}
    unknown = sorted(set(spec) - valid_names)
    if unknown:
        raise TypeError(f"unknown validation spec keys: {unknown}")
    values = dict(spec)
    if values.get("allowed_fields") is not None:
        if isinstance(values["allowed_fields"], (str, bytes)):
            raise TypeError("allowed_fields must be a collection of field names")
        values["allowed_fields"] = frozenset(values["allowed_fields"])
    return ValidationSpec(**values)


@dataclass
class _Inspection:
    node_count: int = 0
    deepest: int = 0
    names: set[str] | None = None
    operators: set[str] | None = None
    issues: list[str] | None = None
    active_ids: set[int] | None = None

    def __post_init__(self) -> None:
        self.names = set()
        self.operators = set()
        self.issues = []
        self.active_ids = set()


@dataclass(frozen=True)
class _SemanticSignature:
    """Static production properties of one already validated DSL node."""

    price_scale_degree: Fraction | None
    contains_cross_sectional: bool
    scale_failure: str | None = None


def _format_degree(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


def _production_semantics(
    node: Mapping[str, Any],
    spec: ValidationSpec,
    state: _Inspection,
    *,
    path: str = "$",
) -> _SemanticSignature:
    """Prove price-anchor homogeneity and production operator ordering.

    A vendor adjustment-anchor change multiplies every adjusted price field for
    one security by the same positive constant.  ``price_scale_degree`` tracks
    the exponent of that constant.  ``None`` is deliberately conservative: it
    means that homogeneity could not be proven from the DSL structure.
    """

    op = str(node["op"])
    if op == "field":
        degree = Fraction(1 if node["name"] in spec.price_scale_fields else 0)
        return _SemanticSignature(degree, False)
    if op == "const":
        return _SemanticSignature(Fraction(0), False)

    children = [
        _production_semantics(
            child,
            spec,
            state,
            path=f"{path}.args[{position}]",
        )
        for position, child in enumerate(node["args"])
    ]
    contains_cross_sectional = op in _CROSS_SECTIONAL or any(
        child.contains_cross_sectional for child in children
    )
    if (
        spec.forbid_time_series_over_cross_sectional
        and op in _TIME_SERIES
        and any(child.contains_cross_sectional for child in children)
    ):
        state.issues.append(
            f"{path}: production time-series operator {op!r} may not wrap "
            "a cross-sectional descendant"
        )

    degrees = [child.price_scale_degree for child in children]
    inherited_failure = next(
        (child.scale_failure for child in children if child.scale_failure), None
    )

    def unknown(reason: str) -> _SemanticSignature:
        return _SemanticSignature(None, contains_cross_sectional, reason)

    if op in {"abs", "neg", "lag", "delta"} | _ROLLING_UNARY:
        if degrees[0] is None:
            return unknown(inherited_failure or f"{path}: child is not homogeneous")
        return _SemanticSignature(degrees[0], contains_cross_sectional)
    if op == "sqrt":
        if degrees[0] is None:
            return unknown(inherited_failure or f"{path}: child is not homogeneous")
        return _SemanticSignature(degrees[0] / 2, contains_cross_sectional)
    if op == "log1p":
        if degrees[0] != 0:
            return unknown(
                inherited_failure
                or f"{path}: log1p requires a price-scale invariant child"
            )
        return _SemanticSignature(Fraction(0), contains_cross_sectional)
    if op in {"add", "sub"}:
        if degrees[0] is None or degrees[1] is None:
            return unknown(inherited_failure or f"{path}: child is not homogeneous")
        if degrees[0] != degrees[1]:
            return unknown(
                f"{path}: {op} combines price-scale degrees "
                f"{_format_degree(degrees[0])} and {_format_degree(degrees[1])}"
            )
        return _SemanticSignature(degrees[0], contains_cross_sectional)
    if op == "mul":
        if degrees[0] is None or degrees[1] is None:
            return unknown(inherited_failure or f"{path}: child is not homogeneous")
        return _SemanticSignature(degrees[0] + degrees[1], contains_cross_sectional)
    if op == "div":
        if degrees[0] is None or degrees[1] is None:
            return unknown(inherited_failure or f"{path}: child is not homogeneous")
        return _SemanticSignature(degrees[0] - degrees[1], contains_cross_sectional)
    if op == "return":
        if degrees[0] is None:
            return unknown(inherited_failure or f"{path}: child is not homogeneous")
        return _SemanticSignature(Fraction(0), contains_cross_sectional)
    if op == "ts_corr":
        if degrees[0] is None or degrees[1] is None:
            return unknown(inherited_failure or f"{path}: child is not homogeneous")
        return _SemanticSignature(Fraction(0), contains_cross_sectional)
    if op in _CROSS_SECTIONAL:
        if degrees[0] != 0:
            return unknown(
                inherited_failure
                or f"{path}: {op} requires a price-scale invariant child"
            )
        return _SemanticSignature(Fraction(0), contains_cross_sectional)
    raise RuntimeError(f"unhandled validated operator {op!r}")


def _expect_exact_keys(
    node: Mapping[str, Any], required: set[str], path: str, state: _Inspection
) -> bool:
    actual = set(node)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing:
        state.issues.append(f"{path}: missing keys {missing}")
    if extra:
        state.issues.append(f"{path}: unexpected keys {extra}")
    return not missing and not extra


def _inspect(
    node: Any,
    spec: ValidationSpec,
    state: _Inspection,
    *,
    path: str = "$",
    depth: int = 1,
) -> int:
    """Inspect a node and return its required number of prior observations."""

    state.node_count += 1
    state.deepest = max(state.deepest, depth)
    if state.node_count > spec.max_nodes:
        state.issues.append(
            f"{path}: expression exceeds max_nodes={spec.max_nodes}"
        )
        return 0
    if depth > spec.max_depth:
        state.issues.append(
            f"{path}: expression exceeds max_depth={spec.max_depth}"
        )
        return 0
    if not isinstance(node, Mapping):
        state.issues.append(f"{path}: node must be a JSON object")
        return 0
    node_id = id(node)
    if node_id in state.active_ids:
        state.issues.append(f"{path}: cyclic expression is not JSON-compatible")
        return 0
    state.active_ids.add(node_id)

    try:
        op = node.get("op")
        if not isinstance(op, str):
            state.issues.append(f"{path}.op: must be a string")
            return 0
        if op not in _ALL_OPERATORS:
            state.issues.append(f"{path}.op: unsupported operator {op!r}")
            return 0
        state.operators.add(op)

        if op == "field":
            if not _expect_exact_keys(node, {"op", "name"}, path, state):
                return 0
            name = node.get("name")
            if not isinstance(name, str) or not _FIELD_RE.fullmatch(name):
                state.issues.append(f"{path}.name: invalid field identifier")
                return 0
            if _LABEL_LIKE_RE.search(name):
                state.issues.append(
                    f"{path}.name: label-like field {name!r} is forbidden"
                )
                return 0
            if spec.allowed_fields is not None and name not in spec.allowed_fields:
                state.issues.append(f"{path}.name: field {name!r} is not whitelisted")
                return 0
            state.names.add(name)
            return 0

        if op == "const":
            if not _expect_exact_keys(node, {"op", "value"}, path, state):
                return 0
            value = node.get("value")
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                state.issues.append(f"{path}.value: constant must be a real number")
            elif not math.isfinite(float(value)):
                state.issues.append(f"{path}.value: constant must be finite")
            return 0

        required_keys = {"op", "args"}
        arity = 2 if op in _BINARY or op in _ROLLING_BINARY else 1
        parameter_name: str | None = None
        if op in _PERIODIC:
            parameter_name = "periods"
            required_keys.add(parameter_name)
        elif op in _ROLLING_UNARY or op in _ROLLING_BINARY:
            parameter_name = "window"
            required_keys.add(parameter_name)
        if not _expect_exact_keys(node, required_keys, path, state):
            return 0

        args = node.get("args")
        if not isinstance(args, list):
            state.issues.append(f"{path}.args: must be a JSON array")
            return 0
        if len(args) != arity:
            state.issues.append(
                f"{path}.args: operator {op!r} expects {arity} argument(s)"
            )
            return 0

        parameter = 0
        if parameter_name is not None:
            raw_parameter = node.get(parameter_name)
            if isinstance(raw_parameter, bool) or not isinstance(raw_parameter, int):
                state.issues.append(
                    f"{path}.{parameter_name}: must be a positive integer"
                )
                return 0
            parameter = raw_parameter
            # Zero-period transforms add no information and can hide generation
            # bugs; negative values are an explicit future-data reference.
            if parameter < 1:
                state.issues.append(
                    f"{path}.{parameter_name}: must be >= 1 (future/zero references forbidden)"
                )
                return 0
            if parameter > spec.max_window:
                state.issues.append(
                    f"{path}.{parameter_name}: exceeds max_window={spec.max_window}"
                )
                return 0

        child_lookbacks = [
            _inspect(
                child,
                spec,
                state,
                path=f"{path}.args[{position}]",
                depth=depth + 1,
            )
            for position, child in enumerate(args)
        ]
        inherited = max(child_lookbacks, default=0)
        if op in _PERIODIC:
            return inherited + parameter
        if op in _ROLLING_UNARY or op in _ROLLING_BINARY:
            return inherited + parameter - 1
        return inherited
    finally:
        state.active_ids.discard(node_id)


def validate_expression(
    expression: JSONExpression,
    spec: ValidationSpec | Mapping[str, Any] | None = None,
) -> ValidationReport:
    """Validate *expression* and return its static properties.

    Invalid expressions raise :class:`DSLValidationError`; they never reach the
    evaluator.  Passing ``allowed_fields`` in ``spec`` is strongly recommended
    for real research runs.
    """

    resolved = _coerce_spec(spec)
    state = _Inspection()
    lookback = _inspect(expression, resolved, state)
    semantic: _SemanticSignature | None = None
    if not state.issues:
        semantic = _production_semantics(expression, resolved, state)
        if (
            resolved.require_price_scale_invariance
            and semantic.price_scale_degree != 0
        ):
            reason = semantic.scale_failure
            if reason is None and semantic.price_scale_degree is not None:
                reason = (
                    "result has price-scale degree "
                    + _format_degree(semantic.price_scale_degree)
                )
            state.issues.append(
                "$: production expression is not provably invariant to "
                f"per-security adjusted-price scaling ({reason or 'unknown reason'})"
            )
    if state.issues:
        # A stable order makes rejection logs reproducible across processes.
        raise DSLValidationError(tuple(dict.fromkeys(state.issues)))
    return ValidationReport(
        node_count=state.node_count,
        max_depth=state.deepest,
        lookback=lookback,
        fields=tuple(sorted(state.names)),
        operators=tuple(sorted(state.operators)),
    )


def _normalized_number(value: numbers.Real) -> int | float:
    as_float = float(value)
    if as_float == 0.0:
        return 0
    if as_float.is_integer() and abs(as_float) <= 2**53:
        return int(as_float)
    return as_float


def _canonicalize_valid(node: Mapping[str, Any]) -> dict[str, Any]:
    op = node["op"]
    if op == "field":
        return {"op": "field", "name": node["name"]}
    if op == "const":
        return {"op": "const", "value": _normalized_number(node["value"])}
    args = [_canonicalize_valid(child) for child in node["args"]]
    if op in _COMMUTATIVE:
        args.sort(key=_canonical_json_valid)
    result: dict[str, Any] = {"op": op, "args": args}
    if op in _PERIODIC:
        result["periods"] = int(node["periods"])
    elif op in _ROLLING_UNARY or op in _ROLLING_BINARY:
        result["window"] = int(node["window"])
    return result


def _canonical_json_valid(node: Mapping[str, Any]) -> str:
    return json.dumps(
        node,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonicalize(
    expression: JSONExpression,
    spec: ValidationSpec | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a normalized, detached JSON AST suitable for persistence."""

    validate_expression(expression, spec)
    return _canonicalize_valid(expression)


def canonical_json(
    expression: JSONExpression,
    spec: ValidationSpec | Mapping[str, Any] | None = None,
) -> str:
    """Serialize an expression in its stable canonical form."""

    return _canonical_json_valid(canonicalize(expression, spec))


def expression_hash(
    expression: JSONExpression,
    spec: ValidationSpec | Mapping[str, Any] | None = None,
) -> str:
    """Return a full SHA-256 identity for the expression's semantics."""

    payload = canonical_json(expression, spec).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def required_lookback(
    expression: JSONExpression,
    spec: ValidationSpec | Mapping[str, Any] | None = None,
) -> int:
    """Return how many prior observations are needed for the first value."""

    return validate_expression(expression, spec).lookback


def _panel_levels(panel: pd.DataFrame) -> tuple[int | str, int | str]:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel index must be a two-level MultiIndex(date, symbol)")
    if not panel.index.is_unique:
        raise ValueError("panel index must contain unique (date, symbol) rows")
    names = list(panel.index.names)
    if "date" in names and "symbol" in names:
        return "date", "symbol"
    # A two-level unnamed panel still has an unambiguous conventional order.
    return 0, 1


def _finite(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").astype(float, copy=False)
    values = result.to_numpy(dtype=float, copy=False)
    if np.isinf(values).any():
        return result.replace([np.inf, -np.inf], np.nan)
    return result


def strict_boolean_mask(values: pd.Series, *, label: str) -> pd.Series:
    """Return a real boolean mask without Python truthiness coercion.

    Strings such as ``"False"`` and arbitrary nonzero integers are truthy in
    Python, which is unsafe for a daily eligibility contract.  Only actual
    booleans and finite numeric 0/1 values are accepted.
    """

    if not isinstance(values, pd.Series):
        raise TypeError(f"{label} must be a pandas Series")
    if values.isna().any():
        raise ValueError(f"{label} must not contain missing values")
    if pd.api.types.is_bool_dtype(values.dtype):
        # The common production path is already a real bool Series.  Avoid a
        # Python callback per tape row while preserving a detached, aligned
        # boolean result.
        return values.astype(bool, copy=True)
    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            invalid = values.loc[~numeric.isin([0, 1])]
            samples = ", ".join(repr(value) for value in invalid.iloc[:3])
            raise ValueError(
                f"{label} must contain only booleans or numeric 0/1 values; "
                f"invalid sample(s): {samples}"
            )
        return numeric.astype(bool).rename(values.name)

    def accepted(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return True
        if isinstance(value, numbers.Real):
            numeric = float(value)
            return math.isfinite(numeric) and numeric in {0.0, 1.0}
        return False

    valid = values.map(accepted)
    if not valid.all():
        invalid = values.loc[~valid]
        samples = ", ".join(repr(value) for value in invalid.iloc[:3])
        raise ValueError(
            f"{label} must contain only booleans or numeric 0/1 values; "
            f"invalid sample(s): {samples}"
        )
    return pd.Series(
        [bool(value) for value in values],
        index=values.index,
        dtype=bool,
        name=values.name,
    )


def _time_transform(
    series: pd.Series,
    date_level: int | str,
    symbol_level: int | str,
    transform: Any,
    *,
    ordered_groups: Sequence[np.ndarray] | None = None,
) -> pd.Series:
    # Positional assignment avoids a costly MultiIndex lookup for every
    # security/operator pair.  Stable sorting and the RangeIndex passed to the
    # transform are exactly the same as in the former Series/.loc path.
    source = series.to_numpy(dtype=float)
    output = np.full(len(series), np.nan, dtype=float)
    groups = (
        _ordered_symbol_positions(series.index, date_level, symbol_level)
        if ordered_groups is None
        else ordered_groups
    )
    for ordered_positions in groups:
        values = pd.Series(source[ordered_positions], dtype=float)
        transformed = transform(values)
        output[ordered_positions] = np.asarray(transformed, dtype=float)
    return pd.Series(output, index=series.index, dtype=float)


def _time_pair_transform(
    left: pd.Series,
    right: pd.Series,
    date_level: int | str,
    symbol_level: int | str,
    transform: Any,
    *,
    ordered_groups: Sequence[np.ndarray] | None = None,
) -> pd.Series:
    left_source = left.to_numpy(dtype=float)
    right_source = right.to_numpy(dtype=float)
    output = np.full(len(left), np.nan, dtype=float)
    groups = (
        _ordered_symbol_positions(left.index, date_level, symbol_level)
        if ordered_groups is None
        else ordered_groups
    )
    for ordered_positions in groups:
        left_values = pd.Series(
            left_source[ordered_positions], dtype=float
        )
        right_values = pd.Series(
            right_source[ordered_positions], dtype=float
        )
        transformed = transform(left_values, right_values)
        output[ordered_positions] = np.asarray(transformed, dtype=float)
    return pd.Series(output, index=left.index, dtype=float)


def _ordered_symbol_positions(
    index: pd.MultiIndex,
    date_level: int | str,
    symbol_level: int | str,
) -> tuple[np.ndarray, ...]:
    dates = index.get_level_values(date_level).to_numpy()
    marker = pd.Series(np.arange(len(index), dtype=np.int64), index=index)
    groups = marker.groupby(level=symbol_level, sort=False).indices
    ordered: list[np.ndarray] = []
    for positions in groups.values():
        positions = np.asarray(positions, dtype=np.int64)
        resolved = positions[np.argsort(dates[positions], kind="stable")]
        resolved.setflags(write=False)
        ordered.append(resolved)
    return tuple(ordered)


def evaluate_expression(
    expression: JSONExpression,
    panel: pd.DataFrame,
    spec: ValidationSpec | Mapping[str, Any] | None = None,
    *,
    epsilon: float = 1e-12,
    cross_sectional_mask: pd.Series | None = None,
) -> pd.Series:
    """Evaluate a validated factor on a ``(date, symbol)`` panel.

    The returned series has exactly the input index and is named using a short
    expression hash.  Undefined domains, zero denominators, and non-finite
    values become ``NaN`` instead of silently becoming extreme scores.
    """

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    date_level, symbol_level = _panel_levels(panel)
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive number")

    if spec is None:
        resolved_spec = ValidationSpec(allowed_fields=frozenset(map(str, panel.columns)))
    else:
        resolved_spec = _coerce_spec(spec)
    normalized = canonicalize(expression, resolved_spec)
    if cross_sectional_mask is None:
        eligible = pd.Series(True, index=panel.index, dtype=bool)
    else:
        if not isinstance(cross_sectional_mask, pd.Series):
            raise TypeError("cross_sectional_mask must be a pandas Series")
        if not cross_sectional_mask.index.equals(panel.index):
            raise ValueError("cross_sectional_mask must exactly align with panel index")
        eligible = strict_boolean_mask(
            cross_sectional_mask,
            label="cross_sectional_mask",
        )
    memo: dict[str, pd.Series] = {}
    ordered_time_groups: tuple[np.ndarray, ...] | None = None

    def time_groups() -> tuple[np.ndarray, ...]:
        nonlocal ordered_time_groups
        if ordered_time_groups is None:
            ordered_time_groups = _ordered_symbol_positions(
                panel.index, date_level, symbol_level
            )
        return ordered_time_groups

    def evaluate(node: Mapping[str, Any]) -> pd.Series:
        key = _canonical_json_valid(node)
        cached = memo.get(key)
        if cached is not None:
            return cached
        op = node["op"]
        if op == "field":
            name = node["name"]
            if name not in panel.columns:
                raise KeyError(f"panel has no field {name!r}")
            result = _finite(panel[name])
        elif op == "const":
            result = pd.Series(float(node["value"]), index=panel.index, dtype=float)
        else:
            args = [evaluate(child) for child in node["args"]]
            if op == "add":
                result = args[0] + args[1]
            elif op == "sub":
                result = args[0] - args[1]
            elif op == "mul":
                result = args[0] * args[1]
            elif op == "div":
                # Exact-zero masking is scale covariant and introduces no
                # hidden time-series lookback.  Non-finite quotients are still
                # removed by _finite below.
                denominator = args[1].where(args[1].ne(0.0))
                result = args[0] / denominator
            elif op == "abs":
                result = args[0].abs()
            elif op == "neg":
                result = -args[0]
            elif op == "log1p":
                domain = args[0].where(args[0] > -1.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    result = np.log1p(domain)
            elif op == "sqrt":
                domain = args[0].where(args[0] >= 0.0)
                with np.errstate(invalid="ignore"):
                    result = np.sqrt(domain)
            elif op in _PERIODIC:
                periods = node["periods"]
                if op == "lag":
                    result = _time_transform(
                        args[0],
                        date_level,
                        symbol_level,
                        lambda values: values.shift(periods),
                        ordered_groups=time_groups(),
                    )
                elif op == "delta":
                    result = _time_transform(
                        args[0],
                        date_level,
                        symbol_level,
                        lambda values: values - values.shift(periods),
                        ordered_groups=time_groups(),
                    )
                else:
                    result = _time_transform(
                        args[0],
                        date_level,
                        symbol_level,
                        lambda values: values.div(
                            values.shift(periods).where(
                                values.shift(periods).ne(0.0)
                            )
                        )
                        - 1.0,
                        ordered_groups=time_groups(),
                    )
            elif op in _ROLLING_UNARY:
                window = node["window"]
                if op == "ts_mean":
                    reducer = lambda values: values.rolling(
                        window, min_periods=window
                    ).mean()
                elif op == "ts_std":
                    reducer = lambda values: values.rolling(
                        window, min_periods=window
                    ).std(ddof=0)
                elif op == "ts_min":
                    reducer = lambda values: values.rolling(
                        window, min_periods=window
                    ).min()
                elif op == "ts_max":
                    reducer = lambda values: values.rolling(
                        window, min_periods=window
                    ).max()
                else:
                    reducer = lambda values: values.rolling(
                        window, min_periods=window
                    ).sum()
                result = _time_transform(
                    args[0],
                    date_level,
                    symbol_level,
                    reducer,
                    ordered_groups=time_groups(),
                )
            elif op == "ts_corr":
                window = node["window"]
                result = _time_pair_transform(
                    args[0],
                    args[1],
                    date_level,
                    symbol_level,
                    lambda left, right: left.rolling(
                        window, min_periods=window
                    ).corr(right),
                    ordered_groups=time_groups(),
                )
            elif op == "cs_rank":
                cross_section = args[0].where(eligible)
                result = cross_section.groupby(level=date_level, sort=False).rank(
                    method="average", pct=True, na_option="keep"
                )
            elif op == "cs_zscore":
                cross_section = args[0].where(eligible)
                grouped = cross_section.groupby(level=date_level, sort=False)
                mean = grouped.transform("mean")
                std = grouped.transform(lambda values: values.std(ddof=0))
                result = (cross_section - mean) / std.where(std > epsilon)
            else:  # pragma: no cover - validation makes this unreachable.
                raise RuntimeError(f"unhandled validated operator {op!r}")
        result = _finite(result).reindex(panel.index)
        memo[key] = result
        return result

    output = evaluate(normalized).copy()
    output.name = f"factor_{expression_hash(normalized, resolved_spec)[:12]}"
    return output


# Short aliases are useful to adapters while the explicit public names remain
# self-documenting in research code.
validate_ast = validate_expression
canonical_hash = expression_hash
lookback = required_lookback
evaluate = evaluate_expression


__all__ = [
    "DSLValidationError",
    "JSONExpression",
    "ValidationReport",
    "ValidationSpec",
    "canonical_hash",
    "canonical_json",
    "canonicalize",
    "evaluate",
    "evaluate_expression",
    "expression_hash",
    "lookback",
    "required_lookback",
    "strict_boolean_mask",
    "validate_ast",
    "validate_expression",
]
