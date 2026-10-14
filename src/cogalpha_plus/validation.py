"""Blinded development scoring and one-time sealed evaluation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data_contract import RESEARCH_PRICE_MAP
from .dsl import (
    DSLValidationError,
    ValidationSpec,
    canonical_json,
    evaluate_expression,
    expression_hash,
    strict_boolean_mask,
    validate_expression,
)
from .evolution import (
    BehavioralChampionPolicy,
    Candidate,
    CandidateScore,
    ChampionSelectionResult,
    EvaluatedCandidate,
    deterministic_candidate_id,
    expression_complexity,
)
from .metrics import (
    PreparedCrossSectionalNeutralizer,
    cross_sectional_neutralize,
    daily_ic,
    endpoint_executable_forward_return,
    factor_correlation,
    forward_return,
    holm_adjusted_pvalues,
    moving_block_max_t_pvalues,
    normalize_trading_calendar,
    one_sided_normal_pvalue,
    rank_turnover,
    summarize_ic,
)
from .manifest import hash_json
from .python_factor import (
    PythonFactorSandboxPolicy,
    PythonFactorValidationError,
    parse_python_factor,
    render_python_factor,
)
from .terminal_labels import (
    TerminalLabelBridge,
    apply_terminal_label_bridges,
    validate_terminal_label_bridge_identity,
)


FEATURE_FIELDS = frozenset({"open", "high", "low", "close", "volume", "amount", "vwap"})
PRICE_FEATURE_FIELDS = frozenset({"open", "high", "low", "close", "vwap"})
PRODUCTION_DSL_SPEC = ValidationSpec(
    allowed_fields=FEATURE_FIELDS,
    max_window=252,
    max_nodes=48,
    max_depth=16,
    price_scale_fields=PRICE_FEATURE_FIELDS,
    require_price_scale_invariance=True,
    forbid_time_series_over_cross_sectional=True,
)


RESEARCH_ELIGIBILITY_RULE_V1: Mapping[str, object] = {
    "schema_version": 1,
    "selection_column": "research_eligible",
    "decision_time": "t_close",
    "current_time_only": True,
    "future_endpoint_availability_filter": False,
    "time_series_scope": "continuous_union_feature_tape",
    "cross_sectional_scope": "same_date_research_eligible_only",
    "legacy_universe_member_allowed_in_production": False,
}

EXACT_FORWARD_LABEL_RULE_V1: Mapping[str, object] = {
    "schema_version": 1,
    "calendar_source": "frozen_normalized_trade_calendar",
    "entry_endpoint": "calendar[t+entry_delay]",
    "exit_endpoint": "calendar[t+entry_delay+horizon]",
    "missing_symbol_bar_policy": (
        "missing_label_no_compression_no_fill_no_drop_except_sourced_terminal_exchange"
    ),
    "terminal_exchange_policy": (
        "exact_endpoint_successor_raw_endpoint_price_valuation_only"
    ),
    "standard_label_required_daily_coverage": 1.0,
}


def research_rule_identity_v1() -> dict[str, object]:
    payload = {
        "eligibility": dict(RESEARCH_ELIGIBILITY_RULE_V1),
        "forward_label": dict(EXACT_FORWARD_LABEL_RULE_V1),
    }
    return {**payload, "rule_hash": hash_json(payload)}


def _semantic_panel_hash(obj: pd.Series | pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(
        obj, index=True, categorize=True
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def build_research_universe_identity(
    full_member_panel: pd.DataFrame,
    feature_tape: pd.DataFrame,
    eligible_panel: pd.DataFrame,
    *,
    terminal_sidecar_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind the complete member view and the exact ex-ante selection mask."""

    full_member_panel = full_member_panel.sort_index()
    feature_tape = feature_tape.sort_index()
    eligible_panel = eligible_panel.sort_index()
    for label, frame in (
        ("full member panel", full_member_panel),
        ("feature tape", feature_tape),
        ("eligible panel", eligible_panel),
    ):
        if (
            not isinstance(frame.index, pd.MultiIndex)
            or tuple(frame.index.names) != ("date", "symbol")
            or frame.index.has_duplicates
        ):
            raise ValueError(f"{label} must use a unique date-symbol index")
    if not full_member_panel.index.isin(feature_tape.index).all():
        raise ValueError("feature tape does not cover the complete member panel")
    if "research_eligible" not in full_member_panel.columns:
        raise ValueError("complete member panel lacks research_eligible")
    member_mask = strict_boolean_mask(
        full_member_panel["research_eligible"],
        label="full member panel research_eligible",
    )
    expected_eligible = full_member_panel.loc[member_mask].index
    if not expected_eligible.equals(eligible_panel.index):
        raise ValueError("eligible panel is not the exact research_eligible subset")
    tape_mask = build_cross_sectional_mask(feature_tape, eligible_panel)
    if not tape_mask.reindex(full_member_panel.index).equals(member_mask):
        raise ValueError("member-panel and feature-tape eligibility masks differ")
    daily = (
        pd.Series(1, index=eligible_panel.index, dtype=np.int64)
        .groupby(level="date", sort=True)
        .sum()
        .rename("eligible_count")
    )
    daily_records = [
        {"date": str(pd.Timestamp(date).date()), "eligible_count": int(count)}
        for date, count in daily.items()
    ]
    rules = research_rule_identity_v1()
    payload: dict[str, object] = {
        "schema_version": 1,
        "rule_hash": rules["rule_hash"],
        "rules": rules,
        "full_member_panel_rows": int(len(full_member_panel)),
        "full_member_panel_hash": _semantic_panel_hash(full_member_panel),
        "eligible_panel_rows": int(len(eligible_panel)),
        "eligible_panel_hash": _semantic_panel_hash(eligible_panel),
        "full_member_eligibility_mask_hash": _semantic_panel_hash(
            member_mask.astype(np.int8).rename("research_eligible")
        ),
        "feature_tape_eligibility_mask_hash": _semantic_panel_hash(
            tape_mask.astype(np.int8).rename("research_eligible")
        ),
        "eligible_daily_counts": daily_records,
        "eligible_daily_counts_hash": hash_json(daily_records),
        "eligible_daily_min": int(daily.min()) if len(daily) else 0,
        "eligible_daily_max": int(daily.max()) if len(daily) else 0,
        "terminal_sidecar_identity": dict(terminal_sidecar_identity or {}),
    }
    return {**payload, "identity_hash": hash_json(payload)}


def sealed_reporting_contract_v1() -> dict[str, object]:
    """Return the closed reporting contract fixed before sealed replay.

    The 23-series family is retained as a conservative multiplicity-control
    family.  It is not a pool from which a model or a headline result may be
    selected after observing sealed outcomes.  Only the standard-label,
    neutral LightGBM composite is eligible for the primary claim; every other
    reported endpoint is an explicitly pre-registered secondary diagnostic.
    """

    return {
        "schema_version": 1,
        "unique_primary_result_key": "composite:lightgbm",
        "unique_primary_backend": "lightgbm",
        "primary_endpoint_key": "sealed_neutral",
        "primary_inference_key": "joint_multiple_testing",
        "joint_multiplicity_family": "all_20_factors_plus_3_composites",
        "joint_multiplicity_family_size": 23,
        "joint_family_role": (
            "conservative_fwer_control_not_a_selection_or_winner_pool"
        ),
        "component_factor_role": (
            "pre_registered_secondary_component_factor_diagnostic"
        ),
        "control_composite_result_keys": [
            "composite:equal_weight",
            "composite:ridge",
        ],
        "control_composite_role": (
            "pre_registered_secondary_model_control"
        ),
        "contrast_family_role": (
            "pre_registered_secondary_model_comparison"
        ),
        "immediate_executable_role": (
            "pre_registered_secondary_execution_sensitivity_diagnostic"
        ),
        "secondary_results_eligible_for_primary_claim": False,
        "backend_selection_from_sealed_results_allowed": False,
        "primary_reassignment_allowed": False,
    }


def build_research_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the DSL input view without mutating canonical raw execution prices.

    Factor Cards continue to use the stable ``open``/``close``/``vwap`` DSL
    vocabulary.  Within this short-lived calculation frame those names are
    sourced from the corresponding ``adj_*`` columns.  The validated panel and
    every execution component retain the canonical names as raw prices.
    """

    sources = {
        name: RESEARCH_PRICE_MAP.get(name, name)
        for name in FEATURE_FIELDS
    }
    missing = sorted(set(sources.values()).difference(panel.columns))
    if missing:
        raise ValueError(f"panel missing research feature fields: {missing}")
    return pd.DataFrame(
        {name: panel[source] for name, source in sources.items()},
        index=panel.index,
    )


def build_cross_sectional_mask(
    feature_tape: pd.DataFrame,
    eligible_panel: pd.DataFrame,
    *,
    allow_legacy_eligibility: bool = False,
) -> pd.Series:
    """Separate continuous time-series history from daily index eligibility.

    Lag and rolling operators need every bar for every historical constituent,
    including bars observed while a security is outside the index.  In
    contrast, ``cs_rank`` and ``cs_zscore`` must only see securities eligible
    on that date.  The frozen real-data tape carries that mask explicitly;
    synthetic/unit-test panels can derive it from the eligible panel index.
    """

    if not eligible_panel.index.isin(feature_tape.index).all():
        raise ValueError("continuous feature tape does not cover every eligible row")
    if "research_eligible" in feature_tape.columns:
        source = feature_tape["research_eligible"]
        mask = strict_boolean_mask(
            source,
            label="feature tape research_eligible",
        )
    elif allow_legacy_eligibility:
        mask = pd.Series(
            feature_tape.index.isin(eligible_panel.index),
            index=feature_tape.index,
            dtype=bool,
            name="research_eligible",
        )
    else:
        raise ValueError(
            "feature tape lacks required research_eligible; legacy fallback is test-only"
        )

    if "research_eligible" in eligible_panel.columns:
        declared = strict_boolean_mask(
            eligible_panel["research_eligible"],
            label="eligible panel research_eligible",
        )
        if not declared.all():
            raise ValueError("eligible panel contains research_eligible=false rows")

    eligible_dates = eligible_panel.index.get_level_values("date")
    first_date = eligible_dates.min()
    last_date = eligible_dates.max()
    tape_dates = feature_tape.index.get_level_values("date")
    in_eligible_span = (tape_dates >= first_date) & (tape_dates <= last_date)
    declared_index = feature_tape.index[
        in_eligible_span & mask.to_numpy(dtype=bool)
    ]
    if not declared_index.equals(eligible_panel.index):
        raise ValueError(
            "feature tape research_eligible mask does not exactly match eligible panel"
        )
    mask.name = "research_eligible"
    return mask


def build_immediate_executable_labels(
    feature_frame: pd.DataFrame,
    tape_panel: pd.DataFrame,
    *,
    horizon: int,
    entry_delay: int,
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex | None,
    allow_test_calendar_from_panel: bool = False,
    price_field: str = "vwap",
) -> tuple[pd.Series, dict[str, object]]:
    """Build a fixed-endpoint executability diagnostic beside the forecast label.

    The primary research target remains the adjusted endpoint return.  It does
    not condition the sample on a future fill state.  This second label asks a
    narrower execution question: could a long-only order trade at both of its
    scheduled endpoints?  Missing endpoints stay missing; they are never
    replaced by the close-price placeholder used to value a suspended stock.

    This is deliberately not a delayed-fill backtest.  Orders that cannot fill
    immediately, holdings frozen at a limit, costs and T+1 accounting belong in
    the event-driven portfolio simulator.
    """

    if not feature_frame.index.equals(tape_panel.index):
        raise ValueError("feature frame and execution-status tape must align exactly")
    index = feature_frame.index

    def status(name: str, default: bool) -> tuple[pd.Series, str]:
        if name not in tape_panel.columns:
            return (
                pd.Series(default, index=index, dtype=bool, name=name),
                f"default_{str(default).lower()}",
            )
        return tape_panel[name], f"tape_column:{name}"

    tradable, tradable_source = status("tradable", True)
    limit_up, limit_up_source = status("limit_up", False)
    limit_down, limit_down_source = status("limit_down", False)
    limit_known, limit_known_source = status("limit_proxy_known", True)
    if price_field not in {"open", "vwap"}:
        raise ValueError("executable label price_field must be open or vwap")
    labels = endpoint_executable_forward_return(
        feature_frame[price_field],
        tradable,
        horizon,
        entry_delay=entry_delay,
        trading_calendar=trading_calendar,
        allow_test_calendar_from_panel=allow_test_calendar_from_panel,
        entry_blocked=limit_up,
        exit_blocked=limit_down,
        execution_status_known=limit_known,
    )
    return labels, {
        "definition": f"exact_frozen_calendar_scheduled_adjusted_{price_field}_endpoints_long_only_no_delay",
        "price_field": price_field,
        "is_full_execution_backtest": False,
        "tradable_source": tradable_source,
        "entry_block_source": limit_up_source,
        "exit_block_source": limit_down_source,
        "execution_status_known_source": limit_known_source,
    }


def summarize_endpoint_availability(
    forecast_labels: pd.Series,
    executable_labels: pd.Series,
    dates: Sequence[pd.Timestamp],
) -> dict[str, object]:
    """Disclose how much of the forecast sample can trade at both endpoints."""

    forecast = _date_slice(forecast_labels, dates)
    executable = _date_slice(executable_labels, dates)
    forecast_available = forecast.notna()
    executable_available = executable.notna() & forecast_available
    forecast_rows = int(forecast_available.sum())
    executable_rows = int(executable_available.sum())
    coverage = (
        float(executable_rows / forecast_rows) if forecast_rows else 0.0
    )
    return {
        "forecast_label_rows": forecast_rows,
        "immediately_executable_label_rows": executable_rows,
        "immediately_executable_share": coverage,
        "not_immediately_executable_share": float(1.0 - coverage),
    }


@dataclass(frozen=True)
class PriceScaleAudit:
    """Result of a per-security adjusted-price anchor perturbation."""

    invariant: bool
    compared_rows: int
    mismatch_rows: int
    maximum_absolute_error: float
    maximum_relative_error: float
    probe: str = "per_security_log_uniform_0.2x_5x_v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "invariant": self.invariant,
            "compared_rows": self.compared_rows,
            "mismatch_rows": self.mismatch_rows,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "probe": self.probe,
        }


def _stable_price_scale(symbol: object) -> float:
    payload = f"cogalpha-price-scale-v1|{symbol}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    unit = integer / float((1 << 64) - 1)
    return float(np.exp(np.log(0.2) + unit * (np.log(5.0) - np.log(0.2))))


def build_price_scale_probe(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Multiply each security's adjusted price history by a stable constant.

    Backward and forward adjustment schemes differ only by a positive constant
    for a security over a fixed history.  A research factor that changes under
    this perturbation is learning an arbitrary vendor/IPO price anchor rather
    than a return path, range ratio or price-volume mechanism.
    """

    if (
        not isinstance(feature_frame.index, pd.MultiIndex)
        or feature_frame.index.names != ["date", "symbol"]
    ):
        raise ValueError("price-scale probe requires MultiIndex(date, symbol)")
    missing = sorted(PRICE_FEATURE_FIELDS.difference(feature_frame.columns))
    if missing:
        raise ValueError("price-scale probe missing price fields: " + ",".join(missing))
    symbols = feature_frame.index.get_level_values("symbol")
    scale_by_symbol = {
        symbol: _stable_price_scale(symbol) for symbol in symbols.unique()
    }
    scales = symbols.map(scale_by_symbol).to_numpy(dtype=float)
    output = feature_frame.copy()
    for field in PRICE_FEATURE_FIELDS:
        values = pd.to_numeric(output[field], errors="coerce")
        if values.isna().any() or not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(f"price-scale probe field {field} must be finite numeric")
        output[field] = values.to_numpy(dtype=float) * scales
    return output


def audit_price_scale_invariance(
    expression: Mapping[str, object],
    feature_frame: pd.DataFrame,
    spec: ValidationSpec,
    *,
    baseline_values: pd.Series | None = None,
    perturbed_feature_frame: pd.DataFrame | None = None,
    cross_sectional_mask: pd.Series | None = None,
    perturbed_expression_memo: OrderedDict[str, pd.Series] | None = None,
    expression_memo_limit: int = 0,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-10,
) -> PriceScaleAudit:
    """Dynamically reject reliance on an arbitrary per-security price anchor."""

    if relative_tolerance <= 0 or absolute_tolerance < 0:
        raise ValueError("price-scale audit tolerances must be positive")
    baseline = (
        evaluate_expression(
            expression,
            feature_frame,
            spec,
            cross_sectional_mask=cross_sectional_mask,
        )
        if baseline_values is None
        else baseline_values
    )
    if not baseline.index.equals(feature_frame.index):
        raise ValueError("baseline factor values do not align with feature frame")
    probe = (
        build_price_scale_probe(feature_frame)
        if perturbed_feature_frame is None
        else perturbed_feature_frame
    )
    if not probe.index.equals(feature_frame.index) or tuple(probe.columns) != tuple(
        feature_frame.columns
    ):
        raise ValueError("perturbed feature frame does not align with feature frame")
    perturbed = evaluate_expression(
        expression,
        probe,
        spec,
        cross_sectional_mask=cross_sectional_mask,
        external_memo=perturbed_expression_memo,
        external_memo_limit=expression_memo_limit,
    )
    left = baseline.to_numpy(dtype=float)
    right = perturbed.to_numpy(dtype=float)
    same_nan = np.isnan(left) == np.isnan(right)
    finite = np.isfinite(left) & np.isfinite(right)
    comparable = same_nan & (finite | (np.isnan(left) & np.isnan(right)))
    close = np.zeros(len(left), dtype=bool)
    both_nan = np.isnan(left) & np.isnan(right)
    close[both_nan] = True
    close[finite] = np.isclose(
        left[finite],
        right[finite],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    passed = comparable & close
    difference = np.abs(left[finite] - right[finite])
    denominator = np.maximum(np.abs(left[finite]), np.abs(right[finite]))
    relative = difference / np.maximum(denominator, 1e-12)
    return PriceScaleAudit(
        invariant=bool(passed.all()),
        compared_rows=len(left),
        mismatch_rows=int((~passed).sum()),
        maximum_absolute_error=float(difference.max()) if len(difference) else 0.0,
        maximum_relative_error=float(relative.max()) if len(relative) else 0.0,
    )


@dataclass(frozen=True)
class SplitPlan:
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    sealed_dates: tuple[pd.Timestamp, ...]
    purge_after_train: tuple[pd.Timestamp, ...]
    purge_after_validation: tuple[pd.Timestamp, ...]
    label_horizon: int
    entry_delay: int
    outcome_tail: tuple[pd.Timestamp, ...] = ()

    @property
    def development_end(self) -> pd.Timestamp:
        return self.purge_after_validation[-1]

    def to_dict(self) -> dict[str, object]:
        def span(values: tuple[pd.Timestamp, ...]) -> dict[str, object]:
            return {
                "start": str(values[0].date()),
                "end": str(values[-1].date()),
                "trading_days": len(values),
            }

        return {
            "train": span(self.train_dates),
            "purge_after_train": span(self.purge_after_train),
            "validation": span(self.validation_dates),
            "purge_after_validation": span(self.purge_after_validation),
            "sealed": span(self.sealed_dates),
            "outcome_tail_not_scored": (
                span(self.outcome_tail) if self.outcome_tail else None
            ),
            "label_horizon": self.label_horizon,
            "entry_delay": self.entry_delay,
        }


def make_split_plan(
    panel: pd.DataFrame,
    *,
    label_horizon: int = 5,
    entry_delay: int = 1,
    train_fraction: float = 0.55,
    validation_fraction: float = 0.22,
) -> SplitPlan:
    dates = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    purge = label_horizon + entry_delay
    if label_horizon < 1 or entry_delay < 0:
        raise ValueError("invalid label timing")
    if not (0.3 <= train_fraction <= 0.75 and 0.1 <= validation_fraction <= 0.35):
        raise ValueError("split fractions are outside safe research ranges")
    train_n = int(len(dates) * train_fraction)
    valid_n = int(len(dates) * validation_fraction)
    sealed_n = len(dates) - train_n - valid_n - 3 * purge
    if min(train_n, valid_n, sealed_n) < 40:
        raise ValueError("not enough dates after two purged boundaries")
    train = tuple(pd.Timestamp(value) for value in dates[:train_n])
    purge1 = tuple(pd.Timestamp(value) for value in dates[train_n : train_n + purge])
    valid_start = train_n + purge
    valid = tuple(pd.Timestamp(value) for value in dates[valid_start : valid_start + valid_n])
    purge2_start = valid_start + valid_n
    purge2 = tuple(pd.Timestamp(value) for value in dates[purge2_start : purge2_start + purge])
    sealed_start = purge2_start + purge
    sealed = tuple(
        pd.Timestamp(value) for value in dates[sealed_start : len(dates) - purge]
    )
    outcome_tail = tuple(pd.Timestamp(value) for value in dates[len(dates) - purge :])
    return SplitPlan(
        train,
        valid,
        sealed,
        purge1,
        purge2,
        label_horizon,
        entry_delay,
        outcome_tail,
    )


def _date_slice(series: pd.Series, dates: Sequence[pd.Timestamp]) -> pd.Series:
    mask = series.index.get_level_values("date").isin(dates)
    return series.loc[mask]


def _panel_date_slice(frame: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    mask = frame.index.get_level_values("date").isin(dates)
    return frame.loc[mask]


def build_style_exposures(
    panel: pd.DataFrame,
    *,
    price_tape: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Open CNE-like controls available at the signal timestamp."""

    tape = panel if price_tape is None else price_tape
    if "adj_close" not in tape.columns:
        raise ValueError("price tape missing adjusted research field: adj_close")
    if not panel.index.isin(tape.index).all():
        raise ValueError("price tape does not cover every eligible member row")
    close = tape["adj_close"].astype(float)
    one_day = close.groupby(level="symbol", sort=False).pct_change(fill_method=None)

    def rolling(series: pd.Series, window: int, method: str) -> pd.Series:
        out = pd.Series(np.nan, index=series.index, dtype=float)
        for _, group in series.groupby(level="symbol", sort=False):
            ordered = group.sort_index(level="date")
            values = pd.Series(ordered.to_numpy(dtype=float))
            if method == "std":
                result = values.rolling(window, min_periods=window).std(ddof=0)
            elif method == "sum":
                result = values.rolling(window, min_periods=window).sum()
            else:
                raise ValueError(method)
            out.loc[ordered.index] = result.to_numpy()
        return out

    exposures = pd.DataFrame(index=panel.index)
    exposures["industry"] = panel["industry"].astype("string")
    exposures["log_mcap"] = pd.to_numeric(panel["log_mcap"], errors="coerce")
    exposures["beta"] = pd.to_numeric(panel["beta"], errors="coerce")
    exposures["momentum_20"] = rolling(one_day, 20, "sum").reindex(panel.index)
    exposures["volatility_20"] = rolling(one_day, 20, "std").reindex(panel.index)
    exposures["liquidity"] = np.log1p(pd.to_numeric(panel["amount"], errors="coerce").clip(lower=0.0))
    return exposures


@dataclass(frozen=True)
class FactorArtifact:
    """Small immutable commitment passed from development to the sealed gate.

    Earlier versions also retained the full signed and neutralized development
    series for every successful trial.  Those arrays are not consumed by the
    sealed gate (which deliberately recomputes the frozen AST) and made the
    faithful 2,384-candidate search require tens of gigabytes of resident
    memory.  Keep only the identity and scalar development commitments needed
    for integrity checking and reporting.
    """

    candidate_id: str
    canonical_expression_json: str
    expression_hash: str
    direction: int
    train_rank_ic: float
    validation_rank_ic: float
    validation_neutral_rank_ic: float


@dataclass(frozen=True)
class ParallelPreparedResult:
    """Order-independent worker result committed by the parent evaluator.

    Workers may calculate expressions, IC paths, neutralization and bootstrap
    statistics concurrently.  They do not own the rolling novelty archive,
    parent-score state, adaptive feedback, or persistent ledgers.  Those remain
    parent-process responsibilities and are applied in candidate order.
    """

    score: CandidateScore
    artifact: FactorArtifact | None
    validation_signed_values: np.ndarray | None
    validation_neutral_rank_values: np.ndarray | None


class DevelopmentEvaluator:
    """Only owns train, purge and inner-validation data—never sealed rows."""

    def __init__(
        self,
        development_panel: pd.DataFrame,
        split: SplitPlan,
        *,
        feature_tape: pd.DataFrame | None = None,
        trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex | None = None,
        allow_test_calendar_from_panel: bool = False,
        allow_legacy_eligibility: bool = False,
        min_names: int = 20,
        development_handoff_policy: Mapping[str, object] | None = None,
        terminal_label_bridges: Sequence[TerminalLabelBridge] = (),
        terminal_label_bridge_identity: Mapping[str, object] | None = None,
        bootstrap_samples: int = 200,
        novelty_archive_limit: int = 64,
        forecast_price_field: str = "vwap",
    ) -> None:
        max_dev = development_panel.index.get_level_values("date").max()
        if max_dev > split.development_end:
            raise ValueError("development evaluator was given sealed data")
        self.panel = development_panel.sort_index()
        self.tape_panel = (
            self.panel if feature_tape is None else feature_tape.sort_index()
        )
        if not self.panel.index.isin(self.tape_panel.index).all():
            raise ValueError("continuous feature tape does not cover every development member row")
        if self.tape_panel.index.get_level_values("date").max() > split.development_end:
            raise ValueError("development feature tape was given sealed data")
        self.cross_sectional_mask = build_cross_sectional_mask(
            self.tape_panel,
            self.panel,
            allow_legacy_eligibility=allow_legacy_eligibility,
        )
        self.trading_calendar = normalize_trading_calendar(
            trading_calendar,
            panel_index=self.tape_panel.index,
            allow_test_calendar_from_panel=allow_test_calendar_from_panel,
        )
        self.allow_test_calendar_from_panel = allow_test_calendar_from_panel
        self.split = split
        self.min_names = min_names
        self.bootstrap_samples = bootstrap_samples
        self.novelty_archive_limit = novelty_archive_limit
        if forecast_price_field not in {"open", "vwap"}:
            raise ValueError("forecast_price_field must be open or vwap")
        self.forecast_price_field = forecast_price_field
        self.spec = PRODUCTION_DSL_SPEC
        self.python_factor_policy = PythonFactorSandboxPolicy()
        self.feature_frame = build_research_feature_frame(self.tape_panel)
        self._prefix_cutoff = split.validation_dates[
            len(split.validation_dates) // 2
        ]
        prefix_rows = (
            self.feature_frame.index.get_level_values("date")
            <= self._prefix_cutoff
        )
        # The truncation boundary is fixed for the run. Materialize it once;
        # recreating this multi-million-row frame for every trial was pure
        # allocation overhead and did not strengthen the leakage audit.
        self._prefix_feature_frame = self.feature_frame.loc[prefix_rows]
        self._prefix_cross_sectional_mask = self.cross_sectional_mask.reindex(
            self._prefix_feature_frame.index
        )
        # Ordinary labels use the same point-in-time-adjusted return basis as
        # research prices. A sourced terminal exchange is valued separately
        # from endpoint fields declared raw; their provenance is authenticated
        # by the upstream frozen-tape lineage, not by this internal relation check.
        self.forecast_labels = forward_return(
            self.feature_frame[self.forecast_price_field],
            split.label_horizon,
            entry_delay=split.entry_delay,
            trading_calendar=self.trading_calendar,
        )
        self.forecast_labels, terminal_label_audit = apply_terminal_label_bridges(
            self.forecast_labels,
            self.tape_panel,
            self.panel.index,
            terminal_label_bridges,
            terminal_label_bridge_identity=terminal_label_bridge_identity,
            endpoint_price_field=self.forecast_price_field,
            horizon=split.label_horizon,
            entry_delay=split.entry_delay,
            trading_calendar=self.trading_calendar,
        )
        self.terminal_label_bridge_identity = dict(
            terminal_label_audit["terminal_label_bridge_identity"]
        )
        self.terminal_label_audit = terminal_label_audit
        # Backward-compatible alias: this is a forecast/mark-to-market target,
        # not a claim that both VWAP endpoints were immediately executable.
        self.labels = self.forecast_labels
        (
            self.executable_labels,
            self.executable_label_definition,
        ) = build_immediate_executable_labels(
            self.feature_frame,
            self.tape_panel,
            horizon=split.label_horizon,
            entry_delay=split.entry_delay,
            trading_calendar=self.trading_calendar,
            price_field=self.forecast_price_field,
        )
        self._member_forecast_labels = self.forecast_labels.reindex(
            self.panel.index
        )
        self.development_label_audit: Mapping[str, object] | None = None
        if development_handoff_policy is not None:
            policy = dict(development_handoff_policy)
            if (
                float(policy.get("minimum_daily_label_coverage", np.nan)) != 1.0
                or policy.get("allow_label_row_drop") is not False
            ):
                raise ValueError(
                    "development label policy must require 100% coverage and no row drops"
                )
            selected_dates = set(split.train_dates) | set(split.validation_dates)
            selected = self._member_forecast_labels.loc[
                self._member_forecast_labels.index.get_level_values("date").isin(
                    selected_dates
                )
            ].sort_index()
            finite = pd.to_numeric(selected, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).notna()
            records: list[dict[str, object]] = []
            failures: list[str] = []
            train_dates_set = set(split.train_dates)
            for date, group in finite.groupby(level="date", sort=True):
                eligible = int(len(group))
                valid = int(group.sum())
                share = float(valid / eligible) if eligible else 0.0
                partition = (
                    "train" if pd.Timestamp(date) in train_dates_set else "inner_validation"
                )
                records.append(
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "partition": partition,
                        "eligible_labels": eligible,
                        "valid_labels": valid,
                        "valid_share": share,
                    }
                )
                if eligible < int(policy.get("min_names", min_names)) or valid != eligible:
                    failures.append(f"{pd.Timestamp(date).date()}:{valid}/{eligible}")
            expected_dates = tuple(split.train_dates) + tuple(split.validation_dates)
            if tuple(pd.Timestamp(item["date"]) for item in records) != expected_dates:
                raise ValueError(
                    "development label audit does not cover exact train/validation dates"
                )
            if failures:
                raise ValueError(
                    f"development adjusted-{self.forecast_price_field} label coverage failed before search; "
                    + ", ".join(failures[:5])
                )
            mask_hash = hashlib.sha256(
                pd.util.hash_pandas_object(finite.astype(int), index=True)
                .to_numpy(dtype=np.uint64)
                .tobytes()
            ).hexdigest()
            audit_hash = hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            by_partition = {
                name: {
                    "dates": len(items),
                    "eligible_rows": sum(int(item["eligible_labels"]) for item in items),
                    "valid_rows": sum(int(item["valid_labels"]) for item in items),
                    "minimum_daily_valid_share": min(
                        float(item["valid_share"]) for item in items
                    ),
                }
                for name in ("train", "inner_validation")
                for items in [[item for item in records if item["partition"] == name]]
            }
            self.development_label_audit = {
                "schema_version": 1,
                "policy": policy,
                "terminal_label_bridge_identity": self.terminal_label_bridge_identity,
                "terminal_label_bridge_audit": self.terminal_label_audit,
                "valid_mask_hash": mask_hash,
                "daily_audit_hash": audit_hash,
                **by_partition,
                "all_development": {
                    "dates": len(records),
                    "eligible_rows": len(finite),
                    "valid_rows": int(finite.sum()),
                    "minimum_daily_valid_share": min(
                        float(item["valid_share"]) for item in records
                    ),
                },
                "all_passed": True,
            }
        # Only after the pre-registered label-completeness gate succeeds do we
        # build candidate-search probes. Thus a bad endpoint cannot consume a
        # 24-generation search budget before failing.
        self.price_scale_probe = build_price_scale_probe(self.feature_frame)
        self._member_executable_labels = self.executable_labels.reindex(
            self.panel.index
        )
        self._train_forecast_labels = _date_slice(
            self._member_forecast_labels, split.train_dates
        )
        self._validation_forecast_labels = _date_slice(
            self._member_forecast_labels, split.validation_dates
        )
        self._validation_executable_labels = _date_slice(
            self._member_executable_labels, split.validation_dates
        )
        self.validation_endpoint_availability = summarize_endpoint_availability(
            self.forecast_labels.reindex(self.panel.index),
            self.executable_labels.reindex(self.panel.index),
            split.validation_dates,
        )
        self.exposures = build_style_exposures(
            self.panel,
            price_tape=self.tape_panel,
        )
        self._validation_exposures = _panel_date_slice(
            self.exposures, split.validation_dates
        )
        self._validation_index = self._validation_exposures.index
        validation_dates = self._validation_index.get_level_values("date")
        validation_codes, _ = pd.factorize(validation_dates, sort=False)
        self._validation_rank_boundaries = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.flatnonzero(np.diff(validation_codes)).astype(np.int64) + 1,
                np.array([len(validation_codes)], dtype=np.int64),
            )
        )
        self._neutralizer = PreparedCrossSectionalNeutralizer(
            self._validation_exposures,
            categorical=("industry",),
            min_names=self.min_names,
        )
        self.artifacts: dict[str, FactorArtifact] = {}
        self.scores: dict[str, CandidateScore] = {}
        self._control_ic_cache: dict[str, float] = {}
        self._novelty_values: list[pd.Series] = []
        self._novelty_complete_ranks: list[np.ndarray | None] = []
        # Residualized validation ranks support low-correlation parent pairing.
        # Keep only a bounded recent working set: the current 96 children plus
        # the previous parent pool and elites fit comfortably without retaining
        # every million-row candidate path.
        self._pair_rank_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._pair_rank_cache_limit = 256
        # Parent pairs are queried repeatedly while 96 children are built.
        # Their frozen validation rank vectors do not change, so memoize the
        # exact scalar correlation instead of repeating 232 daily rank
        # correlations for every child slot.
        self._pair_correlation_cache: OrderedDict[frozenset[str], float] = (
            OrderedDict()
        )
        self._pair_correlation_cache_limit = 4096
        # These memos may reuse subexpressions only inside one candidate.  A
        # cross-candidate LRU makes numerically borderline validation gates
        # depend on which worker evaluated earlier candidates, so __call__
        # clears every memo before beginning a new candidate.
        self._full_expression_memo: OrderedDict[str, pd.Series] = OrderedDict()
        self._prefix_expression_memo: OrderedDict[str, pd.Series] = OrderedDict()
        self._scale_expression_memo: OrderedDict[str, pd.Series] = OrderedDict()
        self._expression_memo_limit = 64

    def _rank_ic_mean(self, values: pd.Series, dates: Sequence[pd.Timestamp]) -> float:
        selected = _date_slice(values.reindex(self.panel.index), dates)
        if tuple(dates) == self.split.train_dates:
            labels = self._train_forecast_labels
        elif tuple(dates) == self.split.validation_dates:
            labels = self._validation_forecast_labels
        else:
            labels = _date_slice(self._member_forecast_labels, dates)
        ic, _ = daily_ic(selected, labels, method="spearman", min_names=self.min_names)
        return float(ic.mean())

    def _neutralized(self, values: pd.Series) -> pd.Series:
        return self._neutralizer.transform(
            values.reindex(self._validation_index),
        )

    def _control_neutral_ic(self, expression: Mapping[str, object]) -> float:
        cache_key = expression_hash(expression, self.spec)
        cached = self._control_ic_cache.get(cache_key)
        if cached is not None:
            return cached
        values = evaluate_expression(
            expression,
            self.feature_frame,
            self.spec,
            cross_sectional_mask=self.cross_sectional_mask,
            external_memo=self._full_expression_memo,
            external_memo_limit=self._expression_memo_limit,
        )
        # Crossover controls are built from already oriented parents and have
        # a direction of +1 by construction.  They must not use labels to pick
        # the more flattering sign.
        neutral = self._neutralized(values)
        ic, _ = daily_ic(
            neutral,
            self._validation_forecast_labels,
            method="spearman",
            min_names=self.min_names,
        )
        result = float(ic.mean())
        self._control_ic_cache[cache_key] = result
        return result

    def _novelty(self, values: pd.Series) -> float:
        valid_values = _date_slice(
            values.reindex(self.panel.index), self.split.validation_dates
        )
        if not self._novelty_values:
            return 1.0
        current_ranks = self._complete_daily_ranks(valid_values)
        correlations: list[float] = []
        for old, old_ranks in zip(
            self._novelty_values, self._novelty_complete_ranks, strict=True
        ):
            if current_ranks is not None and old_ranks is not None:
                correlations.append(
                    abs(self._mean_complete_daily_rank_correlation(current_ranks, old_ranks))
                )
            else:
                correlations.append(
                    abs(factor_correlation(valid_values, old, min_names=self.min_names))
                )
        finite = [value for value in correlations if np.isfinite(value)]
        return float(np.clip(1.0 - max(finite, default=0.0), 0.0, 1.0))

    def _complete_daily_ranks(self, values: pd.Series) -> np.ndarray | None:
        aligned = values.reindex(self._validation_index)
        numeric = aligned.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(numeric).all():
            return None
        ranked = aligned.groupby(level="date", sort=False).rank(
            method="average", na_option="keep"
        )
        return ranked.to_numpy(dtype=float, na_value=np.nan)

    def _mean_complete_daily_rank_correlation(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        daily: list[float] = []
        for start, stop in zip(
            self._validation_rank_boundaries[:-1],
            self._validation_rank_boundaries[1:],
            strict=True,
        ):
            left_day = left[start:stop]
            right_day = right[start:stop]
            if len(left_day) < self.min_names:
                continue
            if np.unique(left_day).size < 2 or np.unique(right_day).size < 2:
                continue
            daily.append(float(np.corrcoef(left_day, right_day)[1, 0]))
        return float(np.mean(daily)) if daily else float("nan")

    def _store_pair_behavior(self, candidate_id: str, neutral: pd.Series) -> None:
        aligned = neutral.reindex(self._validation_index)
        ranked = aligned.groupby(level="date", sort=False).rank(
            method="average", na_option="keep"
        )
        self._pair_rank_cache[candidate_id] = ranked.to_numpy(
            dtype=np.float32, na_value=np.nan
        )
        self._pair_rank_cache.move_to_end(candidate_id)
        while len(self._pair_rank_cache) > self._pair_rank_cache_limit:
            evicted, _ = self._pair_rank_cache.popitem(last=False)
            stale = [key for key in self._pair_correlation_cache if evicted in key]
            for key in stale:
                self._pair_correlation_cache.pop(key, None)

    def residual_factor_correlation(self, left_id: str, right_id: str) -> float:
        """Mean daily absolute Spearman correlation of neutralized factor paths."""

        cache_key = frozenset((left_id, right_id))
        cached = self._pair_correlation_cache.get(cache_key)
        if cached is not None:
            self._pair_correlation_cache.move_to_end(cache_key)
            return cached
        left = self._pair_rank_cache.get(left_id)
        right = self._pair_rank_cache.get(right_id)
        if left is None or right is None:
            return 1.0
        daily: list[float] = []
        for start, stop in zip(
            self._validation_rank_boundaries[:-1],
            self._validation_rank_boundaries[1:],
            strict=True,
        ):
            left_day = left[start:stop]
            right_day = right[start:stop]
            valid = np.isfinite(left_day) & np.isfinite(right_day)
            if int(valid.sum()) < self.min_names:
                continue
            x = left_day[valid]
            y = right_day[valid]
            if np.unique(x).size < 2 or np.unique(y).size < 2:
                continue
            daily.append(float(np.corrcoef(x, y)[0, 1]))
        value = float(np.mean(daily)) if daily else 1.0
        if not np.isfinite(value):
            return 1.0
        self._pair_rank_cache.move_to_end(left_id)
        self._pair_rank_cache.move_to_end(right_id)
        result = float(abs(value))
        self._pair_correlation_cache[cache_key] = result
        self._pair_correlation_cache.move_to_end(cache_key)
        while len(self._pair_correlation_cache) > self._pair_correlation_cache_limit:
            self._pair_correlation_cache.popitem(last=False)
        return result

    def _inner_validation_rank_behavior(self, candidate: Candidate) -> pd.Series:
        """Recompute one frozen candidate's validation-only rank path.

        This intentionally does not use ``_novelty_values``: that bounded,
        rolling archive depends on trial order and is only a soft search term.
        The final Plus gate recomputes exact behavior lazily from the frozen
        AST, retains at most the selected family in memory, and has no sealed
        panel capability.
        """

        artifact = self.artifacts.get(candidate.candidate_id)
        if artifact is None:
            raise ValueError("passed candidate lacks a development artifact")
        if deterministic_candidate_id(candidate) != candidate.candidate_id:
            raise ValueError("candidate identity differs from its frozen behavior identity")
        canonical_expression = canonical_json(candidate.expression, self.spec)
        if canonical_expression != artifact.canonical_expression_json:
            raise ValueError("candidate expression differs from development artifact")
        if expression_hash(candidate.expression, self.spec) != artifact.expression_hash:
            raise ValueError("candidate expression hash differs from development artifact")
        declared_direction = candidate.metadata.get("expected_direction")
        if declared_direction not in {-1, 1} or int(declared_direction) != artifact.direction:
            raise ValueError("candidate direction differs from development artifact")
        values = evaluate_expression(
            json.loads(canonical_expression),
            self.feature_frame,
            self.spec,
            cross_sectional_mask=self.cross_sectional_mask,
            external_memo=self._full_expression_memo,
            external_memo_limit=self._expression_memo_limit,
        )
        validation = _date_slice(
            (values * artifact.direction).reindex(self.panel.index),
            self.split.validation_dates,
        )
        actual_dates = tuple(
            pd.Timestamp(value)
            for value in validation.index.get_level_values("date").unique()
        )
        if actual_dates != tuple(self.split.validation_dates):
            raise ValueError("candidate behavior omits frozen validation dates")
        if np.isinf(validation.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("candidate behavior contains infinite values")
        ranks = validation.groupby(level="date", sort=False).rank(pct=True)
        ranks.name = candidate.candidate_id
        return ranks

    def _complete_daily_behavior_correlation(
        self,
        left: pd.Series,
        right: pd.Series,
    ) -> float:
        """Return the mean daily cross-sectional Spearman, without omission.

        A behavioral hard gate must not turn missing dates into a more
        flattering comparison.  Every represented validation date therefore
        needs at least ``min_names`` overlapping finite, non-constant values;
        otherwise the comparison is invalid and its candidate is rejected.
        Row-level NaNs remain permitted for suspensions and incomplete vendor
        histories, provided the daily cross-section is still auditable.
        """

        if not isinstance(left, pd.Series) or not isinstance(right, pd.Series):
            raise ValueError("candidate behavior must be a pandas Series")
        if not left.index.equals(right.index):
            raise ValueError("candidate behaviors do not share the frozen validation index")
        if not isinstance(left.index, pd.MultiIndex) or tuple(left.index.names) != (
            "date",
            "symbol",
        ):
            raise ValueError("candidate behavior must use the date/symbol panel index")
        if left.index.has_duplicates:
            raise ValueError("candidate behavior index contains duplicate rows")
        left_values = left.to_numpy(dtype=float, na_value=np.nan)
        right_values = right.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(left_values).any() or np.isinf(right_values).any():
            raise ValueError("candidate behavior contains infinite values")
        expected_dates = left.index.get_level_values("date").unique()
        if len(expected_dates) < 1:
            raise ValueError("candidate behavior has no validation dates")
        daily, _ = daily_ic(
            left,
            right,
            method="spearman",
            min_names=self.min_names,
        )
        if (
            len(daily) != len(expected_dates)
            or not daily.index.equals(expected_dates.sort_values())
            or not np.isfinite(daily.to_numpy(dtype=float, na_value=np.nan)).all()
        ):
            raise ValueError(
                "candidate behavior has a missing or non-finite daily correlation"
            )
        result = float(daily.mean())
        if not np.isfinite(result):
            raise ValueError("candidate behavior correlation is non-finite")
        return result

    def select_behaviorally_distinct_champions(
        self,
        eligible: Sequence[EvaluatedCandidate],
        final_factors: int,
        policy: BehavioralChampionPolicy,
    ) -> ChampionSelectionResult:
        """Greedily select an already-ranked family under a hard correlation cap.

        The caller owns the pre-registered ordering (fitness or Pareto/NSGA-II).
        A bounded leading pool is scanned first; only when it cannot fill the
        requested family is the remainder of the passed archive considered.
        The threshold is never relaxed.  Only rank paths for accepted
        candidates remain live during selection, and all are released before
        this method returns.
        """

        if final_factors < 1:
            raise ValueError("final_factors must be positive")
        ranked = list(eligible)
        if any(not item.score.passed for item in ranked):
            raise ValueError("behavioral champion gate accepts only passed candidates")

        top_pool_size = min(
            len(ranked),
            max(final_factors, policy.initial_top_pool_size),
        )
        selected: list[EvaluatedCandidate] = []
        selected_behaviors: dict[str, pd.Series] = {}
        selected_pair_correlations: dict[frozenset[str], float] = {}
        conflict_by_nearest: dict[str, int] = {}
        conflict_examples: list[dict[str, object]] = []
        correlation_conflicts = 0
        correlation_conflict_pairs = 0
        nonfinite_correlation_rejections = 0
        behavior_recompute_failures = 0
        comparisons = 0
        considered = 0
        considered_in_top_pool = 0

        def consider(item: EvaluatedCandidate, *, top_pool: bool) -> None:
            nonlocal correlation_conflicts
            nonlocal correlation_conflict_pairs
            nonlocal nonfinite_correlation_rejections
            nonlocal behavior_recompute_failures
            nonlocal comparisons
            nonlocal considered
            nonlocal considered_in_top_pool
            considered += 1
            considered_in_top_pool += int(top_pool)
            candidate_id = item.candidate.candidate_id
            try:
                behavior = self._inner_validation_rank_behavior(item.candidate)
                # The first candidate has no pairwise comparison.  Validate
                # its complete daily cross-section against itself so a
                # constant, empty or otherwise non-auditable path cannot enter
                # the family and poison every later comparison.
                self._complete_daily_behavior_correlation(behavior, behavior)
            except (DSLValidationError, ValueError, KeyError, FloatingPointError):
                behavior_recompute_failures += 1
                return

            candidate_correlations: dict[str, float] = {}
            invalid_correlation = False
            for selected_id, selected_behavior in selected_behaviors.items():
                try:
                    correlation = abs(
                        self._complete_daily_behavior_correlation(
                            behavior,
                            selected_behavior,
                        )
                    )
                except (ValueError, KeyError, FloatingPointError):
                    invalid_correlation = True
                    break
                comparisons += 1
                if not np.isfinite(correlation):
                    invalid_correlation = True
                    break
                candidate_correlations[selected_id] = float(correlation)
            if invalid_correlation:
                nonfinite_correlation_rejections += 1
                return

            nearest_id: str | None = None
            maximum = 0.0
            if candidate_correlations:
                nearest_id, maximum = max(
                    candidate_correlations.items(),
                    key=lambda pair: (pair[1], pair[0]),
                )
            if maximum > policy.maximum_absolute_correlation:
                correlation_conflicts += 1
                correlation_conflict_pairs += sum(
                    correlation > policy.maximum_absolute_correlation
                    for correlation in candidate_correlations.values()
                )
                assert nearest_id is not None
                conflict_by_nearest[nearest_id] = (
                    conflict_by_nearest.get(nearest_id, 0) + 1
                )
                if len(conflict_examples) < 25:
                    conflict_examples.append(
                        {
                            "candidate_id": candidate_id,
                            "fitness": float(item.score.fitness),
                            "nearest_selected_candidate_id": nearest_id,
                            "maximum_absolute_correlation": float(maximum),
                        }
                    )
                return

            for selected_id, correlation in candidate_correlations.items():
                selected_pair_correlations[
                    frozenset((candidate_id, selected_id))
                ] = correlation
            selected.append(item)
            selected_behaviors[candidate_id] = behavior

        for item in ranked[:top_pool_size]:
            if len(selected) >= final_factors:
                break
            consider(item, top_pool=True)

        expanded = len(selected) < final_factors and top_pool_size < len(ranked)
        if expanded:
            for item in ranked[top_pool_size:]:
                if len(selected) >= final_factors:
                    break
                consider(item, top_pool=False)

        selected_ids = [item.candidate.candidate_id for item in selected]
        champion_relationships: list[dict[str, object]] = []
        for rank, item in enumerate(selected, start=1):
            candidate_id = item.candidate.candidate_id
            correlations = [
                (other_id, selected_pair_correlations[frozenset((candidate_id, other_id))])
                for other_id in selected_ids
                if other_id != candidate_id
                and frozenset((candidate_id, other_id))
                in selected_pair_correlations
            ]
            nearest_id: str | None = None
            maximum = 0.0
            if correlations:
                nearest_id, maximum = max(
                    correlations,
                    key=lambda pair: (pair[1], pair[0]),
                )
            champion_relationships.append(
                {
                    "selection_rank": rank,
                    "candidate_id": candidate_id,
                    "fitness": float(item.score.fitness),
                    "nearest_champion_candidate_id": nearest_id,
                    "maximum_absolute_correlation": float(maximum),
                }
            )

        audit: dict[str, object] = {
            "schema": "behavioral_champion_selection_v1",
            "mode": "plus_inner_validation_hard_behavioral_dedup",
            "metric": "mean_daily_cross_sectional_spearman_of_percentile_ranks",
            "maximum_absolute_correlation_threshold": float(
                policy.maximum_absolute_correlation
            ),
            "initial_top_pool_size_configured": policy.initial_top_pool_size,
            "initial_top_pool_size_resolved": top_pool_size,
            "eligible_passed_count": len(ranked),
            "required_final_factors": final_factors,
            "selected_count": len(selected),
            "candidates_considered": considered,
            "initial_pool_candidates_considered": considered_in_top_pool,
            "expanded_to_all_passed": expanded,
            "full_passed_archive_exhausted": considered == len(ranked),
            "correlation_comparisons": comparisons,
            "correlation_conflict_candidates": correlation_conflicts,
            "correlation_conflict_pairs": correlation_conflict_pairs,
            "nonfinite_correlation_rejections": nonfinite_correlation_rejections,
            "behavior_recompute_failures": behavior_recompute_failures,
            "conflicts_by_nearest_selected_candidate": dict(
                sorted(conflict_by_nearest.items())
            ),
            "conflict_examples": conflict_examples,
            "champions": champion_relationships,
            "behavior_vectors_retained_after_selection": 0,
            "data_scope": "inner_validation_only",
            "sealed_data_consumed": False,
            "threshold_relaxed": False,
            "status": (
                "complete"
                if len(selected) == final_factors
                else "failed_insufficient_behaviorally_distinct_champions"
            ),
        }
        # Drop the only full-length behavior vectors before crossing back into
        # the model-agnostic evolution engine.  The compact scalar pair audit
        # above is the sole retained behavior evidence.
        selected_behaviors.clear()
        return ChampionSelectionResult(tuple(selected), audit)

    def _rejected(self, reason: str) -> CandidateScore:
        return CandidateScore(
            fitness=-1.0,
            signed_rank_ic=0.0,
            neutral_rank_ic=0.0,
            turnover=1.0,
            worst_window_ic=-1.0,
            style_retention=0.0,
            novelty=0.0,
            passed=False,
            diagnostics={"rejection": reason},
        )

    def __call__(self, candidate: Candidate) -> CandidateScore:
        # Candidate validation must be a pure function of the candidate and
        # frozen data.  In particular, worker-local LRU history may not change
        # the price-scale gate or any downstream adaptive feedback.
        self._full_expression_memo.clear()
        self._prefix_expression_memo.clear()
        self._scale_expression_memo.clear()
        self._control_ic_cache.clear()
        try:
            candidate_id = candidate.candidate_id
            canonical_expression_json = canonical_json(
                candidate.expression,
                self.spec,
            )
            evaluated_expression = json.loads(canonical_expression_json)
            python_factor_source = render_python_factor(
                evaluated_expression, self.spec
            )
            parsed_python = parse_python_factor(
                python_factor_source,
                self.spec,
                policy=self.python_factor_policy,
            )
            if (
                canonical_json(parsed_python.expression, self.spec)
                != canonical_expression_json
            ):
                return self._rejected("python_factor_round_trip_mismatch")
            canonical_expression_hash = expression_hash(
                evaluated_expression,
                self.spec,
            )
            report = validate_expression(evaluated_expression, self.spec)
            values = evaluate_expression(
                evaluated_expression,
                self.feature_frame,
                self.spec,
                cross_sectional_mask=self.cross_sectional_mask,
                external_memo=self._full_expression_memo,
                external_memo_limit=self._expression_memo_limit,
            )
            # Prefix invariance is a dynamic complement to static DSL checks.
            prefix_values = evaluate_expression(
                evaluated_expression,
                self._prefix_feature_frame,
                self.spec,
                cross_sectional_mask=self._prefix_cross_sectional_mask,
                external_memo=self._prefix_expression_memo,
                external_memo_limit=self._expression_memo_limit,
            )
            common = values.reindex(prefix_values.index)
            if not np.allclose(common.to_numpy(), prefix_values.to_numpy(), equal_nan=True, rtol=0.0, atol=1e-12):
                return self._rejected("truncation_invariance_failed")
            # A fixed adjusted-price anchor is a data-vendor convention, not
            # an economic signal.  Perturb every security by a different
            # positive constant and fail closed unless the factor path is
            # unchanged.  This also makes accepted factors immune to the
            # constant historical rewrite inherent in forward adjustment.
            scale_audit = audit_price_scale_invariance(
                evaluated_expression,
                self.feature_frame,
                self.spec,
                baseline_values=values,
                perturbed_feature_frame=self.price_scale_probe,
                cross_sectional_mask=self.cross_sectional_mask,
                perturbed_expression_memo=self._scale_expression_memo,
                expression_memo_limit=self._expression_memo_limit,
            )
            if not scale_audit.invariant:
                # The gate decision is categorical.  A mismatch-row count is
                # an implementation-order diagnostic: bounded expression LRU
                # reuse can change the number of numerically borderline rows
                # without changing the invariant/fail decision.  Feeding that
                # incidental count into adaptive feedback would make otherwise
                # equivalent serial and parallel runs carry different evidence.
                return self._rejected("price_scale_invariance_failed")
        except (
            DSLValidationError,
            PythonFactorValidationError,
            ValueError,
            KeyError,
            FloatingPointError,
        ) as exc:
            return self._rejected(f"dsl_or_runtime:{type(exc).__name__}:{exc}")

        train_raw_ic = self._rank_ic_mean(values, self.split.train_dates)
        if not np.isfinite(train_raw_ic):
            return self._rejected("insufficient_train_ic")
        declared_direction = candidate.metadata.get("expected_direction")
        if declared_direction not in {-1, 1}:
            return self._rejected("missing_predeclared_expected_direction")
        direction = int(declared_direction)
        direction_source = "predeclared_hypothesis"
        signed = (values * direction).reindex(self.panel.index)
        valid_signed = _date_slice(signed, self.split.validation_dates)
        valid_labels = self._validation_forecast_labels
        usable_labels = valid_labels.notna()
        coverage = float(valid_signed[usable_labels].notna().mean()) if usable_labels.any() else 0.0
        if coverage < 0.70:
            return self._rejected(f"coverage_below_gate:{coverage:.4f}")

        neutral = self._neutralized(signed)
        raw_ic, names = daily_ic(valid_signed, valid_labels, method="spearman", min_names=self.min_names)
        neutral_ic, neutral_names = daily_ic(
            _date_slice(neutral, self.split.validation_dates),
            valid_labels,
            method="spearman",
            min_names=self.min_names,
        )
        raw_summary = summarize_ic(
            raw_ic,
            names,
            hac_lags=max(self.split.label_horizon - 1, 0),
            block_size=min(40, max(10, self.split.label_horizon * 2)),
            bootstrap_samples=self.bootstrap_samples,
            seed=19,
        )
        neutral_summary = summarize_ic(
            neutral_ic,
            neutral_names,
            hac_lags=max(self.split.label_horizon - 1, 0),
            block_size=min(40, max(10, self.split.label_horizon * 2)),
            bootstrap_samples=self.bootstrap_samples,
            seed=23,
        )
        if not np.isfinite(neutral_summary.mean):
            return self._rejected("insufficient_neutral_validation_ic")

        # This diagnostic uses the same frozen factor and dates, but removes
        # observations whose scheduled entry/exit could not trade.  It is not
        # used in fitness: conditioning model selection on future fill states
        # would change the forecasting objective and invite a different form
        # of selection bias.  The event-driven portfolio gate remains the
        # authority for realizable performance.
        executable_valid_labels = self._validation_executable_labels
        executable_raw_ic, executable_names = daily_ic(
            valid_signed,
            executable_valid_labels,
            method="spearman",
            min_names=self.min_names,
        )
        executable_neutral_ic, executable_neutral_names = daily_ic(
            _date_slice(neutral, self.split.validation_dates),
            executable_valid_labels,
            method="spearman",
            min_names=self.min_names,
        )
        executable_raw_summary = summarize_ic(
            executable_raw_ic,
            executable_names,
            hac_lags=max(self.split.label_horizon - 1, 0),
            block_size=min(40, max(10, self.split.label_horizon * 2)),
            bootstrap_samples=self.bootstrap_samples,
            seed=29,
        )
        executable_neutral_summary = summarize_ic(
            executable_neutral_ic,
            executable_neutral_names,
            hac_lags=max(self.split.label_horizon - 1, 0),
            block_size=min(40, max(10, self.split.label_horizon * 2)),
            bootstrap_samples=self.bootstrap_samples,
            seed=31,
        )

        chunks = np.array_split(neutral_ic.dropna().to_numpy(), 4)
        worst = min(float(chunk.mean()) for chunk in chunks if len(chunk))
        turnover = rank_turnover(valid_signed)
        novelty = self._novelty(valid_signed)
        retention = float(np.clip(abs(neutral_summary.mean) / (abs(raw_summary.mean) + 1e-6), 0.0, 2.0))
        complexity = report.node_count
        control_delta: float | None = None
        comparator = -math.inf
        parent_neutral = [self.scores[parent].neutral_rank_ic for parent in candidate.parents if parent in self.scores]
        if parent_neutral:
            comparator = max(parent_neutral)
        control_ic: float | None = None
        if candidate.control_expression is not None:
            try:
                control_ic = self._control_neutral_ic(candidate.control_expression)
                if np.isfinite(control_ic):
                    comparator = max(comparator, control_ic)
            except (DSLValidationError, ValueError, KeyError):
                control_ic = None
        if comparator > -math.inf:
            control_delta = float(neutral_summary.mean - comparator)

        fitness = (
            0.38 * neutral_summary.mean
            + 0.17 * raw_summary.mean
            + 0.20 * worst
            + 0.12 * neutral_summary.bootstrap_low
            + 0.006 * novelty
            - 0.004 * turnover
            - 0.00008 * complexity
        )
        if control_delta is not None:
            fitness += 0.15 * min(control_delta, 0.0) + 0.04 * max(control_delta, 0.0)
        try:
            current_expression_json = canonical_json(candidate.expression, self.spec)
        except DSLValidationError as exc:
            return self._rejected(
                f"candidate_mutated_during_evaluation:{type(exc).__name__}:{exc}"
            )
        if (
            candidate.candidate_id != candidate_id
            or current_expression_json != canonical_expression_json
        ):
            return self._rejected("candidate_mutated_during_evaluation")
        score = CandidateScore(
            fitness=float(fitness),
            signed_rank_ic=float(raw_summary.mean),
            neutral_rank_ic=float(neutral_summary.mean),
            turnover=float(turnover),
            worst_window_ic=float(worst),
            style_retention=retention,
            novelty=novelty,
            control_delta=control_delta,
            passed=True,
            diagnostics={
                "direction": direction,
                "direction_source": direction_source,
                "train_raw_rank_ic": train_raw_ic,
                "coverage": coverage,
                "ast_nodes": complexity,
                "lookback": report.lookback,
                "price_scale_audit": scale_audit.to_dict(),
                "python_factor_source": python_factor_source,
                "python_factor_source_sha256": parsed_python.source_sha256,
                "python_factor_sandbox": dict(parsed_python.sandbox_policy),
                "forecast_label_definition": (
                    f"adjusted_{self.forecast_price_field}_exact_frozen_calendar_endpoint_return_without_future_fill_conditioning"
                ),
                "immediate_executable_label_definition": dict(
                    self.executable_label_definition
                ),
                "validation_endpoint_availability": dict(
                    self.validation_endpoint_availability
                ),
                "raw_summary": raw_summary.to_dict(),
                "neutral_summary": neutral_summary.to_dict(),
                "immediate_executable_raw_summary": (
                    executable_raw_summary.to_dict()
                ),
                "immediate_executable_neutral_summary": (
                    executable_neutral_summary.to_dict()
                ),
                "linear_control_neutral_rank_ic": control_ic,
            },
        )
        self.artifacts[candidate_id] = FactorArtifact(
            candidate_id=candidate_id,
            canonical_expression_json=canonical_expression_json,
            expression_hash=canonical_expression_hash,
            direction=direction,
            train_rank_ic=float(train_raw_ic * direction),
            validation_rank_ic=float(raw_summary.mean),
            validation_neutral_rank_ic=float(neutral_summary.mean),
        )
        self.scores[candidate_id] = score
        self._store_pair_behavior(
            candidate_id,
            _date_slice(neutral, self.split.validation_dates),
        )
        self._novelty_values.append(valid_signed)
        self._novelty_complete_ranks.append(
            self._complete_daily_ranks(valid_signed)
        )
        if len(self._novelty_values) > self.novelty_archive_limit:
            self._novelty_values.pop(0)
            self._novelty_complete_ranks.pop(0)
        return score

    def commit_parallel_result(
        self,
        candidate: Candidate,
        prepared: ParallelPreparedResult,
    ) -> CandidateScore:
        """Commit a worker result with the exact serial ordering semantics."""

        worker_score = prepared.score
        if not worker_score.passed:
            if prepared.artifact is not None or prepared.validation_signed_values is not None:
                raise ValueError("rejected worker result contains passed-candidate state")
            return worker_score
        artifact = prepared.artifact
        values = prepared.validation_signed_values
        neutral_rank_values = prepared.validation_neutral_rank_values
        if artifact is None or values is None or neutral_rank_values is None:
            raise ValueError("passed worker result lacks artifact or validation values")
        if (
            len(values) != len(self._validation_index)
            or len(neutral_rank_values) != len(self._validation_index)
        ):
            raise ValueError("worker validation values do not align with parent evaluator")
        if (
            artifact.candidate_id != candidate.candidate_id
            or deterministic_candidate_id(candidate) != candidate.candidate_id
            or canonical_json(candidate.expression, self.spec)
            != artifact.canonical_expression_json
        ):
            raise ValueError("candidate identity changed between worker and ordered commit")

        valid_signed = pd.Series(
            np.asarray(values, dtype=float),
            index=self._validation_index,
            dtype=float,
        )
        novelty = self._novelty(valid_signed)
        diagnostics = dict(worker_score.diagnostics)
        neutral_summary = diagnostics.get("neutral_summary")
        if not isinstance(neutral_summary, Mapping):
            raise ValueError("passed worker result lacks neutral IC summary")
        bootstrap_low = float(neutral_summary.get("bootstrap_low", np.nan))
        if not np.isfinite(bootstrap_low):
            raise ValueError("worker neutral bootstrap bound is non-finite")

        comparator = -math.inf
        parent_neutral = [
            self.scores[parent].neutral_rank_ic
            for parent in candidate.parents
            if parent in self.scores
        ]
        if parent_neutral:
            comparator = max(parent_neutral)
        control_ic_raw = diagnostics.get("linear_control_neutral_rank_ic")
        control_ic = (
            float(control_ic_raw)
            if control_ic_raw is not None and np.isfinite(float(control_ic_raw))
            else None
        )
        if control_ic is not None:
            comparator = max(comparator, control_ic)
        control_delta = (
            float(worker_score.neutral_rank_ic - comparator)
            if comparator > -math.inf
            else None
        )

        complexity = int(diagnostics.get("ast_nodes", -1))
        if complexity < 1:
            raise ValueError("passed worker result has invalid expression complexity")
        fitness = (
            0.38 * worker_score.neutral_rank_ic
            + 0.17 * worker_score.signed_rank_ic
            + 0.20 * worker_score.worst_window_ic
            + 0.12 * bootstrap_low
            + 0.006 * novelty
            - 0.004 * worker_score.turnover
            - 0.00008 * complexity
        )
        if control_delta is not None:
            fitness += 0.15 * min(control_delta, 0.0) + 0.04 * max(
                control_delta, 0.0
            )
        score = CandidateScore(
            fitness=float(fitness),
            signed_rank_ic=worker_score.signed_rank_ic,
            neutral_rank_ic=worker_score.neutral_rank_ic,
            turnover=worker_score.turnover,
            worst_window_ic=worker_score.worst_window_ic,
            style_retention=worker_score.style_retention,
            novelty=novelty,
            control_delta=control_delta,
            passed=True,
            diagnostics=diagnostics,
        )
        self.artifacts[candidate.candidate_id] = artifact
        self.scores[candidate.candidate_id] = score
        # The worker already computed this exact neutralized rank vector while
        # producing the score. Reusing it avoids a redundant date-by-date
        # linear solve in the serial commit path without changing any stateful
        # novelty, parent comparison, or ledger ordering.
        self._pair_rank_cache[candidate.candidate_id] = np.asarray(
            neutral_rank_values,
            dtype=np.float32,
        ).copy()
        self._pair_rank_cache.move_to_end(candidate.candidate_id)
        while len(self._pair_rank_cache) > self._pair_rank_cache_limit:
            evicted, _ = self._pair_rank_cache.popitem(last=False)
            stale = [key for key in self._pair_correlation_cache if evicted in key]
            for key in stale:
                self._pair_correlation_cache.pop(key, None)
        self._novelty_values.append(valid_signed)
        self._novelty_complete_ranks.append(
            self._complete_daily_ranks(valid_signed)
        )
        if len(self._novelty_values) > self.novelty_archive_limit:
            self._novelty_values.pop(0)
            self._novelty_complete_ranks.pop(0)
        return score


class SealedGate:
    """One-use outer gate. Results are returned for reporting, never evolution."""

    def __init__(
        self,
        full_panel: pd.DataFrame,
        split: SplitPlan,
        *,
        feature_tape: pd.DataFrame | None = None,
        trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex | None = None,
        research_universe_identity: Mapping[str, object] | None = None,
        trading_calendar_identity: Mapping[str, object] | None = None,
        allow_test_calendar_from_panel: bool = False,
        allow_legacy_eligibility: bool = False,
        min_names: int = 20,
        sealed_claim_capability: object | None = None,
        terminal_label_bridges: Sequence[TerminalLabelBridge] = (),
        terminal_label_bridge_identity: Mapping[str, object] | None = None,
        forecast_price_field: str = "vwap",
    ) -> None:
        self._panel = full_panel.sort_index()
        self._tape_panel = (
            self._panel if feature_tape is None else feature_tape.sort_index()
        )
        if not self._panel.index.isin(self._tape_panel.index).all():
            raise ValueError("continuous feature tape does not cover every sealed member row")
        self._cross_sectional_mask = build_cross_sectional_mask(
            self._tape_panel,
            self._panel,
            allow_legacy_eligibility=allow_legacy_eligibility,
        )
        self._trading_calendar = normalize_trading_calendar(
            trading_calendar,
            panel_index=self._tape_panel.index,
            allow_test_calendar_from_panel=allow_test_calendar_from_panel,
        )
        if research_universe_identity is None:
            if not allow_test_calendar_from_panel:
                raise ValueError(
                    "sealed gate requires the frozen research-universe identity"
                )
            research_universe_identity = build_research_universe_identity(
                self._panel, self._tape_panel, self._panel
            )
        if trading_calendar_identity is None:
            if not allow_test_calendar_from_panel:
                raise ValueError(
                    "sealed gate requires the frozen trading-calendar identity"
                )
            calendar_payload = {
                "schema_version": 1,
                "source": "synthetic_panel_explicit_test_fallback",
                "trading_dates_hash": hash_json(
                    [str(pd.Timestamp(value).date()) for value in self._trading_calendar]
                ),
                "trading_days": len(self._trading_calendar),
                "first_date": str(self._trading_calendar[0].date()),
                "last_date": str(self._trading_calendar[-1].date()),
                "endpoint_rule_hash": research_rule_identity_v1()["rule_hash"],
            }
            trading_calendar_identity = {
                **calendar_payload,
                "identity_hash": hash_json(calendar_payload),
            }
        self._research_universe_identity = dict(research_universe_identity)
        self._trading_calendar_identity = dict(trading_calendar_identity)
        for label, identity in (
            ("research universe", self._research_universe_identity),
            ("trading calendar", self._trading_calendar_identity),
        ):
            payload = {
                key: value for key, value in identity.items() if key != "identity_hash"
            }
            if identity.get("identity_hash") != hash_json(payload):
                raise ValueError(f"sealed {label} identity hash mismatch")
        if self._research_universe_identity.get(
            "eligible_panel_hash"
        ) != _semantic_panel_hash(self._panel):
            raise ValueError("sealed eligible panel differs from universe identity")
        if self._research_universe_identity.get(
            "feature_tape_eligibility_mask_hash"
        ) != _semantic_panel_hash(
            self._cross_sectional_mask.astype(np.int8).rename(
                "research_eligible"
            )
        ):
            raise ValueError("sealed eligibility mask differs from universe identity")
        if self._trading_calendar_identity.get("trading_dates_hash") != hash_json(
            [str(pd.Timestamp(value).date()) for value in self._trading_calendar]
        ):
            raise ValueError("sealed trading calendar differs from calendar identity")
        self._feature_frame = build_research_feature_frame(self._tape_panel)
        self._split = split
        self._min_names = min_names
        self._sealed_claim_capability = sealed_claim_capability
        self._revealed = False
        self._composite_reveal: dict[str, object] | None = None
        self._spec = PRODUCTION_DSL_SPEC
        if forecast_price_field not in {"open", "vwap"}:
            raise ValueError("forecast_price_field must be open or vwap")
        self._forecast_price_field = forecast_price_field
        self._terminal_label_bridges = tuple(terminal_label_bridges)
        self._terminal_label_bridge_identity = (
            validate_terminal_label_bridge_identity(
                self._terminal_label_bridges, terminal_label_bridge_identity
            )
            if self._terminal_label_bridges
            else {}
        )

    @property
    def revealed(self) -> bool:
        return self._revealed

    @property
    def composite_reveal(self) -> Mapping[str, object] | None:
        return self._composite_reveal

    def reveal(
        self,
        champions: Sequence[EvaluatedCandidate],
        development_artifacts: Mapping[str, FactorArtifact],
        *,
        composite_family: object | None = None,
        composite_family_hash: str | None = None,
    ) -> list[dict[str, object]]:
        if self._revealed:
            raise RuntimeError("sealed gate has already been revealed and is now retired")
        self._revealed = True
        labels = forward_return(
            self._feature_frame[self._forecast_price_field],
            self._split.label_horizon,
            entry_delay=self._split.entry_delay,
            trading_calendar=self._trading_calendar,
        )
        labels, terminal_label_audit = apply_terminal_label_bridges(
            labels,
            self._tape_panel,
            self._panel.index,
            self._terminal_label_bridges,
            terminal_label_bridge_identity=self._terminal_label_bridge_identity,
            endpoint_price_field=self._forecast_price_field,
            horizon=self._split.label_horizon,
            entry_delay=self._split.entry_delay,
            trading_calendar=self._trading_calendar,
        )
        executable_labels, executable_label_definition = (
            build_immediate_executable_labels(
                self._feature_frame,
                self._tape_panel,
                horizon=self._split.label_horizon,
                entry_delay=self._split.entry_delay,
                trading_calendar=self._trading_calendar,
                price_field=self._forecast_price_field,
            )
        )
        endpoint_availability = summarize_endpoint_availability(
            labels.reindex(self._panel.index),
            executable_labels.reindex(self._panel.index),
            self._split.sealed_dates,
        )
        sealed_forecast_labels = _date_slice(
            labels.reindex(self._panel.index), self._split.sealed_dates
        )
        assert isinstance(sealed_forecast_labels, pd.Series)
        forecast_finite = pd.to_numeric(
            sealed_forecast_labels, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).notna()
        forecast_daily = forecast_finite.groupby(level="date", sort=True).agg(
            ["size", "sum"]
        )
        expected_sealed_dates = pd.DatetimeIndex(self._split.sealed_dates)
        if (
            not pd.DatetimeIndex(forecast_daily.index).equals(expected_sealed_dates)
            or (forecast_daily["sum"] != forecast_daily["size"]).any()
        ):
            raise ValueError(
                f"sealed adjusted-{self._forecast_price_field} forecast labels are not 100% complete"
            )
        sealed_forecast_label_audit = {
            "rule_identity": research_rule_identity_v1(),
            "terminal_label_bridge_identity": self._terminal_label_bridge_identity,
            "terminal_label_bridge_audit": terminal_label_audit,
            "research_universe_identity_hash": self._research_universe_identity[
                "identity_hash"
            ],
            "trading_calendar_identity_hash": self._trading_calendar_identity[
                "identity_hash"
            ],
            "eligible_rows": int(forecast_daily["size"].sum()),
            "valid_rows": int(forecast_daily["sum"].sum()),
            "valid_share": 1.0,
            "minimum_daily_valid_share": 1.0,
            "sealed_dates_dropped": 0,
            "all_passed": True,
        }
        forecast_daily_records = [
            {
                "date": str(pd.Timestamp(date).date()),
                "eligible_rows": int(row["size"]),
                "valid_rows": int(row["sum"]),
            }
            for date, row in forecast_daily.iterrows()
        ]
        sealed_forecast_label_audit["daily_counts"] = forecast_daily_records
        sealed_forecast_label_audit["daily_counts_hash"] = hash_json(
            forecast_daily_records
        )
        exposures = build_style_exposures(
            self._panel,
            price_tape=self._tape_panel,
        )
        output: list[dict[str, object]] = []
        neutral_ic_family: dict[str, pd.Series] = {}
        executable_neutral_ic_family: dict[str, pd.Series] = {}
        sealed_neutral_features = pd.DataFrame()

        def require_complete_ic_path(
            ic: pd.Series,
            names: pd.Series,
            *,
            label: str,
        ) -> None:
            expected = pd.DatetimeIndex(self._split.sealed_dates)
            actual = pd.DatetimeIndex(ic.index)
            names_index = pd.DatetimeIndex(names.index)
            if not actual.equals(expected) or not names_index.equals(expected):
                raise ValueError(
                    f"{label} daily IC does not cover every sealed signal date"
                )
            numeric_ic = pd.to_numeric(ic, errors="coerce").to_numpy(dtype=float)
            numeric_names = pd.to_numeric(names, errors="coerce").to_numpy(dtype=float)
            if (
                not np.isfinite(numeric_ic).all()
                or not np.isfinite(numeric_names).all()
                or (numeric_names < self._min_names).any()
            ):
                raise ValueError(
                    f"{label} daily IC has a non-finite/under-covered sealed date"
                )

        for evaluated in champions:
            candidate = evaluated.candidate
            artifact = development_artifacts.get(candidate.candidate_id)
            if artifact is None:
                raise ValueError(
                    f"sealed candidate {candidate.candidate_id!r} has no development artifact"
                )
            if artifact.candidate_id != candidate.candidate_id:
                raise ValueError("sealed candidate/artifact id integrity mismatch")
            try:
                artifact_expression = json.loads(artifact.canonical_expression_json)
                stored_canonical_json = canonical_json(
                    artifact_expression,
                    self._spec,
                )
                stored_expression_hash = expression_hash(
                    artifact_expression,
                    self._spec,
                )
                candidate_canonical_json = canonical_json(
                    candidate.expression,
                    self._spec,
                )
                candidate_expression_hash = expression_hash(
                    candidate.expression,
                    self._spec,
                )
            except (DSLValidationError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("sealed expression integrity validation failed") from exc
            if (
                stored_canonical_json != artifact.canonical_expression_json
                or stored_expression_hash != artifact.expression_hash
            ):
                raise ValueError("sealed development artifact expression is corrupted")
            if (
                candidate_canonical_json != artifact.canonical_expression_json
                or candidate_expression_hash != artifact.expression_hash
            ):
                raise ValueError("sealed candidate expression integrity mismatch")
            values = evaluate_expression(
                artifact_expression,
                self._feature_frame,
                self._spec,
                cross_sectional_mask=self._cross_sectional_mask,
            ).reindex(self._panel.index) * artifact.direction
            neutral = cross_sectional_neutralize(values, exposures, categorical=("industry",), min_names=self._min_names)
            test_values = _date_slice(values, self._split.sealed_dates)
            test_neutral = _date_slice(neutral, self._split.sealed_dates)
            test_labels = _date_slice(
                labels.reindex(self._panel.index), self._split.sealed_dates
            )
            test_executable_labels = _date_slice(
                executable_labels.reindex(self._panel.index),
                self._split.sealed_dates,
            )
            raw_ic, raw_names = daily_ic(test_values, test_labels, method="spearman", min_names=self._min_names)
            neutral_ic, neutral_names = daily_ic(test_neutral, test_labels, method="spearman", min_names=self._min_names)
            executable_raw_ic, executable_raw_names = daily_ic(
                test_values,
                test_executable_labels,
                method="spearman",
                min_names=self._min_names,
            )
            executable_neutral_ic, executable_neutral_names = daily_ic(
                test_neutral,
                test_executable_labels,
                method="spearman",
                min_names=self._min_names,
            )
            for path_label, path_ic, path_names in (
                ("factor standard raw", raw_ic, raw_names),
                ("factor standard neutral", neutral_ic, neutral_names),
                (
                    "factor immediate-executable raw",
                    executable_raw_ic,
                    executable_raw_names,
                ),
                (
                    "factor immediate-executable neutral",
                    executable_neutral_ic,
                    executable_neutral_names,
                ),
            ):
                require_complete_ic_path(path_ic, path_names, label=path_label)
            raw = summarize_ic(raw_ic, raw_names, hac_lags=max(self._split.label_horizon - 1, 0), bootstrap_samples=500)
            neutral_result = summarize_ic(
                neutral_ic,
                neutral_names,
                hac_lags=max(self._split.label_horizon - 1, 0),
                bootstrap_samples=500,
            )
            executable_raw = summarize_ic(
                executable_raw_ic,
                executable_raw_names,
                hac_lags=max(self._split.label_horizon - 1, 0),
                bootstrap_samples=500,
                seed=47,
            )
            executable_neutral_result = summarize_ic(
                executable_neutral_ic,
                executable_neutral_names,
                hac_lags=max(self._split.label_horizon - 1, 0),
                bootstrap_samples=500,
                seed=53,
            )
            neutral_ic_family[candidate.candidate_id] = neutral_ic
            executable_neutral_ic_family[candidate.candidate_id] = (
                executable_neutral_ic
            )
            if sealed_neutral_features.empty:
                sealed_neutral_features = pd.DataFrame(index=test_neutral.index)
            sealed_neutral_features[candidate.candidate_id] = test_neutral
            output.append(
                {
                    "result_key": f"factor:{candidate.candidate_id}",
                    "candidate_id": candidate.candidate_id,
                    "expression_hash": artifact.expression_hash,
                    "agent_id": candidate.agent_id,
                    "operator": candidate.operator,
                    "reporting_role": (
                        "pre_registered_secondary_component_factor_diagnostic"
                    ),
                    "pre_registered_before_sealed_reveal": True,
                    "eligible_for_primary_claim": False,
                    "direction_frozen_in_train": artifact.direction,
                    "development_neutral_rank_ic": artifact.validation_neutral_rank_ic,
                    "forecast_label_definition": (
                        f"adjusted_{self._forecast_price_field}_exact_frozen_calendar_endpoint_return_without_future_fill_conditioning"
                    ),
                    "sealed_raw": raw.to_dict(),
                    "sealed_neutral": neutral_result.to_dict(),
                    "immediate_executable_label_definition": dict(
                        executable_label_definition
                    ),
                    "sealed_endpoint_availability": dict(endpoint_availability),
                    "sealed_forecast_label_audit": dict(
                        sealed_forecast_label_audit
                    ),
                    "research_universe_identity": dict(
                        self._research_universe_identity
                    ),
                    "trading_calendar_identity": dict(
                        self._trading_calendar_identity
                    ),
                    "sealed_immediate_executable_raw": executable_raw.to_dict(),
                    "sealed_immediate_executable_neutral": (
                        executable_neutral_result.to_dict()
                    ),
                    "immediate_executable_reporting_role": (
                        "pre_registered_secondary_execution_sensitivity_diagnostic"
                    ),
                    "immediate_executable_eligible_for_primary_claim": False,
                    "forecast_ic_dates_dropped": 0,
                    "immediate_executable_ic_dates_dropped": 0,
                }
            )
        composite_output: list[dict[str, object]] = []
        composite_neutral_ic_family: dict[str, pd.Series] = {}
        composite_executable_ic_family: dict[str, pd.Series] = {}
        composite_neutral_names: dict[str, pd.Series] = {}
        sealed_feature_audit: Mapping[str, object] | None = None
        if composite_family is not None:
            if not isinstance(composite_family_hash, str) or len(composite_family_hash) != 64:
                raise ValueError("sealed composite family hash is invalid")
            predictor = getattr(composite_family, "predict_all_for_sealed_gate", None)
            if predictor is None:
                raise ValueError("sealed composite family has no all-family interface")
            if self._sealed_claim_capability is None:
                raise ValueError(
                    "sealed composite reveal requires persistent-claim capability"
                )
            auditor = getattr(composite_family, "audit_sealed_feature_matrix", None)
            if auditor is None:
                raise ValueError("sealed composite family has no completeness audit")
            # The gate is already retired at method entry. Any coverage,
            # zero-imputation, or constant-column failure therefore fails
            # closed and cannot be retried with a modified family.
            sealed_feature_audit = auditor(sealed_neutral_features)
            predictions = predictor(
                sealed_neutral_features,
                expected_composite_family_hash=composite_family_hash,
                claim_capability=self._sealed_claim_capability,
            )
            if tuple(predictions.columns) != ("equal_weight", "ridge", "lightgbm"):
                raise ValueError("sealed composite prediction family is incomplete")
            test_labels = _date_slice(
                labels.reindex(self._panel.index), self._split.sealed_dates
            )
            test_executable_labels = _date_slice(
                executable_labels.reindex(self._panel.index),
                self._split.sealed_dates,
            )
            sealed_exposures = exposures.reindex(predictions.index)
            for backend in ("equal_weight", "ridge", "lightgbm"):
                raw_values = predictions[backend]
                neutral_values = cross_sectional_neutralize(
                    raw_values,
                    sealed_exposures,
                    categorical=("industry",),
                    min_names=self._min_names,
                )
                raw_ic, raw_names = daily_ic(
                    raw_values,
                    test_labels,
                    method="spearman",
                    min_names=self._min_names,
                )
                neutral_ic, neutral_names = daily_ic(
                    neutral_values,
                    test_labels,
                    method="spearman",
                    min_names=self._min_names,
                )
                executable_raw_ic, executable_raw_names = daily_ic(
                    raw_values,
                    test_executable_labels,
                    method="spearman",
                    min_names=self._min_names,
                )
                executable_neutral_ic, executable_neutral_names = daily_ic(
                    neutral_values,
                    test_executable_labels,
                    method="spearman",
                    min_names=self._min_names,
                )
                for path_label, path_ic, path_names in (
                    ("composite standard raw", raw_ic, raw_names),
                    ("composite standard neutral", neutral_ic, neutral_names),
                    (
                        "composite immediate-executable raw",
                        executable_raw_ic,
                        executable_raw_names,
                    ),
                    (
                        "composite immediate-executable neutral",
                        executable_neutral_ic,
                        executable_neutral_names,
                    ),
                ):
                    require_complete_ic_path(path_ic, path_names, label=path_label)
                key = f"composite:{backend}"
                composite_neutral_ic_family[key] = neutral_ic
                composite_executable_ic_family[key] = executable_neutral_ic
                composite_neutral_names[key] = neutral_names
                composite_output.append(
                    {
                        "result_key": f"composite:{backend}",
                        "backend": backend,
                        "primary_backend_pre_registered": backend == "lightgbm",
                        "reporting_role": (
                            "unique_pre_registered_primary_composite"
                            if backend == "lightgbm"
                            else "pre_registered_secondary_model_control"
                        ),
                        "pre_registered_before_sealed_reveal": True,
                        "eligible_for_primary_claim": backend == "lightgbm",
                        "composite_family_hash": composite_family_hash,
                        "sealed_raw": summarize_ic(
                            raw_ic,
                            raw_names,
                            hac_lags=max(self._split.label_horizon - 1, 0),
                            bootstrap_samples=500,
                            seed=61,
                        ).to_dict(),
                        "sealed_neutral": summarize_ic(
                            neutral_ic,
                            neutral_names,
                            hac_lags=max(self._split.label_horizon - 1, 0),
                            bootstrap_samples=500,
                            seed=67,
                        ).to_dict(),
                        "sealed_immediate_executable_raw": summarize_ic(
                            executable_raw_ic,
                            executable_raw_names,
                            hac_lags=max(self._split.label_horizon - 1, 0),
                            bootstrap_samples=500,
                            seed=71,
                        ).to_dict(),
                        "sealed_immediate_executable_neutral": summarize_ic(
                            executable_neutral_ic,
                            executable_neutral_names,
                            hac_lags=max(self._split.label_horizon - 1, 0),
                            bootstrap_samples=500,
                            seed=73,
                        ).to_dict(),
                        "immediate_executable_reporting_role": (
                            "pre_registered_secondary_execution_sensitivity_diagnostic"
                        ),
                        "immediate_executable_eligible_for_primary_claim": False,
                        "forecast_ic_dates_dropped": 0,
                        "immediate_executable_ic_dates_dropped": 0,
                    }
                )

        if output:
            joint_neutral = {
                **neutral_ic_family,
                **composite_neutral_ic_family,
            }
            family = pd.DataFrame(joint_neutral).sort_index()
            raw_p = np.asarray(
                [
                    one_sided_normal_pvalue(
                        float(
                            next(
                                record["sealed_neutral"]["hac_t"]
                                for record in [*output, *composite_output]
                                if str(record.get("candidate_id", f"composite:{record.get('backend', '')}"))
                                == key
                            )
                        )
                    )
                    for key in family.columns
                ],
                dtype=float,
            )
            holm_p = holm_adjusted_pvalues(raw_p)
            max_t_p = moving_block_max_t_pvalues(
                family,
                hac_lags=max(self._split.label_horizon - 1, 0),
                block_size=max(20, self._split.label_horizon * 2),
                samples=1_000,
                seed=43,
            )
            common_observations = int(family.dropna(axis=0, how="any").shape[0])
            if common_observations != len(self._split.sealed_dates):
                raise ValueError("joint family dropped one or more sealed IC dates")
            joint_positions = {key: position for position, key in enumerate(family.columns)}
            for record in output:
                candidate_id = str(record["candidate_id"])
                block_p = float(max_t_p.loc[candidate_id])
                position = joint_positions[candidate_id]
                record["multiple_testing"] = {
                    "family": "all_20_factors_plus_3_composites",
                    "family_size": len(family.columns),
                    "reporting_role": (
                        "conservative_fwer_control_not_a_selection_or_winner_pool"
                    ),
                    "selection_pool": False,
                    "common_daily_ic_observations": common_observations,
                    "hac_one_sided_p": float(raw_p[position]),
                    "holm_adjusted_p": float(holm_p[position]),
                    "block_max_t_adjusted_p": block_p,
                    "passes_fwer_5pct": block_p < 0.05,
                }

            executable_family = pd.DataFrame(
                {**executable_neutral_ic_family, **composite_executable_ic_family}
            ).sort_index()
            executable_raw_p = np.asarray(
                [
                    one_sided_normal_pvalue(
                        float(
                            next(
                                record["sealed_immediate_executable_neutral"]["hac_t"]
                                for record in [*output, *composite_output]
                                if str(record.get("candidate_id", f"composite:{record.get('backend', '')}"))
                                == key
                            )
                        )
                    )
                    for key in executable_family.columns
                ],
                dtype=float,
            )
            executable_holm_p = holm_adjusted_pvalues(executable_raw_p)
            executable_max_t_p = moving_block_max_t_pvalues(
                executable_family,
                hac_lags=max(self._split.label_horizon - 1, 0),
                block_size=max(20, self._split.label_horizon * 2),
                samples=1_000,
                seed=59,
            )
            executable_common_observations = int(
                executable_family.dropna(axis=0, how="any").shape[0]
            )
            if executable_common_observations != len(self._split.sealed_dates):
                raise ValueError(
                    "joint immediate-executable family dropped sealed IC dates"
                )
            executable_positions = {
                key: position for position, key in enumerate(executable_family.columns)
            }
            for record in output:
                candidate_id = str(record["candidate_id"])
                block_p = float(executable_max_t_p.loc[candidate_id])
                position = executable_positions[candidate_id]
                record["immediate_executable_multiple_testing"] = {
                    "family": "all_20_factors_plus_3_composites_immediate_executable",
                    "family_size": len(executable_family.columns),
                    "reporting_role": (
                        "pre_registered_secondary_execution_sensitivity_diagnostic"
                    ),
                    "eligible_for_primary_claim": False,
                    "selection_pool": False,
                    "common_daily_ic_observations": (
                        executable_common_observations
                    ),
                    "hac_one_sided_p": float(executable_raw_p[position]),
                    "holm_adjusted_p": float(executable_holm_p[position]),
                    "block_max_t_adjusted_p": block_p,
                    "passes_fwer_5pct": block_p < 0.05,
                }
            if composite_output:
                for record in composite_output:
                    key = f"composite:{record['backend']}"
                    position = joint_positions[key]
                    record["joint_multiple_testing"] = {
                        "family": "all_20_factors_plus_3_composites",
                        "family_size": len(family.columns),
                        "reporting_role": (
                            "conservative_fwer_control_not_a_selection_or_winner_pool"
                        ),
                        "selection_pool": False,
                        "hac_one_sided_p": float(raw_p[position]),
                        "holm_adjusted_p": float(holm_p[position]),
                        "block_max_t_adjusted_p": float(max_t_p.loc[key]),
                        "passes_fwer_5pct": float(max_t_p.loc[key]) < 0.05,
                    }
                    executable_position = executable_positions[key]
                    record["joint_immediate_executable_multiple_testing"] = {
                        "family": "all_20_factors_plus_3_composites_immediate_executable",
                        "family_size": len(executable_family.columns),
                        "reporting_role": (
                            "pre_registered_secondary_execution_sensitivity_diagnostic"
                        ),
                        "eligible_for_primary_claim": False,
                        "selection_pool": False,
                        "hac_one_sided_p": float(
                            executable_raw_p[executable_position]
                        ),
                        "holm_adjusted_p": float(
                            executable_holm_p[executable_position]
                        ),
                        "block_max_t_adjusted_p": float(
                            executable_max_t_p.loc[key]
                        ),
                        "passes_fwer_5pct": float(
                            executable_max_t_p.loc[key]
                        ) < 0.05,
                    }

                composite_frame = pd.DataFrame(composite_neutral_ic_family).sort_index()
                composite_p = np.asarray(
                    [
                        one_sided_normal_pvalue(
                            float(record["sealed_neutral"]["hac_t"])
                        )
                        for record in composite_output
                    ]
                )
                composite_holm = holm_adjusted_pvalues(composite_p)
                composite_max_t = moving_block_max_t_pvalues(
                    composite_frame,
                    hac_lags=max(self._split.label_horizon - 1, 0),
                    block_size=max(20, self._split.label_horizon * 2),
                    samples=1_000,
                    seed=79,
                )
                for position, record in enumerate(composite_output):
                    key = f"composite:{record['backend']}"
                    record["composite_subfamily_multiple_testing"] = {
                        "family": "three_pre_registered_composites",
                        "family_size": 3,
                        "reporting_role": (
                            "pre_registered_secondary_family_diagnostic"
                        ),
                        "eligible_for_primary_claim": False,
                        "selection_pool": False,
                        "hac_one_sided_p": float(composite_p[position]),
                        "holm_adjusted_p": float(composite_holm[position]),
                        "block_max_t_adjusted_p": float(composite_max_t.loc[key]),
                    }

                contrast_specs = (
                    ("lightgbm_minus_equal_weight", "lightgbm", "equal_weight"),
                    ("ridge_minus_equal_weight", "ridge", "equal_weight"),
                    ("lightgbm_minus_ridge", "lightgbm", "ridge"),
                )
                contrast_series: dict[str, pd.Series] = {}
                contrast_output: list[dict[str, object]] = []
                for name, left, right in contrast_specs:
                    difference = (
                        composite_neutral_ic_family[f"composite:{left}"]
                        - composite_neutral_ic_family[f"composite:{right}"]
                    ).dropna()
                    names = pd.concat(
                        [
                            composite_neutral_names[f"composite:{left}"],
                            composite_neutral_names[f"composite:{right}"],
                        ],
                        axis=1,
                    ).min(axis=1).reindex(difference.index)
                    summary = summarize_ic(
                        difference,
                        names,
                        hac_lags=max(self._split.label_horizon - 1, 0),
                        bootstrap_samples=500,
                        seed=83,
                    )
                    contrast_series[name] = difference
                    contrast_output.append(
                        {
                            "contrast": name,
                            "reporting_role": (
                                "pre_registered_secondary_model_comparison"
                            ),
                            "pre_registered_before_sealed_reveal": True,
                            "eligible_for_primary_claim": False,
                            "summary": summary.to_dict(),
                        }
                    )
                contrast_frame = pd.DataFrame(contrast_series).sort_index()
                contrast_p = np.asarray(
                    [
                        one_sided_normal_pvalue(float(item["summary"]["hac_t"]))
                        for item in contrast_output
                    ]
                )
                contrast_holm = holm_adjusted_pvalues(contrast_p)
                contrast_max_t = moving_block_max_t_pvalues(
                    contrast_frame,
                    hac_lags=max(self._split.label_horizon - 1, 0),
                    block_size=max(20, self._split.label_horizon * 2),
                    samples=1_000,
                    seed=89,
                )
                for position, item in enumerate(contrast_output):
                    name = str(item["contrast"])
                    item["multiple_testing"] = {
                        "family": "three_pre_registered_neutral_ic_contrasts",
                        "family_size": 3,
                        "reporting_role": (
                            "pre_registered_secondary_model_comparison"
                        ),
                        "eligible_for_primary_claim": False,
                        "selection_pool": False,
                        "hac_one_sided_p": float(contrast_p[position]),
                        "holm_adjusted_p": float(contrast_holm[position]),
                        "block_max_t_adjusted_p": float(contrast_max_t.loc[name]),
                    }
                reporting_contract = sealed_reporting_contract_v1()
                if len(family.columns) != int(
                    reporting_contract["joint_multiplicity_family_size"]
                ):
                    raise ValueError(
                        "sealed joint family differs from the pre-registered "
                        "reporting contract"
                    )
                primary_records = [
                    record
                    for record in composite_output
                    if record.get("eligible_for_primary_claim") is True
                ]
                if (
                    len(primary_records) != 1
                    or primary_records[0].get("result_key")
                    != reporting_contract["unique_primary_result_key"]
                    or primary_records[0].get("backend")
                    != reporting_contract["unique_primary_backend"]
                ):
                    raise ValueError(
                        "sealed composite output does not have exactly one "
                        "pre-registered primary"
                    )
                primary_record = primary_records[0]
                primary_binding = {
                    "result_key": primary_record["result_key"],
                    "backend": primary_record["backend"],
                    "endpoint_key": reporting_contract["primary_endpoint_key"],
                    "inference_key": reporting_contract["primary_inference_key"],
                    "result_record_hash": hash_json(primary_record),
                }
                self._composite_reveal = {
                    "primary_backend_pre_registered": "lightgbm",
                    "backend_selection_performed": False,
                    "sealed_result_ranking_or_winner_selection_performed": False,
                    "composite_family_hash": composite_family_hash,
                    "reporting_contract": reporting_contract,
                    "reporting_contract_hash": hash_json(reporting_contract),
                    "primary_result_binding": primary_binding,
                    "sealed_feature_completeness_audit": dict(
                        sealed_feature_audit or {}
                    ),
                    "sealed_forecast_label_audit": dict(
                        sealed_forecast_label_audit
                    ),
                    "research_universe_identity": dict(
                        self._research_universe_identity
                    ),
                    "trading_calendar_identity": dict(
                        self._trading_calendar_identity
                    ),
                    "sealed_composite_results": composite_output,
                    "pre_registered_neutral_ic_contrasts": contrast_output,
                    "joint_hypothesis_family_size": len(family.columns),
                }
        # Drop references after reveal so the object cannot become an accidental
        # reusable oracle in a long-running process.
        self._panel = pd.DataFrame()
        self._tape_panel = pd.DataFrame()
        self._feature_frame = pd.DataFrame()
        self._cross_sectional_mask = pd.Series(dtype=bool)
        return output
