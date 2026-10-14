"""Faithful-budget, mechanism-aware factor evolution.

The engine keeps the search topology of CogAlpha (population, parent pool,
mutation, crossover, periodic injection and elite carry) while making every
proposal deterministic and serializable. It does not inspect market data; an
external evaluator owns train/validation gates and returns blinded scores.
"""

from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import random
from typing import Callable, Iterable, Mapping, Sequence
from uuid import uuid4

from .dsl import canonical_json as dsl_canonical_json


Expression = dict[str, object]


def field_expr(name: str) -> Expression:
    return {"op": "field", "name": name}


def const_expr(value: float) -> Expression:
    return {"op": "const", "value": float(value)}


def unary(op: str, arg: Expression, **parameters: object) -> Expression:
    return {"op": op, "args": [deepcopy(arg)], **parameters}


def binary(op: str, left: Expression, right: Expression, **parameters: object) -> Expression:
    return {"op": op, "args": [deepcopy(left), deepcopy(right)], **parameters}


def _raw_canonical_expression(expression: Expression) -> str:
    return json.dumps(
        expression,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@lru_cache(maxsize=100_000)
def _dsl_canonical_from_raw(raw_expression: str) -> str:
    return dsl_canonical_json(json.loads(raw_expression))


def canonical_expression(expression: Expression) -> str:
    """Canonical DSL semantics, including commutative child ordering."""

    return _dsl_canonical_from_raw(_raw_canonical_expression(expression))


def expression_digest(expression: Expression) -> str:
    return hashlib.sha256(canonical_expression(expression).encode("utf-8")).hexdigest()


def expression_complexity(expression: Expression) -> int:
    total = 1
    for arg in expression.get("args", []):
        if isinstance(arg, dict):
            total += expression_complexity(arg)
    return total


@dataclass(frozen=True)
class SearchBudget:
    """Original CogAlpha search dimensions; use ``tiny`` only for tests."""

    initial_population: int = 80
    parent_pool: int = 32
    children_per_generation: int = 96
    generations: int = 24
    subcycles: int = 3
    generations_per_subcycle: int = 8
    injection_every: int = 2
    injection_count: int = 8
    elite_carry: int = 2
    final_factors: int = 20

    def __post_init__(self) -> None:
        positive = {
            "initial_population": self.initial_population,
            "parent_pool": self.parent_pool,
            "children_per_generation": self.children_per_generation,
            "generations": self.generations,
            "subcycles": self.subcycles,
            "generations_per_subcycle": self.generations_per_subcycle,
            "injection_every": self.injection_every,
            "elite_carry": self.elite_carry,
            "final_factors": self.final_factors,
        }
        if any(value < 1 for value in positive.values()) or self.injection_count < 0:
            raise ValueError("search budget values must be positive (injection_count may be zero)")
        if self.generations != self.subcycles * self.generations_per_subcycle:
            raise ValueError("generations must equal subcycles * generations_per_subcycle")
        if self.parent_pool > self.initial_population:
            raise ValueError("parent pool cannot exceed initial population")
        if self.elite_carry > self.parent_pool:
            raise ValueError("elite carry cannot exceed parent pool")

    @classmethod
    def tiny(cls) -> "SearchBudget":
        return cls(
            initial_population=8,
            parent_pool=4,
            children_per_generation=8,
            generations=4,
            subcycles=2,
            generations_per_subcycle=2,
            injection_every=2,
            injection_count=2,
            elite_carry=2,
            final_factors=3,
        )


@dataclass
class Candidate:
    expression: Expression
    agent_id: str
    hypothesis: str
    mechanism_tags: tuple[str, ...]
    generation: int = 0
    parents: tuple[str, ...] = ()
    operator: str = "initial"
    candidate_id: str = field(default_factory=lambda: uuid4().hex)
    control_expression: Expression | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def expression_hash(self) -> str:
        return expression_digest(self.expression)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "expression": self.expression,
            "expression_hash": self.expression_hash,
            "agent_id": self.agent_id,
            "hypothesis": self.hypothesis,
            "mechanism_tags": list(self.mechanism_tags),
            "generation": self.generation,
            "parents": list(self.parents),
            "operator": self.operator,
            "control_expression": self.control_expression,
            "metadata": self.metadata,
        }


def _normalized_role_lineage(candidate: Candidate) -> tuple[str, ...]:
    declared = candidate.metadata.get("role_lineage")
    if isinstance(declared, (list, tuple)) and all(
        isinstance(item, str) and item.strip() for item in declared
    ):
        return tuple(sorted(set(str(item).strip() for item in declared)))
    return tuple(
        sorted(
            {
                item.strip()
                for item in candidate.agent_id.split("+")
                if item.strip()
            }
        )
    )


def _identity_feedback(metadata: Mapping[str, object]) -> Mapping[str, object] | None:
    """Normalize the behavioral part of an adaptive directive.

    Floating evidence values and human-readable observation text are excluded:
    they explain why a directive was selected but do not change the already
    materialized child's behavior. The cause/fix/source/role/split instruction
    remains content-addressed.
    """

    value = metadata.get("adaptive_feedback")
    if not isinstance(value, Mapping):
        return None
    target_roles = value.get("target_roles", [])
    allowed_splits = value.get("allowed_splits", [])
    return {
        "schema": str(value.get("schema", "")),
        "source_candidate_id": value.get("source_candidate_id"),
        "target_roles": sorted(
            {
                str(item).strip()
                for item in target_roles
                if isinstance(item, str) and item.strip()
            }
        )
        if isinstance(target_roles, (list, tuple))
        else [],
        "cause": value.get("cause"),
        "fix": value.get("fix"),
        "allowed_splits": sorted(
            {
                str(item).strip()
                for item in allowed_splits
                if isinstance(item, str) and item.strip()
            }
        )
        if isinstance(allowed_splits, (list, tuple))
        else [],
        "sealed_feedback_consumed": bool(
            value.get("sealed_feedback_consumed", False)
        ),
    }


def deterministic_candidate_id(candidate: Candidate) -> str:
    """Canonical behavior/provenance identity for a generated candidate.

    IDs bind everything that can alter scoring, future parent selection or
    lineage interpretation, while deliberately excluding floating diagnostics,
    fitted metrics, hypotheses and other display prose.
    """

    metadata = candidate.metadata
    identity_metadata = {
        "expected_direction": metadata.get("expected_direction"),
        "forecast_horizon": metadata.get("forecast_horizon"),
        "role_lineage": list(_normalized_role_lineage(candidate)),
        "adaptive_feedback": _identity_feedback(metadata),
        "source_kind": metadata.get("source"),
        "candidate_key": metadata.get("candidate_key"),
        "source_factor_card_id": metadata.get("source_factor_card_id"),
        "source_agent_id": metadata.get("source_agent_id"),
        "feedback_fix_applied": metadata.get("feedback_fix_applied"),
        "mutation_class": metadata.get("mutation_class"),
        "crossover_rule": metadata.get("crossover_rule"),
        "parent_expected_directions": metadata.get(
            "parent_expected_directions"
        ),
        "two_stage_operator": bool(metadata.get("two_stage_operator", False)),
        "required_comparisons": sorted(
            str(item)
            for item in metadata.get("required_comparisons", [])
            if isinstance(item, str)
        )
        if isinstance(metadata.get("required_comparisons", []), (list, tuple))
        else [],
    }
    payload = {
        "identity_schema": "candidate_behavior_v2",
        "expression_hash": candidate.expression_hash,
        "control_expression_hash": (
            None
            if candidate.control_expression is None
            else expression_digest(candidate.control_expression)
        ),
        "agent_id": candidate.agent_id,
        "generation": candidate.generation,
        "parents": list(candidate.parents),
        "operator": candidate.operator,
        "mechanism_tags": sorted(set(candidate.mechanism_tags)),
        "metadata": identity_metadata,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:32]


@dataclass(frozen=True)
class CandidateScore:
    """Blinded validation response consumed by the evolution engine."""

    fitness: float
    signed_rank_ic: float
    neutral_rank_ic: float
    turnover: float
    worst_window_ic: float
    style_retention: float
    novelty: float = 1.0
    control_delta: float | None = None
    passed: bool = True
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        numeric = (
            self.fitness,
            self.signed_rank_ic,
            self.neutral_rank_ic,
            self.turnover,
            self.worst_window_ic,
            self.style_retention,
            self.novelty,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("candidate score contains a non-finite value")


@dataclass
class EvaluatedCandidate:
    candidate: Candidate
    score: CandidateScore


@dataclass(frozen=True)
class BehavioralChampionPolicy:
    """Pinned policy for the Plus final-family behavioral independence gate.

    The gate is deliberately separate from fitness and parent selection.  It
    is applied exactly once, after the complete development archive has been
    scored, and receives only inner-validation behavior from its external
    selector.  Faithful mode never consumes this Plus-only policy.
    """

    maximum_absolute_correlation: float = 0.85
    initial_top_pool_size: int = 200

    def __post_init__(self) -> None:
        threshold = self.maximum_absolute_correlation
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError(
                "maximum_absolute_correlation must be finite and in [0, 1]"
            )
        if (
            isinstance(self.initial_top_pool_size, bool)
            or not isinstance(self.initial_top_pool_size, int)
            or self.initial_top_pool_size < 1
        ):
            raise ValueError("initial_top_pool_size must be a positive integer")


@dataclass(frozen=True)
class ChampionSelectionResult:
    """Development-only champion family and its compact selection audit."""

    champions: tuple[EvaluatedCandidate, ...]
    audit: Mapping[str, object]


@dataclass(frozen=True)
class AdaptiveFeedbackPolicy:
    """Pinned development-only thresholds for the Plus feedback controller.

    These values are not tuned on the sealed period.  They translate a small,
    explicitly whitelisted set of train/inner-validation diagnostics into one
    auditable repair objective.  The controller never receives a panel, label,
    date, or sealed-gate object.
    """

    minimum_coverage: float = 0.80
    maximum_turnover: float = 0.45
    minimum_style_retention: float = 0.65
    minimum_worst_window_ic: float = 0.0
    minimum_novelty: float = 0.25
    minimum_control_delta: float = 0.0


@dataclass(frozen=True)
class FeedbackObservation:
    """Sanitized observation retained by the within-run failure memory."""

    candidate_id: str
    generation: int
    roles: tuple[str, ...]
    passed: bool
    cause: str
    preferred_fix: str
    evidence: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "roles": list(self.roles),
            "passed": self.passed,
            "cause": self.cause,
            "preferred_fix": self.preferred_fix,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class FeedbackDirective:
    """One deterministic observation/cause/fix instruction for an offspring."""

    source_candidate_id: str | None
    target_roles: tuple[str, ...]
    observation: str
    cause: str
    fix: str
    evidence: Mapping[str, object]
    allowed_splits: tuple[str, ...] = ("train", "inner_validation")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "adaptive_feedback_v1",
            "source_candidate_id": self.source_candidate_id,
            "target_roles": list(self.target_roles),
            "observation": self.observation,
            "cause": self.cause,
            "fix": self.fix,
            "evidence": dict(self.evidence),
            "allowed_splits": list(self.allowed_splits),
            "sealed_feedback_consumed": False,
        }


def candidate_roles(candidate: Candidate) -> tuple[str, ...]:
    """Return stable original-role lineage without trusting display strings."""

    return _normalized_role_lineage(candidate)


class AdaptiveFeedbackMemory:
    """Deterministic Plus controller over sanitized development scores.

    The evaluator remains the sole owner of train/inner-validation data.  This
    memory consumes only scalar ``CandidateScore`` fields plus three named
    diagnostics (``rejection``, ``coverage``, ``train_raw_rank_ic``).  It never
    iterates over the diagnostic mapping, which makes accidental propagation of
    sealed or other undeclared fields impossible.
    """

    _DIAGNOSTIC_ALLOWLIST = frozenset(
        {"rejection", "coverage", "train_raw_rank_ic"}
    )

    def __init__(self, policy: AdaptiveFeedbackPolicy | None = None) -> None:
        self.policy = policy or AdaptiveFeedbackPolicy()
        self._roles: set[str] = set()
        self._observations: list[FeedbackObservation] = []
        self._observed_ids: set[str] = set()
        self._cause_counts: Counter[str] = Counter()
        self._fix_counts: Counter[str] = Counter()
        self._role_observations: Counter[str] = Counter()
        self._role_failures: Counter[str] = Counter()
        self._role_injections: Counter[str] = Counter()

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(self._roles))

    @staticmethod
    def _finite_or_none(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def register_roles(self, candidates: Sequence[Candidate]) -> None:
        for candidate in candidates:
            self._roles.update(candidate_roles(candidate))

    def _sanitize(self, evaluated: EvaluatedCandidate) -> dict[str, object]:
        score = evaluated.score
        diagnostics = score.diagnostics
        # Exact-key access is intentional. Do not copy/iterate diagnostics.
        rejection = diagnostics.get("rejection")
        coverage = self._finite_or_none(diagnostics.get("coverage"))
        train_ic = self._finite_or_none(diagnostics.get("train_raw_rank_ic"))
        return {
            "passed": bool(score.passed),
            "rejection": str(rejection) if rejection is not None else None,
            "coverage": coverage,
            "train_raw_rank_ic": train_ic,
            "neutral_rank_ic": float(score.neutral_rank_ic),
            "turnover": float(score.turnover),
            "worst_window_ic": float(score.worst_window_ic),
            "style_retention": float(score.style_retention),
            "novelty": float(score.novelty),
            "control_delta": (
                None if score.control_delta is None else float(score.control_delta)
            ),
        }

    def _classify(self, evaluated: EvaluatedCandidate) -> FeedbackObservation:
        clean = self._sanitize(evaluated)
        rejection = clean["rejection"]
        coverage = clean["coverage"]
        train_ic = clean["train_raw_rank_ic"]
        neutral_ic = float(clean["neutral_rank_ic"])
        control_delta = clean["control_delta"]
        evidence: dict[str, object]

        if not bool(clean["passed"]):
            reason = str(rejection or "validation_gate_failed")
            if reason.startswith("coverage_below_gate"):
                cause, fix = "insufficient_coverage", "shorten_lookback"
            elif reason.startswith(("dsl_or_runtime", "price_scale", "truncation")):
                cause, fix = "invalid_or_fragile_structure", "conservative_repair"
            else:
                cause, fix = "validation_rejection", "conservative_repair"
            evidence = {"rejection": reason}
        elif coverage is not None and coverage < self.policy.minimum_coverage:
            cause, fix = "coverage_fragility", "shorten_lookback"
            evidence = {"coverage": coverage, "threshold": self.policy.minimum_coverage}
        elif (
            control_delta is not None
            and control_delta < self.policy.minimum_control_delta
        ):
            cause, fix = "no_increment_over_control", "nonlinear_refinement"
            evidence = {
                "control_delta": control_delta,
                "threshold": self.policy.minimum_control_delta,
            }
        elif float(clean["turnover"]) > self.policy.maximum_turnover:
            cause, fix = "excessive_turnover", "persistence_smoothing"
            evidence = {
                "turnover": clean["turnover"],
                "threshold": self.policy.maximum_turnover,
            }
        elif float(clean["style_retention"]) < self.policy.minimum_style_retention:
            cause, fix = "style_shortcut", "style_robustification"
            evidence = {
                "style_retention": clean["style_retention"],
                "threshold": self.policy.minimum_style_retention,
            }
        elif float(clean["worst_window_ic"]) < self.policy.minimum_worst_window_ic:
            cause, fix = "window_instability", "stability_smoothing"
            evidence = {
                "worst_window_ic": clean["worst_window_ic"],
                "threshold": self.policy.minimum_worst_window_ic,
            }
        elif float(clean["novelty"]) < self.policy.minimum_novelty:
            cause, fix = "behavioral_duplication", "novelty_recombination"
            evidence = {
                "novelty": clean["novelty"],
                "threshold": self.policy.minimum_novelty,
            }
        elif (
            train_ic is not None
            and train_ic != 0.0
            and neutral_ic != 0.0
            and train_ic * neutral_ic < 0.0
        ):
            cause, fix = "train_validation_sign_instability", "stability_smoothing"
            evidence = {
                "train_raw_rank_ic": train_ic,
                "neutral_rank_ic": neutral_ic,
            }
        elif neutral_ic <= 0.0:
            cause, fix = "weak_neutral_alpha", "parameter_refinement"
            evidence = {"neutral_rank_ic": neutral_ic}
        else:
            cause, fix = "effective_mechanism", "parameter_refinement"
            evidence = {
                "neutral_rank_ic": neutral_ic,
                "worst_window_ic": clean["worst_window_ic"],
            }
        candidate = evaluated.candidate
        return FeedbackObservation(
            candidate_id=candidate.candidate_id,
            generation=candidate.generation,
            roles=candidate_roles(candidate),
            passed=bool(clean["passed"]),
            cause=cause,
            preferred_fix=fix,
            evidence=evidence,
        )

    def observe(self, evaluated: Sequence[EvaluatedCandidate]) -> None:
        for item in evaluated:
            candidate_id = item.candidate.candidate_id
            if candidate_id in self._observed_ids:
                continue
            self._observed_ids.add(candidate_id)
            observation = self._classify(item)
            self._observations.append(observation)
            self._roles.update(observation.roles)
            self._cause_counts[observation.cause] += 1
            self._fix_counts[observation.preferred_fix] += 1
            for role in observation.roles:
                self._role_observations[role] += 1
                if not observation.passed:
                    self._role_failures[role] += 1
                if item.candidate.operator.startswith("injection:"):
                    self._role_injections[role] += 1

    @staticmethod
    def _fix_for_category(preferred: str, category: str) -> str:
        if category in {"crossover", "crossover_then_mutation"}:
            return {
                "shorten_lookback": "coverage_guarded_crossover",
                "conservative_repair": "failure_avoidance_crossover",
                "persistence_smoothing": "turnover_guarded_crossover",
                "style_robustification": "retention_guarded_crossover",
                "stability_smoothing": "stability_guarded_crossover",
                "novelty_recombination": "novelty_crossover",
                "nonlinear_refinement": "nonlinear_crossover",
                "parameter_refinement": "complementary_crossover",
            }[preferred]
        return preferred

    def _ranked_observations(self, category: str) -> list[FeedbackObservation]:
        # Failures and concrete weaknesses take precedence over already-effective
        # mechanisms. Stable tuple sorting makes the schedule reproducible.
        cause_priority = {
            "invalid_or_fragile_structure": 0,
            "validation_rejection": 1,
            "insufficient_coverage": 2,
            "coverage_fragility": 3,
            "no_increment_over_control": 4,
            "excessive_turnover": 5,
            "style_shortcut": 6,
            "window_instability": 7,
            "behavioral_duplication": 8,
            "train_validation_sign_instability": 9,
            "weak_neutral_alpha": 10,
            "effective_mechanism": 11,
        }
        observations = list(self._observations)
        if category in {"crossover", "crossover_then_mutation"}:
            crossover_causes = {
                "no_increment_over_control",
                "style_shortcut",
                "behavioral_duplication",
                "window_instability",
                "train_validation_sign_instability",
            }
            focused = [item for item in observations if item.cause in crossover_causes]
            if focused:
                observations = focused
        return sorted(
            observations,
            key=lambda item: (
                cause_priority.get(item.cause, 99),
                -item.generation,
                item.candidate_id,
            ),
        )

    def directive(
        self,
        *,
        category: str,
        generation: int,
        slot: int,
        parent_roles: Sequence[str],
        eligible_target_roles: Sequence[str] | None = None,
    ) -> FeedbackDirective:
        if category == "injection" and self._roles:
            present = set(parent_roles)
            schedule_roles = set(self._roles)
            if eligible_target_roles is not None:
                schedule_roles.intersection_update(eligible_target_roles)
                if not schedule_roles:
                    raise RuntimeError(
                        "no registered role has an unused precommitted injection card"
                    )
            ordered_roles = sorted(
                schedule_roles,
                key=lambda role: (
                    role in present,
                    self._role_injections[role],
                    self._role_observations[role],
                    -self._role_failures[role],
                    role,
                ),
            )
            role = ordered_roles[slot % len(ordered_roles)]
            role_observations = [
                item for item in self._ranked_observations(category) if role in item.roles
            ]
            source = role_observations[slot % len(role_observations)] if role_observations else None
            cause = "role_coverage_gap" if role not in present else (
                source.cause if source is not None else "fresh_mechanism_search"
            )
            preferred = source.preferred_fix if source is not None else "parameter_refinement"
            return FeedbackDirective(
                source_candidate_id=None if source is None else source.candidate_id,
                target_roles=(role,),
                observation=f"generation_{generation}_role_schedule",
                cause=cause,
                fix=preferred,
                evidence={
                    "role_present_in_parent_pool": role in present,
                    "remembered_role_failures": self._role_failures[role],
                    "prior_role_injections": self._role_injections[role],
                    "target_limited_to_unused_precommitted_role": (
                        eligible_target_roles is not None
                    ),
                },
            )

        ranked = self._ranked_observations(category)
        if not ranked:
            return FeedbackDirective(
                source_candidate_id=None,
                target_roles=(),
                observation=f"generation_{generation}_cold_start",
                cause="no_prior_feedback",
                fix=(
                    "complementary_crossover"
                    if category in {"crossover", "crossover_then_mutation"}
                    else "parameter_refinement"
                ),
                evidence={},
            )
        # Round-robin across distinct causes before cycling examples within a
        # cause. A large family of static failures therefore stays remembered
        # without monopolizing all coverage/style/stability/novelty repairs.
        causes = list(dict.fromkeys(item.cause for item in ranked))
        selected_cause = causes[slot % len(causes)]
        cause_examples = [item for item in ranked if item.cause == selected_cause]
        source = cause_examples[(slot // len(causes)) % len(cause_examples)]
        return FeedbackDirective(
            source_candidate_id=source.candidate_id,
            target_roles=source.roles,
            observation=f"remembered_{source.cause}",
            cause=source.cause,
            fix=self._fix_for_category(source.preferred_fix, category),
            evidence=source.evidence,
        )

    def summary(self) -> dict[str, object]:
        failed = sum(not item.passed for item in self._observations)
        return {
            "schema": "adaptive_feedback_memory_v1",
            "allowed_splits": ["train", "inner_validation"],
            "sealed_feedback_consumed": False,
            "diagnostic_allowlist": sorted(self._DIAGNOSTIC_ALLOWLIST),
            "registered_roles": list(self.roles),
            "registered_role_count": len(self._roles),
            "observations": len(self._observations),
            "remembered_failures": failed,
            "causes": dict(sorted(self._cause_counts.items())),
            "preferred_fixes": dict(sorted(self._fix_counts.items())),
            "role_failures": {
                role: self._role_failures[role] for role in self.roles
            },
            "role_injections": {
                role: self._role_injections[role] for role in self.roles
            },
        }


@dataclass(frozen=True)
class EvolutionResult:
    champions: tuple[EvaluatedCandidate, ...]
    archive: tuple[EvaluatedCandidate, ...]
    generations: tuple[Mapping[str, object], ...]
    adaptive_feedback: Mapping[str, object] = field(default_factory=dict)
    champion_selection: Mapping[str, object] = field(default_factory=dict)


class TemplateCandidateFactory:
    """Deterministic seed/injection source and a non-LLM control arm.

    Codex proposals can be loaded as Candidate objects instead. This factory is
    retained because an equal-budget symbolic baseline is necessary to measure
    whether model reasoning adds value.
    """

    WINDOWS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60, 90, 120, 180, 252)

    def __init__(self, seed: int = 17) -> None:
        self.rng = random.Random(seed)
        self._counter = 0
        self._emitted_hashes: set[str] = set()

    def _candidate(self, expression: Expression, hypothesis: str, tags: Sequence[str], agent_id: str) -> Candidate:
        self._counter += 1
        candidate = Candidate(
            expression=expression,
            agent_id=agent_id,
            hypothesis=hypothesis,
            mechanism_tags=tuple(tags),
            metadata={
                "source": "template_control",
                "sequence": self._counter,
                "role_lineage": [agent_id],
                # Every control template is written in its hypothesized Alpha
                # direction.  Evolution may not infer or flip this from labels.
                "expected_direction": 1,
            },
        )
        return candidate

    def generate(
        self,
        count: int,
        *,
        generation: int = 0,
        feedback: FeedbackDirective | None = None,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        attempts = 0
        while len(candidates) < count and attempts < max(count * 500, 2_000):
            attempts += 1
            short = self.rng.choice(self.WINDOWS[:8])
            long = self.rng.choice([w for w in self.WINDOWS if w > short])
            family = self.rng.randrange(7)
            variant = self.rng.randrange(6)
            close, volume, amount = field_expr("close"), field_expr("volume"), field_expr("amount")
            if family == 0:
                expr = unary("neg", unary("return", close, periods=short))
                c = self._candidate(expr, f"{short}日价格冲击均值回归", ("signal", "reversal", "short_scale"), "L4_price_reversal")
            elif family == 1:
                expr = binary(
                    "mul",
                    unary("return", close, periods=long),
                    binary("div", volume, unary("ts_mean", volume, window=long)),
                )
                c = self._candidate(expr, f"{long}日趋势由异常成交量确认", ("signal", "momentum", "confirmation", "volume"), "L3_volume_confirmation")
            elif family == 2:
                expr = binary("div", unary("return", close, periods=short), unary("ts_std", unary("return", close, periods=1), window=long))
                c = self._candidate(expr, "短期收益按实现波动缩放", ("signal", "risk_scale", "volatility"), "L2_vulnerability")
            elif family == 3:
                expr = binary("sub", unary("return", close, periods=short), unary("return", close, periods=long))
                c = self._candidate(expr, "短长尺度收益分歧", ("signal", "multi_scale", "divergence"), "L5_multiscale")
            elif family == 4:
                intraday = binary("div", binary("sub", close, field_expr("open")), field_expr("open"))
                expr = binary("mul", intraday, binary("div", amount, unary("ts_mean", amount, window=long)))
                c = self._candidate(expr, "日内收盘强度由成交金额确认", ("signal", "market_structure", "confirmation"), "L1_market_structure")
            elif family == 5:
                range_ = binary("div", binary("sub", field_expr("high"), field_expr("low")), close)
                expr = binary("div", unary("neg", unary("return", close, periods=short)), binary("add", range_, const_expr(1e-4)))
                c = self._candidate(expr, "按价格脆弱性缩放反转冲击", ("signal", "risk_scale", "range"), "L6_stability")
            else:
                location = binary(
                    "div",
                    binary("sub", close, field_expr("low")),
                    binary("add", binary("sub", field_expr("high"), field_expr("low")), const_expr(1e-4)),
                )
                expr = binary("mul", unary("return", close, periods=short), binary("sub", location, const_expr(0.5)))
                c = self._candidate(expr, "K线几何位置与短期方向耦合", ("signal", "geometry", "fusion"), "L7_geometry")

            # These are economically interpretable representations of the same
            # base hypothesis, not cosmetic scalar rewrites: cross-sectional
            # robustness, short persistence, slow-baseline surprise, or local
            # volatility normalization. They also form the equal-budget
            # non-LLM control arm used to measure Codex's incremental value.
            base = c.expression
            if variant == 1:
                c.expression = unary("cs_rank", base)
                c.hypothesis += "；采用截面分位数抑制极值"
                c.mechanism_tags += ("robust_rank",)
            elif variant == 2:
                c.expression = unary("cs_zscore", base)
                c.hypothesis += "；采用截面标准化比较强度"
                c.mechanism_tags += ("cross_sectional_scale",)
            elif variant == 3:
                c.expression = unary("ts_mean", base, window=self.rng.choice((2, 3, 4, 5)))
                c.hypothesis += "；要求信号短期持续"
                c.mechanism_tags += ("persistence",)
            elif variant == 4:
                c.expression = binary("sub", base, unary("ts_mean", base, window=long))
                c.hypothesis += f"；度量相对{long}日基线的意外变化"
                c.mechanism_tags += ("slow_baseline_surprise",)
            elif variant == 5:
                scale = binary("add", unary("ts_std", base, window=long), const_expr(1e-4))
                c.expression = binary("div", base, scale)
                c.hypothesis += f"；按{long}日自身不稳定性缩放"
                c.mechanism_tags += ("self_risk_scale",)
            c.generation = generation
            if feedback is not None:
                c.metadata["adaptive_feedback"] = feedback.to_dict()
            c.candidate_id = deterministic_candidate_id(c)
            if c.expression_hash not in self._emitted_hashes:
                candidates.append(c)
                self._emitted_hashes.add(c.expression_hash)
        if len(candidates) != count:
            raise RuntimeError(f"could only produce {len(candidates)} unique templates for requested {count}")
        return candidates


class MechanismAwareOperators:
    """Mutation and crossover that preserve an economic role for each parent."""

    WINDOWS = (2, 3, 5, 10, 20, 40, 60, 120)

    def __init__(self, seed: int = 17, *, max_nodes: int = 48) -> None:
        self.rng = random.Random(seed)
        if max_nodes < 16:
            raise ValueError("max_nodes must leave room for two-parent crossover")
        self.max_nodes = max_nodes

    def linear_control(self, left: Candidate, right: Candidate) -> Expression:
        # Five fixed nodes form 0.5 * (z(left) + z(right)); reserve one
        # orientation node per branch and spend the remaining capacity evenly.
        # This uses the same typed parent views as the nonlinear child, so the
        # control is a real two-parent comparison rather than a more severely
        # truncated surrogate.
        branch_budget = max(6, (self.max_nodes - 8) // 2)
        left_expression = _compact_expression(left.expression, branch_budget)
        right_expression = _compact_expression(right.expression, branch_budget)
        left_direction = left.metadata.get("expected_direction")
        right_direction = right.metadata.get("expected_direction")
        if left_direction not in {-1, 1} or right_direction not in {-1, 1}:
            raise ValueError("linear control parents require predeclared expected_direction")
        if int(left_direction) == -1:
            left_expression = unary("neg", left_expression)
        if int(right_direction) == -1:
            right_expression = unary("neg", right_expression)
        return binary(
            "mul",
            binary("add", unary("cs_zscore", left_expression), unary("cs_zscore", right_expression)),
            const_expr(0.5),
        )

    def mutate(
        self,
        parent: Candidate,
        generation: int,
        *,
        mutation_class: str = "parameter",
        feedback_fix: str | None = None,
    ) -> Candidate:
        if mutation_class not in {"parameter", "semantic"}:
            raise ValueError("mutation_class must be parameter or semantic")
        # Repeated evolution must not create exponentially growing expression
        # trees. Preserve the parent mechanism while fitting it into the DSL
        # node budget before adding a mutation wrapper.
        expression = _compact_expression(parent.expression, self.max_nodes - 10)
        options = (
            ["timescale", "robust_rank"]
            if mutation_class == "parameter"
            else ["risk_scale", "persistence", "slow_baseline_surprise"]
        )
        targeted_modes = {
            "shorten_lookback": "shorten_lookback",
            "conservative_repair": "conservative_repair",
            "persistence_smoothing": "persistence",
            "style_robustification": "robust_rank",
            "stability_smoothing": "persistence",
            "novelty_recombination": "slow_baseline_surprise",
            "nonlinear_refinement": "risk_scale",
            "parameter_refinement": "timescale",
        }
        if feedback_fix is not None and feedback_fix not in targeted_modes:
            raise ValueError(f"unsupported mutation feedback fix: {feedback_fix}")
        mode = (
            targeted_modes[feedback_fix]
            if feedback_fix is not None
            else self.rng.choice(options)
        )
        if mode == "timescale":
            nodes = [node for node in _walk_nodes(expression) if "window" in node or "periods" in node]
            if nodes:
                node = self.rng.choice(nodes)
                key = "window" if "window" in node else "periods"
                current = int(node[key])
                choices = sorted(self.WINDOWS, key=lambda value: abs(value - current))[:4]
                node[key] = self.rng.choice([value for value in choices if value != current] or list(self.WINDOWS))
            else:
                expression = _insert_time_series_before_cross_section(
                    expression,
                    "ts_mean",
                    window=self.rng.choice(self.WINDOWS[2:]),
                )
        elif mode == "shorten_lookback":
            nodes = [
                node
                for node in _walk_nodes(expression)
                if "window" in node or "periods" in node
            ]
            if nodes:
                node = self.rng.choice(nodes)
                key = "window" if "window" in node else "periods"
                current = int(node[key])
                shorter = [value for value in self.WINDOWS if value < current]
                node[key] = max(shorter) if shorter else max(1, current - 1)
            else:
                expression = unary("cs_rank", expression)
        elif mode == "conservative_repair":
            expression = _repair_time_series_stage(
                _compact_expression(expression, self.max_nodes - 2)
            )
            nodes = [
                node
                for node in _walk_nodes(expression)
                if "window" in node or "periods" in node
            ]
            if nodes:
                node = self.rng.choice(nodes)
                key = "window" if "window" in node else "periods"
                current = int(node[key])
                node[key] = self.rng.choice(
                    [value for value in self.WINDOWS if value != current]
                )
            expression = unary(self.rng.choice(("cs_rank", "cs_zscore")), expression)
        elif mode == "robust_rank":
            if feedback_fix is not None:
                nodes = [
                    node
                    for node in _walk_nodes(expression)
                    if "window" in node or "periods" in node
                ]
                if nodes:
                    node = self.rng.choice(nodes)
                    key = "window" if "window" in node else "periods"
                    current = int(node[key])
                    choices = [value for value in self.WINDOWS if value != current]
                    node[key] = self.rng.choice(choices)
            expression = unary(
                self.rng.choice(("cs_rank", "cs_zscore"))
                if feedback_fix is not None
                else "cs_rank",
                expression,
            )
        elif mode == "risk_scale":
            risk = binary(
                "add",
                unary("ts_std", unary("return", field_expr("close"), periods=1), window=self.rng.choice(self.WINDOWS[3:])),
                const_expr(1e-4),
            )
            expression = binary("div", expression, risk)
        elif mode == "persistence":
            expression = _insert_time_series_before_cross_section(
                expression,
                "ts_mean",
                window=self.rng.choice((2, 3, 4, 5)),
            )
        else:
            compact = _compact_expression(expression, max(6, (self.max_nodes - 3) // 2))
            slow = _insert_time_series_before_cross_section(
                compact,
                "ts_mean",
                window=self.rng.choice(self.WINDOWS[3:]),
            )
            expression = binary("sub", compact, slow)
        mutation_label = f"feedback_{mode}" if feedback_fix is not None else mode
        candidate = Candidate(
            expression=expression,
            agent_id=parent.agent_id,
            hypothesis=f"{parent.hypothesis}；定向突变={mutation_label}",
            mechanism_tags=parent.mechanism_tags,
            generation=generation,
            parents=(parent.candidate_id,),
            operator=f"mutation:{mutation_class}:{mutation_label}",
            metadata={
                **deepcopy(parent.metadata),
                "mutation_preserves_parent_mechanism": True,
                "mutation_class": mutation_class,
                "role_lineage": list(candidate_roles(parent)),
                "feedback_fix_applied": feedback_fix,
            },
        )
        candidate.candidate_id = deterministic_candidate_id(candidate)
        return candidate

    def crossover(
        self,
        left: Candidate,
        right: Candidate,
        generation: int,
        *,
        feedback_fix: str | None = None,
    ) -> Candidate:
        left_direction = left.metadata.get("expected_direction")
        right_direction = right.metadata.get("expected_direction")
        if left_direction not in {-1, 1} or right_direction not in {-1, 1}:
            raise ValueError("crossover parents require predeclared expected_direction")
        left_tags, right_tags = set(left.mechanism_tags), set(right.mechanism_tags)
        # The largest crossover shell needs five nodes, and the two declared
        # directions can add at most two ``neg`` nodes.  Keep one node of slack
        # and give both mechanisms equal typed budgets.
        branch_budget = max(6, (self.max_nodes - 8) // 2)
        left_expression = _compact_expression(left.expression, branch_budget)
        right_expression = _compact_expression(right.expression, branch_budget)
        if int(left_direction) == -1:
            left_expression = unary("neg", left_expression)
        if int(right_direction) == -1:
            right_expression = unary("neg", right_expression)
        control = self.linear_control(left, right)
        if feedback_fix in {
            "nonlinear_crossover",
            "novelty_crossover",
        }:
            expression = binary(
                "mul",
                unary("cs_zscore", left_expression),
                unary("cs_zscore", right_expression),
            )
            mode = feedback_fix
            hypothesis = (
                f"以非线性交互检验{left.hypothesis}与{right.hypothesis}，"
                "并与等权线性控制比较"
            )
        elif feedback_fix in {
            "retention_guarded_crossover",
            "stability_guarded_crossover",
            "turnover_guarded_crossover",
        }:
            gate = binary(
                "add",
                const_expr(0.25),
                unary("cs_rank", unary("abs", right_expression)),
            )
            expression = binary("mul", left_expression, gate)
            mode = feedback_fix
            hypothesis = f"以{right.hypothesis}作为{feedback_fix}门控{left.hypothesis}"
        elif "risk_scale" in right_tags or "volatility" in right_tags or "range" in right_tags:
            expression = binary("div", left_expression, binary("add", unary("abs", right_expression), const_expr(1e-4)))
            mode = "signal_over_risk"
            hypothesis = f"用{right.hypothesis}刻画风险尺度，缩放{left.hypothesis}"
        elif "risk_scale" in left_tags or "volatility" in left_tags or "range" in left_tags:
            expression = binary("div", right_expression, binary("add", unary("abs", left_expression), const_expr(1e-4)))
            mode = "signal_over_risk"
            hypothesis = f"用{left.hypothesis}刻画风险尺度，缩放{right.hypothesis}"
        elif "confirmation" in right_tags or "volume" in right_tags:
            gate = binary("add", const_expr(0.25), unary("cs_rank", unary("abs", right_expression)))
            expression = binary("mul", left_expression, gate)
            mode = "signal_times_confirmation"
            hypothesis = f"仅在{right.hypothesis}较强时放大{left.hypothesis}"
        elif "confirmation" in left_tags or "volume" in left_tags:
            gate = binary("add", const_expr(0.25), unary("cs_rank", unary("abs", left_expression)))
            expression = binary("mul", right_expression, gate)
            mode = "signal_times_confirmation"
            hypothesis = f"仅在{left.hypothesis}较强时放大{right.hypothesis}"
        elif "multi_scale" in left_tags | right_tags or _different_scales(left_tags, right_tags):
            expression = binary("sub", unary("cs_zscore", left_expression), unary("cs_zscore", right_expression))
            mode = "cross_scale_divergence"
            hypothesis = f"捕捉{left.hypothesis}与{right.hypothesis}的尺度分歧"
        else:
            # A centered interaction has a distinct hypothesis; the uncentered
            # average is retained as the explicit control expression.
            expression = binary("mul", unary("cs_zscore", left_expression), unary("cs_zscore", right_expression))
            mode = "orthogonal_interaction"
            hypothesis = f"检验{left.hypothesis}与{right.hypothesis}的非线性交互"
        tags = tuple(sorted(left_tags | right_tags | {"crossover"}))
        roles = tuple(sorted(set(candidate_roles(left)) | set(candidate_roles(right))))
        embedded_branch_hashes = [
            expression_digest(left_expression),
            expression_digest(right_expression),
        ]
        if not all(
            _contains_expression_digest(expression, digest)
            for digest in embedded_branch_hashes
        ):
            raise RuntimeError("crossover shell failed to retain both typed parent branches")
        if not all(
            _contains_expression_digest(control, digest)
            for digest in embedded_branch_hashes
        ):
            raise RuntimeError("linear control failed to retain both typed parent branches")
        candidate = Candidate(
            expression=expression,
            agent_id="+".join(roles),
            hypothesis=hypothesis,
            mechanism_tags=tags,
            generation=generation,
            parents=(left.candidate_id, right.candidate_id),
            operator=f"crossover:{mode}",
            control_expression=control,
            metadata={
                "crossover_rule": mode,
                "required_comparisons": ["left_parent", "right_parent", "equal_weight_linear_control"],
                # Both branches are explicitly oriented before combination;
                # the resulting hypothesis direction is fixed before scoring.
                "expected_direction": 1,
                "parent_expected_directions": [int(left_direction), int(right_direction)],
                "role_lineage": list(roles),
                "feedback_fix_applied": feedback_fix,
                "parent_expression_hashes": [
                    left.expression_hash,
                    right.expression_hash,
                ],
                "embedded_parent_branch_hashes": embedded_branch_hashes,
                "parent_branch_nodes": [
                    expression_complexity(left_expression),
                    expression_complexity(right_expression),
                ],
                "linear_control_expression_hash": expression_digest(control),
                "typed_semantic_compaction": True,
            },
        )
        candidate.candidate_id = deterministic_candidate_id(candidate)
        return candidate

    def crossover_then_mutate(
        self,
        left: Candidate,
        right: Candidate,
        generation: int,
        *,
        feedback_fix: str | None = None,
    ) -> Candidate:
        crossed = self.crossover(
            left,
            right,
            generation,
            feedback_fix=feedback_fix,
        )
        mutated = self.mutate(crossed, generation, mutation_class="semantic")
        mutated.parents = (left.candidate_id, right.candidate_id)
        mutated.agent_id = crossed.agent_id
        mutated.hypothesis = f"{crossed.hypothesis}；交叉后语义突变"
        mutated.mechanism_tags = tuple(sorted(set(crossed.mechanism_tags) | {"crossover_then_mutation"}))
        mutated.operator = f"crossover_mutation:{crossed.operator.removeprefix('crossover:')}+{mutated.operator.removeprefix('mutation:')}"
        mutated.control_expression = crossed.control_expression
        mutated.metadata = {
            **crossed.metadata,
            **mutated.metadata,
            "two_stage_operator": True,
            "feedback_fix_applied": feedback_fix,
        }
        mutated.candidate_id = deterministic_candidate_id(mutated)
        return mutated


def _walk_nodes(expression: Expression) -> list[Expression]:
    nodes = [expression]
    for arg in expression.get("args", []):
        if isinstance(arg, dict):
            nodes.extend(_walk_nodes(arg))
    return nodes


_CROSS_SECTIONAL_OPERATORS = frozenset({"cs_rank", "cs_zscore"})
_TIME_SERIES_OPERATORS = frozenset(
    {
        "lag",
        "delta",
        "return",
        "ts_mean",
        "ts_std",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_corr",
    }
)
_PRICE_SCALE_FIELDS = frozenset({"open", "high", "low", "close", "vwap"})
_SCALE_PRESERVING_UNARY = frozenset(
    {"abs", "neg", "lag", "delta", "ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max"}
)
_COMPACTION_CACHE: dict[tuple[str, int], Expression] = {}
_COMPACTION_CACHE_LIMIT = 20_000


@dataclass(frozen=True)
class _SemanticType:
    """Production types needed by anti-bloat compaction.

    ``price_scale_degree`` is the exponent induced by multiplying all adjusted
    price fields of one security by the same positive constant.  The CS flag
    is also part of the type because placing a time-series operator above such
    a descendant is forbidden by the production DSL.
    """

    price_scale_degree: Fraction
    contains_cross_sectional: bool


def _contains_cross_sectional(expression: Expression) -> bool:
    return any(
        str(node.get("op", "")) in _CROSS_SECTIONAL_OPERATORS
        for node in _walk_nodes(expression)
    )


def _semantic_type(expression: Expression) -> _SemanticType | None:
    """Return a fail-closed local production type for a DSL expression."""

    op = str(expression.get("op", ""))
    if op == "field":
        degree = Fraction(1 if expression.get("name") in _PRICE_SCALE_FIELDS else 0)
        return _SemanticType(degree, False)
    if op == "const":
        return _SemanticType(Fraction(0), False)
    args = [arg for arg in expression.get("args", []) if isinstance(arg, dict)]
    child_types = [_semantic_type(arg) for arg in args]
    if not args or any(item is None for item in child_types):
        return None
    resolved = [item for item in child_types if item is not None]
    contains_cross_sectional = op in _CROSS_SECTIONAL_OPERATORS or any(
        item.contains_cross_sectional for item in resolved
    )
    if op in _TIME_SERIES_OPERATORS and any(
        item.contains_cross_sectional for item in resolved
    ):
        return None
    degrees = [item.price_scale_degree for item in resolved]
    if op in _SCALE_PRESERVING_UNARY:
        return _SemanticType(degrees[0], contains_cross_sectional)
    if op == "sqrt":
        return _SemanticType(degrees[0] / 2, contains_cross_sectional)
    if op == "log1p":
        return (
            _SemanticType(Fraction(0), contains_cross_sectional)
            if degrees[0] == 0
            else None
        )
    if op in {"add", "sub"}:
        return (
            _SemanticType(degrees[0], contains_cross_sectional)
            if len(degrees) == 2 and degrees[0] == degrees[1]
            else None
        )
    if op == "mul":
        return (
            _SemanticType(degrees[0] + degrees[1], contains_cross_sectional)
            if len(degrees) == 2
            else None
        )
    if op == "div":
        return (
            _SemanticType(degrees[0] - degrees[1], contains_cross_sectional)
            if len(degrees) == 2
            else None
        )
    if op == "return":
        return _SemanticType(Fraction(0), contains_cross_sectional)
    if op == "ts_corr":
        return (
            _SemanticType(Fraction(0), contains_cross_sectional)
            if len(degrees) == 2
            else None
        )
    if op in _CROSS_SECTIONAL_OPERATORS:
        return (
            _SemanticType(Fraction(0), True)
            if degrees[0] == 0
            else None
        )
    return None


def _contains_field(expression: Expression) -> bool:
    return any(str(node.get("op", "")) == "field" for node in _walk_nodes(expression))


def _contains_expression_digest(expression: Expression, digest: str) -> bool:
    return any(expression_digest(node) == digest for node in _walk_nodes(expression))


def _repair_time_series_stage(expression: Expression) -> Expression:
    """Drop only illegal TS-over-CS wrappers while preserving their child.

    Production factors forbid a time-series operator from consuming an already
    cross-sectionally transformed series. This conservative structural repair
    is deterministic and data independent.
    """

    result = deepcopy(expression)
    args = [arg for arg in result.get("args", []) if isinstance(arg, dict)]
    if args:
        result["args"] = [_repair_time_series_stage(arg) for arg in args]
    if str(result.get("op", "")) in _TIME_SERIES_OPERATORS and any(
        _contains_cross_sectional(arg) for arg in result.get("args", [])
    ):
        # Unary TS nodes have one child. For a defensive legacy ``ts_corr``
        # repair choose the first informative, non-CS branch; if both are CS,
        # descend into the first branch deterministically.
        safe = next(
            (
                arg
                for arg in result["args"]
                if not _contains_cross_sectional(arg) and _contains_field(arg)
            ),
            result["args"][0],
        )
        return deepcopy(safe)
    return result


def _insert_time_series_before_cross_section(
    expression: Expression,
    op: str,
    **parameters: object,
) -> Expression:
    """Apply a TS repair without ever wrapping a CS descendant."""

    clean = _repair_time_series_stage(expression)
    if not _contains_cross_sectional(clean):
        return unary(op, clean, **parameters)
    args = [arg for arg in clean.get("args", []) if isinstance(arg, dict)]
    if not args:
        return clean
    result = deepcopy(clean)
    transformed = _insert_time_series_before_cross_section(
        args[0], op, **parameters
    )
    result_args = [deepcopy(arg) for arg in args]
    result_args[0] = transformed
    result["args"] = result_args
    return result


def _compact_expression(expression: Expression, max_nodes: int) -> Expression:
    """Fit an expression into ``max_nodes`` without corrupting its type.

    The old equal-slice pruning could turn a relative price floor such as
    ``eps * vwap`` into a scalar constant.  Its parent then combined scale-one
    and scale-zero values and failed the production invariance proof.  This
    routine treats price homogeneity and CS/TS stage as static types.  It keeps
    an operator only when all compacted children reproduce the original node's
    type, searches binary budget splits deterministically, and falls back only
    to an informative descendant of the same type.

    No labels or market observations enter this process.  Crossover compacts
    the two parents independently and the outer operator still embeds both
    resulting mechanisms.
    """

    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    clean = _repair_time_series_stage(expression)
    # Compaction is type-checked below; its memoization key and deterministic
    # tie-break do not define candidate identity and can use the cheaper exact
    # JSON form. Candidate/seen/control hashes always use the DSL canonical form.
    cache_key = (_raw_canonical_expression(clean), max_nodes)
    cached_compaction = _COMPACTION_CACHE.get(cache_key)
    if cached_compaction is not None:
        return deepcopy(cached_compaction)
    original_type = _semantic_type(clean)

    def legacy_structural(node: Expression, budget: int) -> Expression:
        """Keep the explicitly labelled non-production control arm diverse."""

        if expression_complexity(node) <= budget:
            return deepcopy(node)
        args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
        if not args:
            return deepcopy(node)
        remaining = budget - 1
        if remaining < len(args):
            return legacy_structural(args[0], budget)
        per_arg = max(1, remaining // len(args))
        result: Expression = {
            "op": str(node.get("op", "")),
            "args": [legacy_structural(arg, per_arg) for arg in args],
        }
        for parameter in ("window", "periods"):
            if parameter in node:
                result[parameter] = node[parameter]
        return result

    # ``TemplateCandidateFactory`` is a deliberately labelled generic control
    # arm and predates the production price-type contract.  Preserve its old
    # structural diversity for comparison; all frozen Factor Cards and every
    # descendant in the real run enter the typed path below.
    if original_type is None:
        compacted_control = legacy_structural(clean, max_nodes)
        if len(_COMPACTION_CACHE) >= _COMPACTION_CACHE_LIMIT:
            _COMPACTION_CACHE.clear()
        _COMPACTION_CACHE[cache_key] = deepcopy(compacted_control)
        return compacted_control

    source_token_cache: dict[int, Counter[tuple[str, str, str]]] = {}

    def tokens(node: Expression) -> Counter[tuple[str, str, str]]:
        cached = source_token_cache.get(id(node))
        if cached is not None:
            return cached
        value = Counter(
            (
                str(item.get("op", "")),
                str(item.get("name", "")),
                str(item.get("window", item.get("periods", ""))),
            )
            for item in _walk_nodes(node)
        )
        source_token_cache[id(node)] = value
        return value

    def quality(source: Expression, candidate: Expression) -> tuple[int, int, int, str]:
        source_tokens = tokens(source)
        candidate_tokens = Counter(
            (
                str(item.get("op", "")),
                str(item.get("name", "")),
                str(item.get("window", item.get("periods", ""))),
            )
            for item in _walk_nodes(candidate)
        )
        overlap = sum((source_tokens & candidate_tokens).values())
        fields = sum(
            1 for node in _walk_nodes(candidate) if str(node.get("op", "")) == "field"
        )
        # The final canonical string makes ties reproducible across processes;
        # callers maximize the first three entries and minimize that string.
        return (
            overlap,
            fields,
            expression_complexity(candidate),
            _raw_canonical_expression(candidate),
        )

    def quality_sort_key(source: Expression, candidate: Expression) -> tuple[int, int, int, str]:
        overlap, fields, complexity, canonical = quality(source, candidate)
        return -overlap, -fields, -complexity, canonical

    memo: dict[tuple[int, int], Expression | None] = {}

    def best(node: Expression, budget: int) -> Expression | None:
        key = (id(node), budget)
        if key in memo:
            return memo[key]
        target = _semantic_type(node)
        if target is None or budget < 1:
            memo[key] = None
            return None
        if expression_complexity(node) <= budget:
            result = deepcopy(node)
            memo[key] = result
            return result

        op = str(node.get("op", ""))
        args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
        candidates: list[Expression] = []
        if len(args) == 1 and budget >= 2:
            child = best(args[0], budget - 1)
            if child is not None:
                candidate: Expression = {"op": op, "args": [child]}
                for parameter in ("window", "periods"):
                    if parameter in node:
                        candidate[parameter] = node[parameter]
                if _semantic_type(candidate) == target:
                    candidates.append(candidate)
        elif len(args) == 2 and budget >= 3:
            for left_budget in range(1, budget - 1):
                right_budget = budget - 1 - left_budget
                left = best(args[0], left_budget)
                right = best(args[1], right_budget)
                if left is None or right is None:
                    continue
                candidate = {"op": op, "args": [left, right]}
                if "window" in node:
                    candidate["window"] = node["window"]
                if _semantic_type(candidate) == target:
                    candidates.append(candidate)

        # Root preservation is the first priority.  Only if the operator
        # cannot be represented in the budget may compaction select a typed,
        # informative descendant.
        if not candidates:
            for descendant in args:
                compacted = best(descendant, budget)
                if (
                    compacted is not None
                    and _semantic_type(compacted) == target
                    and _contains_field(compacted)
                ):
                    candidates.append(compacted)
        if not candidates:
            memo[key] = None
            return None
        result = sorted(
            candidates,
            key=lambda candidate: quality_sort_key(node, candidate),
        )[0]
        memo[key] = deepcopy(result)
        return memo[key]

    compacted = best(clean, max_nodes)
    if compacted is None and original_type is not None:
        # Search every descendant before using the universal price-return
        # witness.  Constants are intentionally excluded: a constant branch
        # would technically type-check but erase the parent's mechanism.
        typed_descendants = [
            node
            for node in _walk_nodes(clean)
            if _semantic_type(node) == original_type and _contains_field(node)
        ]
        options = [best(node, max_nodes) for node in typed_descendants]
        valid_options = [item for item in options if item is not None]
        if valid_options:
            compacted = sorted(
                valid_options,
                key=lambda candidate: quality_sort_key(clean, candidate),
            )[0]
    if compacted is None:
        # A production-valid tree always has a same-typed representation at
        # the budgets used by the operators (at least six nodes).  Fail closed
        # if that invariant is ever violated instead of silently erasing the
        # mechanism with a constant or unrelated stock-price proxy.
        raise RuntimeError("no informative same-type compaction fits the node budget")
    compacted_type = _semantic_type(compacted)
    if expression_complexity(compacted) > max_nodes or compacted_type is None:
        raise RuntimeError("typed semantic compaction failed its postcondition")
    if original_type is not None and compacted_type != original_type:
        raise RuntimeError("typed semantic compaction changed the expression type")
    if len(_COMPACTION_CACHE) >= _COMPACTION_CACHE_LIMIT:
        _COMPACTION_CACHE.clear()
    _COMPACTION_CACHE[cache_key] = deepcopy(compacted)
    return deepcopy(compacted)


def _different_scales(left: set[str], right: set[str]) -> bool:
    return ("short_scale" in left and "long_scale" in right) or ("long_scale" in left and "short_scale" in right)


class CodexSeedInjectionFactory:
    """Create independent injections from frozen Codex Factor Cards.

    Real-data Plus runs draw each injection exactly once from a second,
    precommitted Codex library. Sanitized development feedback may select an
    under-covered role and repair class, but it cannot rewrite the chosen AST.
    The older type-safe mutation of generation-zero cards remains available as
    an explicitly labelled fallback for synthetic tests and control studies.
    The factory never sees market data, labels, arbitrary evaluator
    diagnostics, or sealed results.
    """

    def __init__(
        self,
        seeds: Sequence[Candidate],
        *,
        injection_cards: Sequence[Candidate] | None = None,
        seed: int = 17,
        max_nodes: int = 48,
    ) -> None:
        if not seeds:
            raise ValueError("Codex injection factory requires at least one Factor Card")
        self.seeds = tuple(deepcopy(item) for item in seeds)
        self.injection_cards = tuple(
            deepcopy(item) for item in (injection_cards or ())
        )
        self.rng = random.Random(seed)
        self.operators = MechanismAwareOperators(seed=seed + 7_919, max_nodes=max_nodes)
        self._emitted_hashes = {item.expression_hash for item in seeds}
        injection_hashes = [item.expression_hash for item in self.injection_cards]
        if len(set(injection_hashes)) != len(injection_hashes):
            raise ValueError("Codex adaptive injection cards must be structurally unique")
        overlap = self._emitted_hashes.intersection(injection_hashes)
        if overlap:
            raise ValueError(
                "Codex adaptive injection cards duplicate generation-zero seeds"
            )
        self._unused_injection_ids = {
            item.candidate_id for item in self.injection_cards
        }
        self._counter = 0

    @staticmethod
    def _repair_tag(fix: str) -> str:
        return f"repair_{fix}"

    def _select_precommitted_card(
        self,
        feedback: FeedbackDirective | None,
    ) -> tuple[Candidate, bool, bool]:
        available = [
            item
            for item in self.injection_cards
            if item.candidate_id in self._unused_injection_ids
        ]
        if not available:
            raise RuntimeError("precommitted Codex adaptive injection library is exhausted")
        target_roles = set(() if feedback is None else feedback.target_roles)
        repair_tag = None if feedback is None else self._repair_tag(feedback.fix)

        def key(item: Candidate) -> tuple[int, int, int, str]:
            role_match = bool(target_roles.intersection(candidate_roles(item)))
            fix_match = repair_tag is not None and repair_tag in item.mechanism_tags
            both = role_match and fix_match
            return (
                -int(both),
                -int(role_match),
                -int(fix_match),
                item.candidate_id,
            )

        selected = sorted(available, key=key)[0]
        self._unused_injection_ids.remove(selected.candidate_id)
        role_match = bool(target_roles.intersection(candidate_roles(selected)))
        fix_match = repair_tag is not None and repair_tag in selected.mechanism_tags
        return selected, role_match, fix_match

    def available_precommitted_roles(self) -> tuple[str, ...]:
        """Roles represented by cards which have not yet been injected."""

        return tuple(
            sorted(
                {
                    role
                    for item in self.injection_cards
                    if item.candidate_id in self._unused_injection_ids
                    for role in candidate_roles(item)
                }
            )
        )

    def precommitted_expression_hashes(self) -> frozenset[str]:
        """All reserved AST identities, including cards not yet selected."""

        return frozenset(item.expression_hash for item in self.injection_cards)

    def precommitted_card_ids_by_hash(self) -> Mapping[str, str]:
        """Bind every reserved AST to its immutable source Factor Card."""

        return {
            item.expression_hash: item.candidate_id
            for item in self.injection_cards
        }

    def remaining_precommitted_count(self) -> int:
        return len(self._unused_injection_ids)

    def generate(
        self,
        count: int,
        *,
        generation: int = 0,
        feedback: FeedbackDirective | None = None,
    ) -> list[Candidate]:
        if count < 0:
            raise ValueError("injection count cannot be negative")
        if self.injection_cards:
            candidates: list[Candidate] = []
            for _ in range(count):
                base, role_match, fix_match = self._select_precommitted_card(
                    feedback
                )
                self._counter += 1
                child = deepcopy(base)
                child.generation = generation
                child.parents = ()
                child.operator = "injection:codex_precommitted_adaptive_card"
                child.metadata = {
                    **child.metadata,
                    "source": "codex_precommitted_adaptive_injection_card",
                    "source_factor_card_id": base.candidate_id,
                    "source_agent_id": base.agent_id,
                    "injection_sequence": self._counter,
                    "validation_feedback_visible": feedback is not None,
                    "role_lineage": list(candidate_roles(base)),
                    "target_role_matched": role_match,
                    "repair_fix_matched": fix_match,
                    "precommitted_library_remaining": len(
                        self._unused_injection_ids
                    ),
                }
                if feedback is not None:
                    child.metadata["adaptive_feedback"] = feedback.to_dict()
                child.candidate_id = deterministic_candidate_id(child)
                digest = child.expression_hash
                if digest in self._emitted_hashes:
                    raise RuntimeError(
                        "precommitted Codex injection unexpectedly duplicates an "
                        "already evaluated expression"
                    )
                self._emitted_hashes.add(digest)
                candidates.append(child)
            return candidates
        candidates: list[Candidate] = []
        attempts = 0
        maximum_attempts = max(2_000, count * 500)
        while len(candidates) < count and attempts < maximum_attempts:
            attempts += 1
            eligible_seeds = self.seeds
            if feedback is not None and feedback.target_roles:
                targeted = tuple(
                    item
                    for item in self.seeds
                    if set(candidate_roles(item)).intersection(feedback.target_roles)
                )
                if targeted:
                    eligible_seeds = targeted
            base = self.rng.choice(eligible_seeds)
            mutation_class = "parameter" if self.rng.random() < 0.5 else "semantic"
            child = self.operators.mutate(
                base,
                generation,
                mutation_class=mutation_class,
                feedback_fix=None if feedback is None else feedback.fix,
            )
            digest = child.expression_hash
            if digest in self._emitted_hashes:
                continue
            self._counter += 1
            self._emitted_hashes.add(digest)
            child.parents = ()
            child.operator = f"injection:codex_factor_card:{mutation_class}"
            child.metadata = {
                **child.metadata,
                "source": "codex_seed_injection",
                "source_factor_card_id": base.candidate_id,
                "source_agent_id": base.agent_id,
                "injection_sequence": self._counter,
                "validation_feedback_visible": feedback is not None,
                "role_lineage": list(candidate_roles(base)),
            }
            if feedback is not None:
                child.metadata["adaptive_feedback"] = feedback.to_dict()
            child.candidate_id = deterministic_candidate_id(child)
            candidates.append(child)
        if len(candidates) != count:
            raise RuntimeError("could not generate enough unique Codex-derived injections")
        return candidates


class EvolutionEngine:
    """Run the complete genetic loop using a validation-only scoring callback."""

    def __init__(
        self,
        *,
        evaluator: Callable[[Candidate], CandidateScore],
        factory: TemplateCandidateFactory | CodexSeedInjectionFactory,
        operators: MechanismAwareOperators,
        budget: SearchBudget | None = None,
        mode: str = "plus",
        seed: int = 17,
        on_evaluated: Callable[[EvaluatedCandidate], None] | None = None,
        feedback_memory: AdaptiveFeedbackMemory | None = None,
        champion_selector: Callable[
            [Sequence[EvaluatedCandidate], int, BehavioralChampionPolicy],
            ChampionSelectionResult,
        ]
        | None = None,
        behavioral_champion_policy: BehavioralChampionPolicy | None = None,
        on_terminal_audit: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if mode not in {"faithful", "plus"}:
            raise ValueError("mode must be faithful or plus")
        self.evaluator = evaluator
        self.factory = factory
        self.operators = operators
        self.budget = budget or SearchBudget()
        self.mode = mode
        self.rng = random.Random(seed)
        self.on_evaluated = on_evaluated
        self.on_terminal_audit = on_terminal_audit
        self.terminal_audit: Mapping[str, object] = {}
        if mode == "faithful" and (
            champion_selector is not None or behavioral_champion_policy is not None
        ):
            raise ValueError(
                "faithful mode cannot receive the Plus behavioral champion gate"
            )
        if (champion_selector is None) != (behavioral_champion_policy is None):
            raise ValueError(
                "behavioral champion selector and policy must be supplied together"
            )
        self.champion_selector = champion_selector
        self.behavioral_champion_policy = behavioral_champion_policy
        self.feedback_memory = (
            feedback_memory or AdaptiveFeedbackMemory()
            if mode == "plus"
            else None
        )
        reserved_provider = getattr(
            self.factory,
            "precommitted_expression_hashes",
            None,
        )
        self._reserved_injection_hashes = (
            frozenset(reserved_provider()) if callable(reserved_provider) else frozenset()
        )
        source_id_provider = getattr(
            self.factory,
            "precommitted_card_ids_by_hash",
            None,
        )
        self._reserved_injection_source_ids = (
            dict(source_id_provider()) if callable(source_id_provider) else {}
        )
        if set(self._reserved_injection_source_ids) != set(
            self._reserved_injection_hashes
        ):
            raise ValueError(
                "precommitted injection AST/source-card identity is inconsistent"
            )
        if self._reserved_injection_hashes:
            expected_injections = sum(
                self.budget.injection_count
                for generation in range(1, self.budget.generations + 1)
                if generation % self.budget.injection_every == 0
            )
            if len(self._reserved_injection_hashes) != expected_injections:
                raise ValueError(
                    "precommitted injection library size must equal the exact "
                    f"search budget: cards={len(self._reserved_injection_hashes)}, "
                    f"slots={expected_injections}"
                )

    def _record_terminal_failure(self, audit: Mapping[str, object]) -> None:
        """Persist every terminal search failure before propagating it."""

        self.terminal_audit = dict(audit)
        if self.on_terminal_audit is not None:
            self.on_terminal_audit(self.terminal_audit)

    def _evaluate_unique(self, candidates: Iterable[Candidate], seen: set[str]) -> list[EvaluatedCandidate]:
        evaluated: list[EvaluatedCandidate] = []
        for candidate in candidates:
            digest = candidate.expression_hash
            is_precommitted_injection = (
                candidate.operator
                == "injection:codex_precommitted_adaptive_card"
                and candidate.metadata.get("source_factor_card_id")
                == self._reserved_injection_source_ids.get(digest)
            )
            if (
                digest in self._reserved_injection_hashes
                and not is_precommitted_injection
            ):
                # A future Codex card is a reserved trial.  Letting an earlier
                # mutation/crossover consume the same AST would make the
                # without-replacement 96-slot schedule exhaust early.
                continue
            if digest in seen:
                continue
            seen.add(digest)
            result = EvaluatedCandidate(candidate, self.evaluator(candidate))
            evaluated.append(result)
            if self.on_evaluated:
                self.on_evaluated(result)
        return evaluated

    def _select_parents(self, pool: Sequence[EvaluatedCandidate]) -> list[EvaluatedCandidate]:
        eligible = [item for item in pool if item.score.passed]
        if len(eligible) < self.budget.parent_pool:
            eligible = list(pool)
        ranked = sorted(eligible, key=lambda item: item.score.fitness, reverse=True)
        if self.mode == "faithful":
            return ranked[: self.budget.parent_pool]

        # Quality-diversity: guarantee elites and retain one viable island
        # representative per original role before rotating mechanism niches.
        selected = ranked[: self.budget.elite_carry]
        selected_ids = {item.candidate.candidate_id for item in selected}
        if self.feedback_memory is not None:
            for role in self.feedback_memory.roles:
                if len(selected) >= self.budget.parent_pool:
                    break
                representative = next(
                    (
                        item
                        for item in ranked
                        if item.candidate.candidate_id not in selected_ids
                        and role in candidate_roles(item.candidate)
                    ),
                    None,
                )
                if representative is not None:
                    selected.append(representative)
                    selected_ids.add(representative.candidate.candidate_id)
        represented: dict[str, int] = {}
        for item in selected:
            for tag in item.candidate.mechanism_tags:
                represented[tag] = represented.get(tag, 0) + 1
        while len(selected) < self.budget.parent_pool:
            best: EvaluatedCandidate | None = None
            best_value = -math.inf
            for item in ranked:
                if item.candidate.candidate_id in selected_ids:
                    continue
                tags = item.candidate.mechanism_tags
                crowding = sum(represented.get(tag, 0) for tag in tags) / max(len(tags), 1)
                value = item.score.fitness + 0.0025 * item.score.novelty - 0.00025 * crowding
                if value > best_value:
                    best, best_value = item, value
            if best is None:
                break
            selected.append(best)
            selected_ids.add(best.candidate.candidate_id)
            for tag in best.candidate.mechanism_tags:
                represented[tag] = represented.get(tag, 0) + 1
        return selected

    def _allocation(self, generation: int) -> dict[str, int]:
        injection = self.budget.injection_count if generation % self.budget.injection_every == 0 else 0
        evolved_count = self.budget.children_per_generation - injection
        if evolved_count < 0:
            raise ValueError("injection count exceeds children-per-generation budget")
        categories = (
            ["deterministic_mutation", "semantic_mutation", "crossover", "crossover_then_mutation"]
            if self.mode == "plus"
            else ["semantic_mutation", "crossover", "crossover_then_mutation"]
        )
        base, remainder = divmod(evolved_count, len(categories))
        allocation = {name: base + (position < remainder) for position, name in enumerate(categories)}
        allocation["injection"] = injection
        return allocation

    @staticmethod
    def _category(candidate: Candidate) -> str:
        if candidate.operator.startswith("crossover_mutation:"):
            return "crossover_then_mutation"
        if candidate.operator.startswith("crossover:"):
            return "crossover"
        if candidate.operator.startswith("mutation:parameter:"):
            return "deterministic_mutation"
        if candidate.operator.startswith("mutation:semantic:"):
            return "semantic_mutation"
        return "injection"

    def _generate_one(
        self,
        category: str,
        parents: Sequence[EvaluatedCandidate],
        generation: int,
        slot: int,
    ) -> Candidate:
        parent_candidates = [item.candidate for item in parents]
        directive: FeedbackDirective | None = None
        if self.feedback_memory is not None:
            parent_roles = sorted(
                {
                    role
                    for item in parent_candidates
                    for role in candidate_roles(item)
                }
            )
            eligible_target_roles: Sequence[str] | None = None
            if category == "injection":
                available_roles = getattr(
                    self.factory,
                    "available_precommitted_roles",
                    None,
                )
                if callable(available_roles):
                    remaining_roles = tuple(available_roles())
                    if remaining_roles:
                        eligible_target_roles = remaining_roles
            directive = self.feedback_memory.directive(
                category=category,
                generation=generation,
                slot=slot,
                parent_roles=parent_roles,
                eligible_target_roles=eligible_target_roles,
            )
        if category == "deterministic_mutation":
            parent = self._feedback_parent(parents, directive, slot)
            candidate = self.operators.mutate(
                parent.candidate,
                generation,
                mutation_class="parameter",
                feedback_fix=None if directive is None else directive.fix,
            )
        elif category == "semantic_mutation":
            parent = self._feedback_parent(parents, directive, slot)
            candidate = self.operators.mutate(
                parent.candidate,
                generation,
                mutation_class="semantic",
                feedback_fix=None if directive is None else directive.fix,
            )
        elif category == "crossover":
            left, right = self._feedback_pair(parents, directive, slot)
            candidate = self.operators.crossover(
                left.candidate,
                right.candidate,
                generation,
                feedback_fix=None if directive is None else directive.fix,
            )
        elif category == "crossover_then_mutation":
            left, right = self._feedback_pair(parents, directive, slot)
            candidate = self.operators.crossover_then_mutate(
                left.candidate,
                right.candidate,
                generation,
                feedback_fix=None if directive is None else directive.fix,
            )
        elif category == "injection":
            candidate = self.factory.generate(
                1,
                generation=generation,
                feedback=directive,
            )[0]
        else:
            raise ValueError(f"unknown offspring category: {category}")
        if directive is not None:
            candidate.metadata = {
                **candidate.metadata,
                "adaptive_feedback": directive.to_dict(),
            }
        candidate.candidate_id = deterministic_candidate_id(candidate)
        return candidate

    def _feedback_parent(
        self,
        parents: Sequence[EvaluatedCandidate],
        directive: FeedbackDirective | None,
        slot: int,
    ) -> EvaluatedCandidate:
        if directive is None:
            return self.rng.choice(list(parents))
        matching = [
            item
            for item in parents
            if set(candidate_roles(item.candidate)).intersection(directive.target_roles)
        ]
        pool = matching or list(parents)
        source = next(
            (
                item
                for item in pool
                if item.candidate.candidate_id == directive.source_candidate_id
            ),
            None,
        )
        if source is not None:
            return source
        ordered = sorted(
            pool,
            key=lambda item: (-item.score.fitness, item.candidate.candidate_id),
        )
        return ordered[slot % len(ordered)]

    @staticmethod
    def _pair_metric(
        item: EvaluatedCandidate,
        directive: FeedbackDirective,
    ) -> float:
        if directive.fix == "retention_guarded_crossover":
            return item.score.style_retention
        if directive.fix == "stability_guarded_crossover":
            return item.score.worst_window_ic
        if directive.fix == "turnover_guarded_crossover":
            return -item.score.turnover
        if directive.fix == "novelty_crossover":
            return item.score.novelty
        if directive.fix == "coverage_guarded_crossover":
            coverage = item.score.diagnostics.get("coverage")
            try:
                return float(coverage)
            except (TypeError, ValueError):
                return 0.0
        return item.score.fitness

    def _feedback_pair(
        self,
        parents: Sequence[EvaluatedCandidate],
        directive: FeedbackDirective | None,
        slot: int,
    ) -> tuple[EvaluatedCandidate, EvaluatedCandidate]:
        if directive is None:
            left, right = self.rng.sample(list(parents), 2)
            return left, right
        left = self._feedback_parent(parents, directive, slot)
        left_tags = set(left.candidate.mechanism_tags)
        alternatives = [
            item
            for item in parents
            if item.candidate.candidate_id != left.candidate.candidate_id
        ]
        ordered = sorted(
            alternatives,
            key=lambda item: (
                -self._pair_metric(item, directive),
                len(left_tags.intersection(item.candidate.mechanism_tags)),
                item.candidate.candidate_id,
            ),
        )
        # Rotate among the best half to retain deterministic variety while the
        # leading sort key still implements the requested repair objective.
        band = ordered[: max(2, len(ordered) // 2)]
        right = band[slot % len(band)]
        return left, right

    def _children(self, parents: Sequence[EvaluatedCandidate], generation: int) -> list[Candidate]:
        children: list[Candidate] = []
        for category, count in self._allocation(generation).items():
            for slot in range(count):
                children.append(
                    self._generate_one(category, parents, generation, slot)
                )
        return children

    @staticmethod
    def _feedback_generation_audit(
        evaluated: Sequence[EvaluatedCandidate],
        parents: Sequence[EvaluatedCandidate],
    ) -> dict[str, object]:
        directives: list[dict[str, object]] = []
        causes: Counter[str] = Counter()
        fixes: Counter[str] = Counter()
        target_roles: Counter[str] = Counter()
        precommitted_injections = 0
        target_role_matches = 0
        repair_fix_matches = 0
        exact_role_and_fix_matches = 0
        remaining_library_sizes: list[int] = []
        for item in evaluated:
            if (
                item.candidate.operator
                == "injection:codex_precommitted_adaptive_card"
            ):
                precommitted_injections += 1
                role_match = bool(
                    item.candidate.metadata.get("target_role_matched", False)
                )
                fix_match = bool(
                    item.candidate.metadata.get("repair_fix_matched", False)
                )
                target_role_matches += int(role_match)
                repair_fix_matches += int(fix_match)
                exact_role_and_fix_matches += int(role_match and fix_match)
                remaining = item.candidate.metadata.get(
                    "precommitted_library_remaining"
                )
                if isinstance(remaining, int):
                    remaining_library_sizes.append(remaining)
            payload = item.candidate.metadata.get("adaptive_feedback")
            if not isinstance(payload, Mapping):
                continue
            clean = {
                "candidate_id": item.candidate.candidate_id,
                "source_candidate_id": payload.get("source_candidate_id"),
                "cause": payload.get("cause"),
                "fix": payload.get("fix"),
                "target_roles": list(payload.get("target_roles", [])),
            }
            directives.append(clean)
            causes[str(clean["cause"])] += 1
            fixes[str(clean["fix"])] += 1
            for role in clean["target_roles"]:
                target_roles[str(role)] += 1
        serialized = json.dumps(
            sorted(directives, key=lambda item: str(item["candidate_id"])),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        represented_parent_roles = sorted(
            {
                role
                for item in parents
                for role in candidate_roles(item.candidate)
            }
        )
        return {
            "directives_applied": len(directives),
            "causes": dict(sorted(causes.items())),
            "fixes": dict(sorted(fixes.items())),
            "target_roles": dict(sorted(target_roles.items())),
            "parent_role_coverage": represented_parent_roles,
            "precommitted_injections": precommitted_injections,
            "precommitted_target_role_matches": target_role_matches,
            "precommitted_repair_fix_matches": repair_fix_matches,
            "precommitted_exact_role_and_fix_matches": exact_role_and_fix_matches,
            "precommitted_library_remaining": (
                min(remaining_library_sizes) if remaining_library_sizes else None
            ),
            "directive_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "sealed_feedback_consumed": False,
        }

    def run(self, initial: Sequence[Candidate] | None = None) -> EvolutionResult:
        initial_candidates = list(initial or self.factory.generate(self.budget.initial_population, generation=0))
        if len(initial_candidates) != self.budget.initial_population:
            raise ValueError(f"initial population must contain exactly {self.budget.initial_population} candidates")
        seen: set[str] = set()
        archive = self._evaluate_unique(initial_candidates, seen)
        if len(archive) < self.budget.parent_pool:
            raise RuntimeError("deduplication left too few initial parents")
        current = list(archive)
        summaries: list[Mapping[str, object]] = []
        if self.feedback_memory is not None:
            self.feedback_memory.register_roles(initial_candidates)
            self.feedback_memory.observe(archive)

        for generation in range(1, self.budget.generations + 1):
            parents = self._select_parents(current)
            if len(parents) < 2:
                raise RuntimeError("not enough eligible parents for crossover")
            requested = self._children(parents, generation)
            evaluated = self._evaluate_unique(requested, seen)
            # Preserve the declared operator allocation after structural
            # deduplication. A duplicate crossover is replaced by another
            # crossover—not silently by a fresh template.
            desired = self._allocation(generation)
            actual = {name: 0 for name in desired}
            for item in evaluated:
                actual[self._category(item.candidate)] += 1
            refill_attempts = 0
            while any(actual[name] < desired[name] for name in desired) and refill_attempts < 10_000:
                refill_attempts += 1
                category = next(name for name in desired if actual[name] < desired[name])
                candidate = self._generate_one(
                    category,
                    parents,
                    generation,
                    desired[category] + refill_attempts,
                )
                added = self._evaluate_unique([candidate], seen)
                if added:
                    evaluated.extend(added)
                    actual[category] += 1
            if len(evaluated) != self.budget.children_per_generation:
                raise RuntimeError("could not satisfy the unique child budget")
            if actual != desired:
                raise RuntimeError(f"could not preserve operator allocation: desired={desired}, actual={actual}")
            elites = sorted(parents, key=lambda item: item.score.fitness, reverse=True)[: self.budget.elite_carry]
            current = elites + evaluated
            archive.extend(evaluated)
            if self.feedback_memory is not None:
                self.feedback_memory.observe(evaluated)
            ranked_generation = sorted(evaluated, key=lambda item: item.score.fitness, reverse=True)
            generation_summary: dict[str, object] = {
                    "generation": generation,
                    "subcycle": (generation - 1) // self.budget.generations_per_subcycle + 1,
                    "children": len(evaluated),
                    "deterministic_mutations": actual.get("deterministic_mutation", 0),
                    "semantic_mutations": actual.get("semantic_mutation", 0),
                    "crossovers": sum(item.candidate.operator.startswith("crossover:") for item in evaluated),
                    "crossover_then_mutations": actual.get("crossover_then_mutation", 0),
                    "injections": actual.get("injection", 0),
                    "best_fitness": ranked_generation[0].score.fitness,
                    "median_fitness": float(sorted(item.score.fitness for item in evaluated)[len(evaluated) // 2]),
                }
            if self.feedback_memory is not None:
                generation_summary["adaptive_feedback"] = self._feedback_generation_audit(
                    evaluated,
                    parents,
                )
                generation_summary["remembered_failures"] = self.feedback_memory.summary()[
                    "remembered_failures"
                ]
            summaries.append(generation_summary)

        if self._reserved_injection_hashes:
            evaluated_precommitted = {
                item.candidate.expression_hash
                for item in archive
                if item.candidate.operator
                == "injection:codex_precommitted_adaptive_card"
            }
            remaining_provider = getattr(
                self.factory,
                "remaining_precommitted_count",
                None,
            )
            remaining = (
                int(remaining_provider())
                if callable(remaining_provider)
                else -1
            )
            if (
                evaluated_precommitted != self._reserved_injection_hashes
                or remaining != 0
            ):
                self._record_terminal_failure({
                    "status": "failed_precommitted_injection_accounting",
                    "reserved_cards": len(self._reserved_injection_hashes),
                    "evaluated_precommitted_cards": len(evaluated_precommitted),
                    "remaining_precommitted_cards": remaining,
                })
                raise RuntimeError(
                    "precommitted injection accounting failed closed: "
                    f"reserved={len(self._reserved_injection_hashes)}, "
                    f"evaluated={len(evaluated_precommitted)}, remaining={remaining}"
                )

        eligible_champions = [item for item in archive if item.score.passed]
        if len(eligible_champions) < self.budget.final_factors:
            rejection_counts: Counter[str] = Counter()
            for item in archive:
                if item.score.passed:
                    continue
                reason = item.score.diagnostics.get("rejection")
                rejection_counts[str(reason or "unspecified_rejection")] += 1
            self._record_terminal_failure({
                "status": "failed_insufficient_passed_champions",
                "eligible_passed_count": len(eligible_champions),
                "required_final_factors": self.budget.final_factors,
                "rejected_count": len(archive) - len(eligible_champions),
                "rejection_counts": dict(sorted(rejection_counts.items())),
            })
            raise RuntimeError(
                "development search failed closed: "
                f"eligible passed champions={len(eligible_champions)} "
                f"is below required final_factors={self.budget.final_factors}; "
                f"rejections={dict(sorted(rejection_counts.items()))}"
            )
        # Preserve the published faithful control arm exactly: Python's stable
        # sort keeps archive order for tied fitness.  Plus adds an explicit id
        # tie-break before its separate behavior gate for deterministic replay.
        ranked_eligible = (
            sorted(
                eligible_champions,
                key=lambda item: (-item.score.fitness, item.candidate.candidate_id),
            )
            if self.mode == "plus"
            else sorted(
                eligible_champions,
                key=lambda item: item.score.fitness,
                reverse=True,
            )
        )
        if self.mode == "plus" and self.champion_selector is not None:
            assert self.behavioral_champion_policy is not None
            try:
                selection = self.champion_selector(
                    ranked_eligible,
                    self.budget.final_factors,
                    self.behavioral_champion_policy,
                )
            except Exception as exc:
                self._record_terminal_failure(
                    {
                        "status": "failed_behavioral_champion_selector",
                        "eligible_passed_count": len(eligible_champions),
                        "required_final_factors": self.budget.final_factors,
                        "error_type": type(exc).__name__,
                    }
                )
                raise RuntimeError(
                    "development search failed closed: behavioral champion "
                    f"selector raised {type(exc).__name__}"
                ) from exc
            champions = list(selection.champions)
            selection_audit = dict(selection.audit)
            if len(champions) < self.budget.final_factors:
                self._record_terminal_failure({
                    "status": "failed_insufficient_behaviorally_distinct_champions",
                    "eligible_passed_count": len(eligible_champions),
                    "required_final_factors": self.budget.final_factors,
                    "selected_distinct_count": len(champions),
                    "behavioral_selection": selection_audit,
                })
                raise RuntimeError(
                    "development search failed closed: behavior gate selected "
                    f"{len(champions)} distinct champions but requires "
                    f"{self.budget.final_factors}; conflicts="
                    f"{selection_audit.get('correlation_conflict_candidates', 0)}"
                )
            if len(champions) > self.budget.final_factors:
                self._record_terminal_failure(
                    {
                        "status": "failed_behavioral_selector_exceeded_budget",
                        "eligible_passed_count": len(eligible_champions),
                        "required_final_factors": self.budget.final_factors,
                        "selected_count": len(champions),
                        "behavioral_selection": selection_audit,
                    }
                )
                raise RuntimeError(
                    "behavioral champion selector exceeded the frozen final-factor budget"
                )
            if len({item.candidate.candidate_id for item in champions}) != len(
                champions
            ):
                self._record_terminal_failure(
                    {
                        "status": "failed_behavioral_selector_duplicates",
                        "eligible_passed_count": len(eligible_champions),
                        "required_final_factors": self.budget.final_factors,
                        "selected_count": len(champions),
                        "behavioral_selection": selection_audit,
                    }
                )
                raise RuntimeError("behavioral champion selector returned duplicates")
            eligible_by_id = {
                item.candidate.candidate_id: item for item in eligible_champions
            }
            if any(
                item.candidate.candidate_id not in eligible_by_id
                for item in champions
            ):
                self._record_terminal_failure(
                    {
                        "status": "failed_behavioral_selector_ineligible_candidate",
                        "eligible_passed_count": len(eligible_champions),
                        "required_final_factors": self.budget.final_factors,
                        "selected_count": len(champions),
                        "behavioral_selection": selection_audit,
                    }
                )
                raise RuntimeError(
                    "behavioral champion selector returned an ineligible candidate"
                )
            if any(
                item is not eligible_by_id[item.candidate.candidate_id]
                for item in champions
            ):
                self._record_terminal_failure(
                    {
                        "status": "failed_behavioral_selector_altered_candidate",
                        "eligible_passed_count": len(eligible_champions),
                        "required_final_factors": self.budget.final_factors,
                        "selected_count": len(champions),
                        "behavioral_selection": selection_audit,
                    }
                )
                raise RuntimeError(
                    "behavioral champion selector altered an eligible candidate"
                )
        else:
            champions = ranked_eligible[: self.budget.final_factors]
            selection_audit = {
                "schema": "champion_selection_v1",
                "mode": "faithful_fitness_rank"
                if self.mode == "faithful"
                else "fitness_rank_without_external_behavior_selector",
                "eligible_passed_count": len(eligible_champions),
                "required_final_factors": self.budget.final_factors,
                "selected_count": len(champions),
                "sealed_data_consumed": False,
            }
        self.terminal_audit = {
            "status": "complete",
            "eligible_passed_count": len(eligible_champions),
            "required_final_factors": self.budget.final_factors,
            "rejected_count": len(archive) - len(eligible_champions),
            "precommitted_injection_cards": len(
                self._reserved_injection_hashes
            ),
            "precommitted_injection_cards_remaining": 0,
            "behavioral_selection": selection_audit,
        }
        return EvolutionResult(
            tuple(champions),
            tuple(archive),
            tuple(summaries),
            {} if self.feedback_memory is None else self.feedback_memory.summary(),
            selection_audit,
        )
