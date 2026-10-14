"""Deterministic Pareto ranking for quality-first factor evolution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .evolution import EvaluatedCandidate


@dataclass(frozen=True)
class ParetoSelectionPolicy:
    schema_version: int = 1
    objectives: tuple[str, ...] = (
        "raw_rank_ic",
        "neutral_rank_ic",
        "stability",
        "novelty",
        "turnover_efficiency",
        "parent_delta",
    )
    dominance_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        allowed = {
            "raw_rank_ic",
            "neutral_rank_ic",
            "stability",
            "novelty",
            "turnover_efficiency",
            "parent_delta",
            "style_retention",
        }
        if not self.objectives or set(self.objectives) - allowed:
            raise ValueError("Pareto objectives are empty or unsupported")
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError("Pareto objectives must be unique")
        if not math.isfinite(self.dominance_epsilon) or self.dominance_epsilon < 0:
            raise ValueError("dominance_epsilon must be finite and non-negative")


def _objective(item: EvaluatedCandidate, name: str) -> float:
    score = item.score
    if name == "raw_rank_ic":
        value = score.signed_rank_ic
    elif name == "neutral_rank_ic":
        value = score.neutral_rank_ic
    elif name == "stability":
        value = score.worst_window_ic
    elif name == "novelty":
        value = score.novelty
    elif name == "turnover_efficiency":
        value = -score.turnover
    elif name == "parent_delta":
        value = -1.0 if score.control_delta is None else score.control_delta
    elif name == "style_retention":
        value = score.style_retention
    else:  # pragma: no cover - policy validation makes this unreachable.
        raise ValueError(f"unsupported Pareto objective: {name}")
    number = float(value)
    return number if math.isfinite(number) else -math.inf


def objective_vector(
    item: EvaluatedCandidate, policy: ParetoSelectionPolicy
) -> tuple[float, ...]:
    return tuple(_objective(item, name) for name in policy.objectives)


def _dominates(left: np.ndarray, right: np.ndarray, epsilon: float) -> bool:
    return bool(np.all(left >= right - epsilon) and np.any(left > right + epsilon))


def non_dominated_fronts(
    candidates: Sequence[EvaluatedCandidate],
    policy: ParetoSelectionPolicy | None = None,
) -> tuple[tuple[EvaluatedCandidate, ...], ...]:
    """Return exact deterministic Pareto fronts, best first."""

    resolved = policy or ParetoSelectionPolicy()
    ordered = sorted(candidates, key=lambda item: item.candidate.candidate_id)
    if not ordered:
        return ()
    matrix = np.asarray(
        [objective_vector(item, resolved) for item in ordered], dtype=float
    )
    dominates: list[list[int]] = [[] for _ in ordered]
    dominated_count = np.zeros(len(ordered), dtype=int)
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            if _dominates(matrix[left], matrix[right], resolved.dominance_epsilon):
                dominates[left].append(right)
                dominated_count[right] += 1
            elif _dominates(matrix[right], matrix[left], resolved.dominance_epsilon):
                dominates[right].append(left)
                dominated_count[left] += 1
    active = [index for index, count in enumerate(dominated_count) if count == 0]
    fronts: list[tuple[EvaluatedCandidate, ...]] = []
    while active:
        active.sort(key=lambda index: ordered[index].candidate.candidate_id)
        fronts.append(tuple(ordered[index] for index in active))
        following: list[int] = []
        for index in active:
            for dominated in dominates[index]:
                dominated_count[dominated] -= 1
                if dominated_count[dominated] == 0:
                    following.append(dominated)
        active = following
    if sum(map(len, fronts)) != len(ordered):
        raise RuntimeError("Pareto sorting failed to assign every candidate")
    return tuple(fronts)


def _crowding(
    front: Sequence[EvaluatedCandidate], policy: ParetoSelectionPolicy
) -> dict[str, float]:
    if not front:
        return {}
    distances = {item.candidate.candidate_id: 0.0 for item in front}
    if len(front) <= 2:
        return {key: math.inf for key in distances}
    for objective_index in range(len(policy.objectives)):
        ordered = sorted(
            front,
            key=lambda item: (
                objective_vector(item, policy)[objective_index],
                item.candidate.candidate_id,
            ),
        )
        low = objective_vector(ordered[0], policy)[objective_index]
        high = objective_vector(ordered[-1], policy)[objective_index]
        distances[ordered[0].candidate.candidate_id] = math.inf
        distances[ordered[-1].candidate.candidate_id] = math.inf
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            continue
        for position in range(1, len(ordered) - 1):
            key = ordered[position].candidate.candidate_id
            if math.isinf(distances[key]):
                continue
            prior = objective_vector(ordered[position - 1], policy)[objective_index]
            following = objective_vector(ordered[position + 1], policy)[objective_index]
            distances[key] += (following - prior) / (high - low)
    return distances


def pareto_rank(
    candidates: Sequence[EvaluatedCandidate],
    policy: ParetoSelectionPolicy | None = None,
) -> list[EvaluatedCandidate]:
    """NSGA-II ordering with deterministic IDs as the final tie-break."""

    resolved = policy or ParetoSelectionPolicy()
    output: list[EvaluatedCandidate] = []
    for front in non_dominated_fronts(candidates, resolved):
        crowding = _crowding(front, resolved)
        output.extend(
            sorted(
                front,
                key=lambda item: (
                    -crowding[item.candidate.candidate_id],
                    -item.score.fitness,
                    item.candidate.candidate_id,
                ),
            )
        )
    return output


__all__ = [
    "ParetoSelectionPolicy",
    "non_dominated_fronts",
    "objective_vector",
    "pareto_rank",
]
