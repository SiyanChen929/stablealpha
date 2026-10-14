from __future__ import annotations

from cogalpha_plus.evolution import Candidate, CandidateScore, EvaluatedCandidate
from cogalpha_plus.multiobjective import (
    ParetoSelectionPolicy,
    non_dominated_fronts,
    pareto_rank,
)


def evaluated(
    candidate_id: str,
    *,
    raw: float,
    neutral: float,
    stability: float,
    novelty: float,
    turnover: float,
    delta: float | None,
) -> EvaluatedCandidate:
    candidate = Candidate(
        expression={"op": "const", "value": float(len(candidate_id))},
        agent_id="role",
        hypothesis="test",
        mechanism_tags=("test",),
        candidate_id=candidate_id,
    )
    return EvaluatedCandidate(
        candidate,
        CandidateScore(
            fitness=neutral,
            signed_rank_ic=raw,
            neutral_rank_ic=neutral,
            turnover=turnover,
            worst_window_ic=stability,
            style_retention=1.0,
            novelty=novelty,
            control_delta=delta,
        ),
    )


def test_pareto_front_preserves_tradeoffs_and_demotes_dominated_candidate() -> None:
    a = evaluated(
        "a", raw=0.08, neutral=0.03, stability=0.01, novelty=0.9, turnover=0.3, delta=0.01
    )
    b = evaluated(
        "b", raw=0.04, neutral=0.07, stability=0.03, novelty=0.8, turnover=0.2, delta=0.02
    )
    dominated = evaluated(
        "c", raw=0.01, neutral=0.01, stability=-0.01, novelty=0.2, turnover=0.8, delta=-0.05
    )
    fronts = non_dominated_fronts([dominated, b, a])
    assert {item.candidate.candidate_id for item in fronts[0]} == {"a", "b"}
    assert [item.candidate.candidate_id for item in fronts[1]] == ["c"]
    assert pareto_rank([dominated, b, a])[-1].candidate.candidate_id == "c"


def test_pareto_rank_is_input_order_invariant() -> None:
    policy = ParetoSelectionPolicy()
    items = [
        evaluated("a", raw=.03, neutral=.02, stability=.01, novelty=.7, turnover=.4, delta=None),
        evaluated("b", raw=.02, neutral=.03, stability=.02, novelty=.5, turnover=.2, delta=.0),
        evaluated("c", raw=.01, neutral=.01, stability=.00, novelty=.4, turnover=.5, delta=-.01),
    ]
    forward = [item.candidate.candidate_id for item in pareto_rank(items, policy)]
    backward = [item.candidate.candidate_id for item in pareto_rank(items[::-1], policy)]
    assert forward == backward
