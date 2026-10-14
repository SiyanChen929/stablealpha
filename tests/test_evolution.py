from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from cogalpha_plus.dsl import canonical_json, validate_expression
from cogalpha_plus.evolution import (
    AdaptiveFeedbackMemory,
    BehavioralChampionPolicy,
    Candidate,
    CandidateScore,
    ChampionSelectionResult,
    CodexSeedInjectionFactory,
    EvaluatedCandidate,
    EvolutionEngine,
    FeedbackDirective,
    MechanismAwareOperators,
    SearchBudget,
    TemplateCandidateFactory,
    _contains_expression_digest,
    candidate_roles,
    deterministic_candidate_id,
    expression_digest,
    expression_complexity,
)
from cogalpha_plus.hs300_research import (
    load_codex_injection_candidates,
    load_codex_seed_candidates,
)
from cogalpha_plus.validation import PRODUCTION_DSL_SPEC


def score(candidate):
    complexity = expression_complexity(candidate.expression)
    crossover_bonus = 0.005 if candidate.operator.startswith("crossover:") else 0.0
    fitness = 0.03 + crossover_bonus - complexity * 0.0001
    return CandidateScore(
        fitness=fitness,
        signed_rank_ic=0.02,
        neutral_rank_ic=0.018,
        turnover=0.10,
        worst_window_ic=0.005,
        style_retention=0.9,
        novelty=0.8,
    )


def test_original_budget_is_preserved():
    budget = SearchBudget()
    assert budget.initial_population == 80
    assert budget.parent_pool == 32
    assert budget.children_per_generation == 96
    assert budget.generations == 24
    assert budget.subcycles * budget.generations_per_subcycle == 24
    assert budget.final_factors == 20


def test_mechanism_crossover_keeps_lineage_and_control():
    factory = TemplateCandidateFactory(seed=3)
    parents = factory.generate(8)
    child = MechanismAwareOperators(seed=4).crossover(parents[0], parents[1], generation=1)
    assert child.operator.startswith("crossover:")
    assert child.parents == (parents[0].candidate_id, parents[1].candidate_id)
    assert child.control_expression is not None
    assert child.metadata["required_comparisons"] == [
        "left_parent",
        "right_parent",
        "equal_weight_linear_control",
    ]
    assert child.metadata["expected_direction"] == 1


def test_evolution_preserves_or_explicitly_reorients_hypothesis_direction():
    parents = TemplateCandidateFactory(seed=33).generate(2)
    parents[0].metadata["expected_direction"] = -1
    operators = MechanismAwareOperators(seed=34)
    mutation = operators.mutate(parents[0], generation=1, mutation_class="semantic")
    assert mutation.metadata["expected_direction"] == -1
    child = operators.crossover(parents[0], parents[1], generation=1)
    assert child.metadata["expected_direction"] == 1
    assert child.metadata["parent_expected_directions"] == [-1, 1]


def test_real_injections_are_unique_codex_card_derivatives_not_generic_templates():
    seeds = TemplateCandidateFactory(seed=301).generate(20)
    factory = CodexSeedInjectionFactory(seeds, seed=302)
    injected = factory.generate(12, generation=2)
    seed_hashes = {item.expression_hash for item in seeds}
    assert len({item.expression_hash for item in injected}) == 12
    assert not seed_hashes.intersection(item.expression_hash for item in injected)
    assert all(item.operator.startswith("injection:codex_factor_card:") for item in injected)
    assert all(item.parents == () for item in injected)
    assert all(item.metadata["source"] == "codex_seed_injection" for item in injected)
    assert all(item.metadata["validation_feedback_visible"] is False for item in injected)


def test_precommitted_codex_injection_prefers_live_role_and_repair_without_rewriting():
    seeds = TemplateCandidateFactory(seed=401).generate(8)
    cards = []
    specifications = (
        ("R1", "repair_shorten_lookback", 3),
        ("R1", "repair_stability_smoothing", 4),
        ("R2", "repair_shorten_lookback", 6),
        ("R2", "repair_stability_smoothing", 7),
    )
    for role, repair_tag, periods in specifications:
        card = Candidate(
            expression={
                "op": "cs_rank",
                "args": [
                    {
                        "op": "return",
                        "args": [{"op": "field", "name": "vwap"}],
                        "periods": periods,
                    }
                ],
            },
            agent_id=role,
            hypothesis=f"{role}-{repair_tag}",
            mechanism_tags=("injection", repair_tag),
            metadata={
                "expected_direction": 1,
                "forecast_horizon": 10,
                "role_lineage": [role],
                "candidate_key": f"{role}-{repair_tag}",
            },
        )
        card.candidate_id = deterministic_candidate_id(card)
        cards.append(card)

    factory = CodexSeedInjectionFactory(
        seeds,
        injection_cards=cards,
        seed=402,
    )
    assert factory.available_precommitted_roles() == ("R1", "R2")
    directive = FeedbackDirective(
        source_candidate_id=seeds[0].candidate_id,
        target_roles=("R2",),
        observation="window instability",
        cause="window_instability",
        fix="stability_smoothing",
        evidence={},
    )
    selected = factory.generate(1, generation=2, feedback=directive)[0]
    expected = next(
        item
        for item in cards
        if item.agent_id == "R2"
        and "repair_stability_smoothing" in item.mechanism_tags
    )
    assert selected.expression_hash == expected.expression_hash
    assert selected.operator == "injection:codex_precommitted_adaptive_card"
    assert selected.metadata["source_factor_card_id"] == expected.candidate_id
    assert selected.metadata["target_role_matched"] is True
    assert selected.metadata["repair_fix_matched"] is True
    assert selected.metadata["adaptive_feedback"]["sealed_feedback_consumed"] is False

    generation_audit = EvolutionEngine._feedback_generation_audit(
        [EvaluatedCandidate(selected, score(selected))],
        [EvaluatedCandidate(seeds[0], score(seeds[0]))],
    )
    assert generation_audit["precommitted_injections"] == 1
    assert generation_audit["precommitted_target_role_matches"] == 1
    assert generation_audit["precommitted_repair_fix_matches"] == 1
    assert generation_audit["precommitted_exact_role_and_fix_matches"] == 1
    assert generation_audit["precommitted_library_remaining"] == 3

    remainder = factory.generate(3, generation=4)
    assert len({selected.expression_hash, *(item.expression_hash for item in remainder)}) == 4
    assert factory.available_precommitted_roles() == ()
    with pytest.raises(RuntimeError, match="library is exhausted"):
        factory.generate(1, generation=6)


def test_feedback_injection_scheduler_targets_only_roles_with_unused_cards():
    roles = TemplateCandidateFactory(seed=410).generate(2)
    roles[0].agent_id = "R1"
    roles[0].metadata["role_lineage"] = ["R1"]
    roles[1].agent_id = "R2"
    roles[1].metadata["role_lineage"] = ["R2"]
    memory = AdaptiveFeedbackMemory()
    memory.register_roles(roles)

    directive = memory.directive(
        category="injection",
        generation=24,
        slot=0,
        parent_roles=("R1", "R2"),
        eligible_target_roles=("R2",),
    )
    assert directive.target_roles == ("R2",)
    assert directive.evidence["target_limited_to_unused_precommitted_role"] is True

    with pytest.raises(RuntimeError, match="no registered role"):
        memory.directive(
            category="injection",
            generation=24,
            slot=1,
            parent_roles=("R1", "R2"),
            eligible_target_roles=("R3",),
        )


def test_precommitted_ast_reservation_prevents_early_evolution_consumption():
    seeds = TemplateCandidateFactory(seed=420).generate(8)
    roles = (seeds[0].agent_id, seeds[1].agent_id)
    cards = []
    for index, periods in enumerate((13, 14, 16, 18)):
        role = roles[index % len(roles)]
        card = Candidate(
            expression={
                "op": "cs_rank",
                "args": [
                    {
                        "op": "return",
                        "args": [{"op": "field", "name": "vwap"}],
                        "periods": periods,
                    }
                ],
            },
            agent_id=role,
            hypothesis=f"reserved-{index}",
            mechanism_tags=("injection", "repair_parameter_refinement"),
            metadata={
                "expected_direction": 1,
                "forecast_horizon": 10,
                "role_lineage": [role],
                "candidate_key": f"reserved-{index}",
            },
        )
        card.candidate_id = deterministic_candidate_id(card)
        cards.append(card)

    factory = CodexSeedInjectionFactory(
        seeds,
        injection_cards=cards,
        seed=421,
    )
    engine = EvolutionEngine(
        evaluator=score,
        factory=factory,
        operators=MechanismAwareOperators(seed=421),
        budget=SearchBudget.tiny(),
        mode="plus",
        seed=421,
    )
    collision = deepcopy(cards[0])
    collision.operator = "mutation:semantic:forced_reserved_collision"
    collision.generation = 1
    collision.candidate_id = deterministic_candidate_id(collision)
    seen = set()
    assert engine._evaluate_unique([collision], seen) == []
    assert seen == set()
    assert factory.remaining_precommitted_count() == 4
    spoof = deepcopy(cards[0])
    spoof.operator = "injection:codex_precommitted_adaptive_card"
    spoof.metadata["source_factor_card_id"] = "spoofed-card-id"
    spoof.candidate_id = deterministic_candidate_id(spoof)
    assert engine._evaluate_unique([spoof], seen) == []
    assert seen == set()

    result = engine.run(seeds)
    injected = [
        item
        for item in result.archive
        if item.candidate.operator
        == "injection:codex_precommitted_adaptive_card"
    ]
    assert len(injected) == 4
    assert {item.candidate.expression_hash for item in injected} == {
        item.expression_hash for item in cards
    }
    assert all(item.candidate.metadata["target_role_matched"] for item in injected)
    assert factory.remaining_precommitted_count() == 0
    assert engine.terminal_audit["precommitted_injection_cards"] == 4
    assert engine.terminal_audit["precommitted_injection_cards_remaining"] == 0


def test_tiny_evolution_runs_exact_budget_with_crossovers():
    budget = SearchBudget.tiny()
    engine = EvolutionEngine(
        evaluator=score,
        factory=TemplateCandidateFactory(seed=9),
        operators=MechanismAwareOperators(seed=9),
        budget=budget,
        mode="plus",
        seed=9,
    )
    result = engine.run()
    assert len(result.archive) == budget.initial_population + budget.generations * budget.children_per_generation
    assert len(result.champions) == budget.final_factors
    assert any(item.candidate.operator.startswith("crossover:") for item in result.archive)
    assert len(result.generations) == budget.generations


def test_full_faithful_budget_is_unique_and_bloat_bounded():
    budget = SearchBudget()
    engine = EvolutionEngine(
        evaluator=score,
        factory=TemplateCandidateFactory(seed=1701),
        operators=MechanismAwareOperators(seed=1701),
        budget=budget,
        mode="plus",
        seed=1701,
    )
    result = engine.run()
    assert len(result.archive) == 80 + 24 * 96
    assert len({item.candidate.expression_hash for item in result.archive}) == len(result.archive)
    assert max(expression_complexity(item.candidate.expression) for item in result.archive) <= 48
    pure = sum(item.candidate.operator.startswith("crossover:") for item in result.archive)
    crossed_mutation = sum(item.candidate.operator.startswith("crossover_mutation:") for item in result.archive)
    assert pure == 552
    assert crossed_mutation == 552
    assert all(summary["children"] == 96 for summary in result.generations)


def test_full_codex_budget_has_complete_typed_operator_yield():
    """Every declared slot must reach the evaluator as a production factor."""

    project = Path(__file__).resolve().parents[1]
    seeds = load_codex_seed_candidates(project)
    injection_cards = load_codex_injection_candidates(project)
    counts: Counter[str] = Counter()
    control_counts: Counter[str] = Counter()

    def category(candidate):
        if candidate.operator.startswith("crossover_mutation:"):
            return "crossover_then_mutation"
        if candidate.operator.startswith("crossover:"):
            return "crossover"
        if candidate.operator.startswith("mutation:parameter:"):
            return "deterministic_mutation"
        if candidate.operator.startswith("mutation:semantic:"):
            return "semantic_mutation"
        return "initial" if candidate.generation == 0 else "injection"

    def production_score(candidate):
        key = category(candidate)
        report = validate_expression(candidate.expression, PRODUCTION_DSL_SPEC)
        assert report.node_count <= 48
        counts[key] += 1
        if candidate.control_expression is not None:
            control_report = validate_expression(
                candidate.control_expression,
                PRODUCTION_DSL_SPEC,
            )
            assert control_report.node_count <= 48
            control_counts[key] += 1
        return score(candidate)

    budget = SearchBudget()
    injection_factory = CodexSeedInjectionFactory(
        seeds,
        injection_cards=injection_cards,
        seed=1701,
    )
    engine = EvolutionEngine(
        evaluator=production_score,
        factory=injection_factory,
        operators=MechanismAwareOperators(seed=1701, max_nodes=48),
        budget=budget,
        mode="plus",
        seed=1701,
    )
    result = engine.run(seeds)
    assert len(result.archive) == 2_384
    injected = [
        item
        for item in result.archive
        if item.candidate.operator
        == "injection:codex_precommitted_adaptive_card"
    ]
    assert len(injected) == 96
    assert len(
        {item.candidate.metadata["source_factor_card_id"] for item in injected}
    ) == 96
    assert all(item.candidate.metadata["target_role_matched"] for item in injected)
    assert sum(
        item.candidate.metadata["repair_fix_matched"] for item in injected
    ) >= 21
    assert injection_factory.remaining_precommitted_count() == 0
    assert engine.terminal_audit["precommitted_injection_cards"] == 96
    assert sum(
        int(summary["adaptive_feedback"]["precommitted_injections"])
        for summary in result.generations
    ) == 96
    assert counts == Counter(
        {
            "initial": 80,
            "deterministic_mutation": 552,
            "semantic_mutation": 552,
            "crossover": 552,
            "crossover_then_mutation": 552,
            "injection": 96,
        }
    )
    assert control_counts == Counter(
        {"crossover": 552, "crossover_then_mutation": 552}
    )


def test_typed_crossover_is_deterministic_and_embeds_both_parent_views():
    project = Path(__file__).resolve().parents[1]
    parents = load_codex_seed_candidates(project)
    # Use two of the larger cards so this exercises compaction, not only the
    # no-op path for already-small expressions.
    left, right = sorted(
        parents,
        key=lambda candidate: expression_complexity(candidate.expression),
        reverse=True,
    )[:2]
    first = MechanismAwareOperators(seed=7, max_nodes=48).crossover(
        left, right, generation=1
    )
    second = MechanismAwareOperators(seed=7, max_nodes=48).crossover(
        left, right, generation=1
    )
    assert canonical_json(first.expression, PRODUCTION_DSL_SPEC) == canonical_json(
        second.expression, PRODUCTION_DSL_SPEC
    )
    assert canonical_json(
        first.control_expression, PRODUCTION_DSL_SPEC
    ) == canonical_json(second.control_expression, PRODUCTION_DSL_SPEC)
    assert first.candidate_id == second.candidate_id
    assert first.parents == (left.candidate_id, right.candidate_id)
    assert first.metadata["parent_expression_hashes"] == [
        left.expression_hash,
        right.expression_hash,
    ]
    branch_hashes = first.metadata["embedded_parent_branch_hashes"]
    assert len(branch_hashes) == 2
    assert all(
        _contains_expression_digest(first.expression, digest)
        for digest in branch_hashes
    )
    assert all(
        _contains_expression_digest(first.control_expression, digest)
        for digest in branch_hashes
    )
    child_report = validate_expression(first.expression, PRODUCTION_DSL_SPEC)
    control_report = validate_expression(
        first.control_expression, PRODUCTION_DSL_SPEC
    )
    assert child_report.node_count <= 48
    assert control_report.node_count <= 48


def test_candidate_id_binds_behavior_lineage_and_control_but_not_metrics_or_prose():
    expression = {
        "op": "return",
        "args": [{"op": "field", "name": "close"}],
        "periods": 5,
    }
    base = Candidate(
        expression=expression,
        agent_id="L4_reversal",
        hypothesis="display hypothesis A",
        mechanism_tags=("reversal", "liquidity"),
        generation=2,
        parents=("parent-a", "parent-b"),
        operator="crossover:orthogonal_interaction",
        control_expression={"op": "cs_rank", "args": [expression]},
        metadata={
            "expected_direction": 1,
            "forecast_horizon": 10,
            "role_lineage": ["L4_reversal", "L3_liquidity"],
            "source": "codex_seed_injection",
            "source_factor_card_id": "card-a",
            "adaptive_feedback": {
                "schema": "adaptive_feedback_v1",
                "source_candidate_id": "parent-a",
                "target_roles": ["L4_reversal"],
                "observation": "display explanation A",
                "cause": "excessive_turnover",
                "fix": "persistence_smoothing",
                "evidence": {"turnover": 0.81},
                "allowed_splits": ["train", "inner_validation"],
                "sealed_feedback_consumed": False,
            },
        },
    )
    base_id = deterministic_candidate_id(base)

    variants = []
    opposite = deepcopy(base)
    opposite.metadata["expected_direction"] = -1
    variants.append(opposite)
    different_roles = deepcopy(base)
    different_roles.metadata["role_lineage"] = ["L5_herding"]
    variants.append(different_roles)
    different_tags = deepcopy(base)
    different_tags.mechanism_tags = ("momentum",)
    variants.append(different_tags)
    different_directive = deepcopy(base)
    different_directive.metadata["adaptive_feedback"]["fix"] = "robust_rank"
    variants.append(different_directive)
    different_source = deepcopy(base)
    different_source.metadata["source_factor_card_id"] = "card-b"
    variants.append(different_source)
    different_control = deepcopy(base)
    different_control.control_expression = {
        "op": "cs_zscore",
        "args": [expression],
    }
    variants.append(different_control)
    assert all(deterministic_candidate_id(item) != base_id for item in variants)

    non_behavioral = deepcopy(base)
    non_behavioral.hypothesis = "different display hypothesis"
    non_behavioral.metadata["adaptive_feedback"]["observation"] = (
        "different display explanation"
    )
    non_behavioral.metadata["adaptive_feedback"]["evidence"] = {
        "turnover": 0.12,
        "floating_metric": 999.0,
    }
    assert deterministic_candidate_id(non_behavioral) == base_id


def test_development_search_fails_closed_when_passed_family_is_too_small():
    def reject(candidate):
        return CandidateScore(
            fitness=-1.0,
            signed_rank_ic=0.0,
            neutral_rank_ic=0.0,
            turnover=1.0,
            worst_window_ic=-1.0,
            style_retention=0.0,
            novelty=0.0,
            passed=False,
            diagnostics={"rejection": "mock_quality_gate"},
        )

    engine = EvolutionEngine(
        evaluator=reject,
        factory=TemplateCandidateFactory(seed=811),
        operators=MechanismAwareOperators(seed=811),
        budget=SearchBudget.tiny(),
        mode="plus",
        seed=811,
    )
    with pytest.raises(
        RuntimeError,
        match=r"eligible passed champions=0.*required final_factors=3",
    ):
        engine.run()
    assert engine.terminal_audit == {
        "status": "failed_insufficient_passed_champions",
        "eligible_passed_count": 0,
        "required_final_factors": 3,
        "rejected_count": 40,
        "rejection_counts": {"mock_quality_gate": 40},
    }


def test_plus_behavioral_champion_underfill_fails_closed_without_relaxation():
    budget = SearchBudget.tiny()
    persisted_audits = []

    def underfilled_selector(eligible, final_factors, policy):
        assert final_factors == 3
        assert policy.maximum_absolute_correlation == 0.85
        return ChampionSelectionResult(
            champions=(eligible[0], eligible[1]),
            audit={
                "status": "failed_insufficient_behaviorally_distinct_champions",
                "correlation_conflict_candidates": len(eligible) - 2,
                "threshold_relaxed": False,
                "sealed_data_consumed": False,
            },
        )

    engine = EvolutionEngine(
        evaluator=lambda candidate: score(candidate),
        factory=TemplateCandidateFactory(seed=355),
        operators=MechanismAwareOperators(seed=355),
        budget=budget,
        mode="plus",
        seed=355,
        champion_selector=underfilled_selector,
        behavioral_champion_policy=BehavioralChampionPolicy(),
        on_terminal_audit=persisted_audits.append,
    )
    with pytest.raises(RuntimeError, match="behavior gate selected 2 distinct"):
        engine.run()
    assert engine.terminal_audit["status"] == (
        "failed_insufficient_behaviorally_distinct_champions"
    )
    assert engine.terminal_audit["selected_distinct_count"] == 2
    assert engine.terminal_audit["behavioral_selection"][
        "threshold_relaxed"
    ] is False
    assert persisted_audits == [engine.terminal_audit]


def test_plus_behavioral_selector_exception_is_persisted_before_failure():
    persisted_audits = []

    def broken_selector(eligible, final_factors, policy):
        raise ValueError("non-finite behavior")

    engine = EvolutionEngine(
        evaluator=lambda candidate: score(candidate),
        factory=TemplateCandidateFactory(seed=356),
        operators=MechanismAwareOperators(seed=356),
        budget=SearchBudget.tiny(),
        mode="plus",
        seed=356,
        champion_selector=broken_selector,
        behavioral_champion_policy=BehavioralChampionPolicy(),
        on_terminal_audit=persisted_audits.append,
    )

    with pytest.raises(RuntimeError, match="selector raised ValueError"):
        engine.run()

    assert engine.terminal_audit["status"] == "failed_behavioral_champion_selector"
    assert engine.terminal_audit["error_type"] == "ValueError"
    assert persisted_audits == [engine.terminal_audit]


def test_commutative_semantic_dedup_refills_the_same_operator_budget():
    left_term = {
        "op": "return",
        "args": [{"op": "field", "name": "close"}],
        "periods": 5,
    }
    right_term = {
        "op": "return",
        "args": [{"op": "field", "name": "open"}],
        "periods": 5,
    }
    forward = {"op": "add", "args": [left_term, right_term]}
    swapped = {"op": "add", "args": [right_term, left_term]}
    assert expression_digest(forward) == expression_digest(swapped)

    common = dict(
        agent_id="L1_test",
        hypothesis="same behavior",
        mechanism_tags=("trend",),
        generation=1,
        parents=("a", "b"),
        operator="crossover:test",
        metadata={"expected_direction": 1, "role_lineage": ["L1_test"]},
    )
    assert deterministic_candidate_id(
        Candidate(expression=forward, **common)
    ) == deterministic_candidate_id(Candidate(expression=swapped, **common))

    class SwappedDuplicateOperators(MechanismAwareOperators):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pure_crossover_calls = 0

        def crossover(self, left, right, generation, *, feedback_fix=None):
            child = super().crossover(
                left,
                right,
                generation,
                feedback_fix=feedback_fix,
            )
            self.pure_crossover_calls += 1
            if self.pure_crossover_calls == 1:
                child.expression = deepcopy(forward)
            elif self.pure_crossover_calls == 2:
                child.expression = deepcopy(swapped)
            child.candidate_id = deterministic_candidate_id(child)
            return child

    budget = SearchBudget.tiny()
    result = EvolutionEngine(
        evaluator=score,
        factory=TemplateCandidateFactory(seed=919),
        operators=SwappedDuplicateOperators(seed=919),
        budget=budget,
        mode="plus",
        seed=919,
    ).run()
    assert len(result.archive) == (
        budget.initial_population
        + budget.generations * budget.children_per_generation
    )
    for summary in result.generations:
        generation = int(summary["generation"])
        injection = (
            budget.injection_count
            if generation % budget.injection_every == 0
            else 0
        )
        evolved = budget.children_per_generation - injection
        base, remainder = divmod(evolved, 4)
        expected_crossovers = base + (2 < remainder)
        assert summary["crossovers"] == expected_crossovers


def _feedback_score(
    *,
    turnover=0.1,
    style_retention=0.9,
    worst_window_ic=0.01,
    novelty=0.8,
    control_delta=0.01,
    passed=True,
    diagnostics=None,
):
    return CandidateScore(
        fitness=0.02 if passed else -1.0,
        signed_rank_ic=0.02 if passed else 0.0,
        neutral_rank_ic=0.018 if passed else 0.0,
        turnover=turnover,
        worst_window_ic=worst_window_ic,
        style_retention=style_retention,
        novelty=novelty,
        control_delta=control_delta,
        passed=passed,
        diagnostics=diagnostics or {"coverage": 0.95, "train_raw_rank_ic": 0.02},
    )


def test_plus_and_faithful_have_equal_candidate_budget_but_only_plus_adapts():
    budget = SearchBudget.tiny()
    plus = EvolutionEngine(
        evaluator=lambda candidate: _feedback_score(),
        factory=TemplateCandidateFactory(seed=410),
        operators=MechanismAwareOperators(seed=410),
        budget=budget,
        mode="plus",
        seed=410,
    ).run()
    faithful = EvolutionEngine(
        evaluator=lambda candidate: _feedback_score(),
        factory=TemplateCandidateFactory(seed=410),
        operators=MechanismAwareOperators(seed=410),
        budget=budget,
        mode="faithful",
        seed=410,
    ).run()
    expected = budget.initial_population + budget.generations * budget.children_per_generation
    assert len(plus.archive) == len(faithful.archive) == expected
    assert all(item["children"] == budget.children_per_generation for item in plus.generations)
    assert all(item["children"] == budget.children_per_generation for item in faithful.generations)
    assert plus.adaptive_feedback["sealed_feedback_consumed"] is False
    assert faithful.adaptive_feedback == {}
    assert any(
        "adaptive_feedback" in item.candidate.metadata
        for item in plus.archive
        if item.candidate.generation > 0
    )
    assert all(
        "adaptive_feedback" not in item.candidate.metadata
        for item in faithful.archive
    )
    expected_faithful = sorted(
        faithful.archive,
        key=lambda item: item.score.fitness,
        reverse=True,
    )[: budget.final_factors]
    assert [item.candidate.candidate_id for item in faithful.champions] == [
        item.candidate.candidate_id for item in expected_faithful
    ]
    assert faithful.champion_selection["mode"] == "faithful_fitness_rank"


def test_feedback_reads_only_named_development_diagnostics_never_sealed_fields():
    class GuardedDiagnostics:
        def __init__(self):
            self.accessed = []
            self.values = {
                "coverage": 0.96,
                "train_raw_rank_ic": 0.02,
                "rejection": None,
                "sealed_rank_ic": 0.99,
            }

        def get(self, key, default=None):
            if str(key).startswith("sealed"):
                raise AssertionError("sealed diagnostic was accessed")
            self.accessed.append(key)
            return self.values.get(key, default)

        def __iter__(self):
            raise AssertionError("diagnostics must not be enumerated or copied")

    diagnostics = GuardedDiagnostics()
    result = EvolutionEngine(
        evaluator=lambda candidate: _feedback_score(diagnostics=diagnostics),
        factory=TemplateCandidateFactory(seed=420),
        operators=MechanismAwareOperators(seed=420),
        budget=SearchBudget.tiny(),
        mode="plus",
        seed=420,
    ).run()
    assert set(diagnostics.accessed) <= {
        "rejection",
        "coverage",
        "train_raw_rank_ic",
    }
    assert result.adaptive_feedback["diagnostic_allowlist"] == [
        "coverage",
        "rejection",
        "train_raw_rank_ic",
    ]
    assert "sealed_rank_ic" not in str(result.adaptive_feedback)


def test_sanitized_feedback_changes_the_next_generation_repair_mechanism():
    budget = SearchBudget.tiny()
    turnover_result = EvolutionEngine(
        evaluator=lambda candidate: _feedback_score(turnover=0.9),
        factory=TemplateCandidateFactory(seed=430),
        operators=MechanismAwareOperators(seed=430),
        budget=budget,
        mode="plus",
        seed=430,
    ).run()
    style_result = EvolutionEngine(
        evaluator=lambda candidate: _feedback_score(style_retention=0.1),
        factory=TemplateCandidateFactory(seed=430),
        operators=MechanismAwareOperators(seed=430),
        budget=budget,
        mode="plus",
        seed=430,
    ).run()
    turnover_fixes = turnover_result.generations[0]["adaptive_feedback"]["fixes"]
    style_fixes = style_result.generations[0]["adaptive_feedback"]["fixes"]
    assert turnover_fixes["persistence_smoothing"] > 0
    assert turnover_fixes["turnover_guarded_crossover"] > 0
    assert style_fixes["style_robustification"] > 0
    assert style_fixes["retention_guarded_crossover"] > 0
    turnover_hashes = {
        item.candidate.expression_hash
        for item in turnover_result.archive
        if item.candidate.generation == 1
    }
    style_hashes = {
        item.candidate.expression_hash
        for item in style_result.archive
        if item.candidate.generation == 1
    }
    assert turnover_hashes != style_hashes


def test_failure_memory_survives_generations_and_drives_repair_lineage():
    memory = AdaptiveFeedbackMemory()

    def failure_aware(candidate):
        if candidate.generation == 0 and candidate.agent_id == "L4_price_reversal":
            return _feedback_score(
                passed=False,
                turnover=1.0,
                style_retention=0.0,
                worst_window_ic=-1.0,
                novelty=0.0,
                control_delta=None,
                diagnostics={"rejection": "dsl_or_runtime:stage_violation"},
            )
        return _feedback_score()

    result = EvolutionEngine(
        evaluator=failure_aware,
        factory=TemplateCandidateFactory(seed=440),
        operators=MechanismAwareOperators(seed=440),
        budget=SearchBudget.tiny(),
        mode="plus",
        seed=440,
        feedback_memory=memory,
    ).run()
    assert result.adaptive_feedback["remembered_failures"] > 0
    assert result.adaptive_feedback["causes"]["invalid_or_fragile_structure"] > 0
    assert all(summary["remembered_failures"] > 0 for summary in result.generations)
    repaired = [
        item
        for item in result.archive
        if item.candidate.generation > 0
        and item.candidate.metadata.get("adaptive_feedback", {}).get("cause")
        == "invalid_or_fragile_structure"
    ]
    assert repaired
    assert any(
        item.candidate.metadata["adaptive_feedback"]["fix"]
        in {"conservative_repair", "failure_avoidance_crossover"}
        for item in repaired
    )


def test_21_role_memory_and_fresh_injection_repair_missing_islands():
    seeds = TemplateCandidateFactory(seed=450).generate(21)
    for position, candidate in enumerate(seeds):
        role = f"L{position // 3 + 1}_role_{position + 1:02d}"
        candidate.agent_id = role
        candidate.metadata["role_lineage"] = [role]
    budget = SearchBudget(
        initial_population=21,
        parent_pool=21,
        children_per_generation=8,
        generations=2,
        subcycles=1,
        generations_per_subcycle=2,
        injection_every=2,
        injection_count=8,
        elite_carry=2,
        final_factors=3,
    )
    result = EvolutionEngine(
        evaluator=lambda candidate: _feedback_score(),
        factory=CodexSeedInjectionFactory(seeds, seed=451),
        operators=MechanismAwareOperators(seed=451),
        budget=budget,
        mode="plus",
        seed=451,
    ).run(seeds)
    assert result.adaptive_feedback["registered_role_count"] == 21
    assert len(result.generations[0]["adaptive_feedback"]["parent_role_coverage"]) == 21
    injections = [
        item.candidate
        for item in result.archive
        if item.candidate.generation == 2
        and item.candidate.operator.startswith("injection:")
    ]
    assert len(injections) == 8
    targets = [
        tuple(item.metadata["adaptive_feedback"]["target_roles"])
        for item in injections
    ]
    assert len(set(targets)) == 8
    assert all(
        set(target).issubset(candidate_roles(candidate))
        for target, candidate in zip(targets, injections, strict=True)
    )
    assert all(
        item.metadata["adaptive_feedback"]["sealed_feedback_consumed"] is False
        for item in injections
    )


def test_plus_feedback_schedule_and_lineage_are_deterministic_for_fixed_seed():
    def run_once():
        return EvolutionEngine(
            evaluator=lambda candidate: _feedback_score(
                turnover=0.55,
                novelty=0.2 if candidate.generation % 2 else 0.8,
            ),
            factory=TemplateCandidateFactory(seed=460),
            operators=MechanismAwareOperators(seed=460),
            budget=SearchBudget.tiny(),
            mode="plus",
            seed=460,
        ).run()

    first = run_once()
    second = run_once()
    assert [
        (
            item.candidate.candidate_id,
            item.candidate.expression_hash,
            item.candidate.operator,
            item.candidate.parents,
        )
        for item in first.archive
    ] == [
        (
            item.candidate.candidate_id,
            item.candidate.expression_hash,
            item.candidate.operator,
            item.candidate.parents,
        )
        for item in second.archive
    ]
    assert [
        item["adaptive_feedback"]["directive_sha256"]
        for item in first.generations
    ] == [
        item["adaptive_feedback"]["directive_sha256"]
        for item in second.generations
    ]
