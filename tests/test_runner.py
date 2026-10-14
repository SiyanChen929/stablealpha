import json

import pytest

import cogalpha_plus.runner as runner_module
from cogalpha_plus.evolution import (
    AdaptiveFeedbackPolicy,
    BehavioralChampionPolicy,
    SearchBudget,
)
from cogalpha_plus.manifest import RunManifest, hash_json
from cogalpha_plus.runner import (
    experiment_config_payload,
    run_experiment,
    score_payload,
)
from cogalpha_plus.synthetic import make_synthetic_market
from cogalpha_plus.validation import make_split_plan


def test_recorded_micro_experiment_is_auditable_and_binds_runtime_policy(
    tmp_path, monkeypatch
):
    panel = make_synthetic_market(days=220, symbols=24, seed=31)
    split = make_split_plan(panel, label_horizon=2, train_fraction=0.50, validation_fraction=0.20)
    budget = SearchBudget(
        initial_population=4,
        parent_pool=2,
        children_per_generation=4,
        generations=2,
        subcycles=1,
        generations_per_subcycle=2,
        injection_every=2,
        injection_count=1,
        elite_carry=1,
        final_factors=2,
    )
    policy = AdaptiveFeedbackPolicy(minimum_coverage=0.91)
    captured = {}
    original_engine = runner_module.EvolutionEngine

    class CapturingEngine(original_engine):
        def __init__(self, **kwargs):
            captured["feedback_memory"] = kwargs.get("feedback_memory")
            super().__init__(**kwargs)

    monkeypatch.setattr(runner_module, "EvolutionEngine", CapturingEngine)
    _, summary = run_experiment(
        full_panel=panel,
        split=split,
        output_dir=tmp_path / "run",
        budget=budget,
        bootstrap_samples=10,
        seed=31,
            adaptive_feedback_policy=policy,
            allow_test_calendar_from_panel=True,
        )
    assert summary["evaluated_candidates"] == 12
    assert summary["registry"]["total"] == 12
    assert summary["ledger"]["valid"]
    assert not summary["sealed_revealed"]
    assert (tmp_path / "run" / "manifest.json").exists()
    assert (tmp_path / "run" / "summary.json").exists()
    assert captured["feedback_memory"].policy is policy
    manifest = RunManifest.load(tmp_path / "run" / "manifest.json")
    source_identity = manifest.data_snapshot["source_snapshot_identity"]
    assert source_identity["identity_hash"] == hash_json(
        {
            key: value
            for key, value in source_identity.items()
            if key != "identity_hash"
        }
    )
    resolved = experiment_config_payload(
        profile="plus",
        split=split,
        budget=budget,
        bootstrap_samples=10,
        adaptive_feedback_policy=policy,
        research_universe_identity=manifest.data_snapshot[
            "research_universe_identity"
        ],
        trading_calendar_identity=manifest.data_snapshot[
            "trading_calendar_identity"
        ],
        source_lineage_identity=manifest.data_snapshot[
            "source_lineage_identity"
        ],
        source_snapshot_identity=manifest.data_snapshot[
            "source_snapshot_identity"
        ],
    )
    assert manifest.config_hash == hash_json(resolved)
    assert resolved["adaptive_feedback_policy"]["minimum_coverage"] == 0.91
    assert resolved["behavioral_champion_policy"] == {
        "maximum_absolute_correlation": 0.85,
        "initial_top_pool_size": 200,
    }
    assert summary["champion_selection"]["sealed_data_consumed"] is False
    assert all(
        "maximum_absolute_behavioral_correlation" in champion
        and "nearest_champion_candidate_id" in champion
        for champion in summary["champions"]
    )


def test_faithful_profile_has_no_plus_feedback_policy():
    panel = make_synthetic_market(days=220, symbols=20, seed=7)
    split = make_split_plan(
        panel,
        label_horizon=2,
        train_fraction=0.50,
        validation_fraction=0.20,
    )
    payload = experiment_config_payload(
        profile="faithful",
        split=split,
        budget=SearchBudget.tiny(),
        bootstrap_samples=10,
    )
    assert payload["adaptive_feedback_policy"] is None
    assert payload["behavioral_champion_policy"] is None
    with pytest.raises(ValueError, match="faithful profile cannot receive"):
        experiment_config_payload(
            profile="faithful",
            split=split,
            budget=SearchBudget.tiny(),
            bootstrap_samples=10,
            adaptive_feedback_policy=AdaptiveFeedbackPolicy(),
        )
    with pytest.raises(ValueError, match="faithful profile cannot receive"):
        experiment_config_payload(
            profile="faithful",
            split=split,
            budget=SearchBudget.tiny(),
            bootstrap_samples=10,
            behavioral_champion_policy=BehavioralChampionPolicy(),
        )


def test_three_worker_evaluation_matches_serial_candidate_by_candidate(tmp_path):
    panel = make_synthetic_market(days=240, symbols=24, seed=43)
    split = make_split_plan(
        panel,
        label_horizon=2,
        train_fraction=0.50,
        validation_fraction=0.20,
    )
    budget = SearchBudget(
        initial_population=4,
        parent_pool=2,
        children_per_generation=4,
        generations=2,
        subcycles=1,
        generations_per_subcycle=2,
        injection_every=2,
        injection_count=0,
        elite_carry=1,
        final_factors=1,
    )
    serial, _ = run_experiment(
        full_panel=panel,
        split=split,
        output_dir=tmp_path / "serial",
        budget=budget,
        bootstrap_samples=20,
        seed=43,
        evaluation_workers=1,
        allow_test_calendar_from_panel=True,
    )
    parallel, _ = run_experiment(
        full_panel=panel,
        split=split,
        output_dir=tmp_path / "parallel",
        budget=budget,
        bootstrap_samples=20,
        seed=43,
        evaluation_workers=3,
        allow_test_calendar_from_panel=True,
    )

    serial_candidates = [
        (item.candidate.to_dict(), score_payload(item.score))
        for item in serial.archive
    ]
    parallel_candidates = [
        (item.candidate.to_dict(), score_payload(item.score))
        for item in parallel.archive
    ]
    assert parallel_candidates == serial_candidates
    assert parallel.generations == serial.generations
    assert parallel.adaptive_feedback == serial.adaptive_feedback
    assert parallel.champion_selection == serial.champion_selection
    assert [item.candidate.candidate_id for item in parallel.champions] == [
        item.candidate.candidate_id for item in serial.champions
    ]

    def ledger_candidates(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return [
            (
                row["payload"]["candidate_id"],
                row["payload"]["generation"],
                row["payload"]["operator"],
                row["payload"]["score"],
            )
            for row in rows
            if row["event_type"] == "candidate_evaluated"
        ]

    assert ledger_candidates(
        tmp_path / "parallel" / "ledger" / "trials.jsonl"
    ) == ledger_candidates(tmp_path / "serial" / "ledger" / "trials.jsonl")
