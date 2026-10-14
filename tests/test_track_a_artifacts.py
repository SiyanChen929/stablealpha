import json
from types import SimpleNamespace

import pytest

from cogalpha_plus.evolution import CandidateScore, EvaluatedCandidate
from cogalpha_plus.hs300_research import load_codex_seed_candidates
from cogalpha_plus.track_a import make_track_a_split
from cogalpha_plus.track_a_artifacts import (
    freeze_track_a_development,
    load_track_a_development_freeze,
    track_a_implementation_identity,
)


def test_track_a_development_freeze_round_trip_is_content_addressed(tmp_path):
    project = __import__("pathlib").Path(__file__).resolve().parents[1]
    run = tmp_path / "run"
    (run / "ledger").mkdir(parents=True)
    (run / "manifest.json").write_text("{}\n")
    (run / "ledger" / "trials.jsonl").write_text("{}\n")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "sealed_revealed": False,
                "evaluated_candidates": 2384,
                "ledger": {"head_hash": "a" * 64},
            }
        )
    )
    for generation in range(1, 25):
        for kind in ("requests", "responses", "transcripts"):
            path = run / "exchange" / kind / f"generation_{generation:02d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"generation": generation, "kind": kind}))
    candidates = load_codex_seed_candidates(project, horizon=10)[:20]
    champions = []
    for candidate in candidates:
        champions.append(
            EvaluatedCandidate(
                candidate,
                CandidateScore(
                    fitness=0.01,
                    signed_rank_ic=0.02,
                    neutral_rank_ic=0.01,
                    turnover=0.2,
                    worst_window_ic=0.001,
                    style_retention=0.8,
                    novelty=0.7,
                    passed=True,
                    diagnostics={"train_raw_rank_ic": 0.015},
                ),
            )
        )
    calendar = __import__("pandas").bdate_range("2011-01-03", "2025-01-20")
    split = make_track_a_split(calendar)
    protocol = SimpleNamespace(
        config_sha256="b" * 64,
        panel_manifest_sha256="c" * 64,
    )
    bundle = SimpleNamespace(protocol=protocol, split=split)
    target = freeze_track_a_development(
        project_dir=project,
        run_dir=run,
        bundle=bundle,
        champions=champions,
        validation_model_metrics={"lightgbm": {"mean": 0.01}},
        factor_matrix_identity={"sha256": "d" * 64},
    )
    frozen, loaded, artifacts = load_track_a_development_freeze(target)
    assert frozen["status"] == "DEVELOPMENT_FROZEN_TEST_UNSEEN"
    assert len(loaded) == len(artifacts) == 20
    assert [item.candidate.candidate_id for item in loaded] == [
        item.candidate.candidate_id for item in champions
    ]
    assert frozen["implementation_identity"] == track_a_implementation_identity(project)

    with pytest.raises(ValueError, match="changed during development search"):
        freeze_track_a_development(
            project_dir=project,
            run_dir=run,
            bundle=bundle,
            champions=champions,
            validation_model_metrics={"lightgbm": {"mean": 0.01}},
            factor_matrix_identity={"sha256": "d" * 64},
            expected_implementation_identity={"sha256": "wrong"},
        )
