from __future__ import annotations

import argparse
import json
from pathlib import Path

from cogalpha_plus.evolution import SearchBudget
from cogalpha_plus.generation_exchange import FileGenerationReviewer
from cogalpha_plus.hs300_research import (
    load_codex_injection_candidates,
    load_codex_seed_candidates,
)
from cogalpha_plus.runner import run_experiment
from cogalpha_plus.terminal_labels import load_terminal_label_bridges
from cogalpha_plus.track_a import load_track_a_development
from cogalpha_plus.track_a_artifacts import (
    freeze_track_a_development,
    track_a_implementation_identity,
)
from cogalpha_plus.track_a_models import (
    build_track_a_factor_matrix,
    build_track_a_open_labels,
    fit_validation_controls,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    args = parser.parse_args()
    if args.workers != 3:
        raise ValueError("Track A frozen contract requires exactly three workers")
    project = args.project_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Track A output directory must be new or empty")

    bundle = load_track_a_development(project)
    protocol = bundle.protocol
    budget = SearchBudget()
    seeds = load_codex_seed_candidates(project, horizon=10)
    injections = load_codex_injection_candidates(project, horizon=10)
    if len(seeds) != 80 or len(injections) != 96:
        raise ValueError("Track A requires the frozen 80/96 LLM card libraries")
    roles = {
        role
        for candidate in seeds
        for role in candidate.metadata.get("role_lineage", [candidate.agent_id])
    }
    if len(roles) != 21:
        raise ValueError("Track A seed library does not cover all 21 roles")
    bridges, bridge_identity = load_terminal_label_bridges(
        project / protocol.raw["label"]["terminal_bridge_catalog"]
    )
    implementation_identity = track_a_implementation_identity(project)
    reviewer = FileGenerationReviewer(output / "exchange")
    result, summary = run_experiment(
        full_panel=bundle.frozen.eligible_panel,
        feature_tape=bundle.frozen.feature_tape,
        source_member_panel=bundle.frozen.panel,
        trading_calendar=bundle.frozen.trading_calendar,
        trading_calendar_identity=bundle.frozen.trading_calendar_identity,
        research_universe_identity=bundle.frozen.research_universe_identity,
        source_lineage_identity=bundle.frozen.source_lineage_identity,
        source_snapshot_identity=bundle.frozen.source_snapshot_identity,
        split=bundle.split,
        output_dir=output,
        budget=budget,
        profile="plus",
        seed=int(protocol.raw["reproducibility"]["seed"]),
        reveal_sealed=False,
        bootstrap_samples=args.bootstrap_samples,
        evaluation_workers=args.workers,
        model_version="codex-track-a-interactive-quality-first",
        prompt_version="track-a-sanitized-feedback-v1",
        initial_candidates=seeds,
        injection_candidates=injections,
        terminal_label_bridges=bridges,
        terminal_label_bridge_identity=bridge_identity,
        forecast_price_field="open",
        pareto_selection_policy=protocol.pareto_policy,
        generation_reviewer=reviewer,
        generation_reviewer_identity={
            "schema": "codex_generation_reviewer_v1",
            "review_required_every_generation": True,
            "request_schema": "codex_generation_request_v1",
            "response_schema": "codex_generation_response_v1",
            "sealed_feedback": "forbidden",
            "reviewer_model": "codex-current-thread",
        },
        development_handoff_policy={
            "minimum_daily_label_coverage": 1.0,
            "allow_label_row_drop": False,
            "min_names": 20,
        },
        implementation_identity=implementation_identity,
        operational_locators={
            "track_a_config": str(protocol.config_path),
            "generation_exchange": str(output / "exchange"),
        },
    )

    features, feature_identity = build_track_a_factor_matrix(
        result.champions,
        eligible_panel=bundle.frozen.eligible_panel,
        feature_tape=bundle.frozen.feature_tape,
    )
    labels = build_track_a_open_labels(
        eligible_panel=bundle.frozen.eligible_panel,
        feature_tape=bundle.frozen.feature_tape,
        trading_calendar=bundle.frozen.trading_calendar,
        split=bundle.split,
        terminal_label_bridges=bridges,
        terminal_label_bridge_identity=bridge_identity,
    )
    _, validation_metrics = fit_validation_controls(
        features,
        labels,
        bundle.split,
        ridge_alpha=float(protocol.raw["models"]["ridge"]["alpha"]),
        lightgbm_parameters=protocol.raw["models"]["lightgbm"],
        bootstrap_samples=500,
    )
    validation_path = output / "validation_model_metrics.json"
    validation_path.write_text(
        json.dumps(validation_metrics, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    freeze_path = freeze_track_a_development(
        project_dir=project,
        run_dir=output,
        bundle=bundle,
        champions=result.champions,
        validation_model_metrics=validation_metrics,
        factor_matrix_identity=feature_identity,
        expected_implementation_identity=implementation_identity,
    )
    print(
        json.dumps(
            {
                "status": "DEVELOPMENT_FROZEN_TEST_UNSEEN",
                "evaluated_candidates": summary["evaluated_candidates"],
                "generations": len(result.generations),
                "champions": len(result.champions),
                "development_freeze": str(freeze_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
