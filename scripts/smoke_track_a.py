from __future__ import annotations

import argparse
import json
from pathlib import Path

from cogalpha_plus.evolution import SearchBudget
from cogalpha_plus.hs300_research import (
    load_codex_injection_candidates,
    load_codex_seed_candidates,
)
from cogalpha_plus.runner import run_experiment
from cogalpha_plus.track_a import load_track_a_development, split_identity
from cogalpha_plus.terminal_labels import load_terminal_label_bridges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("smoke output directory must be new or empty")
    bundle = load_track_a_development(project)
    budget = SearchBudget.tiny()
    seeds = load_codex_seed_candidates(project, horizon=10)[: budget.initial_population]
    required_injections = sum(
        budget.injection_count
        for generation in range(1, budget.generations + 1)
        if generation % budget.injection_every == 0
    )
    injections = load_codex_injection_candidates(project, horizon=10)[
        :required_injections
    ]
    bridges, bridge_identity = load_terminal_label_bridges(
        project / bundle.protocol.raw["label"]["terminal_bridge_catalog"]
    )
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
        seed=17,
        reveal_sealed=False,
        bootstrap_samples=20,
        evaluation_workers=args.workers,
        model_version="codex-track-a-smoke",
        prompt_version="track-a-sanitized-feedback-v1",
        initial_candidates=seeds,
        injection_candidates=injections,
        forecast_price_field="open",
        terminal_label_bridges=bridges,
        terminal_label_bridge_identity=bridge_identity,
        development_handoff_policy={
            "minimum_daily_label_coverage": 1.0,
            "allow_label_row_drop": False,
            "min_names": 20,
        },
        pareto_selection_policy=bundle.protocol.pareto_policy,
        operational_locators={"track_a_config": str(bundle.protocol.config_path)},
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "archive": len(result.archive),
                "champions": len(result.champions),
                "generations": len(result.generations),
                "split_identity": split_identity(bundle.split),
                "manifest_hash": summary["manifest_hash"],
                "sealed_revealed": summary["sealed_revealed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
