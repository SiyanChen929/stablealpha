from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback

import pandas as pd

from cogalpha_plus.hs300_research import load_frozen_hs300_panel
from cogalpha_plus.terminal_labels import load_terminal_label_bridges
from cogalpha_plus.track_a import load_track_a_protocol, make_track_a_split, split_identity
from cogalpha_plus.track_a_artifacts import load_track_a_development_freeze
from cogalpha_plus.track_a_models import (
    build_track_a_open_labels,
    run_track_a_models,
    score_track_a_predictions,
)
from cogalpha_plus.track_a_portfolio import (
    load_frozen_csi300_open,
    run_paper_top50_drop5,
)
from cogalpha_plus.track_a_report import write_track_a_report
from cogalpha_plus.validation import SealedGate


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-freeze", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.development_freeze.resolve()
    freeze, champions, artifacts = load_track_a_development_freeze(freeze_path)
    project = Path(str(freeze["project_dir"])).resolve()
    run_dir = freeze_path.parent
    protocol = load_track_a_protocol(project)
    if protocol.config_sha256 != freeze["track_a_config_sha256"]:
        raise ValueError("Track A config differs from development freeze")

    registry_path = run_dir / "sealed_reveal_registry.json"
    started = {
        "schema": "track_a_one_time_reveal_registry_v1",
        "state": "STARTED_TEST_DATA_CAPABILITY_OPENED",
        "development_freeze_sha256": freeze["freeze_sha256"],
        "sealed_results_may_not_feed_back": True,
    }
    with registry_path.open("x", encoding="utf-8") as handle:
        json.dump(started, handle, indent=2, sort_keys=True)
        handle.write("\n")

    try:
        full = load_frozen_hs300_panel(protocol.panel_dir)
        if full.manifest_sha256 != freeze["panel_manifest_sha256"]:
            raise ValueError("full panel manifest differs from development freeze")
        split = make_track_a_split(full.trading_calendar)
        if split_identity(split) != freeze["split_identity"]:
            raise ValueError("sealed split differs from development freeze")
        bridges, bridge_identity = load_terminal_label_bridges(
            project / protocol.raw["label"]["terminal_bridge_catalog"]
        )
        gate = SealedGate(
            full.eligible_panel,
            split,
            feature_tape=full.feature_tape,
            trading_calendar=full.trading_calendar,
            research_universe_identity=full.research_universe_identity,
            trading_calendar_identity=full.trading_calendar_identity,
            terminal_label_bridges=bridges,
            terminal_label_bridge_identity=bridge_identity,
            forecast_price_field="open",
            min_names=20,
        )
        factor_results = gate.reveal(champions, artifacts)
        models = run_track_a_models(
            champions,
            eligible_panel=full.eligible_panel,
            feature_tape=full.feature_tape,
            trading_calendar=full.trading_calendar,
            split=split,
            ridge_alpha=float(protocol.raw["models"]["ridge"]["alpha"]),
            lightgbm_parameters=protocol.raw["models"]["lightgbm"],
            bootstrap_samples=500,
            terminal_label_bridges=bridges,
            terminal_label_bridge_identity=bridge_identity,
        )
        for name, frozen_metric in freeze["validation_model_metrics"].items():
            observed = models.validation_metrics[name]
            if abs(float(frozen_metric["mean"]) - float(observed["mean"])) > 1e-12:
                raise RuntimeError("validation model metric changed after sealed materialization")
        labels = build_track_a_open_labels(
            eligible_panel=full.eligible_panel,
            feature_tape=full.feature_tape,
            trading_calendar=full.trading_calendar,
            split=split,
            terminal_label_bridges=bridges,
            terminal_label_bridge_identity=bridge_identity,
        )
        sealed_model_metrics = score_track_a_predictions(
            models.predictions,
            labels,
            bootstrap_samples=500,
        )
        benchmark, benchmark_identity = load_frozen_csi300_open(full.manifest)
        adjusted_open = full.feature_tape["adj_open"]
        portfolio_results = {}
        portfolio_dir = run_dir / "sealed_portfolios"
        portfolio_dir.mkdir(exist_ok=False)
        for name, prediction in models.predictions.items():
            result = run_paper_top50_drop5(
                prediction,
                adjusted_open=adjusted_open,
                trading_calendar=full.trading_calendar,
                benchmark_open=benchmark,
                initial_capital=float(protocol.raw["portfolio"]["initial_capital_cny"]),
                open_cost_rate=float(protocol.raw["portfolio"]["open_cost_rate"]),
                close_cost_rate=float(protocol.raw["portfolio"]["close_cost_rate"]),
                minimum_commission=float(protocol.raw["portfolio"]["minimum_commission_cny"]),
            )
            result.daily.to_parquet(portfolio_dir / f"{name}_daily.parquet")
            portfolio_results[name] = dict(result.statistics)
        output = {
            "schema": "track_a_sealed_result_v1",
            "status": "SEALED_REVEAL_COMPLETE_NO_FEEDBACK_ALLOWED",
            "claim_scope": "paper_compatible_reproduction_not_exact_replication",
            "development_freeze_sha256": freeze["freeze_sha256"],
            "factor_results": factor_results,
            "validation_model_metrics_recomputed": models.validation_metrics,
            "sealed_model_metrics": sealed_model_metrics,
            "rolling_refit_audit": list(models.rolling_audit),
            "factor_matrix_identity": models.feature_identity,
            "benchmark_identity": benchmark_identity,
            "portfolio_results": portfolio_results,
            "primary_result": {
                "backend": "lightgbm",
                "rank_ic": sealed_model_metrics["lightgbm"],
                "portfolio": portfolio_results["lightgbm"],
            },
            "research_validity": "NOT VALIDATED",
            "research_validity_reason": (
                "single historical outer replay on a BaoStock research-proxy snapshot; "
                "not prospective paper trading or capital-grade corporate-action accounting"
            ),
            "sealed_results_used_for_selection": False,
        }
        result_path = run_dir / "sealed_result.json"
        with result_path.open("x", encoding="utf-8") as handle:
            json.dump(
                output,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
        report_path = write_track_a_report(run_dir, output)
        completed = {
            **started,
            "state": "COMPLETED_AND_RETIRED",
            "sealed_result": str(result_path),
            "report": str(report_path),
        }
        registry_path.write_text(
            json.dumps(completed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": completed["state"], "result": str(result_path)}))
    except BaseException as exc:
        failed = {
            **started,
            "state": "FAILED_AFTER_TEST_DATA_CAPABILITY_OPENED_RETIRED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        registry_path.write_text(
            json.dumps(failed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
