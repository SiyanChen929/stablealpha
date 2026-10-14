#!/usr/bin/env python3
"""Freeze, activate, prepare and score the HS300 research portfolio replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

import cogalpha_plus.hs300_research as hs300
from cogalpha_plus.dsl import evaluate_expression
from cogalpha_plus.manifest import canonical_json, hash_json, utc_now
from cogalpha_plus.metrics import cross_sectional_neutralize, forward_return
from cogalpha_plus.portfolio_replay import (
    TargetBook,
    load_portfolio_replay_policy,
    prepare_target_book,
    score_target_book,
    target_book_hash,
)
from cogalpha_plus.research_proxy import (
    SealedCompositeClaimCapability,
    load_frozen_composite_family,
)
from cogalpha_plus.terminal_labels import (
    apply_terminal_label_bridges,
    load_terminal_label_bridge_catalog_snapshot,
    validate_terminal_label_bridge_identity,
)
from cogalpha_plus.validation import (
    PRODUCTION_DSL_SPEC,
    build_cross_sectional_mask,
    build_immediate_executable_labels,
    build_research_feature_frame,
    build_style_exposures,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT / "configs" / "hs300_portfolio_replay_v1.yaml"
DEFAULT_PLAN = PROJECT / "runs" / "hs300_portfolio_plan_v1.json"
BRIDGE_CATALOG_RELATIVE = "configs/hs300_terminal_label_bridges.yaml"
BRIDGE_CATALOG = PROJECT / BRIDGE_CATALOG_RELATIVE
PLAN_FILES = (
    "configs/hs300_portfolio_replay_v1.yaml",
    BRIDGE_CATALOG_RELATIVE,
    "scripts/run_hs300_portfolio_replay.py",
    "src/cogalpha_plus/portfolio_replay.py",
    "src/cogalpha_plus/portfolio.py",
)


def bridge_catalog_snapshot() -> tuple[tuple[object, ...], dict[str, object]]:
    events, identity, raw_identity = load_terminal_label_bridge_catalog_snapshot(
        BRIDGE_CATALOG
    )
    validated = validate_terminal_label_bridge_identity(events, identity)
    return events, {
        "path": BRIDGE_CATALOG_RELATIVE,
        "file_sha256": raw_identity["file_sha256"],
        "file_bytes": raw_identity["file_bytes"],
        "identity": validated,
    }


def require_bridge_catalog_snapshot(
    expected: object,
) -> tuple[tuple[object, ...], dict[str, object]]:
    if not isinstance(expected, Mapping) or set(expected) != {
        "path",
        "file_sha256",
        "file_bytes",
        "identity",
    }:
        raise ValueError("terminal bridge catalog commitment is invalid")
    events, current = bridge_catalog_snapshot()
    if dict(expected) != current:
        raise ValueError("terminal bridge catalog changed after freeze")
    return events, current


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        handle.write("\n")


def load_envelope(path: Path, *, kind: str) -> tuple[dict[str, object], str]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping) or payload.get("kind") != kind:
        raise ValueError(f"invalid {kind} envelope")
    digest = hash_json(dict(payload))
    if envelope.get("payload_hash") != digest:
        raise ValueError(f"{kind} envelope hash mismatch")
    return dict(payload), digest


def verify_plan(path: Path) -> tuple[dict[str, object], str]:
    payload, digest = load_envelope(path, kind="hs300_portfolio_replay_plan")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("portfolio plan file inventory is invalid")
    if [item.get("path") for item in files if isinstance(item, Mapping)] != list(PLAN_FILES):
        raise ValueError("portfolio plan file inventory changed")
    _, bridge_commitment = require_bridge_catalog_snapshot(
        payload.get("terminal_bridge_catalog")
    )
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("portfolio plan file record is invalid")
        target = PROJECT / str(item["path"])
        if str(item["path"]) == BRIDGE_CATALOG_RELATIVE:
            observed_hash = bridge_commitment["file_sha256"]
            observed_bytes = bridge_commitment["file_bytes"]
        else:
            observed_hash = sha256_file(target)
            observed_bytes = target.stat().st_size
        if observed_hash != item.get("sha256") or observed_bytes != item.get("bytes"):
            raise ValueError(f"portfolio plan file changed: {item['path']}")
    policy = load_portfolio_replay_policy(PROJECT / str(payload["policy_path"]))
    if policy.policy_hash != payload.get("policy_hash"):
        raise ValueError("portfolio plan policy hash mismatch")
    return payload, digest


def freeze_plan(args: argparse.Namespace) -> None:
    policy_path = Path(args.policy).resolve()
    if policy_path != DEFAULT_POLICY.resolve():
        raise ValueError("portfolio policy path is fixed")
    policy = load_portfolio_replay_policy(policy_path)
    _, bridge_commitment = bridge_catalog_snapshot()
    files = [
        {
            "path": name,
            "sha256": (
                bridge_commitment["file_sha256"]
                if name == BRIDGE_CATALOG_RELATIVE
                else sha256_file(PROJECT / name)
            ),
            "bytes": (
                bridge_commitment["file_bytes"]
                if name == BRIDGE_CATALOG_RELATIVE
                else (PROJECT / name).stat().st_size
            ),
        }
        for name in PLAN_FILES
    ]
    payload = {
        "schema_version": 1,
        "kind": "hs300_portfolio_replay_plan",
        "status": "frozen_before_historical_replay",
        "created_at": utc_now(),
        "policy_path": str(policy_path.relative_to(PROJECT)),
        "policy_hash": policy.policy_hash,
        "terminal_bridge_catalog": bridge_commitment,
        "files": files,
        "alpha_backend": "lightgbm",
        "sealed_results_seen": False,
        "cash_execution_authorized": False,
        "official_index_enhancement_claim_authorized": False,
    }
    output = Path(args.output).resolve()
    exclusive_json({"payload": payload, "payload_hash": hash_json(payload)}, output)
    print(json.dumps({"plan": str(output), "plan_hash": hash_json(payload)}, indent=2))


def activate(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan).resolve()
    plan, plan_hash = verify_plan(plan_path)
    development_path, development, development_hash = hs300._load_reveal_manifest(
        Path(args.development_manifest).resolve()
    )
    locators = development["operational_locators"]
    sealed_output = Path(str(locators["sealed_output_dir"])).resolve()
    result_path = sealed_output / "sealed_replay_results.json"
    if result_path.exists():
        raise ValueError("portfolio plan activation must precede sealed replay output")
    development_bridge_identity = dict(
        development.get("resolved_experiment_config", {})
    ).get("terminal_label_bridge_identity")
    plan_bridge = dict(plan["terminal_bridge_catalog"])
    if development_bridge_identity != plan_bridge["identity"]:
        raise ValueError(
            "portfolio plan terminal bridge catalog differs from development commitment"
        )
    payload = {
        "schema_version": 1,
        "kind": "hs300_portfolio_replay_activation",
        "status": "activated_before_historical_replay",
        "created_at": utc_now(),
        "plan_path": str(plan_path),
        "plan_hash": plan_hash,
        "terminal_bridge_catalog": plan_bridge,
        "development_manifest": str(development_path),
        "development_manifest_hash": development_hash,
        "combined_data_snapshot_hash": development["source_snapshot_identity"]["combined_data_snapshot_hash"],
        "split_identity_hash": development["split_identity_hash"],
        "champion_family_hash": development["champion_family_hash"],
        "composite_family_hash": development["composite_family_hash"],
        "sealed_result_existed_at_activation": False,
    }
    output = development_path.parent / "portfolio_activation_manifest.json"
    exclusive_json({"payload": payload, "payload_hash": hash_json(payload)}, output)
    print(json.dumps({"activation": str(output), "activation_hash": hash_json(payload)}, indent=2))


def verify_activation(path: Path) -> tuple[dict[str, object], str, dict[str, object]]:
    activation, activation_hash = load_envelope(
        path, kind="hs300_portfolio_replay_activation"
    )
    plan, plan_hash = verify_plan(Path(str(activation["plan_path"])))
    if plan_hash != activation.get("plan_hash"):
        raise ValueError("activation plan hash mismatch")
    if activation.get("terminal_bridge_catalog") != plan.get(
        "terminal_bridge_catalog"
    ):
        raise ValueError("activation terminal bridge catalog mismatch")
    development_path, development, development_hash = hs300._load_reveal_manifest(
        Path(str(activation["development_manifest"]))
    )
    if development_hash != activation.get("development_manifest_hash"):
        raise ValueError("activation development hash mismatch")
    development_bridge_identity = dict(
        development.get("resolved_experiment_config", {})
    ).get("terminal_label_bridge_identity")
    activation_bridge = dict(activation["terminal_bridge_catalog"])
    if development_bridge_identity != activation_bridge["identity"]:
        raise ValueError(
            "activation terminal bridge catalog differs from development commitment"
        )
    for key in (
        "split_identity_hash", "champion_family_hash", "composite_family_hash"
    ):
        if activation.get(key) != development.get(key):
            raise ValueError(f"activation {key} mismatch")
    if activation.get("combined_data_snapshot_hash") != development[
        "source_snapshot_identity"
    ]["combined_data_snapshot_hash"]:
        raise ValueError("activation data snapshot mismatch")
    return activation, activation_hash, development


def completed_reveal(development: Mapping[str, object]) -> tuple[Path, dict[str, object], Path]:
    output = Path(str(development["operational_locators"]["sealed_output_dir"])).resolve()
    result_path = output / "sealed_replay_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("development_manifest_hash") != hash_json(development):
        # The development hash is the envelope payload hash, not a file digest.
        manifest_envelope = json.loads(Path(str(result["development_manifest"])).read_text())
        if result.get("development_manifest_hash") != manifest_envelope.get("manifest_hash"):
            raise ValueError("sealed result development hash mismatch")
    if result.get("composite_family_hash") != development.get("composite_family_hash"):
        raise ValueError("sealed result composite family mismatch")
    claim = Path(str(result["reveal_registry_record"])).resolve()
    completion = claim.with_suffix(".result.json")
    completed = json.loads(completion.read_text(encoding="utf-8"))
    if completed.get("result_path") != str(result_path) or completed.get("result_sha256") != sha256_file(result_path):
        raise ValueError("sealed reveal completion evidence mismatch")
    return result_path, result, claim


def build_primary_alpha(
    development: Mapping[str, object],
    claim: Path,
    development_manifest_path: Path,
) -> tuple[pd.Series, hs300.FrozenPanel, object]:
    panel_dir = Path(str(development["operational_locators"]["panel_dir"])).resolve()
    frozen = hs300.load_frozen_hs300_panel(panel_dir)
    split = hs300._split_from_payload(development["split_exact"])
    champions, artifacts = hs300._champions_from_frozen_manifest(development["champions"])
    feature_frame = build_research_feature_frame(frozen.feature_tape)
    mask = build_cross_sectional_mask(frozen.feature_tape, frozen.eligible_panel)
    exposures = build_style_exposures(frozen.eligible_panel, price_tape=frozen.feature_tape)
    sealed_rows = frozen.eligible_panel.index.get_level_values("date").isin(split.sealed_dates)
    sealed_features = pd.DataFrame(index=frozen.eligible_panel.index[sealed_rows])
    for evaluated in champions:
        artifact = artifacts[evaluated.candidate.candidate_id]
        expression = json.loads(artifact.canonical_expression_json)
        values = evaluate_expression(
            expression, feature_frame, PRODUCTION_DSL_SPEC,
            cross_sectional_mask=mask,
        ).reindex(frozen.eligible_panel.index) * artifact.direction
        neutral = cross_sectional_neutralize(
            values, exposures, categorical=("industry",), min_names=20
        )
        sealed_features[artifact.candidate_id] = neutral.loc[sealed_features.index]
    run_dir = development_manifest_path.resolve().parent
    family = load_frozen_composite_family(
        run_dir / "composite",
        expected_composite_family_hash=str(development["composite_family_hash"]),
    )
    capability = SealedCompositeClaimCapability(
        composite_family_hash=str(development["composite_family_hash"]),
        split_identity_hash=str(development["split_identity_hash"]),
        combined_data_snapshot_hash=str(development["source_snapshot_identity"]["combined_data_snapshot_hash"]),
        source_snapshot_identity_hash=str(development["source_snapshot_identity"]["identity_hash"]),
        persistent_claim_path=str(claim),
        persistent_claim_sha256=sha256_file(claim),
    )
    predictions = family.predict_all_for_sealed_gate(
        sealed_features,
        expected_composite_family_hash=str(development["composite_family_hash"]),
        claim_capability=capability,
    )
    primary = cross_sectional_neutralize(
        predictions["lightgbm"], exposures.reindex(predictions.index),
        categorical=("industry",), min_names=20,
    )
    primary.name = "primary_lightgbm_neutral_alpha"
    return primary, frozen, split


def prepare_targets(args: argparse.Namespace) -> None:
    activation, activation_hash, development = verify_activation(
        Path(args.activation).resolve()
    )
    result_path, _, claim = completed_reveal(development)
    policy = load_portfolio_replay_policy(DEFAULT_POLICY)
    alpha, frozen, split = build_primary_alpha(
        development,
        claim,
        Path(str(activation["development_manifest"])),
    )
    book = prepare_target_book(
        alpha, frozen.eligible_panel, frozen.feature_tape,
        frozen.trading_calendar, split.sealed_dates, policy,
    )
    output = Path(str(development["operational_locators"]["sealed_output_dir"])).resolve() / "portfolio_v1"
    output.mkdir(parents=True, exist_ok=False)
    target_path = output / "targets.parquet"
    benchmark_path = output / "benchmarks.parquet"
    diagnostics_path = output / "target_diagnostics.parquet"
    alpha_path = output / "primary_alpha.parquet"
    book.targets.to_parquet(target_path)
    book.benchmarks.to_parquet(benchmark_path)
    book.diagnostics.to_parquet(diagnostics_path)
    alpha.to_frame().to_parquet(alpha_path)
    payload = {
        "schema_version": 1,
        "kind": "hs300_frozen_portfolio_target_book",
        "created_at": utc_now(),
        "activation_hash": activation_hash,
        "plan_hash": activation["plan_hash"],
        "terminal_bridge_catalog": activation["terminal_bridge_catalog"],
        "development_manifest_hash": activation["development_manifest_hash"],
        "sealed_result_sha256": sha256_file(result_path),
        "policy_hash": policy.policy_hash,
        "target_book_hash": target_book_hash(book),
        "signal_dates": [str(date.date()) for date in book.signal_dates],
        "execution_dates": [str(date.date()) for date in book.execution_dates],
        "exit_dates": [str(date.date()) for date in book.exit_dates],
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (target_path, benchmark_path, diagnostics_path, alpha_path)
        },
        "forward_return_labels_loaded_during_target_construction": False,
    }
    manifest = output / "target_book_manifest.json"
    exclusive_json({"payload": payload, "payload_hash": hash_json(payload)}, manifest)
    print(json.dumps({"target_book_manifest": str(manifest), "target_book_hash": payload["target_book_hash"]}, indent=2))


def load_target_book(path: Path, policy_hash: str) -> tuple[TargetBook, dict[str, object], str]:
    payload, digest = load_envelope(path, kind="hs300_frozen_portfolio_target_book")
    if payload.get("policy_hash") != policy_hash:
        raise ValueError("target book policy mismatch")
    files = payload["files"]
    if not isinstance(files, Mapping):
        raise ValueError("target book files are invalid")
    for name, record in files.items():
        target = path.parent / str(name)
        if not isinstance(record, Mapping) or sha256_file(target) != record.get("sha256"):
            raise ValueError(f"target book artifact changed: {name}")
    targets = pd.read_parquet(path.parent / "targets.parquet").sort_index()
    benchmarks = pd.read_parquet(path.parent / "benchmarks.parquet").sort_index()
    diagnostics = pd.read_parquet(path.parent / "target_diagnostics.parquet").sort_index()
    signals = tuple(pd.Timestamp(value) for value in payload["signal_dates"])
    executions = tuple(pd.Timestamp(value) for value in payload["execution_dates"])
    exits = tuple(pd.Timestamp(value) for value in payload["exit_dates"])
    book = TargetBook(
        targets, benchmarks, diagnostics, signals, executions, exits, policy_hash
    )
    if target_book_hash(book) != payload.get("target_book_hash"):
        raise ValueError("target book content hash mismatch")
    return book, payload, digest


def score(args: argparse.Namespace) -> None:
    policy = load_portfolio_replay_policy(DEFAULT_POLICY)
    target_manifest = Path(args.target_book_manifest).resolve()
    book, target_payload, target_manifest_hash = load_target_book(
        target_manifest, policy.policy_hash
    )
    activation_path = Path(args.activation).resolve()
    activation, activation_hash, development = verify_activation(activation_path)
    if activation_hash != target_payload.get("activation_hash"):
        raise ValueError("target book activation mismatch")
    if target_payload.get("terminal_bridge_catalog") != activation.get(
        "terminal_bridge_catalog"
    ):
        raise ValueError("target book terminal bridge catalog mismatch")
    result_path, _, _ = completed_reveal(development)
    if sha256_file(result_path) != target_payload.get("sealed_result_sha256"):
        raise ValueError("sealed result changed after target construction")
    panel_dir = Path(str(development["operational_locators"]["panel_dir"])).resolve()
    frozen = hs300.load_frozen_hs300_panel(panel_dir)
    split = hs300._split_from_payload(development["split_exact"])
    features = build_research_feature_frame(frozen.feature_tape)
    labels = forward_return(
        features["vwap"], policy.holding_horizon_sessions,
        entry_delay=policy.execution_delay_sessions,
        trading_calendar=frozen.trading_calendar,
    )
    bridges, bridge_commitment = require_bridge_catalog_snapshot(
        activation["terminal_bridge_catalog"]
    )
    bridge_identity = dict(bridge_commitment["identity"])
    labels, label_audit = apply_terminal_label_bridges(
        labels, frozen.feature_tape, frozen.eligible_panel.index,
        bridges, terminal_label_bridge_identity=bridge_identity,
        endpoint_price_field="vwap",
        horizon=policy.holding_horizon_sessions,
        entry_delay=policy.execution_delay_sessions,
        trading_calendar=frozen.trading_calendar,
    )
    executable, executable_definition = build_immediate_executable_labels(
        features, frozen.feature_tape,
        horizon=policy.holding_horizon_sessions,
        entry_delay=policy.execution_delay_sessions,
        trading_calendar=frozen.trading_calendar,
    )
    result = score_target_book(book, labels, executable, policy)
    output = target_manifest.parent
    periods_path = output / "portfolio_periods.parquet"
    scenarios_path = output / "cost_scenarios.parquet"
    summary_path = output / "portfolio_performance.json"
    result.periods.to_parquet(periods_path)
    result.cost_scenarios.to_parquet(scenarios_path)
    summary = {
        **dict(result.summary),
        "created_at": utc_now(),
        "target_book_manifest_hash": target_manifest_hash,
        "activation_hash": activation_hash,
        "development_manifest_hash": activation["development_manifest_hash"],
        "sealed_result_sha256": sha256_file(result_path),
        "terminal_bridge_catalog": bridge_commitment,
        "terminal_label_bridge_audit": label_audit,
        "immediate_executable_label_definition": executable_definition,
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (periods_path, scenarios_path)
        },
    }
    exclusive_json(summary, summary_path)
    print(json.dumps({"performance": str(summary_path), "primary": summary["primary"]}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-plan")
    freeze.add_argument("--policy", default=str(DEFAULT_POLICY))
    freeze.add_argument("--output", default=str(DEFAULT_PLAN))
    freeze.set_defaults(func=freeze_plan)
    activation = sub.add_parser("activate")
    activation.add_argument("--plan", default=str(DEFAULT_PLAN))
    activation.add_argument("--development-manifest", required=True)
    activation.set_defaults(func=activate)
    prepare = sub.add_parser("prepare-targets")
    prepare.add_argument("--activation", required=True)
    prepare.set_defaults(func=prepare_targets)
    scoring = sub.add_parser("score")
    scoring.add_argument("--activation", required=True)
    scoring.add_argument("--target-book-manifest", required=True)
    scoring.set_defaults(func=score)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
