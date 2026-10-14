"""Command-line entry points for deterministic local research tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from .dsl import expression_hash
from .evolution import SearchBudget
from .data_sources.hs300_download import (
    HS300SourceConfig,
    download_adjustment_factor_catalog,
    download_industry_catalog,
    download_membership_catalog,
    download_price_catalog,
    rebuild_industry_catalog_from_cache,
    rebuild_membership_catalog_from_cache,
    rebuild_security_master_from_cache,
    raw_cache_status,
)
from .data_sources.hs300_build import HS300BuildConfig, build_hs300_research_panel
from .data_sources.hs300_audit import HS300AuditConfig, audit_hs300_research_panel
from .data_sources.adjustment_reconstruction_audit import (
    DEFAULT_ABSOLUTE_TOLERANCE as ADJUSTMENT_RECONSTRUCTION_DEFAULT_ATOL,
    DEFAULT_RELATIVE_TOLERANCE as ADJUSTMENT_RECONSTRUCTION_DEFAULT_RTOL,
    AdjustmentReconstructionAuditConfig,
    audit_adjustment_reconstruction,
    configuration_failure_report,
)
from .hs300_research import (
    HS300ResearchConfig,
    SEMANTIC_REVIEW_RELATIVE_PATH,
    load_codex_injection_candidates,
    reveal_hs300_replay,
    run_hs300_research,
)
from .manifest import hash_json
from .paths import data_home
from .proposals import build_agent_input_contract, load_factor_cards
from .runner import run_experiment
from .semantic_audit import validate_semantic_review_manifest
from .synthetic import make_synthetic_market
from .validation import make_split_plan


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_configuration(project: Path) -> dict[str, object]:
    project = project.resolve()
    semantic_review = validate_semantic_review_manifest(
        project,
        project / SEMANTIC_REVIEW_RELATIVE_PATH,
    )
    with (project / "configs" / "agents.yaml").open(encoding="utf-8") as handle:
        agents = yaml.safe_load(handle)
    with (project / "configs" / "faithful.yaml").open(encoding="utf-8") as handle:
        faithful = yaml.safe_load(handle)
    with (project / "configs" / "plus.yaml").open(encoding="utf-8") as handle:
        plus = yaml.safe_load(handle)
    ids = [agent["id"] for agent in agents["agents"]]
    levels = {agent["level"] for agent in agents["agents"]}
    if len(ids) != 21 or len(set(ids)) != 21 or levels != set(range(1, 8)):
        raise ValueError("agent catalog must contain 21 unique agents over seven levels")
    keys = ("initial_population", "parent_pool", "offspring_per_generation", "generations", "elite_carryover", "final_factor_count")
    faithful_budget = faithful["search_budget"]
    plus_budget = plus["search_budget"]
    mismatches = {key: (faithful_budget[key], plus_budget[key]) for key in keys if faithful_budget[key] != plus_budget[key]}
    if mismatches:
        raise ValueError(f"faithful/plus core budget mismatch: {mismatches}")
    allowed = set(ids)
    agent_input_contract = build_agent_input_contract(agents)
    seed_cards = load_factor_cards(
        [
            project / "proposals" / "codex_seed" / name
            for name in ("l1_l2.jsonl", "l3_l4.jsonl", "l5_l7.jsonl")
        ],
        allowed_agent_ids=allowed,
        agent_allowed_inputs=agent_input_contract,
        expected_horizon=10,
        expected_count=80,
    )
    injection_cards = load_factor_cards(
        [
            project / "proposals" / "codex_injection" / name
            for name in ("l1_l4.jsonl", "l5_l7.jsonl")
        ],
        allowed_agent_ids=allowed,
        agent_allowed_inputs=agent_input_contract,
        expected_horizon=10,
        expected_count=96,
    )
    for label, cards in (("seed", seed_cards), ("injection", injection_cards)):
        covered = {card.agent_id for card in cards}
        if covered != allowed:
            raise ValueError(
                f"{label} Factor Card library must cover all 21 agents; "
                f"missing={sorted(allowed - covered)}"
            )
    seed_hashes = {expression_hash(card.expression) for card in seed_cards}
    injection_hashes = {
        expression_hash(card.expression) for card in injection_cards
    }
    overlap = seed_hashes.intersection(injection_hashes)
    if overlap:
        raise ValueError(
            "generation-zero and precommitted injection libraries contain "
            f"{len(overlap)} structural duplicate(s)"
        )
    # Reuse the real-run loader so check-config also enforces the full
    # role-by-repair contract, not just the generic FactorCard schema.
    load_codex_injection_candidates(project, horizon=10)
    injection_contract = plus.get("evolution", {}).get(
        "precommitted_adaptive_injection", {}
    )
    expected_injection_contract = {
        "codex_authored_cards": len(injection_cards),
        "selection": "development_feedback_role_and_repair_match",
        "without_replacement": True,
        "reserve_card_asts_from_other_operators": True,
        "terminal_accounting": "require_all_96_evaluated_and_zero_remaining",
    }
    if injection_contract != expected_injection_contract:
        raise ValueError(
            "Plus precommitted injection declaration does not match runtime: "
            f"declared={injection_contract}, expected={expected_injection_contract}"
        )
    return {
        "agents": len(ids),
        "levels": sorted(levels),
        "faithful_profile": faithful["profile_id"],
        "plus_profile": plus["profile_id"],
        "core_budget_equal": True,
        "core_budget": {key: faithful_budget[key] for key in keys},
        "factor_cards": {
            "generation_zero": len(seed_cards),
            "precommitted_adaptive_injection": len(injection_cards),
            "total_codex_authored": len(seed_cards) + len(injection_cards),
            "all_21_roles_covered_in_each_library": True,
            "structural_overlap_between_libraries": 0,
            "role_input_field_contract_enforced": True,
        },
        "economic_semantic_review": {
            "candidate_count": semantic_review["candidate_count"],
            "review_set_hash": semantic_review["review_set_hash"],
            "approval_binding_hash": semantic_review["approval_binding_hash"],
            "aggregate_hash": hash_json(semantic_review),
        },
    }


def command_check(args: argparse.Namespace) -> int:
    print(json.dumps(check_configuration(Path(args.project)), ensure_ascii=False, indent=2))
    return 0


def command_demo(args: argparse.Namespace) -> int:
    panel = make_synthetic_market(days=args.days, symbols=args.symbols, seed=args.seed)
    split = make_split_plan(
        panel,
        label_horizon=args.horizon,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    budget = SearchBudget.tiny() if args.budget == "tiny" else SearchBudget()
    _, summary = run_experiment(
        full_panel=panel,
        split=split,
        output_dir=args.output,
        budget=budget,
        profile=args.profile,
        seed=args.seed,
        reveal_sealed=args.reveal_synthetic_sealed,
        bootstrap_samples=args.bootstrap_samples,
        evaluation_workers=args.evaluation_workers,
        model_version="template-control-no-llm",
        prompt_version="synthetic-integration-test-v1",
        # The synthetic fixture has no vendor calendar; its own dates are the
        # session calendar.  Real-data commands never set this flag.
        allow_test_calendar_from_panel=True,
    )
    concise = {
        "run_id": summary["run_id"],
        "evaluated_candidates": summary["evaluated_candidates"],
        "registry": summary["registry"],
        "ledger": summary["ledger"],
        "sealed_revealed": summary["sealed_revealed"],
        "top_champion": summary["champions"][0] if summary["champions"] else None,
        "summary_file": str(Path(args.output).resolve() / "summary.json"),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0


def command_validate_proposals(args: argparse.Namespace) -> int:
    project = Path(args.project)
    with (project / "configs" / "agents.yaml").open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    agent_ids = {item["id"] for item in catalog["agents"]}
    agent_input_contract = build_agent_input_contract(catalog)
    cards = load_factor_cards(
        args.files,
        allowed_agent_ids=agent_ids,
        agent_allowed_inputs=agent_input_contract,
        expected_horizon=args.horizon,
        expected_count=args.expected_count,
    )
    counts = {agent_id: 0 for agent_id in sorted(agent_ids)}
    for card in cards:
        counts[card.agent_id] += 1
    missing = [agent_id for agent_id, count in counts.items() if count == 0]
    if args.require_all_agents and missing:
        raise ValueError(f"proposal batch omits agent roles: {missing}")
    print(
        json.dumps(
            {
                "valid": True,
                "cards": len(cards),
                "horizon": args.horizon,
                "covered_agents": sum(count > 0 for count in counts.values()),
                "missing_agents": missing,
                "counts_by_agent": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _hs300_source_config(args: argparse.Namespace) -> HS300SourceConfig:
    project = Path(args.project).resolve()
    root = Path(args.data_root) if args.data_root else data_home(project) / "baostock_hs300"
    if not root.is_absolute():
        root = (project / root).resolve()
    return HS300SourceConfig(
        cache_dir=root / "raw",
        normalized_dir=root / "normalized",
        start_date=args.start,
        end_date=args.end,
        membership_frequency=getattr(args, "membership_frequency", "provider_weekly"),
        max_attempts=args.max_attempts,
    )


def _project_relative_path(project: Path, value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _hs300_build_config(args: argparse.Namespace) -> HS300BuildConfig:
    project = Path(args.project).resolve()
    root = _project_relative_path(
        project,
        args.data_root,
        data_home(project) / "baostock_hs300",
    )
    output = _project_relative_path(project, args.output_dir, root / "panel")
    return HS300BuildConfig(
        cache_dir=root / "raw",
        normalized_dir=root / "normalized",
        output_dir=output,
        research_start=args.research_start,
        research_end=args.research_end,
        expected_members=args.expected_members,
        adv_window=args.adv_window,
        adv_min_periods=args.adv_min_periods,
        beta_window=args.beta_window,
        beta_min_periods=args.beta_min_periods,
        contract_minimum_history_days=args.minimum_history_days,
        minimum_limit_rule_listed_days=args.minimum_limit_rule_listed_days,
        maximum_industry_age_days=args.maximum_industry_age_days,
        membership_source_start_date=args.source_start,
        membership_source_end_date=args.source_end,
        membership_frequency=args.membership_frequency,
        maximum_membership_snapshot_gap_days=(
            args.maximum_membership_snapshot_gap_days
        ),
        maximum_membership_snapshot_age_days=(
            args.maximum_membership_snapshot_age_days
        ),
        adjustment_event_start_date=args.adjustment_event_start,
    )


def _download_progress(every: int):
    def report(done: int, total: int, item: str) -> None:
        if done == 1 or done == total or done % every == 0:
            print(
                json.dumps(
                    {"progress": done, "total": total, "item": item},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return report


def command_data_membership(args: argparse.Namespace) -> int:
    summary = download_membership_catalog(
        _hs300_source_config(args),
        progress=_download_progress(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def command_data_membership_lineage(args: argparse.Namespace) -> int:
    summary = rebuild_membership_catalog_from_cache(
        _hs300_source_config(args),
        progress=_download_progress(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def command_data_security_lineage(args: argparse.Namespace) -> int:
    summary = rebuild_security_master_from_cache(_hs300_source_config(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def command_data_industry_lineage(args: argparse.Namespace) -> int:
    summary = rebuild_industry_catalog_from_cache(
        _hs300_source_config(args),
        progress=_download_progress(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def command_data_prices(args: argparse.Namespace) -> int:
    summary = download_price_catalog(
        _hs300_source_config(args),
        limit_symbols=args.limit_symbols,
        progress=_download_progress(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def command_data_industry(args: argparse.Namespace) -> int:
    summary = download_industry_catalog(
        _hs300_source_config(args),
        progress=_download_progress(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def command_data_adjustments(args: argparse.Namespace) -> int:
    summary = download_adjustment_factor_catalog(
        _hs300_source_config(args),
        limit_symbols=args.limit_symbols,
        progress=_download_progress(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def command_data_adjustment_reconstruction_audit(
    args: argparse.Namespace,
) -> int:
    """Run the offline, read-only gate for a future raw-plus-factor source."""

    project = Path(args.project).resolve()
    root = _project_relative_path(
        project,
        args.data_root,
        data_home(project) / "baostock_hs300",
    )
    cache_dir = root / "raw"
    normalized_dir = root / "normalized"
    try:
        config = AdjustmentReconstructionAuditConfig(
            cache_dir=cache_dir,
            normalized_dir=normalized_dir,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
            maximum_failure_samples=args.maximum_failure_samples,
        )
        report = audit_adjustment_reconstruction(config)
    except Exception as exc:
        report = configuration_failure_report(
            cache_dir=cache_dir,
            normalized_dir=normalized_dir,
            error=exc,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


def command_data_status(args: argparse.Namespace) -> int:
    config = _hs300_source_config(args)
    status = raw_cache_status(config.cache_dir)
    status["cache_dir"] = str(config.cache_dir)
    status["normalized_dir"] = str(config.normalized_dir)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def command_data_build(args: argparse.Namespace) -> int:
    result = build_hs300_research_panel(_hs300_build_config(args))
    manifest_sha256 = result.manifest_sha256_path.read_text(encoding="utf-8").split()[0]
    report = result.validation
    summary = {
        "valid": report.valid,
        "dataset_tier": report.metadata.get("dataset_tier"),
        "research_proxy_only": result.manifest.get("research_proxy_only"),
        "research_watermark": result.manifest.get("research_watermark"),
        "snapshot_hash": report.snapshot_hash,
        "rows": report.rows,
        "symbols": report.symbols,
        "trading_days": report.trading_days,
        "first_date": report.first_date,
        "last_date": report.last_date,
        "warnings": list(report.warnings),
        "manifest": str(result.manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "manifest_sha256_file": str(result.manifest_sha256_path.resolve()),
        "partitions": len(result.partition_paths),
        "continuous_feature_tape_rows": (
            len(result.feature_tape) if result.feature_tape is not None else 0
        ),
        "continuous_feature_tape_partitions": len(
            result.feature_tape_partition_paths
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _hs300_audit_config(args: argparse.Namespace) -> HS300AuditConfig:
    project = Path(args.project).resolve()
    root = _project_relative_path(
        project,
        args.data_root,
        data_home(project) / "baostock_hs300",
    )
    panel_dir = _project_relative_path(project, args.panel_dir, root / "panel")
    normalized_dir = (
        _project_relative_path(project, args.normalized_dir, root / "normalized")
        if args.normalized_dir
        else None
    )
    cache_dir = (
        _project_relative_path(project, args.cache_dir, root / "raw")
        if args.cache_dir
        else None
    )
    return HS300AuditConfig(
        panel_dir=panel_dir,
        expected_members=args.expected_members,
        normalized_dir=normalized_dir,
        cache_dir=cache_dir,
        adjusted_price_tolerance=args.adjusted_price_tolerance,
        adjusted_return_warning_threshold=args.adjusted_return_warning_threshold,
        adjusted_return_fail_threshold=args.adjusted_return_fail_threshold,
        suspension_daily_warning_rate=args.suspension_daily_warning_rate,
        minimum_proxy_mcap_coverage=args.minimum_proxy_mcap_coverage,
        maximum_samples=args.maximum_samples,
    )


def command_data_audit(args: argparse.Namespace) -> int:
    report = audit_hs300_research_panel(_hs300_audit_config(args))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.valid else 1


def command_research_hs300(args: argparse.Namespace) -> int:
    project = Path(args.project).resolve()
    data_root = _project_relative_path(
        project,
        args.data_root,
        data_home(project) / "baostock_hs300",
    )
    panel_dir = _project_relative_path(project, args.panel_dir, data_root / "panel")
    output_dir = _project_relative_path(project, args.output, project / "runs" / "hs300_full_v1")
    result = run_hs300_research(
        HS300ResearchConfig(
            project_dir=project,
            panel_dir=panel_dir,
            output_dir=output_dir,
            train_end=args.train_end,
            validation_end=args.validation_end,
            replay_end=args.replay_end,
            label_horizon=args.horizon,
            entry_delay=args.entry_delay,
            profile=args.profile,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            evaluation_workers=args.evaluation_workers,
            expected_members=args.expected_members,
        )
    )
    experiment = result.summary["experiment_summary"]
    concise = {
        "run_id": experiment["run_id"],
        "output_dir": str(output_dir),
        "data_snapshot_hash": result.data_snapshot_hash,
        "historical_replay_warning": result.historical_replay_warning,
        "historical_replay_revealed": result.summary["historical_replay_revealed"],
        "sealed_market_rows_loaded_by_development": result.summary[
            "sealed_market_rows_loaded_by_development"
        ],
        "sealed_reveal_manifest": result.summary["sealed_reveal_manifest"],
        "sealed_reveal_manifest_hash": result.summary[
            "sealed_reveal_manifest_hash"
        ],
        "split": result.split.to_dict(),
        "evaluated_candidates": experiment["evaluated_candidates"],
        "registry": experiment["registry"],
        "ledger": experiment["ledger"],
        "champions": experiment["champions"],
        "summary_file": str(output_dir / "hs300_research_summary.json"),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, default=str))
    return 0


def command_research_hs300_reveal(args: argparse.Namespace) -> int:
    result = reveal_hs300_replay(Path(args.development_manifest).resolve())
    concise = {
        "development_manifest_hash": result.development_manifest_hash,
        "data_snapshot_hash": result.data_snapshot_hash,
        "historical_replay_warning": result.historical_replay_warning,
        "historical_replay_revealed": True,
        "split": result.split.to_dict(),
        "champions_revealed": len(result.results),
        "registry_record": str(result.registry_record),
        "result_file": str(result.output_path),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, default=str))
    return 0


def _add_data_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=str(_project_root()))
    parser.add_argument("--data-root")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--max-attempts", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cogalpha", description="CogAlpha-Plus local research framework")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check-config", help="validate the seven-level catalog and equal search budgets")
    check.add_argument("--project", default=str(_project_root()))
    check.set_defaults(func=command_check)

    proposals = sub.add_parser("validate-proposals", help="validate LLM Factor Card JSONL files")
    proposals.add_argument("files", nargs="+")
    proposals.add_argument("--project", default=str(_project_root()))
    proposals.add_argument("--horizon", type=int, default=10)
    proposals.add_argument("--expected-count", type=int)
    proposals.add_argument("--require-all-agents", action="store_true")
    proposals.set_defaults(func=command_validate_proposals)

    demo = sub.add_parser("demo", help="run a deterministic synthetic end-to-end integration experiment")
    demo.add_argument("--output", required=True)
    demo.add_argument("--profile", choices=("faithful", "plus"), default="plus")
    demo.add_argument("--budget", choices=("tiny", "full"), default="tiny")
    demo.add_argument("--days", type=int, default=320)
    demo.add_argument("--symbols", type=int, default=35)
    demo.add_argument("--horizon", type=int, default=5)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--train-fraction", type=float, default=0.50)
    demo.add_argument("--validation-fraction", type=float, default=0.20)
    demo.add_argument("--bootstrap-samples", type=int, default=50)
    demo.add_argument("--evaluation-workers", type=int, default=1)
    demo.add_argument(
        "--reveal-synthetic-sealed",
        action="store_true",
        help="reveal the one-time outer gate; safe only because this command uses fake data",
    )
    demo.set_defaults(func=command_demo)

    research = sub.add_parser(
        "research",
        help="run factor discovery on a frozen real-data panel",
    )
    research_sub = research.add_subparsers(dest="research_command", required=True)
    hs300 = research_sub.add_parser(
        "hs300",
        help="run the full 7-level/21-role/24-generation HS300 experiment",
    )
    hs300.add_argument("--project", default=str(_project_root()))
    hs300.add_argument("--data-root")
    hs300.add_argument("--panel-dir")
    hs300.add_argument("--output", required=True)
    hs300.add_argument("--train-end", default="2017-12-31")
    hs300.add_argument("--validation-end", default="2019-12-31")
    hs300.add_argument("--replay-end", default="2025-12-31")
    hs300.add_argument("--horizon", type=int, default=10)
    hs300.add_argument("--entry-delay", type=int, default=1)
    hs300.add_argument("--profile", choices=("faithful", "plus"), default="plus")
    hs300.add_argument("--seed", type=int, default=17)
    hs300.add_argument("--bootstrap-samples", type=int, default=200)
    hs300.add_argument("--evaluation-workers", type=int, default=1)
    hs300.add_argument("--expected-members", type=int, default=300)
    hs300.set_defaults(func=command_research_hs300)

    hs300_reveal = research_sub.add_parser(
        "hs300-reveal",
        help=(
            "reveal one frozen HS300 development manifest in a separate process; "
            "accepts no search or factor overrides"
        ),
    )
    hs300_reveal.add_argument(
        "--development-manifest",
        required=True,
        help="content-hashed sealed_reveal_manifest.json produced by research hs300",
    )
    hs300_reveal.set_defaults(func=command_research_hs300_reveal)

    data = sub.add_parser("data", help="download and audit local market research data")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    membership = data_sub.add_parser(
        "membership",
        help="download the calendar and point-in-time CSI300 membership catalog",
    )
    _add_data_common(membership)
    membership.add_argument(
        "--membership-frequency",
        choices=("provider_weekly", "trading_day"),
        default="provider_weekly",
    )
    membership.add_argument("--progress-every", type=int, default=25)
    membership.set_defaults(func=command_data_membership)

    membership_lineage = data_sub.add_parser(
        "membership-lineage",
        help=(
            "rebuild and freeze membership catalogs from existing immutable "
            "raw cache without network access"
        ),
    )
    _add_data_common(membership_lineage)
    membership_lineage.add_argument(
        "--membership-frequency",
        choices=("provider_weekly", "trading_day"),
        default="provider_weekly",
    )
    membership_lineage.add_argument("--progress-every", type=int, default=25)
    membership_lineage.set_defaults(func=command_data_membership_lineage)

    security_lineage = data_sub.add_parser(
        "security-lineage",
        help=(
            "rebuild and freeze the security master from existing immutable "
            "stock_basic cache without network access"
        ),
    )
    _add_data_common(security_lineage)
    security_lineage.set_defaults(func=command_data_security_lineage)

    prices = data_sub.add_parser(
        "prices",
        help="download raw and backward-adjusted histories for the historical union",
    )
    _add_data_common(prices)
    prices.add_argument("--limit-symbols", type=int)
    prices.add_argument("--progress-every", type=int, default=10)
    prices.set_defaults(func=command_data_prices)

    adjustments = data_sub.add_parser(
        "adjustments",
        help=(
            "download dated BaoStock adjustment-factor events for the "
            "historical union"
        ),
    )
    _add_data_common(adjustments)
    adjustments.add_argument("--limit-symbols", type=int)
    adjustments.add_argument("--progress-every", type=int, default=20)
    adjustments.set_defaults(start="1990-01-01", func=command_data_adjustments)

    adjustment_reconstruction_audit = data_sub.add_parser(
        "adjustment-reconstruction-audit",
        help=(
            "offline read-only comparison of raw+factor reconstruction with "
            "the complete direct adjusted-price cache"
        ),
    )
    adjustment_reconstruction_audit.add_argument(
        "--project", default=str(_project_root())
    )
    adjustment_reconstruction_audit.add_argument("--data-root")
    adjustment_reconstruction_audit.add_argument(
        "--relative-tolerance",
        type=float,
        default=ADJUSTMENT_RECONSTRUCTION_DEFAULT_RTOL,
        help=(
            "relative cell tolerance; may only tighten the registered "
            f"{ADJUSTMENT_RECONSTRUCTION_DEFAULT_RTOL:g} limit"
        ),
    )
    adjustment_reconstruction_audit.add_argument(
        "--absolute-tolerance",
        type=float,
        default=ADJUSTMENT_RECONSTRUCTION_DEFAULT_ATOL,
        help=(
            "absolute cell tolerance; may only tighten the registered "
            f"{ADJUSTMENT_RECONSTRUCTION_DEFAULT_ATOL:g} limit"
        ),
    )
    adjustment_reconstruction_audit.add_argument(
        "--maximum-failure-samples", type=int, default=20
    )
    adjustment_reconstruction_audit.set_defaults(
        func=command_data_adjustment_reconstruction_audit
    )

    industry = data_sub.add_parser(
        "industry",
        help="download dated CSRC industry snapshots for the historical union",
    )
    _add_data_common(industry)
    industry.add_argument("--progress-every", type=int, default=20)
    industry.set_defaults(func=command_data_industry)

    industry_lineage = data_sub.add_parser(
        "industry-lineage",
        help=(
            "rebuild and freeze industry snapshots from existing immutable "
            "raw cache without network access"
        ),
    )
    _add_data_common(industry_lineage)
    industry_lineage.add_argument("--progress-every", type=int, default=20)
    industry_lineage.set_defaults(func=command_data_industry_lineage)

    status = data_sub.add_parser("status", help="summarize immutable raw-cache request state")
    _add_data_common(status)
    status.set_defaults(func=command_data_status)

    build = data_sub.add_parser(
        "build",
        help="validate and freeze the downloaded CSI300 research-proxy panel",
    )
    build.add_argument("--project", default=str(_project_root()))
    build.add_argument("--data-root")
    build.add_argument(
        "--output-dir",
        help="snapshot output directory (default: <data-root>/panel; relative paths use --project)",
    )
    build.add_argument("--research-start", default="2011-01-01")
    build.add_argument("--research-end", default="2025-12-31")
    build.add_argument("--expected-members", type=int, default=300)
    build.add_argument("--adv-window", type=int, default=20)
    build.add_argument("--adv-min-periods", type=int)
    build.add_argument("--beta-window", type=int, default=252)
    build.add_argument("--beta-min-periods", type=int, default=60)
    build.add_argument("--minimum-history-days", type=int, default=252)
    build.add_argument("--minimum-limit-rule-listed-days", type=int, default=30)
    build.add_argument("--maximum-industry-age-days", type=int, default=35)
    build.add_argument("--source-start", default="2010-01-01")
    build.add_argument("--source-end", default="2025-12-31")
    build.add_argument(
        "--membership-frequency",
        choices=("provider_weekly", "trading_day"),
        default="provider_weekly",
    )
    build.add_argument("--maximum-membership-snapshot-gap-days", type=int, default=21)
    build.add_argument("--maximum-membership-snapshot-age-days", type=int, default=21)
    build.add_argument("--adjustment-event-start", default="1990-01-01")
    build.set_defaults(func=command_data_build)

    audit = data_sub.add_parser(
        "audit",
        help="offline integrity and coverage audit of a frozen CSI300 panel",
    )
    audit.add_argument("--project", default=str(_project_root()))
    audit.add_argument("--data-root")
    audit.add_argument(
        "--panel-dir",
        help="frozen panel directory (default: <data-root>/panel)",
    )
    audit.add_argument(
        "--normalized-dir",
        help="override a relocated normalized catalog directory",
    )
    audit.add_argument(
        "--cache-dir",
        help="override a relocated content-addressed raw cache directory",
    )
    audit.add_argument("--expected-members", type=int, default=300)
    audit.add_argument("--adjusted-price-tolerance", type=float, default=1e-8)
    audit.add_argument(
        "--adjusted-return-warning-threshold", type=float, default=0.30
    )
    audit.add_argument(
        "--adjusted-return-fail-threshold", type=float, default=1.00
    )
    audit.add_argument("--suspension-daily-warning-rate", type=float, default=0.10)
    audit.add_argument("--minimum-proxy-mcap-coverage", type=float, default=1.00)
    audit.add_argument("--maximum-samples", type=int, default=10)
    audit.set_defaults(func=command_data_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
