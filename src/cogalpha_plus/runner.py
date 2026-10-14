"""Experiment orchestration: manifest, ledger, registry, evolution and sealing."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .dsl import canonicalize
from .evolution import (
    AdaptiveFeedbackMemory,
    AdaptiveFeedbackPolicy,
    BehavioralChampionPolicy,
    Candidate,
    CandidateScore,
    CodexSeedInjectionFactory,
    EvaluatedCandidate,
    EvolutionEngine,
    EvolutionResult,
    MechanismAwareOperators,
    SearchBudget,
    TemplateCandidateFactory,
)
from .manifest import RunManifest, hash_json
from .registry import FactorLineage, FactorRecord, FactorRegistry, TrialLedger
from .metrics import normalize_trading_calendar
from .terminal_labels import TerminalLabelBridge
from .validation import (
    DevelopmentEvaluator,
    SealedGate,
    SplitPlan,
    build_research_universe_identity,
    research_rule_identity_v1,
)


def panel_snapshot_hash(panel: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(panel, index=True, categorize=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def score_payload(score: CandidateScore) -> dict[str, Any]:
    return _jsonable(asdict(score))


def experiment_config_payload(
    *,
    profile: str,
    split: SplitPlan,
    budget: SearchBudget,
    bootstrap_samples: int,
    adaptive_feedback_policy: AdaptiveFeedbackPolicy | None = None,
    behavioral_champion_policy: BehavioralChampionPolicy | None = None,
    development_handoff_policy: Mapping[str, object] | None = None,
    composite_family_policy: Mapping[str, object] | None = None,
    operational_locators: Mapping[str, object] | None = None,
    implementation_identity: Mapping[str, object] | None = None,
    research_universe_identity: Mapping[str, object] | None = None,
    trading_calendar_identity: Mapping[str, object] | None = None,
    source_lineage_identity: Mapping[str, object] | None = None,
    source_snapshot_identity: Mapping[str, object] | None = None,
    terminal_label_bridge_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the complete resolved configuration bound into a RunManifest."""

    if profile not in {"faithful", "plus"}:
        raise ValueError("profile must be faithful or plus")
    if profile == "faithful" and adaptive_feedback_policy is not None:
        raise ValueError(
            "faithful profile cannot receive the Plus adaptive feedback policy"
        )
    if profile == "faithful" and behavioral_champion_policy is not None:
        raise ValueError(
            "faithful profile cannot receive the Plus behavioral champion policy"
        )
    resolved_feedback_policy = (
        adaptive_feedback_policy or AdaptiveFeedbackPolicy()
        if profile == "plus"
        else None
    )
    resolved_behavioral_policy = (
        behavioral_champion_policy or BehavioralChampionPolicy()
        if profile == "plus"
        else None
    )

    payload: dict[str, object] = {
        "profile": profile,
        "split": split.to_dict(),
        "budget": asdict(budget),
        "bootstrap_samples": bootstrap_samples,
        # Bind the values consumed by AdaptiveFeedbackMemory, rather than only
        # hashing the path of a declarative YAML file.
        "adaptive_feedback_policy": (
            asdict(resolved_feedback_policy)
            if resolved_feedback_policy is not None
            else None
        ),
        "behavioral_champion_policy": (
            asdict(resolved_behavioral_policy)
            if resolved_behavioral_policy is not None
            else None
        ),
        "development_handoff_policy": (
            dict(development_handoff_policy)
            if development_handoff_policy is not None
            else None
        ),
        "composite_family_policy": (
            dict(composite_family_policy)
            if composite_family_policy is not None
            else None
        ),
        "operational_locators": (
            dict(operational_locators) if operational_locators is not None else None
        ),
    }
    if implementation_identity is not None:
        payload["research_implementation_identity"] = _jsonable(
            implementation_identity
        )
    if research_universe_identity is not None:
        payload["research_universe_identity"] = _jsonable(
            research_universe_identity
        )
    if trading_calendar_identity is not None:
        payload["trading_calendar_identity"] = _jsonable(
            trading_calendar_identity
        )
    if source_lineage_identity is not None:
        payload["source_lineage_identity"] = _jsonable(source_lineage_identity)
    if source_snapshot_identity is not None:
        payload["source_snapshot_identity"] = _jsonable(source_snapshot_identity)
    if terminal_label_bridge_identity is not None:
        payload["terminal_label_bridge_identity"] = _jsonable(
            terminal_label_bridge_identity
        )
    payload["research_rule_identity"] = research_rule_identity_v1()
    return payload


class ExperimentRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        manifest: RunManifest,
        data_version: str,
        config_version: str,
        model_version: str,
        prompt_version: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.data_version = data_version
        self.config_version = config_version
        self.model_version = model_version
        self.prompt_version = prompt_version
        self.registry = FactorRegistry(self.output_dir / "registry")
        self.ledger = TrialLedger(self.output_dir / "ledger")
        manifest.save(self.output_dir / "manifest.json")
        self.ledger.append(
            "run_started",
            {"run_id": manifest.run_id, "manifest_hash": manifest.manifest_hash},
        )

    @staticmethod
    def _operation(evaluated: EvaluatedCandidate) -> str:
        candidate = evaluated.candidate
        if candidate.operator.startswith("crossover_mutation:"):
            return "crossover"
        if candidate.operator.startswith("mutation:"):
            return "mutation"
        if candidate.operator.startswith("crossover:"):
            return "crossover"
        if candidate.generation > 0:
            return "injection"
        return "seed"

    def record(self, evaluated: EvaluatedCandidate) -> None:
        candidate, score = evaluated.candidate, evaluated.score
        operation = self._operation(evaluated)
        lineage = FactorLineage(
            parent_factor_ids=tuple(candidate.parents),
            operation=operation,
            operator=candidate.operator if operation in {"mutation", "crossover"} else "",
            agent_id=candidate.agent_id,
            generation=candidate.generation,
        )
        try:
            expression = canonicalize(candidate.expression)
        except Exception:
            expression = candidate.expression
        metrics = score_payload(score)
        rejection = str(score.diagnostics.get("rejection", "validation_gate_failed")) if not score.passed else ""
        record = FactorRecord.create(
            name=candidate.hypothesis,
            formula_ast=expression,
            status="success" if score.passed else "failed",
            lineage=lineage,
            run_id=self.manifest.run_id,
            data_version=self.data_version,
            config_version=self.config_version,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
            metrics=metrics,
            error=rejection,
            metadata={
                "candidate_id": candidate.candidate_id,
                "mechanism_tags": list(candidate.mechanism_tags),
                "control_expression": candidate.control_expression,
                "candidate_metadata": candidate.metadata,
            },
        )
        registration = self.registry.register(record)
        self.ledger.append(
            "candidate_evaluated",
            {
                "candidate_id": candidate.candidate_id,
                "factor_id": registration.record.factor_id,
                "inserted": registration.inserted,
                "generation": candidate.generation,
                "operator": candidate.operator,
                "score": metrics,
            },
        )

    def record_terminal_audit(self, audit: Mapping[str, object]) -> None:
        """Persist a fail-closed search audit even when no summary is produced."""

        payload = _jsonable(audit)
        self.ledger.append("search_terminal_audit", {"audit": payload})
        target = self.output_dir / "terminal_audit.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def finish(
        self,
        result: EvolutionResult,
        sealed: list[dict[str, object]] | None = None,
        development_label_audit: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        selection_records = result.champion_selection.get("champions", [])
        relationship_by_id = {
            str(item.get("candidate_id")): item
            for item in selection_records
            if isinstance(item, Mapping) and item.get("candidate_id") is not None
        }
        summary: dict[str, object] = {
            "run_id": self.manifest.run_id,
            "manifest_hash": self.manifest.manifest_hash,
            "evaluated_candidates": len(result.archive),
            "registry": self.registry.stats(),
            "champions": [
                {
                    "candidate_id": item.candidate.candidate_id,
                    "agent_id": item.candidate.agent_id,
                    "operator": item.candidate.operator,
                    "expression_hash": item.candidate.expression_hash,
                    "fitness": item.score.fitness,
                    "validation_rank_ic": item.score.signed_rank_ic,
                    "validation_neutral_rank_ic": item.score.neutral_rank_ic,
                    "control_delta": item.score.control_delta,
                    "nearest_champion_candidate_id": relationship_by_id.get(
                        item.candidate.candidate_id,
                        {},
                    ).get("nearest_champion_candidate_id"),
                    "maximum_absolute_behavioral_correlation": relationship_by_id.get(
                        item.candidate.candidate_id,
                        {},
                    ).get("maximum_absolute_correlation"),
                }
                for item in result.champions
            ],
            "generations": list(result.generations),
            "adaptive_feedback": _jsonable(result.adaptive_feedback),
            "champion_selection": _jsonable(result.champion_selection),
            "sealed_revealed": sealed is not None,
            "sealed": sealed,
            "development_label_audit": _jsonable(development_label_audit),
        }
        self.ledger.append(
            "run_finished",
            {
                "champion_candidate_ids": [item.candidate.candidate_id for item in result.champions],
                "adaptive_feedback": _jsonable(result.adaptive_feedback),
                "champion_selection": _jsonable(result.champion_selection),
                "sealed_revealed": sealed is not None,
                "development_label_audit": _jsonable(development_label_audit),
            },
        )
        verification = self.ledger.verify()
        summary["ledger"] = {
            "valid": verification.valid,
            "entries": verification.entries,
            "head_hash": verification.head_hash,
        }
        target = self.output_dir / "summary.json"
        target.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary


def run_experiment(
    *,
    full_panel: pd.DataFrame,
    feature_tape: pd.DataFrame | None = None,
    source_member_panel: pd.DataFrame | None = None,
    trading_calendar: pd.DatetimeIndex | None = None,
    trading_calendar_identity: Mapping[str, object] | None = None,
    research_universe_identity: Mapping[str, object] | None = None,
    source_lineage_identity: Mapping[str, object] | None = None,
    source_snapshot_identity: Mapping[str, object] | None = None,
    allow_test_calendar_from_panel: bool = False,
    split: SplitPlan,
    output_dir: str | Path,
    budget: SearchBudget,
    profile: str = "plus",
    seed: int = 17,
    reveal_sealed: bool = False,
    bootstrap_samples: int = 100,
    model_version: str = "codex-desktop-interactive",
    prompt_version: str = "cogalpha-seven-level-v1",
    initial_candidates: list[Candidate] | None = None,
    injection_candidates: list[Candidate] | None = None,
    adaptive_feedback_policy: AdaptiveFeedbackPolicy | None = None,
    behavioral_champion_policy: BehavioralChampionPolicy | None = None,
    development_handoff_policy: Mapping[str, object] | None = None,
    composite_family_policy: Mapping[str, object] | None = None,
    operational_locators: Mapping[str, object] | None = None,
    implementation_identity: Mapping[str, object] | None = None,
    terminal_label_bridges: tuple[TerminalLabelBridge, ...] = (),
    terminal_label_bridge_identity: Mapping[str, object] | None = None,
) -> tuple[EvolutionResult, dict[str, object]]:
    if profile not in {"faithful", "plus"}:
        raise ValueError("profile must be faithful or plus")
    if profile == "faithful" and adaptive_feedback_policy is not None:
        raise ValueError(
            "faithful profile cannot receive the Plus adaptive feedback policy"
        )
    if profile == "faithful" and behavioral_champion_policy is not None:
        raise ValueError(
            "faithful profile cannot receive the Plus behavioral champion policy"
        )
    resolved_feedback_policy = (
        adaptive_feedback_policy or AdaptiveFeedbackPolicy()
        if profile == "plus"
        else None
    )
    resolved_behavioral_policy = (
        behavioral_champion_policy or BehavioralChampionPolicy()
        if profile == "plus"
        else None
    )
    eligible_hash = panel_snapshot_hash(full_panel)
    tape = full_panel if feature_tape is None else feature_tape.sort_index()
    if not full_panel.index.isin(tape.index).all():
        raise ValueError("continuous feature tape does not cover every eligible panel row")
    tape_hash = panel_snapshot_hash(tape)
    member_source = (
        full_panel if source_member_panel is None else source_member_panel.sort_index()
    )
    member_hash = panel_snapshot_hash(member_source)
    calendar = normalize_trading_calendar(
        trading_calendar,
        panel_index=tape.index,
        allow_test_calendar_from_panel=allow_test_calendar_from_panel,
    )
    derived_calendar_identity: dict[str, object] = {
        "schema_version": 1,
        "source": (
            "synthetic_panel_explicit_test_fallback"
            if trading_calendar is None
            else "caller_supplied_frozen_calendar"
        ),
        "trading_dates_hash": hash_json(
            [str(pd.Timestamp(value).date()) for value in calendar]
        ),
        "trading_days": int(len(calendar)),
        "first_date": str(calendar[0].date()),
        "last_date": str(calendar[-1].date()),
        "endpoint_rule_hash": research_rule_identity_v1()["rule_hash"],
    }
    derived_calendar_identity["identity_hash"] = hash_json(
        derived_calendar_identity
    )
    if trading_calendar_identity is not None:
        supplied = dict(trading_calendar_identity)
        supplied_payload = {
            key: value for key, value in supplied.items() if key != "identity_hash"
        }
        if (
            supplied.get("identity_hash") != hash_json(supplied_payload)
            or supplied.get("trading_dates_hash")
            != derived_calendar_identity["trading_dates_hash"]
        ):
            raise ValueError("trading-calendar identity differs from supplied dates")
        calendar_identity = supplied
    else:
        calendar_identity = derived_calendar_identity
    derived_universe_identity = build_research_universe_identity(
        member_source,
        tape,
        full_panel,
    )
    if research_universe_identity is not None:
        supplied_universe = dict(research_universe_identity)
        supplied_universe_payload = {
            key: value
            for key, value in supplied_universe.items()
            if key != "identity_hash"
        }
        if supplied_universe.get("identity_hash") != hash_json(
            supplied_universe_payload
        ):
            raise ValueError("research-universe identity hash mismatch")
        for name in (
            "full_member_panel_hash",
            "eligible_panel_hash",
            "full_member_eligibility_mask_hash",
            "feature_tape_eligibility_mask_hash",
            "eligible_daily_counts_hash",
            "rule_hash",
        ):
            if supplied_universe.get(name) != derived_universe_identity.get(name):
                raise ValueError(f"research-universe identity mismatch: {name}")
        universe_identity = supplied_universe
    else:
        universe_identity = derived_universe_identity
    if source_lineage_identity is None:
        if not allow_test_calendar_from_panel:
            raise ValueError(
                "real experiment requires the frozen source-lineage identity"
            )
        test_lineage_payload = {
            "schema_version": "test-only",
            "complete": True,
            "source": "synthetic_explicit_test_fallback",
            "root_sha256": hash_json(
                {
                    "member_panel_hash": member_hash,
                    "feature_tape_hash": tape_hash,
                }
            ),
        }
        lineage_identity = {
            **test_lineage_payload,
            "identity_hash": hash_json(test_lineage_payload),
        }
    else:
        supplied_lineage = dict(source_lineage_identity)
        lineage_payload = {
            key: value
            for key, value in supplied_lineage.items()
            if key != "identity_hash"
        }
        if supplied_lineage.get("identity_hash") != hash_json(lineage_payload):
            raise ValueError("source-lineage identity hash mismatch")
        lineage_identity = supplied_lineage
    if source_snapshot_identity is None:
        if not allow_test_calendar_from_panel:
            raise ValueError(
                "real experiment requires the full frozen source-snapshot identity"
            )
        test_source_payload = {
            "schema_version": "test-only",
            "complete": True,
            "member_panel_hash": member_hash,
            "feature_tape_hash": tape_hash,
            "research_universe_identity_hash": universe_identity["identity_hash"],
            "trading_calendar_identity_hash": calendar_identity["identity_hash"],
            "source_lineage_identity_hash": lineage_identity["identity_hash"],
        }
        source_identity = {
            **test_source_payload,
            "identity_hash": hash_json(test_source_payload),
        }
    else:
        supplied_source = dict(source_snapshot_identity)
        source_payload = {
            key: value
            for key, value in supplied_source.items()
            if key != "identity_hash"
        }
        if supplied_source.get("identity_hash") != hash_json(source_payload):
            raise ValueError("source-snapshot identity hash mismatch")
        if supplied_source.get("source_lineage_identity") != lineage_identity:
            raise ValueError(
                "source snapshot differs from the frozen source-lineage identity"
            )
        source_identity = supplied_source
    data_hash = hash_json(
        {
            "full_member_panel_hash": member_hash,
            "eligible_panel_hash": eligible_hash,
            "feature_tape_hash": tape_hash,
            "research_universe_identity_hash": universe_identity["identity_hash"],
            "trading_calendar_identity_hash": calendar_identity["identity_hash"],
            "source_lineage_identity_hash": lineage_identity["identity_hash"],
            "source_snapshot_identity_hash": source_identity["identity_hash"],
        }
    )
    config = experiment_config_payload(
        profile=profile,
        split=split,
        budget=budget,
        bootstrap_samples=bootstrap_samples,
        adaptive_feedback_policy=resolved_feedback_policy,
        behavioral_champion_policy=resolved_behavioral_policy,
        development_handoff_policy=development_handoff_policy,
        composite_family_policy=composite_family_policy,
        operational_locators=operational_locators,
        implementation_identity=implementation_identity,
        research_universe_identity=universe_identity,
        trading_calendar_identity=calendar_identity,
        source_lineage_identity=lineage_identity,
        source_snapshot_identity=source_identity,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
    )
    config_hash = hash_json(config)
    manifest = RunManifest.create(
        seed=seed,
        data_snapshot={
            "kind": "full_member_panel_plus_t_eligible_subset_plus_continuous_union_tape",
            "rows": len(member_source),
            "first_date": str(member_source.index.get_level_values("date").min().date()),
            "last_date": str(member_source.index.get_level_values("date").max().date()),
            "member_panel_sha256": member_hash,
            "eligible_panel_rows": len(full_panel),
            "eligible_panel_sha256": eligible_hash,
            "feature_tape_rows": len(tape),
            "feature_tape_first_date": str(tape.index.get_level_values("date").min().date()),
            "feature_tape_last_date": str(tape.index.get_level_values("date").max().date()),
            "feature_tape_sha256": tape_hash,
            "research_universe_identity": universe_identity,
            "trading_calendar_identity": calendar_identity,
            "source_lineage_identity": lineage_identity,
            "source_snapshot_identity": source_identity,
            "sha256": data_hash,
        },
        search_budget={
            "initial_population": budget.initial_population,
            "parent_pool": budget.parent_pool,
            "children_per_generation": budget.children_per_generation,
            "generations": budget.generations,
            "injection_every": budget.injection_every,
            "injection_count": budget.injection_count,
            "elite_carry": budget.elite_carry,
            "final_factors": budget.final_factors,
        },
        config_hash=config_hash,
        model_version=model_version,
        prompt_version=prompt_version,
    )
    recorder = ExperimentRecorder(
        output_dir,
        manifest=manifest,
        data_version=data_hash,
        config_version=config_hash,
        model_version=model_version,
        prompt_version=prompt_version,
    )
    development = full_panel.loc[full_panel.index.get_level_values("date") <= split.development_end]
    development_tape = tape.loc[
        tape.index.get_level_values("date") <= split.development_end
    ]
    evaluator = DevelopmentEvaluator(
        development,
        split,
        feature_tape=development_tape,
        trading_calendar=calendar,
        min_names=min(20, full_panel.index.get_level_values("symbol").nunique() - 2),
        development_handoff_policy=development_handoff_policy,
        terminal_label_bridges=terminal_label_bridges,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
        bootstrap_samples=bootstrap_samples,
    )
    candidate_factory = (
        CodexSeedInjectionFactory(
            initial_candidates,
            injection_cards=injection_candidates,
            seed=seed,
        )
        if initial_candidates is not None
        else TemplateCandidateFactory(seed=seed)
    )
    engine = EvolutionEngine(
        evaluator=evaluator,
        factory=candidate_factory,
        operators=MechanismAwareOperators(seed=seed, max_nodes=48),
        budget=budget,
        mode=profile,
        seed=seed,
        on_evaluated=recorder.record,
        feedback_memory=(
            AdaptiveFeedbackMemory(resolved_feedback_policy)
            if resolved_feedback_policy is not None
            else None
        ),
        champion_selector=(
            evaluator.select_behaviorally_distinct_champions
            if resolved_behavioral_policy is not None
            else None
        ),
        behavioral_champion_policy=resolved_behavioral_policy,
        on_terminal_audit=recorder.record_terminal_audit,
    )
    result = engine.run(initial_candidates)
    sealed: list[dict[str, object]] | None = None
    if reveal_sealed:
        gate = SealedGate(
            full_panel,
            split,
            feature_tape=tape,
            trading_calendar=calendar,
            research_universe_identity=universe_identity,
            trading_calendar_identity=calendar_identity,
            min_names=evaluator.min_names,
            terminal_label_bridges=terminal_label_bridges,
            terminal_label_bridge_identity=terminal_label_bridge_identity,
        )
        sealed = gate.reveal(result.champions, evaluator.artifacts)
        recorder.ledger.append("sealed_gate_revealed_and_retired", {"results": sealed})
    summary = recorder.finish(
        result,
        sealed,
        development_label_audit=evaluator.development_label_audit,
    )
    return result, summary
