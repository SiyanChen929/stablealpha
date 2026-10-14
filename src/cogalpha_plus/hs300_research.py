"""Auditable orchestration for the real-data HS300 factor experiment.

The historical 2020--2025 segment is deliberately called a *replay*, not a
true out-of-sample period: those market outcomes already exist when this code
is being written.  It is nevertheless isolated from factor generation so that
the experiment cannot adapt to it accidentally.  A genuinely unseen test only
starts after the system manifest has been frozen and run prospectively.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .data_sources.hs300_audit import HS300AuditConfig, audit_hs300_research_panel
from .dsl import canonical_json as canonical_expression_json
from .dsl import expression_hash as canonical_expression_hash
from .dsl import strict_boolean_mask
from .evolution import (
    AdaptiveFeedbackPolicy,
    BehavioralChampionPolicy,
    Candidate,
    CandidateScore,
    EvaluatedCandidate,
    EvolutionResult,
    SearchBudget,
    candidate_roles,
    deterministic_candidate_id,
)
from .manifest import RunManifest, canonical_json, hash_json, utc_now
from .proposals import (
    build_agent_input_contract,
    cards_to_candidates,
    load_factor_cards,
)
from .registry import FactorRegistry, TrialLedger
from .research_proxy import (
    CompositeFamilyPolicy,
    DevelopmentHandoffPolicy,
    FrozenDevelopmentCommitment,
    SealedCompositeClaimCapability,
    build_development_handoff_from_commitment,
    dump_frozen_composite_family,
    fit_frozen_development_composite_family,
    load_frozen_composite_family,
)
from .runner import experiment_config_payload, panel_snapshot_hash, run_experiment
from .semantic_audit import validate_semantic_review_manifest
from .terminal_labels import TerminalLabelBridge, load_terminal_label_bridges
from .validation import (
    FactorArtifact,
    PRODUCTION_DSL_SPEC,
    SealedGate,
    SplitPlan,
    build_research_universe_identity,
    research_rule_identity_v1,
    sealed_reporting_contract_v1,
)


HISTORICAL_REPLAY_WARNING = (
    "2020-2025 is an isolated historical replay, not a genuinely unseen OOS test; "
    "true OOS begins only after the frozen system runs prospectively"
)


SEALED_REVEAL_SCHEMA_VERSION = 7

DEVELOPMENT_COMPLETION_EVENT = "development_champion_family_frozen"
DEVELOPMENT_EVIDENCE_LOCATORS = {
    "run_manifest": "manifest.json",
    "summary": "summary.json",
    "registry": "registry/factors.jsonl",
    "ledger": "ledger/trials.jsonl",
}

SEMANTIC_REVIEW_RELATIVE_PATH = "reviews/semantic_review_176.json"

# Only files which can change factor discovery, validation, input lineage or
# sealed scoring are frozen here.  Downstream portfolio/execution modules are
# deliberately excluded so they can evolve after factor discovery without
# silently changing the meaning of the one-time factor replay.
RESEARCH_IMPLEMENTATION_FILES = (
    "pyproject.toml",
    "configs/agents.yaml",
    "configs/faithful.yaml",
    "configs/plus.yaml",
    "configs/hs300_terminal_label_bridges.yaml",
    "proposals/codex_seed/l1_l2.jsonl",
    "proposals/codex_seed/l3_l4.jsonl",
    "proposals/codex_seed/l5_l7.jsonl",
    "proposals/codex_injection/l1_l4.jsonl",
    "proposals/codex_injection/l5_l7.jsonl",
    SEMANTIC_REVIEW_RELATIVE_PATH,
    "src/cogalpha_plus/data_contract.py",
    "src/cogalpha_plus/data_quality.py",
    "src/cogalpha_plus/data_sources/baostock.py",
    "src/cogalpha_plus/data_sources/hs300.py",
    "src/cogalpha_plus/data_sources/hs300_audit.py",
    "src/cogalpha_plus/data_sources/hs300_build.py",
    "src/cogalpha_plus/data_sources/hs300_download.py",
    "src/cogalpha_plus/dsl.py",
    "src/cogalpha_plus/ensemble.py",
    "src/cogalpha_plus/evolution.py",
    "src/cogalpha_plus/hs300_research.py",
    "src/cogalpha_plus/manifest.py",
    "src/cogalpha_plus/metrics.py",
    "src/cogalpha_plus/proposals.py",
    "src/cogalpha_plus/registry.py",
    "src/cogalpha_plus/research_proxy.py",
    "src/cogalpha_plus/runner.py",
    "src/cogalpha_plus/semantic_audit.py",
    "src/cogalpha_plus/terminal_labels.py",
    "src/cogalpha_plus/validation.py",
)

RESEARCH_RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "PyYAML",
    "pyarrow",
    "lightgbm",
)

ADAPTIVE_REPAIR_TAGS = frozenset(
    {
        "repair_shorten_lookback",
        "repair_persistence_smoothing",
        "repair_stability_smoothing",
        "repair_style_robustification",
        "repair_conservative_repair",
        "repair_novelty_recombination",
        "repair_nonlinear_refinement",
        "repair_parameter_refinement",
    }
)


_ADAPTIVE_FEEDBACK_POLICY_FIELDS = tuple(
    field.name for field in fields(AdaptiveFeedbackPolicy)
)

_BEHAVIORAL_CHAMPION_POLICY_FIELDS = tuple(
    field.name for field in fields(BehavioralChampionPolicy)
)


def _composite_policy_payload(
    policy: CompositeFamilyPolicy,
) -> dict[str, object]:
    return policy.to_dict()


def _development_handoff_policy_payload(
    policy: DevelopmentHandoffPolicy,
) -> dict[str, object]:
    return policy.to_dict()


def _load_composite_policies(
    project_dir: Path | str,
    profile: str,
) -> tuple[DevelopmentHandoffPolicy, CompositeFamilyPolicy]:
    if profile not in {"faithful", "plus"}:
        raise ValueError("profile must be faithful or plus")
    path = Path(project_dir).resolve() / "configs" / f"{profile}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ensemble = raw.get("ensemble") if isinstance(raw, Mapping) else None
    if not isinstance(ensemble, Mapping):
        raise ValueError(f"configs/{profile}.yaml omits ensemble policy")
    expected_keys = {
        "scope",
        "invoked_by_hs300_factor_discovery_runtime",
        "status",
        "handoff",
        "composite_family",
        "validation_metrics_usage",
        "backend_selection_allowed",
        "pre_registered_contrasts",
        "sealed_reporting",
        "reporting_contract",
    }
    if set(ensemble) != expected_keys:
        raise ValueError(f"configs/{profile}.yaml ensemble keys are not exact")
    if (
        ensemble.get("scope") != "frozen_development_family_joint_one_time_reveal"
        or ensemble.get("invoked_by_hs300_factor_discovery_runtime") is not True
        or ensemble.get("status") != "fixed_family_no_backend_selection"
        or ensemble.get("sealed_reporting")
        != (
            "unique_lightgbm_primary_with_joint_23_fwer_and_"
            "pre_registered_secondaries"
        )
    ):
        raise ValueError(f"configs/{profile}.yaml ensemble declarations are unsafe")
    handoff = ensemble.get("handoff")
    if not isinstance(handoff, Mapping) or set(handoff) != set(
        DevelopmentHandoffPolicy().to_dict()
    ):
        raise ValueError(f"configs/{profile}.yaml handoff policy is not exact")
    development_policy = DevelopmentHandoffPolicy(**dict(handoff))
    composite_values = ensemble.get("composite_family")
    if not isinstance(composite_values, Mapping) or set(composite_values) != set(
        CompositeFamilyPolicy.fixed_v1().to_dict()
    ):
        raise ValueError(f"configs/{profile}.yaml composite policy is not exact")
    resolved_composite = dict(composite_values)
    members = resolved_composite.get("members")
    if not isinstance(members, list):
        raise ValueError(f"configs/{profile}.yaml members must be a list")
    resolved_composite["members"] = tuple(str(item) for item in members)
    configuration = resolved_composite.get("lightgbm_configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError(f"configs/{profile}.yaml LightGBM configuration is invalid")
    resolved_composite["lightgbm_configuration"] = dict(configuration)
    composite_policy = CompositeFamilyPolicy(**resolved_composite)
    if ensemble.get("validation_metrics_usage") != (
        "descriptive_only_never_selection_or_tuning"
    ) or ensemble.get("backend_selection_allowed") is not False:
        raise ValueError(f"configs/{profile}.yaml permits backend selection")
    if ensemble.get("pre_registered_contrasts") != [
        "lightgbm_minus_equal_weight",
        "ridge_minus_equal_weight",
        "lightgbm_minus_ridge",
    ]:
        raise ValueError(f"configs/{profile}.yaml contrast family is not exact")
    if ensemble.get("reporting_contract") != sealed_reporting_contract_v1():
        raise ValueError(f"configs/{profile}.yaml reporting contract is not exact")
    return development_policy, composite_policy


def _adaptive_feedback_policy_payload(
    policy: AdaptiveFeedbackPolicy | None,
) -> dict[str, float] | None:
    if policy is None:
        return None
    return {
        name: float(value)
        for name, value in asdict(policy).items()
    }


def _behavioral_champion_policy_payload(
    policy: BehavioralChampionPolicy | None,
) -> dict[str, float | int] | None:
    if policy is None:
        return None
    return {
        "maximum_absolute_correlation": float(
            policy.maximum_absolute_correlation
        ),
        "initial_top_pool_size": int(policy.initial_top_pool_size),
    }


def _load_behavioral_champion_policy(
    project_dir: Path | str,
    profile: str,
) -> BehavioralChampionPolicy | None:
    """Resolve the exact Plus final-family behavior gate from its YAML."""

    if profile == "faithful":
        return None
    if profile != "plus":
        raise ValueError("profile must be faithful or plus")
    path = Path(project_dir).resolve() / "configs" / "plus.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("profile_id") != "cogalpha_plus":
        raise ValueError("configs/plus.yaml is not the cogalpha_plus profile")
    deduplication = raw.get("deduplication")
    behavioral = (
        deduplication.get("behavioral")
        if isinstance(deduplication, Mapping)
        else None
    )
    if not isinstance(behavioral, Mapping) or behavioral.get("enabled") is not True:
        raise ValueError("Plus behavioral champion gate must be enabled")
    if behavioral.get("scope") != "final_champion_selection_inner_validation_only":
        raise ValueError("Plus behavioral champion gate has an unexpected scope")
    if behavioral.get("metric") != "mean_daily_cross_sectional_spearman":
        raise ValueError("Plus behavioral champion gate has an unexpected metric")
    if behavioral.get("threshold_relaxation") is not False:
        raise ValueError("Plus behavioral champion threshold relaxation must be false")
    if behavioral.get("insufficient_family") != "fail_closed":
        raise ValueError("Plus behavioral champion underfill policy must fail closed")

    values = {
        name: behavioral.get(name)
        for name in _BEHAVIORAL_CHAMPION_POLICY_FIELDS
    }
    threshold = values["maximum_absolute_correlation"]
    pool_size = values["initial_top_pool_size"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("Plus behavioral correlation threshold must be finite numeric")
    if (
        isinstance(pool_size, bool)
        or not isinstance(pool_size, int)
    ):
        raise ValueError("Plus behavioral initial top pool size must be an integer")
    policy = BehavioralChampionPolicy(
        maximum_absolute_correlation=float(threshold),
        initial_top_pool_size=pool_size,
    )
    if _behavioral_champion_policy_payload(policy) != {
        "maximum_absolute_correlation": float(threshold),
        "initial_top_pool_size": int(pool_size),
    }:
        raise RuntimeError("Plus behavioral YAML did not resolve exactly into runtime policy")
    return policy


def _load_adaptive_feedback_policy(
    project_dir: Path | str,
    profile: str,
) -> AdaptiveFeedbackPolicy | None:
    """Resolve the exact Plus controller thresholds used by the runtime.

    Faithful mode deliberately has no local Plus feedback controller.  Plus
    mode fails closed on missing, extra, boolean or non-finite threshold values
    so a YAML declaration cannot silently drift away from the dataclass passed
    to ``AdaptiveFeedbackMemory``.
    """

    if profile == "faithful":
        return None
    if profile != "plus":
        raise ValueError("profile must be faithful or plus")
    path = Path(project_dir).resolve() / "configs" / "plus.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("profile_id") != "cogalpha_plus":
        raise ValueError("configs/plus.yaml is not the cogalpha_plus profile")
    hierarchy = raw.get("agent_hierarchy")
    feedback = hierarchy.get("adaptive_feedback") if isinstance(hierarchy, Mapping) else None
    if not isinstance(feedback, Mapping):
        raise ValueError("Plus profile omits agent_hierarchy.adaptive_feedback")
    if feedback.get("enabled") is not True:
        raise ValueError("Plus adaptive feedback controller must be enabled")
    if feedback.get("runtime_controller") != "AdaptiveFeedbackMemory":
        raise ValueError("Plus adaptive feedback runtime controller is not exact")
    thresholds = feedback.get("fixed_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("Plus adaptive feedback fixed_thresholds are missing")
    expected = set(_ADAPTIVE_FEEDBACK_POLICY_FIELDS)
    actual = {str(name) for name in thresholds}
    if actual != expected:
        raise ValueError(
            "Plus adaptive feedback threshold keys differ from "
            f"AdaptiveFeedbackPolicy: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    resolved: dict[str, float] = {}
    for name in _ADAPTIVE_FEEDBACK_POLICY_FIELDS:
        value = thresholds[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Plus adaptive feedback threshold {name} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Plus adaptive feedback threshold {name} must be finite")
        resolved[name] = number
    bounded_zero_one = (
        "minimum_coverage",
        "minimum_style_retention",
        "minimum_novelty",
    )
    if any(not 0.0 <= resolved[name] <= 1.0 for name in bounded_zero_one):
        raise ValueError("coverage, style-retention and novelty thresholds must be in [0, 1]")
    if resolved["maximum_turnover"] < 0.0:
        raise ValueError("maximum_turnover must be non-negative")
    if any(
        not -1.0 <= resolved[name] <= 1.0
        for name in ("minimum_worst_window_ic", "minimum_control_delta")
    ):
        raise ValueError("IC and control-delta thresholds must be in [-1, 1]")
    policy = AdaptiveFeedbackPolicy(**resolved)
    if _adaptive_feedback_policy_payload(policy) != resolved:
        raise RuntimeError("Plus YAML thresholds did not resolve exactly into runtime policy")
    return policy


def _research_module_name(relative_name: str) -> str | None:
    prefix = "src/cogalpha_plus/"
    if not relative_name.startswith(prefix) or not relative_name.endswith(".py"):
        return None
    suffix = relative_name[len(prefix) : -3]
    return "cogalpha_plus." + suffix.replace("/", ".")


@dataclass(frozen=True)
class HS300ResearchConfig:
    project_dir: Path | str
    panel_dir: Path | str
    output_dir: Path | str
    train_end: str = "2017-12-31"
    validation_end: str = "2019-12-31"
    replay_end: str = "2025-12-31"
    label_horizon: int = 10
    entry_delay: int = 1
    profile: str = "plus"
    seed: int = 17
    bootstrap_samples: int = 200
    evaluation_workers: int = 1
    reveal_historical_replay: bool = False
    expected_members: int = 300
    reveal_registry_dir: Path | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_dir", Path(self.project_dir).resolve())
        object.__setattr__(self, "panel_dir", Path(self.panel_dir).resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        registry = Path(
            self.project_dir, "runs", "replay_reveal_registry"
        ).resolve()
        if (
            self.reveal_registry_dir is not None
            and Path(self.reveal_registry_dir).resolve() != registry
        ):
            raise ValueError(
                "reveal registry is fixed at project/runs/replay_reveal_registry"
            )
        object.__setattr__(self, "reveal_registry_dir", registry)
        train = pd.Timestamp(self.train_end).normalize()
        valid = pd.Timestamp(self.validation_end).normalize()
        replay = pd.Timestamp(self.replay_end).normalize()
        if not train < valid < replay:
            raise ValueError("train_end, validation_end and replay_end must be strictly ordered")
        if self.label_horizon < 1 or self.entry_delay < 0:
            raise ValueError("invalid label timing")
        if self.profile not in {"faithful", "plus"}:
            raise ValueError("profile must be faithful or plus")
        if self.bootstrap_samples < 20:
            raise ValueError("real-data research requires at least 20 bootstrap samples")
        if not 1 <= self.evaluation_workers <= 8:
            raise ValueError("evaluation_workers must be between 1 and 8")
        if self.expected_members < 1:
            raise ValueError("expected_members must be positive")
        if self.reveal_historical_replay:
            raise ValueError(
                "development search cannot reveal historical replay data; use the "
                "separate 'research hs300-reveal' command after the sealed manifest "
                "has been frozen"
            )


def _trusted_project_root() -> Path:
    """Return the root of the implementation bytes executing this command."""

    return Path(__file__).resolve().parents[2]


def _operational_locators(config: HS300ResearchConfig) -> dict[str, str]:
    return {
        "project_dir": str(config.project_dir),
        "panel_dir": str(config.panel_dir),
        "reveal_registry_dir": str(config.reveal_registry_dir),
        "sealed_output_dir": str(config.output_dir / "sealed_reveal"),
    }


@dataclass(frozen=True)
class FrozenPanel:
    manifest: Mapping[str, object]
    panel: pd.DataFrame
    eligible_panel: pd.DataFrame
    feature_tape: pd.DataFrame
    research_universe_identity: Mapping[str, object]
    source_lineage_identity: Mapping[str, object]
    source_snapshot_identity: Mapping[str, object]
    trading_calendar: pd.DatetimeIndex
    trading_calendar_identity: Mapping[str, object]
    manifest_sha256: str


@dataclass(frozen=True)
class HS300ResearchResult:
    evolution: EvolutionResult
    summary: Mapping[str, object]
    split: SplitPlan
    data_snapshot_hash: str
    historical_replay_warning: str = HISTORICAL_REPLAY_WARNING


@dataclass(frozen=True)
class HS300RevealResult:
    results: tuple[Mapping[str, object], ...]
    output_path: Path
    registry_record: Path
    development_manifest_hash: str
    data_snapshot_hash: str
    split: SplitPlan
    historical_replay_warning: str = HISTORICAL_REPLAY_WARNING


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _research_implementation_snapshot(project_dir: Path | str) -> dict[str, object]:
    """Freeze the exact research implementation used before sealed reveal.

    A data/test split is not reproducible when the adaptive controller, DSL,
    metric semantics, Factor Cards or their resolved configuration can drift
    between development and reveal.  The snapshot is intentionally content
    based and independent of git availability.
    """

    project = Path(project_dir).resolve()
    file_records: list[dict[str, object]] = []
    for relative_name in RESEARCH_IMPLEMENTATION_FILES:
        path = project / relative_name
        if not path.is_file():
            raise FileNotFoundError(
                f"research implementation file is missing: {relative_name}"
            )
        module_name = _research_module_name(relative_name)
        if module_name is not None:
            module = importlib.import_module(module_name)
            module_file = getattr(module, "__file__", None)
            if module_file is None or Path(module_file).resolve() != path.resolve():
                raise RuntimeError(
                    f"loaded research module {module_name} does not originate "
                    f"from the committed project file {relative_name}"
                )
        file_records.append(
            {
                "path": relative_name,
                "module": module_name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    runtime: dict[str, str] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable_name": Path(sys.executable).name,
        "platform_machine": platform.machine(),
    }
    for distribution in RESEARCH_RUNTIME_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required research runtime distribution is missing: {distribution}"
            ) from exc
        runtime[f"distribution:{distribution}"] = version
    content_identity = {
        "files": file_records,
        "runtime": runtime,
    }
    return {
        **content_identity,
        "aggregate_sha256": hash_json(content_identity),
    }


def _validate_research_implementation_snapshot(
    snapshot: object,
    *,
    project_dir: Path | str | None = None,
) -> dict[str, object]:
    if not isinstance(snapshot, Mapping):
        raise ValueError("development reveal manifest omits implementation identity")
    files = snapshot.get("files")
    runtime = snapshot.get("runtime")
    if not isinstance(files, list) or not isinstance(runtime, Mapping):
        raise ValueError("research implementation identity has invalid structure")
    expected_paths = list(RESEARCH_IMPLEMENTATION_FILES)
    actual_paths: list[str] = []
    for record in files:
        if not isinstance(record, Mapping):
            raise ValueError("research implementation file identity is invalid")
        relative_name = str(record.get("path", ""))
        module_name = record.get("module")
        digest = str(record.get("sha256", ""))
        size = record.get("bytes")
        if (
            not relative_name
            or Path(relative_name).is_absolute()
            or ".." in Path(relative_name).parts
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or module_name != _research_module_name(relative_name)
            or set(record) != {"path", "module", "bytes", "sha256"}
        ):
            raise ValueError("research implementation file identity is invalid")
        actual_paths.append(relative_name)
    if actual_paths != expected_paths or len(set(actual_paths)) != len(actual_paths):
        raise ValueError("research implementation file inventory is not canonical")
    required_runtime_keys = {
        "python_implementation",
        "python_version",
        "python_executable_name",
        "platform_machine",
        *(f"distribution:{name}" for name in RESEARCH_RUNTIME_DISTRIBUTIONS),
    }
    if set(str(key) for key in runtime) != required_runtime_keys or any(
        not str(value) for value in runtime.values()
    ):
        raise ValueError("research runtime identity is incomplete")
    content_identity = {
        "files": [dict(record) for record in files],
        "runtime": {str(key): str(value) for key, value in runtime.items()},
    }
    aggregate = str(snapshot.get("aggregate_sha256", ""))
    if aggregate != hash_json(content_identity):
        raise ValueError("research implementation aggregate identity mismatch")
    canonical = {**content_identity, "aggregate_sha256": aggregate}
    if project_dir is not None:
        current = _research_implementation_snapshot(project_dir)
        if current != canonical:
            raise ValueError(
                "research implementation changed after development freeze; "
                "sealed replay cannot be evaluated under different semantics"
            )
    return canonical


def _safe_partition_path(panel_dir: Path, declared: object) -> Path:
    candidate = Path(str(declared or ""))
    if not str(candidate):
        raise ValueError("frozen manifest contains a blank partition path")
    resolved = candidate.resolve() if candidate.is_absolute() else (panel_dir / candidate).resolve()
    allowed_root = (panel_dir / "snapshots").resolve()
    if not resolved.is_relative_to(allowed_root):
        raise ValueError("frozen partition path escapes panel/snapshots")
    return resolved


def _load_frozen_panel_manifest(panel_dir: Path | str) -> tuple[Path, dict[str, object], str]:
    root = Path(panel_dir).resolve()
    manifest_path = root / "manifest.json"
    digest_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        raise FileNotFoundError("frozen panel manifest or manifest.sha256 is missing")
    manifest_bytes = manifest_path.read_bytes()
    actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    tokens = digest_path.read_text(encoding="utf-8").strip().split()
    if len(tokens) < 1 or tokens[0] != actual_manifest_hash:
        raise ValueError("frozen panel manifest SHA-256 mismatch")
    payload = json.loads(manifest_bytes)
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError("frozen panel manifest is not complete")
    if payload.get("research_proxy_only") is not True:
        raise ValueError("free HS300 experiment requires an explicit research-proxy watermark")
    return root, payload, actual_manifest_hash


def _partition_year(record: Mapping[str, object]) -> int | None:
    raw = record.get("year")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen partition has an invalid year") from exc


def _read_frozen_partitions(
    root: Path,
    records: object,
    *,
    end_date: pd.Timestamp | None,
    label: str,
) -> pd.DataFrame:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} has no frozen partitions")
    frames: list[pd.DataFrame] = []
    seen: set[Path] = set()
    cutoff = end_date.normalize() if end_date is not None else None
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid {label} partition record")
        year = _partition_year(record)
        # A development process never opens partitions wholly after the
        # development boundary.  In the boundary year Arrow applies the date
        # predicate before pandas materializes any row, so replay outcomes are
        # not present in process memory.
        if cutoff is not None and year is not None and year > cutoff.year:
            continue
        path = _safe_partition_path(root, record.get("path"))
        if path in seen or not path.is_file():
            raise ValueError(f"{label} partition is duplicate or missing")
        seen.add(path)
        filtered = cutoff is not None and (year is None or year == cutoff.year)
        # Hashing a boundary-year file would stream every later price byte into
        # the development process even though Arrow filters its rows.  Leave
        # the full-file checksum to the reveal process; the development view's
        # semantic hash is frozen below and reveal revalidates both identities.
        if not filtered and _sha256_file(path) != str(record.get("sha256", "")):
            raise ValueError(f"{label} partition SHA-256 mismatch: {path.name}")
        frame = (
            pd.read_parquet(path, filters=[("date", "<=", cutoff)])
            if filtered
            else pd.read_parquet(path)
        )
        if not filtered and len(frame) != int(record.get("rows", -1)):
            raise ValueError(f"{label} partition row-count mismatch: {path.name}")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError(f"{label} has no rows at or before the requested boundary")
    output = pd.concat(frames).sort_index()
    if cutoff is not None and output.index.get_level_values("date").max() > cutoff:
        raise AssertionError(f"{label} loader materialized data after its hard boundary")
    return output


def _declared_terminal_sidecar_identity(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    raw = manifest.get("terminal_stale_membership_exceptions")
    if not isinstance(raw, Mapping) or raw.get("complete") is not True:
        raise ValueError("frozen manifest omits terminal-stale sidecar identity")
    file_record = raw.get("file")
    if not isinstance(file_record, Mapping):
        raise ValueError("terminal-stale sidecar file identity is invalid")
    storage_snapshot_id = str(manifest.get("storage_snapshot_id", "")).strip()
    if (
        not storage_snapshot_id
        or storage_snapshot_id in {".", ".."}
        or Path(storage_snapshot_id).name != storage_snapshot_id
    ):
        raise ValueError("terminal-stale sidecar storage snapshot identity is invalid")
    relative_path = str(
        Path("snapshots")
        / storage_snapshot_id
        / "terminal_stale_membership_exceptions.parquet"
    )
    declared_path = Path(str(file_record.get("path", "")))
    if tuple(declared_path.parts[-3:]) != tuple(Path(relative_path).parts):
        raise ValueError("terminal-stale sidecar does not use its fixed snapshot path")
    payload = {
        "schema_version": str(raw.get("schema_version", "")),
        "policy": str(raw.get("policy", "")),
        "price_fill_policy": str(raw.get("price_fill_policy", "")),
        "terminal_return_policy": str(raw.get("terminal_return_policy", "")),
        "rows": int(raw.get("rows", -1)),
        "symbols": int(raw.get("symbols", -1)),
        "trading_days": int(raw.get("trading_days", -1)),
        "snapshot_hash": str(raw.get("snapshot_hash", "")),
        "file_relative_path": relative_path,
        "file_sha256": str(file_record.get("sha256", "")),
        "file_rows": int(file_record.get("rows", -1)),
        "file_bytes": int(file_record.get("bytes", -1)),
    }
    for name in ("snapshot_hash", "file_sha256"):
        if len(str(payload[name])) != 64:
            raise ValueError("terminal-stale sidecar SHA-256 identity is incomplete")
    if (
        payload["rows"] < 0
        or payload["rows"] != payload["file_rows"]
        or payload["symbols"] < 0
        or payload["trading_days"] < 0
        or payload["file_bytes"] < 0
    ):
        raise ValueError("terminal-stale sidecar row identity is inconsistent")
    return {**payload, "identity_hash": hash_json(payload)}


_TERMINAL_SIDECAR_COLUMNS = (
    "vendor_universe_member",
    "universe_member",
    "research_eligible",
    "reason",
    "stock_basic_out_date",
    "last_raw_bar_date",
    "membership_snapshot_date",
    "security_status",
    "price_fill_policy",
    "terminal_return_policy",
)


_SOURCE_LINEAGE_NORMALIZED_KEYS = {
    "membership": (
        "trade_calendar",
        "membership_snapshots",
        "membership_daily",
        "security_union",
        "membership_request_index",
    ),
    "security": (
        "security_master",
        "security_master_request_index",
    ),
    "industry": (
        "industry_snapshots",
        "industry_request_index",
    ),
}


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _canonical_source_lineage_section(raw: object) -> dict[str, object]:
    """Validate and detach the exact unified source-lineage attestation."""

    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "complete",
        "domains",
        "root_sha256",
    }:
        raise ValueError("frozen manifest source-lineage section is invalid")
    if raw.get("schema_version") != "1.0" or raw.get("complete") is not True:
        raise ValueError("frozen manifest source-lineage section is incomplete")
    domains = raw.get("domains")
    if not isinstance(domains, Mapping) or set(domains) != set(
        _SOURCE_LINEAGE_NORMALIZED_KEYS
    ):
        raise ValueError("frozen manifest source-lineage domains are invalid")
    canonical_domains: dict[str, object] = {}
    for name, normalized_names in _SOURCE_LINEAGE_NORMALIZED_KEYS.items():
        domain = domains.get(name)
        if not isinstance(domain, Mapping) or set(domain) != {
            "policy",
            "cache_objects_sha256",
            "normalized_input_sha256",
            "identity_sha256",
        }:
            raise ValueError(f"frozen {name} source-lineage domain is invalid")
        policy = domain.get("policy")
        normalized = domain.get("normalized_input_sha256")
        if not isinstance(policy, Mapping) or not isinstance(normalized, Mapping):
            raise ValueError(f"frozen {name} source-lineage payload is invalid")
        if set(normalized) != set(normalized_names) or not all(
            _is_sha256(value) for value in normalized.values()
        ):
            raise ValueError(
                f"frozen {name} source-lineage normalized identity is invalid"
            )
        identity = {
            "policy": dict(policy),
            "cache_objects_sha256": str(domain.get("cache_objects_sha256", "")),
            "normalized_input_sha256": {
                key: str(normalized[key]) for key in normalized_names
            },
        }
        if (
            not _is_sha256(identity["cache_objects_sha256"])
            or domain.get("identity_sha256") != hash_json(identity)
        ):
            raise ValueError(f"frozen {name} source-lineage hash is invalid")
        canonical_domains[name] = {
            **identity,
            "identity_sha256": str(domain["identity_sha256"]),
        }
    root_sha256 = str(raw.get("root_sha256", ""))
    if not _is_sha256(root_sha256) or root_sha256 != hash_json(canonical_domains):
        raise ValueError("frozen source-lineage root hash is invalid")
    return {
        "schema_version": "1.0",
        "complete": True,
        "domains": canonical_domains,
        "root_sha256": root_sha256,
    }


def _declared_source_lineage_identity(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Recompute every lineage domain from the surrounding source manifest."""

    section = _canonical_source_lineage_section(manifest.get("source_lineage"))
    normalized_inputs = manifest.get("normalized_input_sha256")
    tape = manifest.get("feature_tape")
    source_contract = tape.get("source_contract") if isinstance(tape, Mapping) else None
    if not isinstance(normalized_inputs, Mapping) or not isinstance(
        source_contract, Mapping
    ):
        raise ValueError("frozen source-lineage surrounding contract is invalid")
    domains = section["domains"]
    assert isinstance(domains, Mapping)
    for name, normalized_names in _SOURCE_LINEAGE_NORMALIZED_KEYS.items():
        domain = domains[name]
        assert isinstance(domain, Mapping)
        cache_records = manifest.get(f"{name}_cache_objects")
        policy = manifest.get(f"{name}_lineage")
        if not isinstance(cache_records, list) or not isinstance(policy, Mapping):
            raise ValueError(f"frozen {name} source-lineage inputs are missing")
        expected_normalized = {
            key: str(normalized_inputs.get(key, "")) for key in normalized_names
        }
        if (
            domain.get("policy") != dict(policy)
            or domain.get("cache_objects_sha256") != hash_json(cache_records)
            or domain.get("normalized_input_sha256") != expected_normalized
        ):
            raise ValueError(
                f"frozen {name} source-lineage domain differs from source artifacts"
            )
    if source_contract.get("source_lineage_root_sha256") != section["root_sha256"]:
        raise ValueError("feature-tape source-lineage root is inconsistent")
    return {**section, "identity_hash": hash_json(section)}


def _verify_terminal_sidecar_artifact(
    root: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Verify the non-price terminal metadata artifact before any research work."""

    identity = _declared_terminal_sidecar_identity(manifest)
    raw = manifest["terminal_stale_membership_exceptions"]
    assert isinstance(raw, Mapping)
    file_record = raw["file"]
    assert isinstance(file_record, Mapping)
    path = _safe_partition_path(root, file_record.get("path"))
    expected_path = (root / str(identity["file_relative_path"])).resolve()
    if path != expected_path:
        raise ValueError("terminal-stale sidecar fixed snapshot path mismatch")
    if not path.is_file():
        raise FileNotFoundError("terminal-stale membership sidecar is missing")
    if path.stat().st_size != int(identity["file_bytes"]):
        raise ValueError("terminal-stale membership sidecar byte count mismatch")
    if _sha256_file(path) != str(identity["file_sha256"]):
        raise ValueError("terminal-stale membership sidecar SHA-256 mismatch")
    sidecar = pd.read_parquet(path)
    if (
        not isinstance(sidecar.index, pd.MultiIndex)
        or tuple(sidecar.index.names) != ("date", "symbol")
        or sidecar.index.has_duplicates
        or not sidecar.index.is_monotonic_increasing
        or tuple(sidecar.columns) != _TERMINAL_SIDECAR_COLUMNS
    ):
        raise ValueError("terminal-stale membership sidecar schema is invalid")
    if len(sidecar) != int(identity["file_rows"]):
        raise ValueError("terminal-stale membership sidecar row count mismatch")
    if (
        sidecar.index.get_level_values("symbol").nunique()
        != int(identity["symbols"])
        or sidecar.index.get_level_values("date").nunique()
        != int(identity["trading_days"])
    ):
        raise ValueError("terminal-stale membership sidecar dimension count mismatch")
    semantic = hashlib.sha256(
        pd.util.hash_pandas_object(
            sidecar, index=True, categorize=True
        ).to_numpy(dtype="uint64").tobytes()
    ).hexdigest()
    if semantic != str(identity["snapshot_hash"]):
        raise ValueError("terminal-stale membership sidecar semantic hash mismatch")
    return identity


def _require_stable_development_source_manifest(
    planned_manifest: Mapping[str, object],
    planned_manifest_sha256: str,
    loaded_manifest: Mapping[str, object],
    loaded_manifest_sha256: str,
) -> None:
    """Reject a source-manifest race between split planning and bounded load."""

    if (
        loaded_manifest_sha256 != planned_manifest_sha256
        or dict(loaded_manifest) != dict(planned_manifest)
    ):
        raise ValueError(
            "frozen panel manifest changed between split planning and development load"
        )


def _trading_calendar_identity(
    trading_dates: Sequence[pd.Timestamp],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    dates = pd.DatetimeIndex(trading_dates)
    expected_source = str(
        dict(manifest.get("normalized_input_sha256", {})).get(
            "trade_calendar", ""
        )
    )
    records = [str(pd.Timestamp(value).date()) for value in dates]
    payload = {
        "schema_version": 1,
        "source": "frozen_normalized_trade_calendar",
        "normalized_trade_calendar_sha256": expected_source,
        "trading_dates_hash": hash_json(records),
        "trading_days": len(records),
        "first_date": records[0] if records else "",
        "last_date": records[-1] if records else "",
        "endpoint_rule_hash": research_rule_identity_v1()["rule_hash"],
    }
    if len(expected_source) != 64 or not records:
        raise ValueError("frozen trading-calendar identity is incomplete")
    return {**payload, "identity_hash": hash_json(payload)}


def load_frozen_hs300_panel(
    panel_dir: Path | str,
    *,
    end_date: str | pd.Timestamp | None = None,
) -> FrozenPanel:
    """Load and verify a frozen panel, optionally as a bounded development view.

    ``end_date`` is a data-capability boundary, not a post-load slice.  Future
    year partitions are never opened and the boundary-year Parquet read uses a
    predicate, which prevents replay rows from entering the development
    process.  A full load (the default) retains the original exact semantic
    snapshot verification used by the reveal process.
    """

    root, payload, actual_manifest_hash = _load_frozen_panel_manifest(panel_dir)
    source_lineage_identity = _declared_source_lineage_identity(payload)
    terminal_identity = _verify_terminal_sidecar_artifact(root, payload)
    cutoff = pd.Timestamp(end_date).normalize() if end_date is not None else None
    records = payload.get("partitions")
    panel = _read_frozen_partitions(
        root,
        records,
        end_date=cutoff,
        label="frozen panel",
    )
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["date", "symbol"]:
        raise ValueError("frozen panel index must be MultiIndex(date, symbol)")
    if not panel.index.is_unique:
        raise ValueError("frozen panel contains duplicate date-symbol rows")
    if cutoff is None:
        if len(panel) != int(payload.get("rows", -1)):
            raise ValueError("frozen panel total row count disagrees with manifest")
        actual_snapshot = panel_snapshot_hash(panel)
        if actual_snapshot != str(payload.get("snapshot_hash", "")):
            raise ValueError("frozen panel semantic snapshot hash mismatch")

    tape_manifest = payload.get("feature_tape")
    if not isinstance(tape_manifest, Mapping):
        raise ValueError("frozen manifest omits the continuous union feature tape")
    tape_records = tape_manifest.get("partitions")
    feature_tape = _read_frozen_partitions(
        root,
        tape_records,
        end_date=cutoff,
        label="feature-tape",
    )
    if (
        not isinstance(feature_tape.index, pd.MultiIndex)
        or feature_tape.index.names != ["date", "symbol"]
        or not feature_tape.index.is_unique
    ):
        raise ValueError("continuous feature tape must have a unique date-symbol index")
    if cutoff is None:
        if len(feature_tape) != int(tape_manifest.get("rows", -1)):
            raise ValueError("feature-tape total rows disagree with manifest")
        tape_hash = panel_snapshot_hash(feature_tape)
        if tape_hash != str(tape_manifest.get("snapshot_hash", "")):
            raise ValueError("feature-tape semantic snapshot hash mismatch")
    if not panel.index.isin(feature_tape.index).all():
        raise ValueError("continuous feature tape does not cover every member row")
    required_masks = {
        "vendor_universe_member",
        "universe_member",
        "research_eligible",
    }
    if not required_masks.issubset(feature_tape.columns) or not required_masks.issubset(
        panel.columns
    ):
        raise ValueError(
            "frozen panel/tape lacks vendor membership or research eligibility"
        )
    panel_start = panel.index.get_level_values("date").min()
    panel_end = panel.index.get_level_values("date").max()
    tape_dates = feature_tape.index.get_level_values("date")
    in_panel_span = (tape_dates >= panel_start) & (tape_dates <= panel_end)
    member_mask = strict_boolean_mask(
        feature_tape["vendor_universe_member"],
        label="frozen feature-tape vendor_universe_member",
    )
    tape_members = feature_tape.loc[
        in_panel_span & member_mask.to_numpy(dtype=bool)
    ].index
    if not tape_members.equals(panel.index):
        raise ValueError("feature-tape vendor-member mask disagrees with member panel")
    panel_vendor = strict_boolean_mask(
        panel["vendor_universe_member"],
        label="frozen member-panel vendor_universe_member",
    )
    if not panel_vendor.all():
        raise ValueError("complete member panel contains non-vendor-member rows")
    for frame, label in (
        (feature_tape, "feature tape"),
        (panel, "member panel"),
    ):
        legacy = strict_boolean_mask(
            frame["universe_member"], label=f"{label} universe_member"
        )
        vendor = strict_boolean_mask(
            frame["vendor_universe_member"],
            label=f"{label} vendor_universe_member",
        )
        if not legacy.equals(vendor):
            raise ValueError("legacy universe_member alias differs from vendor membership")
    panel_eligible = strict_boolean_mask(
        panel["research_eligible"],
        label="frozen member-panel research_eligible",
    )
    tape_eligible = strict_boolean_mask(
        feature_tape["research_eligible"],
        label="frozen feature-tape research_eligible",
    )
    if not tape_eligible.reindex(panel.index).equals(panel_eligible):
        raise ValueError("member-panel and feature-tape research eligibility differ")
    if (tape_eligible & ~member_mask).any():
        raise ValueError("research eligibility contains a non-vendor-member row")
    eligible_panel = panel.loc[panel_eligible].copy()
    if eligible_panel.empty:
        raise ValueError("frozen research_eligible subset is empty")
    universe_identity = build_research_universe_identity(
        panel,
        feature_tape,
        eligible_panel,
        terminal_sidecar_identity=terminal_identity,
    )
    declared_eligibility = payload.get("research_eligibility")
    if not isinstance(declared_eligibility, Mapping):
        raise ValueError("frozen manifest omits research-eligibility identity")
    if (
        declared_eligibility.get("selection_column") != "research_eligible"
        or declared_eligibility.get("current_time_only") is not True
        or declared_eligibility.get("future_endpoint_availability_filter") is not False
    ):
        raise ValueError("frozen research-eligibility rule is not point-in-time")
    if cutoff is None:
        eligibility_columns = [
            "vendor_universe_member",
            "universe_member",
            "research_eligible",
        ]
        eligibility_hashed = pd.util.hash_pandas_object(
            feature_tape.loc[:, eligibility_columns], index=True, categorize=True
        ).to_numpy(dtype="uint64")
        actual_eligibility_hash = hashlib.sha256(
            eligibility_hashed.tobytes()
        ).hexdigest()
        if actual_eligibility_hash != str(
            declared_eligibility.get("snapshot_hash", "")
        ):
            raise ValueError("frozen research-eligibility snapshot hash mismatch")
    trading_calendar = _load_bound_trading_dates(root, payload)
    calendar_identity = _trading_calendar_identity(trading_calendar, payload)
    source_snapshot_identity = _source_snapshot_identity(
        payload, actual_manifest_hash
    )
    return FrozenPanel(
        payload,
        panel,
        eligible_panel,
        feature_tape,
        universe_identity,
        source_lineage_identity,
        source_snapshot_identity,
        trading_calendar,
        calendar_identity,
        actual_manifest_hash,
    )


def make_historical_replay_split(
    panel: pd.DataFrame,
    *,
    train_end: str,
    validation_end: str,
    replay_end: str,
    label_horizon: int = 10,
    entry_delay: int = 1,
) -> SplitPlan:
    """Create calendar-explicit train/validation/replay spans with hard purges."""

    if not isinstance(panel.index, pd.MultiIndex) or "date" not in panel.index.names:
        raise ValueError("panel must use a date-symbol MultiIndex")
    dates = pd.DatetimeIndex(
        panel.index.get_level_values("date")
    ).unique()
    return _make_historical_replay_split_from_dates(
        dates,
        train_end=train_end,
        validation_end=validation_end,
        replay_end=replay_end,
        label_horizon=label_horizon,
        entry_delay=entry_delay,
    )


def _make_historical_replay_split_from_dates(
    dates: Sequence[pd.Timestamp],
    *,
    train_end: str,
    validation_end: str,
    replay_end: str,
    label_horizon: int = 10,
    entry_delay: int = 1,
) -> SplitPlan:
    if label_horizon < 1 or entry_delay < 0:
        raise ValueError("invalid label timing")
    dates = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce")).normalize()
    if (
        dates.empty
        or dates.hasnans
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
    ):
        raise ValueError(
            "split trading dates must be valid, unique, and strictly increasing"
        )
    train_cut = pd.Timestamp(train_end).normalize()
    valid_cut = pd.Timestamp(validation_end).normalize()
    replay_cut = pd.Timestamp(replay_end).normalize()
    if not train_cut < valid_cut < replay_cut:
        raise ValueError("split cut dates must be strictly ordered")
    dates = dates[dates <= replay_cut]
    train = dates[dates <= train_cut]
    remaining = dates[dates > train_cut]
    purge_n = label_horizon + entry_delay
    if len(remaining) <= purge_n:
        raise ValueError("not enough dates after training for purge and validation")
    purge1 = remaining[:purge_n]
    validation_pool = remaining[purge_n:]
    validation = validation_pool[validation_pool <= valid_cut]
    after_validation = dates[dates > valid_cut]
    if len(after_validation) <= purge_n:
        raise ValueError("not enough dates after validation for purge and replay")
    purge2 = after_validation[:purge_n]
    replay_with_outcomes = after_validation[purge_n:]
    if len(replay_with_outcomes) <= purge_n:
        raise ValueError("not enough replay dates to reserve complete label outcomes")
    replay = replay_with_outcomes[:-purge_n]
    outcome_tail = replay_with_outcomes[-purge_n:]
    if min(len(train), len(validation), len(replay)) < 40:
        raise ValueError("each research span must contain at least 40 trading sessions")
    if len(purge1) != purge_n or len(purge2) != purge_n:
        raise AssertionError("purge construction failed")
    collections: Sequence[pd.DatetimeIndex] = (
        train,
        purge1,
        validation,
        purge2,
        replay,
        outcome_tail,
    )
    flattened = [pd.Timestamp(value) for group in collections for value in group]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("split dates overlap")
    return SplitPlan(
        tuple(map(pd.Timestamp, train)),
        tuple(map(pd.Timestamp, validation)),
        tuple(map(pd.Timestamp, replay)),
        tuple(map(pd.Timestamp, purge1)),
        tuple(map(pd.Timestamp, purge2)),
        label_horizon,
        entry_delay,
        tuple(map(pd.Timestamp, outcome_tail)),
    )


def _load_bound_trading_dates(
    panel_dir: Path,
    manifest: Mapping[str, object],
) -> pd.DatetimeIndex:
    """Load the committed calendar, which carries no price/return outcomes."""

    expected = str(
        dict(manifest.get("normalized_input_sha256", {})).get(
            "trade_calendar", ""
        )
    )
    config = manifest.get("config")
    candidates: list[Path] = [panel_dir.parent / "normalized" / "trade_calendar.parquet"]
    if isinstance(config, Mapping) and config.get("normalized_dir"):
        candidates.insert(
            0,
            Path(str(config["normalized_dir"])).resolve() / "trade_calendar.parquet",
        )
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique_candidates:
            unique_candidates.append(resolved)
    calendar_path = next((path for path in unique_candidates if path.is_file()), None)
    if calendar_path is None:
        raise FileNotFoundError(
            "frozen trade calendar is missing; development split cannot inspect "
            "future price partitions to infer dates"
        )
    if not expected or _sha256_file(calendar_path) != expected:
        raise ValueError("trade calendar does not match the frozen panel manifest")
    calendar = pd.read_parquet(calendar_path)
    required = {"calendar_date", "is_trading_day"}
    if not required.issubset(calendar.columns):
        raise ValueError("frozen trade calendar schema is incomplete")
    domain = calendar["is_trading_day"].astype("string").str.strip()
    if not domain.isin(["0", "1"]).all():
        raise ValueError("frozen trade calendar has a non-boolean trading-day domain")
    trading = pd.DatetimeIndex(
        pd.to_datetime(
            calendar.loc[domain == "1", "calendar_date"],
            errors="coerce",
        )
    ).normalize()
    if (
        trading.hasnans
        or trading.has_duplicates
        or not trading.is_monotonic_increasing
    ):
        raise ValueError(
            "frozen trade calendar must be valid, unique, and strictly increasing"
        )
    tape_manifest = manifest.get("feature_tape")
    tape_first = (
        tape_manifest.get("first_date")
        if isinstance(tape_manifest, Mapping)
        else None
    )
    first = pd.Timestamp(
        str(tape_first if tape_first is not None else manifest.get("first_date"))
    ).normalize()
    last = pd.Timestamp(str(manifest.get("last_date"))).normalize()
    bounded = trading[(trading >= first) & (trading <= last)]
    if bounded.empty:
        raise ValueError("frozen trade calendar has no sessions in panel span")
    return bounded


def load_codex_seed_candidates(
    project_dir: Path | str,
    *,
    horizon: int = 10,
) -> list[Candidate]:
    """Load exactly 80 frozen Factor Cards spanning all 21 original roles."""

    project = Path(project_dir).resolve()
    with (project / "configs" / "agents.yaml").open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    allowed = {str(item["id"]) for item in catalog["agents"]}
    agent_input_contract = build_agent_input_contract(catalog)
    paths = [
        project / "proposals" / "codex_seed" / name
        for name in ("l1_l2.jsonl", "l3_l4.jsonl", "l5_l7.jsonl")
    ]
    cards = load_factor_cards(
        paths,
        allowed_agent_ids=allowed,
        agent_allowed_inputs=agent_input_contract,
        expected_horizon=horizon,
        expected_count=80,
    )
    covered = {card.agent_id for card in cards}
    if covered != allowed:
        missing = sorted(allowed.difference(covered))
        extra = sorted(covered.difference(allowed))
        raise ValueError(f"LLM seed batch does not cover all 21 roles: missing={missing}, extra={extra}")
    return cards_to_candidates(cards)


def load_codex_injection_candidates(
    project_dir: Path | str,
    *,
    horizon: int = 10,
) -> list[Candidate]:
    """Load 96 LLM-authored repair cards reserved for live injections.

    These cards are not part of generation zero.  Every two generations the
    adaptive controller selects eight still-unused cards according to the
    observed development-only failure class and original seven-level role.
    """

    project = Path(project_dir).resolve()
    with (project / "configs" / "agents.yaml").open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    allowed = {str(item["id"]) for item in catalog["agents"]}
    agent_input_contract = build_agent_input_contract(catalog)
    paths = [
        project / "proposals" / "codex_injection" / name
        for name in ("l1_l4.jsonl", "l5_l7.jsonl")
    ]
    cards = load_factor_cards(
        paths,
        allowed_agent_ids=allowed,
        agent_allowed_inputs=agent_input_contract,
        expected_horizon=horizon,
        expected_count=96,
    )
    covered = {card.agent_id for card in cards}
    if covered != allowed:
        missing = sorted(allowed.difference(covered))
        extra = sorted(covered.difference(allowed))
        raise ValueError(
            "LLM injection library does not cover all 21 roles: "
            f"missing={missing}, extra={extra}"
        )
    repairs_by_role = {role: set() for role in allowed}
    cards_by_role = {role: 0 for role in allowed}
    for card in cards:
        cards_by_role[card.agent_id] += 1
        declared_repairs = {
            tag for tag in card.mechanism_tags if tag.startswith("repair_")
        }
        unknown_repairs = declared_repairs.difference(ADAPTIVE_REPAIR_TAGS)
        if unknown_repairs:
            raise ValueError(
                f"LLM injection card {card.candidate_key!r} declares unknown "
                f"repair tags: {sorted(unknown_repairs)}"
            )
        if not declared_repairs:
            raise ValueError(
                f"LLM injection card {card.candidate_key!r} has no adaptive repair tag"
            )
        repairs_by_role[card.agent_id].update(declared_repairs)
    incomplete_repairs = {
        role: sorted(ADAPTIVE_REPAIR_TAGS.difference(tags))
        for role, tags in repairs_by_role.items()
        if not ADAPTIVE_REPAIR_TAGS.issubset(tags)
    }
    if incomplete_repairs:
        raise ValueError(
            "each LLM injection role must cover every adaptive repair class: "
            f"{incomplete_repairs}"
        )
    underfilled_roles = {
        role: count for role, count in cards_by_role.items() if count < 4
    }
    if underfilled_roles:
        raise ValueError(
            "each LLM injection role requires at least four distinct "
            f"mechanism cards: {underfilled_roles}"
        )
    candidates = cards_to_candidates(cards)
    seed_hashes = {
        candidate.expression_hash
        for candidate in load_codex_seed_candidates(project, horizon=horizon)
    }
    overlaps = sorted(
        candidate.expression_hash
        for candidate in candidates
        if candidate.expression_hash in seed_hashes
    )
    if overlaps:
        raise ValueError(
            "LLM injection library structurally duplicates generation-zero "
            f"Factor Cards: {overlaps[:5]}"
        )
    for candidate in candidates:
        candidate.metadata = {
            **candidate.metadata,
            "source": "codex_precommitted_adaptive_injection_card",
            "role_lineage": list(candidate_roles(candidate)),
        }
        candidate.candidate_id = deterministic_candidate_id(candidate)
    return candidates


def _atomic_json(payload: Mapping[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exclusive_json(payload: Mapping[str, object], target: Path) -> Path:
    """Create an immutable JSON file; replacement is never permitted."""

    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(payload) + "\n"
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to replace frozen file: {target}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return target


_SPLIT_SEQUENCE_FIELDS = (
    "train_dates",
    "validation_dates",
    "sealed_dates",
    "purge_after_train",
    "purge_after_validation",
    "outcome_tail",
)


def _split_payload(split: SplitPlan) -> dict[str, object]:
    return {
        **{
            name: [str(pd.Timestamp(value).date()) for value in getattr(split, name)]
            for name in _SPLIT_SEQUENCE_FIELDS
        },
        "label_horizon": split.label_horizon,
        "entry_delay": split.entry_delay,
    }


def _split_from_payload(payload: object) -> SplitPlan:
    if not isinstance(payload, Mapping):
        raise ValueError("sealed manifest split payload is invalid")
    converted: dict[str, tuple[pd.Timestamp, ...]] = {}
    for name in _SPLIT_SEQUENCE_FIELDS:
        raw = payload.get(name)
        if not isinstance(raw, list) or (name != "outcome_tail" and not raw):
            raise ValueError(f"sealed manifest split field {name} is invalid")
        dates = tuple(pd.Timestamp(str(item)).normalize() for item in raw)
        if len(dates) != len(set(dates)) or tuple(sorted(dates)) != dates:
            raise ValueError(f"sealed manifest split field {name} is not ordered and unique")
        converted[name] = dates
    split = SplitPlan(
        train_dates=converted["train_dates"],
        validation_dates=converted["validation_dates"],
        sealed_dates=converted["sealed_dates"],
        purge_after_train=converted["purge_after_train"],
        purge_after_validation=converted["purge_after_validation"],
        label_horizon=int(payload.get("label_horizon", 0)),
        entry_delay=int(payload.get("entry_delay", -1)),
        outcome_tail=converted["outcome_tail"],
    )
    if split.label_horizon < 1 or split.entry_delay < 0:
        raise ValueError("sealed manifest split timing is invalid")
    flattened = [value for name in _SPLIT_SEQUENCE_FIELDS for value in converted[name]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("sealed manifest split spans overlap")
    return split


def _source_snapshot_identity(
    manifest: Mapping[str, object],
    manifest_sha256: str,
) -> dict[str, object]:
    tape = manifest.get("feature_tape")
    if not isinstance(tape, Mapping):
        raise ValueError("frozen panel manifest omits feature-tape identity")
    member_hash = str(manifest.get("snapshot_hash", ""))
    tape_hash = str(tape.get("snapshot_hash", ""))
    member_partitions = manifest.get("partitions")
    tape_partitions = tape.get("partitions")
    source_contract = tape.get("source_contract")
    if (
        not isinstance(member_partitions, list)
        or not isinstance(tape_partitions, list)
        or not isinstance(source_contract, Mapping)
    ):
        raise ValueError("frozen panel manifest omits full partition/source contracts")
    eligibility = manifest.get("research_eligibility")
    if not isinstance(eligibility, Mapping):
        raise ValueError("frozen panel manifest omits research eligibility identity")
    eligibility_hash = str(eligibility.get("snapshot_hash", ""))
    eligibility_payload = dict(eligibility)
    eligibility_identity = {
        **eligibility_payload,
        "identity_hash": hash_json(eligibility_payload),
    }
    source_lineage_identity = _declared_source_lineage_identity(manifest)
    terminal_identity = _declared_terminal_sidecar_identity(manifest)
    calendar_hash = str(
        dict(manifest.get("normalized_input_sha256", {})).get(
            "trade_calendar", ""
        )
    )
    if any(
        len(value) != 64
        for value in (
            manifest_sha256,
            member_hash,
            tape_hash,
            eligibility_hash,
            calendar_hash,
            str(source_lineage_identity["root_sha256"]),
            str(source_lineage_identity["identity_hash"]),
            str(terminal_identity["identity_hash"]),
        )
    ):
        raise ValueError("frozen panel snapshot identity is incomplete")
    combined = hashlib.sha256(
        (
            f"{member_hash}|{tape_hash}|{eligibility_hash}|"
            f"{terminal_identity['identity_hash']}|{calendar_hash}|"
            f"{source_lineage_identity['identity_hash']}"
        ).encode("utf-8")
    ).hexdigest()
    identity = {
        "panel_manifest_sha256": manifest_sha256,
        "storage_snapshot_id": str(manifest.get("storage_snapshot_id", "")),
        "member_panel_snapshot_hash": member_hash,
        "member_partition_records_hash": hash_json(member_partitions),
        "feature_tape_snapshot_hash": tape_hash,
        "feature_tape_partition_records_hash": hash_json(tape_partitions),
        "full_source_contract": dict(source_contract),
        "full_source_contract_hash": hash_json(dict(source_contract)),
        "research_eligibility_snapshot_hash": eligibility_hash,
        "research_eligibility_identity": eligibility_identity,
        "source_lineage_root_sha256": source_lineage_identity["root_sha256"],
        "source_lineage_identity": source_lineage_identity,
        "terminal_sidecar_identity": terminal_identity,
        "frozen_trade_calendar_sha256": calendar_hash,
        "research_rule_identity": research_rule_identity_v1(),
        "combined_data_snapshot_hash": combined,
        "first_date": str(manifest.get("first_date", "")),
        "last_date": str(manifest.get("last_date", "")),
    }
    return {**identity, "identity_hash": hash_json(identity)}


def _champions_payload(
    champions: Sequence[EvaluatedCandidate],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for evaluated in champions:
        candidate = evaluated.candidate
        score = evaluated.score
        expression_json = canonical_expression_json(
            candidate.expression,
            PRODUCTION_DSL_SPEC,
        )
        expression_ast = json.loads(expression_json)
        expression_digest = canonical_expression_hash(
            expression_ast,
            PRODUCTION_DSL_SPEC,
        )
        control_ast: dict[str, object] | None = None
        control_json: str | None = None
        control_digest: str | None = None
        if candidate.control_expression is not None:
            control_json = canonical_expression_json(
                candidate.control_expression,
                PRODUCTION_DSL_SPEC,
            )
            control_ast = json.loads(control_json)
            control_digest = canonical_expression_hash(
                control_ast,
                PRODUCTION_DSL_SPEC,
            )
        direction = score.diagnostics.get("direction")
        if direction not in {-1, 1}:
            raise ValueError(
                f"champion {candidate.candidate_id!r} lacks a frozen train-only direction"
            )
        metadata = json.loads(canonical_json(candidate.metadata))
        if metadata.get("expected_direction") != direction:
            raise ValueError(
                f"champion {candidate.candidate_id!r} direction differs from its "
                "predeclared candidate metadata"
            )
        if deterministic_candidate_id(candidate) != candidate.candidate_id:
            raise ValueError(
                f"champion {candidate.candidate_id!r} is not its deterministic "
                "behavior/provenance identity"
            )
        train_raw = float(score.diagnostics.get("train_raw_rank_ic", 0.0))
        output.append(
            {
                "candidate_id": candidate.candidate_id,
                "canonical_expression_ast": expression_ast,
                "canonical_expression_json": expression_json,
                "expression_hash": expression_digest,
                "control_expression_ast": control_ast,
                "control_expression_json": control_json,
                "control_expression_hash": control_digest,
                "direction": int(direction),
                "agent_id": candidate.agent_id,
                "operator": candidate.operator,
                "hypothesis": candidate.hypothesis,
                "mechanism_tags": list(candidate.mechanism_tags),
                "generation": candidate.generation,
                "parents": list(candidate.parents),
                "metadata": metadata,
                "train_rank_ic": train_raw * int(direction),
                "validation_rank_ic": score.signed_rank_ic,
                "validation_neutral_rank_ic": score.neutral_rank_ic,
                "score": {
                    "fitness": score.fitness,
                    "signed_rank_ic": score.signed_rank_ic,
                    "neutral_rank_ic": score.neutral_rank_ic,
                    "turnover": score.turnover,
                    "worst_window_ic": score.worst_window_ic,
                    "style_retention": score.style_retention,
                    "novelty": score.novelty,
                    "control_delta": score.control_delta,
                    "passed": score.passed,
                },
            }
        )
    if len({str(item["candidate_id"]) for item in output}) != len(output):
        raise ValueError("frozen champion family contains duplicate candidate ids")
    return output


def _candidate_from_frozen_champion(item: Mapping[str, object]) -> Candidate:
    """Rebuild and authenticate the exact development Candidate identity."""

    metadata = item.get("metadata")
    mechanism_tags = item.get("mechanism_tags")
    parents = item.get("parents")
    if not isinstance(metadata, Mapping):
        raise ValueError("frozen champion metadata is invalid")
    if not isinstance(mechanism_tags, list) or not all(
        isinstance(tag, str) for tag in mechanism_tags
    ):
        raise ValueError("frozen champion mechanism tags are invalid")
    if not isinstance(parents, list) or not all(
        isinstance(parent, str) and parent for parent in parents
    ):
        raise ValueError("frozen champion parents are invalid")
    generation = item.get("generation")
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
    ):
        raise ValueError("frozen champion generation is invalid")

    ast = item.get("canonical_expression_ast")
    expression_json = canonical_expression_json(ast, PRODUCTION_DSL_SPEC)
    expression_digest = canonical_expression_hash(ast, PRODUCTION_DSL_SPEC)
    if (
        expression_json != str(item.get("canonical_expression_json", ""))
        or expression_digest != str(item.get("expression_hash", ""))
    ):
        raise ValueError("frozen champion canonical AST/hash integrity mismatch")

    control_ast = item.get("control_expression_ast")
    control_json = item.get("control_expression_json")
    control_digest = item.get("control_expression_hash")
    if control_ast is None:
        if control_json is not None or control_digest is not None:
            raise ValueError("frozen champion null control commitment is inconsistent")
        canonical_control = None
    else:
        canonical_control_json = canonical_expression_json(
            control_ast,
            PRODUCTION_DSL_SPEC,
        )
        canonical_control_digest = canonical_expression_hash(
            control_ast,
            PRODUCTION_DSL_SPEC,
        )
        if (
            canonical_control_json != control_json
            or canonical_control_digest != control_digest
        ):
            raise ValueError("frozen champion control AST/hash integrity mismatch")
        canonical_control = json.loads(canonical_control_json)

    direction = item.get("direction")
    if direction not in {-1, 1}:
        raise ValueError("frozen champion direction is invalid")
    canonical_metadata = json.loads(canonical_json(dict(metadata)))
    if canonical_metadata.get("expected_direction") != direction:
        raise ValueError(
            "frozen champion direction differs from predeclared candidate metadata"
        )
    candidate = Candidate(
        expression=json.loads(expression_json),
        agent_id=str(item.get("agent_id", "")),
        hypothesis=str(item.get("hypothesis", "")),
        mechanism_tags=tuple(mechanism_tags),
        generation=generation,
        parents=tuple(parents),
        operator=str(item.get("operator", "")),
        candidate_id=str(item.get("candidate_id", "")),
        control_expression=canonical_control,
        metadata=canonical_metadata,
    )
    recomputed_id = deterministic_candidate_id(candidate)
    if recomputed_id != candidate.candidate_id:
        raise ValueError(
            "frozen champion candidate_id differs from deterministic identity"
        )
    return candidate


def _fixed_development_evidence_paths(
    run_dir: Path,
    locators: object,
) -> dict[str, Path]:
    if not isinstance(locators, Mapping) or dict(locators) != (
        DEVELOPMENT_EVIDENCE_LOCATORS
    ):
        raise ValueError(
            "development evidence locators are not the fixed run-relative set"
        )
    root = run_dir.resolve()
    paths = {
        name: (root / relative).resolve()
        for name, relative in DEVELOPMENT_EVIDENCE_LOCATORS.items()
    }
    if any(
        path != root / DEVELOPMENT_EVIDENCE_LOCATORS[name]
        for name, path in paths.items()
    ):
        raise ValueError("development evidence locator escapes its run directory")
    return paths


def _read_json_mapping(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"development {label} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"development {label} is not a JSON object")
    return dict(payload)


def _validate_development_summary(
    summary: Mapping[str, object],
    *,
    run_manifest: RunManifest,
    champions: Sequence[Mapping[str, object]],
    pre_completion_entries: int,
    pre_completion_head: str,
) -> None:
    if (
        summary.get("run_id") != run_manifest.run_id
        or summary.get("manifest_hash") != run_manifest.manifest_hash
        or summary.get("sealed_revealed") is not False
    ):
        raise ValueError(
            "development summary disagrees with the unrevealed RunManifest"
        )
    summary_champions = summary.get("champions")
    if not isinstance(summary_champions, list) or len(summary_champions) != len(
        champions
    ):
        raise ValueError("development summary champion family is incomplete")
    score_fields = (
        "fitness",
        "validation_rank_ic",
        "validation_neutral_rank_ic",
        "control_delta",
    )
    for frozen, recorded in zip(champions, summary_champions, strict=True):
        if not isinstance(recorded, Mapping):
            raise ValueError("development summary champion record is invalid")
        score = frozen.get("score")
        if not isinstance(score, Mapping):
            raise ValueError("frozen champion score is invalid")
        expected = {
            "candidate_id": frozen.get("candidate_id"),
            "agent_id": frozen.get("agent_id"),
            "operator": frozen.get("operator"),
            "expression_hash": frozen.get("expression_hash"),
            "fitness": score.get("fitness"),
            "validation_rank_ic": score.get("signed_rank_ic"),
            "validation_neutral_rank_ic": score.get("neutral_rank_ic"),
            "control_delta": score.get("control_delta"),
        }
        actual = {
            name: recorded.get(name)
            for name in (
                "candidate_id",
                "agent_id",
                "operator",
                "expression_hash",
                *score_fields,
            )
        }
        if actual != expected:
            raise ValueError("development summary champion record mismatch")
    ledger_summary = summary.get("ledger")
    if not isinstance(ledger_summary, Mapping) or (
        ledger_summary.get("valid") is not True
        or ledger_summary.get("entries") != pre_completion_entries
        or ledger_summary.get("head_hash") != pre_completion_head
    ):
        raise ValueError("development summary does not bind the pre-completion ledger")


def _champion_registry_bindings(
    *,
    champions: Sequence[Mapping[str, object]],
    run_manifest: RunManifest,
    registry: FactorRegistry,
    ledger_entries: Sequence[object],
) -> list[dict[str, object]]:
    bindings: list[dict[str, object]] = []
    for frozen in champions:
        candidate = _candidate_from_frozen_champion(frozen)
        matching_events = [
            entry
            for entry in ledger_entries
            if getattr(entry, "event_type", "") == "candidate_evaluated"
            and isinstance(getattr(entry, "payload", None), Mapping)
            and entry.payload.get("candidate_id") == candidate.candidate_id
        ]
        if len(matching_events) != 1:
            raise ValueError(
                "each frozen champion must have exactly one development ledger evaluation"
            )
        event = matching_events[0]
        factor_id = str(event.payload.get("factor_id", ""))
        record = registry.get(factor_id)
        if record is None or record.status != "success":
            raise ValueError("frozen champion has no successful registry record")
        if record.run_id != run_manifest.run_id:
            raise ValueError("frozen champion registry record belongs to another run")
        record_payload = record.to_dict()
        record_metadata = record_payload["metadata"]
        if not isinstance(record_metadata, Mapping):
            raise ValueError("frozen champion registry metadata is invalid")
        if (
            record_metadata.get("candidate_id") != candidate.candidate_id
            or list(record_metadata.get("mechanism_tags", ()))
            != list(candidate.mechanism_tags)
            or dict(record_metadata.get("candidate_metadata", {}))
            != candidate.metadata
        ):
            raise ValueError("frozen champion identity differs from its registry record")
        record_control = record_metadata.get("control_expression")
        if record_control is None:
            if candidate.control_expression is not None:
                raise ValueError(
                    "frozen champion control differs from its registry record"
                )
        elif candidate.control_expression is None or canonical_expression_json(
            record_control,
            PRODUCTION_DSL_SPEC,
        ) != canonical_expression_json(
            candidate.control_expression,
            PRODUCTION_DSL_SPEC,
        ):
            raise ValueError("frozen champion control differs from its registry record")
        if canonical_expression_hash(
            record_payload["formula_ast"],
            PRODUCTION_DSL_SPEC,
        ) != str(frozen.get("expression_hash", "")):
            raise ValueError("frozen champion AST differs from its registry record")
        score = frozen.get("score")
        event_score = event.payload.get("score")
        if not isinstance(score, Mapping) or not isinstance(event_score, Mapping):
            raise ValueError("frozen champion development score evidence is invalid")
        for frozen_name, recorded_name in (
            ("fitness", "fitness"),
            ("signed_rank_ic", "signed_rank_ic"),
            ("neutral_rank_ic", "neutral_rank_ic"),
            ("turnover", "turnover"),
            ("worst_window_ic", "worst_window_ic"),
            ("style_retention", "style_retention"),
            ("novelty", "novelty"),
            ("control_delta", "control_delta"),
            ("passed", "passed"),
        ):
            if score.get(frozen_name) != event_score.get(recorded_name):
                raise ValueError(
                    "frozen champion score differs from its ledger evaluation"
                )
        diagnostics = event_score.get("diagnostics")
        if not isinstance(diagnostics, Mapping) or diagnostics.get(
            "direction"
        ) != frozen.get("direction"):
            raise ValueError("frozen champion direction differs from its ledger evaluation")
        bindings.append(
            {
                "candidate_id": candidate.candidate_id,
                "expression_hash": str(frozen["expression_hash"]),
                "factor_id": factor_id,
                "factor_record_hash": record.record_hash,
                "candidate_evaluation_seq": event.seq,
                "candidate_evaluation_entry_hash": event.entry_hash,
            }
        )
    return bindings


def _completion_event_payload(
    *,
    run_manifest: RunManifest,
    run_manifest_sha256: str,
    summary_sha256: str,
    registry_sha256: str,
    champion_family_hash: str,
    champion_registry_bindings: Sequence[Mapping[str, object]],
    composite_family_hash: str,
    composite_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "completion_schema_version": 1,
        "run_id": run_manifest.run_id,
        "run_manifest_hash": run_manifest.manifest_hash,
        "run_manifest_sha256": run_manifest_sha256,
        "summary_sha256": summary_sha256,
        "registry_sha256": registry_sha256,
        "champion_family_hash": champion_family_hash,
        "champion_registry_bindings": [
            dict(item) for item in champion_registry_bindings
        ],
        "composite_family_hash": composite_family_hash,
        "composite_manifest_sha256": composite_manifest_sha256,
        "sealed_market_rows_loaded": 0,
    }


def _append_development_completion_evidence(
    *,
    run_dir: Path,
    run_manifest: RunManifest,
    champions: Sequence[Mapping[str, object]],
    champion_family_hash: str,
    composite_family_hash: str,
    composite_manifest_sha256: str,
) -> dict[str, object]:
    paths = _fixed_development_evidence_paths(
        run_dir,
        DEVELOPMENT_EVIDENCE_LOCATORS,
    )
    if (
        RunManifest.load(paths["run_manifest"]).manifest_hash
        != run_manifest.manifest_hash
    ):
        raise ValueError("development RunManifest changed before completion freeze")
    summary = _read_json_mapping(paths["summary"], label="summary")
    registry = FactorRegistry(paths["registry"])
    ledger = TrialLedger(paths["ledger"])
    entries = ledger.entries()
    if not entries or entries[-1].event_type != "run_finished":
        raise ValueError("development ledger is not at its completed pre-freeze tail")
    if any(entry.event_type == DEVELOPMENT_COMPLETION_EVENT for entry in entries):
        raise ValueError("development completion evidence was already appended")
    verification = ledger.verify()
    if not verification:
        raise ValueError("development trial ledger is corrupt before completion freeze")
    _validate_development_summary(
        summary,
        run_manifest=run_manifest,
        champions=champions,
        pre_completion_entries=verification.entries,
        pre_completion_head=verification.head_hash,
    )
    bindings = _champion_registry_bindings(
        champions=champions,
        run_manifest=run_manifest,
        registry=registry,
        ledger_entries=entries,
    )
    run_manifest_sha256 = _sha256_file(paths["run_manifest"])
    summary_sha256 = _sha256_file(paths["summary"])
    registry_sha256 = _sha256_file(paths["registry"])
    completion_payload = _completion_event_payload(
        run_manifest=run_manifest,
        run_manifest_sha256=run_manifest_sha256,
        summary_sha256=summary_sha256,
        registry_sha256=registry_sha256,
        champion_family_hash=champion_family_hash,
        champion_registry_bindings=bindings,
        composite_family_hash=composite_family_hash,
        composite_manifest_sha256=composite_manifest_sha256,
    )
    completion = ledger.append(DEVELOPMENT_COMPLETION_EVENT, completion_payload)
    final_verification = ledger.verify(
        expected_head=completion.entry_hash,
        expected_entries=completion.seq + 1,
    )
    if not final_verification:
        raise ValueError("development trial ledger failed its final completion freeze")
    return {
        "schema_version": 1,
        "locators": dict(DEVELOPMENT_EVIDENCE_LOCATORS),
        "run_manifest_sha256": run_manifest_sha256,
        "summary_sha256": summary_sha256,
        "registry_sha256": registry_sha256,
        "champion_registry_bindings": bindings,
        "composite_family_hash": composite_family_hash,
        "composite_manifest_sha256": composite_manifest_sha256,
        "ledger_sha256": _sha256_file(paths["ledger"]),
        "ledger_entries": final_verification.entries,
        "ledger_head_hash": final_verification.head_hash,
        "completion_event_seq": completion.seq,
        "completion_event_hash": completion.entry_hash,
    }


def _validate_development_completion_evidence(
    *,
    run_dir: Path,
    payload: Mapping[str, object],
    champions: Sequence[Mapping[str, object]],
    run_manifest: RunManifest,
) -> None:
    evidence = payload.get("development_completion_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != 1:
        raise ValueError("development completion evidence is missing or invalid")
    paths = _fixed_development_evidence_paths(run_dir, evidence.get("locators"))
    expected_sha = {
        "run_manifest_sha256": _sha256_file(paths["run_manifest"]),
        "summary_sha256": _sha256_file(paths["summary"]),
        "registry_sha256": _sha256_file(paths["registry"]),
        "ledger_sha256": _sha256_file(paths["ledger"]),
    }
    for name, digest in expected_sha.items():
        if evidence.get(name) != digest:
            raise ValueError(f"development completion evidence {name} mismatch")
    if evidence.get("run_manifest_sha256") != payload.get(
        "development_run_manifest_sha256"
    ):
        raise ValueError("development evidence and manifest byte identities disagree")
    if (
        evidence.get("composite_family_hash")
        != payload.get("composite_family_hash")
        or evidence.get("composite_manifest_sha256")
        != payload.get("composite_manifest_sha256")
    ):
        raise ValueError("development evidence and composite identities disagree")

    summary = _read_json_mapping(paths["summary"], label="summary")
    registry = FactorRegistry(paths["registry"])
    ledger = TrialLedger(paths["ledger"])
    expected_entries = evidence.get("ledger_entries")
    if not isinstance(expected_entries, int) or isinstance(expected_entries, bool):
        raise ValueError("development completion ledger entry count is invalid")
    expected_head = str(evidence.get("ledger_head_hash", ""))
    verification = ledger.verify(
        expected_head=expected_head,
        expected_entries=expected_entries,
    )
    if not verification:
        raise ValueError("development completion ledger hash chain mismatch")
    entries = ledger.entries()
    if len(entries) < 2:
        raise ValueError("development completion ledger is incomplete")
    completion = entries[-1]
    if (
        completion.event_type != DEVELOPMENT_COMPLETION_EVENT
        or entries[-2].event_type != "run_finished"
        or completion.seq != evidence.get("completion_event_seq")
        or completion.entry_hash != evidence.get("completion_event_hash")
        or completion.entry_hash != expected_head
    ):
        raise ValueError("development completion event is not the frozen ledger tail")
    _validate_development_summary(
        summary,
        run_manifest=run_manifest,
        champions=champions,
        pre_completion_entries=completion.seq,
        pre_completion_head=completion.prev_hash,
    )
    bindings = _champion_registry_bindings(
        champions=champions,
        run_manifest=run_manifest,
        registry=registry,
        ledger_entries=entries,
    )
    if evidence.get("champion_registry_bindings") != bindings:
        raise ValueError("development champion-to-registry bindings mismatch")
    champion_family_hash = str(payload.get("champion_family_hash", ""))
    expected_completion_payload = _completion_event_payload(
        run_manifest=run_manifest,
        run_manifest_sha256=expected_sha["run_manifest_sha256"],
        summary_sha256=expected_sha["summary_sha256"],
        registry_sha256=expected_sha["registry_sha256"],
        champion_family_hash=champion_family_hash,
        champion_registry_bindings=bindings,
        composite_family_hash=str(payload.get("composite_family_hash", "")),
        composite_manifest_sha256=str(
            payload.get("composite_manifest_sha256", "")
        ),
    )
    recorded_payload = completion.to_dict()["payload"]
    if recorded_payload != expected_completion_payload:
        raise ValueError(
            "development completion event disagrees with champion family evidence"
        )


def _run_manifest_snapshot_identity(
    run_manifest: RunManifest,
    name: str,
) -> object:
    """Return one persisted identity in ordinary JSON list/dict form."""

    return run_manifest.to_dict()["data_snapshot"].get(name)


def _freeze_reveal_manifest(
    *,
    config: HS300ResearchConfig,
    source_manifest: Mapping[str, object],
    source_manifest_sha256: str,
    development_panel: FrozenPanel,
    split: SplitPlan,
    evolution: EvolutionResult,
    implementation_identity: Mapping[str, object],
    adaptive_feedback_policy: AdaptiveFeedbackPolicy | None,
    behavioral_champion_policy: BehavioralChampionPolicy | None,
    development_handoff_policy: DevelopmentHandoffPolicy,
    composite_family_policy: CompositeFamilyPolicy,
    pre_search_development_label_audit: Mapping[str, object],
    terminal_label_bridges: Sequence[TerminalLabelBridge],
    terminal_label_bridge_identity: Mapping[str, object],
) -> tuple[Path, str]:
    run_manifest_path = config.output_dir / "manifest.json"
    run_manifest = RunManifest.load(run_manifest_path)
    # RunManifest deliberately deep-freezes JSON arrays as tuples in memory.
    # Compare identities in their serialized JSON domain so a persisted
    # list-valued lineage record remains equal to the same freshly loaded
    # source identity.  Direct MappingProxy/tuple versus dict/list equality
    # spuriously rejects a valid completed development run.
    champions = _champions_payload(evolution.champions)
    champion_family_hash = hash_json(champions)
    split_exact = _split_payload(split)
    split_identity_hash = hash_json(split_exact)
    source_identity = _source_snapshot_identity(
        source_manifest,
        source_manifest_sha256,
    )
    if development_panel.source_snapshot_identity != source_identity:
        raise ValueError(
            "bounded development loader disagrees with the full source identity"
        )
    implementation_identity = _validate_research_implementation_snapshot(
        implementation_identity,
        project_dir=config.project_dir,
    )
    current_policy = _load_adaptive_feedback_policy(
        config.project_dir,
        config.profile,
    )
    if _adaptive_feedback_policy_payload(current_policy) != (
        _adaptive_feedback_policy_payload(adaptive_feedback_policy)
    ):
        raise ValueError(
            "adaptive feedback YAML changed after the runtime policy was resolved"
        )
    current_behavioral_policy = _load_behavioral_champion_policy(
        config.project_dir,
        config.profile,
    )
    if _behavioral_champion_policy_payload(current_behavioral_policy) != (
        _behavioral_champion_policy_payload(behavioral_champion_policy)
    ):
        raise ValueError(
            "behavioral champion YAML changed after the runtime policy was resolved"
        )
    current_handoff_policy, current_composite_policy = _load_composite_policies(
        config.project_dir,
        config.profile,
    )
    if (
        current_handoff_policy.to_dict() != development_handoff_policy.to_dict()
        or current_composite_policy.to_dict() != composite_family_policy.to_dict()
    ):
        raise ValueError("composite YAML changed after runtime policy resolution")
    current_bridges, current_bridge_identity = load_terminal_label_bridges(
        config.project_dir / "configs" / "hs300_terminal_label_bridges.yaml"
    )
    if (
        current_bridges != tuple(terminal_label_bridges)
        or current_bridge_identity != dict(terminal_label_bridge_identity)
    ):
        raise ValueError("terminal label bridge catalog changed after policy resolution")
    resolved_experiment_config = experiment_config_payload(
        profile=config.profile,
        split=split,
        budget=SearchBudget(),
        bootstrap_samples=config.bootstrap_samples,
        evaluation_workers=config.evaluation_workers,
        adaptive_feedback_policy=adaptive_feedback_policy,
        behavioral_champion_policy=behavioral_champion_policy,
        development_handoff_policy=development_handoff_policy.to_dict(),
        composite_family_policy=composite_family_policy.to_dict(),
        operational_locators=_operational_locators(config),
        implementation_identity=implementation_identity,
        research_universe_identity=development_panel.research_universe_identity,
        trading_calendar_identity=development_panel.trading_calendar_identity,
        source_lineage_identity=development_panel.source_lineage_identity,
        source_snapshot_identity=source_identity,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
    )
    if hash_json(resolved_experiment_config) != run_manifest.config_hash:
        raise ValueError(
            "development RunManifest does not bind the frozen implementation identity"
        )
    if _run_manifest_snapshot_identity(
        run_manifest, "source_lineage_identity"
    ) != dict(
        development_panel.source_lineage_identity
    ):
        raise ValueError(
            "development RunManifest does not bind the frozen source-lineage identity"
        )
    if _run_manifest_snapshot_identity(
        run_manifest, "source_snapshot_identity"
    ) != source_identity:
        raise ValueError(
            "development RunManifest does not bind the full frozen source snapshot"
        )
    development_full_member_hash = panel_snapshot_hash(development_panel.panel)
    development_member_hash = panel_snapshot_hash(
        development_panel.eligible_panel
    )
    development_tape_hash = panel_snapshot_hash(development_panel.feature_tape)
    commitment = FrozenDevelopmentCommitment.from_records(
        development_root_hash=run_manifest.manifest_hash,
        champion_family_hash=champion_family_hash,
        split_identity_hash=split_identity_hash,
        split=split,
        development_member_panel_hash=development_member_hash,
        development_feature_tape_hash=development_tape_hash,
        research_universe_identity_hash=str(
            development_panel.research_universe_identity["identity_hash"]
        ),
        trading_calendar_identity_hash=str(
            development_panel.trading_calendar_identity["identity_hash"]
        ),
        source_lineage_identity_hash=str(
            development_panel.source_lineage_identity["identity_hash"]
        ),
        source_snapshot_identity_hash=str(source_identity["identity_hash"]),
        champion_records=champions,
    )
    handoff = build_development_handoff_from_commitment(
        commitment,
        development_panel.eligible_panel,
        development_panel.feature_tape,
        trading_calendar=development_panel.trading_calendar,
        research_universe_identity=development_panel.research_universe_identity,
        trading_calendar_identity=development_panel.trading_calendar_identity,
        source_lineage_identity=development_panel.source_lineage_identity,
        source_snapshot_identity=source_identity,
        policy=development_handoff_policy,
        expected_policy_hash=development_handoff_policy.policy_hash,
        terminal_label_bridges=terminal_label_bridges,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
    )
    handoff_label_audit = handoff.provenance.get("development_label_audit")
    if (
        not isinstance(handoff_label_audit, Mapping)
        or pre_search_development_label_audit.get("all_passed") is not True
        or pre_search_development_label_audit.get("policy")
        != development_handoff_policy.to_dict()
        or pre_search_development_label_audit.get("all_development")
        != handoff_label_audit.get("all_development")
        or pre_search_development_label_audit.get("train")
        != handoff_label_audit.get("train")
        or pre_search_development_label_audit.get("inner_validation")
        != handoff_label_audit.get("inner_validation")
    ):
        raise ValueError(
            "pre-search and frozen-handoff development-label audits disagree"
        )
    composite_family = fit_frozen_development_composite_family(
        handoff,
        policy=composite_family_policy,
        expected_policy_hash=composite_family_policy.policy_hash,
    )
    composite_manifest_path = dump_frozen_composite_family(
        composite_family,
        config.output_dir / "composite",
    )
    composite_manifest_sha256 = _sha256_file(composite_manifest_path)
    loaded_family = load_frozen_composite_family(
        composite_manifest_path.parent,
        expected_composite_family_hash=composite_family.composite_family_hash,
    )
    loaded_family.verify_integrity()
    development_evidence = _append_development_completion_evidence(
        run_dir=config.output_dir,
        run_manifest=run_manifest,
        champions=champions,
        champion_family_hash=champion_family_hash,
        composite_family_hash=composite_family.composite_family_hash,
        composite_manifest_sha256=composite_manifest_sha256,
    )
    reporting_contract = sealed_reporting_contract_v1()
    payload: dict[str, object] = {
        "schema_version": SEALED_REVEAL_SCHEMA_VERSION,
        "kind": "hs300-frozen-development-to-sealed-reveal",
        "status": "frozen_before_reveal",
        "created_at": utc_now(),
        "warning": HISTORICAL_REPLAY_WARNING,
        "development_run_manifest_filename": DEVELOPMENT_EVIDENCE_LOCATORS[
            "run_manifest"
        ],
        "development_run_manifest_sha256": _sha256_file(run_manifest_path),
        "development_run_manifest_hash": run_manifest.manifest_hash,
        "development_completion_evidence": development_evidence,
        "composite_manifest_locator": "composite/family_manifest.json",
        "composite_manifest_sha256": composite_manifest_sha256,
        "composite_family_hash": composite_family.composite_family_hash,
        "composite_primary_backend": composite_family.primary_backend,
        "sealed_reporting_contract": reporting_contract,
        "sealed_reporting_contract_hash": hash_json(reporting_contract),
        "development_label_audit": dict(
            composite_family.family_commitment["development_label_audit"]
        ),
        "pre_search_development_label_audit": dict(
            pre_search_development_label_audit
        ),
        "source_snapshot_identity": source_identity,
        "research_implementation_identity": implementation_identity,
        "resolved_experiment_config": resolved_experiment_config,
        "development_view_identity": {
            "hard_end_date": str(split.development_end.date()),
            "member_panel_snapshot_hash": development_member_hash,
            "eligible_member_panel_snapshot_hash": development_member_hash,
            "full_member_panel_snapshot_hash": development_full_member_hash,
            "feature_tape_snapshot_hash": development_tape_hash,
            "research_universe_identity": dict(
                development_panel.research_universe_identity
            ),
            "research_universe_identity_hash": str(
                development_panel.research_universe_identity["identity_hash"]
            ),
            "trading_calendar_identity": dict(
                development_panel.trading_calendar_identity
            ),
            "trading_calendar_identity_hash": str(
                development_panel.trading_calendar_identity["identity_hash"]
            ),
            "source_lineage_identity": dict(
                development_panel.source_lineage_identity
            ),
            "source_lineage_identity_hash": str(
                development_panel.source_lineage_identity["identity_hash"]
            ),
            "source_snapshot_identity": dict(source_identity),
            "source_snapshot_identity_hash": str(source_identity["identity_hash"]),
            "sealed_market_rows_loaded": 0,
        },
        "split_cut_spec": {
            "train_end": config.train_end,
            "validation_end": config.validation_end,
            "replay_end": config.replay_end,
            "label_horizon": config.label_horizon,
            "entry_delay": config.entry_delay,
        },
        "split_exact": split_exact,
        "split_identity_hash": split_identity_hash,
        "champions": champions,
        "champion_family_hash": champion_family_hash,
        "search_budget": dict(run_manifest.search_budget),
        "expected_members": config.expected_members,
        "operational_locators": _operational_locators(config),
    }
    envelope = {"manifest": payload, "manifest_hash": hash_json(payload)}
    target = config.output_dir / "sealed_reveal_manifest.json"
    _exclusive_json(envelope, target)
    return target, str(envelope["manifest_hash"])


def _load_reveal_manifest(path: Path | str) -> tuple[Path, dict[str, object], str]:
    target = Path(path).resolve()
    envelope = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(envelope, Mapping) or not isinstance(envelope.get("manifest"), Mapping):
        raise ValueError("development reveal manifest envelope is invalid")
    payload = dict(envelope["manifest"])
    manifest_hash = hash_json(payload)
    if str(envelope.get("manifest_hash", "")) != manifest_hash:
        raise ValueError("development reveal manifest hash mismatch")
    if (
        payload.get("schema_version") != SEALED_REVEAL_SCHEMA_VERSION
        or payload.get("kind") != "hs300-frozen-development-to-sealed-reveal"
        or payload.get("status") != "frozen_before_reveal"
    ):
        raise ValueError("development reveal manifest schema or status is invalid")

    reporting_contract = sealed_reporting_contract_v1()
    if (
        payload.get("sealed_reporting_contract") != reporting_contract
        or payload.get("sealed_reporting_contract_hash")
        != hash_json(reporting_contract)
    ):
        raise ValueError("development reveal manifest reporting contract is invalid")

    _validate_research_implementation_snapshot(
        payload.get("research_implementation_identity")
    )

    source_identity = payload.get("source_snapshot_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("development reveal manifest source identity is invalid")
    hash_names = (
        "panel_manifest_sha256",
        "member_panel_snapshot_hash",
        "member_partition_records_hash",
        "feature_tape_snapshot_hash",
        "feature_tape_partition_records_hash",
        "full_source_contract_hash",
        "research_eligibility_snapshot_hash",
        "source_lineage_root_sha256",
        "frozen_trade_calendar_sha256",
        "combined_data_snapshot_hash",
        "identity_hash",
    )
    for name in hash_names:
        value = str(source_identity.get(name, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"development reveal manifest {name} is not SHA-256")
    terminal_sidecar_identity = source_identity.get("terminal_sidecar_identity")
    full_source_contract = source_identity.get("full_source_contract")
    eligibility_identity = source_identity.get("research_eligibility_identity")
    source_lineage_identity = source_identity.get("source_lineage_identity")
    if not isinstance(source_lineage_identity, Mapping):
        raise ValueError("development reveal source-lineage identity is invalid")
    canonical_lineage = _canonical_source_lineage_section(
        {
            key: value
            for key, value in source_lineage_identity.items()
            if key != "identity_hash"
        }
    )
    if (
        not isinstance(terminal_sidecar_identity, Mapping)
        or not isinstance(full_source_contract, Mapping)
        or source_identity.get("full_source_contract_hash")
        != hash_json(dict(full_source_contract))
        or not isinstance(eligibility_identity, Mapping)
        or eligibility_identity.get("identity_hash")
        != hash_json(
            {
                key: value
                for key, value in eligibility_identity.items()
                if key != "identity_hash"
            }
        )
        or eligibility_identity.get("snapshot_hash")
        != source_identity.get("research_eligibility_snapshot_hash")
        or not str(source_identity.get("storage_snapshot_id", ""))
        or terminal_sidecar_identity.get("identity_hash")
        != hash_json(
            {
                key: value
                for key, value in terminal_sidecar_identity.items()
                if key != "identity_hash"
            }
        )
        or source_lineage_identity.get("identity_hash")
        != hash_json(canonical_lineage)
        or source_lineage_identity.get("root_sha256")
        != source_identity.get("source_lineage_root_sha256")
        or source_identity.get("research_rule_identity")
        != research_rule_identity_v1()
    ):
        raise ValueError(
            "development reveal source sidecar/rule identity is invalid"
        )
    expected_combined = hashlib.sha256(
        (
            str(source_identity["member_panel_snapshot_hash"])
            + "|"
            + str(source_identity["feature_tape_snapshot_hash"])
            + "|"
            + str(source_identity["research_eligibility_snapshot_hash"])
            + "|"
            + str(
                terminal_sidecar_identity.get(
                    "identity_hash", ""
                )
            )
            + "|"
            + str(source_identity["frozen_trade_calendar_sha256"])
            + "|"
            + str(source_lineage_identity.get("identity_hash", ""))
        ).encode("utf-8")
    ).hexdigest()
    if expected_combined != str(source_identity["combined_data_snapshot_hash"]):
        raise ValueError("development reveal manifest combined snapshot hash mismatch")
    if source_identity.get("identity_hash") != hash_json(
        {
            key: value
            for key, value in source_identity.items()
            if key != "identity_hash"
        }
    ):
        raise ValueError("development reveal full source-snapshot identity is invalid")

    development_identity = payload.get("development_view_identity")
    if not isinstance(development_identity, Mapping):
        raise ValueError("development reveal manifest development identity is invalid")
    if development_identity.get("sealed_market_rows_loaded") != 0:
        raise ValueError("development manifest does not attest a zero-row sealed boundary")

    split_exact = payload.get("split_exact")
    split = _split_from_payload(split_exact)
    if hash_json(split_exact) != str(payload.get("split_identity_hash", "")):
        raise ValueError("development reveal manifest split identity mismatch")
    if _split_payload(split) != split_exact:
        raise ValueError("development reveal manifest split is not canonical")
    if str(development_identity.get("hard_end_date", "")) != str(
        split.development_end.date()
    ):
        raise ValueError("development-view boundary disagrees with the exact split")
    for name in (
        "member_panel_snapshot_hash",
        "eligible_member_panel_snapshot_hash",
        "full_member_panel_snapshot_hash",
        "feature_tape_snapshot_hash",
        "research_universe_identity_hash",
        "trading_calendar_identity_hash",
        "source_lineage_identity_hash",
        "source_snapshot_identity_hash",
    ):
        value = str(development_identity.get(name, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"development-view {name} is not SHA-256")
    for name in (
        "research_universe_identity",
        "trading_calendar_identity",
        "source_lineage_identity",
        "source_snapshot_identity",
    ):
        identity = development_identity.get(name)
        expected = development_identity.get(f"{name}_hash")
        if (
            not isinstance(identity, Mapping)
            or identity.get("identity_hash") != expected
            or hash_json(
                {key: value for key, value in identity.items() if key != "identity_hash"}
            )
            != expected
        ):
            raise ValueError(f"development-view {name} is not authenticated")
    if development_identity.get("source_lineage_identity") != dict(
        source_lineage_identity
    ):
        raise ValueError(
            "development view and full source snapshot disagree on source lineage"
        )
    if development_identity.get("source_snapshot_identity") != dict(
        source_identity
    ):
        raise ValueError(
            "development view is not bound to the full source snapshot"
        )
    development_universe = development_identity.get(
        "research_universe_identity"
    )
    development_calendar = development_identity.get(
        "trading_calendar_identity"
    )
    if (
        not isinstance(development_universe, Mapping)
        or development_universe.get("terminal_sidecar_identity")
        != terminal_sidecar_identity
        or not isinstance(development_calendar, Mapping)
        or development_calendar.get("normalized_trade_calendar_sha256")
        != source_identity.get("frozen_trade_calendar_sha256")
        or full_source_contract.get("trade_calendar_sha256")
        != source_identity.get("frozen_trade_calendar_sha256")
        or full_source_contract.get("source_lineage_root_sha256")
        != source_identity.get("source_lineage_root_sha256")
    ):
        raise ValueError(
            "development universe/calendar contracts differ from the full source"
        )

    cut_spec = payload.get("split_cut_spec")
    if not isinstance(cut_spec, Mapping):
        raise ValueError("development reveal manifest split cut specification is invalid")
    train_cut = pd.Timestamp(str(cut_spec.get("train_end", ""))).normalize()
    validation_cut = pd.Timestamp(
        str(cut_spec.get("validation_end", ""))
    ).normalize()
    replay_cut = pd.Timestamp(str(cut_spec.get("replay_end", ""))).normalize()
    if not train_cut < validation_cut < replay_cut:
        raise ValueError("development reveal manifest split cuts are not ordered")
    if (
        int(cut_spec.get("label_horizon", 0)) != split.label_horizon
        or int(cut_spec.get("entry_delay", -1)) != split.entry_delay
        or max(split.train_dates) > train_cut
        or max(split.validation_dates) > validation_cut
        or max(split.outcome_tail) > replay_cut
    ):
        raise ValueError("development reveal manifest cut spec disagrees with exact split")

    champions = payload.get("champions")
    if not isinstance(champions, list) or not champions:
        raise ValueError("development reveal manifest has no frozen champions")
    if hash_json(champions) != str(payload.get("champion_family_hash", "")):
        raise ValueError("development reveal manifest champion-family hash mismatch")
    candidate_ids: set[str] = set()
    expression_hashes: set[str] = set()
    for item in champions:
        if not isinstance(item, Mapping):
            raise ValueError("development reveal manifest champion is invalid")
        candidate = _candidate_from_frozen_champion(item)
        candidate_id = candidate.candidate_id
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("development reveal manifest champion ids are invalid")
        candidate_ids.add(candidate_id)
        expression_digest = candidate.expression_hash
        if expression_digest in expression_hashes:
            raise ValueError("development reveal manifest repeats a champion expression")
        expression_hashes.add(expression_digest)
        score = item.get("score")
        if not isinstance(score, Mapping) or score.get("passed") is not True:
            raise ValueError("frozen champion score is not a passed development result")

    run_manifest_name = str(payload.get("development_run_manifest_filename", ""))
    if Path(run_manifest_name).name != run_manifest_name or run_manifest_name != "manifest.json":
        raise ValueError("development run manifest locator is not a fixed sibling")
    run_manifest_path = target.parent / run_manifest_name
    if _sha256_file(run_manifest_path) != str(
        payload.get("development_run_manifest_sha256", "")
    ):
        raise ValueError("development run manifest byte identity mismatch")
    run_manifest = RunManifest.load(run_manifest_path)
    if run_manifest.manifest_hash != str(payload.get("development_run_manifest_hash", "")):
        raise ValueError("development run manifest semantic identity mismatch")
    resolved_experiment_config = payload.get("resolved_experiment_config")
    if not isinstance(resolved_experiment_config, Mapping):
        raise ValueError("development manifest omits its resolved experiment config")
    if hash_json(resolved_experiment_config) != run_manifest.config_hash:
        raise ValueError(
            "development RunManifest does not bind the resolved experiment config"
        )
    if resolved_experiment_config.get(
        "research_implementation_identity"
    ) != payload.get("research_implementation_identity"):
        raise ValueError(
            "resolved experiment config disagrees with implementation identity"
        )
    if (
        resolved_experiment_config.get("research_universe_identity")
        != development_identity.get("research_universe_identity")
        or resolved_experiment_config.get("trading_calendar_identity")
        != development_identity.get("trading_calendar_identity")
        or resolved_experiment_config.get("source_lineage_identity")
        != development_identity.get("source_lineage_identity")
        or resolved_experiment_config.get("source_snapshot_identity")
        != development_identity.get("source_snapshot_identity")
    ):
        raise ValueError(
            "resolved experiment config disagrees with universe/calendar/source-lineage identity"
        )
    if run_manifest.data_snapshot.get("source_lineage_identity") != (
        development_identity.get("source_lineage_identity")
    ):
        raise ValueError(
            "development RunManifest data snapshot disagrees with source lineage"
        )
    if run_manifest.data_snapshot.get("source_snapshot_identity") != dict(
        source_identity
    ):
        raise ValueError(
            "development RunManifest does not anchor the full source snapshot"
        )
    operational_locators = payload.get("operational_locators")
    if (
        not isinstance(operational_locators, Mapping)
        or resolved_experiment_config.get("operational_locators")
        != dict(operational_locators)
    ):
        raise ValueError(
            "development RunManifest does not bind operational locators"
        )
    trusted_root = _trusted_project_root()
    trusted_registry = (trusted_root / "runs" / "replay_reveal_registry").resolve()
    if (
        Path(str(operational_locators.get("project_dir", ""))).resolve()
        != trusted_root
        or Path(
            str(operational_locators.get("reveal_registry_dir", ""))
        ).resolve()
        != trusted_registry
    ):
        raise ValueError("development manifest does not use trusted operational roots")
    composite_locator = payload.get("composite_manifest_locator")
    if composite_locator != "composite/family_manifest.json":
        raise ValueError("development manifest composite locator is not fixed")
    composite_manifest_path = (target.parent / str(composite_locator)).resolve()
    if composite_manifest_path != (
        target.parent / "composite" / "family_manifest.json"
    ).resolve():
        raise ValueError("development manifest composite locator escapes run dir")
    composite_manifest_sha256 = str(
        payload.get("composite_manifest_sha256", "")
    )
    if (
        not composite_manifest_path.is_file()
        or _sha256_file(composite_manifest_path) != composite_manifest_sha256
    ):
        raise ValueError("frozen composite manifest byte identity mismatch")
    composite_family_hash = str(payload.get("composite_family_hash", ""))
    composite_family = load_frozen_composite_family(
        composite_manifest_path.parent,
        expected_composite_family_hash=composite_family_hash,
    )
    commitment = composite_family.family_commitment
    development_identity = payload.get("development_view_identity")
    if not isinstance(development_identity, Mapping):
        raise ValueError("development view identity is missing for composite")
    expected_composite_links = {
        "development_root_hash": run_manifest.manifest_hash,
        "champion_family_hash": str(payload.get("champion_family_hash", "")),
        "split_identity_hash": str(payload.get("split_identity_hash", "")),
        "development_member_panel_hash": str(
            development_identity.get("member_panel_snapshot_hash", "")
        ),
        "development_feature_tape_hash": str(
            development_identity.get("feature_tape_snapshot_hash", "")
        ),
        "research_universe_identity_hash": str(
            development_identity.get("research_universe_identity_hash", "")
        ),
        "trading_calendar_identity_hash": str(
            development_identity.get("trading_calendar_identity_hash", "")
        ),
        "source_lineage_identity_hash": str(
            development_identity.get("source_lineage_identity_hash", "")
        ),
        "source_snapshot_identity_hash": str(
            development_identity.get("source_snapshot_identity_hash", "")
        ),
    }
    if (
        any(
            commitment.get(name) != value
            for name, value in expected_composite_links.items()
        )
        or commitment.get("source_lineage_identity")
        != development_identity.get("source_lineage_identity")
        or commitment.get("source_snapshot_identity")
        != development_identity.get("source_snapshot_identity")
    ):
        raise ValueError("frozen composite family differs from development roots")
    expected_feature_columns = tuple(
        str(item.get("candidate_id", "")) for item in champions
    )
    expected_expression_hashes = tuple(
        str(item.get("expression_hash", "")) for item in champions
    )
    expected_directions = {
        str(item.get("candidate_id", "")): int(item.get("direction", 0))
        for item in champions
    }
    if (
        tuple(commitment.get("feature_columns", ())) != expected_feature_columns
        or tuple(commitment.get("expression_hashes", ()))
        != expected_expression_hashes
        or dict(commitment.get("frozen_train_only_directions", {}))
        != expected_directions
    ):
        raise ValueError(
            "frozen composite columns/expressions/directions differ from champions"
        )
    if (
        composite_family.primary_backend != "lightgbm"
        or payload.get("composite_primary_backend") != "lightgbm"
        or commitment.get("primary_backend_pre_registered") != "lightgbm"
    ):
        raise ValueError("frozen composite primary backend is not exact LightGBM")
    if resolved_experiment_config.get("composite_family_policy") != commitment.get(
        "composite_family_policy"
    ):
        raise ValueError("resolved config differs from frozen composite policy")
    if resolved_experiment_config.get(
        "development_handoff_policy"
    ) != commitment.get("development_handoff_policy"):
        raise ValueError("resolved config differs from frozen handoff policy")
    if payload.get("development_label_audit") != commitment.get(
        "development_label_audit"
    ):
        raise ValueError("outer manifest differs from frozen development-label audit")
    summary_payload = json.loads(
        (target.parent / DEVELOPMENT_EVIDENCE_LOCATORS["summary"]).read_text(
            encoding="utf-8"
        )
    )
    if payload.get("pre_search_development_label_audit") != summary_payload.get(
        "development_label_audit"
    ):
        raise ValueError("outer manifest differs from pre-search label audit")
    evidence = payload.get("development_completion_evidence")
    if not isinstance(evidence, Mapping) or (
        evidence.get("composite_family_hash") != composite_family_hash
        or evidence.get("composite_manifest_sha256")
        != composite_manifest_sha256
    ):
        raise ValueError("completion evidence does not bind the composite family")
    if dict(run_manifest.search_budget) != dict(payload.get("search_budget", {})):
        raise ValueError("development search budget identity mismatch")
    default_budget = SearchBudget()
    required_budget = {
        "initial_population": default_budget.initial_population,
        "parent_pool": default_budget.parent_pool,
        "children_per_generation": default_budget.children_per_generation,
        "generations": default_budget.generations,
        "injection_every": default_budget.injection_every,
        "injection_count": default_budget.injection_count,
        "elite_carry": default_budget.elite_carry,
        "final_factors": default_budget.final_factors,
    }
    if dict(run_manifest.search_budget) != required_budget:
        raise ValueError("real-data reveal requires the frozen full search budget")
    if len(champions) != int(run_manifest.search_budget.get("final_factors", -1)):
        raise ValueError("frozen champion count disagrees with the search manifest")
    if int(payload.get("expected_members", 0)) < 1:
        raise ValueError("development reveal manifest expected_members is invalid")
    _validate_development_completion_evidence(
        run_dir=target.parent,
        payload=payload,
        champions=champions,
        run_manifest=run_manifest,
    )
    # Validate every scalar score and reconstructed identity before a reveal
    # process is allowed to retire the persistent gate.
    _champions_from_frozen_manifest(champions)
    return target, payload, manifest_hash


def _claim_replay_gate(
    registry_dir: Path,
    *,
    source_snapshot_identity: Mapping[str, object],
    split_identity_hash: str,
    development_manifest_hash: str,
    champion_family_hash: str,
    composite_family_hash: str,
) -> Path:
    # Deliberately exclude run/champion identities from the filename.  One
    # snapshot/split gate may be opened only once, even if someone reruns
    # development and offers a different champion family after seeing results.
    gate_identity = {
        "combined_data_snapshot_hash": str(
            source_snapshot_identity.get("combined_data_snapshot_hash", "")
        ),
        "split_identity_hash": split_identity_hash,
    }
    digest = hash_json(gate_identity)
    target = registry_dir.resolve() / f"{digest}.json"
    payload = {
        **gate_identity,
        "source_snapshot_identity_hash": str(
            source_snapshot_identity.get("identity_hash", "")
        ),
        "schema_version": 1,
        "gate_kind": "known-history-isolated-replay",
        "status": "retired_before_any_sealed_row_was_loaded",
        "claimed_at": utc_now(),
        "development_manifest_hash": development_manifest_hash,
        "champion_family_hash": champion_family_hash,
        # Bound as evidence only. Deliberately excluded from gate_identity and
        # therefore from the filename: a different family cannot reveal the
        # same snapshot/split a second time.
        "composite_family_hash": composite_family_hash,
        "panel_manifest_sha256": str(
            source_snapshot_identity.get("panel_manifest_sha256", "")
        ),
        "warning": HISTORICAL_REPLAY_WARNING,
    }
    try:
        return _exclusive_json(payload, target)
    except FileExistsError as exc:
        raise RuntimeError(
            "this data-snapshot/split historical replay gate was already revealed"
        ) from exc


def _artifact_from_frozen_champion(item: Mapping[str, object]) -> FactorArtifact:
    kwargs: dict[str, Any] = {
        "candidate_id": str(item["candidate_id"]),
        "canonical_expression_json": str(item["canonical_expression_json"]),
        "expression_hash": str(item["expression_hash"]),
        "direction": int(item["direction"]),
        "train_rank_ic": float(item["train_rank_ic"]),
        "validation_rank_ic": float(item["validation_rank_ic"]),
        "validation_neutral_rank_ic": float(item["validation_neutral_rank_ic"]),
    }
    # Compatibility with an older in-memory artifact schema.  These values are
    # never serialized and SealedGate never consumes them.
    artifact_fields = {field.name for field in fields(FactorArtifact)}
    if "development_values" in artifact_fields:
        kwargs["development_values"] = pd.Series(dtype=float)
    if "neutral_values" in artifact_fields:
        kwargs["neutral_values"] = pd.Series(dtype=float)
    return FactorArtifact(**kwargs)


def _champions_from_frozen_manifest(
    records: Sequence[Mapping[str, object]],
) -> tuple[list[EvaluatedCandidate], dict[str, FactorArtifact]]:
    champions: list[EvaluatedCandidate] = []
    artifacts: dict[str, FactorArtifact] = {}
    for item in records:
        score_payload = item.get("score")
        if not isinstance(score_payload, Mapping):
            raise ValueError("frozen champion score is invalid")
        candidate = _candidate_from_frozen_champion(item)
        score = CandidateScore(
            fitness=float(score_payload["fitness"]),
            signed_rank_ic=float(score_payload["signed_rank_ic"]),
            neutral_rank_ic=float(score_payload["neutral_rank_ic"]),
            turnover=float(score_payload["turnover"]),
            worst_window_ic=float(score_payload["worst_window_ic"]),
            style_retention=float(score_payload["style_retention"]),
            novelty=float(score_payload["novelty"]),
            control_delta=(
                None
                if score_payload.get("control_delta") is None
                else float(score_payload["control_delta"])
            ),
            passed=bool(score_payload["passed"]),
        )
        champions.append(EvaluatedCandidate(candidate, score))
        artifacts[candidate.candidate_id] = _artifact_from_frozen_champion(item)
    return champions, artifacts


def run_hs300_research(
    config: HS300ResearchConfig,
    *,
    budget: SearchBudget | None = None,
) -> HS300ResearchResult:
    """Run development search without materializing any historical-replay row."""

    if config.project_dir != _trusted_project_root():
        raise ValueError(
            "real HS300 research project_dir must equal the executing implementation root"
        )
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError("research output directory must be new or empty")
    # Freeze implementation bytes before any Factor Card is parsed.  A file
    # change after this point will fail the manifest validation at the end of
    # development, rather than letting candidates from one byte version be
    # attributed to another (load/snapshot TOCTOU).
    implementation_identity = _research_implementation_snapshot(
        config.project_dir
    )
    # Economic semantics are a hard pre-data gate, not metadata attached after
    # a successful search. Validate all 176 explicit independent review
    # records against the code-side proposal anchors before reading any market
    # partition or constructing a label. Factor Cards are then parsed under
    # the same byte snapshot and the complete implementation is re-hashed at
    # once, closing the validate/load TOCTOU window.
    semantic_review_manifest = validate_semantic_review_manifest(
        config.project_dir,
        config.project_dir / SEMANTIC_REVIEW_RELATIVE_PATH,
    )
    candidates = load_codex_seed_candidates(
        config.project_dir,
        horizon=config.label_horizon,
    )
    injection_candidates = load_codex_injection_candidates(
        config.project_dir,
        horizon=config.label_horizon,
    )
    _validate_research_implementation_snapshot(
        implementation_identity,
        project_dir=config.project_dir,
    )
    # Resolve every adaptive, selection and composite policy before even the
    # source manifest or a market partition is opened.  The same immutable
    # objects are later bound into RunManifest and rechecked at freeze.
    adaptive_feedback_policy = _load_adaptive_feedback_policy(
        config.project_dir,
        config.profile,
    )
    behavioral_champion_policy = _load_behavioral_champion_policy(
        config.project_dir,
        config.profile,
    )
    development_handoff_policy, composite_family_policy = (
        _load_composite_policies(config.project_dir, config.profile)
    )
    terminal_label_bridges, terminal_label_bridge_identity = (
        load_terminal_label_bridges(
            config.project_dir / "configs" / "hs300_terminal_label_bridges.yaml"
        )
    )
    _validate_research_implementation_snapshot(
        implementation_identity,
        project_dir=config.project_dir,
    )
    panel_root, source_manifest, source_manifest_sha256 = (
        _load_frozen_panel_manifest(config.panel_dir)
    )
    declared_replay_end = pd.Timestamp(
        str(source_manifest.get("last_date"))
    ).normalize()
    if pd.Timestamp(config.replay_end).normalize() != declared_replay_end:
        raise ValueError(
            "replay_end must equal the frozen panel last_date so its sealed "
            "snapshot has a pre-reveal semantic identity"
        )
    validation_report = source_manifest.get("validation")
    if isinstance(validation_report, Mapping) and validation_report.get("valid") is not True:
        raise ValueError("frozen panel build validation is not valid")
    trading_dates = _load_bound_trading_dates(panel_root, source_manifest)
    research_start = pd.Timestamp(str(source_manifest.get("first_date"))).normalize()
    split_dates = trading_dates[trading_dates >= research_start]
    split = _make_historical_replay_split_from_dates(
        split_dates,
        train_end=config.train_end,
        validation_end=config.validation_end,
        replay_end=config.replay_end,
        label_horizon=config.label_horizon,
        entry_delay=config.entry_delay,
    )
    # This is the hard capability boundary.  The loader filters inside Arrow;
    # it is not a full load followed by a pandas slice.
    frozen = load_frozen_hs300_panel(
        config.panel_dir,
        end_date=split.development_end,
    )
    _require_stable_development_source_manifest(
        source_manifest,
        source_manifest_sha256,
        frozen.manifest,
        frozen.manifest_sha256,
    )
    full_member_panel = frozen.panel
    panel = frozen.eligible_panel
    feature_tape = frozen.feature_tape
    if full_member_panel.index.get_level_values("date").max() != split.development_end:
        raise ValueError("development panel does not reach the frozen purge boundary")
    daily_counts = panel.groupby(level="date").size()
    required_daily_names = min(20, max(1, config.expected_members - 2))
    if daily_counts.empty or daily_counts.min() < required_daily_names:
        raise ValueError("development research_eligible panel has too few daily names")
    expected_development_dates = trading_dates[
        (trading_dates >= full_member_panel.index.get_level_values("date").min())
        & (trading_dates <= split.development_end)
    ]
    actual_development_dates = pd.DatetimeIndex(
        full_member_panel.index.get_level_values("date").unique()
    ).sort_values()
    if not actual_development_dates.equals(expected_development_dates):
        raise ValueError("development panel does not cover every committed trading session")
    selected_budget = budget or SearchBudget()
    if selected_budget != SearchBudget():
        raise ValueError(
            "real HS300 run requires the frozen full 80/32/96/24 search budget; "
            "reduced budgets are reserved for synthetic tests"
        )
    source_identity = _source_snapshot_identity(
        source_manifest,
        source_manifest_sha256,
    )
    combined_snapshot_hash = str(source_identity["combined_data_snapshot_hash"])
    evolution, summary = run_experiment(
        full_panel=panel,
        feature_tape=feature_tape,
        source_member_panel=full_member_panel,
        trading_calendar=frozen.trading_calendar,
        trading_calendar_identity=frozen.trading_calendar_identity,
        research_universe_identity=frozen.research_universe_identity,
        source_lineage_identity=frozen.source_lineage_identity,
        source_snapshot_identity=source_identity,
        split=split,
        output_dir=config.output_dir,
        budget=selected_budget,
        profile=config.profile,
        seed=config.seed,
        reveal_sealed=False,
        bootstrap_samples=config.bootstrap_samples,
        evaluation_workers=config.evaluation_workers,
        model_version="codex-desktop-precommitted-176-factor-cards",
        prompt_version="cogalpha-seven-level-21-role-adaptive-repair-v2",
        initial_candidates=candidates,
        injection_candidates=injection_candidates,
        adaptive_feedback_policy=adaptive_feedback_policy,
        behavioral_champion_policy=behavioral_champion_policy,
        development_handoff_policy=development_handoff_policy.to_dict(),
        composite_family_policy=composite_family_policy.to_dict(),
        operational_locators=_operational_locators(config),
        implementation_identity=implementation_identity,
        terminal_label_bridges=terminal_label_bridges,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
    )
    reveal_manifest_path, reveal_manifest_hash = _freeze_reveal_manifest(
        config=config,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        development_panel=frozen,
        split=split,
        evolution=evolution,
        implementation_identity=implementation_identity,
        adaptive_feedback_policy=adaptive_feedback_policy,
        behavioral_champion_policy=behavioral_champion_policy,
        development_handoff_policy=development_handoff_policy,
        composite_family_policy=composite_family_policy,
        pre_search_development_label_audit=dict(
            summary.get("development_label_audit") or {}
        ),
        terminal_label_bridges=terminal_label_bridges,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
    )
    wrapper = {
        "historical_replay_warning": HISTORICAL_REPLAY_WARNING,
        "historical_replay_revealed": False,
        "sealed_market_rows_loaded_by_development": 0,
        "sealed_reveal_manifest": str(reveal_manifest_path),
        "sealed_reveal_manifest_hash": reveal_manifest_hash,
        "persistent_reveal_record": None,
        "frozen_panel_manifest_sha256": frozen.manifest_sha256,
        "frozen_panel_snapshot_hash": str(
            source_identity["member_panel_snapshot_hash"]
        ),
        "continuous_feature_tape_snapshot_hash": str(
            source_identity["feature_tape_snapshot_hash"]
        ),
        "development_member_panel_snapshot_hash": panel_snapshot_hash(panel),
        "development_full_member_panel_snapshot_hash": panel_snapshot_hash(
            full_member_panel
        ),
        "development_feature_tape_snapshot_hash": panel_snapshot_hash(feature_tape),
        "development_research_universe_identity": dict(
            frozen.research_universe_identity
        ),
        "development_source_lineage_identity": dict(
            frozen.source_lineage_identity
        ),
        "frozen_full_source_snapshot_identity": dict(source_identity),
        "frozen_trading_calendar_identity": dict(
            frozen.trading_calendar_identity
        ),
        "combined_data_snapshot_hash": combined_snapshot_hash,
        "signal_scoring_end": str(split.sealed_dates[-1].date()),
        "outcome_data_end": str(split.outcome_tail[-1].date()),
        "tail_signal_dates_not_scored": len(split.outcome_tail),
        "economic_semantic_review": {
            "schema": semantic_review_manifest["schema"],
            "candidate_count": semantic_review_manifest["candidate_count"],
            "review_set_hash": semantic_review_manifest["review_set_hash"],
            "approval_binding_hash": semantic_review_manifest[
                "approval_binding_hash"
            ],
            "aggregate_hash": hash_json(semantic_review_manifest),
        },
        "split": split.to_dict(),
        "core_search_budget": {
            "initial_population": selected_budget.initial_population,
            "parent_pool": selected_budget.parent_pool,
            "children_per_generation": selected_budget.children_per_generation,
            "generations": selected_budget.generations,
            "final_factors": selected_budget.final_factors,
        },
        "experiment_summary": summary,
    }
    _atomic_json(wrapper, config.output_dir / "hs300_research_summary.json")
    return HS300ResearchResult(
        evolution=evolution,
        summary=wrapper,
        split=split,
        data_snapshot_hash=combined_snapshot_hash,
    )


def reveal_hs300_replay(
    development_manifest: Path | str,
) -> HS300RevealResult:
    """Consume one frozen development manifest in a separate reveal process.

    No search parameters or candidate expressions are accepted from the CLI.
    The exact ASTs, directions, snapshot, split, budget and operational paths
    all come from the content-hashed development manifest.
    """

    manifest_path, payload, development_manifest_hash = _load_reveal_manifest(
        development_manifest
    )
    source_identity = payload.get("source_snapshot_identity")
    locators = payload.get("operational_locators")
    if not isinstance(source_identity, Mapping) or not isinstance(locators, Mapping):
        raise ValueError("development reveal manifest omits bound identities or locators")
    project_dir = _trusted_project_root()
    panel_dir = Path(str(locators.get("panel_dir", ""))).resolve()
    registry_dir = Path(str(locators.get("reveal_registry_dir", ""))).resolve()
    output_dir = Path(str(locators.get("sealed_output_dir", ""))).resolve()
    if not str(locators.get("project_dir", "")) or not str(
        locators.get("panel_dir", "")
    ) or not str(
        locators.get("reveal_registry_dir", "")
    ) or not str(locators.get("sealed_output_dir", "")):
        raise ValueError("development reveal manifest has blank operational locators")
    if (
        Path(str(locators.get("project_dir", ""))).resolve() != project_dir
        or registry_dir
        != (project_dir / "runs" / "replay_reveal_registry").resolve()
    ):
        raise ValueError("reveal manifest operational roots are not trusted")

    # This check must precede both the persistent one-time claim and any sealed
    # price-partition read.  A changed evaluator/DSL/controller/config must not
    # consume the gate or score known history under different semantics.
    _validate_research_implementation_snapshot(
        payload.get("research_implementation_identity"),
        project_dir=project_dir,
    )
    resolved_experiment_config = payload.get("resolved_experiment_config")
    if not isinstance(resolved_experiment_config, Mapping):
        raise ValueError("development manifest omits its resolved experiment config")
    profile = str(resolved_experiment_config.get("profile", ""))
    current_policy = _load_adaptive_feedback_policy(project_dir, profile)
    if resolved_experiment_config.get("adaptive_feedback_policy") != (
        _adaptive_feedback_policy_payload(current_policy)
    ):
        raise ValueError(
            "resolved adaptive feedback policy differs from configs/plus.yaml"
        )
    current_behavioral_policy = _load_behavioral_champion_policy(
        project_dir,
        profile,
    )
    if resolved_experiment_config.get("behavioral_champion_policy") != (
        _behavioral_champion_policy_payload(current_behavioral_policy)
    ):
        raise ValueError(
            "resolved behavioral champion policy differs from configs/plus.yaml"
        )
    current_handoff_policy, current_composite_policy = _load_composite_policies(
        project_dir,
        profile,
    )
    if resolved_experiment_config.get("development_handoff_policy") != (
        current_handoff_policy.to_dict()
    ) or resolved_experiment_config.get("composite_family_policy") != (
        current_composite_policy.to_dict()
    ):
        raise ValueError("resolved composite policies differ from profile YAML")
    terminal_label_bridges, terminal_label_bridge_identity = (
        load_terminal_label_bridges(
            project_dir / "configs" / "hs300_terminal_label_bridges.yaml"
        )
    )
    if resolved_experiment_config.get("terminal_label_bridge_identity") != (
        terminal_label_bridge_identity
    ):
        raise ValueError("resolved terminal label bridges differ from frozen catalog")
    composite_locator = str(payload.get("composite_manifest_locator", ""))
    composite_manifest_path = (manifest_path.parent / composite_locator).resolve()
    if composite_manifest_path != (
        manifest_path.parent / "composite" / "family_manifest.json"
    ).resolve():
        raise ValueError("composite artifact locator is not the fixed run-relative path")
    if _sha256_file(composite_manifest_path) != str(
        payload.get("composite_manifest_sha256", "")
    ):
        raise ValueError("composite artifact manifest changed before reveal")
    composite_family_hash = str(payload.get("composite_family_hash", ""))
    composite_family = load_frozen_composite_family(
        composite_manifest_path.parent,
        expected_composite_family_hash=composite_family_hash,
    )
    composite_family.verify_integrity()

    # Manifest-only verification is safe before retirement; no price partition
    # has been opened yet.  The atomic claim occurs before audit/full loading,
    # so crashes cannot turn the known-history gate into a reusable oracle.
    _, current_source_manifest, current_source_manifest_sha = (
        _load_frozen_panel_manifest(panel_dir)
    )
    _verify_terminal_sidecar_artifact(panel_dir, current_source_manifest)
    current_source_identity = _source_snapshot_identity(
        current_source_manifest,
        current_source_manifest_sha,
    )
    if current_source_identity != dict(source_identity):
        raise ValueError("current frozen panel identity differs from development freeze")
    split_identity_hash = str(payload.get("split_identity_hash", ""))
    claim = _claim_replay_gate(
        registry_dir,
        source_snapshot_identity=source_identity,
        split_identity_hash=split_identity_hash,
        development_manifest_hash=development_manifest_hash,
        champion_family_hash=str(payload.get("champion_family_hash", "")),
        composite_family_hash=composite_family_hash,
    )

    expected_members = int(payload.get("expected_members", 0))
    audit = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=panel_dir, expected_members=expected_members)
    )
    if not audit.valid:
        codes = [
            item.code for item in audit.findings if item.severity.value == "fail"
        ]
        raise ValueError("frozen HS300 panel failed reveal audit: " + ",".join(codes))
    frozen = load_frozen_hs300_panel(panel_dir)
    actual_identity = _source_snapshot_identity(
        frozen.manifest,
        frozen.manifest_sha256,
    )
    if actual_identity != dict(source_identity):
        raise ValueError("loaded sealed data does not match the frozen snapshot identity")

    cut_spec = payload.get("split_cut_spec")
    if not isinstance(cut_spec, Mapping):
        raise ValueError("development reveal manifest omits split cut specification")
    independently_derived_split = _make_historical_replay_split_from_dates(
        frozen.trading_calendar,
        train_end=str(cut_spec["train_end"]),
        validation_end=str(cut_spec["validation_end"]),
        replay_end=str(cut_spec["replay_end"]),
        label_horizon=int(cut_spec["label_horizon"]),
        entry_delay=int(cut_spec["entry_delay"]),
    )
    split = _split_from_payload(payload.get("split_exact"))
    if _split_payload(independently_derived_split) != _split_payload(split):
        raise ValueError("sealed data calendar disagrees with the frozen split identity")
    development_identity = payload.get("development_view_identity")
    if not isinstance(development_identity, Mapping):
        raise ValueError("development reveal manifest omits development-view identity")
    development_full_panel = frozen.panel.loc[
        frozen.panel.index.get_level_values("date") <= split.development_end
    ]
    development_panel = frozen.eligible_panel.loc[
        frozen.eligible_panel.index.get_level_values("date") <= split.development_end
    ]
    development_tape = frozen.feature_tape.loc[
        frozen.feature_tape.index.get_level_values("date") <= split.development_end
    ]
    development_universe_identity = build_research_universe_identity(
        development_full_panel,
        development_tape,
        development_panel,
        terminal_sidecar_identity=_declared_terminal_sidecar_identity(
            frozen.manifest
        ),
    )
    if (
        panel_snapshot_hash(development_panel)
        != str(development_identity.get("member_panel_snapshot_hash", ""))
        or panel_snapshot_hash(development_full_panel)
        != str(development_identity.get("full_member_panel_snapshot_hash", ""))
        or panel_snapshot_hash(development_tape)
        != str(development_identity.get("feature_tape_snapshot_hash", ""))
        or frozen.trading_calendar_identity
        != development_identity.get("trading_calendar_identity")
        or frozen.source_lineage_identity
        != development_identity.get("source_lineage_identity")
        or frozen.source_snapshot_identity
        != development_identity.get("source_snapshot_identity")
        or development_universe_identity
        != development_identity.get("research_universe_identity")
    ):
        raise ValueError(
            "full frozen source cannot reproduce the exact development view"
        )
    champion_records = payload.get("champions")
    if not isinstance(champion_records, list):
        raise ValueError("development reveal manifest champions are invalid")
    champions, artifacts = _champions_from_frozen_manifest(champion_records)
    min_names = min(
        20,
        frozen.eligible_panel.index.get_level_values("symbol").nunique() - 2,
    )
    gate = SealedGate(
        frozen.eligible_panel,
        split,
        feature_tape=frozen.feature_tape,
        trading_calendar=frozen.trading_calendar,
        research_universe_identity=frozen.research_universe_identity,
        trading_calendar_identity=frozen.trading_calendar_identity,
        min_names=min_names,
        sealed_claim_capability=SealedCompositeClaimCapability(
            composite_family_hash=composite_family_hash,
            split_identity_hash=split_identity_hash,
            combined_data_snapshot_hash=str(
                source_identity["combined_data_snapshot_hash"]
            ),
            source_snapshot_identity_hash=str(source_identity["identity_hash"]),
            persistent_claim_path=str(claim),
            persistent_claim_sha256=_sha256_file(claim),
        ),
        terminal_label_bridges=terminal_label_bridges,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
    )
    sealed_results = gate.reveal(
        champions,
        artifacts,
        composite_family=composite_family,
        composite_family_hash=composite_family_hash,
    )
    composite_reveal = gate.composite_reveal
    reporting_contract = sealed_reporting_contract_v1()
    if (
        not isinstance(composite_reveal, Mapping)
        or composite_reveal.get("primary_backend_pre_registered") != "lightgbm"
        or composite_reveal.get("backend_selection_performed") is not False
        or composite_reveal.get(
            "sealed_result_ranking_or_winner_selection_performed"
        )
        is not False
        or composite_reveal.get("joint_hypothesis_family_size") != 23
        or composite_reveal.get("reporting_contract") != reporting_contract
        or composite_reveal.get("reporting_contract_hash")
        != hash_json(reporting_contract)
    ):
        raise ValueError(
            "sealed gate did not return the exact frozen composite/reporting family"
        )
    composite_records = composite_reveal.get("sealed_composite_results")
    primary_binding = composite_reveal.get("primary_result_binding")
    if not isinstance(composite_records, list) or not isinstance(
        primary_binding, Mapping
    ):
        raise ValueError("sealed reporting contract omits result records or binding")
    primary_records = [
        record
        for record in composite_records
        if isinstance(record, Mapping)
        and record.get("eligible_for_primary_claim") is True
    ]
    secondary_control_keys = {
        str(record.get("result_key"))
        for record in composite_records
        if isinstance(record, Mapping)
        and record.get("reporting_role")
        == "pre_registered_secondary_model_control"
        and record.get("eligible_for_primary_claim") is False
    }
    primary_record = primary_records[0] if len(primary_records) == 1 else None
    if (
        primary_record is None
        or primary_record.get("result_key")
        != reporting_contract["unique_primary_result_key"]
        or primary_binding.get("result_key") != primary_record.get("result_key")
        or primary_binding.get("backend") != primary_record.get("backend")
        or primary_binding.get("endpoint_key")
        != reporting_contract["primary_endpoint_key"]
        or primary_binding.get("inference_key")
        != reporting_contract["primary_inference_key"]
        or primary_binding.get("result_record_hash") != hash_json(primary_record)
        or secondary_control_keys
        != set(reporting_contract["control_composite_result_keys"])
        or any(
            record.get("eligible_for_primary_claim") is not False
            or record.get("reporting_role")
            != "pre_registered_secondary_component_factor_diagnostic"
            or record.get("immediate_executable_eligible_for_primary_claim")
            is not False
            for record in sealed_results
        )
        or any(
            not isinstance(record, Mapping)
            or record.get("eligible_for_primary_claim") is not False
            or record.get("reporting_role")
            != "pre_registered_secondary_model_comparison"
            for record in composite_reveal.get(
                "pre_registered_neutral_ic_contrasts", []
            )
        )
        or any(
            not isinstance(record, Mapping)
            or record.get("immediate_executable_eligible_for_primary_claim")
            is not False
            for record in composite_records
        )
    ):
        raise ValueError("sealed result roles violate the frozen reporting contract")
    result_payload: dict[str, object] = {
        "schema_version": 3,
        "kind": "hs300-one-time-historical-replay-result",
        "created_at": utc_now(),
        "warning": HISTORICAL_REPLAY_WARNING,
        "development_manifest": str(manifest_path),
        "development_manifest_hash": development_manifest_hash,
        "reveal_registry_record": str(claim),
        "source_snapshot_identity": dict(source_identity),
        "split_identity_hash": split_identity_hash,
        "champion_family_hash": str(payload["champion_family_hash"]),
        "composite_family_hash": composite_family_hash,
        "primary_backend_pre_registered": "lightgbm",
        "backend_selection_performed": False,
        "sealed_result_ranking_or_winner_selection_performed": False,
        "reporting_contract": reporting_contract,
        "reporting_contract_hash": hash_json(reporting_contract),
        "primary_result_binding": dict(primary_binding),
        "sealed_results": sealed_results,
        "sealed_factor_results": sealed_results,
        "sealed_composite_reveal": dict(composite_reveal),
    }
    output_path = output_dir / "sealed_replay_results.json"
    _exclusive_json(result_payload, output_path)
    completion_path = claim.with_suffix(".result.json")
    _exclusive_json(
        {
            "status": "reveal_completed",
            "gate_record": str(claim),
            "result_path": str(output_path),
            "result_sha256": _sha256_file(output_path),
            "completed_at": utc_now(),
        },
        completion_path,
    )
    return HS300RevealResult(
        results=tuple(sealed_results),
        output_path=output_path,
        registry_record=claim,
        development_manifest_hash=development_manifest_hash,
        data_snapshot_hash=str(source_identity["combined_data_snapshot_hash"]),
        split=split,
    )


__all__ = [
    "FrozenPanel",
    "HISTORICAL_REPLAY_WARNING",
    "HS300ResearchConfig",
    "HS300ResearchResult",
    "HS300RevealResult",
    "SEMANTIC_REVIEW_RELATIVE_PATH",
    "load_codex_injection_candidates",
    "load_codex_seed_candidates",
    "load_frozen_hs300_panel",
    "make_historical_replay_split",
    "reveal_hs300_replay",
    "run_hs300_research",
]
