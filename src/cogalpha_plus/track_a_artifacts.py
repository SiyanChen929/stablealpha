"""Immutable development freeze and reconstruction for Track A reveal."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .dsl import canonical_json, expression_hash
from .evolution import Candidate, CandidateScore, EvaluatedCandidate, deterministic_candidate_id
from .runner import score_payload
from .track_a import TrackADevelopmentBundle, split_identity
from .validation import FactorArtifact, PRODUCTION_DSL_SPEC


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def track_a_implementation_identity(project_dir: Path | str) -> dict[str, object]:
    project = Path(project_dir).resolve()
    fixed = [
        project / "pyproject.toml",
        project / "configs" / "track_a.yaml",
        project / "configs" / "agents.yaml",
        project / "configs" / "hs300_terminal_label_bridges.yaml",
        project / "reviews" / "semantic_review_176.json",
        *sorted((project / "proposals").rglob("*.jsonl")),
        *sorted((project / "src" / "cogalpha_plus").rglob("*.py")),
        *sorted((project / "scripts").glob("*track_a*.py")),
    ]
    files = []
    for path in fixed:
        if not path.is_file():
            raise FileNotFoundError(f"Track A implementation input is missing: {path}")
        files.append(
            {
                "path": str(path.relative_to(project)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema": "track_a_implementation_identity_v1",
        "files": files,
    }
    return {**payload, "sha256": _hash_json(payload)}


def exchange_identity(exchange_dir: Path | str, *, generations: int = 24) -> dict[str, object]:
    root = Path(exchange_dir).resolve()
    records = []
    for generation in range(1, generations + 1):
        for kind in ("requests", "responses", "transcripts"):
            path = root / kind / f"generation_{generation:02d}.json"
            if not path.is_file():
                raise FileNotFoundError(f"generation exchange artifact missing: {path}")
            records.append(
                {
                    "generation": generation,
                    "kind": kind,
                    "sha256": sha256_file(path),
                    "bytes": int(path.stat().st_size),
                }
            )
    payload = {
        "schema": "track_a_generation_exchange_identity_v1",
        "generations": generations,
        "records": records,
    }
    return {**payload, "sha256": _hash_json(payload)}


def _candidate_record(item: EvaluatedCandidate) -> dict[str, object]:
    return {
        "candidate": item.candidate.to_dict(),
        "score": score_payload(item.score),
    }


def freeze_track_a_development(
    *,
    project_dir: Path | str,
    run_dir: Path | str,
    bundle: TrackADevelopmentBundle,
    champions: Sequence[EvaluatedCandidate],
    validation_model_metrics: Mapping[str, Mapping[str, object]],
    factor_matrix_identity: Mapping[str, object],
    expected_implementation_identity: Mapping[str, object] | None = None,
) -> Path:
    if len(champions) != 20:
        raise ValueError("Track A development freeze requires 20 champions")
    project = Path(project_dir).resolve()
    run = Path(run_dir).resolve()
    summary_path = run / "summary.json"
    manifest_path = run / "manifest.json"
    ledger_path = run / "ledger" / "trials.jsonl"
    if not all(path.is_file() for path in (summary_path, manifest_path, ledger_path)):
        raise FileNotFoundError("Track A search artifacts are incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("sealed_revealed") is not False:
        raise ValueError("development freeze requires an unseen sealed partition")
    champion_records = [_candidate_record(item) for item in champions]
    family_hash = _hash_json(champion_records)
    current_implementation_identity = track_a_implementation_identity(project)
    if (
        expected_implementation_identity is not None
        and current_implementation_identity != dict(expected_implementation_identity)
    ):
        raise ValueError("Track A implementation changed during development search")
    payload = {
        "schema": "track_a_development_freeze_v1",
        "status": "DEVELOPMENT_FROZEN_TEST_UNSEEN",
        "project_dir": str(project),
        "track_a_config_sha256": bundle.protocol.config_sha256,
        "panel_manifest_sha256": bundle.protocol.panel_manifest_sha256,
        "split_identity": split_identity(bundle.split),
        "search_manifest_sha256": sha256_file(manifest_path),
        "search_summary_sha256": sha256_file(summary_path),
        "trial_ledger_sha256": sha256_file(ledger_path),
        "trial_ledger_head_hash": summary["ledger"]["head_hash"],
        "evaluated_candidates": int(summary["evaluated_candidates"]),
        "generation_exchange_identity": exchange_identity(run / "exchange"),
        "implementation_identity": current_implementation_identity,
        "factor_matrix_identity": dict(factor_matrix_identity),
        "validation_model_metrics": {
            str(key): dict(value) for key, value in validation_model_metrics.items()
        },
        "champion_family_hash": family_hash,
        "champions": champion_records,
        "model_selection_after_sealed_results": False,
        "test_metrics_seen": False,
    }
    payload["freeze_sha256"] = _hash_json(payload)
    target = run / "development_freeze.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return target


def _load_candidate(payload: Mapping[str, object]) -> Candidate:
    candidate = Candidate(
        expression=dict(payload["expression"]),
        agent_id=str(payload["agent_id"]),
        hypothesis=str(payload["hypothesis"]),
        mechanism_tags=tuple(payload["mechanism_tags"]),
        generation=int(payload["generation"]),
        parents=tuple(payload["parents"]),
        operator=str(payload["operator"]),
        candidate_id=str(payload["candidate_id"]),
        control_expression=(
            None
            if payload.get("control_expression") is None
            else dict(payload["control_expression"])
        ),
        metadata=dict(payload["metadata"]),
    )
    if candidate.expression_hash != payload.get("expression_hash"):
        raise ValueError("frozen candidate expression hash mismatch")
    if deterministic_candidate_id(candidate) != candidate.candidate_id:
        raise ValueError("frozen candidate id is not content-addressed")
    return candidate


def _load_score(payload: Mapping[str, object]) -> CandidateScore:
    return CandidateScore(
        fitness=float(payload["fitness"]),
        signed_rank_ic=float(payload["signed_rank_ic"]),
        neutral_rank_ic=float(payload["neutral_rank_ic"]),
        turnover=float(payload["turnover"]),
        worst_window_ic=float(payload["worst_window_ic"]),
        style_retention=float(payload["style_retention"]),
        novelty=float(payload["novelty"]),
        control_delta=(
            None if payload.get("control_delta") is None else float(payload["control_delta"])
        ),
        passed=bool(payload["passed"]),
        diagnostics=dict(payload.get("diagnostics", {})),
    )


def load_track_a_development_freeze(
    path: Path | str,
    *,
    verify_current_implementation: bool = True,
) -> tuple[Mapping[str, object], tuple[EvaluatedCandidate, ...], dict[str, FactorArtifact]]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != "track_a_development_freeze_v1":
        raise ValueError("Track A development freeze schema mismatch")
    clean = {key: value for key, value in payload.items() if key != "freeze_sha256"}
    if payload.get("freeze_sha256") != _hash_json(clean):
        raise ValueError("Track A development freeze hash mismatch")
    records = payload.get("champions")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("Track A development freeze must contain 20 champions")
    if payload.get("champion_family_hash") != _hash_json(records):
        raise ValueError("Track A champion family hash mismatch")
    if verify_current_implementation:
        current = track_a_implementation_identity(str(payload["project_dir"]))
        if current != payload.get("implementation_identity"):
            raise ValueError("Track A implementation changed after development freeze")
    champions: list[EvaluatedCandidate] = []
    artifacts: dict[str, FactorArtifact] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Track A champion record is invalid")
        candidate_payload = record.get("candidate")
        score_payload_raw = record.get("score")
        if not isinstance(candidate_payload, Mapping) or not isinstance(score_payload_raw, Mapping):
            raise ValueError("Track A champion candidate/score record is invalid")
        candidate = _load_candidate(candidate_payload)
        score = _load_score(score_payload_raw)
        if not score.passed:
            raise ValueError("Track A frozen champion did not pass development")
        champions.append(EvaluatedCandidate(candidate, score))
        direction = int(candidate.metadata["expected_direction"])
        train_raw = float(score.diagnostics["train_raw_rank_ic"]) * direction
        artifacts[candidate.candidate_id] = FactorArtifact(
            candidate_id=candidate.candidate_id,
            canonical_expression_json=canonical_json(candidate.expression, PRODUCTION_DSL_SPEC),
            expression_hash=expression_hash(candidate.expression, PRODUCTION_DSL_SPEC),
            direction=direction,
            train_rank_ic=train_raw,
            validation_rank_ic=score.signed_rank_ic,
            validation_neutral_rank_ic=score.neutral_rank_ic,
        )
    return payload, tuple(champions), artifacts


__all__ = [
    "exchange_identity",
    "freeze_track_a_development",
    "load_track_a_development_freeze",
    "sha256_file",
    "track_a_implementation_identity",
]
