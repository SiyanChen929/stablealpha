from __future__ import annotations

import json

import pytest

from cogalpha_plus.manifest import RunManifest, hash_json
from cogalpha_plus.registry import (
    FactorLineage,
    FactorRecord,
    FactorRegistry,
    LedgerIntegrityError,
    TrialLedger,
)


def _lineage(
    operation: str = "seed", parents: tuple[str, ...] = ()
) -> FactorLineage:
    return FactorLineage(
        parent_factor_ids=parents,
        operation=operation,
        operator="identity" if operation != "seed" else "",
        agent_id="price_behavior.reversal",
        generation=2,
    )


def _record(
    formula: dict,
    *,
    name: str = "candidate",
    status: str = "success",
    error: str = "",
    lineage: FactorLineage | None = None,
) -> FactorRecord:
    return FactorRecord.create(
        name=name,
        formula_ast=formula,
        status=status,
        lineage=lineage or _lineage(),
        run_id="run_test",
        data_version="cn_equity_snapshot_sha256:abc",
        config_version="cfg_sha256:def",
        model_version="codex:gpt-5",
        prompt_version="factor_prompt:v1",
        metrics={"rank_ic": 0.031} if status == "success" else {},
        error=error,
        created_at="2026-08-29T12:00:00.000000Z",
    )


def test_factor_record_is_content_addressed_and_canonical() -> None:
    left = _record(
        {
            "op": "sub",
            "args": [{"field": "close"}, {"field": "open"}],
            "params": {"safe": True, "epsilon": 1e-12},
        }
    )
    right = _record(
        {
            "params": {"epsilon": 1e-12, "safe": True},
            "args": [{"field": "close"}, {"field": "open"}],
            "op": "sub",
        },
        name="same_expression_from_another_agent",
    )

    assert left.formula_ast_hash == right.formula_ast_hash
    assert left.factor_id == right.factor_id
    assert left.record_hash != right.record_hash  # provenance/name remain auditable
    with pytest.raises(TypeError):
        left.formula_ast["op"] = "add"


def test_lineage_enforces_evolution_parent_semantics() -> None:
    with pytest.raises(ValueError, match="mutation"):
        _lineage("mutation")
    with pytest.raises(ValueError, match="crossover"):
        _lineage("crossover", ("fac_parent_1",))

    lineage = _lineage("crossover", ("fac_parent_1", "fac_parent_2"))
    assert lineage.operator == "identity"
    assert lineage.generation == 2


def test_registry_keeps_success_and_failure_and_deduplicates(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factor_library")
    success = _record({"op": "returns", "args": [{"field": "close"}], "window": 5})
    failed = _record(
        {"op": "div", "args": [{"field": "amount"}, {"const": 0}]},
        name="unsafe_division",
        status="failed",
        error="static_gate: denominator is constant zero",
    )

    assert registry.register(success).inserted
    assert registry.register(failed).inserted
    duplicate = _record(
        {"window": 5, "args": [{"field": "close"}], "op": "returns"},
        name="duplicate_alias",
    )
    result = registry.register(duplicate)
    assert not result.inserted
    assert result.duplicate_of == success.factor_id

    reopened = FactorRegistry(tmp_path / "factor_library")
    assert reopened.stats() == {"total": 2, "success": 1, "failed": 1}
    assert reopened.get(success.factor_id) == success
    assert reopened.contains_formula(
        success.formula_ast_hash,
        data_version=success.data_version,
        config_version=success.config_version,
    )
    assert reopened.records(status="failed")[0].error.startswith("static_gate")


def test_trial_ledger_verifies_chain_and_detects_tampering(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    ledger = TrialLedger(path)
    first = ledger.append(
        "candidate_generated",
        {"agent_id": "price_behavior.reversal", "generation": 0},
        timestamp="2026-08-29T12:00:00.000000Z",
    )
    second = ledger.append(
        "candidate_evaluated",
        {"factor_id": "fac_123", "rank_ic": 0.03},
        timestamp="2026-08-29T12:01:00.000000Z",
    )

    verified = ledger.verify(expected_entries=2, expected_head=second.entry_hash)
    assert verified
    assert first.prev_hash == "0" * 64
    assert second.prev_hash == first.entry_hash

    # Simulate an out-of-band edit to the first line without recomputing hashes.
    lines = path.read_text(encoding="utf-8").splitlines()
    first_payload = json.loads(lines[0])
    first_payload["payload"]["generation"] = 99
    lines[0] = json.dumps(first_payload, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    damaged = ledger.verify()
    assert not damaged
    assert damaged.bad_sequence == 0
    assert "hash mismatch" in damaged.error
    with pytest.raises(LedgerIntegrityError):
        ledger.append("should_be_refused", {})


def test_manifest_freezes_reproduction_inputs_and_checks_hash(tmp_path) -> None:
    config = {"mode": "faithful", "universe": "CSI300", "label_horizon": 5}
    budget = {
        "initial_pool": 80,
        "parents": 32,
        "children_per_generation": 96,
        "generations": 24,
        "subcycles": 3,
        "generations_per_subcycle": 8,
        "inject_every": 2,
        "elites": 2,
        "final_factors": 20,
    }
    manifest = RunManifest.create(
        seed=20260829,
        data_snapshot={
            "vendor": "test",
            "as_of": "2026-08-28",
            "sha256": "abc123",
        },
        search_budget=budget,
        config=config,
        model_version="codex:gpt-5",
        prompt_version="cogalpha-faithful:v1",
        run_id="run_reproducible_test",
        created_at="2026-08-29T12:00:00.000000Z",
    )
    assert manifest.config_hash == hash_json(config)
    assert manifest.search_budget["initial_pool"] == 80
    with pytest.raises(TypeError):
        manifest.search_budget["initial_pool"] = 1

    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = RunManifest.load(path)
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.manifest_hash == manifest.manifest_hash
    with pytest.raises(FileExistsError):
        manifest.save(path)

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["manifest"]["seed"] = 7
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        RunManifest.load(path)
