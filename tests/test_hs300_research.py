import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import cogalpha_plus.hs300_research as hs300_research
from cogalpha_plus.hs300_research import (
    HS300ResearchConfig,
    _claim_replay_gate,
    _load_reveal_manifest,
    _split_payload,
    load_codex_injection_candidates,
    load_codex_seed_candidates,
    load_frozen_hs300_panel,
    make_historical_replay_split,
)
from cogalpha_plus.dsl import canonical_json, expression_hash
from cogalpha_plus.manifest import RunManifest, hash_json
from cogalpha_plus.runner import ExperimentRecorder, panel_snapshot_hash
from cogalpha_plus.research_proxy import (
    DevelopmentFeatureHandoff,
    dump_frozen_composite_family,
    fit_frozen_development_composite_family,
)
from cogalpha_plus.synthetic import make_synthetic_market
from cogalpha_plus.validation import PRODUCTION_DSL_SPEC, sealed_reporting_contract_v1


PROJECT = Path(__file__).resolve().parents[1]


def test_research_implementation_inventory_freezes_input_lineage_and_composites():
    required = {
        "src/cogalpha_plus/data_sources/hs300_download.py",
        "src/cogalpha_plus/ensemble.py",
        "src/cogalpha_plus/research_proxy.py",
    }
    assert required.issubset(set(hs300_research.RESEARCH_IMPLEMENTATION_FILES))
    assert "lightgbm" in hs300_research.RESEARCH_RUNTIME_DISTRIBUTIONS


def test_run_manifest_identity_comparison_uses_serialized_json_domain(tmp_path):
    """Nested JSON arrays must survive RunManifest's immutable tuple view."""

    source_lineage = {
        "schema_version": 1,
        "components": [
            {"name": "membership", "hashes": ["a" * 64, "b" * 64]}
        ],
    }
    source_snapshot = {
        "schema_version": 1,
        "source_lineage_identity": source_lineage,
        "partitions": ["2018", "2019"],
    }
    manifest = RunManifest.create(
        seed=17,
        data_snapshot={
            "source_lineage_identity": source_lineage,
            "source_snapshot_identity": source_snapshot,
        },
        search_budget={"initial_population": 1},
        config={"profile": "test"},
    )
    path = manifest.save(tmp_path / "manifest.json")
    loaded = RunManifest.load(path)

    assert hs300_research._run_manifest_snapshot_identity(
        loaded, "source_lineage_identity"
    ) == source_lineage
    assert hs300_research._run_manifest_snapshot_identity(
        loaded, "source_snapshot_identity"
    ) == source_snapshot


def test_profile_reporting_contract_is_exact_and_fail_closed(tmp_path):
    expected = sealed_reporting_contract_v1()
    for profile in ("faithful", "plus"):
        raw = yaml.safe_load(
            (PROJECT / "configs" / f"{profile}.yaml").read_text(encoding="utf-8")
        )
        assert raw["ensemble"]["reporting_contract"] == expected
        hs300_research._load_composite_policies(PROJECT, profile)

    raw = yaml.safe_load(
        (PROJECT / "configs" / "plus.yaml").read_text(encoding="utf-8")
    )
    raw["ensemble"]["reporting_contract"][
        "primary_reassignment_allowed"
    ] = True
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "plus.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="reporting contract is not exact"):
        hs300_research._load_composite_policies(tmp_path, "plus")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(kind: str) -> dict[str, object]:
    payload: dict[str, object] = {"schema_version": 1, "kind": kind}
    return {**payload, "identity_hash": hash_json(payload)}


def _fixture_source_lineage_identity(
    normalized_inputs: dict[str, str] | None = None,
) -> dict[str, object]:
    hashes = {
        key: hash_json({"fixture_normalized_input": key})
        for names in hs300_research._SOURCE_LINEAGE_NORMALIZED_KEYS.values()
        for key in names
    }
    hashes.update(normalized_inputs or {})
    domains = {}
    for name, names in hs300_research._SOURCE_LINEAGE_NORMALIZED_KEYS.items():
        policy = {"schema_version": "fixture-1.0", "domain": name}
        identity = {
            "policy": policy,
            "cache_objects_sha256": hash_json([]),
            "normalized_input_sha256": {key: hashes[key] for key in names},
        }
        domains[name] = {**identity, "identity_sha256": hash_json(identity)}
    section = {
        "schema_version": "1.0",
        "complete": True,
        "domains": domains,
        "root_sha256": hash_json(domains),
    }
    return {**section, "identity_hash": hash_json(section)}


def _add_source_lineage_contract(manifest: dict[str, object]) -> dict[str, object]:
    normalized = manifest.setdefault("normalized_input_sha256", {})
    assert isinstance(normalized, dict)
    identity = _fixture_source_lineage_identity(dict(normalized))
    domains = identity["domains"]
    for name, names in hs300_research._SOURCE_LINEAGE_NORMALIZED_KEYS.items():
        domain = domains[name]
        manifest[f"{name}_cache_objects"] = []
        manifest[f"{name}_lineage"] = dict(domain["policy"])
        normalized.update(domain["normalized_input_sha256"])
    manifest["source_lineage"] = {
        key: value for key, value in identity.items() if key != "identity_hash"
    }
    tape = manifest.setdefault("feature_tape", {})
    assert isinstance(tape, dict)
    source_contract = tape.setdefault("source_contract", {})
    assert isinstance(source_contract, dict)
    source_contract["source_lineage_root_sha256"] = identity["root_sha256"]
    return identity


def _add_selection_contract(
    root: Path,
    manifest: dict[str, object],
    panel: pd.DataFrame,
    tape: pd.DataFrame,
) -> None:
    dates = pd.DatetimeIndex(
        panel.index.get_level_values("date").unique()
    ).sort_values()
    normalized = root.parent / "normalized"
    normalized.mkdir(exist_ok=True)
    calendar_path = normalized / "trade_calendar.parquet"
    pd.DataFrame(
        {"calendar_date": dates, "is_trading_day": "1"}
    ).to_parquet(calendar_path, index=False)
    columns = [
        "vendor_universe_member",
        "universe_member",
        "research_eligible",
    ]
    eligibility_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            tape.loc[:, columns], index=True, categorize=True
        ).to_numpy(dtype=np.uint64).tobytes()
    ).hexdigest()
    manifest["first_date"] = str(dates[0].date())
    manifest["last_date"] = str(dates[-1].date())
    manifest["storage_snapshot_id"] = "fixture"
    manifest["normalized_input_sha256"] = {"trade_calendar": _sha(calendar_path)}
    manifest["research_eligibility"] = {
        "complete": True,
        "selection_column": "research_eligible",
        "current_time_only": True,
        "future_endpoint_availability_filter": False,
        "snapshot_hash": eligibility_hash,
    }
    terminal_path = (
        root
        / "snapshots"
        / "fixture"
        / "terminal_stale_membership_exceptions.parquet"
    )
    terminal_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([], name="date"), pd.Index([], name="symbol")],
        names=["date", "symbol"],
    )
    terminal_columns = list(hs300_research._TERMINAL_SIDECAR_COLUMNS)
    terminal = pd.DataFrame(columns=terminal_columns, index=terminal_index)
    terminal.to_parquet(terminal_path)
    terminal_snapshot_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            terminal, index=True, categorize=True
        ).to_numpy(dtype=np.uint64).tobytes()
    ).hexdigest()
    terminal_payload = {
        "schema_version": "1.0",
        "complete": True,
        "policy": "terminal_stale_membership_after_verified_out_date",
        "price_fill_policy": "none",
        "terminal_return_policy": "unresolved_no_imputation",
        "rows": 0,
        "symbols": 0,
        "trading_days": 0,
        "snapshot_hash": terminal_snapshot_hash,
        "file": {
            "path": str(terminal_path),
            "rows": 0,
            "sha256": _sha(terminal_path),
            "bytes": terminal_path.stat().st_size,
        },
    }
    manifest["terminal_stale_membership_exceptions"] = terminal_payload
    _add_source_lineage_contract(manifest)


def _dump_composite_fixture(
    run_dir: Path,
    *,
    development_root_hash: str,
    champion_family_hash: str,
    split_identity_hash: str,
    champion_ids: tuple[str, ...],
    expression_hashes: tuple[str, ...],
    handoff_policy,
    composite_policy,
    research_universe_identity,
    trading_calendar_identity,
    source_lineage_identity,
    source_snapshot_identity,
):
    """Build a real safe-format family without loading any sealed fixture rows."""

    dates = pd.bdate_range("2018-01-02", periods=70, name="date")
    symbols = [f"S{i:02d}" for i in range(30)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    rng = np.random.default_rng(41)
    features = pd.DataFrame(
        rng.normal(size=(len(index), len(champion_ids))),
        index=index,
        columns=champion_ids,
    )
    labels = pd.Series(
        0.4 * features.iloc[:, 0]
        - 0.2 * features.iloc[:, 1]
        + rng.normal(scale=0.7, size=len(index)),
        index=index,
        name="label",
    )
    train_mask = index.get_level_values("date").isin(dates[:50])
    train_features = features.loc[train_mask]
    validation_features = features.loc[~train_mask]
    train_labels = labels.reindex(train_features.index)
    validation_labels = labels.reindex(validation_features.index)
    audit_index = pd.MultiIndex.from_product(
        [dates, champion_ids], names=["date", "factor"]
    )
    audit = pd.DataFrame(
        {
            "valid_names": len(symbols),
            "total_names": len(symbols),
            "coverage": 1.0,
            "zero_impute_share": 0.0,
            "constant_or_all_missing": False,
            "passes_gate": True,
        },
        index=audit_index,
    )
    provenance = {
        "development_root_hash": development_root_hash,
        "champion_family_hash": champion_family_hash,
        "split_identity_hash": split_identity_hash,
        "development_member_panel_hash": "e" * 64,
        "development_feature_tape_hash": "f" * 64,
        "research_universe_identity": dict(research_universe_identity),
        "research_universe_identity_hash": research_universe_identity[
            "identity_hash"
        ],
        "trading_calendar_identity": dict(trading_calendar_identity),
        "trading_calendar_identity_hash": trading_calendar_identity[
            "identity_hash"
        ],
        "source_lineage_identity": dict(source_lineage_identity),
        "source_lineage_identity_hash": source_lineage_identity[
            "identity_hash"
        ],
        "source_snapshot_identity": dict(source_snapshot_identity),
        "source_snapshot_identity_hash": source_snapshot_identity[
            "identity_hash"
        ],
        "train_features_hash": panel_snapshot_hash(train_features),
        "validation_features_hash": panel_snapshot_hash(validation_features),
        "train_labels_hash": panel_snapshot_hash(train_labels.to_frame()),
        "validation_labels_hash": panel_snapshot_hash(validation_labels.to_frame()),
        "expression_hashes": list(expression_hashes),
        "directions": {name: 1 for name in champion_ids},
        "development_handoff_policy": handoff_policy.to_dict(),
        "development_handoff_policy_hash": handoff_policy.policy_hash,
        "development_label_audit": {
            "schema_version": 1,
            "policy_hash": handoff_policy.policy_hash,
            "minimum_daily_label_coverage": 1.0,
            "allow_label_row_drop": False,
            "daily_audit_hash": "7" * 64,
            "valid_mask_hash": "8" * 64,
            "retained_row_hash": "9" * 64,
            "train": {"eligible_rows": len(train_labels), "valid_rows": len(train_labels)},
            "inner_validation": {"eligible_rows": len(validation_labels), "valid_rows": len(validation_labels)},
            "all_development": {
                "eligible_rows": len(labels),
                "valid_rows": len(labels),
            },
            "all_passed": True,
        },
        "neutralization": {"method": "unit_test_same_date"},
        "neutral_feature_audit": {"coverage_audit_hash": panel_snapshot_hash(audit)},
        "sealed_rows_loaded": 0,
        "sealed_results_consumed": False,
    }
    handoff = DevelopmentFeatureHandoff(
        train_features,
        validation_features,
        train_labels,
        validation_labels,
        audit,
        champion_ids,
        provenance,
    )
    family = fit_frozen_development_composite_family(
        handoff,
        policy=composite_policy,
        expected_policy_hash=composite_policy.policy_hash,
    )
    manifest = dump_frozen_composite_family(family, run_dir / "composite")
    return family, manifest


def _frozen_fixture(tmp_path: Path):
    root = tmp_path / "panel"
    snapshot = root / "snapshots" / "fixture" / "year=2020"
    snapshot.mkdir(parents=True)
    original = make_synthetic_market(days=80, symbols=24, seed=91)
    partition = snapshot / "part-000.parquet"
    original.to_parquet(partition, index=True)
    restored = pd.read_parquet(partition).sort_index()
    tape = restored.copy()
    tape["universe_member"] = True
    tape_partition = root / "snapshots" / "fixture" / "feature_tape" / "year=2020" / "part-000.parquet"
    tape_partition.parent.mkdir(parents=True)
    tape.to_parquet(tape_partition, index=True)
    tape = pd.read_parquet(tape_partition).sort_index()
    manifest = {
        "complete": True,
        "research_proxy_only": True,
        "rows": len(restored),
        "snapshot_hash": panel_snapshot_hash(restored),
        "partitions": [
            {
                "path": str(partition),
                "rows": len(restored),
                "sha256": _sha(partition),
            }
        ],
        "feature_tape": {
            "rows": len(tape),
            "snapshot_hash": panel_snapshot_hash(tape),
            "partitions": [
                {
                    "path": str(tape_partition),
                    "rows": len(tape),
                    "sha256": _sha(tape_partition),
                }
            ],
        },
    }
    _add_selection_contract(root, manifest, restored, tape)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest.sha256").write_text(
        f"{_sha(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    return root, restored, partition


def test_real_seed_batch_is_exactly_80_and_covers_21_roles():
    candidates = load_codex_seed_candidates(PROJECT, horizon=10)
    assert len(candidates) == 80
    assert len({item.candidate_id for item in candidates}) == 80
    assert len({item.expression_hash for item in candidates}) == 80
    assert len({item.agent_id for item in candidates}) == 21
    assert {item.metadata["forecast_horizon"] for item in candidates} == {10}


def test_precommitted_injection_library_is_96_unique_role_complete_repairs():
    seeds = load_codex_seed_candidates(PROJECT, horizon=10)
    candidates = load_codex_injection_candidates(PROJECT, horizon=10)
    repairs = {
        "repair_shorten_lookback",
        "repair_persistence_smoothing",
        "repair_stability_smoothing",
        "repair_style_robustification",
        "repair_conservative_repair",
        "repair_novelty_recombination",
        "repair_nonlinear_refinement",
        "repair_parameter_refinement",
    }
    roles = {item.agent_id for item in seeds}
    assert len(candidates) == 96
    assert len({item.candidate_id for item in candidates}) == 96
    assert len({item.expression_hash for item in candidates}) == 96
    assert {item.agent_id for item in candidates} == roles
    assert {item.metadata["forecast_horizon"] for item in candidates} == {10}
    assert not {item.expression_hash for item in seeds}.intersection(
        item.expression_hash for item in candidates
    )
    for role in roles:
        role_cards = [item for item in candidates if item.agent_id == role]
        assert len(role_cards) >= 4
        assert repairs.issubset(
            {
                tag
                for item in role_cards
                for tag in item.mechanism_tags
            }
        )
        assert all(
            item.metadata["source"]
            == "codex_precommitted_adaptive_injection_card"
            for item in role_cards
        )


def test_injection_loader_fails_closed_on_missing_or_unknown_repair_contract(
    monkeypatch,
):
    with (PROJECT / "configs" / "agents.yaml").open(encoding="utf-8") as handle:
        allowed = {item["id"] for item in yaml.safe_load(handle)["agents"]}
    paths = [
        PROJECT / "proposals" / "codex_injection" / name
        for name in ("l1_l4.jsonl", "l5_l7.jsonl")
    ]
    cards = hs300_research.load_factor_cards(
        paths,
        allowed_agent_ids=allowed,
        expected_horizon=10,
        expected_count=96,
    )
    target_role = "L1_market_cycle"
    incomplete = [
        replace(
            card,
            mechanism_tags=tuple(
                tag
                for tag in card.mechanism_tags
                if not (
                    card.agent_id == target_role
                    and tag == "repair_parameter_refinement"
                )
            ),
        )
        for card in cards
    ]
    monkeypatch.setattr(
        hs300_research,
        "load_factor_cards",
        lambda *args, **kwargs: incomplete,
    )
    monkeypatch.setattr(
        hs300_research,
        "load_codex_seed_candidates",
        lambda *args, **kwargs: [],
    )
    with pytest.raises(ValueError, match="every adaptive repair class"):
        load_codex_injection_candidates(PROJECT, horizon=10)

    unknown = list(cards)
    unknown[0] = replace(
        unknown[0],
        mechanism_tags=unknown[0].mechanism_tags + ("repair_future_leak",),
    )
    monkeypatch.setattr(
        hs300_research,
        "load_factor_cards",
        lambda *args, **kwargs: unknown,
    )
    with pytest.raises(ValueError, match="unknown repair tags"):
        load_codex_injection_candidates(PROJECT, horizon=10)

    target_cards = [card for card in cards if card.agent_id == target_role]
    retained_key = target_cards[0].candidate_key
    underfilled = []
    for card in cards:
        if card.agent_id == target_role and card.candidate_key != retained_key:
            underfilled.append(replace(card, agent_id="L1_volatility_regime"))
        elif card.candidate_key == retained_key:
            underfilled.append(
                replace(
                    card,
                    mechanism_tags=tuple(
                        sorted(set(card.mechanism_tags).union(
                            hs300_research.ADAPTIVE_REPAIR_TAGS
                        ))
                    ),
                )
            )
        else:
            underfilled.append(card)
    monkeypatch.setattr(
        hs300_research,
        "load_factor_cards",
        lambda *args, **kwargs: underfilled,
    )
    with pytest.raises(ValueError, match="at least four distinct"):
        load_codex_injection_candidates(PROJECT, horizon=10)


def test_plus_yaml_thresholds_resolve_exactly_and_change_config_hash(tmp_path):
    original = yaml.safe_load(
        (PROJECT / "configs" / "plus.yaml").read_text(encoding="utf-8")
    )
    baseline = hs300_research._load_adaptive_feedback_policy(PROJECT, "plus")
    declared = original["agent_hierarchy"]["adaptive_feedback"]["fixed_thresholds"]
    assert hs300_research._adaptive_feedback_policy_payload(baseline) == {
        name: float(value) for name, value in declared.items()
    }
    assert hs300_research._load_adaptive_feedback_policy(PROJECT, "faithful") is None

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    changed_yaml = json.loads(json.dumps(original))
    changed_yaml["agent_hierarchy"]["adaptive_feedback"]["fixed_thresholds"][
        "minimum_coverage"
    ] = 0.81
    (config_dir / "plus.yaml").write_text(
        yaml.safe_dump(changed_yaml, sort_keys=False),
        encoding="utf-8",
    )
    changed = hs300_research._load_adaptive_feedback_policy(tmp_path, "plus")
    assert changed.minimum_coverage == 0.81
    assert changed != baseline

    panel = make_synthetic_market(days=220, symbols=20, seed=2)
    split = hs300_research.make_historical_replay_split(
        panel,
        train_end=str(panel.index.get_level_values("date").unique()[70].date()),
        validation_end=str(panel.index.get_level_values("date").unique()[140].date()),
        replay_end=str(panel.index.get_level_values("date").max().date()),
        label_horizon=2,
        entry_delay=1,
    )
    common = {
        "profile": "plus",
        "split": split,
        "budget": hs300_research.SearchBudget(),
        "bootstrap_samples": 200,
    }
    baseline_config = hs300_research.experiment_config_payload(
        **common,
        adaptive_feedback_policy=baseline,
    )
    changed_config = hs300_research.experiment_config_payload(
        **common,
        adaptive_feedback_policy=changed,
    )
    assert hash_json(baseline_config) != hash_json(changed_config)

    baseline_behavior = hs300_research._load_behavioral_champion_policy(
        PROJECT,
        "plus",
    )
    assert baseline_behavior.maximum_absolute_correlation == 0.85
    assert (
        hs300_research._load_behavioral_champion_policy(PROJECT, "faithful")
        is None
    )
    changed_behavior_yaml = json.loads(json.dumps(original))
    changed_behavior_yaml["deduplication"]["behavioral"][
        "maximum_absolute_correlation"
    ] = 0.80
    (config_dir / "plus.yaml").write_text(
        yaml.safe_dump(changed_behavior_yaml, sort_keys=False),
        encoding="utf-8",
    )
    changed_behavior = hs300_research._load_behavioral_champion_policy(
        tmp_path,
        "plus",
    )
    assert changed_behavior.maximum_absolute_correlation == 0.80
    baseline_behavior_config = hs300_research.experiment_config_payload(
        **common,
        adaptive_feedback_policy=baseline,
        behavioral_champion_policy=baseline_behavior,
    )
    changed_behavior_config = hs300_research.experiment_config_payload(
        **common,
        adaptive_feedback_policy=baseline,
        behavioral_champion_policy=changed_behavior,
    )
    assert hash_json(baseline_behavior_config) != hash_json(
        changed_behavior_config
    )


def test_calendar_split_has_two_full_nonoverlapping_purges():
    dates = pd.bdate_range("2011-01-03", "2025-12-31")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    panel = pd.DataFrame({"x": 1.0}, index=index)
    split = make_historical_replay_split(
        panel,
        train_end="2017-12-31",
        validation_end="2019-12-31",
        replay_end="2025-12-31",
        label_horizon=10,
        entry_delay=1,
    )
    assert len(split.purge_after_train) == 11
    assert len(split.purge_after_validation) == 11
    groups = (
        set(split.train_dates),
        set(split.purge_after_train),
        set(split.validation_dates),
        set(split.purge_after_validation),
        set(split.sealed_dates),
    )
    assert all(not left.intersection(right) for i, left in enumerate(groups) for right in groups[i + 1 :])
    assert max(split.train_dates) <= pd.Timestamp("2017-12-31")
    assert max(split.validation_dates) <= pd.Timestamp("2019-12-31")
    assert min(split.sealed_dates) > pd.Timestamp("2019-12-31")


def test_calendar_split_rejects_duplicate_or_out_of_order_direct_dates():
    dates = pd.bdate_range("2011-01-03", "2025-12-31")
    duplicate = dates.insert(3, dates[2])
    with pytest.raises(ValueError, match="strictly increasing"):
        hs300_research._make_historical_replay_split_from_dates(
            duplicate,
            train_end="2017-12-31",
            validation_end="2019-12-31",
            replay_end="2025-12-31",
        )
    out_of_order = dates.to_list()
    out_of_order[2], out_of_order[3] = out_of_order[3], out_of_order[2]
    with pytest.raises(ValueError, match="strictly increasing"):
        hs300_research._make_historical_replay_split_from_dates(
            out_of_order,
            train_end="2017-12-31",
            validation_end="2019-12-31",
            replay_end="2025-12-31",
        )


def test_frozen_loader_rechecks_manifest_partition_and_semantic_hash(tmp_path):
    root, expected, partition = _frozen_fixture(tmp_path)
    loaded = load_frozen_hs300_panel(root)
    pd.testing.assert_frame_equal(loaded.panel, expected)
    assert loaded.feature_tape["universe_member"].all()
    assert loaded.manifest_sha256 == _sha(root / "manifest.json")

    damaged = pd.read_parquet(partition)
    damaged.iloc[0, damaged.columns.get_loc("close")] *= 1.01
    damaged.to_parquet(partition, index=True)
    with pytest.raises(ValueError, match="partition SHA-256 mismatch"):
        load_frozen_hs300_panel(root)


def test_frozen_manifest_hash_and_payload_share_one_byte_snapshot(
    tmp_path, monkeypatch
):
    root, _, _ = _frozen_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    original_bytes = manifest_path.read_bytes()
    original_payload = json.loads(original_bytes)
    replacement_payload = {**original_payload, "storage_snapshot_id": "replacement"}
    replacement_bytes = json.dumps(replacement_payload).encode("utf-8")
    original_read_bytes = Path.read_bytes
    manifest_reads = 0

    def snapshot_then_replace(path):
        nonlocal manifest_reads
        snapshot = original_read_bytes(path)
        if path.resolve() == manifest_path.resolve():
            manifest_reads += 1
            if manifest_reads == 1:
                manifest_path.write_bytes(replacement_bytes)
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", snapshot_then_replace)

    loaded_root, loaded_payload, loaded_hash = (
        hs300_research._load_frozen_panel_manifest(root)
    )

    assert loaded_root == root.resolve()
    assert manifest_reads == 1
    assert loaded_payload == original_payload
    assert loaded_hash == hashlib.sha256(original_bytes).hexdigest()
    assert original_read_bytes(manifest_path) == replacement_bytes


def test_frozen_loader_rejects_partition_path_escape(tmp_path):
    root, _, partition = _frozen_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(partition.read_bytes())
    manifest["partitions"][0]["path"] = str(outside)
    manifest["partitions"][0]["sha256"] = _sha(outside)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest.sha256").write_text(
        f"{_sha(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="escapes panel/snapshots"):
        load_frozen_hs300_panel(root)


def test_frozen_loader_rejects_missing_or_byte_tampered_terminal_sidecar(tmp_path):
    missing_root, _, _ = _frozen_fixture(tmp_path / "missing")
    missing_manifest = json.loads(
        (missing_root / "manifest.json").read_text(encoding="utf-8")
    )
    missing_path = Path(
        missing_manifest["terminal_stale_membership_exceptions"]["file"]["path"]
    )
    missing_path.unlink()
    with pytest.raises(FileNotFoundError, match="sidecar is missing"):
        load_frozen_hs300_panel(missing_root)

    tampered_root, _, _ = _frozen_fixture(tmp_path / "tampered")
    tampered_manifest = json.loads(
        (tampered_root / "manifest.json").read_text(encoding="utf-8")
    )
    tampered_path = Path(
        tampered_manifest["terminal_stale_membership_exceptions"]["file"]["path"]
    )
    with tampered_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="byte count mismatch|SHA-256 mismatch"):
        load_frozen_hs300_panel(tampered_root)


def test_frozen_loader_rejects_coordinated_terminal_file_manifest_reseal(tmp_path):
    root, _, _ = _frozen_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terminal = manifest["terminal_stale_membership_exceptions"]
    terminal_path = Path(terminal["file"]["path"])
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2020-01-10"), "S0000")],
        names=["date", "symbol"],
    )
    forged = pd.DataFrame(
        [
            {
                "vendor_universe_member": True,
                "universe_member": True,
                "research_eligible": False,
                "reason": "terminal_stale_membership_after_verified_out_date",
                "stock_basic_out_date": pd.Timestamp("2020-01-09"),
                "last_raw_bar_date": pd.Timestamp("2020-01-09"),
                "membership_snapshot_date": pd.Timestamp("2020-01-01"),
                "security_status": "0",
                "price_fill_policy": "none",
                "terminal_return_policy": "unresolved_no_imputation",
            }
        ],
        index=index,
    ).loc[:, list(hs300_research._TERMINAL_SIDECAR_COLUMNS)]
    forged.to_parquet(terminal_path)
    terminal["rows"] = 1
    terminal["symbols"] = 1
    terminal["trading_days"] = 1
    terminal["file"].update(
        {
            "rows": 1,
            "bytes": terminal_path.stat().st_size,
            "sha256": _sha(terminal_path),
        }
    )
    # Coordinately reseal the outer manifest/file identity but deliberately
    # retain the previously committed semantic snapshot root.
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest.sha256").write_text(
        f"{_sha(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        load_frozen_hs300_panel(root)


def test_frozen_loader_rejects_terminal_sidecar_dimension_count_mismatch(tmp_path):
    root, _, _ = _frozen_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["terminal_stale_membership_exceptions"]["symbols"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest.sha256").write_text(
        f"{_sha(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="dimension count mismatch"):
        load_frozen_hs300_panel(root)


def test_frozen_loader_rejects_same_bytes_terminal_sidecar_relocation(tmp_path):
    root, _, _ = _frozen_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terminal_file = manifest["terminal_stale_membership_exceptions"]["file"]
    original = Path(terminal_file["path"])
    relocated = (
        root
        / "snapshots"
        / "relocated"
        / "terminal_stale_membership_exceptions.parquet"
    )
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(original.read_bytes())
    terminal_file["path"] = str(relocated)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest.sha256").write_text(
        f"{_sha(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fixed snapshot path"):
        load_frozen_hs300_panel(root)


def test_development_source_manifest_race_is_fail_closed():
    planned = {"complete": True, "nested": {"value": 1}}
    digest = "a" * 64
    hs300_research._require_stable_development_source_manifest(
        planned, digest, json.loads(json.dumps(planned)), digest
    )
    with pytest.raises(ValueError, match="changed between split planning"):
        hs300_research._require_stable_development_source_manifest(
            planned, digest, planned, "b" * 64
        )
    with pytest.raises(ValueError, match="changed between split planning"):
        hs300_research._require_stable_development_source_manifest(
            planned,
            digest,
            {"complete": True, "nested": {"value": 2}},
            digest,
        )


@pytest.mark.parametrize("domain_name", ["membership", "security", "industry"])
def test_frozen_loader_rejects_coordinately_resealed_source_lineage_domain(
    tmp_path, domain_name
):
    root, _, _ = _frozen_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lineage = manifest["source_lineage"]
    domain = lineage["domains"][domain_name]
    normalized_name = next(iter(domain["normalized_input_sha256"]))
    domain["normalized_input_sha256"][normalized_name] = "0" * 64
    identity = {
        "policy": domain["policy"],
        "cache_objects_sha256": domain["cache_objects_sha256"],
        "normalized_input_sha256": domain["normalized_input_sha256"],
    }
    domain["identity_sha256"] = hash_json(identity)
    lineage["root_sha256"] = hash_json(lineage["domains"])
    manifest["feature_tape"]["source_contract"][
        "source_lineage_root_sha256"
    ] = lineage["root_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest.sha256").write_text(
        f"{_sha(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=f"{domain_name} source-lineage domain"):
        load_frozen_hs300_panel(root)


def test_frozen_loader_rejects_coordinately_resealed_source_lineage_root(tmp_path):
    root, _, _ = _frozen_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_lineage"]["root_sha256"] = "0" * 64
    manifest["feature_tape"]["source_contract"][
        "source_lineage_root_sha256"
    ] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest.sha256").write_text(
        f"{_sha(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source-lineage root hash"):
        load_frozen_hs300_panel(root)


def _multiyear_frozen_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "panel"
    member_records = []
    tape_records = []
    member_frames = []
    tape_frames = []
    for year, dates in (
        (2019, pd.bdate_range("2019-12-16", "2019-12-31")),
        (2020, pd.bdate_range("2020-01-02", "2020-02-28")),
        (2021, pd.bdate_range("2021-01-04", "2021-01-15")),
    ):
        frame = make_synthetic_market(days=80, symbols=24, seed=year)
        source_dates = pd.DatetimeIndex(
            frame.index.get_level_values("date").unique()
        ).sort_values()[: len(dates)]
        frame = frame.loc[
            frame.index.get_level_values("date").isin(source_dates)
        ]
        mapping = dict(zip(source_dates, dates, strict=True))
        frame = frame.reset_index()
        frame["date"] = frame["date"].map(mapping)
        frame = frame.set_index(["date", "symbol"]).sort_index()
        member_path = (
            root / "snapshots" / "fixture" / f"year={year}" / "part-000.parquet"
        )
        member_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(member_path)
        member_frames.append(pd.read_parquet(member_path))
        member_records.append(
            {
                "year": year,
                "path": str(member_path),
                "rows": len(frame),
                "sha256": _sha(member_path),
            }
        )
        tape = frame.copy()
        tape["universe_member"] = True
        tape_path = (
            root
            / "snapshots"
            / "fixture"
            / "feature_tape"
            / f"year={year}"
            / "part-000.parquet"
        )
        tape_path.parent.mkdir(parents=True, exist_ok=True)
        tape.to_parquet(tape_path)
        tape_frames.append(pd.read_parquet(tape_path))
        tape_records.append(
            {
                "year": year,
                "path": str(tape_path),
                "rows": len(tape),
                "sha256": _sha(tape_path),
            }
        )
    panel = pd.concat(member_frames).sort_index()
    tape = pd.concat(tape_frames).sort_index()
    manifest = {
        "complete": True,
        "research_proxy_only": True,
        "rows": len(panel),
        "first_date": "2019-12-16",
        "last_date": "2021-01-15",
        "snapshot_hash": panel_snapshot_hash(panel),
        "partitions": member_records,
        "feature_tape": {
            "rows": len(tape),
            "snapshot_hash": panel_snapshot_hash(tape),
            "partitions": tape_records,
        },
    }
    _add_selection_contract(root, manifest, panel, tape)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest.sha256").write_text(
        f"{_sha(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    return root


def test_development_loader_never_materializes_sealed_rows_or_opens_future_year(
    tmp_path, monkeypatch
):
    root = _multiyear_frozen_fixture(tmp_path)
    original = pd.read_parquet
    calls = []

    def recording_read(path, *args, **kwargs):
        if "year=2021" in str(path):
            raise AssertionError("development process opened a wholly sealed partition")
        frame = original(path, *args, **kwargs)
        calls.append((str(path), kwargs.get("filters"), frame))
        return frame

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    cutoff = pd.Timestamp("2020-01-10")
    loaded = load_frozen_hs300_panel(root, end_date=cutoff)
    assert loaded.panel.index.get_level_values("date").max() == cutoff
    assert loaded.feature_tape.index.get_level_values("date").max() == cutoff
    assert all(
        frame.index.get_level_values("date").max() <= cutoff
        for _, _, frame in calls
        if isinstance(frame.index, pd.MultiIndex) and not frame.empty
    )
    boundary_calls = [item for item in calls if "year=2020" in item[0]]
    assert boundary_calls
    assert all(item[1] == [("date", "<=", cutoff)] for item in boundary_calls)


def _sealed_input_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    budget = {
        "initial_population": 80,
        "parent_pool": 32,
        "children_per_generation": 96,
        "generations": 24,
        "injection_every": 2,
        "injection_count": 8,
        "elite_carry": 2,
        "final_factors": 20,
    }
    dates = pd.bdate_range("2011-01-03", "2025-12-31")
    panel = pd.DataFrame(
        {"x": 1.0},
        index=pd.MultiIndex.from_product(
            [dates, ["A", "B"]], names=["date", "symbol"]
        ),
    )
    split = make_historical_replay_split(
        panel,
        train_end="2017-12-31",
        validation_end="2019-12-31",
        replay_end="2025-12-31",
    )
    evaluated_champions = []
    for position in range(20):
        expression = {
            "op": "cs_rank",
            "args": [
                {
                    "op": "neg",
                    "args": [
                        {
                            "op": "return",
                            "args": [{"op": "field", "name": "close"}],
                            "periods": position + 2,
                        }
                    ],
                }
            ],
        }
        expression_json = canonical_json(expression, PRODUCTION_DSL_SPEC)
        expression_ast = json.loads(expression_json)
        candidate = hs300_research.Candidate(
            expression=expression_ast,
            agent_id="agent",
            operator="seed",
            hypothesis="test",
            mechanism_tags=("reversal",),
            metadata={
                "expected_direction": 1,
                "forecast_horizon": 10,
                "role_lineage": ["agent"],
                "source": "fixture",
                "candidate_key": f"fixture-{position + 1}",
            },
        )
        candidate.candidate_id = hs300_research.deterministic_candidate_id(candidate)
        evaluated_champions.append(
            hs300_research.EvaluatedCandidate(
                candidate,
                hs300_research.CandidateScore(
                    fitness=0.01,
                    signed_rank_ic=0.02,
                    neutral_rank_ic=0.01,
                    turnover=0.2,
                    worst_window_ic=0.0,
                    style_retention=0.5,
                    novelty=1.0,
                    control_delta=None,
                    passed=True,
                    diagnostics={"direction": 1, "train_raw_rank_ic": 0.02},
                ),
            )
        )
    exact = _split_payload(split)
    source_member_hash = "b" * 64
    source_tape_hash = "c" * 64
    source_eligibility_hash = "4" * 64
    source_calendar_hash = "5" * 64
    source_terminal_identity = _identity("fixture_terminal_sidecar")
    source_lineage_identity = _fixture_source_lineage_identity()
    source_combined_hash = hashlib.sha256(
        (
            f"{source_member_hash}|{source_tape_hash}|"
            f"{source_eligibility_hash}|"
            f"{source_terminal_identity['identity_hash']}|{source_calendar_hash}|"
            f"{source_lineage_identity['identity_hash']}"
        ).encode("utf-8")
    ).hexdigest()
    source_contract = {
        "fixture": "sealed_source_contract",
        "source_lineage_root_sha256": source_lineage_identity["root_sha256"],
        "trade_calendar_sha256": source_calendar_hash,
    }
    source_eligibility_payload = {
        "schema_version": "fixture-1.0",
        "complete": True,
        "snapshot_hash": source_eligibility_hash,
    }
    source_eligibility_identity = {
        **source_eligibility_payload,
        "identity_hash": hash_json(source_eligibility_payload),
    }
    source_snapshot_payload = {
        "panel_manifest_sha256": "a" * 64,
        "storage_snapshot_id": "fixture-sealed-source",
        "member_panel_snapshot_hash": source_member_hash,
        "member_partition_records_hash": hash_json([]),
        "feature_tape_snapshot_hash": source_tape_hash,
        "feature_tape_partition_records_hash": hash_json([]),
        "full_source_contract": source_contract,
        "full_source_contract_hash": hash_json(source_contract),
        "research_eligibility_snapshot_hash": source_eligibility_hash,
        "research_eligibility_identity": source_eligibility_identity,
        "source_lineage_root_sha256": source_lineage_identity[
            "root_sha256"
        ],
        "source_lineage_identity": source_lineage_identity,
        "terminal_sidecar_identity": source_terminal_identity,
        "frozen_trade_calendar_sha256": source_calendar_hash,
        "research_rule_identity": hs300_research.research_rule_identity_v1(),
        "combined_data_snapshot_hash": source_combined_hash,
        "first_date": "2011-01-03",
        "last_date": "2025-12-31",
    }
    source_snapshot_identity = {
        **source_snapshot_payload,
        "identity_hash": hash_json(source_snapshot_payload),
    }
    implementation_content = {
        "files": [
            {
                "path": name,
                "module": hs300_research._research_module_name(name),
                "bytes": 1,
                "sha256": "1" * 64,
            }
            for name in hs300_research.RESEARCH_IMPLEMENTATION_FILES
        ],
        "runtime": {
            "python_implementation": "CPython",
            "python_version": "3.12.0",
            "python_executable_name": "python3.12",
            "platform_machine": "arm64",
            **{
                f"distribution:{name}": "fixture"
                for name in hs300_research.RESEARCH_RUNTIME_DISTRIBUTIONS
            },
        },
    }
    implementation_identity = {
        **implementation_content,
        "aggregate_sha256": hash_json(implementation_content),
    }
    handoff_policy, composite_policy = hs300_research._load_composite_policies(
        PROJECT, "plus"
    )
    operational_locators = {
        "project_dir": str(PROJECT.resolve()),
        "panel_dir": str((tmp_path / "panel").resolve()),
        "reveal_registry_dir": str(
            (PROJECT / "runs" / "replay_reveal_registry").resolve()
        ),
        "sealed_output_dir": str((run_dir / "sealed_reveal").resolve()),
    }
    research_universe_payload = {
        "schema_version": 1,
        "kind": "fixture_research_universe",
        "terminal_sidecar_identity": source_terminal_identity,
    }
    research_universe_identity = {
        **research_universe_payload,
        "identity_hash": hash_json(research_universe_payload),
    }
    trading_calendar_payload = {
        "schema_version": 1,
        "kind": "fixture_trading_calendar",
        "normalized_trade_calendar_sha256": source_calendar_hash,
    }
    trading_calendar_identity = {
        **trading_calendar_payload,
        "identity_hash": hash_json(trading_calendar_payload),
    }
    resolved_experiment_config = hs300_research.experiment_config_payload(
        profile="plus",
        split=split,
        budget=hs300_research.SearchBudget(),
        bootstrap_samples=200,
        development_handoff_policy=handoff_policy.to_dict(),
        composite_family_policy=composite_policy.to_dict(),
        operational_locators=operational_locators,
        implementation_identity=implementation_identity,
        research_universe_identity=research_universe_identity,
        trading_calendar_identity=trading_calendar_identity,
        source_lineage_identity=source_lineage_identity,
        source_snapshot_identity=source_snapshot_identity,
    )
    run_manifest = RunManifest.create(
        seed=17,
        data_snapshot={
            "sha256": "d" * 64,
            "source_lineage_identity": source_lineage_identity,
            "source_snapshot_identity": source_snapshot_identity,
        },
        search_budget=budget,
        config_hash=hash_json(resolved_experiment_config),
        model_version="test",
        prompt_version="test",
    )
    recorder = ExperimentRecorder(
        run_dir,
        manifest=run_manifest,
        data_version="d" * 64,
        config_version=run_manifest.config_hash,
        model_version="test",
        prompt_version="test",
    )
    run_path = run_dir / "manifest.json"
    for evaluated in evaluated_champions:
        recorder.record(evaluated)
    evolution = hs300_research.EvolutionResult(
        champions=tuple(evaluated_champions),
        archive=tuple(evaluated_champions),
        generations=(),
    )
    recorder.finish(evolution)
    champions = hs300_research._champions_payload(evaluated_champions)
    champion_family_hash = hash_json(champions)
    split_identity_hash = hash_json(exact)
    family, composite_manifest = _dump_composite_fixture(
        run_dir,
        development_root_hash=run_manifest.manifest_hash,
        champion_family_hash=champion_family_hash,
        split_identity_hash=split_identity_hash,
        champion_ids=tuple(str(item["candidate_id"]) for item in champions),
        expression_hashes=tuple(str(item["expression_hash"]) for item in champions),
        handoff_policy=handoff_policy,
        composite_policy=composite_policy,
        research_universe_identity=research_universe_identity,
        trading_calendar_identity=trading_calendar_identity,
        source_lineage_identity=source_lineage_identity,
        source_snapshot_identity=source_snapshot_identity,
    )
    frozen_label_audit = family.family_commitment["development_label_audit"]
    pre_search_label_audit = {
        "schema_version": 1,
        "policy": handoff_policy.to_dict(),
        "valid_mask_hash": "4" * 64,
        "daily_audit_hash": "5" * 64,
        "train": dict(frozen_label_audit["train"]),
        "inner_validation": dict(frozen_label_audit["inner_validation"]),
        "all_development": dict(frozen_label_audit["all_development"]),
        "all_passed": True,
    }
    summary_path = run_dir / "summary.json"
    recorded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recorded_summary["development_label_audit"] = pre_search_label_audit
    summary_path.write_text(json.dumps(recorded_summary), encoding="utf-8")
    composite_manifest_sha256 = _sha(composite_manifest)
    completion_evidence = hs300_research._append_development_completion_evidence(
        run_dir=run_dir,
        run_manifest=run_manifest,
        champions=champions,
        champion_family_hash=champion_family_hash,
        composite_family_hash=family.composite_family_hash,
        composite_manifest_sha256=composite_manifest_sha256,
    )
    payload = {
        "schema_version": hs300_research.SEALED_REVEAL_SCHEMA_VERSION,
        "kind": "hs300-frozen-development-to-sealed-reveal",
        "status": "frozen_before_reveal",
        "development_run_manifest_filename": "manifest.json",
        "development_run_manifest_sha256": _sha(run_path),
        "development_run_manifest_hash": run_manifest.manifest_hash,
        "development_completion_evidence": completion_evidence,
        "composite_manifest_locator": "composite/family_manifest.json",
        "composite_manifest_sha256": composite_manifest_sha256,
        "composite_family_hash": family.composite_family_hash,
        "composite_primary_backend": "lightgbm",
        "sealed_reporting_contract": sealed_reporting_contract_v1(),
        "sealed_reporting_contract_hash": hash_json(
            sealed_reporting_contract_v1()
        ),
        "development_label_audit": dict(
            family.family_commitment["development_label_audit"]
        ),
        "pre_search_development_label_audit": pre_search_label_audit,
        "source_snapshot_identity": source_snapshot_identity,
        "research_implementation_identity": implementation_identity,
        "resolved_experiment_config": resolved_experiment_config,
        "development_view_identity": {
            "hard_end_date": str(split.development_end.date()),
            "member_panel_snapshot_hash": "e" * 64,
            "eligible_member_panel_snapshot_hash": "e" * 64,
            "full_member_panel_snapshot_hash": "6" * 64,
            "feature_tape_snapshot_hash": "f" * 64,
            "research_universe_identity": research_universe_identity,
            "research_universe_identity_hash": research_universe_identity[
                "identity_hash"
            ],
            "trading_calendar_identity": trading_calendar_identity,
            "trading_calendar_identity_hash": trading_calendar_identity[
                "identity_hash"
            ],
            "source_lineage_identity": source_lineage_identity,
            "source_lineage_identity_hash": source_lineage_identity[
                "identity_hash"
            ],
            "source_snapshot_identity": source_snapshot_identity,
            "source_snapshot_identity_hash": source_snapshot_identity[
                "identity_hash"
            ],
            "sealed_market_rows_loaded": 0,
        },
        "split_cut_spec": {
            "train_end": "2017-12-31",
            "validation_end": "2019-12-31",
            "replay_end": "2025-12-31",
            "label_horizon": 10,
            "entry_delay": 1,
        },
        "split_exact": exact,
        "split_identity_hash": split_identity_hash,
        "champions": champions,
        "champion_family_hash": champion_family_hash,
        "search_budget": budget,
        "expected_members": 300,
        "operational_locators": operational_locators,
    }
    envelope = {"manifest": payload, "manifest_hash": hash_json(payload)}
    target = run_dir / "sealed_reveal_manifest.json"
    target.write_text(json.dumps(envelope), encoding="utf-8")
    return target


def test_frozen_reveal_manifest_rejects_ast_tampering_even_if_attacker_reseals(
    tmp_path,
):
    target = _sealed_input_fixture(tmp_path)
    _, payload, _ = _load_reveal_manifest(target)
    assert payload["development_view_identity"]["sealed_market_rows_loaded"] == 0

    envelope = json.loads(target.read_text(encoding="utf-8"))
    envelope["manifest"]["champions"][0]["canonical_expression_ast"] = {
        "op": "field",
        "name": "volume",
    }
    target.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        _load_reveal_manifest(target)

    payload = envelope["manifest"]
    payload["champion_family_hash"] = hash_json(payload["champions"])
    envelope["manifest_hash"] = hash_json(payload)
    target.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical AST/hash"):
        _load_reveal_manifest(target)


@pytest.mark.parametrize(
    "attack_kind", ["lineage", "terminal", "calendar", "sealed_source"]
)
def test_reveal_preclaim_rejects_coordinated_full_source_swap_without_registry(
    tmp_path, attack_kind
):
    case_dir = tmp_path / attack_kind
    case_dir.mkdir()
    target = _sealed_input_fixture(case_dir)
    registry = case_dir / "registry"
    registry.mkdir()
    envelope = json.loads(target.read_text(encoding="utf-8"))
    source = envelope["manifest"]["source_snapshot_identity"]

    if attack_kind == "lineage":
        alternate = _fixture_source_lineage_identity(
            {"membership_daily": "9" * 64}
        )
        source["source_lineage_identity"] = alternate
        source["source_lineage_root_sha256"] = alternate["root_sha256"]
        source["full_source_contract"][
            "source_lineage_root_sha256"
        ] = alternate["root_sha256"]
        source["full_source_contract_hash"] = hash_json(
            source["full_source_contract"]
        )
    elif attack_kind == "terminal":
        source["terminal_sidecar_identity"] = _identity(
            "attacker_terminal_sidecar"
        )
    elif attack_kind == "calendar":
        source["frozen_trade_calendar_sha256"] = "9" * 64
    else:
        source["panel_manifest_sha256"] = "9" * 64
        source["member_panel_snapshot_hash"] = "8" * 64
        source["feature_tape_snapshot_hash"] = "7" * 64
        source["member_partition_records_hash"] = "6" * 64
        source["feature_tape_partition_records_hash"] = "5" * 64
        source["research_eligibility_snapshot_hash"] = "4" * 64
        eligibility = source["research_eligibility_identity"]
        eligibility["snapshot_hash"] = "4" * 64
        eligibility["identity_hash"] = hash_json(
            {
                key: value
                for key, value in eligibility.items()
                if key != "identity_hash"
            }
        )
        source["full_source_contract"]["price_cache_objects_sha256"] = "3" * 64
        source["full_source_contract_hash"] = hash_json(
            source["full_source_contract"]
        )

    source["combined_data_snapshot_hash"] = hashlib.sha256(
        (
            f"{source['member_panel_snapshot_hash']}|"
            f"{source['feature_tape_snapshot_hash']}|"
            f"{source['research_eligibility_snapshot_hash']}|"
            f"{source['terminal_sidecar_identity']['identity_hash']}|"
            f"{source['frozen_trade_calendar_sha256']}|"
            f"{source['source_lineage_identity']['identity_hash']}"
        ).encode("utf-8")
    ).hexdigest()
    source["identity_hash"] = hash_json(
        {key: value for key, value in source.items() if key != "identity_hash"}
    )
    envelope["manifest_hash"] = hash_json(envelope["manifest"])
    target.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="development view|RunManifest"):
        _load_reveal_manifest(target)
    assert list(registry.iterdir()) == []


def test_frozen_reveal_manifest_rejects_reporting_contract_reassignment(
    tmp_path,
):
    target = _sealed_input_fixture(tmp_path)
    envelope = json.loads(target.read_text(encoding="utf-8"))
    contract = envelope["manifest"]["sealed_reporting_contract"]
    contract["unique_primary_result_key"] = "composite:ridge"
    contract["unique_primary_backend"] = "ridge"
    envelope["manifest"]["sealed_reporting_contract_hash"] = hash_json(contract)
    envelope["manifest_hash"] = hash_json(envelope["manifest"])
    target.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="reporting contract is invalid"):
        _load_reveal_manifest(target)


def test_coordinated_champion_reseal_fails_without_development_evidence_rewrite(
    tmp_path,
):
    target = _sealed_input_fixture(tmp_path)
    envelope = json.loads(target.read_text(encoding="utf-8"))
    payload = envelope["manifest"]
    champion = payload["champions"][0]
    changed_expression = {
        "op": "cs_rank",
        "args": [
            {
                "op": "neg",
                "args": [
                    {
                        "op": "return",
                        "args": [{"op": "field", "name": "close"}],
                        "periods": 60,
                    }
                ],
            }
        ],
    }
    changed_json = canonical_json(changed_expression, PRODUCTION_DSL_SPEC)
    champion["canonical_expression_ast"] = json.loads(changed_json)
    champion["canonical_expression_json"] = changed_json
    champion["expression_hash"] = expression_hash(
        champion["canonical_expression_ast"],
        PRODUCTION_DSL_SPEC,
    )
    changed_candidate = hs300_research.Candidate(
        expression=champion["canonical_expression_ast"],
        agent_id=champion["agent_id"],
        hypothesis=champion["hypothesis"],
        mechanism_tags=tuple(champion["mechanism_tags"]),
        generation=champion["generation"],
        parents=tuple(champion["parents"]),
        operator=champion["operator"],
        control_expression=champion["control_expression_ast"],
        metadata=champion["metadata"],
    )
    champion["candidate_id"] = hs300_research.deterministic_candidate_id(
        changed_candidate
    )
    # The attacker coordinates every champion-local commitment and the outer
    # envelope, but deliberately leaves the completed development
    # summary/registry/ledger untouched.
    payload["champion_family_hash"] = hash_json(payload["champions"])
    envelope["manifest_hash"] = hash_json(payload)
    target.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="frozen composite family differs|development summary champion record mismatch",
    ):
        _load_reveal_manifest(target)


def test_research_implementation_snapshot_detects_semantic_version_drift():
    snapshot = hs300_research._research_implementation_snapshot(PROJECT)
    validated = hs300_research._validate_research_implementation_snapshot(
        snapshot,
        project_dir=PROJECT,
    )
    assert validated["aggregate_sha256"] == snapshot["aggregate_sha256"]

    drifted = json.loads(json.dumps(snapshot))
    drifted["files"][0]["sha256"] = "0" * 64
    content = {"files": drifted["files"], "runtime": drifted["runtime"]}
    drifted["aggregate_sha256"] = hash_json(content)
    with pytest.raises(ValueError, match="changed after development freeze"):
        hs300_research._validate_research_implementation_snapshot(
            drifted,
            project_dir=PROJECT,
        )


def test_attacker_cannot_reseal_changed_implementation_without_run_binding(tmp_path):
    target = _sealed_input_fixture(tmp_path)
    envelope = json.loads(target.read_text(encoding="utf-8"))
    payload = envelope["manifest"]
    identity = payload["research_implementation_identity"]
    identity["files"][0]["sha256"] = "0" * 64
    content = {"files": identity["files"], "runtime": identity["runtime"]}
    identity["aggregate_sha256"] = hash_json(content)
    payload["resolved_experiment_config"][
        "research_implementation_identity"
    ] = identity
    envelope["manifest_hash"] = hash_json(payload)
    target.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="does not bind the resolved"):
        _load_reveal_manifest(target)


def test_attacker_cannot_reseal_changed_feedback_threshold_without_run_binding(
    tmp_path,
):
    target = _sealed_input_fixture(tmp_path)
    envelope = json.loads(target.read_text(encoding="utf-8"))
    payload = envelope["manifest"]
    payload["resolved_experiment_config"]["adaptive_feedback_policy"][
        "minimum_coverage"
    ] = 0.81
    # Re-hashing only the outer sealed envelope cannot rewrite the already
    # frozen development RunManifest.config_hash.
    envelope["manifest_hash"] = hash_json(payload)
    target.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="does not bind the resolved"):
        _load_reveal_manifest(target)


def test_attacker_cannot_reseal_changed_behavior_threshold_without_run_binding(
    tmp_path,
):
    target = _sealed_input_fixture(tmp_path)
    envelope = json.loads(target.read_text(encoding="utf-8"))
    payload = envelope["manifest"]
    payload["resolved_experiment_config"]["behavioral_champion_policy"][
        "maximum_absolute_correlation"
    ] = 0.80
    # The outer envelope can be made internally consistent, but the completed
    # development RunManifest still binds the original 0.85 runtime policy.
    envelope["manifest_hash"] = hash_json(payload)
    target.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="does not bind the resolved"):
        _load_reveal_manifest(target)


def test_reveal_registry_is_once_per_snapshot_split_not_per_champion_family(tmp_path):
    source = {
        "combined_data_snapshot_hash": "a" * 64,
        "panel_manifest_sha256": "b" * 64,
        "identity_hash": "9" * 64,
    }
    first = _claim_replay_gate(
        tmp_path / "registry",
        source_snapshot_identity=source,
        split_identity_hash="c" * 64,
        development_manifest_hash="d" * 64,
        champion_family_hash="e" * 64,
        composite_family_hash="1" * 64,
    )
    assert first.is_file()
    assert first.name == f"{hash_json({'combined_data_snapshot_hash': 'a' * 64, 'split_identity_hash': 'c' * 64})}.json"
    with pytest.raises(RuntimeError, match="already revealed"):
        _claim_replay_gate(
            tmp_path / "registry",
            source_snapshot_identity={
                **source,
                # Re-serializing an identical semantic snapshot into another
                # panel manifest must not create a fresh replay gate.
                "panel_manifest_sha256": "8" * 64,
            },
            split_identity_hash="c" * 64,
            development_manifest_hash="f" * 64,
            champion_family_hash="9" * 64,
            # A different composite family is evidence, never a fresh gate.
            composite_family_hash="2" * 64,
        )


def test_registry_override_and_coordinated_locator_reseal_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="registry is fixed"):
        HS300ResearchConfig(
            project_dir=PROJECT,
            panel_dir=tmp_path / "panel",
            output_dir=tmp_path / "run-a",
            reveal_registry_dir=tmp_path / "alternate-registry",
        )

    target = _sealed_input_fixture(tmp_path)
    envelope = json.loads(target.read_text(encoding="utf-8"))
    payload = envelope["manifest"]
    malicious = dict(payload["operational_locators"])
    malicious["project_dir"] = str(tmp_path / "copied-project")
    malicious["reveal_registry_dir"] = str(tmp_path / "fresh-registry")
    payload["operational_locators"] = malicious
    payload["resolved_experiment_config"]["operational_locators"] = malicious
    envelope["manifest_hash"] = hash_json(payload)
    target.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="RunManifest|trusted operational"):
        _load_reveal_manifest(target)


def test_development_config_cannot_reenable_in_process_reveal(tmp_path):
    with pytest.raises(ValueError, match="separate 'research hs300-reveal'"):
        HS300ResearchConfig(
            project_dir=tmp_path,
            panel_dir=tmp_path / "panel",
            output_dir=tmp_path / "run",
            reveal_historical_replay=True,
        )


def test_bound_calendar_includes_feature_tape_warmup(tmp_path):
    panel_dir = tmp_path / "panel"
    normalized = tmp_path / "normalized"
    panel_dir.mkdir()
    normalized.mkdir()
    dates = pd.bdate_range("2010-01-04", "2011-01-10")
    calendar_path = normalized / "trade_calendar.parquet"
    pd.DataFrame(
        {"calendar_date": dates, "is_trading_day": "1"}
    ).to_parquet(calendar_path, index=False)
    manifest = {
        "first_date": "2011-01-04",
        "last_date": "2011-01-10",
        "feature_tape": {"first_date": "2010-01-04"},
        "normalized_input_sha256": {"trade_calendar": _sha(calendar_path)},
        "config": {"normalized_dir": str(normalized)},
    }

    loaded = hs300_research._load_bound_trading_dates(panel_dir, manifest)

    assert loaded[0] == pd.Timestamp("2010-01-04")
    assert loaded[-1] == pd.Timestamp("2011-01-10")


def test_real_development_orchestrator_never_calls_full_audit_or_full_loader(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(hs300_research, "_trusted_project_root", lambda: tmp_path.resolve())
    dates = pd.bdate_range("2017-01-02", "2021-12-31")
    split = hs300_research._make_historical_replay_split_from_dates(
        dates,
        train_end="2018-06-30",
        validation_end="2019-06-30",
        replay_end="2021-12-31",
        label_horizon=2,
        entry_delay=1,
    )
    member_hash = "a" * 64
    tape_hash = "b" * 64
    source_manifest = {
        "complete": True,
        "research_proxy_only": True,
        "storage_snapshot_id": "bounded-fixture",
        "first_date": str(dates[0].date()),
        "last_date": str(dates[-1].date()),
        "snapshot_hash": member_hash,
        "partitions": [],
        "validation": {"valid": True},
        "feature_tape": {"snapshot_hash": tape_hash, "partitions": []},
        "normalized_input_sha256": {"trade_calendar": "d" * 64},
        "research_eligibility": {
                "snapshot_hash": "e" * 64,
                "selection_column": "research_eligible",
                "current_time_only": True,
                "future_endpoint_availability_filter": False,
        },
        "terminal_stale_membership_exceptions": {
                "schema_version": "1.0",
                "complete": True,
                "policy": "terminal_stale_membership_after_verified_out_date",
                "price_fill_policy": "none",
                "terminal_return_policy": "unresolved_no_imputation",
                "rows": 0,
                "symbols": 0,
                "trading_days": 0,
                "snapshot_hash": "f" * 64,
            "file": {
                "path": (
                    "snapshots/bounded-fixture/"
                    "terminal_stale_membership_exceptions.parquet"
                ),
                "rows": 0,
                "sha256": "0" * 64,
                "bytes": 0,
            },
        },
    }
    bounded_source_lineage_identity = _add_source_lineage_contract(source_manifest)
    manifest_sha = "c" * 64
    bounded_source_snapshot_identity = hs300_research._source_snapshot_identity(
        source_manifest, manifest_sha
    )
    calls = {}
    identity_order = []

    monkeypatch.setattr(
        hs300_research,
        "_load_frozen_panel_manifest",
        lambda panel_dir: identity_order.append("panel_manifest")
        or (Path(panel_dir), source_manifest, manifest_sha),
    )
    monkeypatch.setattr(
        hs300_research,
        "_load_bound_trading_dates",
        lambda panel_dir, manifest: dates,
    )

    def forbidden_audit(*args, **kwargs):
        raise AssertionError("development command invoked the full sealed-data audit")

    monkeypatch.setattr(hs300_research, "audit_hs300_research_panel", forbidden_audit)

    def bounded_loader(panel_dir, *, end_date=None):
        calls["loader_end_date"] = end_date
        assert end_date == split.development_end
        development_dates = dates[dates <= end_date]
        index = pd.MultiIndex.from_product(
            [development_dates, ["A", "B"]], names=["date", "symbol"]
        )
        panel = pd.DataFrame(
            {
                "x": 1.0,
                "vendor_universe_member": True,
                "universe_member": True,
                "research_eligible": True,
            },
            index=index,
        )
        tape = panel.copy()
        universe_identity = hs300_research.build_research_universe_identity(
            panel, tape, panel
        )
        calendar_identity = _identity("bounded_fixture_calendar")
        return hs300_research.FrozenPanel(
            source_manifest,
            panel,
            panel,
            tape,
            universe_identity,
            bounded_source_lineage_identity,
            bounded_source_snapshot_identity,
            dates,
            calendar_identity,
            manifest_sha,
        )

    monkeypatch.setattr(hs300_research, "load_frozen_hs300_panel", bounded_loader)
    monkeypatch.setattr(
        hs300_research,
        "load_codex_seed_candidates",
        lambda *a, **k: identity_order.append("seed_cards") or [],
    )
    monkeypatch.setattr(
        hs300_research,
        "load_codex_injection_candidates",
        lambda *a, **k: identity_order.append("injection_cards") or [],
    )
    monkeypatch.setattr(
        hs300_research,
        "_research_implementation_snapshot",
        lambda project_dir: identity_order.append("implementation_snapshot")
        or {"aggregate_sha256": "9" * 64},
    )
    monkeypatch.setattr(
        hs300_research,
        "validate_semantic_review_manifest",
        lambda project_dir, path: identity_order.append("semantic_review")
        or {
            "schema": "cogalpha-plus.semantic-review/v1",
            "candidate_count": 176,
            "review_set_hash": "7" * 64,
            "approval_binding_hash": "8" * 64,
        },
    )
    monkeypatch.setattr(
        hs300_research,
        "_validate_research_implementation_snapshot",
        lambda snapshot, project_dir=None: identity_order.append(
            "implementation_revalidated"
        )
        or snapshot,
    )
    expected_policy = hs300_research._load_adaptive_feedback_policy(
        PROJECT,
        "plus",
    )
    monkeypatch.setattr(
        hs300_research,
        "_load_adaptive_feedback_policy",
        lambda project_dir, profile: expected_policy,
    )
    expected_behavioral_policy = (
        hs300_research._load_behavioral_champion_policy(PROJECT, "plus")
    )
    monkeypatch.setattr(
        hs300_research,
        "_load_behavioral_champion_policy",
        lambda project_dir, profile: expected_behavioral_policy,
    )
    expected_handoff_policy, expected_composite_policy = (
        hs300_research._load_composite_policies(PROJECT, "plus")
    )
    monkeypatch.setattr(
        hs300_research,
        "_load_composite_policies",
        lambda project_dir, profile: identity_order.append("composite_policies")
        or (
            expected_handoff_policy,
            expected_composite_policy,
            ),
        )
    monkeypatch.setattr(
        hs300_research,
        "load_terminal_label_bridges",
        lambda path: ((), {"schema_version": 1, "identity_hash": "6" * 64}),
    )

    def fake_experiment(**kwargs):
        calls["run_panel_max"] = kwargs["full_panel"].index.get_level_values(
            "date"
        ).max()
        calls["reveal_sealed"] = kwargs["reveal_sealed"]
        assert kwargs["adaptive_feedback_policy"] is expected_policy
        assert kwargs["behavioral_champion_policy"] is expected_behavioral_policy
        assert kwargs["development_handoff_policy"] == (
            expected_handoff_policy.to_dict()
        )
        assert kwargs["composite_family_policy"] == (
            expected_composite_policy.to_dict()
        )
        return SimpleNamespace(champions=()), {"run_id": "dev-only"}

    monkeypatch.setattr(hs300_research, "run_experiment", fake_experiment)
    frozen_path = tmp_path / "run" / "sealed_reveal_manifest.json"
    monkeypatch.setattr(
        hs300_research,
        "_freeze_reveal_manifest",
        lambda **kwargs: (frozen_path, "f" * 64),
    )
    result = hs300_research.run_hs300_research(
        HS300ResearchConfig(
            project_dir=tmp_path,
            panel_dir=tmp_path / "panel",
            output_dir=tmp_path / "run",
            train_end="2018-06-30",
            validation_end="2019-06-30",
            replay_end="2021-12-31",
            label_horizon=2,
            entry_delay=1,
            bootstrap_samples=20,
            expected_members=2,
        )
    )
    assert calls == {
        "loader_end_date": split.development_end,
        "run_panel_max": split.development_end,
        "reveal_sealed": False,
    }
    assert identity_order == [
        "implementation_snapshot",
        "semantic_review",
        "seed_cards",
        "injection_cards",
        "implementation_revalidated",
        "composite_policies",
        "implementation_revalidated",
        "panel_manifest",
    ]
    assert result.summary["sealed_market_rows_loaded_by_development"] == 0
    assert result.summary["economic_semantic_review"]["candidate_count"] == 176


def test_real_development_rejects_semantic_review_before_factor_or_market_reads(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(hs300_research, "_trusted_project_root", lambda: tmp_path.resolve())
    calls: list[str] = []
    monkeypatch.setattr(
        hs300_research,
        "_research_implementation_snapshot",
        lambda project_dir: calls.append("implementation_snapshot")
        or {"aggregate_sha256": "9" * 64},
    )

    def rejected_review(project_dir, path):
        calls.append("semantic_review")
        raise ValueError("semantic review contains a rejected card")

    def forbidden(*args, **kwargs):
        calls.append("forbidden_read")
        raise AssertionError("factor or market data was read before semantic approval")

    monkeypatch.setattr(
        hs300_research,
        "validate_semantic_review_manifest",
        rejected_review,
    )
    monkeypatch.setattr(hs300_research, "load_codex_seed_candidates", forbidden)
    monkeypatch.setattr(
        hs300_research,
        "load_codex_injection_candidates",
        forbidden,
    )
    monkeypatch.setattr(hs300_research, "_load_frozen_panel_manifest", forbidden)

    with pytest.raises(ValueError, match="rejected card"):
        hs300_research.run_hs300_research(
            HS300ResearchConfig(
                project_dir=tmp_path,
                panel_dir=tmp_path / "panel",
                output_dir=tmp_path / "run",
                replay_end="2025-12-31",
                bootstrap_samples=20,
            )
        )

    assert calls == ["implementation_snapshot", "semantic_review"]
