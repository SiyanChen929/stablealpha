from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.manifest import hash_json
from cogalpha_plus.portfolio import ConstrainedPortfolioOptimizer, PortfolioConstraints
from cogalpha_plus.research_proxy import (
    CompositeFamilyPolicy,
    DevelopmentFeatureHandoff,
    DevelopmentHandoffPolicy,
    RESEARCH_PROXY_TIER,
    ResearchProxyBoundaryError,
    ResearchProxyControlSnapshot,
    SealedCompositeClaimCapability,
    _audit_neutral_feature_matrix,
    _audit_development_labels,
    _aligned_model_frame,
    _exact_development_member_dates,
    build_log_mcap_benchmark_proxy,
    dump_frozen_composite_family,
    fit_frozen_development_composite_family,
    load_frozen_composite_family,
    optimize_research_proxy_portfolio,
)
from cogalpha_plus.runner import panel_snapshot_hash
from cogalpha_plus.validation import SplitPlan


def _hash(value: str) -> str:
    return (value * 64)[:64]


def _identity(kind: str) -> dict[str, object]:
    payload: dict[str, object] = {"schema_version": 1, "kind": kind}
    return {**payload, "identity_hash": hash_json(payload)}


def _handoff(seed: int = 7) -> DevelopmentFeatureHandoff:
    rng = np.random.default_rng(seed)
    policy = DevelopmentHandoffPolicy()
    dates = pd.bdate_range("2020-01-02", periods=80, name="date")
    symbols = pd.Index([f"S{i:03d}" for i in range(30)], name="symbol")
    index = pd.MultiIndex.from_product([dates, symbols])
    features = pd.DataFrame(
        rng.normal(size=(len(index), 20)),
        index=index,
        columns=[f"champion_{i:02d}" for i in range(20)],
    )
    labels = pd.Series(
        0.5 * features.iloc[:, 0]
        - 0.3 * features.iloc[:, 1]
        + rng.normal(scale=0.8, size=len(index)),
        index=index,
        name="label",
    )
    train_dates = dates[:55]
    valid_dates = dates[55:]
    train = features.loc[index.get_level_values("date").isin(train_dates)]
    valid = features.loc[index.get_level_values("date").isin(valid_dates)]
    train_y = labels.reindex(train.index)
    valid_y = labels.reindex(valid.index)
    audit_index = pd.MultiIndex.from_product(
        [dates, features.columns], names=["date", "factor"]
    )
    audit = pd.DataFrame(
        {
            "valid_names": 30,
            "total_names": 30,
            "coverage": 1.0,
            "zero_impute_share": 0.0,
            "constant_or_all_missing": False,
            "passes_gate": True,
        },
        index=audit_index,
    )
    provenance = {
        "development_root_hash": _hash("a"),
        "champion_family_hash": _hash("b"),
        "split_identity_hash": _hash("c"),
        "development_member_panel_hash": _hash("d"),
        "development_feature_tape_hash": _hash("e"),
        "research_universe_identity": _identity("synthetic_research_universe"),
        "trading_calendar_identity": _identity("synthetic_trading_calendar"),
        "source_lineage_identity": _identity("synthetic_source_lineage"),
        "source_snapshot_identity": _identity("synthetic_source_snapshot"),
        "train_features_hash": panel_snapshot_hash(train),
        "validation_features_hash": panel_snapshot_hash(valid),
        "train_labels_hash": panel_snapshot_hash(train_y.to_frame()),
        "validation_labels_hash": panel_snapshot_hash(valid_y.to_frame()),
        "expression_hashes": [hashlib.sha256(str(i).encode()).hexdigest() for i in range(20)],
        "directions": {name: 1 for name in features.columns},
        "development_handoff_policy": policy.to_dict(),
        "development_handoff_policy_hash": policy.policy_hash,
        "development_label_audit": _audit_development_labels(
            labels,
            policy,
            train_dates=train_dates,
            validation_dates=valid_dates,
        ),
        "neutralization": {"method": "unit_test_same_date"},
        "neutral_feature_audit": {
            "coverage_audit_hash": panel_snapshot_hash(audit),
        },
        "sealed_rows_loaded": 0,
        "sealed_results_consumed": False,
    }
    provenance["research_universe_identity_hash"] = provenance[
        "research_universe_identity"
    ]["identity_hash"]
    provenance["trading_calendar_identity_hash"] = provenance[
        "trading_calendar_identity"
    ]["identity_hash"]
    provenance["source_lineage_identity_hash"] = provenance[
        "source_lineage_identity"
    ]["identity_hash"]
    provenance["source_snapshot_identity_hash"] = provenance[
        "source_snapshot_identity"
    ]["identity_hash"]
    return DevelopmentFeatureHandoff(
        train,
        valid,
        train_y,
        valid_y,
        audit,
        tuple(features.columns),
        provenance,
    )


def test_warmup_is_leading_train_only_and_validation_never_dropped():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2021-01-04", periods=80, name="date")
    symbols = pd.Index([f"S{i:02d}" for i in range(25)], name="symbol")
    index = pd.MultiIndex.from_product([dates, symbols])
    frame = pd.DataFrame(
        rng.normal(size=(len(index), 20)),
        index=index,
        columns=[f"f{i}" for i in range(20)],
    )
    frame.loc[frame.index.get_level_values("date").isin(dates[:5]), :] = np.nan
    policy = DevelopmentHandoffPolicy(minimum_retained_train_dates=40)
    audit, summary, retained = _audit_neutral_feature_matrix(
        frame,
        policy,
        train_dates=dates[:60],
        validation_dates=dates[60:],
    )
    assert retained[0] == dates[5]
    assert summary["warmup_dropped_train_date_count"] == 5
    assert not audit.empty

    broken = frame.copy()
    broken.loc[(dates[60], slice(None)), "f0"] = np.nan
    with pytest.raises(ResearchProxyBoundaryError, match="inner-validation"):
        _audit_neutral_feature_matrix(
            broken,
            policy,
            train_dates=dates[:60],
            validation_dates=dates[60:],
        )


def test_model_feature_ranks_are_independent_of_future_label_missing_mask():
    dates = pd.bdate_range("2023-01-02", periods=2, name="date")
    symbols = pd.Index([f"S{i:02d}" for i in range(30)], name="symbol")
    index = pd.MultiIndex.from_product([dates, symbols])
    features = pd.DataFrame(
        {
            "f0": np.tile(np.arange(30, dtype=float), 2),
            "f1": np.tile(np.arange(30, 0, -1, dtype=float), 2),
        },
        index=index,
    )
    labels = pd.Series(np.linspace(-1.0, 1.0, len(index)), index=index)
    complete_x, _ = _aligned_model_frame(features, labels)
    missing = labels.copy()
    missing.loc[(dates[0], symbols[-1])] = np.nan
    masked_x, _ = _aligned_model_frame(features, missing)
    expected = complete_x.drop(index=(dates[0], symbols[-1]))
    pd.testing.assert_frame_equal(masked_x, expected)


def test_development_label_audit_rejects_any_missing_row_or_whole_date():
    dates = pd.bdate_range("2023-01-02", periods=4, name="date")
    symbols = pd.Index([f"S{i:02d}" for i in range(30)], name="symbol")
    index = pd.MultiIndex.from_product([dates, symbols])
    labels = pd.Series(0.1, index=index)
    policy = DevelopmentHandoffPolicy()
    audit = _audit_development_labels(
        labels,
        policy,
        train_dates=dates[:2],
        validation_dates=dates[2:],
    )
    assert audit["all_development"]["valid_rows"] == len(labels)
    assert audit["valid_mask_hash"]
    assert audit["retained_row_hash"]

    one_missing = labels.copy()
    one_missing.loc[(dates[0], symbols[0])] = np.nan
    with pytest.raises(ResearchProxyBoundaryError, match="label coverage failed"):
        _audit_development_labels(
            one_missing,
            policy,
            train_dates=dates[:2],
            validation_dates=dates[2:],
        )

    whole_date = labels.copy()
    whole_date.loc[(dates[3], slice(None))] = np.nan
    with pytest.raises(ResearchProxyBoundaryError, match="label coverage failed"):
        _audit_development_labels(
            whole_date,
            policy,
            train_dates=dates[:2],
            validation_dates=dates[2:],
        )


def test_exact_development_dates_include_each_split_partition_once():
    dates = pd.bdate_range("2022-01-03", periods=12)
    split = SplitPlan(
        train_dates=tuple(dates[:4]),
        validation_dates=tuple(dates[6:9]),
        sealed_dates=tuple(dates[11:]),
        purge_after_train=tuple(dates[4:6]),
        purge_after_validation=tuple(dates[9:11]),
        label_horizon=1,
        entry_delay=1,
    )
    expected = tuple(dates[:11])
    assert _exact_development_member_dates(split) == expected
    assert len(set(_exact_development_member_dates(split))) == len(expected)


def test_fixed_family_is_jointly_frozen_and_round_trips_without_pickle(tmp_path):
    handoff = _handoff()
    policy = CompositeFamilyPolicy.fixed_v1()
    family = fit_frozen_development_composite_family(
        handoff,
        policy=policy,
        expected_policy_hash=policy.policy_hash,
    )
    assert family.primary_backend == "lightgbm"
    assert tuple(family.family_commitment["family_members"]) == (
        "equal_weight",
        "ridge",
        "lightgbm",
    )
    assert family.family_commitment["source_snapshot_identity"] == (
        handoff.provenance["source_snapshot_identity"]
    )
    assert family.family_commitment["source_snapshot_identity_hash"] == (
        handoff.provenance["source_snapshot_identity_hash"]
    )
    assert family.family_commitment["validation_metrics_used_for_selection_or_tuning"] is False
    assert family.family_commitment["configurations"]["ridge"]["alpha"] == 1.0
    assert family.family_commitment["configurations"]["lightgbm"]["n_jobs"] == 1
    assert family.family_commitment["missing_feature_policy"] == (
        "same_date_rank_on_full_eligible_feature_cross_section_then_zero_"
        "impute_before_label_join_and_mask"
    )
    assert family.family_commitment["fit_scope"] == (
        "refit_on_all_development_train_plus_inner_validation_before_"
        "family_commitment_and_sealed_reveal"
    )
    assert "after_label_alignment" not in family.family_commitment[
        "missing_feature_policy"
    ]
    assert "after_family_freeze" not in family.family_commitment["fit_scope"]
    before = family.predict_all_development(handoff, partition="inner_validation")

    bundle = tmp_path / "family"
    dump_frozen_composite_family(family, bundle)
    loaded = load_frozen_composite_family(
        bundle,
        expected_composite_family_hash=family.composite_family_hash,
    )
    after = loaded.predict_all_development(handoff, partition="inner_validation")
    assert loaded.composite_family_hash == family.composite_family_hash
    np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-12)

    loaded._fitted_models["ridge"].model.coef_[0] += 1.0
    with pytest.raises(ResearchProxyBoundaryError, match="mutated"):
        loaded.predict_development(handoff, partition="inner_validation")


def test_policy_hash_is_mandatory_and_no_validation_backend_selection():
    handoff = _handoff()
    policy = CompositeFamilyPolicy.fixed_v1()
    with pytest.raises(ResearchProxyBoundaryError, match="pre-registration"):
        fit_frozen_development_composite_family(
            handoff,
            policy=policy,
            expected_policy_hash=_hash("f"),
        )
    with pytest.raises(ResearchProxyBoundaryError, match="unique primary"):
        CompositeFamilyPolicy(
            primary_backend="equal_weight",
            members=("equal_weight", "ridge", "lightgbm"),
            ridge_alpha=1.0,
            random_state=17,
            lightgbm_configuration=dict(policy.lightgbm_configuration),
        )


def test_sealed_family_audit_is_no_drop_and_precedes_imputation(tmp_path):
    handoff = _handoff(19)
    policy = CompositeFamilyPolicy.fixed_v1()
    family = fit_frozen_development_composite_family(
        handoff,
        policy=policy,
        expected_policy_hash=policy.policy_hash,
    )
    sealed = handoff.validation_features.copy()
    audit = family.audit_sealed_feature_matrix(sealed)
    assert audit["all_passed"] is True
    assert audit["sealed_dates_dropped"] == 0
    assert audit["policy_hash"] == DevelopmentHandoffPolicy().policy_hash

    reordered = sealed.loc[:, list(reversed(sealed.columns))]
    claim_payload = {
        "schema_version": 1,
        "gate_kind": "known-history-isolated-replay",
        "status": "retired_before_any_sealed_row_was_loaded",
        "combined_data_snapshot_hash": _hash("d"),
        "source_snapshot_identity_hash": handoff.provenance[
            "source_snapshot_identity_hash"
        ],
        "split_identity_hash": handoff.provenance["split_identity_hash"],
        "composite_family_hash": family.composite_family_hash,
    }
    claim_path = tmp_path / (
        hash_json(
                {
                    "combined_data_snapshot_hash": _hash("d"),
                    "split_identity_hash": handoff.provenance["split_identity_hash"],
                }
        )
        + ".json"
    )
    claim_path.write_text(json.dumps(claim_payload), encoding="utf-8")
    capability = SealedCompositeClaimCapability(
        composite_family_hash=family.composite_family_hash,
        split_identity_hash=handoff.provenance["split_identity_hash"],
        combined_data_snapshot_hash=_hash("d"),
        source_snapshot_identity_hash=handoff.provenance[
            "source_snapshot_identity_hash"
        ],
        persistent_claim_path=str(claim_path),
        persistent_claim_sha256=hashlib.sha256(claim_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ResearchProxyBoundaryError, match="columns differ"):
        family.predict_all_for_sealed_gate(
            reordered,
            expected_composite_family_hash=family.composite_family_hash,
            claim_capability=capability,
        )
    with pytest.raises(ResearchProxyBoundaryError, match="already consumed"):
        family.predict_all_for_sealed_gate(
            sealed,
            expected_composite_family_hash=family.composite_family_hash,
            claim_capability=capability,
        )

    one_date = sealed.index.get_level_values("date").min()
    low_coverage = sealed.copy()
    names = low_coverage.loc[(one_date, slice(None)), :].index[10:]
    low_coverage.loc[names, low_coverage.columns[0]] = np.nan
    with pytest.raises(ResearchProxyBoundaryError, match="no-drop coverage"):
        family.audit_sealed_feature_matrix(low_coverage)

    constant = sealed.copy()
    constant.loc[(one_date, slice(None)), constant.columns[1]] = 0.0
    with pytest.raises(ResearchProxyBoundaryError, match="constant"):
        family.audit_sealed_feature_matrix(constant)


def test_safe_artifact_tamper_is_rejected_before_prediction(tmp_path):
    handoff = _handoff(23)
    policy = CompositeFamilyPolicy.fixed_v1()
    family = fit_frozen_development_composite_family(
        handoff, policy=policy, expected_policy_hash=policy.policy_hash
    )
    bundle = tmp_path / "family"
    dump_frozen_composite_family(family, bundle)
    ridge_path = bundle / "ridge.json"
    ridge_path.write_text(ridge_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ResearchProxyBoundaryError, match="artifact"):
        load_frozen_composite_family(
            bundle,
            expected_composite_family_hash=family.composite_family_hash,
        )


def test_strict_long_only_proxy_requires_complete_same_date_controls(tmp_path):
    handoff = _handoff(11)
    policy = CompositeFamilyPolicy.fixed_v1()
    family = fit_frozen_development_composite_family(
        handoff,
        policy=policy,
        expected_policy_hash=policy.policy_hash,
    )
    day = handoff.validation_features.index.get_level_values("date").max()
    alpha = family.development_alpha_snapshot(
        handoff,
        as_of=day,
        partition="inner_validation",
    )
    names = alpha.values.index
    benchmark = build_log_mcap_benchmark_proxy(
        pd.Series(np.linspace(10.0, 10.2, len(names)), index=names),
        as_of=day,
    )
    styles = pd.DataFrame(
        {
            "beta": np.linspace(0.8, 1.2, len(names)),
            "log_mcap": np.linspace(-1.0, 1.0, len(names)),
            "momentum": np.sin(np.arange(len(names))),
            "volatility": np.cos(np.arange(len(names))),
            "liquidity": np.linspace(1.0, -1.0, len(names)),
        },
        index=names,
    )
    covariance = pd.DataFrame(
        np.eye(len(names)) * 0.0001,
        index=names,
        columns=names,
    )
    controls = ResearchProxyControlSnapshot(
        day,
        covariance,
        pd.Series([f"I{i % 5}" for i in range(len(names))], index=names),
        styles,
        pd.Series(20_000_000.0, index=names),
        pd.Series(True, index=names),
    )
    optimizer = ConstrainedPortfolioOptimizer(
        PortfolioConstraints(
            max_weight=0.05,
            max_active_weight=0.02,
            max_one_way_turnover=0.20,
            max_industry_active=0.05,
        )
    )
    result = optimize_research_proxy_portfolio(
        optimizer,
        family=family,
        alpha_snapshot=alpha,
        benchmark=benchmark,
        previous_weights=benchmark.weights.copy(),
        controls=controls,
        portfolio_value=1_000_000.0,
    )
    assert result.tier == RESEARCH_PROXY_TIER
    assert result.pnl_authorized is False
    with pytest.raises(ResearchProxyBoundaryError, match="cannot generate P&L"):
        result.require_event_driven_pnl_authority()

    incomplete = ResearchProxyControlSnapshot(
        day,
        covariance,
        controls.industry,
        styles.drop(columns="liquidity"),
        controls.adv20_dollars,
        controls.tradable,
    )
    with pytest.raises(ResearchProxyBoundaryError, match="omit required"):
        optimize_research_proxy_portfolio(
            optimizer,
            family=family,
            alpha_snapshot=alpha,
            benchmark=benchmark,
            previous_weights=benchmark.weights.copy(),
            controls=incomplete,
            portfolio_value=1_000_000.0,
        )


def test_cash_bootstrap_never_silently_relaxes_turnover():
    # The full optimizer fixture is covered above; this check targets the
    # explicit pre-solver cash-leg guard through the same strict interface.
    handoff = _handoff(17)
    policy = CompositeFamilyPolicy.fixed_v1()
    family = fit_frozen_development_composite_family(
        handoff, policy=policy, expected_policy_hash=policy.policy_hash
    )
    day = handoff.validation_features.index.get_level_values("date").max()
    alpha = family.development_alpha_snapshot(
        handoff, as_of=day, partition="inner_validation"
    )
    names = alpha.values.index
    benchmark = build_log_mcap_benchmark_proxy(
        pd.Series(10.0, index=names), as_of=day
    )
    controls = ResearchProxyControlSnapshot(
        day,
        pd.DataFrame(np.eye(len(names)) * 0.0001, index=names, columns=names),
        pd.Series("I", index=names),
        pd.DataFrame(
            {name: np.linspace(-1, 1, len(names)) for name in ("beta", "log_mcap", "momentum", "volatility", "liquidity")},
            index=names,
        ),
        pd.Series(20_000_000.0, index=names),
        pd.Series(True, index=names),
    )
    with pytest.raises(ResearchProxyBoundaryError, match="bootstrap"):
        optimize_research_proxy_portfolio(
            ConstrainedPortfolioOptimizer(),
            family=family,
            alpha_snapshot=alpha,
            benchmark=benchmark,
            previous_weights=pd.Series(0.0, index=names),
            controls=controls,
            portfolio_value=1_000_000.0,
        )
