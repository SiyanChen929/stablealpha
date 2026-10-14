import pytest
import numpy as np
import pandas as pd

from cogalpha_plus.evolution import (
    BehavioralChampionPolicy,
    Candidate,
    CandidateScore,
    EvaluatedCandidate,
    MechanismAwareOperators,
    TemplateCandidateFactory,
)
from cogalpha_plus.metrics import cross_sectional_neutralize, forward_return
from cogalpha_plus.research_proxy import DevelopmentHandoffPolicy
from cogalpha_plus.synthetic import make_synthetic_market
from cogalpha_plus.validation import (
    DevelopmentEvaluator as _ProductionDevelopmentEvaluator,
    SealedGate as _ProductionSealedGate,
    audit_price_scale_invariance,
    build_cross_sectional_mask,
    build_research_feature_frame,
    make_split_plan,
    sealed_reporting_contract_v1,
)
from cogalpha_plus.manifest import hash_json
from cogalpha_plus.dsl import ValidationSpec, canonical_json, evaluate_expression


class DevelopmentEvaluator(_ProductionDevelopmentEvaluator):
    """Synthetic fixture wrapper with an explicit panel-calendar fallback."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_test_calendar_from_panel", True)
        super().__init__(*args, **kwargs)


class SealedGate(_ProductionSealedGate):
    """Synthetic fixture wrapper with an explicit panel-calendar fallback."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_test_calendar_from_panel", True)
        super().__init__(*args, **kwargs)


def _calendar(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        panel.index.get_level_values("date").unique()
    ).sort_values()


def _behavior_test_candidate(candidate_id: str, fitness: float) -> EvaluatedCandidate:
    candidate = Candidate(
        expression={"op": "field", "name": "volume"},
        agent_id="L4_behavior_test",
        hypothesis=candidate_id,
        mechanism_tags=("test",),
        candidate_id=candidate_id,
        metadata={"expected_direction": 1},
    )
    return EvaluatedCandidate(
        candidate,
        CandidateScore(
            fitness=fitness,
            signed_rank_ic=0.02,
            neutral_rank_ic=0.018,
            turnover=0.1,
            worst_window_ic=0.005,
            style_retention=0.9,
        ),
    )


def _rank_behavior(seed: int, *, dates: int = 6, symbols: int = 50) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2019-01-02", periods=dates), [f"S{i:03d}" for i in range(symbols)]],
        names=["date", "symbol"],
    )
    chunks = [rng.permutation(symbols).astype(float) for _ in range(dates)]
    return pd.Series(np.concatenate(chunks), index=index, dtype=float)


def _behavior_only_evaluator(behaviors: dict[str, pd.Series]) -> DevelopmentEvaluator:
    evaluator = object.__new__(DevelopmentEvaluator)
    evaluator.min_names = 10

    def load(candidate: Candidate) -> pd.Series:
        if candidate.candidate_id.startswith("sealed"):
            raise AssertionError("sealed candidate was accessed")
        return behaviors[candidate.candidate_id]

    evaluator._inner_validation_rank_behavior = load
    return evaluator


def test_plus_behavior_gate_rejects_correlated_copy_and_reports_nearest():
    base = _rank_behavior(1)
    behaviors = {
        "best": base,
        "copy": base.copy(),
        "independent": _rank_behavior(2),
    }
    eligible = [
        _behavior_test_candidate("best", 3.0),
        _behavior_test_candidate("copy", 2.0),
        _behavior_test_candidate("independent", 1.0),
    ]
    result = _behavior_only_evaluator(behaviors).select_behaviorally_distinct_champions(
        eligible,
        2,
        BehavioralChampionPolicy(
            maximum_absolute_correlation=0.85,
            initial_top_pool_size=3,
        ),
    )
    assert [item.candidate.candidate_id for item in result.champions] == [
        "best",
        "independent",
    ]
    assert result.audit["correlation_conflict_candidates"] == 1
    assert result.audit["conflict_examples"][0]["candidate_id"] == "copy"
    assert result.audit["conflict_examples"][0][
        "nearest_selected_candidate_id"
    ] == "best"
    assert result.audit["conflict_examples"][0][
        "maximum_absolute_correlation"
    ] == pytest.approx(1.0)
    assert result.audit["sealed_data_consumed"] is False
    assert result.audit["behavior_vectors_retained_after_selection"] == 0


def test_plus_behavior_gate_expands_beyond_top_pool_and_selects_20_independent():
    base = _rank_behavior(100)
    behaviors = {"leader": base}
    eligible = [_behavior_test_candidate("leader", 100.0)]
    # The leading pool contains only one independent mechanism and 19 copies.
    for position in range(19):
        candidate_id = f"copy_{position:02d}"
        behaviors[candidate_id] = base.copy()
        eligible.append(_behavior_test_candidate(candidate_id, 99.0 - position))
    # The selector must expand to the remainder instead of relaxing 0.85.
    for position in range(19):
        candidate_id = f"independent_{position:02d}"
        behaviors[candidate_id] = _rank_behavior(200 + position)
        eligible.append(_behavior_test_candidate(candidate_id, 70.0 - position))

    result = _behavior_only_evaluator(behaviors).select_behaviorally_distinct_champions(
        eligible,
        20,
        BehavioralChampionPolicy(
            maximum_absolute_correlation=0.85,
            initial_top_pool_size=20,
        ),
    )
    assert len(result.champions) == 20
    assert result.audit["expanded_to_all_passed"] is True
    assert result.audit["correlation_conflict_candidates"] == 19
    assert result.audit["threshold_relaxed"] is False
    assert result.audit["status"] == "complete"
    assert all(
        item["maximum_absolute_correlation"] <= 0.85
        for item in result.audit["champions"]
    )


def test_plus_behavior_gate_rejects_invalid_first_behavior_fail_closed():
    invalid = _rank_behavior(301)
    first_date = invalid.index.get_level_values("date").unique()[0]
    invalid.loc[first_date] = np.nan
    infinite = _rank_behavior(302)
    infinite.iloc[0] = np.inf
    valid = _rank_behavior(303)
    behaviors = {
        "missing_daily_cross_section": invalid,
        "infinite_value": infinite,
        "valid": valid,
    }
    eligible = [
        _behavior_test_candidate("missing_daily_cross_section", 3.0),
        _behavior_test_candidate("infinite_value", 2.0),
        _behavior_test_candidate("valid", 1.0),
    ]

    result = _behavior_only_evaluator(behaviors).select_behaviorally_distinct_champions(
        eligible,
        1,
        BehavioralChampionPolicy(initial_top_pool_size=3),
    )

    assert [item.candidate.candidate_id for item in result.champions] == ["valid"]
    assert result.audit["behavior_recompute_failures"] == 2
    assert result.audit["status"] == "complete"


def test_behavior_metric_is_mean_of_complete_daily_cross_sectional_spearman():
    left = _rank_behavior(401, dates=2, symbols=50)
    right = left.copy()
    second_date = left.index.get_level_values("date").unique()[1]
    right.loc[second_date] = -left.loc[second_date].to_numpy()
    evaluator = _behavior_only_evaluator({})

    measured = evaluator._complete_daily_behavior_correlation(left, right)
    per_date = []
    for date in left.index.get_level_values("date").unique():
        per_date.append(left.loc[date].corr(right.loc[date], method="spearman"))

    assert measured == pytest.approx(float(np.mean(per_date)))
    assert measured == pytest.approx(0.0)


def test_behavior_pair_comparison_rejects_one_missing_daily_overlap():
    base = _rank_behavior(501)
    partial = _rank_behavior(502)
    first_date = partial.index.get_level_values("date").unique()[0]
    partial.loc[first_date] = np.nan
    fallback = _rank_behavior(503)
    behaviors = {"base": base, "partial": partial, "fallback": fallback}
    eligible = [
        _behavior_test_candidate("base", 3.0),
        _behavior_test_candidate("partial", 2.0),
        _behavior_test_candidate("fallback", 1.0),
    ]

    result = _behavior_only_evaluator(behaviors).select_behaviorally_distinct_champions(
        eligible,
        2,
        BehavioralChampionPolicy(initial_top_pool_size=3),
    )

    assert [item.candidate.candidate_id for item in result.champions] == [
        "base",
        "fallback",
    ]
    # The candidate is rejected at its self-audit before pairwise omission can
    # make its remaining dates look acceptable.
    assert result.audit["behavior_recompute_failures"] == 1


def test_research_feature_frame_uses_adjusted_prices_without_mutating_raw_panel():
    panel = make_synthetic_market(days=80, symbols=20)
    raw_close = panel["close"].copy()
    panel["adj_close"] = raw_close * 2.0
    feature_frame = build_research_feature_frame(panel)
    assert np.array_equal(feature_frame["close"].to_numpy(), panel["adj_close"].to_numpy())
    assert np.array_equal(panel["close"].to_numpy(), raw_close.to_numpy())


def test_development_labels_use_adjusted_vwap_not_cash_execution_vwap():
    panel = make_synthetic_market(days=260, symbols=25, seed=31)
    # Introduce a synthetic corporate-action scale change in raw cash prices,
    # while the research series remains continuous.
    dates = panel.index.get_level_values("date").unique().sort_values()
    event = dates[120]
    after = panel.index.get_level_values("date") >= event
    panel.loc[after, ["open", "high", "low", "close", "vwap"]] *= 0.5
    panel.loc[after, "adj_factor"] *= 2.0
    split = make_split_plan(panel, label_horizon=5, entry_delay=1)
    development = panel.loc[panel.index.get_level_values("date") <= split.development_end]
    evaluator = DevelopmentEvaluator(development, split, min_names=20, bootstrap_samples=10)
    expected = forward_return(
        evaluator.feature_frame["vwap"],
        split.label_horizon,
        entry_delay=split.entry_delay,
        trading_calendar=evaluator.trading_calendar,
    )
    raw = forward_return(
        development["vwap"], split.label_horizon,
        entry_delay=split.entry_delay, trading_calendar=evaluator.trading_calendar,
    )
    assert evaluator.labels.equals(expected)
    assert not evaluator.labels.equals(raw)


def test_development_behavior_selector_cannot_be_constructed_with_sealed_rows():
    panel = make_synthetic_market(days=220, symbols=24, seed=33)
    split = make_split_plan(
        panel,
        label_horizon=2,
        train_fraction=0.50,
        validation_fraction=0.20,
    )
    assert panel.index.get_level_values("date").max() > split.development_end
    with pytest.raises(ValueError, match="development evaluator was given sealed data"):
        DevelopmentEvaluator(panel, split, min_names=20, bootstrap_samples=10)


def test_behavior_recompute_binds_full_candidate_identity_and_direction():
    panel = make_synthetic_market(days=220, symbols=30, seed=334)
    split = make_split_plan(
        panel,
        label_horizon=2,
        train_fraction=0.50,
        validation_fraction=0.20,
    )
    development = panel.loc[
        panel.index.get_level_values("date") <= split.development_end
    ]
    evaluator = DevelopmentEvaluator(
        development,
        split,
        min_names=20,
        bootstrap_samples=10,
    )
    candidate = None
    score = None
    for proposed in TemplateCandidateFactory(seed=334).generate(16):
        proposed_score = evaluator(proposed)
        if proposed_score.passed:
            candidate = proposed
            score = proposed_score
            break
    assert candidate is not None and score is not None and score.passed
    original_agent = candidate.agent_id
    candidate.agent_id = "tampered_role"

    with pytest.raises(ValueError, match="frozen behavior identity"):
        evaluator._inner_validation_rank_behavior(candidate)

    candidate.agent_id = original_agent
    candidate.metadata["expected_direction"] *= -1
    with pytest.raises(ValueError, match="frozen behavior identity|direction"):
        evaluator._inner_validation_rank_behavior(candidate)


def test_development_reports_forecast_and_immediate_executable_labels_separately():
    panel = make_synthetic_market(days=260, symbols=25, seed=32)
    panel["limit_up"] = False
    panel["limit_down"] = False
    panel["limit_proxy_known"] = True
    split = make_split_plan(panel, label_horizon=5, entry_delay=1)
    dates = panel.index.get_level_values("date").unique().sort_values()
    symbol = panel.index.get_level_values("symbol").unique()[0]
    signal_date = split.validation_dates[5]
    signal_position = dates.get_loc(signal_date)
    entry_date = dates[signal_position + split.entry_delay]
    exit_date = dates[
        signal_position + split.entry_delay + split.label_horizon
    ]

    # A long cannot enter at an up-limit endpoint and cannot exit at a
    # down-limit endpoint.  The adjusted forecast target remains defined; the
    # execution diagnostic must disclose the missing fills.
    panel.loc[(entry_date, symbol), "limit_up"] = True
    second_symbol = panel.index.get_level_values("symbol").unique()[1]
    panel.loc[(exit_date, second_symbol), "limit_down"] = True
    development = panel.loc[
        panel.index.get_level_values("date") <= split.development_end
    ]
    evaluator = DevelopmentEvaluator(
        development,
        split,
        min_names=20,
        bootstrap_samples=10,
    )

    assert np.isfinite(evaluator.forecast_labels.loc[(signal_date, symbol)])
    assert np.isnan(evaluator.executable_labels.loc[(signal_date, symbol)])
    assert np.isfinite(
        evaluator.forecast_labels.loc[(signal_date, second_symbol)]
    )
    assert np.isnan(
        evaluator.executable_labels.loc[(signal_date, second_symbol)]
    )
    availability = evaluator.validation_endpoint_availability
    assert availability["immediately_executable_share"] < 1.0
    assert availability["not_immediately_executable_share"] > 0.0


def test_executable_label_does_not_condition_on_future_index_membership():
    tape = make_synthetic_market(days=260, symbols=25, seed=33)
    tape["limit_up"] = False
    tape["limit_down"] = False
    tape["limit_proxy_known"] = True
    tape["universe_member"] = True
    provisional_split = make_split_plan(tape, label_horizon=5, entry_delay=1)
    dates = tape.index.get_level_values("date").unique().sort_values()
    symbol = tape.index.get_level_values("symbol").unique()[0]
    signal_date = provisional_split.validation_dates[5]
    signal_position = dates.get_loc(signal_date)
    exit_date = dates[
        signal_position
        + provisional_split.entry_delay
        + provisional_split.label_horizon
    ]
    tape.loc[(exit_date, symbol), "universe_member"] = False
    tape.loc[(exit_date, symbol), "vendor_universe_member"] = False
    tape.loc[(exit_date, symbol), "research_eligible"] = False
    member_panel = tape.drop(index=(exit_date, symbol)).sort_index()
    split = make_split_plan(member_panel, label_horizon=5, entry_delay=1)
    development = member_panel.loc[
        member_panel.index.get_level_values("date") <= split.development_end
    ]
    development_tape = tape.loc[
        tape.index.get_level_values("date") <= split.development_end
    ]
    evaluator = DevelopmentEvaluator(
        development,
        split,
        feature_tape=development_tape,
        min_names=20,
        bootstrap_samples=10,
    )

    # Index deletion is a portfolio rule, not an ex-post label filter.  The
    # continuous tape still has an executable exit price, so both labels stay.
    assert evaluator.cross_sectional_mask.loc[(signal_date, symbol)]
    assert not evaluator.cross_sectional_mask.loc[(exit_date, symbol)]
    assert np.isfinite(evaluator.forecast_labels.loc[(signal_date, symbol)])
    assert np.isfinite(evaluator.executable_labels.loc[(signal_date, symbol)])


def test_continuous_union_tape_prevents_reentry_lag_from_crossing_membership_gap():
    tape = make_synthetic_market(days=240, symbols=25, seed=45)
    dates = tape.index.get_level_values("date").unique().sort_values()
    symbol = tape.index.get_level_values("symbol").unique()[0]
    gap_dates = dates[70:100]
    tape.loc[(gap_dates, symbol), "research_eligible"] = False
    member_panel = tape.drop(
        index=[(date, symbol) for date in gap_dates]
    ).sort_index()
    split = make_split_plan(
        member_panel,
        label_horizon=3,
        train_fraction=0.50,
        validation_fraction=0.20,
    )
    development = member_panel.loc[
        member_panel.index.get_level_values("date") <= split.development_end
    ]
    development_tape = tape.loc[
        tape.index.get_level_values("date") <= split.development_end
    ]
    evaluator = DevelopmentEvaluator(
        development,
        split,
        feature_tape=development_tape,
        min_names=20,
        bootstrap_samples=10,
    )
    expression = {
        "op": "return",
        "args": [{"op": "field", "name": "close"}],
        "periods": 1,
    }
    values = evaluate_expression(expression, evaluator.feature_frame, evaluator.spec)
    reentry = dates[100]
    previous_session = dates[99]
    expected = (
        evaluator.feature_frame.loc[(reentry, symbol), "close"]
        / evaluator.feature_frame.loc[(previous_session, symbol), "close"]
        - 1.0
    )
    stale_member_only = (
        member_panel.loc[(reentry, symbol), "adj_close"]
        / member_panel.loc[(dates[69], symbol), "adj_close"]
        - 1.0
    )
    assert values.loc[(reentry, symbol)] == pytest.approx(expected)
    assert values.loc[(reentry, symbol)] != pytest.approx(stale_member_only)


def test_validation_only_neutralizer_matches_full_development_path_exactly():
    panel = make_synthetic_market(days=240, symbols=30, seed=47)
    split = make_split_plan(
        panel,
        label_horizon=3,
        train_fraction=0.50,
        validation_fraction=0.20,
    )
    development = panel.loc[
        panel.index.get_level_values("date") <= split.development_end
    ]
    evaluator = DevelopmentEvaluator(
        development,
        split,
        min_names=20,
        bootstrap_samples=10,
    )
    values = evaluate_expression(
        {
            "op": "neg",
            "args": [
                {
                    "op": "return",
                    "args": [{"op": "field", "name": "close"}],
                    "periods": 3,
                }
            ],
        },
        evaluator.feature_frame,
        evaluator.spec,
        cross_sectional_mask=evaluator.cross_sectional_mask,
    ).reindex(evaluator.panel.index)
    full = cross_sectional_neutralize(
        values,
        evaluator.exposures,
        categorical=("industry",),
        min_names=evaluator.min_names,
    )
    expected = full.loc[evaluator._validation_index]
    actual = evaluator._neutralized(values)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)


def test_continuous_tape_member_mask_rejects_string_truthiness():
    tape = make_synthetic_market(days=80, symbols=20, seed=46)
    tape["research_eligible"] = True
    eligible = tape.copy()
    tape["research_eligible"] = tape["research_eligible"].astype(object)
    tape.iloc[0, tape.columns.get_loc("research_eligible")] = "False"
    with pytest.raises(ValueError, match="booleans or numeric 0/1"):
        build_cross_sectional_mask(tape, eligible)


def test_price_scale_gate_accepts_returns_and_rejects_vendor_price_level():
    panel = make_synthetic_market(days=80, symbols=20, seed=71)
    frame = build_research_feature_frame(panel)
    spec = ValidationSpec(
        allowed_fields=frozenset(frame.columns),
        max_window=252,
        max_nodes=48,
        max_depth=16,
    )
    path_expression = {
        "op": "return",
        "args": [{"op": "field", "name": "close"}],
        "periods": 5,
    }
    price_level_expression = {
        "op": "cs_rank",
        "args": [{"op": "field", "name": "close"}],
    }
    assert audit_price_scale_invariance(path_expression, frame, spec).invariant
    level_audit = audit_price_scale_invariance(price_level_expression, frame, spec)
    assert not level_audit.invariant
    assert level_audit.mismatch_rows > 0


def test_candidate_evaluation_ignores_cross_candidate_expression_cache_history():
    panel = make_synthetic_market(days=220, symbols=30, seed=171)
    split = make_split_plan(
        panel,
        label_horizon=3,
        train_fraction=0.50,
        validation_fraction=0.20,
    )
    development = panel.loc[
        panel.index.get_level_values("date") <= split.development_end
    ]
    clean = DevelopmentEvaluator(
        development,
        split,
        min_names=20,
        bootstrap_samples=10,
    )
    polluted = DevelopmentEvaluator(
        development,
        split,
        min_names=20,
        bootstrap_samples=10,
    )
    candidate = Candidate(
        expression={
            "op": "return",
            "args": [{"op": "field", "name": "close"}],
            "periods": 5,
        },
        agent_id="cache_history_test",
        hypothesis="five-day return",
        mechanism_tags=("return",),
        metadata={"expected_direction": 1},
    )
    root_key = canonical_json(candidate.expression, polluted.spec)
    polluted._full_expression_memo[root_key] = pd.Series(
        0.0, index=polluted.feature_frame.index
    )
    polluted._prefix_expression_memo[root_key] = pd.Series(
        0.0, index=polluted._prefix_feature_frame.index
    )
    polluted._scale_expression_memo[root_key] = pd.Series(
        0.0, index=polluted.price_scale_probe.index
    )
    polluted._control_ic_cache[root_key] = 999.0

    assert polluted(candidate) == clean(candidate)
    assert root_key not in polluted._control_ic_cache


def test_development_evaluator_statically_rejects_adjusted_price_anchor_signal():
    panel = make_synthetic_market(days=220, symbols=30, seed=72)
    split = make_split_plan(
        panel,
        label_horizon=3,
        train_fraction=0.50,
        validation_fraction=0.20,
    )
    development = panel.loc[
        panel.index.get_level_values("date") <= split.development_end
    ]
    evaluator = DevelopmentEvaluator(
        development,
        split,
        min_names=20,
        bootstrap_samples=10,
    )
    level = Candidate(
        expression={
            "op": "cs_rank",
            "args": [{"op": "field", "name": "close"}],
        },
        agent_id="red_team",
        hypothesis="arbitrary adjusted price level",
        mechanism_tags=("invalid_price_anchor",),
    )
    score = evaluator(level)
    assert not score.passed
    assert "not provably invariant" in score.diagnostics["rejection"]


def test_development_evaluator_and_one_time_sealed_gate():
    panel = make_synthetic_market(days=220, symbols=30, seed=10)
    split = make_split_plan(panel, label_horizon=3, train_fraction=0.50, validation_fraction=0.20)
    dev = panel.loc[panel.index.get_level_values("date") <= split.development_end]
    evaluator = DevelopmentEvaluator(dev, split, min_names=20, bootstrap_samples=30)
    candidate = TemplateCandidateFactory(seed=2).generate(1)[0]
    score = evaluator(candidate)
    assert score.passed
    assert candidate.candidate_id in evaluator.artifacts
    assert score.diagnostics["python_factor_source"].startswith("def factor(ctx):")
    assert len(score.diagnostics["python_factor_source_sha256"]) == 64
    assert score.diagnostics["python_factor_sandbox"]["network"] == "denied"

    gate = SealedGate(panel, split, min_names=20)
    result = gate.reveal([EvaluatedCandidate(candidate, score)], evaluator.artifacts)
    assert len(result) == 1
    assert result[0]["multiple_testing"]["family_size"] == 1
    assert 0.0 <= result[0]["multiple_testing"]["block_max_t_adjusted_p"] <= 1.0
    assert result[0]["forecast_label_definition"].startswith("adjusted_vwap")
    assert "sealed_immediate_executable_neutral" in result[0]
    assert result[0]["immediate_executable_multiple_testing"]["family_size"] == 1
    assert (
        0.0
        <= result[0]["immediate_executable_multiple_testing"][
            "block_max_t_adjusted_p"
        ]
        <= 1.0
    )
    assert gate.revealed
    with pytest.raises(RuntimeError, match="already"):
        gate.reveal([EvaluatedCandidate(candidate, score)], evaluator.artifacts)


def test_real_label_policy_fails_before_search_and_sealed_forecast_is_complete():
    panel = make_synthetic_market(days=220, symbols=30, seed=211)
    split = make_split_plan(
        panel, label_horizon=3, train_fraction=0.50, validation_fraction=0.20
    )
    development = panel.loc[
        panel.index.get_level_values("date") <= split.development_end
    ]
    policy = DevelopmentHandoffPolicy().to_dict()
    clean = DevelopmentEvaluator(
        development,
        split,
        min_names=20,
        development_handoff_policy=policy,
        bootstrap_samples=5,
    )
    assert clean.development_label_audit["all_development"]["valid_rows"] == (
        clean.development_label_audit["all_development"]["eligible_rows"]
    )

    all_dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    symbol = panel.index.get_level_values("symbol").unique()[0]
    train_signal = split.train_dates[0]
    train_endpoint = all_dates[
        all_dates.get_loc(train_signal) + split.entry_delay + split.label_horizon
    ]
    broken_development_tape = development.copy()
    broken_development_tape.loc[(train_endpoint, symbol), "adj_vwap"] = np.nan
    with pytest.raises(ValueError, match="coverage failed before search"):
        DevelopmentEvaluator(
            development,
            split,
            feature_tape=broken_development_tape,
            min_names=20,
            development_handoff_policy=policy,
            bootstrap_samples=5,
        )

    sealed_signal = split.sealed_dates[0]
    sealed_endpoint = all_dates[
        all_dates.get_loc(sealed_signal) + split.entry_delay + split.label_horizon
    ]
    broken_sealed_tape = panel.copy()
    broken_sealed_tape.loc[(sealed_endpoint, symbol), "adj_vwap"] = np.nan
    sealed_gate = SealedGate(
        panel,
        split,
        feature_tape=broken_sealed_tape,
        min_names=20,
    )
    with pytest.raises(ValueError, match="forecast labels are not 100% complete"):
        sealed_gate.reveal([], {})
    assert sealed_gate.revealed


def test_joint_gate_reports_23_family_three_contrasts_and_never_switches_primary():
    panel = make_synthetic_market(days=260, symbols=30, seed=109)
    split = make_split_plan(
        panel, label_horizon=3, train_fraction=0.50, validation_fraction=0.20
    )
    development = panel.loc[
        panel.index.get_level_values("date") <= split.development_end
    ]
    evaluator = DevelopmentEvaluator(
        development, split, min_names=20, bootstrap_samples=5
    )
    champions = []
    for periods in range(2, 22):
        candidate = Candidate(
            expression={
                "op": "cs_rank",
                "args": [
                    {
                        "op": "neg",
                        "args": [
                            {
                                "op": "return",
                                "args": [{"op": "field", "name": "close"}],
                                "periods": periods,
                            }
                        ],
                    }
                ],
            },
            agent_id="joint_gate_fixture",
            hypothesis=f"reversal_{periods}",
            mechanism_tags=("reversal",),
            metadata={"expected_direction": 1},
        )
        score = evaluator(candidate)
        assert score.passed
        champions.append(EvaluatedCandidate(candidate, score))

    sealed_labels = forward_return(
        build_research_feature_frame(panel)["vwap"],
        split.label_horizon,
        entry_delay=split.entry_delay,
        trading_calendar=_calendar(panel),
    )

    class FixedJointFamily:
        def audit_sealed_feature_matrix(self, features):
            assert tuple(features.columns) == tuple(
                item.candidate.candidate_id for item in champions
            )
            return {
                "schema_version": 1,
                "policy_hash": "b" * 64,
                "audit_hash": "c" * 64,
                "dates": int(features.index.get_level_values("date").nunique()),
                "date_factor_rows": int(
                    features.index.get_level_values("date").nunique() * 20
                ),
                "minimum_observed_coverage": 1.0,
                "maximum_observed_zero_impute_share": 0.0,
                "sealed_dates_dropped": 0,
                "all_passed": True,
            }

        def predict_all_for_sealed_gate(
            self, features, *, expected_composite_family_hash, claim_capability
        ):
            assert expected_composite_family_hash == "a" * 64
            assert claim_capability is sentinel_claim
            target = sealed_labels.reindex(features.index)
            return pd.DataFrame(
                {
                    "equal_weight": features.iloc[:, 0],
                    # The fixture intentionally makes Ridge superior. That is
                    # descriptive only and must never replace LightGBM primary.
                    "ridge": target,
                    "lightgbm": -target + 0.2 * features.iloc[:, 1],
                },
                index=features.index,
            )

    sentinel_claim = object()
    gate = SealedGate(
        panel,
        split,
        min_names=20,
        sealed_claim_capability=sentinel_claim,
    )
    result = gate.reveal(
        champions,
        evaluator.artifacts,
        composite_family=FixedJointFamily(),
        composite_family_hash="a" * 64,
    )
    assert len(result) == 20
    assert {item["multiple_testing"]["family_size"] for item in result} == {23}
    reveal = gate.composite_reveal
    assert reveal is not None
    assert reveal["joint_hypothesis_family_size"] == 23
    assert reveal["primary_backend_pre_registered"] == "lightgbm"
    assert reveal["backend_selection_performed"] is False
    assert reveal["sealed_result_ranking_or_winner_selection_performed"] is False
    contract = sealed_reporting_contract_v1()
    assert reveal["reporting_contract"] == contract
    assert reveal["reporting_contract_hash"] == hash_json(contract)
    composites = reveal["sealed_composite_results"]
    assert len(composites) == 3
    primary = [item for item in composites if item["eligible_for_primary_claim"]]
    assert [item["result_key"] for item in primary] == ["composite:lightgbm"]
    assert primary[0]["reporting_role"] == "unique_pre_registered_primary_composite"
    controls = [item for item in composites if not item["eligible_for_primary_claim"]]
    assert {item["result_key"] for item in controls} == {
        "composite:equal_weight",
        "composite:ridge",
    }
    assert {item["reporting_role"] for item in controls} == {
        "pre_registered_secondary_model_control"
    }
    binding = reveal["primary_result_binding"]
    assert binding["result_key"] == "composite:lightgbm"
    assert binding["endpoint_key"] == "sealed_neutral"
    assert binding["inference_key"] == "joint_multiple_testing"
    assert binding["result_record_hash"] == hash_json(primary[0])
    assert all(not item["eligible_for_primary_claim"] for item in result)
    assert {
        item["reporting_role"] for item in result
    } == {"pre_registered_secondary_component_factor_diagnostic"}
    assert all(
        item["immediate_executable_eligible_for_primary_claim"] is False
        for item in [*result, *composites]
    )
    assert {
        item["composite_subfamily_multiple_testing"]["family_size"]
        for item in composites
    } == {3}
    assert len(reveal["pre_registered_neutral_ic_contrasts"]) == 3
    assert all(
        item["reporting_role"]
        == "pre_registered_secondary_model_comparison"
        and item["eligible_for_primary_claim"] is False
        for item in reveal["pre_registered_neutral_ic_contrasts"]
    )
    assert composites[1]["sealed_neutral"]["mean"] > composites[2]["sealed_neutral"]["mean"]
    # Ridge is deliberately superior in this fixture, but the primary binding
    # remains LightGBM and the sealed table is never a winner-selection pool.
    assert binding["result_key"] != composites[1]["result_key"]
    assert reveal["sealed_feature_completeness_audit"]["sealed_dates_dropped"] == 0

    class ConstantCompositeFamily(FixedJointFamily):
        def predict_all_for_sealed_gate(
            self, features, *, expected_composite_family_hash, claim_capability
        ):
            predicted = super().predict_all_for_sealed_gate(
                features,
                expected_composite_family_hash=expected_composite_family_hash,
                claim_capability=claim_capability,
            )
            predicted["lightgbm"] = 0.0
            return predicted

    constant_gate = SealedGate(
        panel,
        split,
        min_names=20,
        sealed_claim_capability=sentinel_claim,
    )
    with pytest.raises(ValueError, match="non-finite/under-covered"):
        constant_gate.reveal(
            champions,
            evaluator.artifacts,
            composite_family=ConstantCompositeFamily(),
            composite_family_hash="a" * 64,
        )
    assert constant_gate.revealed

    execution_tape = panel.copy()
    execution_tape["tradable"] = True
    all_dates = pd.DatetimeIndex(
        execution_tape.index.get_level_values("date").unique()
    ).sort_values()
    first_signal = split.sealed_dates[0]
    entry_date = all_dates[all_dates.get_loc(first_signal) + split.entry_delay]
    symbols = execution_tape.index.get_level_values("symbol").unique()[:15]
    execution_tape.loc[(entry_date, symbols), "tradable"] = False
    undercovered_gate = SealedGate(
        panel,
        split,
        feature_tape=execution_tape,
        min_names=20,
    )
    with pytest.raises(ValueError, match="immediate-executable.*under-covered"):
        undercovered_gate.reveal([champions[0]], evaluator.artifacts)
    assert undercovered_gate.revealed


def test_sealed_gate_binds_candidate_id_and_canonical_expression_snapshot():
    panel = make_synthetic_market(days=220, symbols=30, seed=73)
    split = make_split_plan(
        panel,
        label_horizon=3,
        train_fraction=0.50,
        validation_fraction=0.20,
    )
    development = panel.loc[
        panel.index.get_level_values("date") <= split.development_end
    ]
    evaluator = DevelopmentEvaluator(
        development,
        split,
        min_names=20,
        bootstrap_samples=10,
    )
    candidate = Candidate(
        expression={
            "op": "neg",
            "args": [
                {
                    "op": "return",
                    "args": [{"op": "field", "name": "close"}],
                    "periods": 2,
                }
            ],
        },
        agent_id="integrity_test",
        hypothesis="short reversal",
        mechanism_tags=("reversal",),
        metadata={"expected_direction": 1},
    )
    score = evaluator(candidate)
    assert score.passed
    original_id = candidate.candidate_id
    artifact = evaluator.artifacts[original_id]
    assert artifact.candidate_id == original_id
    assert artifact.canonical_expression_json
    assert len(artifact.expression_hash) == 64

    candidate.expression = {
        "op": "return",
        "args": [{"op": "field", "name": "close"}],
        "periods": 3,
    }
    mutated_gate = SealedGate(panel, split, min_names=20)
    with pytest.raises(ValueError, match="expression integrity"):
        mutated_gate.reveal(
            [EvaluatedCandidate(candidate, score)],
            evaluator.artifacts,
        )

    candidate.expression = {
        "op": "neg",
        "args": [
            {
                "op": "return",
                "args": [{"op": "field", "name": "close"}],
                "periods": 2,
            }
        ],
    }
    candidate.candidate_id = "spoofed-candidate-id"
    spoofed_gate = SealedGate(panel, split, min_names=20)
    with pytest.raises(ValueError, match="id integrity"):
        spoofed_gate.reveal(
            [EvaluatedCandidate(candidate, score)],
            {candidate.candidate_id: artifact},
        )


def test_crossover_is_scored_against_control_and_parents():
    panel = make_synthetic_market(days=220, symbols=30, seed=12)
    split = make_split_plan(panel, label_horizon=3, train_fraction=0.50, validation_fraction=0.20)
    dev = panel.loc[panel.index.get_level_values("date") <= split.development_end]
    evaluator = DevelopmentEvaluator(dev, split, min_names=20, bootstrap_samples=20)
    parents = TemplateCandidateFactory(seed=5).generate(8)
    usable = []
    for parent in parents:
        score = evaluator(parent)
        if score.passed:
            usable.append(parent)
        if len(usable) == 2:
            break
    child = MechanismAwareOperators(seed=6).crossover(usable[0], usable[1], generation=1)
    child_score = evaluator(child)
    assert child_score.passed
    assert child_score.control_delta is not None
    assert child_score.diagnostics["linear_control_neutral_rank_ic"] is not None


def test_future_reference_is_rejected_closed():
    panel = make_synthetic_market(days=220, symbols=30)
    split = make_split_plan(panel, label_horizon=3, train_fraction=0.50, validation_fraction=0.20)
    dev = panel.loc[panel.index.get_level_values("date") <= split.development_end]
    evaluator = DevelopmentEvaluator(dev, split, min_names=20, bootstrap_samples=10)
    bad = Candidate(
        expression={"op": "lag", "args": [{"op": "field", "name": "close"}], "periods": -1},
        agent_id="red_team",
        hypothesis="illegal lead",
        mechanism_tags=("invalid",),
    )
    score = evaluator(bad)
    assert not score.passed
    assert score.diagnostics["rejection"].startswith("dsl_or_runtime")
