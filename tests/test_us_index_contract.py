import pandas as pd
import pytest

from cogalpha_plus.data_sources.us_index_contract import (
    FractionalTreatment,
    MembershipAction,
    MembershipEvent,
    MembershipReplayError,
    MembershipSeed,
    PermanentSecurity,
    SecurityIdentityCatalog,
    TickerAlias,
    USCorporateActionCatalog,
    USCorporateActionContractError,
    USCorporateActionEvent,
    USCorporateActionKind,
    USCorporateActionLeg,
    USCorporateActionLegKind,
    USIndexContractError,
    USIndexMembershipTimeline,
    USIndexTapeCoverageError,
    US_INDEX_EVENT_DAY_ORDER_CONTRACT_V1,
    US_INDEX_EVENT_DAY_ORDER_SHA256,
    US_INDEX_EVENT_DAY_ORDER_V1,
    UnknownSecurityIdentifierError,
    validate_us_index_universe_views,
)


def _identity_catalog(*ids: str) -> SecurityIdentityCatalog:
    securities = [
        PermanentSecurity(
            security_id=security_id,
            id_scheme="unit_test_permanent_id",
            issuer_id=f"issuer-{security_id}",
            source="unit-test-security-master",
        )
        for security_id in ids
    ]
    aliases = [
        TickerAlias(
            security_id=security_id,
            ticker=f"T{position}",
            valid_from="2020-01-02",
            valid_to_exclusive=None,
            known_at="2019-12-31T15:00:00-05:00",
            source="unit-test-alias-history",
        )
        for position, security_id in enumerate(ids)
    ]
    return SecurityIdentityCatalog(securities, aliases)


def _membership_timeline() -> tuple[SecurityIdentityCatalog, USIndexMembershipTimeline]:
    identities = _identity_catalog("sec-A", "sec-B")
    seed = MembershipSeed(
        index_id="SP500",
        member_security_ids=("sec-A",),
        known_at="2023-12-29T16:00:00-05:00",
        effective_at="2024-01-02T09:30:00-05:00",
        source="unit-test-initial-snapshot",
    )
    events = (
        MembershipEvent(
            event_id="add-B",
            index_id="SP500",
            security_id="sec-B",
            action=MembershipAction.ADD,
            announcement_at="2024-01-02T16:05:00-05:00",
            known_at="2024-01-02T16:06:00-05:00",
            effective_at="2024-01-03T09:30:00-05:00",
            source="unit-test-index-notice",
            reason="replacement",
        ),
        MembershipEvent(
            event_id="remove-A",
            index_id="SP500",
            security_id="sec-A",
            action="REMOVE",
            announcement_at="2024-01-02T16:05:00-05:00",
            known_at="2024-01-02T16:06:00-05:00",
            effective_at="2024-01-04T09:30:00-05:00",
            source="unit-test-index-notice",
            reason="acquisition",
        ),
    )
    return identities, USIndexMembershipTimeline(identities, seed, events)


def _session_decisions() -> dict[pd.Timestamp, str]:
    return {
        pd.Timestamp("2024-01-02"): "2024-01-02T16:00:00-05:00",
        pd.Timestamp("2024-01-03"): "2024-01-03T16:00:00-05:00",
        pd.Timestamp("2024-01-04"): "2024-01-04T16:00:00-05:00",
        pd.Timestamp("2024-01-05"): "2024-01-05T16:00:00-05:00",
    }


def test_ticker_alias_reuse_resolves_to_permanent_security_by_date():
    securities = [
        PermanentSecurity("sec-old", "perm", "unit-test", "issuer-old"),
        PermanentSecurity("sec-new", "perm", "unit-test", "issuer-new"),
    ]
    aliases = [
        TickerAlias(
            "sec-old",
            "XYZ",
            "2010-01-04",
            "2020-01-02",
            "2009-12-31T16:00:00-05:00",
            "unit-test",
        ),
        TickerAlias(
            "sec-new",
            "XYZ",
            "2020-01-02",
            None,
            "2019-12-31T16:00:00-05:00",
            "unit-test",
        ),
    ]
    catalog = SecurityIdentityCatalog(securities, aliases)
    assert catalog.resolve_ticker(
        "xyz", "2019-12-31", decision_at="2019-12-31T16:00:00-05:00"
    ) == "sec-old"
    assert catalog.resolve_ticker(
        "XYZ", "2020-01-02", decision_at="2020-01-02T16:00:00-05:00"
    ) == "sec-new"


def test_overlapping_ticker_reuse_is_rejected_instead_of_stitching_history():
    securities = [
        PermanentSecurity("sec-old", "perm", "unit-test"),
        PermanentSecurity("sec-new", "perm", "unit-test"),
    ]
    aliases = [
        TickerAlias(
            "sec-old",
            "XYZ",
            "2010-01-04",
            "2020-02-01",
            "2009-12-31T16:00:00-05:00",
            "unit-test",
        ),
        TickerAlias(
            "sec-new",
            "XYZ",
            "2020-01-02",
            None,
            "2019-12-31T16:00:00-05:00",
            "unit-test",
        ),
    ]
    with pytest.raises(USIndexContractError, match="overlapping ticker alias"):
        SecurityIdentityCatalog(securities, aliases)


def test_ticker_alias_must_be_known_by_first_valid_session():
    with pytest.raises(USIndexContractError, match="not known"):
        TickerAlias(
            "sec-A",
            "AAA",
            "2024-01-02",
            None,
            "2024-01-03T10:00:00-05:00",
            "unit-test",
        )


def test_membership_replay_uses_announcement_known_and_effective_times():
    _, timeline = _membership_timeline()
    assert timeline.members_at("2024-01-02T16:00:00-05:00") == {"sec-A"}
    assert timeline.members_at("2024-01-03T16:00:00-05:00") == {
        "sec-A",
        "sec-B",
    }
    assert timeline.members_at("2024-01-04T16:00:00-05:00") == {"sec-B"}
    expanded = timeline.expand_membership(_session_decisions())
    assert bool(expanded.loc[(pd.Timestamp("2024-01-02"), "sec-A")])
    assert not bool(expanded.loc[(pd.Timestamp("2024-01-02"), "sec-B")])
    assert bool(expanded.loc[(pd.Timestamp("2024-01-03"), "sec-B")])
    assert not bool(expanded.loc[(pd.Timestamp("2024-01-04"), "sec-A")])


def test_membership_event_known_after_effective_fails_closed():
    with pytest.raises(MembershipReplayError, match="announcement_at <= known_at"):
        MembershipEvent(
            event_id="late-event",
            index_id="SP500",
            security_id="sec-A",
            action="ADD",
            announcement_at="2024-01-02T16:00:00-05:00",
            known_at="2024-01-04T16:00:00-05:00",
            effective_at="2024-01-03T09:30:00-05:00",
            source="unit-test",
        )


def test_membership_expansion_rejects_future_session_decision_timestamp():
    _, timeline = _membership_timeline()
    decisions = _session_decisions()
    decisions[pd.Timestamp("2024-01-02")] = "2024-01-03T16:00:00-05:00"
    with pytest.raises(MembershipReplayError, match="different US market date"):
        timeline.expand_membership(decisions)


def test_membership_event_rejects_ticker_in_place_of_permanent_id():
    identities, timeline = _membership_timeline()
    seed = timeline.seed
    event = MembershipEvent(
        event_id="ticker-key",
        index_id="SP500",
        security_id="T0",
        action="REMOVE",
        announcement_at="2024-01-04T16:00:00-05:00",
        known_at="2024-01-04T16:01:00-05:00",
        effective_at="2024-01-05T09:30:00-05:00",
        source="unit-test",
    )
    with pytest.raises(UnknownSecurityIdentifierError, match="unknown permanent"):
        USIndexMembershipTimeline(identities, seed, [event])


def _cash_leg() -> USCorporateActionLeg:
    return USCorporateActionLeg(
        leg_id="cash-consideration",
        kind=USCorporateActionLegKind.CASH,
        settlement_at="2024-01-12T09:30:00-05:00",
        known_at="2024-01-02T16:00:00-05:00",
        source="unit-test-merger-terms",
        cash_per_source_share=2.0,
        currency="USD",
    )


def _stock_leg() -> USCorporateActionLeg:
    return USCorporateActionLeg(
        leg_id="stock-consideration",
        kind="security",
        settlement_at="2024-01-12T09:30:00-05:00",
        known_at="2024-01-02T16:00:00-05:00",
        source="unit-test-merger-terms",
        destination_security_id="sec-B",
        shares_per_source_share=0.25,
        fractional_treatment=FractionalTreatment.CASH_IN_LIEU,
    )


def _cil_leg() -> USCorporateActionLeg:
    return USCorporateActionLeg(
        leg_id="fractional-settlement",
        kind="fractional_cash_in_lieu",
        settlement_at="2024-01-12T09:30:00-05:00",
        known_at="2024-01-11T16:00:00-05:00",
        source="unit-test-agent-sale",
        destination_security_id="sec-B",
        currency="USD",
        applies_to_leg_id="stock-consideration",
        cash_per_fractional_destination_share=40.0,
    )


def _mixed_merger(*legs: USCorporateActionLeg) -> USCorporateActionEvent:
    return USCorporateActionEvent(
        event_id="sec-A-merger",
        kind=USCorporateActionKind.MERGER,
        source_security_id="sec-A",
        announcement_at="2024-01-02T15:55:00-05:00",
        known_at="2024-01-02T16:00:00-05:00",
        effective_at="2024-01-10T09:30:00-05:00",
        source_security_terminates=True,
        source="unit-test-merger-terms",
        legs=tuple(legs),
    )


def test_mixed_merger_books_cash_stock_and_fractional_cash_in_lieu():
    identities = _identity_catalog("sec-A", "sec-B")
    event = _mixed_merger(_cash_leg(), _stock_leg(), _cil_leg())
    catalog = USCorporateActionCatalog(identities, [event])
    entitlement = event.entitlements(10)
    assert entitlement.source_security_terminates is True
    assert entitlement.security_quantities == {"sec-B": 2.0}
    assert entitlement.cash_by_currency == {"USD": pytest.approx(40.0)}
    assert catalog.terminal_session_dates() == {
        "sec-A": pd.Timestamp("2024-01-10")
    }


def test_security_leg_requiring_cash_in_lieu_cannot_omit_settlement_leg():
    with pytest.raises(
        USCorporateActionContractError, match="fractional terms are incomplete"
    ):
        _mixed_merger(_cash_leg(), _stock_leg())


def test_spinoff_preserves_parent_and_delivers_child_security():
    identities = _identity_catalog("sec-parent", "sec-child")
    stock = USCorporateActionLeg(
        leg_id="child-distribution",
        kind="security",
        settlement_at="2024-02-02T09:30:00-05:00",
        known_at="2024-01-12T16:00:00-05:00",
        source="unit-test-spinoff-terms",
        destination_security_id="sec-child",
        shares_per_source_share=0.2,
        fractional_treatment="deliver_fractional",
    )
    event = USCorporateActionEvent(
        event_id="parent-spin",
        kind="spinoff",
        source_security_id="sec-parent",
        announcement_at="2024-01-12T15:55:00-05:00",
        known_at="2024-01-12T16:00:00-05:00",
        effective_at="2024-02-01T09:30:00-05:00",
        source_security_terminates=False,
        source="unit-test-spinoff-terms",
        legs=(stock,),
    )
    USCorporateActionCatalog(identities, [event])
    entitlement = event.entitlements(7)
    assert entitlement.source_security_terminates is False
    assert entitlement.security_quantities["sec-child"] == pytest.approx(1.4)


def test_spinoff_cannot_silently_terminate_parent():
    stock = USCorporateActionLeg(
        leg_id="child-distribution",
        kind="security",
        settlement_at="2024-02-02T09:30:00-05:00",
        known_at="2024-01-12T16:00:00-05:00",
        source="unit-test",
        destination_security_id="sec-child",
        shares_per_source_share=0.2,
        fractional_treatment="deliver_fractional",
    )
    with pytest.raises(
        USCorporateActionContractError, match="termination is inconsistent"
    ):
        USCorporateActionEvent(
            event_id="bad-spin",
            kind="spinoff",
            source_security_id="sec-parent",
            announcement_at="2024-01-12T15:55:00-05:00",
            known_at="2024-01-12T16:00:00-05:00",
            effective_at="2024-02-01T09:30:00-05:00",
            source_security_terminates=True,
            source="unit-test",
            legs=(stock,),
        )


def test_terminal_event_cannot_use_an_after_close_effective_phase():
    with pytest.raises(
        USCorporateActionContractError, match="opening event phase"
    ):
        USCorporateActionEvent(
            event_id="bad-phase",
            kind="merger",
            source_security_id="sec-A",
            announcement_at="2024-01-02T15:55:00-05:00",
            known_at="2024-01-02T16:00:00-05:00",
            effective_at="2024-01-10T16:05:00-05:00",
            source_security_terminates=True,
            source="unit-test",
            legs=(_cash_leg(),),
            effective_phase="after_close",
        )


def _universe_views():
    identities, timeline = _membership_timeline()
    membership = timeline.expand_membership(_session_decisions())
    sessions = pd.DatetimeIndex(_session_decisions())
    full_index = membership.index
    feature = pd.DataFrame({"observed_bar": True}, index=full_index)
    execution = pd.DataFrame(
        {"valuation_known": True},
        index=pd.MultiIndex.from_product(
            [sessions, ["sec-A", "sec-B"]], names=["date", "security_id"]
        ),
    )
    return identities, timeline, sessions, membership, feature, execution


def test_new_entrant_cannot_enter_cross_section_before_effective_add():
    identities, timeline, sessions, membership, feature, execution = _universe_views()
    selection = membership.copy()
    selection.loc[(pd.Timestamp("2024-01-02"), "sec-B")] = True
    with pytest.raises(USIndexTapeCoverageError, match="pre-entry"):
        validate_us_index_universe_views(
            identities,
            timeline,
            membership,
            selection,
            feature,
            execution,
            trading_sessions=sessions,
            session_decision_times=_session_decisions(),
        )


def test_declared_membership_must_equal_event_replay():
    identities, timeline, sessions, membership, feature, execution = _universe_views()
    forged = membership.copy()
    forged.loc[(pd.Timestamp("2024-01-02"), "sec-B")] = True
    with pytest.raises(MembershipReplayError, match="differs from ADD/REMOVE"):
        validate_us_index_universe_views(
            identities,
            timeline,
            forged,
            forged.copy(),
            feature,
            execution,
            trading_sessions=sessions,
            session_decision_times=_session_decisions(),
        )


def test_removed_security_is_retained_on_execution_tape_through_end():
    identities, timeline, sessions, membership, feature, execution = _universe_views()
    report = validate_us_index_universe_views(
        identities,
        timeline,
        membership,
        membership.copy(),
        feature,
        execution,
        trading_sessions=sessions,
        session_decision_times=_session_decisions(),
    )
    assert report.new_entrant_security_ids == ("sec-B",)
    assert report.removed_security_ids == ("sec-A",)
    assert report.execution_required_rows == 7

    incomplete = execution.drop(index=(pd.Timestamp("2024-01-05"), "sec-A"))
    with pytest.raises(USIndexTapeCoverageError, match="first missing"):
        validate_us_index_universe_views(
            identities,
            timeline,
            membership,
            membership.copy(),
            feature,
            incomplete,
            trading_sessions=sessions,
            session_decision_times=_session_decisions(),
        )


def test_opening_terminal_event_ends_execution_retention_before_that_session():
    identities, timeline, sessions, membership, feature, execution = _universe_views()
    terminal_execution = execution.drop(
        index=[
            (pd.Timestamp("2024-01-04"), "sec-A"),
            (pd.Timestamp("2024-01-05"), "sec-A"),
        ]
    )
    cash_out = USCorporateActionEvent(
        event_id="sec-A-cash-out",
        kind="merger",
        source_security_id="sec-A",
        announcement_at="2024-01-02T15:55:00-05:00",
        known_at="2024-01-02T16:00:00-05:00",
        effective_at="2024-01-04T09:30:00-05:00",
        source_security_terminates=True,
        source="unit-test-cash-merger",
        legs=(
            USCorporateActionLeg(
                leg_id="cash-out",
                kind="cash",
                settlement_at="2024-01-04T09:30:00-05:00",
                known_at="2024-01-02T16:00:00-05:00",
                source="unit-test-cash-merger",
                cash_per_source_share=10.0,
                currency="USD",
            ),
        ),
    )
    actions = USCorporateActionCatalog(identities, [cash_out])
    report = validate_us_index_universe_views(
        identities,
        timeline,
        membership,
        membership.copy(),
        feature,
        terminal_execution,
        trading_sessions=sessions,
        session_decision_times=_session_decisions(),
        corporate_actions=actions,
    )
    assert report.terminal_security_ids == ("sec-A",)
    assert report.execution_required_rows == 5


def test_unknown_execution_valuation_fails_closed():
    identities, timeline, sessions, membership, feature, execution = _universe_views()
    execution.loc[(pd.Timestamp("2024-01-05"), "sec-A"), "valuation_known"] = False
    with pytest.raises(USIndexTapeCoverageError, match="unknown valuation"):
        validate_us_index_universe_views(
            identities,
            timeline,
            membership,
            membership.copy(),
            feature,
            execution,
            trading_sessions=sessions,
            session_decision_times=_session_decisions(),
        )


def test_event_day_order_is_frozen_and_content_addressed():
    assert US_INDEX_EVENT_DAY_ORDER_V1 == (
        "opening_corporate_actions",
        "effective_membership_and_benchmark",
        "raw_vwap_execution",
        "close_valuation",
    )
    assert dict(US_INDEX_EVENT_DAY_ORDER_CONTRACT_V1) == {
        "schema_version": 1,
        "market_timezone": "America/New_York",
        "steps": US_INDEX_EVENT_DAY_ORDER_V1,
    }
    assert len(US_INDEX_EVENT_DAY_ORDER_SHA256) == 64
    int(US_INDEX_EVENT_DAY_ORDER_SHA256, 16)
    assert US_INDEX_EVENT_DAY_ORDER_SHA256 == (
        "0e327476a45ca6774fac132c36e2287a8c2a263c48ff18f05832e569480cbcb1"
    )
