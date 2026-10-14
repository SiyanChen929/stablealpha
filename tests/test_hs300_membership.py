import pandas as pd
import pytest

from cogalpha_plus.data_sources.hs300 import (
    attach_membership,
    expand_membership_snapshots,
    validate_membership_snapshots,
)


def snapshots_fixture():
    return pd.DataFrame(
        {
            "updateDate": ["2020-01-06"] * 3 + ["2020-01-13"] * 3,
            "code": ["A", "B", "C", "B", "C", "D"],
            "code_name": ["a", "b", "c", "b", "c", "d"],
        }
    )


def test_snapshot_completeness_and_backward_asof_expansion():
    calendar = pd.bdate_range("2020-01-06", "2020-01-17")
    report = validate_membership_snapshots(
        snapshots_fixture(), expected_members=3, trading_dates=calendar
    )
    assert report.valid
    assert report.counts_by_snapshot == {"2020-01-06": 3, "2020-01-13": 3}

    expanded = expand_membership_snapshots(
        snapshots_fixture(), calendar, expected_members=3
    )
    assert len(expanded) == len(calendar) * 3
    assert set(expanded.xs("2020-01-10", level="date").index) == {"A", "B", "C"}
    assert set(expanded.xs("2020-01-13", level="date").index) == {"B", "C", "D"}
    assert "D" not in expanded.xs("2020-01-10", level="date").index
    assert expanded.loc[(pd.Timestamp("2020-01-10"), "A"), "membership_snapshot_date"] == pd.Timestamp(
        "2020-01-06"
    )

    downloader_shape = snapshots_fixture().rename(columns={"updateDate": "vendor_update_date"})
    downloader_report = validate_membership_snapshots(
        downloader_shape, expected_members=3, trading_dates=calendar
    )
    assert downloader_report.valid


def test_bad_member_count_duplicates_and_stale_snapshots_fail_closed():
    incomplete = snapshots_fixture().iloc[:-1]
    report = validate_membership_snapshots(incomplete, expected_members=3)
    assert not report.valid
    assert any(item.startswith("snapshot_member_count_not_3") for item in report.errors)
    with pytest.raises(ValueError, match="member_count"):
        expand_membership_snapshots(
            incomplete,
            pd.bdate_range("2020-01-06", "2020-01-17"),
            expected_members=3,
        )

    duplicate = snapshots_fixture().copy()
    duplicate.loc[2, "code"] = "A"
    report = validate_membership_snapshots(duplicate, expected_members=3)
    assert not report.valid
    assert any(item.startswith("duplicate_symbols") for item in report.errors)

    stale = snapshots_fixture().query("updateDate == '2020-01-06'")
    report = validate_membership_snapshots(
        stale,
        expected_members=3,
        trading_dates=["2020-02-03"],
        max_snapshot_age_days=21,
    )
    assert not report.valid
    assert any(item.startswith("membership_snapshot_stale") for item in report.errors)


def test_attach_membership_marks_union_without_backfilling_future_constituents():
    dates = pd.bdate_range("2020-01-10", "2020-01-13", name="date")
    symbols = pd.Index(["A", "B", "C", "D"], name="symbol")
    index = pd.MultiIndex.from_product([dates, symbols])
    market = pd.DataFrame({"close": 10.0}, index=index)
    expanded = expand_membership_snapshots(
        snapshots_fixture(), dates, expected_members=3
    )
    attached = attach_membership(market, expanded)
    assert not bool(attached.loc[(pd.Timestamp("2020-01-10"), "D"), "universe_member"])
    assert bool(attached.loc[(pd.Timestamp("2020-01-13"), "D"), "universe_member"])
    assert not bool(attached.loc[(pd.Timestamp("2020-01-13"), "A"), "universe_member"])
