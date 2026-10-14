from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from cogalpha_plus.data_sources.baostock import (
    ADJUST_FACTOR_FIELDS,
    HS300_FIELDS,
    INDUSTRY_FIELDS,
    STOCK_BASIC_FIELDS,
    TRADE_DATE_FIELDS,
    CacheIntegrityError,
    make_request_key,
)
from cogalpha_plus.data_sources import hs300_download
from cogalpha_plus.data_sources.hs300_download import (
    HS300SourceConfig,
    _deduplicate_vendor_snapshots,
    _freeze_parquet_exact,
    _membership_request_dates,
    _validate_membership_snapshot,
    download_adjustment_factor_catalog,
    expand_membership_daily,
)


def _snapshot(update: str, prefix: int = 0) -> pd.DataFrame:
    codes = [f"sh.{prefix + value:06d}" for value in range(300)]
    return pd.DataFrame({"updateDate": update, "code": codes, "code_name": codes})


def _commit_raw_request(
    cache_dir,
    *,
    endpoint,
    args,
    fields,
    frame,
    provider_version,
):
    request_key = make_request_key(
        endpoint,
        args,
        provider_version=provider_version,
    )
    relative = f"objects/{endpoint}/{request_key}.parquet"
    object_path = cache_dir / relative
    object_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(object_path, index=False)
    state = {
        "state_schema_version": 1,
        "request_key": request_key,
        "provider": "baostock",
        "provider_version": provider_version,
        "endpoint": endpoint,
        "args": dict(args),
        "attempt_count": 1,
        "status": "success",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "rows": len(frame),
        "fields": list(fields),
        "object": relative,
        "parquet_sha256": hashlib.sha256(object_path.read_bytes()).hexdigest(),
    }
    state_path = cache_dir / "state" / f"{request_key}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return request_key, object_path


def test_provider_weekly_plan_includes_boundaries_and_mondays():
    trading = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    dates = _membership_request_dates(trading, "2020-01-02", "2020-01-09", "provider_weekly")
    assert dates == tuple(pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-09"]))


def test_offline_lineage_migration_is_idempotent_and_never_overwrites_semantics(
    tmp_path,
):
    target = tmp_path / "request_index.parquet"
    first = pd.DataFrame({"request_key": ["a" * 64], "cache_hit": [False]})
    _freeze_parquet_exact(first, target, nonsemantic_columns=("cache_hit",))
    original_hash = hashlib.sha256(target.read_bytes()).hexdigest()

    cache_hit_only = first.assign(cache_hit=True)
    _freeze_parquet_exact(
        cache_hit_only, target, nonsemantic_columns=("cache_hit",)
    )
    assert hashlib.sha256(target.read_bytes()).hexdigest() == original_hash

    changed = first.assign(request_key=["b" * 64])
    with pytest.raises(CacheIntegrityError, match="values differ from raw reconstruction"):
        _freeze_parquet_exact(
            changed, target, nonsemantic_columns=("cache_hit",)
        )


def test_snapshot_rejects_future_incomplete_and_duplicate_rows():
    valid = _validate_membership_snapshot(_snapshot("2019-12-30"), pd.Timestamp("2020-01-02"))
    assert len(valid) == 300
    with pytest.raises(ValueError, match="expected 300"):
        _validate_membership_snapshot(_snapshot("2019-12-30").iloc[:-1], pd.Timestamp("2020-01-02"))
    future = _snapshot("2020-01-06")
    with pytest.raises(ValueError, match="future"):
        _validate_membership_snapshot(future, pd.Timestamp("2020-01-02"))
    duplicate = _snapshot("2019-12-30")
    duplicate.loc[299, "code"] = duplicate.loc[0, "code"]
    with pytest.raises(ValueError, match="duplicate"):
        _validate_membership_snapshot(duplicate, pd.Timestamp("2020-01-02"))


def test_same_vendor_date_must_have_identical_membership():
    first = _validate_membership_snapshot(_snapshot("2019-12-30"), pd.Timestamp("2020-01-02"))
    same = _validate_membership_snapshot(_snapshot("2019-12-30"), pd.Timestamp("2020-01-03"))
    assert len(_deduplicate_vendor_snapshots([first, same])) == 300
    changed = _validate_membership_snapshot(_snapshot("2019-12-30", prefix=1000), pd.Timestamp("2020-01-03"))
    with pytest.raises(ValueError, match="two different sets"):
        _deduplicate_vendor_snapshots([first, changed])


def test_daily_expansion_uses_latest_nonfuture_snapshot():
    first = _validate_membership_snapshot(_snapshot("2019-12-30"), pd.Timestamp("2020-01-02"))
    second = _validate_membership_snapshot(_snapshot("2020-01-06", prefix=1000), pd.Timestamp("2020-01-06"))
    snapshots = _deduplicate_vendor_snapshots([first, second])
    days = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    daily = expand_membership_daily(snapshots, days)
    assert daily.groupby("date")["code"].nunique().eq(300).all()
    assert daily.loc[daily["date"] == pd.Timestamp("2020-01-03"), "code"].str.startswith("sh.000").all()
    assert daily.loc[daily["date"] == pd.Timestamp("2020-01-06"), "code"].str.startswith("sh.001").all()


def test_daily_expansion_fails_before_first_known_snapshot():
    snapshot = _validate_membership_snapshot(_snapshot("2020-01-06"), pd.Timestamp("2020-01-06"))
    with pytest.raises(ValueError, match="calendar_precedes_first_snapshot"):
        expand_membership_daily(snapshot, pd.DatetimeIndex([pd.Timestamp("2020-01-03")]))


def test_daily_expansion_rejects_a_147_day_stale_membership_snapshot():
    snapshot = _validate_membership_snapshot(
        _snapshot("2020-01-06"), pd.Timestamp("2020-01-06")
    )
    calendar = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-06-01")]
    )
    with pytest.raises(ValueError, match="membership_snapshot_stale"):
        expand_membership_daily(
            snapshot,
            calendar,
            maximum_snapshot_gap_days=200,
            maximum_snapshot_age_days=21,
        )


def test_adjustment_factor_catalog_commits_one_serial_request_per_union_name(
    monkeypatch, tmp_path
):
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    pd.DataFrame(
        {"code": ["sh.600001", "sz.000001"], "code_name": ["A", "B"]}
    ).to_parquet(normalized / "security_union.parquet", index=False)

    calls = []

    class FakeSession:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query_adjustment_factors(
            self, code, start_date, end_date, *, allow_empty
        ):
            calls.append((code, start_date, end_date, allow_empty))
            if code == "sh.600001":
                records = [
                    [code, "2018-06-15", "0.8", "1.25", "1.0"]
                ]
            else:
                records = []
            frame = pd.DataFrame(records, columns=ADJUST_FACTOR_FIELDS, dtype=object)
            frame.attrs["baostock_cache"] = {
                "request_key": hashlib.sha256(code.encode()).hexdigest(),
                "cache_hit": False,
            }
            return frame

    monkeypatch.setattr(hs300_download, "BaoStockSession", FakeSession)
    config = HS300SourceConfig(
        cache_dir=tmp_path / "raw",
        normalized_dir=normalized,
        start_date="1990-01-01",
        end_date="2025-12-31",
    )
    summary = download_adjustment_factor_catalog(config)

    assert calls == [
        ("sh.600001", "1990-01-01", "2025-12-31", True),
        ("sz.000001", "1990-01-01", "2025-12-31", True),
    ]
    assert summary["complete"] is True
    assert summary["adjustment_factor_requests"] == 2
    assert summary["event_rows"] == 1
    index = pd.read_parquet(normalized / "adjustment_request_index.parquet")
    assert index["code"].tolist() == ["sh.600001", "sz.000001"]
    assert index["rows"].tolist() == [1, 0]
    manifest = json.loads(
        (normalized / "adjustment_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["file"]["sha256"] == summary["file"]["sha256"]


def test_membership_rebuild_selects_frozen_key_when_provider_versions_coexist(
    tmp_path,
):
    cache_dir = tmp_path / "raw"
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    config = HS300SourceConfig(
        cache_dir=cache_dir,
        normalized_dir=normalized,
        start_date="2020-01-02",
        end_date="2020-01-02",
    )
    calendar_args = {"start_date": config.start_date, "end_date": config.end_date}
    calendar = pd.DataFrame(
        [["2020-01-02", "1"]],
        columns=TRADE_DATE_FIELDS,
    )
    membership_args = {"date": "2020-01-02"}
    membership = _snapshot("2020-01-02")
    selected_calendar, _ = _commit_raw_request(
        cache_dir,
        endpoint="trade_dates",
        args=calendar_args,
        fields=TRADE_DATE_FIELDS,
        frame=calendar,
        provider_version="selected-v1",
    )
    selected_membership, _ = _commit_raw_request(
        cache_dir,
        endpoint="hs300_stocks",
        args=membership_args,
        fields=HS300_FIELDS,
        frame=membership,
        provider_version="selected-v1",
    )
    _commit_raw_request(
        cache_dir,
        endpoint="trade_dates",
        args=calendar_args,
        fields=TRADE_DATE_FIELDS,
        frame=calendar,
        provider_version="coexisting-v2",
    )
    _commit_raw_request(
        cache_dir,
        endpoint="hs300_stocks",
        args=membership_args,
        fields=HS300_FIELDS,
        frame=membership,
        provider_version="coexisting-v2",
    )
    source_index = pd.DataFrame(
        [
            {
                "endpoint": "trade_dates",
                "requested_date": "",
                "start_date": config.start_date,
                "end_date": config.end_date,
                "vendor_update_date": "",
                "rows": 1,
                "request_key": selected_calendar,
                "cache_hit": False,
            },
            {
                "endpoint": "hs300_stocks",
                "requested_date": "2020-01-02",
                "start_date": "",
                "end_date": "",
                "vendor_update_date": "2020-01-02",
                "rows": 300,
                "request_key": selected_membership,
                "cache_hit": False,
            },
        ],
        columns=hs300_download._MEMBERSHIP_REQUEST_INDEX_COLUMNS,
    ).sort_values(["endpoint", "requested_date"], kind="stable").reset_index(drop=True)
    source_index.to_parquet(
        normalized / "membership_request_index.parquet",
        index=False,
    )

    summary = hs300_download.rebuild_membership_catalog_from_cache(config)

    assert summary["complete"] is True
    manifest_path = normalized / "membership_manifest.json"
    first_manifest = manifest_path.read_bytes()
    second_summary = hs300_download._publish_membership_manifest(
        {**summary, "created_at": "2099-01-01T00:00:00+00:00"},
        manifest_path,
    )
    assert manifest_path.read_bytes() == first_manifest
    assert second_summary["created_at"] == summary["created_at"]
    rebuilt = pd.read_parquet(normalized / "membership_request_index.parquet")
    assert set(rebuilt["request_key"]) == {
        selected_calendar,
        selected_membership,
    }


def test_security_rebuild_uses_selected_key_and_never_falls_back(tmp_path):
    cache_dir = tmp_path / "raw"
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    codes = ["sh.600000", "sz.000001"]
    pd.DataFrame({"code": codes}).to_parquet(
        normalized / "security_union.parquet",
        index=False,
    )
    config = HS300SourceConfig(cache_dir=cache_dir, normalized_dir=normalized)
    args = {"code": "", "code_name": ""}
    frame = pd.DataFrame(
        [
            ["sh.600000", "A", "1990-01-01", "", "1", "1"],
            ["sz.000001", "B", "1991-01-01", "", "1", "1"],
        ],
        columns=STOCK_BASIC_FIELDS,
    )
    selected_key, selected_object = _commit_raw_request(
        cache_dir,
        endpoint="stock_basic",
        args=args,
        fields=STOCK_BASIC_FIELDS,
        frame=frame,
        provider_version="selected-v1",
    )
    _commit_raw_request(
        cache_dir,
        endpoint="stock_basic",
        args=args,
        fields=STOCK_BASIC_FIELDS,
        frame=frame,
        provider_version="coexisting-v2",
    )
    pd.DataFrame(
        [["stock_basic", "", "", 2, selected_key, False]],
        columns=hs300_download._SECURITY_REQUEST_INDEX_COLUMNS,
    ).to_parquet(
        normalized / "security_master_request_index.parquet",
        index=False,
    )

    summary = hs300_download.rebuild_security_master_from_cache(config)
    assert summary["complete"] is True
    assert pd.read_parquet(
        normalized / "security_master_request_index.parquet"
    )["request_key"].tolist() == [selected_key]

    selected_object.write_bytes(b"corrupt selected object")
    with pytest.raises(CacheIntegrityError, match="cache object hash mismatch"):
        hs300_download.rebuild_security_master_from_cache(config)


def test_industry_rebuild_selects_frozen_key_when_provider_versions_coexist(
    tmp_path,
):
    cache_dir = tmp_path / "raw"
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    requested = "2020-01-06"
    membership = pd.DataFrame(
        {
            "requested_date": pd.to_datetime([requested]),
            "vendor_update_date": pd.to_datetime([requested]),
            "code": ["sh.600000"],
            "code_name": ["A"],
        }
    )
    membership.to_parquet(normalized / "membership_snapshots.parquet", index=False)
    pd.DataFrame(
        {"endpoint": ["hs300_stocks"], "requested_date": [requested]}
    ).to_parquet(normalized / "membership_request_index.parquet", index=False)
    pd.DataFrame({"code": ["sh.600000"]}).to_parquet(
        normalized / "security_union.parquet",
        index=False,
    )
    args = {"code": "", "date": requested}
    frame = pd.DataFrame(
        [[requested, "sh.600000", "A", "Industry A", "CSRC"]],
        columns=INDUSTRY_FIELDS,
    )
    selected_key, selected_object = _commit_raw_request(
        cache_dir,
        endpoint="stock_industry",
        args=args,
        fields=INDUSTRY_FIELDS,
        frame=frame,
        provider_version="selected-v1",
    )
    _commit_raw_request(
        cache_dir,
        endpoint="stock_industry",
        args=args,
        fields=INDUSTRY_FIELDS,
        frame=frame,
        provider_version="coexisting-v2",
    )
    pd.DataFrame(
        [["stock_industry", requested, requested, 1, selected_key, False]],
        columns=hs300_download._INDUSTRY_REQUEST_INDEX_COLUMNS,
    ).to_parquet(normalized / "industry_request_index.parquet", index=False)
    config = HS300SourceConfig(cache_dir=cache_dir, normalized_dir=normalized)

    summary = hs300_download.rebuild_industry_catalog_from_cache(config)
    assert summary["complete"] is True
    assert pd.read_parquet(normalized / "industry_request_index.parquet")[
        "request_key"
    ].tolist() == [selected_key]

    selected_object.write_bytes(b"corrupt selected object")
    with pytest.raises(CacheIntegrityError, match="cache object hash mismatch"):
        hs300_download.rebuild_industry_catalog_from_cache(config)


def test_price_catalog_refuses_a_second_cache_writer(monkeypatch, tmp_path):
    lock_calls = []

    def contested(_descriptor, operation):
        lock_calls.append(operation)
        if operation & hs300_download.fcntl.LOCK_NB:
            raise BlockingIOError("busy")

    monkeypatch.setattr(hs300_download.fcntl, "flock", contested)
    config = HS300SourceConfig(
        cache_dir=tmp_path / "raw",
        normalized_dir=tmp_path / "normalized",
    )

    with pytest.raises(
        hs300_download.PriceCatalogAlreadyRunning,
        match="writer lock is already held",
    ):
        hs300_download.download_price_catalog(config)

    assert lock_calls == [
        hs300_download.fcntl.LOCK_EX | hs300_download.fcntl.LOCK_NB
    ]


def test_price_pair_validation_rejects_one_sided_provider_truncation():
    raw = pd.DataFrame(
        {"date": ["2020-01-02", "2020-01-03"], "code": ["A", "A"]}
    )
    adjusted = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "code": ["A", "A", "A"],
        }
    )

    with pytest.raises(CacheIntegrityError, match="different dates"):
        hs300_download._validate_downloaded_price_pair("A", raw, adjusted)


@pytest.mark.parametrize(
    "download",
    [
        hs300_download.download_membership_catalog,
        hs300_download.rebuild_membership_catalog_from_cache,
        hs300_download.rebuild_security_master_from_cache,
        hs300_download.download_adjustment_factor_catalog,
        hs300_download.download_industry_catalog,
        hs300_download.rebuild_industry_catalog_from_cache,
    ],
)
def test_all_catalog_writers_share_the_price_writer_lock(
    monkeypatch, tmp_path, download
):
    """Direct CLIs cannot race prices, each other, or a watchdog retry."""

    lock_calls = []

    def contested(_descriptor, operation):
        lock_calls.append(operation)
        if operation & hs300_download.fcntl.LOCK_NB:
            raise BlockingIOError("busy")

    monkeypatch.setattr(hs300_download.fcntl, "flock", contested)
    config = HS300SourceConfig(
        cache_dir=tmp_path / "raw",
        normalized_dir=tmp_path / "normalized",
    )

    with pytest.raises(
        hs300_download.PriceCatalogAlreadyRunning,
        match="raw-catalog writer lock is already held",
    ):
        download(config)

    assert lock_calls == [
        hs300_download.fcntl.LOCK_EX | hs300_download.fcntl.LOCK_NB
    ]
