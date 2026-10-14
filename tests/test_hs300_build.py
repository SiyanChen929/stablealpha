from __future__ import annotations

from dataclasses import replace
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cogalpha_plus.data_sources.baostock import (
    ADJUST_FACTOR_FIELDS,
    CacheIntegrityError,
    EQUITY_DAILY_FIELDS,
    HS300_FIELDS,
    INDUSTRY_FIELDS,
    INDEX_DAILY_FIELDS,
    RawRequestCache,
    STOCK_BASIC_FIELDS,
    TRADE_DATE_FIELDS,
)
from cogalpha_plus.data_sources.hs300_download import (
    _materialize_industry_catalog,
    _membership_request_dates,
)
from cogalpha_plus.data_sources.hs300_build import (
    HS300BuildConfig,
    _derive_research_eligibility,
    _industry_asof,
    build_hs300_research_panel,
    derive_conservative_one_price_limits,
)
from cogalpha_plus.data_sources.hs300_audit import (
    HS300AuditConfig,
    audit_hs300_research_panel,
)
from cogalpha_plus.runner import panel_snapshot_hash


SYMBOLS = ("sh.600001", "sh.600002", "sz.300001")


def test_research_eligibility_requires_current_risk_and_capacity_inputs():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "sh.600001")],
        names=["date", "symbol"],
    )
    features = pd.DataFrame(
        {
            "tradable": [True],
            "is_st": [False],
            "listed_days": [500],
            "log_mcap": [22.0],
            "beta": [np.nan],
            "adv20_dollars": [10_000_000.0],
        },
        index=index,
    )
    limits = pd.DataFrame(
        {
            "research_close_limit_up_proxy": [False],
            "research_close_limit_down_proxy": [False],
            "research_close_limit_known_proxy": [True],
        },
        index=index,
    )

    eligible = _derive_research_eligibility(features, limits, index)
    assert not bool(eligible.iloc[0])

    features.loc[index[0], "beta"] = 1.0
    assert bool(_derive_research_eligibility(features, limits, index).iloc[0])

    features.loc[index[0], "adv20_dollars"] = 0.0
    assert not bool(_derive_research_eligibility(features, limits, index).iloc[0])


def test_industry_freshness_uses_successful_observation_not_vendor_effective_date():
    membership = pd.DataFrame(
        {
            "requested_date": [pd.Timestamp("2020-01-02")],
            "vendor_update_date": [pd.Timestamp("2020-01-02")],
            "code": ["sh.600001"],
            "code_name": ["A"],
        }
    )
    requests = []
    for number, requested in enumerate(
        pd.to_datetime(["2020-01-02", "2020-04-01"]), 1
    ):
        raw = pd.DataFrame(
            {
                "updateDate": ["2020-01-02"],
                "code": ["sh.600001"],
                "code_name": ["A"],
                "industry": ["Industry A"],
                "industryClassification": ["CSRC"],
            },
            columns=INDUSTRY_FIELDS,
        )
        raw.attrs["baostock_cache"] = {
            "request_key": f"{number:x}" * 64,
            "cache_hit": False,
        }
        requests.append((requested, raw))

    catalog, request_index = _materialize_industry_catalog(
        membership,
        ["sh.600001"],
        requests,
        expected_request_dates=pd.to_datetime(["2020-01-02", "2020-04-01"]),
    )

    assert len(catalog) == 2
    assert catalog["vendor_update_date"].nunique() == 1
    assert len(request_index) == 2
    member_index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2020-04-01"), "sh.600001")],
        names=["date", "symbol"],
    )
    joined = _industry_asof(member_index, catalog, maximum_age_days=35)
    assert joined.iloc[0]["industry_snapshot_date"] == pd.Timestamp("2020-04-01")


def test_industry_catalog_ignores_observation_only_members_outside_daily_union():
    membership = pd.DataFrame(
        {
            "requested_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "vendor_update_date": pd.to_datetime(["2019-12-30", "2019-12-30"]),
            "code": ["sh.600001", "sh.600099"],
            "code_name": ["Daily member", "Superseded before next session"],
        }
    )
    raw = pd.DataFrame(
        [
            [
                "2019-12-30",
                "sh.600001",
                "Daily member",
                "Industry A",
                "CSRC",
            ]
        ],
        columns=INDUSTRY_FIELDS,
    )
    raw.attrs["baostock_cache"] = {
        "request_key": "a" * 64,
        "cache_hit": False,
    }

    catalog, _ = _materialize_industry_catalog(
        membership,
        ["sh.600001"],
        [(pd.Timestamp("2019-12-30"), raw)],
        expected_request_dates=pd.to_datetime(["2019-12-30"]),
    )

    assert catalog["code"].tolist() == ["sh.600001"]


def _equity_frame(code, dates, closes, *, adjustflag, factor, limit_date=None):
    closes = np.asarray(closes, dtype=float)
    preclose = np.r_[closes[0] / 1.005, closes[:-1]]
    open_ = closes * 0.999
    high = closes * 1.006
    low = closes * 0.994
    if limit_date is not None:
        location = dates.get_loc(limit_date)
        # A one-price +10% limit day still uses the actual previous cash close;
        # fabricating a different preclose would look like a corporate action.
        closes[location] = np.floor(preclose[location] * 1.10 * 100.0 + 0.5) / 100.0
        open_[location] = high[location] = low[location] = closes[location]
        if location + 1 < len(preclose):
            preclose[location + 1] = closes[location]
    volume = 1_000.0 + 10.0 * np.arange(len(dates))
    amount = volume * closes
    scale = factor if adjustflag == "1" else 1.0
    values = {
        "date": [str(date.date()) for date in dates],
        "code": [code] * len(dates),
        "open": open_ * scale,
        "high": high * scale,
        "low": low * scale,
        "close": closes * scale,
        "preclose": preclose * scale,
        "volume": volume,
        "amount": amount,
        "adjustflag": [adjustflag] * len(dates),
        "turn": ["1.0"] * len(dates),
        "tradestatus": ["1"] * len(dates),
        "pctChg": (closes / preclose - 1.0) * 100.0,
        "peTTM": ["12.0"] * len(dates),
        "pbMRQ": ["1.2"] * len(dates),
        "psTTM": ["1.0"] * len(dates),
        "pcfNcfTTM": ["8.0"] * len(dates),
        "isST": ["0"] * len(dates),
    }
    return pd.DataFrame({field: values[field] for field in EQUITY_DAILY_FIELDS}, dtype=object)


def _index_frame(dates, closes):
    closes = np.asarray(closes, dtype=float)
    previous = np.r_[closes[0], closes[:-1]]
    values = {
        "date": [str(date.date()) for date in dates],
        "code": ["sh.000300"] * len(dates),
        "open": closes * 0.999,
        "high": closes * 1.004,
        "low": closes * 0.996,
        "close": closes,
        "preclose": previous,
        "volume": np.full(len(dates), 1_000_000.0),
        "amount": closes * 1_000_000.0,
        "pctChg": (closes / previous - 1.0) * 100.0,
    }
    return pd.DataFrame({field: values[field] for field in INDEX_DAILY_FIELDS}, dtype=object)


def _write_fixture(root: Path, *, start: str = "2020-01-02"):
    cache_dir = root / "cache"
    normalized_dir = root / "normalized"
    output_dir = root / "output"
    normalized_dir.mkdir(parents=True)
    dates = pd.bdate_range(start, periods=18)
    benchmark_returns = np.array(
        [0.0, 0.01, -0.005, 0.014, -0.009, 0.007, -0.004, 0.011, -0.006] * 2
    )
    index_close = 100.0 * np.cumprod(1.0 + benchmark_returns)

    cache = RawRequestCache(cache_dir, provider_version="0.9.3-test")
    start, end = str(dates[0].date()), str(dates[-1].date())

    calendar_days = pd.date_range(dates[0], dates[-1], freq="D")
    trading_set = set(dates)
    calendar_raw = pd.DataFrame(
        {
            "calendar_date": [str(date.date()) for date in calendar_days],
            "is_trading_day": ["1" if date in trading_set else "0" for date in calendar_days],
        },
        columns=TRADE_DATE_FIELDS,
        dtype=object,
    )
    calendar_stored = cache.get_or_fetch(
        "trade_dates",
        {"start_date": start, "end_date": end},
        expected_fields=TRADE_DATE_FIELDS,
        fetcher=lambda: calendar_raw,
    )
    calendar_key = calendar_stored.attrs["baostock_cache"]["request_key"]
    calendar_normalized = calendar_raw.copy()
    calendar_normalized["date"] = pd.to_datetime(calendar_normalized["calendar_date"])
    calendar_normalized.to_parquet(
        normalized_dir / "trade_calendar.parquet", index=False
    )

    membership_rows = []
    change_date = dates[11]
    for date in dates:
        members = SYMBOLS[:2] if date < change_date else SYMBOLS[1:]
        vendor = dates[0] if date < change_date else change_date
        for code in members:
            membership_rows.append(
                {
                    "date": date,
                    "code": code,
                    "code_name": code,
                    "vendor_update_date": vendor,
                }
            )
    membership_daily = pd.DataFrame(membership_rows)
    membership_daily.to_parquet(
        normalized_dir / "membership_daily.parquet", index=False
    )
    requested_dates = _membership_request_dates(
        dates, start, end, "provider_weekly"
    )
    membership_request_rows = [
        {
            "endpoint": "trade_dates",
            "requested_date": "",
            "start_date": start,
            "end_date": end,
            "vendor_update_date": "",
            "rows": len(calendar_raw),
            "request_key": calendar_key,
            "cache_hit": False,
        }
    ]
    native_membership: dict[pd.Timestamp, pd.DataFrame] = {}
    for requested in requested_dates:
        vendor = dates[0] if requested < change_date else change_date
        members = SYMBOLS[:2] if vendor < change_date else SYMBOLS[1:]
        raw_snapshot = pd.DataFrame(
            {
                "updateDate": [str(vendor.date())] * 2,
                "code": members,
                "code_name": members,
            },
            columns=HS300_FIELDS,
            dtype=object,
        )
        stored_snapshot = cache.get_or_fetch(
            "hs300_stocks",
            {"date": str(requested.date())},
            expected_fields=HS300_FIELDS,
            fetcher=lambda frame=raw_snapshot: frame,
        )
        membership_request_rows.append(
            {
                "endpoint": "hs300_stocks",
                "requested_date": str(requested.date()),
                "start_date": "",
                "end_date": "",
                "vendor_update_date": str(vendor.date()),
                "rows": len(raw_snapshot),
                "request_key": stored_snapshot.attrs["baostock_cache"]["request_key"],
                "cache_hit": False,
            }
        )
        native_membership.setdefault(
            vendor,
            pd.DataFrame(
                {
                    "requested_date": [requested] * 2,
                    "vendor_update_date": [vendor] * 2,
                    "code": members,
                    "code_name": members,
                }
            ),
        )
    membership_snapshots = pd.concat(native_membership.values(), ignore_index=True)
    membership_snapshots.sort_values(
        ["vendor_update_date", "code"]
    ).to_parquet(normalized_dir / "membership_snapshots.parquet", index=False)
    pd.DataFrame(membership_request_rows).sort_values(
        ["endpoint", "requested_date"], kind="stable"
    ).to_parquet(normalized_dir / "membership_request_index.parquet", index=False)
    security_union = (
        membership_daily.groupby("code", as_index=False)
        .agg(
            code_name=("code_name", "last"),
            first_member_date=("date", "min"),
            last_member_date=("date", "max"),
            member_sessions=("date", "size"),
        )
        .sort_values("code")
    )
    security_union.to_parquet(normalized_dir / "security_union.parquet", index=False)

    security_master = pd.DataFrame(
        {
            "code": SYMBOLS,
            "code_name": SYMBOLS,
            "ipoDate": ["2018-01-01", "2018-01-01", "2018-01-01"],
            "outDate": ["", "", ""],
            "type": ["1", "1", "1"],
            "status": ["1", "1", "1"],
        },
        columns=STOCK_BASIC_FIELDS,
        dtype=object,
    )
    stored_basic = cache.get_or_fetch(
        "stock_basic",
        {"code": "", "code_name": ""},
        expected_fields=STOCK_BASIC_FIELDS,
        fetcher=lambda: security_master,
    )
    security_master.to_parquet(
        normalized_dir / "security_master.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "endpoint": "stock_basic",
                "code": "",
                "code_name": "",
                "rows": len(security_master),
                "request_key": stored_basic.attrs["baostock_cache"]["request_key"],
                "cache_hit": False,
            }
        ]
    ).to_parquet(
        normalized_dir / "security_master_request_index.parquet", index=False
    )
    industry = []
    industry_requests = []
    industry_request_dates = sorted(set(requested_dates).union({dates[0], change_date}))
    for requested in industry_request_dates:
        vendor = dates[0] if requested < change_date else change_date
        labels = (
            ("OldA", "OldB", "Growth")
            if requested < change_date
            else ("OldA", "NewB", "Growth")
        )
        raw_industry = pd.DataFrame(
            {
                "updateDate": [str(vendor.date())] * len(SYMBOLS),
                "code": SYMBOLS,
                "code_name": SYMBOLS,
                "industry": labels,
                "industryClassification": ["CSRC"] * len(SYMBOLS),
            },
            columns=INDUSTRY_FIELDS,
            dtype=object,
        )
        stored_industry = cache.get_or_fetch(
            "stock_industry",
            {"code": "", "date": str(requested.date())},
            expected_fields=INDUSTRY_FIELDS,
            fetcher=lambda frame=raw_industry: frame,
        )
        industry_requests.append(
            {
                "endpoint": "stock_industry",
                "requested_date": str(requested.date()),
                "vendor_update_date": str(vendor.date()),
                "rows": len(raw_industry),
                "request_key": stored_industry.attrs["baostock_cache"]["request_key"],
                "cache_hit": False,
            }
        )
        for code, label in zip(SYMBOLS, labels, strict=True):
            industry.append(
                {
                    "requested_date": requested,
                    "vendor_update_date": vendor,
                    "code": code,
                    "code_name": code,
                    "industry": label,
                    "industryClassification": "CSRC",
                }
            )
    pd.DataFrame(industry).to_parquet(normalized_dir / "industry_snapshots.parquet", index=False)
    pd.DataFrame(industry_requests).to_parquet(
        normalized_dir / "industry_request_index.parquet", index=False
    )

    request_rows = []
    raw_keys = []
    for number, code in enumerate(SYMBOLS):
        asset_returns = (0.8 + 0.2 * number) * benchmark_returns
        close = (10.0 + number) * np.cumprod(1.0 + asset_returns)
        limit_date = dates[10] if code == "sh.600002" else None
        for flag in ("3", "1"):
            args = {
                "code": code,
                "fields": ",".join(EQUITY_DAILY_FIELDS),
                "start_date": start,
                "end_date": end,
                "frequency": "d",
                "adjustflag": flag,
            }
            frame = _equity_frame(
                code,
                dates,
                close.copy(),
                adjustflag=flag,
                factor=1.25,
                limit_date=limit_date,
            )
            stored = cache.get_or_fetch(
                "history_k_data",
                args,
                expected_fields=EQUITY_DAILY_FIELDS,
                fetcher=lambda frame=frame: frame,
            )
            key = stored.attrs["baostock_cache"]["request_key"]
            if flag == "3":
                raw_keys.append(key)
            request_rows.append(
                {
                    "code": code,
                    "adjustflag": flag,
                    "basis": "raw_unadjusted" if flag == "3" else "backward_adjusted",
                    "rows": len(frame),
                    "first_date": start,
                    "last_date": end,
                    "request_key": key,
                    "cache_hit": False,
                }
            )
    pd.DataFrame(request_rows).to_parquet(
        normalized_dir / "price_request_index.parquet", index=False
    )

    adjustment_rows = []
    for code in SYMBOLS:
        factor_data = pd.DataFrame(
            [
                {
                    "code": code,
                    "dividOperateDate": start,
                    "foreAdjustFactor": "0.8",
                    "backAdjustFactor": "1.25",
                    "adjustFactor": "1.0",
                }
            ],
            columns=ADJUST_FACTOR_FIELDS,
            dtype=object,
        )
        factor_args = {
            "code": code,
            "start_date": "1990-01-01",
            "end_date": end,
        }
        stored_factor = cache.get_or_fetch(
            "adjust_factor",
            factor_args,
            expected_fields=ADJUST_FACTOR_FIELDS,
            fetcher=lambda frame=factor_data: frame,
        )
        factor_key = stored_factor.attrs["baostock_cache"]["request_key"]
        adjustment_rows.append(
            {
                "code": code,
                "rows": 1,
                "first_event_date": start,
                "last_event_date": start,
                "request_key": factor_key,
                "cache_hit": False,
            }
        )
    pd.DataFrame(adjustment_rows).to_parquet(
        normalized_dir / "adjustment_request_index.parquet", index=False
    )

    index_args = {
        "code": "sh.000300",
        "fields": ",".join(INDEX_DAILY_FIELDS),
        "start_date": start,
        "end_date": end,
        "frequency": "d",
        "adjustflag": "3",
    }
    index_data = _index_frame(dates, index_close)
    stored_index = cache.get_or_fetch(
        "index_history",
        index_args,
        expected_fields=INDEX_DAILY_FIELDS,
        fetcher=lambda: index_data,
    )
    index_key = stored_index.attrs["baostock_cache"]["request_key"]
    (normalized_dir / "price_manifest.json").write_text(
        json.dumps({"index_request": {"request_key": index_key}}), encoding="utf-8"
    )
    config = HS300BuildConfig(
        cache_dir=cache_dir,
        normalized_dir=normalized_dir,
        output_dir=output_dir,
        research_start=str(dates[8].date()),
        research_end=str(dates[-1].date()),
        expected_members=2,
        adv_window=3,
        adv_min_periods=3,
        beta_window=4,
        beta_min_periods=3,
        contract_minimum_history_days=8,
        minimum_limit_rule_listed_days=30,
        maximum_industry_age_days=35,
        membership_source_start_date=start,
        membership_source_end_date=end,
        membership_frequency="provider_weekly",
    )
    return config, dates, change_date, raw_keys


def test_build_validates_and_freezes_point_in_time_research_panel(tmp_path):
    config, dates, change_date, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    panel = result.panel
    tape = result.feature_tape

    assert result.validation.valid
    assert tape is not None
    assert len(tape) > len(panel)
    assert tape.index.get_level_values("date").min() < panel.index.get_level_values("date").min()
    in_research_span = tape.loc[
        tape.index.get_level_values("date") >= panel.index.get_level_values("date").min()
    ]
    assert in_research_span["universe_member"].groupby(level="date").sum().eq(2).all()
    assert panel.index.get_level_values("date").nunique() == 10
    assert panel.groupby(level="date").size().eq(2).all()
    assert set(panel["universe_member"]) == {True}
    assert "sz.300001" not in panel.xs(dates[10], level="date").index
    assert "sz.300001" in panel.xs(change_date, level="date").index
    assert panel.loc[(dates[10], "sh.600002"), "industry"] == "OldB"
    assert panel.loc[(change_date, "sh.600002"), "industry"] == "NewB"
    assert panel.loc[(dates[10], "sh.600002"), "industry_snapshot_date"] <= dates[10]

    # Rolling features retain pre-research history rather than restarting at
    # research_start.
    first_research = panel.index.get_level_values("date").min()
    assert panel.xs(first_research, level="date")["adv20_dollars"].notna().all()
    assert panel.xs(first_research, level="date")["beta"].notna().all()
    assert bool(panel.loc[(dates[10], "sh.600002"), "limit_up"])
    assert bool(panel.loc[(dates[10], "sh.600002"), "limit_proxy_known"])
    assert panel.loc[(dates[10], "sh.600002"), "limit_up_quality"] == "research_proxy"

    sums = panel["benchmark_weight"].groupby(level="date").sum()
    assert np.allclose(sums, 1.0)
    assert set(panel["benchmark_weight_quality"]) == {"research_proxy"}
    assert result.validation.metadata["dataset_tier"] == "research_proxy"
    assert any("not_portfolio_grade" in warning for warning in result.validation.warnings)

    assert result.manifest_path.is_file()
    assert result.manifest_sha256_path.is_file()
    assert result.partition_paths
    assert result.feature_tape_partition_paths
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["snapshot_hash"] == result.validation.snapshot_hash
    assert manifest["feature_tape"]["rows"] == len(tape)
    assert manifest["feature_tape"]["complete"] is True
    assert manifest["feature_tape"]["index_policy"] == (
        "raw_cache_union_after_strict_pre_membership_adjusted_lineage_boundaries"
    )
    assert manifest["feature_tape"]["columns"] == list(tape.columns)
    assert set(manifest["feature_tape"]["dtypes"]) == set(tape.columns)
    assert manifest["feature_tape"]["source_contract"]["price_cache_pairs"] == len(
        SYMBOLS
    )
    assert len(
        manifest["feature_tape"]["source_contract"][
            "price_cache_objects_sha256"
        ]
    ) == 64
    for record in manifest["partitions"]:
        path = Path(record["path"])
        assert path.is_file()
        assert len(pd.read_parquet(path)) == record["rows"]


def test_cache_hash_tampering_fails_before_panel_construction(tmp_path):
    config, _, _, raw_keys = _write_fixture(tmp_path)
    state_path = config.cache_dir / "state" / f"{raw_keys[0]}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["parquet_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(CacheIntegrityError, match="hash mismatch"):
        build_hs300_research_panel(config)


def test_cache_request_key_is_recomputed_from_adjustment_args(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    requests = pd.read_parquet(
        config.normalized_dir / "adjustment_request_index.parquet"
    )
    key = str(requests.iloc[0]["request_key"])
    state_path = config.cache_dir / "state" / f"{key}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["args"]["end_date"] = "2099-12-31"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(CacheIntegrityError, match="not canonical"):
        build_hs300_research_panel(config)


def test_membership_cache_state_args_cannot_be_resealed_under_old_key(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    requests = pd.read_parquet(
        config.normalized_dir / "membership_request_index.parquet"
    )
    row = requests.loc[requests["endpoint"].astype(str).eq("hs300_stocks")].iloc[0]
    key = str(row["request_key"])
    state_path = config.cache_dir / "state" / f"{key}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["args"]["date"] = "2099-12-31"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(CacheIntegrityError, match="not canonical"):
        build_hs300_research_panel(config)


def test_offline_audit_recomputes_cache_key_and_exact_date_policy(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    record = result.manifest["adjustment_cache_objects"][0]
    key = str(record["request_key"])
    state_path = config.cache_dir / "state" / f"{key}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["args"]["end_date"] = "2099-12-31"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )
    assert not report.valid
    assert "cache_state_contract_mismatch" in {
        finding.code for finding in report.findings
    }


def test_future_entrant_membership_reseal_without_raw_snapshot_fails(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    path = config.normalized_dir / "membership_daily.parquet"
    daily = pd.read_parquet(path)
    attacked = daily["date"].eq(dates[2]) & daily["code"].eq(SYMBOLS[0])
    daily.loc[attacked, ["code", "code_name"]] = SYMBOLS[2]
    daily.to_parquet(path, index=False)

    with pytest.raises(CacheIntegrityError, match="membership_daily differs"):
        build_hs300_research_panel(config)


def test_deleted_or_changed_membership_snapshot_fails_raw_replay(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    path = config.normalized_dir / "membership_snapshots.parquet"
    snapshots = pd.read_parquet(path)
    snapshots = snapshots.iloc[1:].reset_index(drop=True)
    snapshots.to_parquet(path, index=False)

    with pytest.raises(CacheIntegrityError, match="membership_snapshots differs"):
        build_hs300_research_panel(config)


def test_security_master_and_industry_cannot_be_coordinately_resealed(tmp_path):
    security_config, _, _, _ = _write_fixture(tmp_path / "security")
    security_path = security_config.normalized_dir / "security_master.parquet"
    security = pd.read_parquet(security_path)
    security.loc[security["code"].eq(SYMBOLS[0]), "ipoDate"] = "2099-01-01"
    security.to_parquet(security_path, index=False)
    with pytest.raises(CacheIntegrityError, match="security_master differs"):
        build_hs300_research_panel(security_config)

    industry_config, _, _, _ = _write_fixture(tmp_path / "industry")
    industry_path = industry_config.normalized_dir / "industry_snapshots.parquet"
    industry = pd.read_parquet(industry_path)
    industry.loc[industry.index[0], "industry"] = "Invented"
    industry.to_parquet(industry_path, index=False)
    with pytest.raises(CacheIntegrityError, match="industry_snapshots differs"):
        build_hs300_research_panel(industry_config)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _rewrite_price_cache(
    config: HS300BuildConfig,
    code: str,
    flag: str,
    rewrite,
) -> None:
    request_path = config.normalized_dir / "price_request_index.parquet"
    requests = pd.read_parquet(request_path)
    selected = requests["code"].astype(str).eq(code) & requests[
        "adjustflag"
    ].astype(str).eq(flag)
    request_key = str(requests.loc[selected, "request_key"].iloc[0])
    state_path = config.cache_dir / "state" / f"{request_key}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    object_path = config.cache_dir / str(state["object"])
    frame = rewrite(pd.read_parquet(object_path).copy()).reset_index(drop=True)
    frame.to_parquet(object_path, index=False)
    state["rows"] = len(frame)
    state["parquet_sha256"] = _file_sha256(object_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    requests.loc[selected, "rows"] = len(frame)
    requests.loc[selected, "first_date"] = str(
        pd.Timestamp(frame["date"].min()).date()
    )
    requests.loc[selected, "last_date"] = str(
        pd.Timestamp(frame["date"].max()).date()
    )
    requests.to_parquet(request_path, index=False)


def _rewrite_security_cache(config: HS300BuildConfig, rewrite) -> None:
    request_path = config.normalized_dir / "security_master_request_index.parquet"
    requests = pd.read_parquet(request_path)
    selected = requests["code"].fillna("").astype(str).eq("") & requests[
        "code_name"
    ].fillna("").astype(str).eq("")
    request_key = str(requests.loc[selected, "request_key"].iloc[0])
    state_path = config.cache_dir / "state" / f"{request_key}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    object_path = config.cache_dir / str(state["object"])
    frame = rewrite(pd.read_parquet(object_path).copy()).reset_index(drop=True)
    frame.to_parquet(object_path, index=False)
    state["rows"] = len(frame)
    state["parquet_sha256"] = _file_sha256(object_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    requests.loc[selected, "rows"] = len(frame)
    requests.to_parquet(request_path, index=False)
    frame.loc[frame["code"].astype(str).isin(SYMBOLS)].sort_values("code").to_parquet(
        config.normalized_dir / "security_master.parquet", index=False
    )


def _make_terminal_stale_fixture(config: HS300BuildConfig, dates) -> None:
    terminal = dates[8]

    def terminate(security):
        selected = security["code"].astype(str).eq(SYMBOLS[0])
        security.loc[selected, "outDate"] = str(terminal.date())
        security.loc[selected, "status"] = "0"
        return security

    _rewrite_security_cache(config, terminate)
    for flag in ("1", "3"):
        _rewrite_price_cache(
            config,
            SYMBOLS[0],
            flag,
            lambda frame, terminal=terminal: frame.loc[
                pd.to_datetime(frame["date"]) <= terminal
            ],
        )


def test_terminal_stale_membership_is_explicit_and_never_filled(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    _make_terminal_stale_fixture(config, dates)

    result = build_hs300_research_panel(config)
    sidecar = result.terminal_stale_membership_exceptions

    assert sidecar is not None
    expected = pd.MultiIndex.from_product(
        [dates[9:11], [SYMBOLS[0]]], names=["date", "symbol"]
    )
    assert sidecar.index.equals(expected)
    assert sidecar["vendor_universe_member"].eq(True).all()
    assert sidecar["research_eligible"].eq(False).all()
    assert set(sidecar["price_fill_policy"]) == {"none"}
    assert set(sidecar["terminal_return_policy"]) == {
        "unresolved_no_imputation"
    }
    assert result.feature_tape is not None
    assert not len(result.feature_tape.index.intersection(expected))
    assert result.panel.groupby(level="date").size().loc[dates[9:11]].eq(1).all()
    assert result.manifest["terminal_stale_membership_exceptions"]["rows"] == 2
    assert audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    ).valid


def test_terminal_out_date_without_a_bar_accepts_only_prior_market_session(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    terminal = dates[8]

    def terminate(security):
        selected = security["code"].astype(str).eq(SYMBOLS[0])
        security.loc[selected, "outDate"] = str(terminal.date())
        security.loc[selected, "status"] = "0"
        return security

    _rewrite_security_cache(config, terminate)
    for flag in ("1", "3"):
        _rewrite_price_cache(
            config,
            SYMBOLS[0],
            flag,
            lambda frame, terminal=terminal: frame.loc[
                pd.to_datetime(frame["date"]) < terminal
            ],
        )

    result = build_hs300_research_panel(config)

    expected = pd.MultiIndex.from_product(
        [dates[8:11], [SYMBOLS[0]]], names=["date", "symbol"]
    )
    assert result.terminal_stale_membership_exceptions.index.equals(expected)
    assert audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    ).valid


def test_missing_member_bar_without_terminal_proof_remains_hard_failure(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    missing_date = dates[9]
    for flag in ("1", "3"):
        _rewrite_price_cache(
            config,
            SYMBOLS[1],
            flag,
            lambda frame, missing_date=missing_date: frame.loc[
                pd.to_datetime(frame["date"]) != missing_date
            ],
        )

    with pytest.raises(ValueError, match="non-terminal-stale"):
        build_hs300_research_panel(config)


def test_research_eligibility_is_current_time_only_and_distinct_from_membership(
    tmp_path,
):
    config, dates, _, _ = _write_fixture(tmp_path / "base")
    base = build_hs300_research_panel(config)
    assert base.feature_tape is not None
    tape = base.feature_tape
    limit_key = (dates[10], SYMBOLS[1])
    assert bool(tape.loc[limit_key, "vendor_universe_member"])
    assert not bool(tape.loc[limit_key, "research_eligible"])
    nonmember_key = (dates[2], SYMBOLS[2])
    assert not bool(tape.loc[nonmember_key, "vendor_universe_member"])
    assert not bool(tape.loc[nonmember_key, "research_eligible"])

    second_config, _, _, _ = _write_fixture(tmp_path / "future_outdate")
    _rewrite_security_cache(
        second_config,
        lambda security: security.assign(
            outDate=np.where(
                security["code"].astype(str).eq(SYMBOLS[1]),
                "2099-12-31",
                security["outDate"],
            )
        ),
    )
    second = build_hs300_research_panel(second_config)
    assert second.feature_tape is not None
    pd.testing.assert_series_equal(
        tape["research_eligible"],
        second.feature_tape["research_eligible"],
    )
    policy = second.manifest["research_eligibility"]
    assert policy["current_time_only"] is True
    assert policy["future_endpoint_availability_filter"] is False
    assert policy["required_finite_fields"] == [
        "log_mcap",
        "beta",
        "adv20_dollars",
    ]
    assert policy["missing_current_risk_or_capacity_policy"] == (
        "ineligible_fail_closed"
    )
    assert len(policy["snapshot_hash"]) == 64


def test_vendor_adjustment_event_transition_mismatch_fails_build(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    request_path = config.normalized_dir / "adjustment_request_index.parquet"
    requests = pd.read_parquet(request_path)
    selected = requests["code"].astype(str).eq(SYMBOLS[0])
    request_key = str(requests.loc[selected, "request_key"].iloc[0])
    state_path = config.cache_dir / "state" / f"{request_key}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    object_path = config.cache_dir / str(state["object"])
    events = pd.read_parquet(object_path)
    attack = events.iloc[[0]].copy()
    attack["dividOperateDate"] = str(dates[5].date())
    attack["backAdjustFactor"] = "1.30"
    events = pd.concat([events, attack], ignore_index=True)
    events.to_parquet(object_path, index=False)
    state["rows"] = len(events)
    state["parquet_sha256"] = _file_sha256(object_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    requests.loc[selected, "rows"] = len(events)
    requests.loc[selected, "last_event_date"] = str(dates[5].date())
    requests.to_parquet(request_path, index=False)

    with pytest.raises(ValueError, match="vendor adjustment-event as-of audit failed"):
        build_hs300_research_panel(config)


def test_close_limit_research_proxy_is_separate_from_one_price_execution_proxy():
    rows = [
        # Closed at +10% after intraday trading: research exclusion only.
        ("2021-01-04", "sh.600001", 10.50, 11.00, 10.40, 11.00, 10.00, False),
        # Touched +10% intraday but did not close there: neither proxy.
        ("2021-01-05", "sh.600001", 10.60, 11.00, 10.50, 10.80, 10.00, False),
        # Main-board ST +5% rule.
        ("2021-01-04", "sh.600002", 10.20, 10.50, 10.10, 10.50, 10.00, True),
        # STAR +20% rule.
        ("2021-01-04", "sh.688001", 11.00, 12.00, 10.80, 12.00, 10.00, False),
        # GEM before and on the 2020-08-24 reform boundary.
        ("2020-08-21", "sz.300001", 10.50, 11.00, 10.20, 11.00, 10.00, False),
        ("2020-08-24", "sz.300001", 11.00, 12.00, 10.90, 12.00, 10.00, False),
        # Half-up tick rounding: 10.05 * 1.10 = 11.055 -> 11.06.
        ("2021-01-04", "sh.600003", 10.80, 11.06, 10.70, 11.06, 10.05, False),
        # A true one-price down-limit is both an execution lock and close limit.
        ("2021-01-04", "sh.600004", 9.00, 9.00, 9.00, 9.00, 10.00, False),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "is_st",
        ],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["listed_days"] = 500
    frame["tradable"] = True
    frame = frame.set_index(["date", "symbol"]).sort_index()

    limits = derive_conservative_one_price_limits(frame)

    assert bool(
        limits.loc[(pd.Timestamp("2021-01-04"), "sh.600001"), "research_close_limit_up_proxy"]
    )
    assert not bool(limits.loc[(pd.Timestamp("2021-01-04"), "sh.600001"), "limit_up"])
    assert not bool(
        limits.loc[(pd.Timestamp("2021-01-05"), "sh.600001"), "research_close_limit_up_proxy"]
    )
    for key in (
        (pd.Timestamp("2021-01-04"), "sh.600002"),
        (pd.Timestamp("2021-01-04"), "sh.688001"),
        (pd.Timestamp("2020-08-21"), "sz.300001"),
        (pd.Timestamp("2020-08-24"), "sz.300001"),
        (pd.Timestamp("2021-01-04"), "sh.600003"),
    ):
        assert bool(limits.loc[key, "research_close_limit_up_proxy"])
    down_key = (pd.Timestamp("2021-01-04"), "sh.600004")
    assert bool(limits.loc[down_key, "research_close_limit_down_proxy"])
    assert bool(limits.loc[down_key, "limit_down"])


def test_non_one_price_close_limit_is_excluded_from_research_mask(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    limit_date = dates[10]

    def make_intraday_traded(frame):
        selected = pd.to_datetime(frame["date"]).eq(limit_date)
        close = pd.to_numeric(frame.loc[selected, "close"], errors="raise")
        frame.loc[selected, "open"] = close.to_numpy() * 0.99
        frame.loc[selected, "low"] = close.to_numpy() * 0.985
        frame.loc[selected, "high"] = close.to_numpy()
        return frame

    for flag in ("1", "3"):
        _rewrite_price_cache(
            config, SYMBOLS[1], flag, make_intraday_traded
        )

    result = build_hs300_research_panel(config)
    key = (limit_date, SYMBOLS[1])

    assert not bool(result.panel.loc[key, "limit_up"])
    assert bool(result.panel.loc[key, "research_close_limit_up_proxy"])
    assert not bool(result.panel.loc[key, "research_eligible"])


def test_st_suspension_and_less_than_one_year_only_change_current_eligibility(
    tmp_path,
):
    config, dates, _, _ = _write_fixture(tmp_path)
    st_date = dates[9]
    suspended_date = dates[8]

    def mark_st(frame):
        selected = pd.to_datetime(frame["date"]).eq(st_date)
        frame.loc[selected, "isST"] = "1"
        return frame

    def mark_suspended(frame):
        selected = pd.to_datetime(frame["date"]).eq(suspended_date)
        frame.loc[selected, "tradestatus"] = "0"
        frame.loc[selected, "volume"] = 0.0
        frame.loc[selected, "amount"] = 0.0
        return frame

    _rewrite_price_cache(config, SYMBOLS[0], "3", mark_st)
    _rewrite_price_cache(config, SYMBOLS[1], "3", mark_suspended)
    def recent_ipo(security):
        security.loc[
            security["code"].astype(str).eq(SYMBOLS[2]), "ipoDate"
        ] = str((dates[0] - pd.Timedelta(days=100)).date())
        return security

    _rewrite_security_cache(config, recent_ipo)

    result = build_hs300_research_panel(config)
    tape = result.feature_tape
    assert tape is not None

    assert not bool(tape.loc[(st_date, SYMBOLS[0]), "research_eligible"])
    assert bool(tape.loc[(dates[8], SYMBOLS[0]), "research_eligible"])
    assert not bool(
        tape.loc[(suspended_date, SYMBOLS[1]), "research_eligible"]
    )
    assert bool(tape.loc[(dates[9], SYMBOLS[1]), "research_eligible"])
    symbol2_member = tape.loc[
        (tape.index.get_level_values("symbol").astype(str) == SYMBOLS[2])
        & tape["vendor_universe_member"].to_numpy(dtype=bool)
    ]
    assert len(symbol2_member)
    assert not symbol2_member["research_eligible"].any()


def _rewrite_frozen_panel(result, panel: pd.DataFrame) -> None:
    assert len(result.partition_paths) == 1
    partition = result.partition_paths[0]
    panel.to_parquet(partition, index=True)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    record = manifest["partitions"][0]
    record["rows"] = len(panel)
    record["bytes"] = partition.stat().st_size
    record["sha256"] = _file_sha256(partition)
    dates = panel.index.get_level_values("date")
    manifest["rows"] = len(panel)
    manifest["symbols"] = panel.index.get_level_values("symbol").nunique()
    manifest["trading_days"] = dates.nunique()
    manifest["first_date"] = str(dates.min().date())
    manifest["last_date"] = str(dates.max().date())
    manifest["snapshot_hash"] = panel_snapshot_hash(panel.sort_index())
    result.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    result.manifest_sha256_path.write_text(
        f"{_file_sha256(result.manifest_path)}  manifest.json\n", encoding="utf-8"
    )


def _rewrite_feature_tape_partition(result, partition_index: int, tape: pd.DataFrame) -> None:
    partition = result.feature_tape_partition_paths[partition_index]
    tape.to_parquet(partition, index=True)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    record = manifest["feature_tape"]["partitions"][partition_index]
    record["rows"] = len(tape)
    record["bytes"] = partition.stat().st_size
    record["sha256"] = _file_sha256(partition)
    restored = pd.concat(
        [pd.read_parquet(path) for path in result.feature_tape_partition_paths],
        copy=False,
    )
    feature = manifest["feature_tape"]
    dates = pd.DatetimeIndex(restored.index.get_level_values("date"))
    feature["rows"] = len(restored)
    feature["symbols"] = int(restored.index.get_level_values("symbol").nunique())
    feature["trading_days"] = int(dates.nunique())
    feature["first_date"] = str(dates.min().date())
    feature["last_date"] = str(dates.max().date())
    feature["columns"] = [str(name) for name in restored.columns]
    feature["dtypes"] = {
        str(name): str(dtype) for name, dtype in restored.dtypes.items()
    }
    feature["snapshot_hash"] = panel_snapshot_hash(restored)
    result.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    result.manifest_sha256_path.write_text(
        f"{_file_sha256(result.manifest_path)}  manifest.json\n", encoding="utf-8"
    )


def _rewrite_terminal_sidecar(result, sidecar: pd.DataFrame) -> None:
    path = result.terminal_stale_membership_exception_path
    assert path is not None
    sidecar.sort_index().to_parquet(path, index=True)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    section = manifest["terminal_stale_membership_exceptions"]
    record = section["file"]
    dates = pd.DatetimeIndex(sidecar.index.get_level_values("date"))
    symbols = pd.Index(sidecar.index.get_level_values("symbol").astype(str))
    snapshot_hash = panel_snapshot_hash(sidecar.sort_index())
    section["rows"] = len(sidecar)
    section["symbols"] = int(symbols.nunique())
    section["trading_days"] = int(dates.nunique())
    section["snapshot_hash"] = snapshot_hash
    record["rows"] = len(sidecar)
    record["bytes"] = path.stat().st_size
    record["sha256"] = _file_sha256(path)
    manifest["feature_tape"]["source_contract"][
        "terminal_stale_membership_exceptions_sha256"
    ] = snapshot_hash
    result.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    result.manifest_sha256_path.write_text(
        f"{_file_sha256(result.manifest_path)}  manifest.json\n", encoding="utf-8"
    )


def test_audit_rejects_resealed_fake_terminal_exception_for_a_real_bar(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    columns = list(result.terminal_stale_membership_exceptions.columns)
    forged = pd.DataFrame(
        [
            {
                "date": dates[10],
                "symbol": SYMBOLS[1],
                "vendor_universe_member": True,
                "universe_member": True,
                "research_eligible": False,
                "reason": "terminal_stale_membership_after_verified_out_date",
                "stock_basic_out_date": dates[9],
                "last_raw_bar_date": dates[9],
                "membership_snapshot_date": dates[0],
                "security_status": "0",
                "price_fill_policy": "none",
                "terminal_return_policy": "unresolved_no_imputation",
            }
        ]
    ).set_index(["date", "symbol"])[columns]
    _rewrite_terminal_sidecar(result, forged)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )
    assert not report.valid
    assert "terminal_exception_source_replay_mismatch" in {
        finding.code for finding in report.findings
    }


def test_audit_rejects_resealed_terminal_proof_field_or_missing_row(tmp_path):
    proof_config, dates, _, _ = _write_fixture(tmp_path / "proof")
    _make_terminal_stale_fixture(proof_config, dates)
    proof_result = build_hs300_research_panel(proof_config)
    proof = proof_result.terminal_stale_membership_exceptions.copy()
    proof.iloc[0, proof.columns.get_loc("membership_snapshot_date")] = dates[1]
    _rewrite_terminal_sidecar(proof_result, proof)
    proof_report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=proof_config.output_dir, expected_members=2)
    )
    assert "terminal_exception_source_replay_mismatch" in {
        finding.code for finding in proof_report.findings
    }

    missing_config, missing_dates, _, _ = _write_fixture(tmp_path / "missing")
    _make_terminal_stale_fixture(missing_config, missing_dates)
    missing_result = build_hs300_research_panel(missing_config)
    missing = missing_result.terminal_stale_membership_exceptions.iloc[:-1].copy()
    _rewrite_terminal_sidecar(missing_result, missing)
    missing_report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=missing_config.output_dir, expected_members=2)
    )
    assert "terminal_exception_source_replay_mismatch" in {
        finding.code for finding in missing_report.findings
    }


def test_snapshot_namespace_binds_nonmember_tape_and_source_contract(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    first = build_hs300_research_panel(config)
    first_storage_id = first.manifest["storage_snapshot_id"]
    first_panel_hash = first.validation.snapshot_hash
    first_partition_paths = tuple(first.partition_paths)

    request_index = pd.read_parquet(
        config.normalized_dir / "price_request_index.parquet"
    )
    request_key = str(
        request_index.loc[
            request_index["code"].astype(str).eq(SYMBOLS[2])
            & request_index["adjustflag"].astype(str).eq("3"),
            "request_key",
        ].iloc[0]
    )
    state_path = config.cache_dir / "state" / f"{request_key}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    object_path = config.cache_dir / str(state["object"])
    raw = pd.read_parquet(object_path)
    attacked = raw["date"].astype(str).eq(str(dates[2].date()))
    raw.loc[attacked, "amount"] = (
        pd.to_numeric(raw.loc[attacked, "amount"], errors="raise") * 1.25
    )
    raw.to_parquet(object_path, index=False)
    state["parquet_sha256"] = _file_sha256(object_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    second = build_hs300_research_panel(config)

    assert second.validation.snapshot_hash == first_panel_hash
    assert second.manifest["storage_snapshot_id"] != first_storage_id
    assert all(path.is_file() for path in first_partition_paths)
    assert all(first_storage_id in path.parts for path in first_partition_paths)
    assert all(
        second.manifest["storage_snapshot_id"] in path.parts
        for path in second.partition_paths
    )


def test_audit_allows_historical_union_symbol_inactive_before_custom_end(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    request_path = config.normalized_dir / "price_request_index.parquet"
    requests = pd.read_parquet(request_path)
    for flag in ("1", "3"):
        selected = requests["code"].astype(str).eq(SYMBOLS[2]) & requests[
            "adjustflag"
        ].astype(str).eq(flag)
        request_key = str(requests.loc[selected, "request_key"].iloc[0])
        state_path = config.cache_dir / "state" / f"{request_key}.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        object_path = config.cache_dir / str(state["object"])
        frame = pd.read_parquet(object_path)
        frame = frame.loc[
            pd.to_datetime(frame["date"]) > dates[10]
        ].reset_index(drop=True)
        frame.to_parquet(object_path, index=False)
        state["rows"] = len(frame)
        state["parquet_sha256"] = _file_sha256(object_path)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        requests.loc[selected, "rows"] = len(frame)
        requests.loc[selected, "first_date"] = str(
            pd.Timestamp(frame["date"].min()).date()
        )
        requests.loc[selected, "last_date"] = str(
            pd.Timestamp(frame["date"].max()).date()
        )
    requests.to_parquet(request_path, index=False)
    config = replace(
        config,
        research_end=str(dates[10].date()),
        contract_minimum_history_days=3,
    )
    result = build_hs300_research_panel(config)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert report.valid
    assert result.feature_tape is not None
    assert SYMBOLS[2] not in set(
        result.feature_tape.index.get_level_values("symbol").astype(str)
    )
    assert report.metrics["continuous_feature_tape"][
        "source_symbols_inactive_through_tape_end"
    ] == 1


def test_offline_post_build_audit_rehashes_sources_and_reports_proxy_limits(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path, start="2019-12-16")
    def add_outdate(security):
        security.loc[security["code"] == SYMBOLS[0], "outDate"] = str(
            dates[-1].date()
        )
        return security

    _rewrite_security_cache(config, add_outdate)
    result = build_hs300_research_panel(config)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert report.valid
    assert report.status == "warn"
    assert report.research_valid is True
    assert report.portfolio_grade is False
    assert report.capital_deployment_allowed is False
    assert "research_proxy_not_portfolio_grade" in {
        item.code for item in report.findings
    }
    assert report.rows == result.validation.rows
    assert report.actual_snapshot_hash == result.validation.snapshot_hash
    assert report.metrics["membership"]["bad_dates"] == 0
    assert report.metrics["raw_adjusted_contract"]["mismatch_by_field"] == {}
    assert report.metrics["integrity"]["partitions_loaded"] == 2
    assert report.metrics["source_integrity"]["normalized_hashes_verified"] == 11
    assert report.metrics["source_integrity"]["cache_hashes_verified"] == 23
    assert report.metrics["delisted_history"]["securities_with_out_date"] == 1
    assert report.metrics["delisted_history"]["raw_adjusted_pairs_covered"] == 1
    assert report.metrics["limit_proxy"]["unknown_tradable_rows"] == 0


def test_post_build_audit_detects_daily_member_loss_even_with_resealed_hashes(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    panel = result.panel.iloc[1:].copy()
    _rewrite_frozen_panel(result, panel)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    codes = {item.code for item in report.findings}
    assert "daily_member_count_not_expected" in codes


def test_post_build_audit_detects_resealed_feature_tape_member_mask_attack(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    partition_index = len(result.feature_tape_partition_paths) - 1
    tape = pd.read_parquet(result.feature_tape_partition_paths[partition_index])
    member_row = tape.index.intersection(result.panel.index)[0]
    tape.loc[member_row, "universe_member"] = False
    _rewrite_feature_tape_partition(result, partition_index, tape)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    codes = {item.code for item in report.findings}
    assert "feature_tape_member_mask_mismatch" in codes
    assert "feature_tape_daily_member_count_not_expected" in codes


def test_post_build_audit_detects_resealed_feature_tape_value_attack(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    tape = pd.read_parquet(result.feature_tape_partition_paths[0])
    attacked = tape.index.intersection(result.panel.index)[0]
    tape.loc[attacked, "adj_close"] *= 4.0
    _rewrite_feature_tape_partition(result, 0, tape)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    codes = {item.code for item in report.findings}
    assert "feature_tape_member_panel_field_mismatch" in codes
    assert "feature_tape_source_lineage_mismatch" in codes


def test_post_build_audit_detects_resealed_nonmember_tape_source_attack(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    tape = pd.read_parquet(result.feature_tape_partition_paths[0])
    attacked = (dates[2], SYMBOLS[2])
    assert not bool(tape.loc[attacked, "universe_member"])
    tape.loc[attacked, "adj_close"] *= 4.0
    _rewrite_feature_tape_partition(result, 0, tape)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    codes = {item.code for item in report.findings}
    assert "feature_tape_source_lineage_mismatch" in codes
    assert "feature_tape_member_panel_field_mismatch" not in codes


def test_post_build_audit_rejects_resealed_member_panel_as_continuous_tape(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    _rewrite_feature_tape_partition(result, 0, result.panel.copy())

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    assert "feature_tape_source_index_mismatch" in {
        item.code for item in report.findings
    }


def test_post_build_audit_detects_resealed_warmup_member_mask_swap(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    tape = pd.read_parquet(result.feature_tape_partition_paths[0])
    warmup_date = dates[2]
    tape.loc[(warmup_date, SYMBOLS[0]), "universe_member"] = False
    tape.loc[(warmup_date, SYMBOLS[2]), "universe_member"] = True
    assert int(tape.xs(warmup_date, level="date")["universe_member"].sum()) == 2
    _rewrite_feature_tape_partition(result, 0, tape)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    codes = {item.code for item in report.findings}
    assert "feature_tape_normalized_membership_mask_mismatch" in codes
    assert "feature_tape_daily_member_count_not_expected" not in codes


def test_post_build_audit_detects_resealed_whole_session_deletion(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    tape = pd.read_parquet(result.feature_tape_partition_paths[0])
    attacked_date = dates[2]
    tape = tape.loc[tape.index.get_level_values("date") != attacked_date].copy()
    _rewrite_feature_tape_partition(result, 0, tape)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    codes = {item.code for item in report.findings}
    assert "feature_tape_trade_calendar_mismatch" in codes
    assert "feature_tape_source_index_mismatch" in codes


def test_post_build_audit_rejects_resealed_non_boolean_tape_domain(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    tape = pd.read_parquet(result.feature_tape_partition_paths[0])
    tape["universe_member"] = tape["universe_member"].astype("int8")
    tape.loc[(dates[2], SYMBOLS[2]), "universe_member"] = 2
    _rewrite_feature_tape_partition(result, 0, tape)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    assert "feature_tape_boolean_domain_invalid" in {
        item.code for item in report.findings
    }


def test_post_build_audit_binds_feature_partition_to_current_snapshot_subtree(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    panel_path = result.partition_paths[0]
    panel = pd.read_parquet(panel_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    feature = manifest["feature_tape"]
    record = feature["partitions"][0]
    record["path"] = str(panel_path)
    record["rows"] = len(panel)
    record["bytes"] = panel_path.stat().st_size
    record["sha256"] = _file_sha256(panel_path)
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    feature["rows"] = len(panel)
    feature["symbols"] = int(panel.index.get_level_values("symbol").nunique())
    feature["trading_days"] = int(dates.nunique())
    feature["first_date"] = str(dates.min().date())
    feature["last_date"] = str(dates.max().date())
    feature["columns"] = [str(name) for name in panel.columns]
    feature["dtypes"] = {str(name): str(dtype) for name, dtype in panel.dtypes.items()}
    feature["snapshot_hash"] = panel_snapshot_hash(panel)
    result.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    result.manifest_sha256_path.write_text(
        f"{_file_sha256(result.manifest_path)}  manifest.json\n", encoding="utf-8"
    )

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    assert "feature_tape_partition_not_bound_to_snapshot" in {
        item.code for item in report.findings
    }


def test_post_build_audit_rejects_resealed_cross_partition_schema_drift(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path, start="2019-12-16")
    result = build_hs300_research_panel(config)
    assert len(result.feature_tape_partition_paths) == 2
    tape = pd.read_parquet(result.feature_tape_partition_paths[1])
    tape["limit_up"] = tape["limit_up"].astype("int8")
    _rewrite_feature_tape_partition(result, 1, tape)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    assert "feature_tape_partition_schema_mismatch" in {
        item.code for item in report.findings
    }


def test_post_build_audit_detects_adjustment_contract_and_extreme_returns(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    panel = result.panel.copy()
    mismatch_row = panel.index[2]
    panel.loc[mismatch_row, "adj_open"] *= 1.02
    jump_row = (dates[12], "sh.600002")
    panel.loc[jump_row, "adj_factor"] *= 3.0
    for field in ("open", "high", "low", "close", "vwap"):
        panel.loc[jump_row, f"adj_{field}"] = (
            panel.loc[jump_row, field] * panel.loc[jump_row, "adj_factor"]
        )
    _rewrite_frozen_panel(result, panel)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    codes = {item.code for item in report.findings}
    assert "adjusted_price_factor_mismatch" in codes
    assert "extreme_adjusted_returns" in codes


def test_post_build_audit_fails_closed_on_partition_hash_tampering(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    partition = result.partition_paths[0]
    partition.write_bytes(partition.read_bytes() + b"tamper")

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    assert "partition_sha_mismatch" in {item.code for item in report.findings}


def test_post_build_audit_reconstructs_semantic_lineage_not_just_hashes(tmp_path):
    config, dates, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    panel = result.panel.copy()

    # Keep the modified snapshot internally hash-consistent.  The independent
    # source lineage, not a byte mismatch, must catch every attack.
    original = panel.index[0]
    rewritten = (original[0], "sh.699999")
    tuples = list(panel.index)
    tuples[0] = rewritten
    panel.index = pd.MultiIndex.from_tuples(tuples, names=["date", "symbol"])
    panel.loc[panel.index[1], "industry"] = "Invented Industry"
    panel.loc[(dates[10], "sh.600002"), ["limit_up", "limit_proxy_known"]] = False
    panel.loc[panel.index[3], "preclose"] *= 1.01
    panel["benchmark_weight"] = 0.5
    panel = panel.sort_index()
    _rewrite_frozen_panel(result, panel)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    codes = {item.code for item in report.findings}
    assert "panel_membership_lineage_mismatch" in codes
    assert "panel_industry_lineage_mismatch" in codes
    assert "limit_proxy_lineage_mismatch" in codes
    assert "benchmark_weight_lineage_mismatch" in codes
    assert "panel_price_or_derived_lineage_mismatch" in codes


def test_post_build_audit_rejects_unparsable_asof_dates_and_missing_watermark(tmp_path):
    config, _, _, _ = _write_fixture(tmp_path)
    result = build_hs300_research_panel(config)
    panel = result.panel.copy()
    panel["membership_snapshot_date"] = "nonsense"
    panel["industry_snapshot_date"] = "nonsense"
    panel = panel.drop(columns="research_watermark")
    _rewrite_frozen_panel(result, panel)

    report = audit_hs300_research_panel(
        HS300AuditConfig(panel_dir=config.output_dir, expected_members=2)
    )

    assert not report.valid
    codes = {item.code for item in report.findings}
    assert "invalid_membership_snapshot_date" in codes
    assert "invalid_industry_snapshot_date" in codes
    assert "panel_research_watermark_column_missing" in codes


def test_audit_policy_rejects_nonfinite_or_overly_loose_adjustment_tolerance(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        HS300AuditConfig(panel_dir=tmp_path, adjusted_price_tolerance=float("inf"))
    with pytest.raises(ValueError, match="fail-closed ceiling"):
        HS300AuditConfig(panel_dir=tmp_path, adjusted_price_tolerance=1e-4)
