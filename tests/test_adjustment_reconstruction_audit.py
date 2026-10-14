from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from cogalpha_plus import cli
from cogalpha_plus.data_sources.adjustment_reconstruction_audit import (
    AdjustmentReconstructionAuditConfig,
    audit_adjustment_reconstruction,
)
from cogalpha_plus.data_sources.baostock import (
    ADJUST_FACTOR_FIELDS,
    EQUITY_DAILY_FIELDS,
    make_request_key,
)


_VERSION = "00.9.30"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _commit(
    cache: Path,
    endpoint: str,
    args: dict[str, object],
    frame: pd.DataFrame,
) -> str:
    key = make_request_key(endpoint, args, provider_version=_VERSION)
    object_path = cache / "objects" / endpoint / f"{key}.parquet"
    state_path = cache / "state" / f"{key}.json"
    object_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(object_path, index=False)
    state = {
        "state_schema_version": 1,
        "provider": "baostock",
        "provider_version": _VERSION,
        "endpoint": endpoint,
        "args": args,
        "request_key": key,
        "status": "success",
        "rows": len(frame),
        "fields": list(frame.columns),
        "object": str(object_path.relative_to(cache)),
        "parquet_sha256": _sha256(object_path),
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return key


def _price_frame(code: str, flag: str, *, mismatch: bool = False) -> pd.DataFrame:
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    raw_prices = {
        "open": [9.6, 10.2, 5.7],
        "high": [10.2, 11.2, 6.2],
        "low": [9.4, 10.0, 5.4],
        "close": [10.0, 11.0, 6.0],
        "preclose": [9.5, 10.0, 5.5],
    }
    factors = [1.0, 1.0, 2.0] if flag == "1" else [1.0, 1.0, 1.0]
    prices = {
        field: [value * factor for value, factor in zip(values, factors)]
        for field, values in raw_prices.items()
    }
    if mismatch and flag == "1":
        prices["close"][1] += 0.1
    records = []
    for number, day in enumerate(dates):
        record = {
            "date": day,
            "code": code,
            "open": str(prices["open"][number]),
            "high": str(prices["high"][number]),
            "low": str(prices["low"][number]),
            "close": str(prices["close"][number]),
            "preclose": str(prices["preclose"][number]),
            "volume": "1000",
            "amount": "10000",
            "adjustflag": flag,
            "turn": "1.0",
            "tradestatus": "1",
            "pctChg": "0.0",
            "peTTM": "12.0",
            "pbMRQ": "1.2",
            "psTTM": "1.0",
            "pcfNcfTTM": "8.0",
            "isST": "0",
        }
        records.append(record)
    return pd.DataFrame.from_records(records, columns=EQUITY_DAILY_FIELDS)


def _event_frame(code: str, *, missing_anchor: bool = False) -> pd.DataFrame:
    rows = []
    if not missing_anchor:
        rows.append([code, "2019-01-01", "1.0", "1.0", "1.0"])
    rows.append([code, "2020-01-03", "0.5", "2.0", "1.0"])
    return pd.DataFrame(rows, columns=ADJUST_FACTOR_FIELDS, dtype=object)


def _fixture(
    tmp_path: Path,
    *,
    mismatch: bool = False,
    missing_anchor: bool = False,
) -> tuple[Path, AdjustmentReconstructionAuditConfig]:
    root = tmp_path / "market"
    cache = root / "raw"
    normalized = root / "normalized"
    normalized.mkdir(parents=True)
    code = "sh.600001"
    union_path = normalized / "security_union.parquet"
    pd.DataFrame({"code": [code], "code_name": ["A"]}).to_parquet(
        union_path, index=False
    )
    membership_manifest = {
        "phase": "membership_catalog",
        "complete": True,
        "historical_union_symbols": 1,
        "files": {"security_union": _metadata(union_path)},
    }
    (normalized / "membership_manifest.json").write_text(
        json.dumps(membership_manifest), encoding="utf-8"
    )

    price_rows = []
    price_fields = ",".join(EQUITY_DAILY_FIELDS)
    for flag, basis in (("3", "raw_unadjusted"), ("1", "backward_adjusted")):
        frame = _price_frame(code, flag, mismatch=mismatch)
        args = {
            "code": code,
            "fields": price_fields,
            "start_date": "2020-01-01",
            "end_date": "2020-01-03",
            "frequency": "d",
            "adjustflag": flag,
        }
        key = _commit(cache, "history_k_data", args, frame)
        price_rows.append(
            {
                "code": code,
                "adjustflag": flag,
                "basis": basis,
                "rows": len(frame),
                "first_date": "2020-01-01",
                "last_date": "2020-01-03",
                "request_key": key,
                "cache_hit": False,
            }
        )
    price_index_path = normalized / "price_request_index.parquet"
    pd.DataFrame(price_rows).to_parquet(price_index_path, index=False)
    price_manifest = {
        "phase": "price_catalog",
        "complete": True,
        "start_date": "2020-01-01",
        "end_date": "2020-01-03",
        "historical_union_symbols": 1,
        "downloaded_symbols": 1,
        "price_requests": 2,
        "files": {"price_request_index": _metadata(price_index_path)},
    }
    (normalized / "price_manifest.json").write_text(
        json.dumps(price_manifest), encoding="utf-8"
    )

    events = _event_frame(code, missing_anchor=missing_anchor)
    event_args = {
        "code": code,
        "start_date": "1990-01-01",
        "end_date": "2020-01-03",
    }
    event_key = _commit(cache, "adjust_factor", event_args, events)
    adjustment_index_path = normalized / "adjustment_request_index.parquet"
    pd.DataFrame(
        {
            "code": [code],
            "rows": [len(events)],
            "first_event_date": [str(events["dividOperateDate"].min())],
            "last_event_date": [str(events["dividOperateDate"].max())],
            "request_key": [event_key],
            "cache_hit": [False],
        }
    ).to_parquet(adjustment_index_path, index=False)
    adjustment_manifest = {
        "phase": "adjustment_factor_catalog",
        "complete": True,
        "start_date": "1990-01-01",
        "end_date": "2020-01-03",
        "historical_union_symbols": 1,
        "downloaded_symbols": 1,
        "adjustment_factor_requests": 1,
        "event_rows": len(events),
        "file": _metadata(adjustment_index_path),
    }
    (normalized / "adjustment_manifest.json").write_text(
        json.dumps(adjustment_manifest), encoding="utf-8"
    )
    config = AdjustmentReconstructionAuditConfig(
        cache_dir=cache, normalized_dir=normalized
    )
    return root, config


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_complete_reconstruction_is_valid_and_read_only(tmp_path):
    root, config = _fixture(tmp_path)
    before = _tree_snapshot(root)
    report = audit_adjustment_reconstruction(config)
    after = _tree_snapshot(root)

    assert report["valid"] is True
    assert report["decision"] == "eligible_for_preregistered_raw_plus_adjust_factor_v2"
    assert report["read_only"] is True
    assert report["network_access"] is False
    assert report["coverage"] == {
        "expected_symbols": 1,
        "source_covered_symbols": 1,
        "within_tolerance_symbols": 1,
        "source_coverage_fraction": 1.0,
        "within_tolerance_fraction": 1.0,
        "aligned_rows": 3,
        "compared_cells": 15,
        "within_tolerance_cells": 15,
    }
    assert report["contract"]["comparison_fields"] == [
        "open",
        "high",
        "low",
        "close",
        "preclose",
    ]
    assert before == after


def test_any_cell_outside_tolerance_fails_closed(tmp_path):
    _root, config = _fixture(tmp_path, mismatch=True)
    report = audit_adjustment_reconstruction(config)

    assert report["valid"] is False
    assert report["decision"] == "retain_direct_adjusted_history"
    assert report["coverage"]["within_tolerance_fraction"] == 0.0
    assert report["coverage"]["compared_cells"] == 15
    assert report["coverage"]["within_tolerance_cells"] == 14
    assert report["metrics"]["outside_tolerance_cells_by_field"]["close"] == 1
    assert report["failed_symbols"] == ["sh.600001"]
    assert "outside_tolerance:close" in report["failure_samples"][0]["detail"]


def test_missing_initial_factor_fails_source_coverage(tmp_path):
    _root, config = _fixture(tmp_path, missing_anchor=True)
    report = audit_adjustment_reconstruction(config)

    assert report["valid"] is False
    assert report["coverage"]["source_coverage_fraction"] == 0.0
    assert report["failed_symbols"] == ["sh.600001"]
    assert "on or before the first raw price" in report["failure_samples"][0]["detail"]


def test_missing_committed_object_prevents_complete_coverage(tmp_path):
    root, config = _fixture(tmp_path)
    adjustment_index = pd.read_parquet(
        root / "normalized" / "adjustment_request_index.parquet"
    )
    request_key = str(adjustment_index.loc[0, "request_key"])
    (
        root / "raw" / "objects" / "adjust_factor" / f"{request_key}.parquet"
    ).unlink()

    report = audit_adjustment_reconstruction(config)

    assert report["valid"] is False
    assert report["coverage"]["source_coverage_fraction"] == 0.0
    assert report["coverage"]["within_tolerance_fraction"] == 0.0
    assert report["failed_symbols"] == ["sh.600001"]
    assert "missing" in report["failure_samples"][0]["detail"]


def test_cli_emits_machine_readable_json_and_rejects_looser_threshold(
    tmp_path, capsys
):
    root, _config = _fixture(tmp_path)
    valid_code = cli.main(
        [
            "data",
            "adjustment-reconstruction-audit",
            "--project",
            str(tmp_path),
            "--data-root",
            str(root),
        ]
    )
    valid_payload = json.loads(capsys.readouterr().out)
    assert valid_code == 0
    assert valid_payload["valid"] is True

    invalid_code = cli.main(
        [
            "data",
            "adjustment-reconstruction-audit",
            "--project",
            str(tmp_path),
            "--data-root",
            str(root),
            "--relative-tolerance",
            "0.001",
        ]
    )
    invalid_payload = json.loads(capsys.readouterr().out)
    assert invalid_code == 2
    assert invalid_payload["valid"] is False
    assert invalid_payload["failure_samples"][0]["scope"] == "configuration"
