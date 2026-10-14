from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import pandas as pd

from cogalpha_plus.data_sources.baostock import (
    ADJUST_FACTOR_FIELDS,
    INDUSTRY_FIELDS,
)
from cogalpha_plus.data_sources.hs300_download import _materialize_industry_catalog


PROJECT = Path(__file__).resolve().parents[1]


def _watcher_module():
    path = PROJECT / "scripts" / "run_hs300_after_prices.py"
    spec = importlib.util.spec_from_file_location("hs300_pipeline_watcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_watcher_runs_semantic_configuration_gate_first_and_never_reveals():
    module = _watcher_module()
    commands = module._commands(PROJECT, PROJECT / "runs" / "test")

    assert commands[0][-3:] == ["check-config", "--project", str(PROJECT)]
    flattened = [token for command in commands for token in command]
    assert "hs300-reveal" not in flattened
    phases = [
        next(
            (token for token in command if token in {
                "membership-lineage",
                "security-lineage",
                "adjustments",
                "industry",
                "build",
                "audit",
            }),
            None,
        )
        for command in commands
    ]
    assert [phase for phase in phases if phase is not None] == [
        "membership-lineage",
        "security-lineage",
        "adjustments",
        "industry",
        "build",
        "audit",
    ]
    adjustment = next(command for command in commands if "adjustments" in command)
    assert adjustment[-6:] == [
        "--start",
        "1990-01-01",
        "--end",
        "2025-12-31",
        "--progress-every",
        "20",
    ]
    assert commands[-1][-2:] == ["--output", str(PROJECT / "runs" / "test")]


def test_price_gate_requires_canonical_supervisor_handoff(monkeypatch, tmp_path):
    module = _watcher_module()
    manifest = tmp_path / "price_manifest.json"
    manifest.write_text(json.dumps({"complete": True}), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_canonical_price_handoff",
        lambda _project: (None, "price_request_index_hash_mismatch"),
    )

    with pytest.raises(RuntimeError, match="canonical handoff validation"):
        module._wait_for_prices(
            tmp_path,
            manifest,
            poll_seconds=0.01,
            timeout_hours=1.0,
        )


def test_industry_watchdog_contract_uses_frozen_membership_query_dates(tmp_path):
    module = _watcher_module()
    normalized = tmp_path / "data" / "baostock_hs300" / "normalized"
    normalized.mkdir(parents=True)
    pd.DataFrame({"code": ["sh.600000"]}).to_parquet(
        normalized / "security_union.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "vendor_update_date": pd.to_datetime(["2020-01-06"]),
            "code": ["sh.600000"],
        }
    ).to_parquet(normalized / "membership_snapshots.parquet", index=False)
    pd.DataFrame(
        {
            "endpoint": ["hs300_stocks", "hs300_stocks"],
            "requested_date": ["2020-01-02", "2020-01-13"],
        }
    ).to_parquet(normalized / "membership_request_index.parquet", index=False)

    contract = module._network_phase_contract(tmp_path, "industry")

    assert contract.endpoint == "stock_industry"
    assert contract.progress_items == (
        "2020-01-02",
        "2020-01-06",
        "2020-01-13",
    )
    assert contract.expected_args["2020-01-06"] == {
        "code": "",
        "date": "2020-01-06",
    }
    assert set(contract.bound_inputs) == {
        normalized / "security_union.parquet",
        normalized / "membership_snapshots.parquet",
        normalized / "membership_request_index.parquet",
    }


def test_post_price_implementation_freeze_is_idempotent_and_detects_drift(
    tmp_path,
):
    module = _watcher_module()
    files = {
        "src/cogalpha_plus/core.py": "VALUE = 1\n",
        "scripts/run_hs300_after_prices.py": "# runner\n",
        "scripts/supervise_hs300_pipeline.py": "# supervisor\n",
        "configs/agents.yaml": "agents: []\n",
        "proposals/seed.jsonl": "{}\n",
        "reviews/review.json": "{}\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    frozen = module._freeze_post_price_implementation(tmp_path)
    assert module._freeze_post_price_implementation(tmp_path) == frozen
    module._assert_post_price_implementation(tmp_path, frozen)

    (tmp_path / "src/cogalpha_plus/core.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="changed after freeze"):
        module._assert_post_price_implementation(tmp_path, frozen)
    with pytest.raises(RuntimeError, match="already frozen"):
        module._freeze_post_price_implementation(tmp_path)


def test_shared_catalog_reader_refuses_an_active_exclusive_writer(
    monkeypatch, tmp_path
):
    module = _watcher_module()

    def contested(_descriptor, operation):
        if operation & module.fcntl.LOCK_NB:
            raise BlockingIOError("writer active")

    monkeypatch.setattr(module.fcntl, "flock", contested)
    with pytest.raises(RuntimeError, match="catalog writer is active"):
        with module._shared_catalog_read_lock(tmp_path / "raw"):
            raise AssertionError("unreachable")


def test_price_handoff_identity_is_rechecked_against_frozen_file(
    monkeypatch, tmp_path
):
    module = _watcher_module()
    target = (
        tmp_path
        / "data"
        / "baostock_hs300"
        / "normalized"
        / "price_handoff_attestation.json"
    )
    target.parent.mkdir(parents=True)
    frozen = {"schema_version": 1, "aggregate_sha256": "a" * 64}
    target.write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_canonical_price_handoff",
        lambda _project: ({"complete": True}, "complete"),
    )
    monkeypatch.setattr(
        module,
        "_price_handoff_identity",
        lambda _project, _manifest: dict(frozen),
    )
    module._assert_price_handoff_identity(tmp_path, frozen)

    target.write_text(
        json.dumps({**frozen, "aggregate_sha256": "b" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        module._assert_price_handoff_identity(tmp_path, frozen)


def test_price_handoff_supports_large_bounded_attestation(monkeypatch, tmp_path):
    module = _watcher_module()
    identity = {
        "schema_version": 1,
        "requests": ["x" * 1024 for _ in range(1100)],
        "aggregate_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        module,
        "_price_handoff_identity",
        lambda _project, _manifest: identity,
    )

    frozen = module._freeze_price_handoff_identity(
        tmp_path,
        {"complete": True},
    )

    target = (
        tmp_path
        / "data"
        / "baostock_hs300"
        / "normalized"
        / "price_handoff_attestation.json"
    )
    assert target.stat().st_size > module.DEFAULT_SECURE_JSON_MAXIMUM_BYTES
    assert frozen == identity
    assert module._load_regular_json(
        target,
        maximum_bytes=module.PRICE_HANDOFF_MAXIMUM_BYTES,
    ) == identity


def test_adjustment_completion_requires_full_cache_index_and_manifest(tmp_path):
    module = _watcher_module()
    data_root = tmp_path / "data" / "baostock_hs300"
    contract = _adjustment_contract(
        module,
        data_root,
        "sh.600000",
    )
    frame = pd.DataFrame(
        [["sh.600000", "2020-01-02", "1", "1", "1"]],
        columns=contract.fields,
    )
    request_key, object_path = _commit_success(
        module,
        contract,
        "sh.600000",
        frame,
    )
    normalized = data_root / "normalized"
    normalized.mkdir()
    index_path = normalized / "adjustment_request_index.parquet"
    pd.DataFrame(
        {
            "code": ["sh.600000"],
            "rows": [1],
            "first_event_date": ["2020-01-02"],
            "last_event_date": ["2020-01-02"],
            "request_key": [request_key],
            "cache_hit": [False],
        }
    ).to_parquet(index_path, index=False)
    manifest = {
        "phase": "adjustment_factor_catalog",
        "complete": True,
        "created_at": "2026-01-01T00:00:02+00:00",
        "start_date": module.FROZEN_ADJUSTMENT_START_DATE,
        "end_date": module.FROZEN_END_DATE,
        "historical_union_symbols": 1,
        "downloaded_symbols": 1,
        "adjustment_factor_requests": 1,
        "event_rows": 1,
        "file": _file_declaration(index_path),
    }
    (normalized / "adjustment_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    module._validate_network_phase_complete(contract)

    _commit_success(
        module,
        contract,
        "sh.600000",
        frame,
        provider_version="0.9.4-watcher-test",
    )
    module._validate_network_phase_complete(contract)

    object_path.write_bytes(b"not parquet")
    with pytest.raises(RuntimeError, match="selects no canonical success"):
        module._validate_network_phase_complete(contract)


def test_industry_completion_requires_full_cache_index_and_manifest(tmp_path):
    module = _watcher_module()
    data_root = tmp_path / "data" / "baostock_hs300"
    item = "2020-01-06"
    normalized = data_root / "normalized"
    normalized.mkdir(parents=True)
    membership_path = normalized / "membership_snapshots.parquet"
    membership_frame = pd.DataFrame(
        {
            "requested_date": pd.to_datetime([item]),
            "vendor_update_date": pd.to_datetime([item]),
            "code": ["sh.600000"],
            "code_name": ["A"],
        }
    )
    membership_frame.to_parquet(membership_path, index=False)
    membership_index_path = normalized / "membership_request_index.parquet"
    pd.DataFrame(
        {"endpoint": ["hs300_stocks"], "requested_date": [item]}
    ).to_parquet(membership_index_path, index=False)
    union_path = normalized / "security_union.parquet"
    pd.DataFrame({"code": ["sh.600000"]}).to_parquet(union_path, index=False)
    contract = module._NetworkPhaseContract(
        action="industry",
        endpoint="stock_industry",
        fields=tuple(INDUSTRY_FIELDS),
        progress_items=(item,),
        expected_args={item: {"code": "", "date": item}},
        state_dir=data_root / "raw" / "state",
        bound_inputs={
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (membership_path, membership_index_path, union_path)
        },
        union_size=1,
        provider_version="0.9.3-watcher-test",
    )
    raw = pd.DataFrame(
        [["2020-01-06", "sh.600000", "A", "Industry A", "CSRC"]],
        columns=contract.fields,
    )
    request_key, _object_path = _commit_success(
        module,
        contract,
        item,
        raw,
    )
    replay = raw.copy()
    replay.attrs["baostock_cache"] = {
        "request_key": request_key,
        "cache_hit": False,
    }
    expected_snapshots, expected_index = _materialize_industry_catalog(
        membership_frame,
        {"sh.600000"},
        [(pd.Timestamp(item), replay)],
        expected_request_dates=pd.to_datetime([item]),
    )
    index_path = normalized / "industry_request_index.parquet"
    expected_index.to_parquet(index_path, index=False)
    snapshots_path = normalized / "industry_snapshots.parquet"
    expected_snapshots.to_parquet(snapshots_path, index=False)
    manifest = {
        "phase": "industry_catalog",
        "complete": True,
        "created_at": "2026-01-01T00:00:02+00:00",
        "native_snapshots": 1,
        "requested_snapshots": 1,
        "historical_union_symbols": 1,
        "rows": 1,
        "source_contract": {
            "schema_version": "1.0",
            "endpoint": "stock_industry",
            "query_code": "",
            "query_dates": "membership_requested_observation_dates_union_vendor_effective_dates",
            "freshness_clock": "successful_requested_observation_date",
            "vendor_update_policy": "nonfuture_effective_metadata",
            "deterministic_union_filter": True,
            "future_update_policy": "forbidden",
        },
        "files": {
            "industry_snapshots": _file_declaration(snapshots_path),
            "industry_request_index": _file_declaration(index_path),
        },
    }
    manifest_path = normalized / "industry_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    module._validate_network_phase_complete(contract)

    manifest["requested_snapshots"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="industry manifest"):
        module._validate_network_phase_complete(contract)


def _adjustment_contract(
    module,
    root: Path,
    *codes: str,
    bound_inputs=None,
    progress_every=20,
    provider_version="0.9.3-watcher-test",
):
    items = tuple(codes)
    return module._NetworkPhaseContract(
        action="adjustments",
        endpoint="adjust_factor",
        fields=tuple(ADJUST_FACTOR_FIELDS),
        progress_items=items,
        expected_args={
            code: {
                "code": code,
                "start_date": module.FROZEN_ADJUSTMENT_START_DATE,
                "end_date": module.FROZEN_END_DATE,
            }
            for code in items
        },
        state_dir=root / "raw" / "state",
        bound_inputs={} if bound_inputs is None else bound_inputs,
        progress_every=progress_every,
        union_size=len(items),
        provider_version=provider_version,
    )


def _state_payload(
    module,
    contract,
    code,
    *,
    status,
    attempt=1,
    provider_version="0.9.3-watcher-test",
    **extra,
):
    args = dict(contract.expected_args[code])
    request_key = module._canonical_request_key(
        contract.endpoint,
        args,
        provider_version,
    )
    payload = {
        "state_schema_version": 1,
        "request_key": request_key,
        "provider": "baostock",
        "provider_version": provider_version,
        "endpoint": contract.endpoint,
        "args": args,
        "attempt_count": attempt,
        "status": status,
        "started_at": "2026-01-01T00:00:00+00:00",
        **extra,
    }
    return request_key, payload


def _commit_success(
    module,
    contract,
    item,
    frame,
    *,
    provider_version="0.9.3-watcher-test",
):
    request_key, payload = _state_payload(
        module,
        contract,
        item,
        status="running",
        provider_version=provider_version,
    )
    relative = Path("objects") / contract.endpoint / f"{request_key}.parquet"
    object_path = contract.state_dir.parent / relative
    object_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(object_path, index=False)
    payload.update(
        {
            "status": "success",
            "completed_at": "2026-01-01T00:00:01+00:00",
            "rows": len(frame),
            "fields": list(contract.fields),
            "object": str(relative),
            "parquet_sha256": hashlib.sha256(object_path.read_bytes()).hexdigest(),
        }
    )
    contract.state_dir.mkdir(parents=True, exist_ok=True)
    (contract.state_dir / f"{request_key}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return request_key, object_path


def _file_declaration(path):
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def test_network_watchdog_accepts_only_bound_monotonic_cli_progress(tmp_path):
    module = _watcher_module()
    contract = _adjustment_contract(
        module,
        tmp_path,
        "sh.600000",
        "sh.600001",
        "sh.600002",
        progress_every=1,
    )
    contract.state_dir.mkdir(parents=True)
    for code in contract.progress_items:
        request_key, payload = _state_payload(
            module,
            contract,
            code,
            status="running",
        )
        relative = Path("objects") / contract.endpoint / f"{request_key}.parquet"
        object_path = contract.state_dir.parent / relative
        object_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=contract.fields).to_parquet(object_path, index=False)
        payload.update(
            {
                "status": "success",
                "completed_at": "2026-01-01T00:00:01+00:00",
                "rows": 0,
                "fields": list(contract.fields),
                "object": str(relative),
                "parquet_sha256": hashlib.sha256(object_path.read_bytes()).hexdigest(),
            }
        )
        (contract.state_dir / f"{request_key}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    script = "\n".join(
        [
            "import json, time",
            f"items = {list(contract.progress_items)!r}",
            "for number, item in enumerate(items, 1):",
            "    print(json.dumps({'progress': number, 'total': len(items), 'item': item}), flush=True)",
            "    time.sleep(0.1)",
        ]
    )

    result = module._run_network_phase_once(
        [sys.executable, "-c", script],
        project=tmp_path,
        environment={},
        contract=contract,
        inactivity_timeout_seconds=0.15,
        poll_seconds=0.01,
    )

    assert result.returncode == 0


@pytest.mark.parametrize(
    "line",
    [
        {"progress": 1, "total": 2, "item": "sh.600001"},
        {"progress": 1, "total": 999, "item": "sh.600000"},
        {"progress": 1, "total": 2, "item": "sh.600000", "extra": True},
    ],
)
def test_false_cli_progress_cannot_keep_stale_network_child_alive(
    tmp_path, line
):
    module = _watcher_module()
    contract = _adjustment_contract(
        module,
        tmp_path,
        "sh.600000",
        "sh.600001",
    )
    script = "\n".join(
        [
            "import json, time",
            f"line = {line!r}",
            "for _ in range(20):",
            "    print(json.dumps(line), flush=True)",
            "    time.sleep(0.03)",
        ]
    )

    with pytest.raises(module.NetworkInactivityTimeout):
        module._run_network_phase_once(
            [sys.executable, "-c", script],
            project=tmp_path,
            environment={},
            contract=contract,
            inactivity_timeout_seconds=0.12,
            poll_seconds=0.01,
        )


def test_canonical_cache_state_transitions_keep_network_child_alive(tmp_path):
    module = _watcher_module()
    code = "sh.600000"
    contract = _adjustment_contract(module, tmp_path, code)
    contract.state_dir.mkdir(parents=True)
    request_key, running = _state_payload(
        module,
        contract,
        code,
        status="running",
    )
    relative = Path("objects") / contract.endpoint / f"{request_key}.parquet"
    object_path = contract.state_dir.parent / relative
    object_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [[code, "2020-01-01", "1", "1", "1"]],
        columns=contract.fields,
    ).to_parquet(object_path, index=False)
    completed = {
        **running,
        "status": "success",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "rows": 1,
        "fields": list(contract.fields),
        "object": str(relative),
        "parquet_sha256": hashlib.sha256(object_path.read_bytes()).hexdigest(),
    }
    state_path = contract.state_dir / f"{request_key}.json"
    script = "\n".join(
        [
            "import json, time",
            "from pathlib import Path",
            f"path = Path({str(state_path)!r})",
            f"running = {running!r}",
            f"completed = {completed!r}",
            "time.sleep(0.05)",
            "path.write_text(json.dumps(running))",
            "time.sleep(0.1)",
            "path.write_text(json.dumps(completed))",
            "time.sleep(0.1)",
        ]
    )

    result = module._run_network_phase_once(
        [sys.executable, "-c", script],
        project=tmp_path,
        environment={},
        contract=contract,
        inactivity_timeout_seconds=0.14,
        poll_seconds=0.01,
    )

    assert result.returncode == 0


def test_noncanonical_cache_rewrites_do_not_count_as_progress(tmp_path):
    module = _watcher_module()
    contract = _adjustment_contract(module, tmp_path, "sh.600000")
    contract.state_dir.mkdir(parents=True)
    state_path = contract.state_dir / ("a" * 64 + ".json")
    script = "\n".join(
        [
            "import json, time",
            "from pathlib import Path",
            f"path = Path({str(state_path)!r})",
            "for attempt in range(20):",
            "    path.write_text(json.dumps({'request_key': 'wrong', 'status': 'running', 'attempt_count': attempt}))",
            "    time.sleep(0.03)",
        ]
    )

    with pytest.raises(module.NetworkInactivityTimeout):
        module._run_network_phase_once(
            [sys.executable, "-c", script],
            project=tmp_path,
            environment={},
            contract=contract,
            inactivity_timeout_seconds=0.12,
            poll_seconds=0.01,
        )


def test_another_provider_version_cannot_count_as_current_cache_progress(tmp_path):
    module = _watcher_module()
    code = "sh.600000"
    contract = _adjustment_contract(module, tmp_path, code)
    tracker = module._CanonicalNetworkStateTracker(contract)
    assert tracker.observe() == (False, 0)

    frame = pd.DataFrame(columns=contract.fields)
    _commit_success(
        module,
        contract,
        code,
        frame,
        provider_version="0.9.4-old-cache",
    )
    assert tracker.observe() == (False, 0)

    _commit_success(module, contract, code, frame)
    assert tracker.observe() == (True, 1)


def test_inactivity_retry_preserves_cache_and_exact_command(monkeypatch, tmp_path):
    module = _watcher_module()
    contract = _adjustment_contract(module, tmp_path, "sh.600000")
    contract.state_dir.mkdir(parents=True)
    committed = contract.state_dir / ("f" * 64 + ".json")
    committed.write_bytes(b"already committed")
    command = [sys.executable, "-c", "pass"]
    calls = []

    def timeout_then_complete(observed, **_kwargs):
        calls.append(list(observed))
        assert committed.read_bytes() == b"already committed"
        if len(calls) == 1:
            raise module.NetworkInactivityTimeout(
                observed,
                30.0,
                action="adjustments",
                successful_requests=17,
            )
        return subprocess.CompletedProcess(list(observed), 0)

    monkeypatch.setattr(module, "_run_network_phase_once", timeout_then_complete)
    monkeypatch.setattr(module, "_validate_network_phase_complete", lambda _contract: None)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    result = module._run_network_phase(
        command,
        project=tmp_path,
        environment={},
        contract=contract,
        inactivity_timeout_seconds=30.0,
        attempts=2,
        initial_backoff_seconds=0.0,
        maximum_backoff_seconds=0.0,
    )

    assert result.returncode == 0
    assert calls == [command, command]
    assert committed.read_bytes() == b"already committed"


def test_nonzero_network_exit_is_not_retried(monkeypatch, tmp_path):
    module = _watcher_module()
    contract = _adjustment_contract(module, tmp_path, "sh.600000")
    calls = []

    def fail(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(list(command), 7)

    monkeypatch.setattr(module, "_run_network_phase_once", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        module._run_network_phase(
            [sys.executable, "-c", "pass"],
            project=tmp_path,
            environment={},
            contract=contract,
            inactivity_timeout_seconds=30.0,
            attempts=8,
            initial_backoff_seconds=0.0,
            maximum_backoff_seconds=0.0,
        )
    assert error.value.returncode == 7
    assert len(calls) == 1


def test_bound_request_plan_input_change_fails_closed(tmp_path):
    module = _watcher_module()
    bound = tmp_path / "security_union.parquet"
    bound.write_bytes(b"frozen")
    contract = _adjustment_contract(
        module,
        tmp_path,
        "sh.600000",
        bound_inputs={bound: hashlib.sha256(bound.read_bytes()).hexdigest()},
    )
    bound.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="request-plan input changed"):
        module._run_network_phase_once(
            [sys.executable, "-c", "pass"],
            project=tmp_path,
            environment={},
            contract=contract,
            inactivity_timeout_seconds=30.0,
        )


@pytest.mark.parametrize("delivered_signal", [signal.SIGTERM, signal.SIGHUP])
def test_controlled_signal_unwinds_and_reaps_network_child(
    monkeypatch, tmp_path, delivered_signal
):
    module = _watcher_module()
    contract = _adjustment_contract(module, tmp_path, "sh.600000")
    installed = {}

    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(b"")
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -signal.SIGTERM

        def wait(self, timeout=None):
            assert timeout is not None
            return self.returncode

        def kill(self):
            raise AssertionError("cooperative child must not need SIGKILL")

    process = FakeProcess()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(module.signal, "getsignal", lambda signum: f"old-{signum}")

    def install(signum, handler):
        installed[signum] = handler

    monkeypatch.setattr(module.signal, "signal", install)

    def deliver_term(_seconds):
        installed[delivered_signal](delivered_signal, None)

    monkeypatch.setattr(module.time, "sleep", deliver_term)

    with pytest.raises(SystemExit) as error:
        module._run_network_phase_once(
            [sys.executable, "-c", "pass"],
            project=tmp_path,
            environment={},
            contract=contract,
            inactivity_timeout_seconds=30.0,
        )

    assert error.value.code == 128 + delivered_signal
    assert process.terminated is True
    assert installed[signal.SIGTERM] == f"old-{signal.SIGTERM}"
    assert installed[signal.SIGHUP] == f"old-{signal.SIGHUP}"


@pytest.mark.parametrize("delivered_signal", [signal.SIGTERM, signal.SIGHUP])
def test_signal_during_child_spawn_is_deferred_then_reaped(
    monkeypatch, tmp_path, delivered_signal
):
    module = _watcher_module()
    contract = _adjustment_contract(module, tmp_path, "sh.600000")
    installed = {}

    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(b"")
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -signal.SIGTERM

        def wait(self, timeout=None):
            assert timeout is not None
            return self.returncode

        def kill(self):
            raise AssertionError("cooperative child must not need SIGKILL")

    process = FakeProcess()
    monkeypatch.setattr(module.signal, "getsignal", lambda signum: f"old-{signum}")

    def install(signum, handler):
        installed[signum] = handler

    monkeypatch.setattr(module.signal, "signal", install)

    def spawn(*_args, **_kwargs):
        installed[delivered_signal](delivered_signal, None)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", spawn)

    with pytest.raises(SystemExit) as error:
        module._run_network_phase_once(
            [sys.executable, "-c", "pass"],
            project=tmp_path,
            environment={},
            contract=contract,
            inactivity_timeout_seconds=30.0,
        )

    assert error.value.code == 128 + delivered_signal
    assert process.terminated is True
    assert installed[signal.SIGTERM] == f"old-{signal.SIGTERM}"
    assert installed[signal.SIGHUP] == f"old-{signal.SIGHUP}"


def test_child_termination_reaps_after_process_lookup_race():
    module = _watcher_module()

    class VanishedProcess:
        def __init__(self):
            self.waits = 0

        def poll(self):
            return None

        def terminate(self):
            raise ProcessLookupError

        def wait(self, timeout=None):
            self.waits += 1
            return -signal.SIGTERM

    process = VanishedProcess()
    module._terminate_child_only(process, grace_seconds=0.01)
    assert process.waits == 1


def test_child_termination_uses_bounded_sigkill_fallback():
    module = _watcher_module()

    class StubbornProcess:
        def __init__(self):
            self.waits = 0
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["child"], timeout)
            return -signal.SIGKILL

        def kill(self):
            self.killed = True

    process = StubbornProcess()
    module._terminate_child_only(process, grace_seconds=0.01)
    assert process.killed is True
    assert process.waits == 2
