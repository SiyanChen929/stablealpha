from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import pytest

from cogalpha_plus.data_sources.baostock import (
    EQUITY_DAILY_FIELDS,
    INDEX_DAILY_FIELDS,
    make_request_key,
)


PROJECT = Path(__file__).resolve().parents[1]


def _supervisor_module():
    path = PROJECT / "scripts" / "supervise_hs300_pipeline.py"
    spec = importlib.util.spec_from_file_location("hs300_pipeline_supervisor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules while decorating.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(module, project: Path, **overrides):
    values = {
        "project": project,
        "output": project / "runs" / "development",
        "price_attempts": 3,
        "provider_attempts": 2,
        "initial_backoff_seconds": 2.0,
        "maximum_backoff_seconds": 3.0,
    }
    values.update(overrides)
    return module.SupervisorConfig(**values)


def _write_watchdog_union(project: Path, *codes: str) -> None:
    normalized = project / "data" / "baostock_hs300" / "normalized"
    normalized.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": list(codes)}).to_parquet(
        normalized / "security_union.parquet",
        index=False,
    )


def _write_complete_manifest(
    module,
    config,
    *,
    union: int = 2,
):
    normalized = config.project / "data" / "baostock_hs300" / "normalized"
    normalized.mkdir(parents=True, exist_ok=True)
    codes = [f"sh.{600000 + number:06d}" for number in range(union)]
    pd.DataFrame({"code": codes, "code_name": codes}).to_parquet(
        normalized / "security_union.parquet", index=False
    )
    records = []
    provider_version = "0.9.3-supervisor-test"
    for code in codes:
        for flag, basis in (("1", "backward_adjusted"), ("3", "raw_unadjusted")):
            args = {
                "code": code,
                "fields": ",".join(EQUITY_DAILY_FIELDS),
                "start_date": config.start_date,
                "end_date": config.end_date,
                "frequency": "d",
                "adjustflag": flag,
            }
            request_key = make_request_key(
                "history_k_data", args, provider_version=provider_version
            )
            values = {
                "date": config.start_date,
                "code": code,
                "open": "10.0",
                "high": "10.2",
                "low": "9.9",
                "close": "10.1",
                "preclose": "10.0",
                "volume": "1000",
                "amount": "10000",
                "adjustflag": flag,
                "turn": "0.1",
                "tradestatus": "1",
                "pctChg": "1.0",
                "peTTM": "12",
                "pbMRQ": "1.2",
                "psTTM": "1.0",
                "pcfNcfTTM": "8",
                "isST": "0",
            }
            frame = pd.DataFrame(
                [[values[field] for field in EQUITY_DAILY_FIELDS]],
                columns=EQUITY_DAILY_FIELDS,
            )
            relative_object = (
                Path("objects") / "history_k_data" / f"{request_key}.parquet"
            )
            object_path = (
                config.project / "data" / "baostock_hs300" / "raw" / relative_object
            )
            object_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(object_path, index=False)
            object_hash = hashlib.sha256(object_path.read_bytes()).hexdigest()
            state_path = (
                config.project
                / "data"
                / "baostock_hs300"
                / "raw"
                / "state"
                / f"{request_key}.json"
            )
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "state_schema_version": 1,
                        "request_key": request_key,
                        "provider": "baostock",
                        "provider_version": provider_version,
                        "endpoint": "history_k_data",
                        "args": args,
                        "status": "success",
                        "fields": list(EQUITY_DAILY_FIELDS),
                        "rows": 1,
                        "object": str(relative_object),
                        "parquet_sha256": object_hash,
                    }
                ),
                encoding="utf-8",
            )
            records.append(
                {
                    "code": code,
                    "adjustflag": flag,
                    "basis": basis,
                    "rows": 1,
                    "first_date": config.start_date,
                    "last_date": config.start_date,
                    "request_key": request_key,
                    "cache_hit": False,
                }
            )
    benchmark_args = {
        "code": "sh.000300",
        "fields": ",".join(INDEX_DAILY_FIELDS),
        "start_date": config.start_date,
        "end_date": config.end_date,
        "frequency": "d",
        "adjustflag": "3",
    }
    benchmark_key = make_request_key(
        "index_history",
        benchmark_args,
        provider_version=provider_version,
    )
    benchmark_values = {
        "date": config.start_date,
        "code": "sh.000300",
        "open": "4000",
        "high": "4010",
        "low": "3990",
        "close": "4005",
        "preclose": "3995",
        "volume": "100000",
        "amount": "100000000",
        "pctChg": "0.25",
    }
    benchmark_frame = pd.DataFrame(
        [[benchmark_values[field] for field in INDEX_DAILY_FIELDS]],
        columns=INDEX_DAILY_FIELDS,
    )
    benchmark_relative = (
        Path("objects") / "index_history" / f"{benchmark_key}.parquet"
    )
    benchmark_path = (
        config.project / "data" / "baostock_hs300" / "raw" / benchmark_relative
    )
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_frame.to_parquet(benchmark_path, index=False)
    benchmark_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    benchmark_state = (
        config.project
        / "data"
        / "baostock_hs300"
        / "raw"
        / "state"
        / f"{benchmark_key}.json"
    )
    benchmark_state.write_text(
        json.dumps(
            {
                "state_schema_version": 1,
                "request_key": benchmark_key,
                "provider": "baostock",
                "provider_version": provider_version,
                "endpoint": "index_history",
                "args": benchmark_args,
                "status": "success",
                "fields": list(INDEX_DAILY_FIELDS),
                "rows": 1,
                "object": str(benchmark_relative),
                "parquet_sha256": benchmark_hash,
            }
        ),
        encoding="utf-8",
    )
    index_path = config.project / module.PRICE_INDEX_RELATIVE_PATH
    pd.DataFrame.from_records(records).to_parquet(index_path, index=False)
    content = index_path.read_bytes()
    manifest = {
        "phase": "price_catalog",
        "complete": True,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "historical_union_symbols": union,
        "downloaded_symbols": union,
        "price_requests": 2 * union,
        "index_request": {"request_key": benchmark_key, "cache_hit": False},
        "files": {
            "price_request_index": {
                "path": str(index_path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        },
    }
    manifest_path = config.project / module.PRICE_MANIFEST_RELATIVE_PATH
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _safe_plan(module, config):
    cli = [sys.executable, "-m", "cogalpha_plus.cli"]
    return [
        cli + arguments
        for arguments in module._expected_post_price_arguments(config)
    ]


def _write_runner_stub(module, config):
    path = config.project / module.POST_PRICE_RUNNER_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# trusted test runner\n", encoding="utf-8")


def test_price_retries_are_bounded_backed_off_and_resume_the_exact_command(
    monkeypatch, tmp_path
):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    commands = []
    sleeps = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if len(commands) == 2:
            _write_complete_manifest(module, config)
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 23)

    monkeypatch.setattr(module, "_run_streaming", fake_run)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    manifest = module._run_price_phase(config, environment={})

    assert manifest["complete"] is True
    assert commands == [module._price_command(config)] * 2
    assert sleeps == [2.0]
    assert "--max-attempts" in commands[0]
    assert commands[0][commands[0].index("--max-attempts") + 1] == "2"


def test_price_phase_stops_after_budget_and_caps_backoff(monkeypatch, tmp_path):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    calls = []
    sleeps = []

    def always_fail(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(module, "_run_streaming", always_fail)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="after 3 attempt"):
        module._run_price_phase(config, environment={})

    assert calls == [module._price_command(config)] * 3
    assert sleeps == [2.0, 3.0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_backoff_seconds", float("nan")),
        ("maximum_backoff_seconds", float("inf")),
        ("price_inactivity_timeout_seconds", float("nan")),
    ],
)
def test_retry_timing_must_be_finite(field, value):
    module = _supervisor_module()
    config = _config(module, PROJECT, **{field: value})
    with pytest.raises(ValueError, match="finite"):
        config.validate()


def test_noncanonical_price_dates_fail_before_any_process(tmp_path):
    module = _supervisor_module()
    config = _config(module, tmp_path, end_date="2024-12-31")
    with pytest.raises(ValueError, match="frozen post-price source contract"):
        config.validate()


def test_restart_with_complete_manifest_skips_all_price_processes(
    monkeypatch, tmp_path
):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    expected = _write_complete_manifest(module, config)

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("a complete frozen price catalog must not be rerun")

    monkeypatch.setattr(module, "_run_streaming", forbidden_run)
    actual = module._run_price_phase(config, environment={})

    assert actual == expected


def test_manifest_must_bind_dates_counts_path_size_and_hash(tmp_path):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    _write_complete_manifest(module, config)
    _write_runner_stub(module, config)
    manifest, reason = module._complete_price_manifest(config)
    assert manifest is not None and reason == "complete"

    index_path = config.project / module.PRICE_INDEX_RELATIVE_PATH
    index_path.write_bytes(b"changed")
    manifest, reason = module._complete_price_manifest(config)
    assert manifest is None
    assert reason in {
        "price_request_index_size_mismatch",
        "price_request_index_hash_mismatch",
    }


def test_manifest_rejects_duplicate_pairs_and_corrupt_cache_objects(tmp_path):
    module = _supervisor_module()
    duplicate_project = tmp_path / "duplicate"
    duplicate_config = _config(module, duplicate_project)
    _write_complete_manifest(module, duplicate_config)
    index_path = duplicate_project / module.PRICE_INDEX_RELATIVE_PATH
    index = pd.read_parquet(index_path)
    index.loc[1, ["code", "adjustflag"]] = index.loc[0, ["code", "adjustflag"]]
    index.to_parquet(index_path, index=False)
    manifest_path = duplicate_project / module.PRICE_MANIFEST_RELATIVE_PATH
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"]["price_request_index"].update(
        {
            "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "bytes": index_path.stat().st_size,
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest, reason = module._complete_price_manifest(duplicate_config)
    assert manifest is None and reason == "price_request_index_duplicate_pair"

    corrupt_project = tmp_path / "corrupt"
    corrupt_config = _config(module, corrupt_project)
    _write_complete_manifest(module, corrupt_config)
    corrupt_index = pd.read_parquet(corrupt_project / module.PRICE_INDEX_RELATIVE_PATH)
    request_key = str(corrupt_index.iloc[0]["request_key"])
    state_path = (
        corrupt_project
        / "data"
        / "baostock_hs300"
        / "raw"
        / "state"
        / f"{request_key}.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    object_path = corrupt_project / "data" / "baostock_hs300" / "raw" / state["object"]
    object_path.write_bytes(b"corrupt")
    manifest, reason = module._complete_price_manifest(corrupt_config)
    assert manifest is None
    assert reason == "price_cache_object_invalid:CacheIntegrityError"


def test_manifest_requires_bound_committed_hs300_benchmark(tmp_path):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    payload = _write_complete_manifest(module, config)
    payload.pop("index_request")
    manifest_path = tmp_path / module.PRICE_MANIFEST_RELATIVE_PATH
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest, reason = module._complete_price_manifest(config)

    assert manifest is None
    assert reason == "benchmark_index_request_missing"


def test_project_lock_refuses_a_second_owner(monkeypatch, tmp_path):
    module = _supervisor_module()
    lock_calls = []

    def contested(_fd, operation):
        lock_calls.append(operation)
        if operation & module.fcntl.LOCK_NB:
            raise BlockingIOError("busy")

    monkeypatch.setattr(module.fcntl, "flock", contested)
    with pytest.raises(module.SupervisorAlreadyRunning, match="already held"):
        with module._exclusive_project_lock(tmp_path):
            raise AssertionError("unreachable")

    assert lock_calls[0] == module.fcntl.LOCK_EX | module.fcntl.LOCK_NB


def test_supervise_uses_lock_skips_prices_and_starts_only_safe_runner(
    monkeypatch, tmp_path
):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    _write_complete_manifest(module, config)
    _write_runner_stub(module, config)
    lifecycle = []
    subprocesses = []

    @contextmanager
    def fake_lock(project):
        lifecycle.append(("lock", project))
        yield project / "fake.lock"
        lifecycle.append(("unlock", project))

    def fake_run(command, **_kwargs):
        subprocesses.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module, "_exclusive_project_lock", fake_lock)
    monkeypatch.setattr(module, "_load_post_price_plan", lambda _config: _safe_plan(module, config))
    monkeypatch.setattr(module, "_run_streaming", fake_run)

    assert module.supervise(config) == 0
    assert lifecycle == [("lock", tmp_path), ("unlock", tmp_path)]
    assert subprocesses == [module._post_price_command(config)]
    flattened = [token.lower() for token in subprocesses[0]]
    assert "hs300-reveal" not in flattened
    assert "trading" not in flattened


def test_price_inactivity_timeout_is_retryable_but_still_bounded(
    monkeypatch, tmp_path
):
    module = _supervisor_module()
    config = _config(
        module,
        tmp_path,
        price_attempts=1,
        price_inactivity_timeout_seconds=30.0,
    )

    def timeout(command, **kwargs):
        assert kwargs["timeout_seconds"] is None
        assert kwargs["inactivity_timeout_seconds"] == 30.0
        raise module.PriceInactivityTimeout(
            command,
            30.0,
            successful_requests=17,
        )

    monkeypatch.setattr(module, "_run_streaming", timeout)
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("no retry remains")),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "price_process_inactivity_timeout:30_seconds:"
            "successful_requests=17"
        ),
    ):
        module._run_price_phase(config, environment={})


def test_timeout_after_atomic_manifest_freeze_does_not_start_another_process(
    monkeypatch, tmp_path
):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    calls = []
    sleeps = []

    def froze_then_timed_out(command, **_kwargs):
        calls.append(list(command))
        _write_complete_manifest(module, config)
        raise module.PriceInactivityTimeout(
            command,
            config.price_inactivity_timeout_seconds,
            successful_requests=4,
        )

    monkeypatch.setattr(module, "_run_streaming", froze_then_timed_out)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    result = module._run_price_phase(config, environment={})

    assert result["complete"] is True
    assert calls == [module._price_command(config)]
    assert sleeps == []


def test_nonzero_price_exit_with_complete_manifest_does_not_spawn_again(
    monkeypatch, tmp_path
):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    calls = []

    def froze_then_exited(command, **_kwargs):
        calls.append(list(command))
        _write_complete_manifest(module, config)
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(module, "_run_streaming", froze_then_exited)
    result = module._run_price_phase(config, environment={})

    assert result["complete"] is True
    assert calls == [module._price_command(config)]


def test_inactivity_retry_preserves_committed_cache_and_exact_command(
    monkeypatch, tmp_path
):
    module = _supervisor_module()
    config = _config(
        module,
        tmp_path,
        price_inactivity_timeout_seconds=30.0,
    )
    state_dir = tmp_path / "data" / "baostock_hs300" / "raw" / "state"
    state_dir.mkdir(parents=True)
    committed = state_dir / ("f" * 64 + ".json")
    committed.write_text(
        json.dumps(
            {
                "request_key": "f" * 64,
                "provider": "baostock",
                "endpoint": "history_k_data",
                "args": {
                    "code": "sh.600000",
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "frequency": "d",
                    "adjustflag": "3",
                },
                "status": "success",
            }
        ),
        encoding="utf-8",
    )
    original_bytes = committed.read_bytes()
    calls = []

    def timeout_then_complete(command, **kwargs):
        calls.append((list(command), kwargs))
        assert committed.read_bytes() == original_bytes
        if len(calls) == 1:
            raise module.PriceInactivityTimeout(
                command,
                30.0,
                successful_requests=1,
            )
        _write_complete_manifest(module, config)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module, "_run_streaming", timeout_then_complete)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    result = module._run_price_phase(config, environment={})

    assert result["complete"] is True
    assert [call[0] for call in calls] == [module._price_command(config)] * 2
    assert all(call[1]["timeout_seconds"] is None for call in calls)
    assert all(
        call[1]["inactivity_timeout_seconds"] == 30.0 for call in calls
    )
    assert committed.read_bytes() == original_bytes


def test_catalog_is_revalidated_after_plan_load_and_before_runner(
    monkeypatch, tmp_path
):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    _write_complete_manifest(module, config)
    _write_runner_stub(module, config)

    @contextmanager
    def fake_lock(project):
        yield project / "fake.lock"

    def mutate_while_loading(_config):
        index = pd.read_parquet(tmp_path / module.PRICE_INDEX_RELATIVE_PATH)
        request_key = str(index.iloc[0]["request_key"])
        state_path = (
            tmp_path
            / "data"
            / "baostock_hs300"
            / "raw"
            / "state"
            / f"{request_key}.json"
        )
        state_path.unlink()
        return _safe_plan(module, config)

    monkeypatch.setattr(module, "_exclusive_project_lock", fake_lock)
    monkeypatch.setattr(module, "_load_post_price_plan", mutate_while_loading)
    monkeypatch.setattr(
        module,
        "_run_streaming",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("runner must not start after cache mutation")
        ),
    )

    with pytest.raises(RuntimeError, match="changed after completion"):
        module.supervise(config)


def test_benign_extra_downstream_argument_is_rejected(tmp_path):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    plan = _safe_plan(module, config)
    plan[-1].extend(["--seed", "999"])

    with pytest.raises(RuntimeError, match="frozen supervisor contract"):
        module._assert_safe_post_price_plan(
            plan,
            expected_arguments=module._expected_post_price_arguments(config),
        )


@pytest.mark.parametrize("mutation_point", ["plan_load", "final_manifest_check"])
def test_runner_bytes_are_bound_until_invocation(
    monkeypatch, tmp_path, mutation_point
):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    _write_complete_manifest(module, config)
    _write_runner_stub(module, config)
    runner_path = tmp_path / module.POST_PRICE_RUNNER_RELATIVE_PATH

    @contextmanager
    def fake_lock(project):
        yield project / "fake.lock"

    original_complete = module._complete_price_manifest
    complete_calls = 0

    def complete_with_optional_mutation(_config):
        nonlocal complete_calls
        complete_calls += 1
        result = original_complete(_config)
        if mutation_point == "final_manifest_check" and complete_calls == 2:
            runner_path.write_text("# changed during final check\n", encoding="utf-8")
        return result

    def plan_with_optional_mutation(_config):
        if mutation_point == "plan_load":
            runner_path.write_text("# changed during plan load\n", encoding="utf-8")
        return _safe_plan(module, config)

    monkeypatch.setattr(module, "_exclusive_project_lock", fake_lock)
    monkeypatch.setattr(module, "_complete_price_manifest", complete_with_optional_mutation)
    monkeypatch.setattr(module, "_load_post_price_plan", plan_with_optional_mutation)
    monkeypatch.setattr(
        module,
        "_run_streaming",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("changed runner must not start")
        ),
    )

    with pytest.raises(RuntimeError, match="post-price runner changed"):
        module.supervise(config)


def test_unsafe_downstream_plan_is_rejected_before_subprocess(
    monkeypatch, tmp_path
):
    module = _supervisor_module()
    config = _config(module, tmp_path)
    _write_complete_manifest(module, config)
    _write_runner_stub(module, config)
    unsafe = _safe_plan(module, config)
    unsafe[-1] = [
        sys.executable,
        "-m",
        "cogalpha_plus.cli",
        "research",
        "hs300-reveal",
    ]

    @contextmanager
    def fake_lock(project):
        yield project / "fake.lock"

    monkeypatch.setattr(module, "_exclusive_project_lock", fake_lock)
    monkeypatch.setattr(module, "_load_post_price_plan", lambda _config: unsafe)
    monkeypatch.setattr(
        module,
        "_run_streaming",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe plan must fail before subprocess")
        ),
    )

    with pytest.raises(RuntimeError, match="reveal or trading"):
        module.supervise(config)


def test_subprocess_output_is_streamed_and_stderr_is_merged(monkeypatch, tmp_path):
    module = _supervisor_module()
    observed = {}

    def fake_subprocess_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_subprocess_run)
    module._run_streaming(["python", "task.py"], project=tmp_path, environment={})

    assert observed["stdout"] is None
    assert observed["stderr"] == subprocess.STDOUT
    assert observed["check"] is False
    assert observed["cwd"] == tmp_path
    assert observed["timeout"] is None


def test_watchdog_allows_total_runtime_longer_than_timeout_when_state_advances(
    tmp_path,
):
    module = _supervisor_module()
    _write_watchdog_union(tmp_path, "sh.600000")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    provider_version = "0.9.3-watchdog-test"
    args = {
        "code": "sh.600000",
        "fields": ",".join(EQUITY_DAILY_FIELDS),
        "start_date": module.FROZEN_START_DATE,
        "end_date": module.FROZEN_END_DATE,
        "frequency": "d",
        "adjustflag": "3",
    }
    request_key = make_request_key(
        "history_k_data",
        args,
        provider_version=provider_version,
    )
    state_path = state_dir / f"{request_key}.json"
    base = {
        "state_schema_version": 1,
        "request_key": request_key,
        "provider": "baostock",
        "provider_version": provider_version,
        "endpoint": "history_k_data",
        "args": args,
    }
    script = "\n".join(
        [
            "import json, time",
            "from pathlib import Path",
            f"path = Path({str(state_path)!r})",
            f"base = {base!r}",
            "for attempt in range(1, 4):",
            "    path.write_text(json.dumps({**base, 'status': 'running', 'attempt_count': attempt}))",
            "    time.sleep(0.2)",
            f"path.write_text(json.dumps({{**base, 'status': 'success', 'attempt_count': 4, 'fields': {list(EQUITY_DAILY_FIELDS)!r}}}))",
            "time.sleep(0.2)",
        ]
    )

    result = module._run_with_price_inactivity_watchdog(
        [sys.executable, "-c", script],
        project=tmp_path,
        environment={},
        inactivity_timeout_seconds=0.5,
        state_dir=state_dir,
        start_date=module.FROZEN_START_DATE,
        end_date=module.FROZEN_END_DATE,
        poll_seconds=0.01,
    )

    assert result.returncode == 0
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"


def test_structured_cli_progress_prevents_false_inactivity_kill(tmp_path):
    module = _supervisor_module()
    _write_watchdog_union(tmp_path, "sh.600000")
    state_dir = tmp_path / "state"
    script = "\n".join(
        [
            "import json, time",
            "for number in range(1, 5):",
            "    print(json.dumps({'progress': number, 'total': 4, 'item': str(number)}), flush=True)",
            "    time.sleep(0.2)",
        ]
    )

    result = module._run_with_price_inactivity_watchdog(
        [sys.executable, "-c", script],
        project=tmp_path,
        environment={},
        inactivity_timeout_seconds=0.5,
        state_dir=state_dir,
        start_date=module.FROZEN_START_DATE,
        end_date=module.FROZEN_END_DATE,
        poll_seconds=0.01,
    )

    assert result.returncode == 0


def test_repeated_cli_progress_does_not_keep_stale_child_alive(tmp_path):
    module = _supervisor_module()
    _write_watchdog_union(tmp_path, "sh.600000")
    script = "\n".join(
        [
            "import json, time",
            "for _ in range(12):",
            "    print(json.dumps({'progress': 1, 'total': 10, 'item': 'same'}), flush=True)",
            "    time.sleep(0.05)",
        ]
    )

    with pytest.raises(module.PriceInactivityTimeout):
        module._run_with_price_inactivity_watchdog(
            [sys.executable, "-c", script],
            project=tmp_path,
            environment={},
            inactivity_timeout_seconds=0.15,
            state_dir=tmp_path / "state",
            start_date=module.FROZEN_START_DATE,
            end_date=module.FROZEN_END_DATE,
            poll_seconds=0.01,
        )


def test_noncanonical_state_changes_do_not_count_as_progress(tmp_path):
    module = _supervisor_module()
    _write_watchdog_union(tmp_path, "sh.600000")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / ("b" * 64 + ".json")
    script = "\n".join(
        [
            "import json, time",
            "from pathlib import Path",
            f"path = Path({str(state_path)!r})",
            "for attempt in range(12):",
            "    path.write_text(json.dumps({'request_key': 'wrong', 'provider': 'baostock', 'endpoint': 'history_k_data', 'status': 'running', 'attempt': attempt}))",
            "    time.sleep(0.05)",
        ]
    )

    with pytest.raises(module.PriceInactivityTimeout):
        module._run_with_price_inactivity_watchdog(
            [sys.executable, "-c", script],
            project=tmp_path,
            environment={},
            inactivity_timeout_seconds=0.15,
            state_dir=state_dir,
            start_date=module.FROZEN_START_DATE,
            end_date=module.FROZEN_END_DATE,
            poll_seconds=0.01,
        )


def test_valid_state_corruption_does_not_keep_stale_child_alive(tmp_path):
    module = _supervisor_module()
    _write_watchdog_union(tmp_path, "sh.600000")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    provider_version = "0.9.3-watchdog-test"
    args = {
        "code": "sh.600000",
        "fields": ",".join(EQUITY_DAILY_FIELDS),
        "start_date": module.FROZEN_START_DATE,
        "end_date": module.FROZEN_END_DATE,
        "frequency": "d",
        "adjustflag": "3",
    }
    request_key = make_request_key(
        "history_k_data",
        args,
        provider_version=provider_version,
    )
    state_path = state_dir / f"{request_key}.json"
    valid = {
        "state_schema_version": 1,
        "request_key": request_key,
        "provider": "baostock",
        "provider_version": provider_version,
        "endpoint": "history_k_data",
        "args": args,
        "attempt_count": 1,
        "status": "running",
    }
    script = "\n".join(
        [
            "import json, time",
            "from pathlib import Path",
            f"path = Path({str(state_path)!r})",
            f"valid = {valid!r}",
            "path.write_text(json.dumps(valid))",
            "time.sleep(0.05)",
            "for attempt in range(12):",
            "    path.write_text(json.dumps({**valid, 'request_key': 'corrupt', 'attempt_count': attempt + 2}))",
            "    time.sleep(0.05)",
        ]
    )

    with pytest.raises(module.PriceInactivityTimeout):
        module._run_with_price_inactivity_watchdog(
            [sys.executable, "-c", script],
            project=tmp_path,
            environment={},
            inactivity_timeout_seconds=0.2,
            state_dir=state_dir,
            start_date=module.FROZEN_START_DATE,
            end_date=module.FROZEN_END_DATE,
            poll_seconds=0.01,
        )


def test_thread_start_failure_reaps_price_child(monkeypatch, tmp_path):
    module = _supervisor_module()
    _write_watchdog_union(tmp_path, "sh.600000")
    real_popen = module.subprocess.Popen
    captured = {}

    def capture_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        captured["process"] = process
        return process

    monkeypatch.setattr(module.subprocess, "Popen", capture_process)
    monkeypatch.setattr(
        module.threading.Thread,
        "start",
        lambda _thread: (_ for _ in ()).throw(RuntimeError("thread start failed")),
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        module._run_with_price_inactivity_watchdog(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            project=tmp_path,
            environment={},
            inactivity_timeout_seconds=0.15,
            state_dir=tmp_path / "state",
            start_date=module.FROZEN_START_DATE,
            end_date=module.FROZEN_END_DATE,
            poll_seconds=0.01,
        )

    assert captured["process"].poll() is not None


def test_sigterm_unwinds_and_reaps_price_child(monkeypatch, tmp_path):
    module = _supervisor_module()
    _write_watchdog_union(tmp_path, "sh.600000")
    real_popen = module.subprocess.Popen
    captured = {}

    def capture_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        captured["process"] = process
        return process

    monkeypatch.setattr(module.subprocess, "Popen", capture_process)
    timer = module.threading.Timer(
        0.1,
        lambda: module.os.kill(module.os.getpid(), module.signal.SIGTERM),
    )
    timer.start()
    try:
        with pytest.raises(SystemExit) as raised:
            module._run_with_price_inactivity_watchdog(
                [sys.executable, "-c", "import time; time.sleep(2)"],
                project=tmp_path,
                environment={},
                inactivity_timeout_seconds=1.0,
                state_dir=tmp_path / "state",
                start_date=module.FROZEN_START_DATE,
                end_date=module.FROZEN_END_DATE,
                poll_seconds=0.01,
            )
    finally:
        timer.join()

    assert raised.value.code == 128 + module.signal.SIGTERM
    assert captured["process"].poll() is not None


def test_stale_child_is_terminated_after_inactivity(tmp_path):
    module = _supervisor_module()
    _write_watchdog_union(tmp_path, "sh.600000")
    started = time.monotonic()

    with pytest.raises(module.PriceInactivityTimeout) as raised:
        module._run_with_price_inactivity_watchdog(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            project=tmp_path,
            environment={},
            inactivity_timeout_seconds=0.15,
            state_dir=tmp_path / "state",
            start_date=module.FROZEN_START_DATE,
            end_date=module.FROZEN_END_DATE,
            poll_seconds=0.01,
        )

    assert time.monotonic() - started < 1.5
    assert raised.value.successful_requests == 0
    assert not any(
        thread.name == "hs300-price-output-relay" and thread.is_alive()
        for thread in module.threading.enumerate()
    )


@pytest.mark.parametrize("value", ["0", "29.9", "nan", "inf", "86401"])
def test_cli_rejects_unreasonable_inactivity_timeout(value):
    module = _supervisor_module()
    with pytest.raises(SystemExit):
        module._parser().parse_args(
            ["--price-inactivity-timeout-seconds", value]
        )


def test_cli_defaults_to_conservative_inactivity_timeout():
    module = _supervisor_module()
    args = module._parser().parse_args([])
    assert args.price_inactivity_timeout_seconds == 600.0


def test_removed_total_timeout_flag_is_not_silently_reinterpreted():
    module = _supervisor_module()
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--price-timeout-seconds", "600"])


def test_current_post_price_runner_plan_passes_strict_allowlist():
    module = _supervisor_module()
    config = _config(module, PROJECT)
    plan = module._load_post_price_plan(config)

    module._assert_safe_post_price_plan(
        plan,
        expected_arguments=module._expected_post_price_arguments(config),
    )
    flat = [str(token).lower() for command in plan for token in command]
    assert "hs300-reveal" not in flat
    assert "trading" not in flat
