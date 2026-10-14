from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cogalpha_plus import cli
from cogalpha_plus.data_contract import DataValidationReport
from cogalpha_plus.data_sources.hs300_build import HS300BuildResult
from cogalpha_plus.data_sources.hs300_audit import (
    AuditFinding,
    AuditSeverity,
    HS300AuditReport,
)


def _fake_result(tmp_path: Path) -> HS300BuildResult:
    output = tmp_path / "frozen"
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest_sha256_path = output / "manifest.sha256"
    manifest_sha256_path.write_text("a" * 64 + "  manifest.json\n", encoding="utf-8")
    report = DataValidationReport(
        valid=True,
        rows=40,
        symbols=3,
        trading_days=20,
        first_date="2020-01-02",
        last_date="2020-02-06",
        snapshot_hash="b" * 64,
        errors=(),
        warnings=("research_proxy_not_portfolio_grade",),
        metadata={"dataset_tier": "research_proxy"},
    )
    return HS300BuildResult(
        panel=pd.DataFrame(),
        validation=report,
        manifest_path=manifest_path,
        manifest_sha256_path=manifest_sha256_path,
        partition_paths=(output / "year=2020" / "part-000.parquet",),
        manifest={
            "research_proxy_only": True,
            "research_watermark": "research proxy only",
        },
    )


def test_data_build_maps_all_parameters_and_prints_manifest_summary(
    monkeypatch, tmp_path, capsys
):
    captured = {}
    result = _fake_result(tmp_path)

    def fake_build(config):
        captured["config"] = config
        return result

    monkeypatch.setattr(cli, "build_hs300_research_panel", fake_build)
    project = tmp_path / "project"
    return_code = cli.main(
        [
            "data",
            "build",
            "--project",
            str(project),
            "--data-root",
            "market-data",
            "--output-dir",
            "frozen-panel",
            "--research-start",
            "2020-01-02",
            "--research-end",
            "2020-02-06",
            "--expected-members",
            "3",
            "--adv-window",
            "7",
            "--adv-min-periods",
            "4",
            "--beta-window",
            "12",
            "--beta-min-periods",
            "5",
            "--minimum-history-days",
            "20",
            "--minimum-limit-rule-listed-days",
            "10",
            "--maximum-industry-age-days",
            "14",
        ]
    )

    assert return_code == 0
    config = captured["config"]
    assert config.cache_dir == (project / "market-data" / "raw").resolve()
    assert config.normalized_dir == (project / "market-data" / "normalized").resolve()
    assert config.output_dir == (project / "frozen-panel").resolve()
    assert config.research_start == "2020-01-02"
    assert config.research_end == "2020-02-06"
    assert config.expected_members == 3
    assert config.adv_window == 7
    assert config.adv_min_periods == 4
    assert config.beta_window == 12
    assert config.beta_min_periods == 5
    assert config.contract_minimum_history_days == 20
    assert config.minimum_limit_rule_listed_days == 10
    assert config.maximum_industry_age_days == 14

    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["dataset_tier"] == "research_proxy"
    assert payload["research_proxy_only"] is True
    assert payload["snapshot_hash"] == "b" * 64
    assert payload["manifest"] == str(result.manifest_path.resolve())
    assert payload["manifest_sha256"] == "a" * 64
    assert payload["manifest_sha256_file"] == str(result.manifest_sha256_path.resolve())
    assert payload["partitions"] == 1


def test_data_build_errors_are_nonzero_and_sent_to_stderr(monkeypatch, tmp_path, capsys):
    def fail_build(_config):
        raise FileNotFoundError("run the download phases first")

    monkeypatch.setattr(cli, "build_hs300_research_panel", fail_build)
    return_code = cli.main(
        ["data", "build", "--project", str(tmp_path), "--research-end", "2020-01-01"]
    )

    captured = capsys.readouterr()
    assert return_code == 2
    assert captured.out == ""
    assert "error: FileNotFoundError: run the download phases first" in captured.err


def test_data_build_rejects_invalid_build_configuration_before_io(tmp_path, capsys):
    return_code = cli.main(
        [
            "data",
            "build",
            "--project",
            str(tmp_path),
            "--beta-window",
            "4",
            "--beta-min-periods",
            "5",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 2
    assert captured.out == ""
    assert "error: ValueError: invalid beta_min_periods" in captured.err


def _fake_audit_report(tmp_path: Path, *, valid: bool) -> HS300AuditReport:
    finding = (
        ()
        if valid
        else (
            AuditFinding(
                severity=AuditSeverity.FAIL,
                code="daily_member_count_not_expected",
                message="bad daily member count",
                count=1,
            ),
        )
    )
    return HS300AuditReport(
        valid=valid,
        status="pass" if valid else "fail",
        panel_dir=str(tmp_path / "panel"),
        manifest_path=str(tmp_path / "panel" / "manifest.json"),
        rows=40,
        symbols=3,
        trading_days=20,
        first_date="2020-01-02",
        last_date="2020-02-06",
        expected_snapshot_hash="b" * 64,
        actual_snapshot_hash="b" * 64,
        research_valid=valid,
        portfolio_grade=False,
        capital_deployment_allowed=False,
        findings=finding,
        metrics={"membership": {"bad_dates": 0 if valid else 1}},
    )


def test_data_audit_maps_paths_thresholds_and_prints_json(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_audit(config):
        captured["config"] = config
        return _fake_audit_report(tmp_path, valid=True)

    monkeypatch.setattr(cli, "audit_hs300_research_panel", fake_audit)
    project = tmp_path / "project"
    return_code = cli.main(
        [
            "data",
            "audit",
            "--project",
            str(project),
            "--data-root",
            "market-data",
            "--panel-dir",
            "frozen-panel",
            "--normalized-dir",
            "catalogs",
            "--cache-dir",
            "cache",
            "--expected-members",
            "3",
            "--adjusted-price-tolerance",
            "1e-9",
            "--adjusted-return-warning-threshold",
            "0.25",
            "--adjusted-return-fail-threshold",
            "0.8",
            "--suspension-daily-warning-rate",
            "0.08",
            "--minimum-proxy-mcap-coverage",
            "1.0",
            "--maximum-samples",
            "4",
        ]
    )

    assert return_code == 0
    config = captured["config"]
    assert config.panel_dir == (project / "frozen-panel").resolve()
    assert config.normalized_dir == (project / "catalogs").resolve()
    assert config.cache_dir == (project / "cache").resolve()
    assert config.expected_members == 3
    assert config.adjusted_price_tolerance == 1e-9
    assert config.adjusted_return_warning_threshold == 0.25
    assert config.adjusted_return_fail_threshold == 0.8
    assert config.suspension_daily_warning_rate == 0.08
    assert config.minimum_proxy_mcap_coverage == 1.0
    assert config.maximum_samples == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["finding_counts"] == {"fail": 0, "warn": 0}


def test_data_audit_returns_one_and_still_prints_json_for_failed_gate(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        cli,
        "audit_hs300_research_panel",
        lambda _config: _fake_audit_report(tmp_path, valid=False),
    )

    return_code = cli.main(
        ["data", "audit", "--project", str(tmp_path), "--expected-members", "3"]
    )

    assert return_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["status"] == "fail"
    assert payload["finding_counts"]["fail"] == 1
