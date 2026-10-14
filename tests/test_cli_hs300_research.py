from types import SimpleNamespace

import pandas as pd

import cogalpha_plus.cli as cli
from cogalpha_plus.validation import SplitPlan


def _split():
    return SplitPlan(
        train_dates=(pd.Timestamp("2017-12-28"),),
        validation_dates=(pd.Timestamp("2019-12-27"),),
        sealed_dates=(pd.Timestamp("2020-01-20"),),
        purge_after_train=(pd.Timestamp("2018-01-02"),),
        purge_after_validation=(pd.Timestamp("2020-01-02"),),
        label_horizon=10,
        entry_delay=1,
    )


def test_hs300_research_cli_maps_frozen_full_run(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return SimpleNamespace(
                summary={
                    "historical_replay_revealed": False,
                    "sealed_market_rows_loaded_by_development": 0,
                    "sealed_reveal_manifest": str(
                        tmp_path / "sealed_reveal_manifest.json"
                    ),
                    "sealed_reveal_manifest_hash": "f" * 64,
                    "experiment_summary": {
                    "run_id": "run-1",
                    "evaluated_candidates": 2384,
                    "registry": {"success": 20},
                    "ledger": {"valid": True},
                    "champions": [],
                },
            },
            split=_split(),
            data_snapshot_hash="abc123",
            historical_replay_warning="known history is not true OOS",
        )

    monkeypatch.setattr(cli, "run_hs300_research", fake_run)
    project = tmp_path / "project"
    project.mkdir()
    code = cli.main(
        [
            "research",
            "hs300",
            "--project",
            str(project),
            "--panel-dir",
            "frozen-panel",
            "--output",
            "runs/full-v1",
        ]
    )
    assert code == 0
    config = captured["config"]
    assert config.panel_dir == (project / "frozen-panel").resolve()
    assert config.output_dir == (project / "runs" / "full-v1").resolve()
    assert config.label_horizon == 10
    assert config.bootstrap_samples == 200
    assert config.reveal_historical_replay is False
    output = capsys.readouterr().out
    assert '"evaluated_candidates": 2384' in output
    assert "not true OOS" in output
    assert '"sealed_market_rows_loaded_by_development": 0' in output


def test_hs300_research_parser_has_no_reduced_budget_switch():
    parser = cli.build_parser()
    args = parser.parse_args(["research", "hs300", "--output", "runs/full"])
    assert args.profile == "plus"
    assert args.horizon == 10
    assert args.bootstrap_samples == 200
    assert not hasattr(args, "budget")
    assert not hasattr(args, "reveal_historical_replay")


def test_hs300_reveal_cli_accepts_only_frozen_manifest(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_reveal(path):
        captured["path"] = path
        return SimpleNamespace(
            development_manifest_hash="d" * 64,
            data_snapshot_hash="s" * 64,
            historical_replay_warning="known history is not true OOS",
            split=_split(),
            results=({},),
            registry_record=tmp_path / "registry" / "gate.json",
            output_path=tmp_path / "sealed" / "result.json",
        )

    monkeypatch.setattr(cli, "reveal_hs300_replay", fake_reveal)
    manifest = tmp_path / "run" / "sealed_reveal_manifest.json"
    code = cli.main(
        ["research", "hs300-reveal", "--development-manifest", str(manifest)]
    )
    assert code == 0
    assert captured["path"] == manifest.resolve()
    args = cli.build_parser().parse_args(
        ["research", "hs300-reveal", "--development-manifest", str(manifest)]
    )
    assert vars(args).keys() == {
        "command",
        "research_command",
        "development_manifest",
        "func",
    }
    assert '"champions_revealed": 1' in capsys.readouterr().out
