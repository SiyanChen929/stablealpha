from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_hs300_portfolio_replay as replay_script

from cogalpha_plus.portfolio_replay import (
    TargetBook,
    load_portfolio_replay_policy,
    prepare_target_book,
    score_target_book,
    target_book_hash,
)


PROJECT = Path(__file__).resolve().parents[1]
POLICY = PROJECT / "configs" / "hs300_portfolio_replay_v1.yaml"


def test_policy_freezes_primary_cost_and_non_execution_boundary():
    policy = load_portfolio_replay_policy(POLICY)
    assert policy.primary_cost_bps_one_way == 20.0
    assert policy.cost_sensitivity_bps_one_way == (10.0, 20.0, 40.0)
    assert policy.cash_execution_authorized is False
    assert policy.official_index_enhancement_claim_authorized is False
    assert policy.execution_delay_sessions == 1
    assert policy.holding_horizon_sessions == policy.rebalance_every_sessions


def test_score_target_book_applies_initial_and_subsequent_turnover_costs():
    policy = load_portfolio_replay_policy(POLICY)
    dates = (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-16"))
    executions = (pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-17"))
    index = pd.MultiIndex.from_product(
        [dates, ["A", "B"]], names=["signal_date", "symbol"]
    )
    targets = pd.DataFrame(
        {"target_weight": [0.6, 0.4, 0.5, 0.5]}, index=index
    )
    benchmarks = pd.DataFrame(
        {"benchmark_weight": [0.5, 0.5, 0.5, 0.5]}, index=index
    )
    diagnostics = pd.DataFrame(index=pd.Index(dates, name="signal_date"))
    book = TargetBook(
        targets,
        benchmarks,
        diagnostics,
        dates,
        executions,
        (pd.Timestamp("2020-01-17"), pd.Timestamp("2020-02-07")),
        policy.policy_hash,
    )
    label_index = pd.MultiIndex.from_product(
        [dates, ["A", "B"]], names=["date", "symbol"]
    )
    returns = pd.Series([0.10, 0.00, -0.02, 0.04], index=label_index)
    executable = returns.copy()
    result = score_target_book(book, returns, executable, policy)

    assert result.periods.iloc[0]["one_way_turnover"] == 1.0
    prior_drifted_a = 0.6 * 1.10 / (0.6 * 1.10 + 0.4)
    assert result.periods.iloc[1]["one_way_turnover"] == pytest.approx(
        abs(0.5 - prior_drifted_a)
    )
    first_gross = 0.6 * 0.10 + 0.4 * 0.00
    assert result.periods.iloc[0]["portfolio_net_return"] == pytest.approx(
        (1.0 - 0.002) * (1.0 + first_gross) - 1.0
    )
    assert result.summary["research_status"] == "NOT VALIDATED"


def _synthetic_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2019-07-01", periods=150)
    symbols = [f"S{i:03d}" for i in range(30)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    symbol_number = np.tile(np.arange(len(symbols)), len(dates))
    shocks = rng.normal(0.0002, 0.01, size=(len(dates), len(symbols)))
    close_matrix = 20.0 * np.exp(np.cumsum(shocks, axis=0))
    close = close_matrix.reshape(-1)
    panel = pd.DataFrame(index=index)
    panel["log_mcap"] = np.log(1.0e9 + symbol_number * 2.0e7)
    panel["beta"] = 0.8 + symbol_number / 100.0
    panel["amount"] = 2.0e8 + symbol_number * 1.0e6
    panel["industry"] = pd.Series(
        [f"I{i % 6}" for i in symbol_number], index=index, dtype="string"
    )
    panel["adv20_dollars"] = 2.0e8 + symbol_number * 1.0e6
    panel["tradable"] = True
    tape = panel.copy()
    tape["adj_close"] = close
    tape["adj_vwap"] = close * 1.0002
    alpha = pd.Series(
        np.sin(symbol_number / 4.0) + rng.normal(0, 0.05, len(index)),
        index=index,
        name="alpha",
    )
    return panel.sort_index(), tape.sort_index(), alpha.sort_index(), dates


def test_prepare_target_book_is_finite_constrained_and_outcome_free():
    policy = load_portfolio_replay_policy(POLICY)
    panel, tape, alpha, calendar = _synthetic_panel()
    # Keep the same 11-session outcome tail used by the real sealed split.
    sealed = calendar[-31:-11]
    book = prepare_target_book(alpha, panel, tape, calendar, sealed, policy)

    assert len(book.signal_dates) == 2
    sums = book.targets["target_weight"].groupby(level="signal_date").sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-6)
    assert float(book.targets["target_weight"].min()) >= -1e-12
    assert float(book.targets["target_weight"].max()) <= 0.080001
    assert float(book.diagnostics["maximum_constraint_violation"].max()) <= 1e-6
    assert len(target_book_hash(book)) == 64


def test_prepare_target_book_rejects_horizon_spacing_mismatch():
    policy = load_portfolio_replay_policy(POLICY)
    panel, tape, alpha, calendar = _synthetic_panel()
    with pytest.raises(ValueError, match="horizon equal rebalance spacing"):
        prepare_target_book(
            alpha,
            panel,
            tape,
            calendar,
            calendar[-31:-11],
            replace(policy, holding_horizon_sessions=5),
        )


def test_portfolio_plan_freezes_terminal_bridge_raw_bytes_and_identity(tmp_path):
    target = tmp_path / "plan.json"
    replay_script.freeze_plan(
        SimpleNamespace(policy=replay_script.DEFAULT_POLICY, output=target)
    )
    payload, _ = replay_script.load_envelope(
        target, kind="hs300_portfolio_replay_plan"
    )
    bridge = payload["terminal_bridge_catalog"]
    assert bridge["path"] == replay_script.BRIDGE_CATALOG_RELATIVE
    assert len(bridge["file_sha256"]) == 64
    assert len(bridge["identity"]["identity_hash"]) == 64
    assert replay_script.BRIDGE_CATALOG_RELATIVE in [
        item["path"] for item in payload["files"]
    ]
    replay_script.verify_plan(target)


def test_bridge_catalog_snapshot_rejects_note_only_byte_change(tmp_path, monkeypatch):
    copied = tmp_path / "bridges.yaml"
    copied.write_bytes(replay_script.BRIDGE_CATALOG.read_bytes())
    monkeypatch.setattr(replay_script, "BRIDGE_CATALOG", copied)
    _, frozen = replay_script.bridge_catalog_snapshot()
    copied.write_bytes(copied.read_bytes() + b"\n# changed after freeze\n")
    with pytest.raises(ValueError, match="changed after freeze"):
        replay_script.require_bridge_catalog_snapshot(frozen)


def test_activation_rejects_bridge_identity_different_from_development(
    tmp_path, monkeypatch
):
    _, commitment = replay_script.bridge_catalog_snapshot()
    plan = {"terminal_bridge_catalog": commitment}
    development_path = tmp_path / "development.json"
    development = {
        "operational_locators": {"sealed_output_dir": str(tmp_path / "out")},
        "resolved_experiment_config": {
            "terminal_label_bridge_identity": {
                **commitment["identity"],
                "identity_hash": "0" * 64,
            }
        },
    }
    monkeypatch.setattr(replay_script, "verify_plan", lambda _: (plan, "p" * 64))
    monkeypatch.setattr(
        replay_script.hs300,
        "_load_reveal_manifest",
        lambda _: (development_path, development, "d" * 64),
    )
    with pytest.raises(ValueError, match="differs from development commitment"):
        replay_script.activate(
            SimpleNamespace(plan=tmp_path / "plan.json", development_manifest=development_path)
        )


def test_verify_activation_rechecks_development_bridge_identity(tmp_path, monkeypatch):
    _, commitment = replay_script.bridge_catalog_snapshot()
    activation = {
        "plan_path": str(tmp_path / "plan.json"),
        "plan_hash": "p" * 64,
        "terminal_bridge_catalog": commitment,
        "development_manifest": str(tmp_path / "development.json"),
        "development_manifest_hash": "d" * 64,
    }
    development = {
        "resolved_experiment_config": {
            "terminal_label_bridge_identity": {
                **commitment["identity"],
                "identity_hash": "0" * 64,
            }
        }
    }
    monkeypatch.setattr(
        replay_script,
        "load_envelope",
        lambda *_args, **_kwargs: (activation, "a" * 64),
    )
    monkeypatch.setattr(
        replay_script,
        "verify_plan",
        lambda _: ({"terminal_bridge_catalog": commitment}, "p" * 64),
    )
    monkeypatch.setattr(
        replay_script.hs300,
        "_load_reveal_manifest",
        lambda _: (tmp_path / "development.json", development, "d" * 64),
    )
    with pytest.raises(ValueError, match="differs from development commitment"):
        replay_script.verify_activation(tmp_path / "activation.json")


def test_score_rejects_target_bridge_commitment_mismatch_before_reveal_access(
    tmp_path, monkeypatch
):
    policy = load_portfolio_replay_policy(POLICY)
    _, commitment = replay_script.bridge_catalog_snapshot()
    activation = {"terminal_bridge_catalog": commitment}
    target_payload = {
        "activation_hash": "a" * 64,
        "terminal_bridge_catalog": {**commitment, "file_sha256": "0" * 64},
    }
    monkeypatch.setattr(
        replay_script,
        "load_target_book",
        lambda *_: (object(), target_payload, "t" * 64),
    )
    monkeypatch.setattr(
        replay_script,
        "verify_activation",
        lambda _: (activation, "a" * 64, {}),
    )
    monkeypatch.setattr(
        replay_script,
        "load_portfolio_replay_policy",
        lambda _: policy,
    )
    with pytest.raises(ValueError, match="target book terminal bridge catalog mismatch"):
        replay_script.score(
            SimpleNamespace(
                target_book_manifest=tmp_path / "target.json",
                activation=tmp_path / "activation.json",
            )
        )
