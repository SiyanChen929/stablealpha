"""CogAlpha Track A paper-compatible, quality-first research protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yaml

from .hs300_research import (
    FrozenPanel,
    _load_bound_trading_dates,
    _load_frozen_panel_manifest,
    load_frozen_hs300_panel,
)
from .multiobjective import ParetoSelectionPolicy
from .paths import resolve_data_path
from .terminal_labels import load_terminal_label_bridges
from .validation import DevelopmentEvaluator, SplitPlan


TRACK_A_CONFIG_RELATIVE = Path("configs/track_a.yaml")
TRACK_A_REQUIRED_SEARCH = {
    "levels": 7,
    "roles": 21,
    "initial_population": 80,
    "parent_pool": 32,
    "children_per_generation": 96,
    "generations": 24,
    "subcycles": 3,
    "generations_per_subcycle": 8,
    "elite_carry": 2,
    "final_factors": 20,
}
TRACK_A_LIGHTGBM = {
    "objective": "regression",
    "learning_rate": 0.0001,
    "n_estimators": 1000,
    "num_leaves": 32,
    "max_depth": 12,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 3,
    "random_state": 17,
}
_TRACK_A_REQUIRED_LABEL = {
    "signal_time": "t_close",
    "entry": "open_t_plus_1",
    "exit": "open_t_plus_11",
    "formula": "adjusted_open_t_plus_11_div_adjusted_open_t_plus_1_minus_1",
    "missing_endpoint_policy": "missing_no_forward_fill",
    "terminal_bridge_use": "announced_symbol_exchange_exact_endpoint_only",
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrackAProtocol:
    project_dir: Path
    config_path: Path
    config_sha256: str
    raw: Mapping[str, object]
    panel_dir: Path
    panel_manifest_sha256: str
    pareto_policy: ParetoSelectionPolicy


@dataclass(frozen=True)
class TrackADevelopmentBundle:
    protocol: TrackAProtocol
    split: SplitPlan
    frozen: FrozenPanel


def _validate_track_a_label(label: Mapping[str, object]) -> None:
    if (
        type(label.get("horizon_trading_days")) is not int
        or label.get("horizon_trading_days") != 10
        or type(label.get("entry_delay_trading_days")) is not int
        or label.get("entry_delay_trading_days") != 1
        or any(
            label.get(key) != value
            for key, value in _TRACK_A_REQUIRED_LABEL.items()
        )
    ):
        raise ValueError("Track A label is not exact 10-day open-to-open")


def load_track_a_protocol(project_dir: Path | str) -> TrackAProtocol:
    project = Path(project_dir).resolve()
    config_path = project / TRACK_A_CONFIG_RELATIVE
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("profile_id") != "cogalpha_track_a_paper_compatible":
        raise ValueError("Track A profile identity is missing or invalid")
    data = raw.get("data")
    dates = raw.get("dates")
    label = raw.get("label")
    search = raw.get("factor_search")
    models = raw.get("models")
    portfolio = raw.get("portfolio")
    sandbox = raw.get("python_factor_sandbox")
    for name, section in (
        ("data", data),
        ("dates", dates),
        ("label", label),
        ("factor_search", search),
        ("models", models),
        ("portfolio", portfolio),
        ("python_factor_sandbox", sandbox),
    ):
        if not isinstance(section, Mapping):
            raise ValueError(f"Track A config section {name} is missing")
    assert isinstance(data, Mapping)
    tracks = data.get("tracks")
    if not isinstance(tracks, Mapping):
        raise ValueError("Track A data tracks are missing")
    primary = tracks.get("A1_baostock_pit_qlib_compatible")
    calibration = tracks.get("A0_official_qlib_frozen")
    if not isinstance(primary, Mapping) or not isinstance(calibration, Mapping):
        raise ValueError("Track A requires both A0 and A1 frozen tracks")
    panel_dir = resolve_data_path(project, str(primary.get("panel_dir", "")))
    manifest = panel_dir / "manifest.json"
    expected_manifest_hash = str(primary.get("panel_manifest_sha256", ""))
    if not manifest.is_file() or sha256_file(manifest) != expected_manifest_hash:
        raise ValueError("Track A A1 panel manifest hash mismatch")
    archive = resolve_data_path(project, str(calibration.get("archive", "")))
    if not archive.is_file() or sha256_file(archive) != str(
        calibration.get("archive_sha256", "")
    ):
        raise ValueError("Track A A0 archive hash mismatch")
    assert isinstance(dates, Mapping) and isinstance(label, Mapping)
    if {
        "train_start": str(dates.get("train_start")),
        "train_end": str(dates.get("train_end")),
        "validation_start": str(dates.get("validation_start")),
        "validation_end": str(dates.get("validation_end")),
        "test_start": str(dates.get("test_start")),
        "test_end": str(dates.get("test_end")),
    } != {
        "train_start": "2011-01-01",
        "train_end": "2019-12-31",
        "validation_start": "2020-01-01",
        "validation_end": "2020-12-31",
        "test_start": "2021-01-01",
        "test_end": "2024-12-31",
    }:
        raise ValueError("Track A date boundaries differ from the paper-compatible contract")
    _validate_track_a_label(label)
    bridge_path = project / str(label.get("terminal_bridge_catalog", ""))
    if (
        not bridge_path.is_file()
        or sha256_file(bridge_path)
        != str(label.get("terminal_bridge_catalog_sha256", ""))
    ):
        raise ValueError("Track A terminal bridge catalog hash mismatch")
    _, bridge_identity = load_terminal_label_bridges(bridge_path)
    if bridge_identity.get("identity_hash") != label.get(
        "terminal_bridge_identity_hash"
    ):
        raise ValueError("Track A terminal bridge identity mismatch")
    assert isinstance(search, Mapping)
    for key, value in TRACK_A_REQUIRED_SEARCH.items():
        if search.get(key) != value:
            raise ValueError(f"Track A search budget mismatch: {key}")
    if search.get("codex_feedback_each_generation") is not True:
        raise ValueError("Track A requires LLM feedback on every generation")
    selection = search.get("selection")
    if not isinstance(selection, Mapping) or selection.get("algorithm") != "deterministic_nsga2":
        raise ValueError("Track A requires deterministic Pareto/NSGA-II selection")
    objectives = tuple(selection.get("objectives_maximize", ()))
    pareto = ParetoSelectionPolicy(objectives=objectives)
    assert isinstance(models, Mapping)
    if int(models.get("refit_schedule_trading_days", 0)) != 126:
        raise ValueError("Track A model refit schedule must be 126 trading days")
    ridge = models.get("ridge")
    lightgbm = models.get("lightgbm")
    if not isinstance(ridge, Mapping) or float(ridge.get("alpha", 0.0)) != 10.0:
        raise ValueError("Track A Ridge alpha must equal 10")
    if not isinstance(lightgbm, Mapping) or dict(lightgbm) != TRACK_A_LIGHTGBM:
        raise ValueError("Track A LightGBM parameters differ from the frozen paper contract")
    assert isinstance(portfolio, Mapping)
    required_portfolio = {
        "selection": "top_50",
        "drop": 5,
        "entry_price": "next_open",
        "open_cost_rate": 0.0005,
        "close_cost_rate": 0.0015,
        "minimum_commission_cny": 5.0,
    }
    for key, expected in required_portfolio.items():
        if portfolio.get(key) != expected:
            raise ValueError(f"Track A portfolio contract mismatch: {key}")
    assert isinstance(sandbox, Mapping)
    if (
        sandbox.get("network") != "denied"
        or sandbox.get("filesystem") != "no_generated_code_access"
        or int(sandbox.get("workers", 0)) != 3
    ):
        raise ValueError("Track A sandbox boundary is not exact")
    return TrackAProtocol(
        project_dir=project,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        raw=raw,
        panel_dir=panel_dir,
        panel_manifest_sha256=expected_manifest_hash,
        pareto_policy=pareto,
    )


def make_track_a_split(
    trading_calendar: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> SplitPlan:
    """Build exact signal periods with tail purges before each next partition."""

    dates = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize().unique().sort_values()
    if dates.empty or dates.hasnans or dates.has_duplicates:
        raise ValueError("Track A calendar must be valid and unique")
    train_pool = dates[(dates >= "2011-01-01") & (dates <= "2019-12-31")]
    validation_pool = dates[(dates >= "2020-01-01") & (dates <= "2020-12-31")]
    sealed = dates[(dates >= "2021-01-01") & (dates <= "2024-12-31")]
    outcome_tail = dates[dates > "2024-12-31"][:11]
    purge = 11
    if min(len(train_pool), len(validation_pool), len(sealed)) < 40:
        raise ValueError("Track A calendar does not cover every research partition")
    if len(outcome_tail) != purge:
        raise ValueError("Track A calendar lacks the 11-session sealed outcome tail")
    train = train_pool[:-purge]
    purge_train = train_pool[-purge:]
    validation = validation_pool[:-purge]
    purge_validation = validation_pool[-purge:]
    split = SplitPlan(
        train_dates=tuple(map(pd.Timestamp, train)),
        validation_dates=tuple(map(pd.Timestamp, validation)),
        sealed_dates=tuple(map(pd.Timestamp, sealed)),
        purge_after_train=tuple(map(pd.Timestamp, purge_train)),
        purge_after_validation=tuple(map(pd.Timestamp, purge_validation)),
        label_horizon=10,
        entry_delay=1,
        outcome_tail=tuple(map(pd.Timestamp, outcome_tail)),
    )
    flattened = [
        *split.train_dates,
        *split.purge_after_train,
        *split.validation_dates,
        *split.purge_after_validation,
        *split.sealed_dates,
        *split.outcome_tail,
    ]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Track A split contains overlapping sessions")
    return split


def load_track_a_development(project_dir: Path | str) -> TrackADevelopmentBundle:
    """Load only development rows; sealed 2021+ prices remain unmaterialized."""

    protocol = load_track_a_protocol(project_dir)
    panel_root, manifest, manifest_hash = _load_frozen_panel_manifest(protocol.panel_dir)
    if manifest_hash != protocol.panel_manifest_sha256:
        raise ValueError("Track A manifest changed after protocol validation")
    calendar = _load_bound_trading_dates(panel_root, manifest)
    split = make_track_a_split(calendar)
    frozen = load_frozen_hs300_panel(
        protocol.panel_dir,
        end_date=split.development_end,
    )
    maximum_loaded = frozen.panel.index.get_level_values("date").max()
    if maximum_loaded != split.development_end or maximum_loaded >= split.sealed_dates[0]:
        raise RuntimeError("Track A development loader crossed the sealed boundary")
    return TrackADevelopmentBundle(protocol=protocol, split=split, frozen=frozen)


def make_track_a_development_evaluator(
    bundle: TrackADevelopmentBundle,
    *,
    bootstrap_samples: int = 200,
) -> DevelopmentEvaluator:
    bridge_path = bundle.protocol.project_dir / str(
        bundle.protocol.raw["label"]["terminal_bridge_catalog"]
    )
    bridges, bridge_identity = load_terminal_label_bridges(bridge_path)
    return DevelopmentEvaluator(
        bundle.frozen.eligible_panel,
        bundle.split,
        feature_tape=bundle.frozen.feature_tape,
        trading_calendar=bundle.frozen.trading_calendar,
        bootstrap_samples=bootstrap_samples,
        forecast_price_field="open",
        terminal_label_bridges=bridges,
        terminal_label_bridge_identity=bridge_identity,
        development_handoff_policy={
            "minimum_daily_label_coverage": 1.0,
            "allow_label_row_drop": False,
            "min_names": 20,
        },
    )


def split_identity(split: SplitPlan) -> dict[str, object]:
    payload = split.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"split": payload, "sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "TRACK_A_CONFIG_RELATIVE",
    "TrackADevelopmentBundle",
    "TrackAProtocol",
    "load_track_a_development",
    "load_track_a_protocol",
    "make_track_a_development_evaluator",
    "make_track_a_split",
    "sha256_file",
    "split_identity",
]
