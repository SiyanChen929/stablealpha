"""Sealed-safe development-to-portfolio research-proxy handoff.

This module is intentionally downstream from factor discovery.  It consumes
the immutable development/reveal manifest, recomputes the frozen champion ASTs
without loading a sealed row, freezes a pre-registered three-member composite
family without validation-driven model selection, and can emit constrained
long-only *target weights*.

It does not expose a P&L runner.  BaoStock's historical membership proxy lacks
official CSI300 weights and sourced corporate-action events, so an output from
this module is permanently labelled research-only and must not be represented
as a live or capital-grade portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .dsl import canonical_json as canonical_expression_json
from .dsl import evaluate_expression, expression_hash
from .ensemble import equal_date_sample_weights, normalize_factor_matrix, rank_ic_mean
from .manifest import canonical_json, hash_json
from .metrics import PreparedCrossSectionalNeutralizer, daily_ic, forward_return
from .portfolio import ConstrainedPortfolioOptimizer, OptimizationResult
from .runner import panel_snapshot_hash
from .terminal_labels import TerminalLabelBridge, apply_terminal_label_bridges
from .validation import (
    PRODUCTION_DSL_SPEC,
    SplitPlan,
    build_cross_sectional_mask,
    build_research_feature_frame,
    build_style_exposures,
)


RESEARCH_PROXY_TIER = "research_proxy_not_portfolio_grade"
RESEARCH_PROXY_DISCLAIMER = (
    "Research proxy only: benchmark weights are not official CSI300 weights; "
    "no event-driven P&L or live-capital claim is authorized without a complete "
    "official-weight history and sourced corporate-action ledger."
)
FROZEN_CHAMPION_COUNT = 20
REQUIRED_PORTFOLIO_STYLES = (
    "beta",
    "log_mcap",
    "momentum",
    "volatility",
    "liquidity",
)


@dataclass(frozen=True)
class DevelopmentHandoffPolicy:
    """Pre-registered completeness/integrity gates for the 20-column matrix."""

    min_names: int = 20
    minimum_daily_factor_coverage: float = 0.80
    maximum_zero_impute_share: float = 0.20
    maximum_absolute_neutral_correlation: float = 0.999
    allow_leading_train_warmup: bool = True
    minimum_retained_train_dates: int = 40
    minimum_daily_label_coverage: float = 1.0
    allow_label_row_drop: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ResearchProxyBoundaryError("unsupported handoff-policy schema")
        if self.min_names < 2:
            raise ResearchProxyBoundaryError("handoff min_names must be at least two")
        for name in (
            "minimum_daily_factor_coverage",
            "maximum_zero_impute_share",
            "maximum_absolute_neutral_correlation",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ResearchProxyBoundaryError(f"handoff policy {name} must be in [0,1]")
        if not np.isclose(
            self.maximum_zero_impute_share,
            1.0 - self.minimum_daily_factor_coverage,
            atol=1e-12,
        ):
            raise ResearchProxyBoundaryError(
                "coverage and zero-imputation policy thresholds must be complementary"
            )
        if self.allow_leading_train_warmup is not True:
            raise ResearchProxyBoundaryError(
                "v1 handoff policy requires explicit leading-train warm-up handling"
            )
        if self.minimum_retained_train_dates < 40:
            raise ResearchProxyBoundaryError(
                "handoff policy requires at least 40 retained train dates"
            )
        if not np.isclose(
            float(self.minimum_daily_label_coverage), 1.0, rtol=0.0, atol=0.0
        ):
            raise ResearchProxyBoundaryError(
                "v1 handoff policy requires 100% daily adjusted-VWAP label coverage"
            )
        if self.allow_label_row_drop is not False:
            raise ResearchProxyBoundaryError(
                "v1 handoff policy forbids selective label-row dropping"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "min_names": int(self.min_names),
            "minimum_daily_factor_coverage": float(
                self.minimum_daily_factor_coverage
            ),
            "maximum_zero_impute_share": float(self.maximum_zero_impute_share),
            "maximum_absolute_neutral_correlation": float(
                self.maximum_absolute_neutral_correlation
            ),
            "allow_leading_train_warmup": self.allow_leading_train_warmup,
            "minimum_retained_train_dates": int(self.minimum_retained_train_dates),
            "minimum_daily_label_coverage": float(
                self.minimum_daily_label_coverage
            ),
            "allow_label_row_drop": self.allow_label_row_drop,
        }

    @property
    def policy_hash(self) -> str:
        return hash_json(self.to_dict())


@dataclass(frozen=True)
class CompositeFamilyPolicy:
    """Externally pre-registrable, canonical policy for the full model family."""

    primary_backend: str
    members: tuple[str, ...]
    ridge_alpha: float
    random_state: int
    lightgbm_configuration: Mapping[str, object]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ResearchProxyBoundaryError("unsupported composite-policy schema")
        exact_members = ("equal_weight", "ridge", "lightgbm")
        if self.members != exact_members:
            raise ResearchProxyBoundaryError(
                "composite policy must contain the exact ordered three-model family"
            )
        if self.primary_backend != "lightgbm":
            raise ResearchProxyBoundaryError(
                "v1 composite policy pre-registers LightGBM as the unique primary"
            )
        if not np.isclose(float(self.ridge_alpha), 1.0, rtol=0.0, atol=0.0):
            raise ResearchProxyBoundaryError(
                "v1 composite policy fixes Ridge alpha at 1.0"
            )
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ResearchProxyBoundaryError("policy random_state must be an integer")
        if self.random_state != 17:
            raise ResearchProxyBoundaryError(
                "v1 composite policy fixes random_state at 17"
            )
        config = dict(self.lightgbm_configuration)
        expected = dict(_FixedLightGBMComposite.CONFIGURATION)
        expected["random_state"] = int(self.random_state)
        if config != expected:
            raise ResearchProxyBoundaryError(
                "LightGBM policy differs from the fixed deterministic configuration"
            )
        object.__setattr__(self, "lightgbm_configuration", MappingProxyType(config))

    @classmethod
    def fixed_v1(
        cls,
    ) -> "CompositeFamilyPolicy":
        config = dict(_FixedLightGBMComposite.CONFIGURATION)
        config["random_state"] = 17
        return cls(
            primary_backend="lightgbm",
            members=("equal_weight", "ridge", "lightgbm"),
            ridge_alpha=1.0,
            random_state=17,
            lightgbm_configuration=config,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "primary_backend": self.primary_backend,
            "members": list(self.members),
            "ridge_alpha": float(self.ridge_alpha),
            "random_state": int(self.random_state),
            "lightgbm_configuration": dict(self.lightgbm_configuration),
        }

    @property
    def policy_hash(self) -> str:
        return hash_json(self.to_dict())


class ResearchProxyBoundaryError(ValueError):
    """A research-only or sealed-data boundary would be crossed."""


def _require_panel_index(obj: pd.Series | pd.DataFrame, *, label: str) -> None:
    if not isinstance(obj.index, pd.MultiIndex) or tuple(obj.index.names) != (
        "date",
        "symbol",
    ):
        raise ResearchProxyBoundaryError(
            f"{label} must use MultiIndex(date, symbol)"
        )
    if obj.index.has_duplicates or not obj.index.is_monotonic_increasing:
        raise ResearchProxyBoundaryError(f"{label} must be sorted and unique")


def _date_slice(
    obj: pd.Series | pd.DataFrame,
    dates: Sequence[pd.Timestamp],
) -> pd.Series | pd.DataFrame:
    mask = obj.index.get_level_values("date").isin(dates)
    return obj.loc[mask]


def _exact_development_member_dates(split: SplitPlan) -> tuple[pd.Timestamp, ...]:
    """Return the only date sequence permitted in a development member panel."""

    dates = (
        tuple(split.train_dates)
        + tuple(split.purge_after_train)
        + tuple(split.validation_dates)
        + tuple(split.purge_after_validation)
    )
    if not dates or len(dates) != len(set(dates)):
        raise ResearchProxyBoundaryError(
            "development split partitions overlap or are empty"
        )
    if tuple(sorted(dates)) != dates:
        raise ResearchProxyBoundaryError(
            "development split partitions are not strictly chronological"
        )
    return dates


def _series_snapshot_hash(series: pd.Series) -> str:
    return panel_snapshot_hash(series.rename(series.name or "value").to_frame())


def _require_finite_series(
    values: pd.Series,
    *,
    label: str,
    positive: bool = False,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ResearchProxyBoundaryError(f"{label} must be finite and complete")
    if positive and np.any(array <= 0.0):
        raise ResearchProxyBoundaryError(f"{label} must be strictly positive")
    return numeric.astype(float)


@dataclass(frozen=True)
class FrozenChampionCommitment:
    candidate_id: str
    expression: Mapping[str, object]
    canonical_expression_json: str
    expression_hash: str
    direction: int
    validation_neutral_rank_ic: float

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "FrozenChampionCommitment":
        candidate_id = str(record.get("candidate_id", ""))
        if not candidate_id:
            raise ResearchProxyBoundaryError("frozen champion has a blank candidate_id")
        ast = record.get("canonical_expression_ast")
        canonical = canonical_expression_json(ast, PRODUCTION_DSL_SPEC)
        digest = expression_hash(ast, PRODUCTION_DSL_SPEC)
        if canonical != str(record.get("canonical_expression_json", "")):
            raise ResearchProxyBoundaryError(
                f"frozen champion {candidate_id!r} canonical AST differs from manifest"
            )
        if digest != str(record.get("expression_hash", "")):
            raise ResearchProxyBoundaryError(
                f"frozen champion {candidate_id!r} expression hash differs from manifest"
            )
        direction = record.get("direction")
        if direction not in {-1, 1}:
            raise ResearchProxyBoundaryError(
                f"frozen champion {candidate_id!r} has no train-only direction"
            )
        score = record.get("score")
        if not isinstance(score, Mapping) or score.get("passed") is not True:
            raise ResearchProxyBoundaryError(
                f"frozen champion {candidate_id!r} is not a passed development result"
            )
        neutral_ic = float(record.get("validation_neutral_rank_ic", np.nan))
        if not np.isfinite(neutral_ic):
            raise ResearchProxyBoundaryError(
                f"frozen champion {candidate_id!r} has no finite neutral validation IC"
            )
        return cls(
            candidate_id=candidate_id,
            expression=json.loads(canonical),
            canonical_expression_json=canonical,
            expression_hash=digest,
            direction=int(direction),
            validation_neutral_rank_ic=neutral_ic,
        )


@dataclass(frozen=True)
class FrozenDevelopmentCommitment:
    """Authenticated development-only inputs needed by the downstream handoff."""

    development_root_hash: str
    champion_family_hash: str
    split_identity_hash: str
    split: SplitPlan
    development_member_panel_hash: str
    development_feature_tape_hash: str
    research_universe_identity_hash: str
    trading_calendar_identity_hash: str
    source_lineage_identity_hash: str
    source_snapshot_identity_hash: str
    champions: tuple[FrozenChampionCommitment, ...]
    sealed_rows_loaded: int = 0

    def __post_init__(self) -> None:
        for name in (
            "development_root_hash",
            "champion_family_hash",
            "split_identity_hash",
            "development_member_panel_hash",
            "development_feature_tape_hash",
            "research_universe_identity_hash",
            "trading_calendar_identity_hash",
            "source_lineage_identity_hash",
            "source_snapshot_identity_hash",
        ):
            value = str(getattr(self, name))
            if len(value) != 64:
                raise ResearchProxyBoundaryError(f"{name} is not a SHA-256 identity")
        if self.sealed_rows_loaded != 0:
            raise ResearchProxyBoundaryError(
                "development commitment reports sealed rows already loaded"
            )
        if len(self.champions) != FROZEN_CHAMPION_COUNT:
            raise ResearchProxyBoundaryError(
                f"handoff requires exactly {FROZEN_CHAMPION_COUNT} frozen champions"
            )
        ids = [item.candidate_id for item in self.champions]
        digests = [item.expression_hash for item in self.champions]
        if len(ids) != len(set(ids)) or len(digests) != len(set(digests)):
            raise ResearchProxyBoundaryError(
                "frozen champion family contains duplicate ids or expressions"
            )

    @classmethod
    def from_manifest(cls, path: Path | str) -> "FrozenDevelopmentCommitment":
        """Load the same authenticated envelope accepted by the one-time gate.

        The private helpers are used deliberately: they validate the completion
        evidence, run manifest, implementation snapshot, split, candidate
        identities and champion-family hash without materializing sealed data.
        """

        from .hs300_research import (  # Imported lazily to keep this downstream.
            _champions_from_frozen_manifest,
            _load_reveal_manifest,
            _split_from_payload,
        )

        _, payload, _ = _load_reveal_manifest(path)
        records = payload.get("champions")
        if not isinstance(records, list):
            raise ResearchProxyBoundaryError("development manifest has no champion family")
        # Authenticate deterministic Candidate identities in addition to the
        # scalar/AST checks below.  No values or sealed rows are reconstructed.
        _champions_from_frozen_manifest(records)
        if len(records) != FROZEN_CHAMPION_COUNT:
            raise ResearchProxyBoundaryError(
                f"development manifest must commit exactly {FROZEN_CHAMPION_COUNT} champions"
            )
        if hash_json(records) != str(payload.get("champion_family_hash", "")):
            raise ResearchProxyBoundaryError("champion-family hash mismatch")
        split_payload = payload.get("split_exact")
        split = _split_from_payload(split_payload)
        if hash_json(split_payload) != str(payload.get("split_identity_hash", "")):
            raise ResearchProxyBoundaryError("split identity hash mismatch")
        identity = payload.get("development_view_identity")
        if not isinstance(identity, Mapping):
            raise ResearchProxyBoundaryError("development manifest omits view identity")
        if identity.get("sealed_market_rows_loaded") != 0:
            raise ResearchProxyBoundaryError(
                "development view does not attest a zero-row sealed boundary"
            )
        return cls(
            development_root_hash=str(payload["development_run_manifest_hash"]),
            champion_family_hash=str(payload["champion_family_hash"]),
            split_identity_hash=str(payload["split_identity_hash"]),
            split=split,
            development_member_panel_hash=str(
                identity.get("member_panel_snapshot_hash", "")
            ),
            development_feature_tape_hash=str(
                identity.get("feature_tape_snapshot_hash", "")
            ),
            research_universe_identity_hash=str(
                identity.get("research_universe_identity_hash", "")
            ),
            trading_calendar_identity_hash=str(
                identity.get("trading_calendar_identity_hash", "")
            ),
            source_lineage_identity_hash=str(
                identity.get("source_lineage_identity_hash", "")
            ),
            source_snapshot_identity_hash=str(
                identity.get("source_snapshot_identity_hash", "")
            ),
            champions=tuple(
                FrozenChampionCommitment.from_record(record) for record in records
            ),
            sealed_rows_loaded=0,
        )

    @classmethod
    def from_records(
        cls,
        *,
        development_root_hash: str,
        champion_family_hash: str,
        split_identity_hash: str,
        split: SplitPlan,
        development_member_panel_hash: str,
        development_feature_tape_hash: str,
        research_universe_identity_hash: str,
        trading_calendar_identity_hash: str,
        source_lineage_identity_hash: str,
        source_snapshot_identity_hash: str,
        champion_records: Sequence[Mapping[str, object]],
    ) -> "FrozenDevelopmentCommitment":
        """Construct the pre-manifest handoff from already frozen dev records."""

        if len(champion_records) != FROZEN_CHAMPION_COUNT:
            raise ResearchProxyBoundaryError(
                f"in-memory handoff requires {FROZEN_CHAMPION_COUNT} champions"
            )
        if hash_json(list(champion_records)) != champion_family_hash:
            raise ResearchProxyBoundaryError(
                "in-memory champion records differ from their family hash"
            )
        return cls(
            development_root_hash=development_root_hash,
            champion_family_hash=champion_family_hash,
            split_identity_hash=split_identity_hash,
            split=split,
            development_member_panel_hash=development_member_panel_hash,
            development_feature_tape_hash=development_feature_tape_hash,
            research_universe_identity_hash=research_universe_identity_hash,
            trading_calendar_identity_hash=trading_calendar_identity_hash,
            source_lineage_identity_hash=source_lineage_identity_hash,
            source_snapshot_identity_hash=source_snapshot_identity_hash,
            champions=tuple(
                FrozenChampionCommitment.from_record(record)
                for record in champion_records
            ),
            sealed_rows_loaded=0,
        )


@dataclass(frozen=True)
class DevelopmentFeatureHandoff:
    train_features: pd.DataFrame
    validation_features: pd.DataFrame
    train_labels: pd.Series
    validation_labels: pd.Series
    coverage_audit: pd.DataFrame
    champion_ids: tuple[str, ...]
    provenance: Mapping[str, object]
    tier: str = RESEARCH_PROXY_TIER
    disclaimer: str = RESEARCH_PROXY_DISCLAIMER

    def __post_init__(self) -> None:
        if self.tier != RESEARCH_PROXY_TIER:
            raise ResearchProxyBoundaryError("research handoff tier cannot be upgraded")
        if len(self.champion_ids) != FROZEN_CHAMPION_COUNT:
            raise ResearchProxyBoundaryError("research handoff does not contain 20 factors")
        if tuple(self.train_features.columns) != self.champion_ids:
            raise ResearchProxyBoundaryError("train feature columns differ from frozen family")
        if tuple(self.validation_features.columns) != self.champion_ids:
            raise ResearchProxyBoundaryError(
                "validation feature columns differ from frozen family"
            )
        if self.provenance.get("sealed_rows_loaded") != 0:
            raise ResearchProxyBoundaryError("handoff provenance consumed sealed rows")
        if self.coverage_audit.empty:
            raise ResearchProxyBoundaryError("handoff omits factor/date coverage audit")
        self.verify_integrity()

    def verify_integrity(self) -> None:
        for label, frame in (
            ("handoff train features", self.train_features),
            ("handoff validation features", self.validation_features),
            ("handoff train labels", self.train_labels),
            ("handoff validation labels", self.validation_labels),
        ):
            _require_panel_index(frame, label=label)
        if not self.train_features.index.equals(self.train_labels.index):
            raise ResearchProxyBoundaryError("train feature/label indices differ")
        if not self.validation_features.index.equals(self.validation_labels.index):
            raise ResearchProxyBoundaryError("validation feature/label indices differ")
        expected_hashes = {
            "train_features_hash": panel_snapshot_hash(self.train_features),
            "validation_features_hash": panel_snapshot_hash(
                self.validation_features
            ),
            "train_labels_hash": _series_snapshot_hash(self.train_labels),
            "validation_labels_hash": _series_snapshot_hash(
                self.validation_labels
            ),
        }
        for key, actual in expected_hashes.items():
            if self.provenance.get(key) != actual:
                raise ResearchProxyBoundaryError(f"handoff mutated or omits {key}")
        neutral_audit = self.provenance.get("neutral_feature_audit")
        if (
            not isinstance(neutral_audit, Mapping)
            or neutral_audit.get("coverage_audit_hash")
            != panel_snapshot_hash(self.coverage_audit)
        ):
            raise ResearchProxyBoundaryError(
                "handoff coverage audit differs from its commitment"
            )
        for key in (
            "development_root_hash",
            "champion_family_hash",
            "split_identity_hash",
            "development_member_panel_hash",
            "development_feature_tape_hash",
            "research_universe_identity_hash",
            "trading_calendar_identity_hash",
            "source_lineage_identity_hash",
            "source_snapshot_identity_hash",
        ):
            if len(str(self.provenance.get(key, ""))) != 64:
                raise ResearchProxyBoundaryError(f"handoff omits valid {key}")
        for name in (
            "research_universe_identity",
            "trading_calendar_identity",
            "source_lineage_identity",
            "source_snapshot_identity",
        ):
            identity = self.provenance.get(name)
            expected = self.provenance.get(f"{name}_hash")
            if (
                not isinstance(identity, Mapping)
                or identity.get("identity_hash") != expected
                or hash_json(
                    {key: value for key, value in identity.items() if key != "identity_hash"}
                )
                != expected
            ):
                raise ResearchProxyBoundaryError(f"handoff {name} is not bound")
        raw_handoff_policy = self.provenance.get("development_handoff_policy")
        if not isinstance(raw_handoff_policy, Mapping):
            raise ResearchProxyBoundaryError("handoff omits its frozen policy")
        handoff_policy = DevelopmentHandoffPolicy(**dict(raw_handoff_policy))
        if handoff_policy.policy_hash != self.provenance.get(
            "development_handoff_policy_hash"
        ):
            raise ResearchProxyBoundaryError("handoff policy hash mismatch")
        label_audit = self.provenance.get("development_label_audit")
        if (
            not isinstance(label_audit, Mapping)
            or label_audit.get("policy_hash") != handoff_policy.policy_hash
            or label_audit.get("all_passed") is not True
            or label_audit.get("allow_label_row_drop") is not False
            or not isinstance(label_audit.get("all_development"), Mapping)
            or label_audit["all_development"].get("eligible_rows")
            != label_audit["all_development"].get("valid_rows")
        ):
            raise ResearchProxyBoundaryError(
                "handoff development-label audit is incomplete"
            )
        expressions = self.provenance.get("expression_hashes")
        directions = self.provenance.get("directions")
        if (
            not isinstance(expressions, Sequence)
            or isinstance(expressions, (str, bytes))
            or len(expressions) != FROZEN_CHAMPION_COUNT
            or not all(len(str(value)) == 64 for value in expressions)
            or not isinstance(directions, Mapping)
            or tuple(directions) != self.champion_ids
            or any(value not in {-1, 1} for value in directions.values())
        ):
            raise ResearchProxyBoundaryError(
                "handoff champion expressions/directions are incomplete"
            )


def _audit_neutral_feature_matrix(
    features: pd.DataFrame,
    policy: DevelopmentHandoffPolicy,
    *,
    train_dates: Sequence[pd.Timestamp],
    validation_dates: Sequence[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, object], tuple[pd.Timestamp, ...]]:
    records: list[dict[str, object]] = []
    date_passes: dict[pd.Timestamp, bool] = {}
    for date, group in features.groupby(level="date", sort=True):
        finite = group.replace([np.inf, -np.inf], np.nan)
        total = len(group)
        for name in features.columns:
            values = pd.to_numeric(finite[name], errors="coerce").dropna()
            valid = int(len(values))
            coverage = float(valid / total) if total else 0.0
            zero_share = float(1.0 - coverage)
            constant = valid < 2 or values.nunique() < 2
            passes = bool(
                valid >= policy.min_names
                and coverage >= policy.minimum_daily_factor_coverage - 1e-12
                and zero_share <= policy.maximum_zero_impute_share + 1e-12
                and not constant
            )
            records.append(
                {
                    "date": pd.Timestamp(date),
                    "factor": str(name),
                    "valid_names": valid,
                    "total_names": int(total),
                    "coverage": coverage,
                    "zero_impute_share": zero_share,
                    "constant_or_all_missing": bool(constant),
                    "passes_gate": passes,
                }
            )
        date_passes[pd.Timestamp(date)] = bool(
            all(record["passes_gate"] for record in records[-len(features.columns) :])
        )
    audit = pd.DataFrame.from_records(records).set_index(["date", "factor"]).sort_index()
    train = tuple(pd.Timestamp(value) for value in train_dates)
    validation = tuple(pd.Timestamp(value) for value in validation_dates)
    first_good = next((position for position, date in enumerate(train) if date_passes.get(date, False)), None)
    if first_good is None:
        raise ResearchProxyBoundaryError("no training date survives the neutral warm-up gate")
    dropped_train = train[:first_good]
    retained_train = train[first_good:]
    later_train_failures = [date for date in retained_train if not date_passes.get(date, False)]
    if later_train_failures:
        raise ResearchProxyBoundaryError(
            "neutral feature coverage failed after warm-up; first="
            + str(later_train_failures[0].date())
        )
    validation_failures = [date for date in validation if not date_passes.get(date, False)]
    if validation_failures:
        raise ResearchProxyBoundaryError(
            "inner-validation feature coverage is incomplete; first="
            + str(validation_failures[0].date())
        )
    if len(retained_train) < policy.minimum_retained_train_dates:
        raise ResearchProxyBoundaryError(
            "too few training dates remain after the deterministic warm-up cut"
        )
    keep_dates = set(retained_train) | set(validation)
    kept_features = features.loc[
        features.index.get_level_values("date").isin(keep_dates)
    ]
    correlations = [
        group.replace([np.inf, -np.inf], np.nan).corr(
            method="spearman", min_periods=policy.min_names
        )
        for _, group in kept_features.groupby(level="date", sort=True)
    ]
    stacked = np.stack(
        [item.reindex(index=features.columns, columns=features.columns).to_numpy(dtype=float) for item in correlations]
    )
    with np.errstate(invalid="ignore"):
        mean_correlation = np.nanmean(stacked, axis=0)
    off_diagonal = ~np.eye(len(features.columns), dtype=bool)
    finite_off = np.abs(mean_correlation[off_diagonal])
    finite_off = finite_off[np.isfinite(finite_off)]
    maximum = float(finite_off.max()) if finite_off.size else np.nan
    if not np.isfinite(maximum):
        raise ResearchProxyBoundaryError("neutral factor-correlation audit is undefined")
    if maximum > policy.maximum_absolute_neutral_correlation + 1e-12:
        raise ResearchProxyBoundaryError(
            "neutral factor family exceeds the pre-registered correlation ceiling"
        )
    by_factor = audit.groupby(level="factor")[["coverage", "zero_impute_share"]].agg(
        ["min", "mean", "max"]
    )
    retained_index = pd.DataFrame(
        {"date": list(retained_train), "retained": True}
    ).set_index("date")
    return audit, {
        "policy": policy.to_dict(),
        "policy_hash": policy.policy_hash,
        "coverage_audit_hash": panel_snapshot_hash(audit),
        "factor_coverage_summary": json.loads(by_factor.to_json(orient="index")),
        "maximum_absolute_mean_daily_neutral_spearman": maximum,
        "mean_daily_neutral_correlation_matrix_hash": panel_snapshot_hash(
            pd.DataFrame(
                mean_correlation,
                index=features.columns,
                columns=features.columns,
            )
        ),
        "zero_imputation_stage": "after_same_date_neutral_residual_rank_only",
        "warmup_policy": "drop_only_leading_train_dates_before_all_20_pass;validation_never_dropped",
        "warmup_dropped_train_dates": [str(value.date()) for value in dropped_train],
        "warmup_dropped_train_date_count": len(dropped_train),
        "retained_train_dates_hash": panel_snapshot_hash(retained_index),
        "retained_train_date_count": len(retained_train),
    }, retained_train


def _audit_development_labels(
    labels: pd.Series,
    policy: DevelopmentHandoffPolicy,
    *,
    train_dates: Sequence[pd.Timestamp],
    validation_dates: Sequence[pd.Timestamp],
) -> dict[str, object]:
    """Require complete adjusted-VWAP labels before any model-fit row filtering."""

    train_set = {pd.Timestamp(value) for value in train_dates}
    validation_set = {pd.Timestamp(value) for value in validation_dates}
    selected_dates = train_set | validation_set
    selected = labels.loc[
        labels.index.get_level_values("date").isin(selected_dates)
    ].sort_index()
    numeric = pd.to_numeric(selected, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    valid_mask = numeric.notna().rename("label_valid")
    records: list[dict[str, object]] = []
    failures: list[str] = []
    for date, group in valid_mask.groupby(level="date", sort=True):
        eligible = int(len(group))
        valid = int(group.sum())
        share = float(valid / eligible) if eligible else 0.0
        partition = "train" if pd.Timestamp(date) in train_set else "inner_validation"
        passed = bool(
            eligible >= policy.min_names
            and valid >= policy.min_names
            and share >= policy.minimum_daily_label_coverage - 1e-12
            and (policy.allow_label_row_drop or valid == eligible)
        )
        records.append(
            {
                "date": pd.Timestamp(date),
                "partition": partition,
                "eligible_labels": eligible,
                "valid_labels": valid,
                "valid_share": share,
                "passes_gate": passed,
            }
        )
        if not passed:
            failures.append(f"{pd.Timestamp(date).date()}:{partition}:{valid}/{eligible}")
    expected_dates = tuple(pd.Timestamp(value) for value in (*train_dates, *validation_dates))
    actual_dates = tuple(record["date"] for record in records)
    if actual_dates != expected_dates:
        raise ResearchProxyBoundaryError(
            "development label audit does not cover the exact train/validation dates"
        )
    if failures:
        raise ResearchProxyBoundaryError(
            "development adjusted-VWAP label coverage failed; "
            + ", ".join(failures[:5])
        )
    audit = pd.DataFrame.from_records(records).set_index("date")
    retained = valid_mask.loc[valid_mask].astype(int).rename("retained_label_row")
    partition_summary = {
        name: {
            "dates": int(len(group)),
            "eligible_rows": int(group["eligible_labels"].sum()),
            "valid_rows": int(group["valid_labels"].sum()),
            "minimum_daily_valid_share": float(group["valid_share"].min()),
        }
        for name, group in audit.groupby("partition", sort=True)
    }
    return {
        "schema_version": 1,
        "policy_hash": policy.policy_hash,
        "minimum_daily_label_coverage": policy.minimum_daily_label_coverage,
        "allow_label_row_drop": policy.allow_label_row_drop,
        "daily_audit_hash": panel_snapshot_hash(audit),
        "valid_mask_hash": _series_snapshot_hash(valid_mask.astype(int)),
        "retained_row_hash": _series_snapshot_hash(retained),
        "train": partition_summary["train"],
        "inner_validation": partition_summary["inner_validation"],
        "all_development": {
            "dates": int(len(audit)),
            "eligible_rows": int(audit["eligible_labels"].sum()),
            "valid_rows": int(audit["valid_labels"].sum()),
            "minimum_daily_valid_share": float(audit["valid_share"].min()),
        },
        "all_passed": True,
    }


def _recompute_development_handoff(
    commitment: FrozenDevelopmentCommitment,
    development_member_panel: pd.DataFrame,
    development_feature_tape: pd.DataFrame,
    *,
    trading_calendar: Sequence[pd.Timestamp],
    research_universe_identity: Mapping[str, object],
    trading_calendar_identity: Mapping[str, object],
    source_lineage_identity: Mapping[str, object],
    source_snapshot_identity: Mapping[str, object],
    policy: DevelopmentHandoffPolicy,
    expected_policy_hash: str,
    terminal_label_bridges: Sequence[TerminalLabelBridge] = (),
    terminal_label_bridge_identity: Mapping[str, object] | None = None,
    scalar_tolerance: float = 1e-10,
) -> DevelopmentFeatureHandoff:
    """Pure recomputation core; callers must obtain ``commitment`` from manifest."""

    member_panel = development_member_panel.sort_index()
    feature_tape = development_feature_tape.sort_index()
    _require_panel_index(member_panel, label="development member panel")
    _require_panel_index(feature_tape, label="development feature tape")
    split = commitment.split
    if not isinstance(policy, DevelopmentHandoffPolicy):
        raise ResearchProxyBoundaryError("an explicit DevelopmentHandoffPolicy is required")
    if expected_policy_hash != policy.policy_hash:
        raise ResearchProxyBoundaryError(
            "development handoff policy differs from external pre-registration hash"
        )
    if (
        research_universe_identity.get("identity_hash")
        != commitment.research_universe_identity_hash
        or trading_calendar_identity.get("identity_hash")
        != commitment.trading_calendar_identity_hash
        or source_lineage_identity.get("identity_hash")
        != commitment.source_lineage_identity_hash
        or source_snapshot_identity.get("identity_hash")
        != commitment.source_snapshot_identity_hash
    ):
        raise ResearchProxyBoundaryError(
            "development universe/calendar/source-lineage identity differs from commitment"
        )
    if hash_json(
        {
            key: value
            for key, value in source_lineage_identity.items()
            if key != "identity_hash"
        }
    ) != commitment.source_lineage_identity_hash:
        raise ResearchProxyBoundaryError(
            "development source-lineage identity is not self-authenticating"
        )
    if hash_json(
        {
            key: value
            for key, value in source_snapshot_identity.items()
            if key != "identity_hash"
        }
    ) != commitment.source_snapshot_identity_hash:
        raise ResearchProxyBoundaryError(
            "development full source-snapshot identity is not self-authenticating"
        )
    if source_snapshot_identity.get("source_lineage_identity") != dict(
        source_lineage_identity
    ):
        raise ResearchProxyBoundaryError(
            "development full source snapshot differs from source lineage"
        )
    for label, frame in (
        ("development member panel", member_panel),
        ("development feature tape", feature_tape),
    ):
        dates = frame.index.get_level_values("date")
        if dates.max() != split.development_end:
            raise ResearchProxyBoundaryError(
                f"{label} must end exactly at the committed development boundary"
            )
        if dates.isin(split.sealed_dates).any() or dates.max() > split.development_end:
            raise ResearchProxyBoundaryError(f"{label} contains sealed rows")
    if panel_snapshot_hash(member_panel) != commitment.development_member_panel_hash:
        raise ResearchProxyBoundaryError("development member-panel snapshot hash mismatch")
    if panel_snapshot_hash(feature_tape) != commitment.development_feature_tape_hash:
        raise ResearchProxyBoundaryError("development feature-tape snapshot hash mismatch")
    if not member_panel.index.isin(feature_tape.index).all():
        raise ResearchProxyBoundaryError(
            "development feature tape does not cover every eligible member row"
        )

    member_dates = pd.DatetimeIndex(
        member_panel.index.get_level_values("date").unique()
    )
    required_dates = pd.DatetimeIndex(_exact_development_member_dates(split))
    if not member_dates.equals(required_dates):
        raise ResearchProxyBoundaryError(
            "development member panel does not match the exact committed dates"
        )

    feature_frame = build_research_feature_frame(feature_tape)
    cross_sectional_mask = build_cross_sectional_mask(feature_tape, member_panel)
    if (
        research_universe_identity.get("eligible_panel_hash")
        != panel_snapshot_hash(member_panel)
        or research_universe_identity.get("feature_tape_eligibility_mask_hash")
        != hashlib.sha256(
            pd.util.hash_pandas_object(
                cross_sectional_mask.astype(np.int8).rename(
                    "research_eligible"
                ),
                index=True,
                categorize=True,
            ).to_numpy(dtype=np.uint64).tobytes()
        ).hexdigest()
    ):
        raise ResearchProxyBoundaryError(
            "development eligibility mask differs from frozen universe identity"
        )
    calendar_records = [
        str(pd.Timestamp(value).date()) for value in trading_calendar
    ]
    if trading_calendar_identity.get("trading_dates_hash") != hash_json(
        calendar_records
    ):
        raise ResearchProxyBoundaryError(
            "development calendar dates differ from frozen calendar identity"
        )
    exposures = build_style_exposures(member_panel, price_tape=feature_tape)
    neutralizer = PreparedCrossSectionalNeutralizer(
        exposures,
        categorical=("industry",),
        min_names=policy.min_names,
    )
    labels = forward_return(
        feature_frame["vwap"],
        split.label_horizon,
        entry_delay=split.entry_delay,
        trading_calendar=trading_calendar,
    )
    labels, terminal_label_audit = apply_terminal_label_bridges(
        labels,
        feature_tape["vwap"],
        member_panel.index,
        terminal_label_bridges,
        horizon=split.label_horizon,
        entry_delay=split.entry_delay,
        trading_calendar=trading_calendar,
    )
    if terminal_label_bridges and terminal_label_bridge_identity is None:
        raise ResearchProxyBoundaryError(
            "terminal label bridges require a frozen catalog identity"
        )
    labels = labels.reindex(member_panel.index)
    development_label_audit = _audit_development_labels(
        labels,
        policy,
        train_dates=split.train_dates,
        validation_dates=split.validation_dates,
    )

    neutral_features = pd.DataFrame(index=member_panel.index)
    validation_ic_recomputed: dict[str, float] = {}
    validation_labels = _date_slice(labels, split.validation_dates)
    assert isinstance(validation_labels, pd.Series)
    for champion in commitment.champions:
        if (
            canonical_expression_json(champion.expression, PRODUCTION_DSL_SPEC)
            != champion.canonical_expression_json
            or expression_hash(champion.expression, PRODUCTION_DSL_SPEC)
            != champion.expression_hash
        ):
            raise ResearchProxyBoundaryError(
                f"champion {champion.candidate_id!r} mutated after freeze"
            )
        values = evaluate_expression(
            champion.expression,
            feature_frame,
            PRODUCTION_DSL_SPEC,
            cross_sectional_mask=cross_sectional_mask,
        )
        signed = (values * champion.direction).reindex(member_panel.index)
        neutral = neutralizer.transform(signed)
        neutral.name = champion.candidate_id
        neutral_features[champion.candidate_id] = neutral
        validation_values = _date_slice(neutral, split.validation_dates)
        assert isinstance(validation_values, pd.Series)
        daily, _ = daily_ic(
            validation_values,
            validation_labels,
            method="spearman",
            min_names=policy.min_names,
        )
        recomputed = float(daily.mean())
        validation_ic_recomputed[champion.candidate_id] = recomputed
        if not np.isfinite(recomputed) or not np.isclose(
            recomputed,
            champion.validation_neutral_rank_ic,
            rtol=0.0,
            atol=scalar_tolerance,
        ):
            raise ResearchProxyBoundaryError(
                f"champion {champion.candidate_id!r} neutral validation IC "
                "does not reproduce its frozen scalar commitment"
            )

    train_features = _date_slice(neutral_features, split.train_dates)
    valid_features = _date_slice(neutral_features, split.validation_dates)
    train_labels = _date_slice(labels, split.train_dates)
    valid_labels = _date_slice(labels, split.validation_dates)
    assert isinstance(train_features, pd.DataFrame)
    assert isinstance(valid_features, pd.DataFrame)
    assert isinstance(train_labels, pd.Series)
    assert isinstance(valid_labels, pd.Series)
    model_features = pd.concat([train_features, valid_features]).sort_index()
    coverage_audit, feature_audit, retained_train_dates = _audit_neutral_feature_matrix(
        model_features,
        policy,
        train_dates=split.train_dates,
        validation_dates=split.validation_dates,
    )
    train_features = _date_slice(train_features, retained_train_dates)
    train_labels = _date_slice(train_labels, retained_train_dates)
    assert isinstance(train_features, pd.DataFrame)
    assert isinstance(train_labels, pd.Series)
    champion_ids = tuple(item.candidate_id for item in commitment.champions)
    provenance: dict[str, object] = {
        "tier": RESEARCH_PROXY_TIER,
        "development_root_hash": commitment.development_root_hash,
        "champion_family_hash": commitment.champion_family_hash,
        "split_identity_hash": commitment.split_identity_hash,
        "development_member_panel_hash": commitment.development_member_panel_hash,
        "development_feature_tape_hash": commitment.development_feature_tape_hash,
        "research_universe_identity": dict(research_universe_identity),
        "research_universe_identity_hash": commitment.research_universe_identity_hash,
        "trading_calendar_identity": dict(trading_calendar_identity),
        "trading_calendar_identity_hash": commitment.trading_calendar_identity_hash,
        "source_lineage_identity": dict(source_lineage_identity),
        "source_lineage_identity_hash": commitment.source_lineage_identity_hash,
        "source_snapshot_identity": dict(source_snapshot_identity),
        "source_snapshot_identity_hash": commitment.source_snapshot_identity_hash,
        "train_features_hash": panel_snapshot_hash(train_features),
        "validation_features_hash": panel_snapshot_hash(valid_features),
        "train_labels_hash": _series_snapshot_hash(train_labels),
        "validation_labels_hash": _series_snapshot_hash(valid_labels),
        "champion_ids": list(champion_ids),
        "expression_hashes": [item.expression_hash for item in commitment.champions],
        "directions": {
            item.candidate_id: item.direction for item in commitment.champions
        },
        "validation_neutral_rank_ic_recomputed": validation_ic_recomputed,
        "development_handoff_policy": policy.to_dict(),
        "development_handoff_policy_hash": policy.policy_hash,
        "development_label_audit": development_label_audit,
        "terminal_label_bridge_identity": dict(
            terminal_label_bridge_identity or {}
        ),
        "terminal_label_bridge_audit": terminal_label_audit,
        "neutral_feature_audit": feature_audit,
        "neutralization": {
            "method": "same_date_cross_sectional_ridge_residual_v1",
            "categorical": ["industry"],
            "continuous": [
                "log_mcap",
                "beta",
                "momentum_20",
                "volatility_20",
                "liquidity",
            ],
        },
        "committed_train_start": str(split.train_dates[0].date()),
        "model_fit_train_start_after_warmup": str(retained_train_dates[0].date()),
        "train_end": str(retained_train_dates[-1].date()),
        "validation_start": str(split.validation_dates[0].date()),
        "validation_end": str(split.validation_dates[-1].date()),
        "development_hard_end": str(split.development_end.date()),
        "sealed_rows_loaded": 0,
        "sealed_results_consumed": False,
    }
    return DevelopmentFeatureHandoff(
        train_features=train_features,
        validation_features=valid_features,
        train_labels=train_labels,
        validation_labels=valid_labels,
        coverage_audit=coverage_audit,
        champion_ids=champion_ids,
        provenance=provenance,
    )


def build_development_handoff(
    development_manifest: Path | str,
    development_member_panel: pd.DataFrame,
    development_feature_tape: pd.DataFrame,
    *,
    trading_calendar: Sequence[pd.Timestamp],
    research_universe_identity: Mapping[str, object],
    trading_calendar_identity: Mapping[str, object],
    source_lineage_identity: Mapping[str, object],
    source_snapshot_identity: Mapping[str, object],
    policy: DevelopmentHandoffPolicy,
    expected_policy_hash: str,
    terminal_label_bridges: Sequence[TerminalLabelBridge] = (),
    terminal_label_bridge_identity: Mapping[str, object] | None = None,
) -> DevelopmentFeatureHandoff:
    """Recompute the exact 20-factor development matrix without sealed access."""

    commitment = FrozenDevelopmentCommitment.from_manifest(development_manifest)
    return _recompute_development_handoff(
        commitment,
        development_member_panel,
        development_feature_tape,
        trading_calendar=trading_calendar,
        research_universe_identity=research_universe_identity,
        trading_calendar_identity=trading_calendar_identity,
        source_lineage_identity=source_lineage_identity,
        source_snapshot_identity=source_snapshot_identity,
        policy=policy,
        expected_policy_hash=expected_policy_hash,
        terminal_label_bridges=terminal_label_bridges,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
    )


def build_development_handoff_from_commitment(
    commitment: FrozenDevelopmentCommitment,
    development_member_panel: pd.DataFrame,
    development_feature_tape: pd.DataFrame,
    *,
    trading_calendar: Sequence[pd.Timestamp],
    research_universe_identity: Mapping[str, object],
    trading_calendar_identity: Mapping[str, object],
    source_lineage_identity: Mapping[str, object],
    source_snapshot_identity: Mapping[str, object],
    policy: DevelopmentHandoffPolicy,
    expected_policy_hash: str,
    terminal_label_bridges: Sequence[TerminalLabelBridge] = (),
    terminal_label_bridge_identity: Mapping[str, object] | None = None,
) -> DevelopmentFeatureHandoff:
    """Build the pre-manifest handoff from an authenticated in-memory root."""

    if not isinstance(commitment, FrozenDevelopmentCommitment):
        raise ResearchProxyBoundaryError(
            "in-memory handoff requires FrozenDevelopmentCommitment"
        )
    return _recompute_development_handoff(
        commitment,
        development_member_panel,
        development_feature_tape,
        trading_calendar=trading_calendar,
        research_universe_identity=research_universe_identity,
        trading_calendar_identity=trading_calendar_identity,
        source_lineage_identity=source_lineage_identity,
        source_snapshot_identity=source_snapshot_identity,
        policy=policy,
        expected_policy_hash=expected_policy_hash,
        terminal_label_bridges=terminal_label_bridges,
        terminal_label_bridge_identity=terminal_label_bridge_identity,
    )


class _EqualWeightComposite:
    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if tuple(features.columns) != self.columns:
            raise ResearchProxyBoundaryError(
                "prediction features differ from the frozen champion family"
            )
        prediction = normalize_factor_matrix(features).fillna(0.0).mean(axis=1)
        prediction.name = "composite_equal_weight"
        return prediction


def _aligned_model_frame(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    # Rank and impute on the complete eligible same-date feature cross-section
    # first. A future endpoint's availability must never change another
    # security's feature rank. The label mask is applied only afterwards.
    normalized = normalize_factor_matrix(features).fillna(0.0)
    aligned = normalized.join(labels.rename("__label__"), how="inner")
    label = pd.to_numeric(aligned.pop("__label__"), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    keep = label.notna()
    return aligned.loc[keep], label.loc[keep].astype(float)


class _FixedRidgeComposite:
    def __init__(self, columns: Sequence[str], *, alpha: float) -> None:
        self.columns = tuple(columns)
        self.alpha = float(alpha)
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        if tuple(features.columns) != self.columns:
            raise ResearchProxyBoundaryError("Ridge features differ from frozen family")
        x, y = _aligned_model_frame(features, labels)
        if len(x) < 100:
            raise ResearchProxyBoundaryError("insufficient rows for fixed Ridge")
        self.model.fit(
            x.to_numpy(),
            y.to_numpy(),
            sample_weight=equal_date_sample_weights(x.index),
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if tuple(features.columns) != self.columns:
            raise ResearchProxyBoundaryError("Ridge features differ from frozen family")
        x = normalize_factor_matrix(features).fillna(0.0)
        output = pd.Series(
            self.model.predict(x.to_numpy()),
            index=x.index,
            name="composite_ridge",
        )
        return output


class _FixedLightGBMComposite:
    """Single-thread deterministic LightGBM with every sampling knob effective."""

    CONFIGURATION = {
        "objective": "regression",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 1,
        "verbosity": -1,
    }

    def __init__(self, columns: Sequence[str], *, random_state: int) -> None:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is a mandatory frozen-family member and cannot be replaced"
            ) from exc
        self.columns = tuple(columns)
        self.random_state = int(random_state)
        self.model = LGBMRegressor(
            **self.CONFIGURATION,
            random_state=self.random_state,
        )

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        if tuple(features.columns) != self.columns:
            raise ResearchProxyBoundaryError(
                "LightGBM features differ from frozen family"
            )
        x, y = _aligned_model_frame(features, labels)
        if len(x) < 100:
            raise ResearchProxyBoundaryError("insufficient rows for fixed LightGBM")
        self.model.fit(
            x.to_numpy(),
            y.to_numpy(),
            sample_weight=equal_date_sample_weights(x.index),
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if tuple(features.columns) != self.columns:
            raise ResearchProxyBoundaryError(
                "LightGBM features differ from frozen family"
            )
        x = normalize_factor_matrix(features).fillna(0.0)
        output = pd.Series(
            self.model.predict(x.to_numpy()),
            index=x.index,
            name="composite_lightgbm",
        )
        return output


class _LoadedLightGBMComposite:
    def __init__(self, columns: Sequence[str], booster: object) -> None:
        self.columns = tuple(columns)
        self.booster = booster

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if tuple(features.columns) != self.columns:
            raise ResearchProxyBoundaryError(
                "LightGBM features differ from frozen family"
            )
        x = normalize_factor_matrix(features).fillna(0.0)
        return pd.Series(
            self.booster.predict(x.to_numpy()),
            index=x.index,
            name="composite_lightgbm",
        )


def _fitted_model_state_hash(backend: str, model: object) -> str:
    """Commit the fitted state, not only its declared hyperparameters."""

    digest = hashlib.sha256()
    digest.update(backend.encode("utf-8"))
    if backend == "equal_weight":
        columns = getattr(model, "columns", ())
        digest.update(hash_json(list(columns)).encode("utf-8"))
        return digest.hexdigest()
    wrapped = getattr(model, "model", None)
    if backend == "ridge":
        coefficient = np.asarray(getattr(wrapped, "coef_", ()), dtype=np.float64)
        intercept = np.asarray([getattr(wrapped, "intercept_", np.nan)], dtype=np.float64)
        if coefficient.size == 0 or not np.isfinite(coefficient).all() or not np.isfinite(
            intercept
        ).all():
            raise ResearchProxyBoundaryError("fitted Ridge state is incomplete")
        digest.update(coefficient.tobytes())
        digest.update(intercept.tobytes())
        return digest.hexdigest()
    if backend == "lightgbm":
        booster = getattr(wrapped, "booster_", None)
        if booster is None:
            booster = getattr(model, "booster", None)
        if booster is None:
            raise ResearchProxyBoundaryError("fitted LightGBM state is incomplete")
        digest.update(booster.model_to_string().encode("utf-8"))
        return digest.hexdigest()
    raise ResearchProxyBoundaryError(f"unknown composite backend: {backend}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _composite_implementation_identity() -> dict[str, object]:
    package_dir = Path(__file__).resolve().parent
    files = []
    for name in ("ensemble.py", "research_proxy.py"):
        path = package_dir / name
        files.append(
            {
                "path": f"src/cogalpha_plus/{name}",
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    runtime: dict[str, str] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    for distribution in ("numpy", "pandas", "scikit-learn", "lightgbm"):
        try:
            runtime[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            # LightGBM is a required member of this frozen family.  This also
            # guarantees the implementation manifest cannot omit its runtime.
            raise RuntimeError(
                f"composite-family runtime is missing required distribution {distribution}"
            ) from exc
    identity: dict[str, object] = {"files": files, "runtime": runtime}
    identity["aggregate_sha256"] = hash_json(identity)
    return identity


def _artifact_texts(
    models: Mapping[str, object],
    *,
    columns: Sequence[str],
    policy: CompositeFamilyPolicy,
) -> dict[str, str]:
    ridge_wrapper = models.get("ridge")
    ridge_model = getattr(ridge_wrapper, "model", None)
    coefficient = np.asarray(getattr(ridge_model, "coef_", ()), dtype=np.float64)
    intercept = float(getattr(ridge_model, "intercept_", np.nan))
    if coefficient.shape != (len(columns),) or not np.isfinite(coefficient).all():
        raise ResearchProxyBoundaryError("Ridge artifact has invalid coefficients")
    if not np.isfinite(intercept):
        raise ResearchProxyBoundaryError("Ridge artifact has invalid intercept")
    lightgbm_wrapper = models.get("lightgbm")
    lightgbm_model = getattr(lightgbm_wrapper, "model", None)
    booster = getattr(lightgbm_model, "booster_", None)
    if booster is None:
        booster = getattr(lightgbm_wrapper, "booster", None)
    if booster is None:
        raise ResearchProxyBoundaryError("LightGBM artifact has no fitted booster")
    return {
        "equal_weight.json": canonical_json(
            {
                "schema_version": 1,
                "backend": "equal_weight",
                "columns": list(columns),
            }
        )
        + "\n",
        "ridge.json": canonical_json(
            {
                "schema_version": 1,
                "backend": "ridge",
                "columns": list(columns),
                "alpha": float(policy.ridge_alpha),
                "coefficient": coefficient.tolist(),
                "intercept": intercept,
            }
        )
        + "\n",
        "lightgbm.txt": booster.model_to_string(),
    }


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SealedCompositeClaimCapability:
    """In-process evidence that the persistent outer gate was retired first."""

    composite_family_hash: str
    split_identity_hash: str
    combined_data_snapshot_hash: str
    source_snapshot_identity_hash: str
    persistent_claim_path: str
    persistent_claim_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "composite_family_hash",
            "split_identity_hash",
            "combined_data_snapshot_hash",
            "source_snapshot_identity_hash",
            "persistent_claim_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ResearchProxyBoundaryError(f"sealed claim capability has invalid {name}")
        self.verify_persistent_claim()

    def verify_persistent_claim(self) -> None:
        path = Path(self.persistent_claim_path).resolve()
        if not path.is_file():
            raise ResearchProxyBoundaryError("persistent sealed claim file is missing")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.persistent_claim_sha256:
            raise ResearchProxyBoundaryError("persistent sealed claim hash mismatch")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResearchProxyBoundaryError("persistent sealed claim is invalid") from exc
        expected = {
            "combined_data_snapshot_hash": self.combined_data_snapshot_hash,
            "source_snapshot_identity_hash": self.source_snapshot_identity_hash,
            "split_identity_hash": self.split_identity_hash,
            "composite_family_hash": self.composite_family_hash,
        }
        if not isinstance(payload, Mapping) or any(
            payload.get(name) != value for name, value in expected.items()
        ):
            raise ResearchProxyBoundaryError(
                "persistent sealed claim differs from capability roots"
            )
        gate_identity = {
            "combined_data_snapshot_hash": self.combined_data_snapshot_hash,
            "split_identity_hash": self.split_identity_hash,
        }
        if path.name != f"{hash_json(gate_identity)}.json":
            raise ResearchProxyBoundaryError(
                "persistent sealed claim filename differs from snapshot/split identity"
            )
        if (
            payload.get("schema_version") != 1
            or payload.get("gate_kind") != "known-history-isolated-replay"
            or payload.get("status")
            != "retired_before_any_sealed_row_was_loaded"
        ):
            raise ResearchProxyBoundaryError(
                "persistent sealed claim retirement evidence is invalid"
            )


@dataclass(frozen=True)
class FrozenDevelopmentCompositeFamily:
    """All pre-registered composites frozen together before sealed reveal."""

    primary_backend: str
    descriptive_validation_rank_ic: Mapping[str, float]
    _fitted_models: Mapping[str, object]
    artifact_texts: Mapping[str, str]
    family_commitment: Mapping[str, object]
    composite_family_hash: str
    provenance: Mapping[str, object]
    tier: str = RESEARCH_PROXY_TIER
    disclaimer: str = RESEARCH_PROXY_DISCLAIMER

    def __post_init__(self) -> None:
        ordered_family = ("equal_weight", "ridge", "lightgbm")
        if tuple(self._fitted_models) != ordered_family:
            raise ResearchProxyBoundaryError("frozen composite family is incomplete")
        if self.primary_backend != "lightgbm":
            raise ResearchProxyBoundaryError("frozen primary backend must be LightGBM")
        if self.tier != RESEARCH_PROXY_TIER:
            raise ResearchProxyBoundaryError("composite-family tier cannot be upgraded")
        if hash_json(dict(self.family_commitment)) != self.composite_family_hash:
            raise ResearchProxyBoundaryError("composite-family commitment hash mismatch")
        if self.provenance.get("sealed_rows_loaded") != 0:
            raise ResearchProxyBoundaryError("composite family consumed sealed rows")
        object.__setattr__(
            self, "descriptive_validation_rank_ic",
            MappingProxyType(dict(self.descriptive_validation_rank_ic)),
        )
        object.__setattr__(
            self, "_fitted_models", MappingProxyType(dict(self._fitted_models))
        )
        object.__setattr__(
            self, "artifact_texts", MappingProxyType(dict(self.artifact_texts))
        )
        # JSON round-trips detach caller-owned nested containers.  Re-hashing on
        # every use below detects even a deliberate mutation through a nested
        # object retained from this read-only top-level mapping.
        object.__setattr__(
            self,
            "family_commitment",
            MappingProxyType(json.loads(canonical_json(self.family_commitment))),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(json.loads(canonical_json(self.provenance))),
        )
        object.__setattr__(self, "_sealed_gate_consumed", False)
        self.verify_integrity()

    def verify_integrity(self) -> None:
        if hash_json(dict(self.family_commitment)) != self.composite_family_hash:
            raise ResearchProxyBoundaryError("composite-family commitment mutated")
        if self.family_commitment.get("schema_version") != 1:
            raise ResearchProxyBoundaryError("unsupported composite-family schema")
        if self.family_commitment.get("tier") != RESEARCH_PROXY_TIER:
            raise ResearchProxyBoundaryError("composite-family tier commitment changed")
        if self.family_commitment.get("sealed_rows_loaded") != 0 or self.family_commitment.get(
            "sealed_results_consumed"
        ) is not False:
            raise ResearchProxyBoundaryError("composite family crossed the sealed boundary")
        policy_payload = self.family_commitment.get("composite_family_policy")
        if not isinstance(policy_payload, Mapping):
            raise ResearchProxyBoundaryError("family commitment omits composite policy")
        policy = CompositeFamilyPolicy(
            primary_backend=str(policy_payload.get("primary_backend", "")),
            members=tuple(policy_payload.get("members", ())),
            ridge_alpha=float(policy_payload.get("ridge_alpha", np.nan)),
            random_state=int(policy_payload.get("random_state", -1)),
            lightgbm_configuration=dict(
                policy_payload.get("lightgbm_configuration", {})
            ),
            schema_version=int(policy_payload.get("schema_version", 0)),
        )
        if policy.policy_hash != self.family_commitment.get(
            "composite_family_policy_hash"
        ):
            raise ResearchProxyBoundaryError("composite-family policy hash changed")
        if (
            self.primary_backend != policy.primary_backend
            or self.family_commitment.get("primary_backend_pre_registered")
            != policy.primary_backend
            or tuple(self.family_commitment.get("family_members", ()))
            != policy.members
        ):
            raise ResearchProxyBoundaryError(
                "composite primary or family membership differs from policy"
            )
        feature_columns = tuple(self.family_commitment.get("feature_columns", ()))
        if len(feature_columns) != FROZEN_CHAMPION_COUNT or len(set(feature_columns)) != len(
            feature_columns
        ):
            raise ResearchProxyBoundaryError("composite feature family is invalid")
        expression_hashes = self.family_commitment.get("expression_hashes")
        if (
            not isinstance(expression_hashes, Sequence)
            or isinstance(expression_hashes, (str, bytes))
            or len(expression_hashes) != FROZEN_CHAMPION_COUNT
            or len(set(expression_hashes)) != FROZEN_CHAMPION_COUNT
            or any(
                len(str(value)) != 64
                or any(character not in "0123456789abcdef" for character in str(value))
                for value in expression_hashes
            )
        ):
            raise ResearchProxyBoundaryError(
                "composite expression-hash family is invalid"
            )
        directions = self.family_commitment.get("frozen_train_only_directions")
        if (
            not isinstance(directions, Mapping)
            or set(directions) != set(feature_columns)
            or any(value not in {-1, 1} for value in directions.values())
        ):
            raise ResearchProxyBoundaryError(
                "composite train-only direction family is invalid"
            )
        raw_handoff_policy = self.family_commitment.get(
            "development_handoff_policy"
        )
        if not isinstance(raw_handoff_policy, Mapping):
            raise ResearchProxyBoundaryError("family commitment omits handoff policy")
        handoff_policy = DevelopmentHandoffPolicy(**dict(raw_handoff_policy))
        if handoff_policy.policy_hash != self.family_commitment.get(
            "development_handoff_policy_hash"
        ):
            raise ResearchProxyBoundaryError("family handoff-policy hash changed")
        label_audit = self.family_commitment.get("development_label_audit")
        if (
            not isinstance(label_audit, Mapping)
            or label_audit.get("policy_hash") != handoff_policy.policy_hash
            or label_audit.get("all_passed") is not True
            or label_audit.get("allow_label_row_drop") is not False
            or not isinstance(label_audit.get("all_development"), Mapping)
            or label_audit["all_development"].get("eligible_rows")
            != label_audit["all_development"].get("valid_rows")
        ):
            raise ResearchProxyBoundaryError(
                "family development-label audit is incomplete"
            )
        for name in (
            "research_universe_identity",
            "trading_calendar_identity",
            "source_lineage_identity",
            "source_snapshot_identity",
        ):
            identity = self.family_commitment.get(name)
            expected = self.family_commitment.get(f"{name}_hash")
            if (
                not isinstance(identity, Mapping)
                or identity.get("identity_hash") != expected
                or hash_json(
                    {key: value for key, value in identity.items() if key != "identity_hash"}
                )
                != expected
            ):
                raise ResearchProxyBoundaryError(
                    f"composite family {name} commitment is invalid"
                )
        if tuple(self._fitted_models) != policy.members:
            raise ResearchProxyBoundaryError("in-memory model family/order changed")
        if set(self.artifact_texts) != {
            "equal_weight.json",
            "ridge.json",
            "lightgbm.txt",
        }:
            raise ResearchProxyBoundaryError("in-memory artifact family changed")
        if self.family_commitment.get(
            "composite_implementation_identity"
        ) != _composite_implementation_identity():
            raise ResearchProxyBoundaryError(
                "current composite code/runtime differs from frozen family"
            )
        expected = self.family_commitment.get("artifacts")
        if not isinstance(expected, Mapping):
            raise ResearchProxyBoundaryError("family commitment omits artifact hashes")
        if set(expected) != {
            "equal_weight.json",
            "ridge.json",
            "lightgbm.txt",
        }:
            raise ResearchProxyBoundaryError("family commitment has an invalid artifact set")
        actual_texts = _artifact_texts(
            self._fitted_models,
            columns=feature_columns,
            policy=policy,
        )
        if actual_texts != dict(self.artifact_texts):
            raise ResearchProxyBoundaryError("in-memory fitted model artifacts mutated")
        expected_states = self.family_commitment.get("model_state_hashes")
        actual_states = {
            name: _fitted_model_state_hash(name, model)
            for name, model in self._fitted_models.items()
        }
        if not isinstance(expected_states, Mapping) or dict(expected_states) != actual_states:
            raise ResearchProxyBoundaryError("in-memory fitted model state mutated")
        for filename, text in actual_texts.items():
            record = expected.get(filename)
            if (
                not isinstance(record, Mapping)
                or record.get("sha256") != _text_sha256(text)
                or int(record.get("bytes", -1)) != len(text.encode("utf-8"))
            ):
                raise ResearchProxyBoundaryError(
                    f"fitted model artifact hash mismatch: {filename}"
                )

    def _predict_unchecked(
        self,
        features: pd.DataFrame,
        *,
        backend: str,
    ) -> pd.Series:
        self.verify_integrity()
        selected = backend
        if selected not in self._fitted_models:
            raise ResearchProxyBoundaryError("requested backend is outside frozen family")
        predictor = getattr(self._fitted_models[selected], "predict", None)
        if predictor is None:
            raise RuntimeError("frozen composite has no prediction interface")
        return predictor(features)

    def predict_all_for_sealed_gate(
        self,
        features: pd.DataFrame,
        *,
        expected_composite_family_hash: str,
        claim_capability: SealedCompositeClaimCapability,
    ) -> pd.DataFrame:
        """Score the exact three-member family; no backend choice is accepted."""

        if not isinstance(claim_capability, SealedCompositeClaimCapability):
            raise ResearchProxyBoundaryError(
                "sealed family prediction requires a persistent-claim capability"
            )
        claim_capability.verify_persistent_claim()
        if expected_composite_family_hash != self.composite_family_hash:
            raise ResearchProxyBoundaryError(
                "sealed gate composite-family commitment mismatch"
            )
        if (
            claim_capability.composite_family_hash != self.composite_family_hash
            or claim_capability.split_identity_hash
            != self.family_commitment.get("split_identity_hash")
            or claim_capability.source_snapshot_identity_hash
            != self.family_commitment.get("source_snapshot_identity_hash")
        ):
            raise ResearchProxyBoundaryError(
                "sealed claim capability differs from frozen family roots"
            )
        if bool(getattr(self, "_sealed_gate_consumed", False)):
            raise ResearchProxyBoundaryError(
                "frozen composite family was already consumed by a sealed gate"
            )
        # Consume before any audit/model call so a failure cannot be used as a
        # retryable oracle within this process.
        object.__setattr__(self, "_sealed_gate_consumed", True)
        self.verify_integrity()
        expected_columns = tuple(self.family_commitment["feature_columns"])
        if tuple(features.columns) != expected_columns:
            raise ResearchProxyBoundaryError(
                "sealed factor matrix columns differ from frozen composite family"
            )
        # Fail closed before any model can turn missing values into zeros.  A
        # sealed date cannot be dropped as development warm-up: every date and
        # all 20 columns must satisfy the pre-registered handoff policy.
        self.audit_sealed_feature_matrix(features)
        output = pd.DataFrame(index=features.index)
        for backend in ("equal_weight", "ridge", "lightgbm"):
            output[backend] = self._predict_unchecked(features, backend=backend)
        return output

    def audit_sealed_feature_matrix(
        self,
        features: pd.DataFrame,
    ) -> dict[str, object]:
        """Apply the frozen handoff completeness policy to every sealed date."""

        self.verify_integrity()
        expected_columns = tuple(self.family_commitment["feature_columns"])
        if tuple(features.columns) != expected_columns:
            raise ResearchProxyBoundaryError(
                "sealed factor matrix columns differ from frozen composite family"
            )
        raw_policy = self.family_commitment.get("development_handoff_policy")
        if not isinstance(raw_policy, Mapping):
            raise ResearchProxyBoundaryError(
                "composite family omits its development handoff policy"
            )
        policy = DevelopmentHandoffPolicy(**dict(raw_policy))
        if policy.policy_hash != self.family_commitment.get(
            "development_handoff_policy_hash"
        ):
            raise ResearchProxyBoundaryError(
                "composite family handoff-policy hash mismatch"
            )
        _require_panel_index(features, label="sealed factor matrix")
        records: list[dict[str, object]] = []
        failures: list[str] = []
        for date, group in features.groupby(level="date", sort=True):
            finite = group.replace([np.inf, -np.inf], np.nan)
            total = int(len(group))
            for factor in expected_columns:
                values = finite[factor].dropna()
                valid = int(len(values))
                coverage = float(valid / total) if total else 0.0
                zero_share = float(1.0 - coverage)
                constant = valid < 2 or int(values.nunique(dropna=True)) < 2
                passed = bool(
                    valid >= policy.min_names
                    and coverage >= policy.minimum_daily_factor_coverage - 1e-12
                    and zero_share <= policy.maximum_zero_impute_share + 1e-12
                    and not constant
                )
                records.append(
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "factor": factor,
                        "valid_names": valid,
                        "total_names": total,
                        "coverage": coverage,
                        "zero_impute_share": zero_share,
                        "constant_or_all_missing": bool(constant),
                        "passes_gate": passed,
                    }
                )
                if not passed:
                    failures.append(f"{pd.Timestamp(date).date()}:{factor}")
        if not records:
            raise ResearchProxyBoundaryError("sealed factor matrix is empty")
        audit = pd.DataFrame.from_records(records).set_index(["date", "factor"])
        if failures:
            preview = ", ".join(failures[:5])
            raise ResearchProxyBoundaryError(
                "sealed factor matrix fails the frozen no-drop coverage/constant "
                f"gate ({len(failures)} date-factor failures; {preview})"
            )
        return {
            "schema_version": 1,
            "policy_hash": policy.policy_hash,
            "audit_hash": panel_snapshot_hash(audit),
            "dates": int(audit.index.get_level_values("date").nunique()),
            "date_factor_rows": int(len(audit)),
            "minimum_observed_coverage": float(audit["coverage"].min()),
            "maximum_observed_zero_impute_share": float(
                audit["zero_impute_share"].max()
            ),
            "sealed_dates_dropped": 0,
            "all_passed": True,
        }

    def predict_development(
        self,
        handoff: DevelopmentFeatureHandoff,
        *,
        partition: str,
        backend: str | None = None,
    ) -> pd.Series:
        """Predict only an exact committed development partition.

        There is intentionally no generic or sealed prediction method here.
        A future sealed-family gate must consume the artifact bundle once and
        evaluate all three members together without returning data to research.
        """

        handoff.verify_integrity()
        if partition not in {"train", "inner_validation"}:
            raise ResearchProxyBoundaryError(
                "development prediction partition must be train or inner_validation"
            )
        for key in (
            "development_root_hash",
            "champion_family_hash",
            "split_identity_hash",
            "source_lineage_identity_hash",
            "source_snapshot_identity_hash",
            "train_features_hash",
            "validation_features_hash",
        ):
            if handoff.provenance.get(key) != self.family_commitment.get(key):
                raise ResearchProxyBoundaryError(
                    f"development prediction handoff differs on {key}"
                )
        features = (
            handoff.train_features
            if partition == "train"
            else handoff.validation_features
        )
        selected = self.primary_backend if backend is None else backend
        return self._predict_unchecked(features, backend=selected)

    def predict_all_development(
        self,
        handoff: DevelopmentFeatureHandoff,
        *,
        partition: str,
    ) -> pd.DataFrame:
        output = pd.DataFrame(
            index=(
                handoff.train_features.index
                if partition == "train"
                else handoff.validation_features.index
            )
        )
        for name in ("equal_weight", "ridge", "lightgbm"):
            output[name] = self.predict_development(
                handoff,
                partition=partition,
                backend=name,
            )
        return output

    def development_alpha_snapshot(
        self,
        handoff: DevelopmentFeatureHandoff,
        *,
        as_of: pd.Timestamp | str,
        partition: str,
    ) -> "DevelopmentAlphaSnapshot":
        """Create a typed primary-alpha snapshot on a committed dev date."""

        day = pd.Timestamp(as_of).normalize()
        features = (
            handoff.train_features
            if partition == "train"
            else handoff.validation_features
            if partition == "inner_validation"
            else None
        )
        if features is None:
            raise ResearchProxyBoundaryError("invalid development alpha partition")
        dates = features.index.get_level_values("date")
        if day not in dates:
            raise ResearchProxyBoundaryError(
                "alpha as-of date is outside the committed development partition"
            )
        date_features = features.loc[dates == day]
        prediction = self.predict_development(
            handoff,
            partition=partition,
            backend=self.primary_backend,
        ).loc[dates == day]
        values = prediction.droplevel("date")
        values.index = values.index.astype(str)
        values.index.name = "symbol"
        return DevelopmentAlphaSnapshot(
            values=values,
            as_of=day,
            backend=self.primary_backend,
            composite_family_hash=self.composite_family_hash,
            development_root_hash=str(
                self.family_commitment["development_root_hash"]
            ),
            source_feature_hash=panel_snapshot_hash(date_features),
            values_hash=_series_snapshot_hash(values),
            sealed_rows_loaded=0,
        )


def fit_frozen_development_composite_family(
    handoff: DevelopmentFeatureHandoff,
    *,
    policy: CompositeFamilyPolicy,
    expected_policy_hash: str,
) -> FrozenDevelopmentCompositeFamily:
    """Fit and freeze equal, fixed-Ridge and fixed-LightGBM as one family.

    The 20 champions were already selected adaptively on inner-validation.
    Therefore validation metrics here are descriptive only: they never choose a
    backend, tune Ridge or decide advancement.  ``policy`` must match an
    externally stored pre-registration hash.  All three are refit on the
    complete development sample and committed together for a future one-time
    sealed evaluation.  A missing LightGBM dependency fails loudly.
    """

    handoff.verify_integrity()
    if handoff.provenance.get("sealed_rows_loaded") != 0:
        raise ResearchProxyBoundaryError("composite fit received sealed provenance")
    if not isinstance(policy, CompositeFamilyPolicy):
        raise ResearchProxyBoundaryError("an explicit CompositeFamilyPolicy is required")
    if expected_policy_hash != policy.policy_hash:
        raise ResearchProxyBoundaryError(
            "composite policy differs from its external pre-registration hash"
        )
    primary_backend = policy.primary_backend
    ridge_alpha = float(policy.ridge_alpha)
    random_state = int(policy.random_state)
    train_x = handoff.train_features
    valid_x = handoff.validation_features
    train_y = handoff.train_labels
    valid_y = handoff.validation_labels
    equal_model = _EqualWeightComposite(handoff.champion_ids)
    equal_prediction = equal_model.predict(valid_x)
    equal_ic = rank_ic_mean(equal_prediction, valid_y)

    ridge_probe = _FixedRidgeComposite(
        handoff.champion_ids,
        alpha=ridge_alpha,
    )
    ridge_probe.fit(train_x, train_y)
    ridge_validation_prediction = ridge_probe.predict(valid_x)

    lightgbm_probe = _FixedLightGBMComposite(
        handoff.champion_ids,
        random_state=random_state,
    )
    lightgbm_probe.fit(train_x, train_y)
    lightgbm_validation_prediction = lightgbm_probe.predict(valid_x)

    descriptive_scores = {
        "equal_weight": float(equal_ic),
        "ridge": float(rank_ic_mean(ridge_validation_prediction, valid_y)),
        "lightgbm": float(rank_ic_mean(lightgbm_validation_prediction, valid_y)),
    }
    if not all(np.isfinite(value) for value in descriptive_scores.values()):
        raise ResearchProxyBoundaryError(
            "descriptive validation scores are incomplete; family freeze aborted"
        )

    development_x = pd.concat([train_x, valid_x]).sort_index()
    development_y = pd.concat([train_y, valid_y]).sort_index()
    ridge_refit = _FixedRidgeComposite(
        handoff.champion_ids,
        alpha=ridge_alpha,
    )
    ridge_refit.fit(development_x, development_y)
    lightgbm_refit = _FixedLightGBMComposite(
        handoff.champion_ids,
        random_state=random_state,
    )
    lightgbm_refit.fit(development_x, development_y)
    fitted_models: dict[str, object] = {
        "equal_weight": equal_model,
        "ridge": ridge_refit,
        "lightgbm": lightgbm_refit,
    }
    model_state_hashes = {
        name: _fitted_model_state_hash(name, model)
        for name, model in fitted_models.items()
    }
    artifacts = _artifact_texts(
        fitted_models,
        columns=handoff.champion_ids,
        policy=policy,
    )
    artifact_records = {
        filename: {
            "format": (
                "canonical_json_v1" if filename.endswith(".json") else "lightgbm_model_text"
            ),
            "bytes": len(text.encode("utf-8")),
            "sha256": _text_sha256(text),
        }
        for filename, text in artifacts.items()
    }
    configurations = {
        "equal_weight": {"method": "mean_of_20_same_date_rank_normalized_factors"},
        "ridge": {"alpha": float(ridge_alpha), "fit_intercept": True},
        "lightgbm": dict(policy.lightgbm_configuration),
    }
    implementation_identity = _composite_implementation_identity()
    family_commitment: dict[str, object] = {
        "schema_version": 1,
        "kind": "development-frozen-composite-family-before-sealed-reveal",
        "tier": RESEARCH_PROXY_TIER,
        "primary_backend_pre_registered": primary_backend,
        "composite_family_policy": policy.to_dict(),
        "composite_family_policy_hash": policy.policy_hash,
        "family_members": ["equal_weight", "ridge", "lightgbm"],
        "configurations": configurations,
        "model_state_hashes": model_state_hashes,
        "artifacts": artifact_records,
        "composite_implementation_identity": implementation_identity,
        "development_root_hash": handoff.provenance[
            "development_root_hash"
        ],
        "champion_family_hash": handoff.provenance["champion_family_hash"],
        "split_identity_hash": handoff.provenance["split_identity_hash"],
        "development_member_panel_hash": handoff.provenance[
            "development_member_panel_hash"
        ],
        "development_feature_tape_hash": handoff.provenance[
            "development_feature_tape_hash"
        ],
        "research_universe_identity": dict(
            handoff.provenance["research_universe_identity"]
        ),
        "research_universe_identity_hash": handoff.provenance[
            "research_universe_identity_hash"
        ],
        "trading_calendar_identity": dict(
            handoff.provenance["trading_calendar_identity"]
        ),
        "trading_calendar_identity_hash": handoff.provenance[
            "trading_calendar_identity_hash"
        ],
        "source_lineage_identity": dict(
            handoff.provenance["source_lineage_identity"]
        ),
        "source_lineage_identity_hash": handoff.provenance[
            "source_lineage_identity_hash"
        ],
        "source_snapshot_identity": dict(
            handoff.provenance["source_snapshot_identity"]
        ),
        "source_snapshot_identity_hash": handoff.provenance[
            "source_snapshot_identity_hash"
        ],
        "train_features_hash": handoff.provenance["train_features_hash"],
        "validation_features_hash": handoff.provenance[
            "validation_features_hash"
        ],
        "train_labels_hash": handoff.provenance["train_labels_hash"],
        "validation_labels_hash": handoff.provenance["validation_labels_hash"],
        "feature_columns": list(handoff.champion_ids),
        "expression_hashes": list(handoff.provenance["expression_hashes"]),
        "frozen_train_only_directions": dict(handoff.provenance["directions"]),
        "development_handoff_policy": dict(
            handoff.provenance["development_handoff_policy"]
        ),
        "development_handoff_policy_hash": handoff.provenance[
            "development_handoff_policy_hash"
        ],
        "development_label_audit": dict(
            handoff.provenance["development_label_audit"]
        ),
        "neutralization": dict(handoff.provenance["neutralization"]),
        "neutral_feature_audit": dict(
            handoff.provenance["neutral_feature_audit"]
        ),
        "missing_feature_policy": (
            "same_date_rank_on_full_eligible_feature_cross_section_then_zero_"
            "impute_before_label_join_and_mask"
        ),
        "fit_scope": (
            "refit_on_all_development_train_plus_inner_validation_before_"
            "family_commitment_and_sealed_reveal"
        ),
        "descriptive_train_only_probe_validation_rank_ic": descriptive_scores,
        "validation_metrics_used_for_selection_or_tuning": False,
        "sealed_rows_loaded": 0,
        "sealed_results_consumed": False,
    }
    family_hash = hash_json(family_commitment)

    provenance = {
        "tier": RESEARCH_PROXY_TIER,
        "development_root_hash": handoff.provenance[
            "development_root_hash"
        ],
        "champion_family_hash": handoff.provenance["champion_family_hash"],
        "split_identity_hash": handoff.provenance["split_identity_hash"],
        "development_member_panel_hash": handoff.provenance[
            "development_member_panel_hash"
        ],
        "development_feature_tape_hash": handoff.provenance[
            "development_feature_tape_hash"
        ],
        "research_universe_identity_hash": handoff.provenance[
            "research_universe_identity_hash"
        ],
        "trading_calendar_identity_hash": handoff.provenance[
            "trading_calendar_identity_hash"
        ],
        "source_lineage_identity_hash": handoff.provenance[
            "source_lineage_identity_hash"
        ],
        "source_snapshot_identity_hash": handoff.provenance[
            "source_snapshot_identity_hash"
        ],
        "train_features_hash": handoff.provenance["train_features_hash"],
        "validation_features_hash": handoff.provenance[
            "validation_features_hash"
        ],
        "feature_columns": list(handoff.champion_ids),
        "descriptive_train_only_probe_validation_rank_ic": descriptive_scores,
        "validation_metrics_used_for_selection_or_tuning": False,
        "primary_backend_pre_registered": primary_backend,
        "composite_family_policy_hash": policy.policy_hash,
        "ridge_alpha_pre_registered": float(ridge_alpha),
        "random_state": int(random_state),
        "refit_scope": (
            "all_development_train_plus_inner_validation_before_family_"
            "commitment_and_sealed_reveal"
        ),
        "model_state_hashes": model_state_hashes,
        "composite_implementation_identity": implementation_identity,
        "composite_family_hash": family_hash,
        "sealed_rows_loaded": 0,
        "sealed_results_consumed": False,
    }
    return FrozenDevelopmentCompositeFamily(
        primary_backend=primary_backend,
        descriptive_validation_rank_ic=descriptive_scores,
        _fitted_models=fitted_models,
        artifact_texts=artifacts,
        family_commitment=family_commitment,
        composite_family_hash=family_hash,
        provenance=provenance,
    )


def _exclusive_text(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def dump_frozen_composite_family(
    family: FrozenDevelopmentCompositeFamily,
    directory: Path | str,
) -> Path:
    """Persist a non-pickle, content-addressed cross-process handoff bundle."""

    family.verify_integrity()
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for filename, text in family.artifact_texts.items():
        _exclusive_text(target / filename, text)
    envelope = {
        "schema_version": 1,
        "kind": "frozen-development-composite-family-artifact-bundle",
        "family_commitment": dict(family.family_commitment),
        "composite_family_hash": family.composite_family_hash,
    }
    _exclusive_text(target / "family_manifest.json", canonical_json(envelope) + "\n")
    return target / "family_manifest.json"


def load_frozen_composite_family(
    directory: Path | str,
    *,
    expected_composite_family_hash: str,
) -> FrozenDevelopmentCompositeFamily:
    """Verify and reconstruct a bundle against an external commitment hash."""

    source = Path(directory).resolve()
    expected_files = {
        "equal_weight.json",
        "ridge.json",
        "lightgbm.txt",
        "family_manifest.json",
    }
    actual_files = {path.name for path in source.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ResearchProxyBoundaryError(
            "composite artifact bundle has missing or unexpected files"
        )
    envelope = json.loads((source / "family_manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(envelope, Mapping)
        or envelope.get("schema_version") != 1
        or envelope.get("kind")
        != "frozen-development-composite-family-artifact-bundle"
    ):
        raise ResearchProxyBoundaryError("composite artifact envelope is invalid")
    commitment = envelope.get("family_commitment")
    if not isinstance(commitment, Mapping):
        raise ResearchProxyBoundaryError("artifact envelope omits family commitment")
    family_hash = str(envelope.get("composite_family_hash", ""))
    if (
        len(expected_composite_family_hash) != 64
        or family_hash != expected_composite_family_hash
    ):
        raise ResearchProxyBoundaryError(
            "artifact bundle differs from the externally committed family hash"
        )
    if hash_json(commitment) != family_hash:
        raise ResearchProxyBoundaryError("artifact envelope family hash mismatch")
    implementation = commitment.get("composite_implementation_identity")
    if implementation != _composite_implementation_identity():
        raise ResearchProxyBoundaryError(
            "current composite code/runtime differs from frozen family"
        )
    artifacts = commitment.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ResearchProxyBoundaryError("family commitment omits artifacts")
    texts: dict[str, str] = {}
    for filename in ("equal_weight.json", "ridge.json", "lightgbm.txt"):
        text = (source / filename).read_text(encoding="utf-8")
        record = artifacts.get(filename)
        if (
            not isinstance(record, Mapping)
            or int(record.get("bytes", -1)) != len(text.encode("utf-8"))
            or str(record.get("sha256", "")) != _text_sha256(text)
        ):
            raise ResearchProxyBoundaryError(
                f"composite artifact failed byte/hash verification: {filename}"
            )
        texts[filename] = text

    policy_payload = commitment.get("composite_family_policy")
    if not isinstance(policy_payload, Mapping):
        raise ResearchProxyBoundaryError("family commitment omits policy")
    policy = CompositeFamilyPolicy(
        primary_backend=str(policy_payload.get("primary_backend", "")),
        members=tuple(policy_payload.get("members", ())),
        ridge_alpha=float(policy_payload.get("ridge_alpha", np.nan)),
        random_state=int(policy_payload.get("random_state", -1)),
        lightgbm_configuration=dict(policy_payload.get("lightgbm_configuration", {})),
        schema_version=int(policy_payload.get("schema_version", 0)),
    )
    if policy.policy_hash != str(commitment.get("composite_family_policy_hash", "")):
        raise ResearchProxyBoundaryError("loaded composite policy hash mismatch")

    equal_payload = json.loads(texts["equal_weight.json"])
    ridge_payload = json.loads(texts["ridge.json"])
    columns = tuple(equal_payload.get("columns", ()))
    if (
        equal_payload.get("backend") != "equal_weight"
        or tuple(ridge_payload.get("columns", ())) != columns
        or tuple(commitment.get("feature_columns", ())) != columns
        or ridge_payload.get("backend") != "ridge"
        or float(ridge_payload.get("alpha", np.nan)) != policy.ridge_alpha
    ):
        raise ResearchProxyBoundaryError("JSON model artifacts disagree with family")
    equal = _EqualWeightComposite(columns)
    ridge = _FixedRidgeComposite(columns, alpha=policy.ridge_alpha)
    coefficient = np.asarray(ridge_payload.get("coefficient", ()), dtype=np.float64)
    intercept = float(ridge_payload.get("intercept", np.nan))
    if coefficient.shape != (len(columns),) or not np.isfinite(coefficient).all():
        raise ResearchProxyBoundaryError("loaded Ridge coefficients are invalid")
    if not np.isfinite(intercept):
        raise ResearchProxyBoundaryError("loaded Ridge intercept is invalid")
    ridge.model.coef_ = coefficient
    ridge.model.intercept_ = intercept
    ridge.model.n_features_in_ = len(columns)

    try:
        from lightgbm import Booster
    except ImportError as exc:
        raise RuntimeError("LightGBM is required to load the frozen family") from exc
    lightgbm = _LoadedLightGBMComposite(
        columns,
        Booster(model_str=texts["lightgbm.txt"]),
    )
    models: dict[str, object] = {
        "equal_weight": equal,
        "ridge": ridge,
        "lightgbm": lightgbm,
    }
    descriptive = commitment.get(
        "descriptive_train_only_probe_validation_rank_ic"
    )
    if not isinstance(descriptive, Mapping):
        raise ResearchProxyBoundaryError("family commitment omits descriptive metrics")
    provenance = {
        "tier": RESEARCH_PROXY_TIER,
        "development_root_hash": commitment["development_root_hash"],
        "champion_family_hash": commitment["champion_family_hash"],
        "split_identity_hash": commitment["split_identity_hash"],
        "source_lineage_identity_hash": commitment[
            "source_lineage_identity_hash"
        ],
        "source_snapshot_identity_hash": commitment[
            "source_snapshot_identity_hash"
        ],
        "train_features_hash": commitment["train_features_hash"],
        "validation_features_hash": commitment["validation_features_hash"],
        "composite_family_hash": family_hash,
        "model_state_hashes": dict(commitment["model_state_hashes"]),
        "sealed_rows_loaded": 0,
        "sealed_results_consumed": False,
    }
    return FrozenDevelopmentCompositeFamily(
        primary_backend=policy.primary_backend,
        descriptive_validation_rank_ic={
            str(name): float(value) for name, value in descriptive.items()
        },
        _fitted_models=models,
        artifact_texts=texts,
        family_commitment=dict(commitment),
        composite_family_hash=family_hash,
        provenance=provenance,
    )


@dataclass(frozen=True)
class DevelopmentAlphaSnapshot:
    values: pd.Series
    as_of: pd.Timestamp
    backend: str
    composite_family_hash: str
    development_root_hash: str
    source_feature_hash: str
    values_hash: str
    sealed_rows_loaded: int = 0
    tier: str = RESEARCH_PROXY_TIER

    def __post_init__(self) -> None:
        if self.tier != RESEARCH_PROXY_TIER or self.sealed_rows_loaded != 0:
            raise ResearchProxyBoundaryError("development alpha crossed its tier boundary")
        if self.backend not in {"equal_weight", "ridge", "lightgbm"}:
            raise ResearchProxyBoundaryError("development alpha backend is invalid")
        for name in (
            "composite_family_hash",
            "development_root_hash",
            "source_feature_hash",
            "values_hash",
        ):
            if len(str(getattr(self, name))) != 64:
                raise ResearchProxyBoundaryError(f"development alpha {name} is invalid")
        if self.values.index.has_duplicates:
            raise ResearchProxyBoundaryError("development alpha symbols are duplicated")
        _require_finite_series(self.values, label="development alpha values")
        if _series_snapshot_hash(self.values) != self.values_hash:
            raise ResearchProxyBoundaryError("development alpha values hash mismatch")


@dataclass(frozen=True)
class ResearchProxyControlSnapshot:
    as_of: pd.Timestamp
    covariance: pd.DataFrame
    industry: pd.Series
    style_exposures: pd.DataFrame
    adv20_dollars: pd.Series
    tradable: pd.Series

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", pd.Timestamp(self.as_of).normalize())


@dataclass(frozen=True)
class ResearchBenchmarkProxy:
    weights: pd.Series
    as_of: pd.Timestamp
    method: str
    input_hash: str
    weights_hash: str
    is_official: bool = False
    tier: str = RESEARCH_PROXY_TIER

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", pd.Timestamp(self.as_of).normalize())
        if self.is_official:
            raise ResearchProxyBoundaryError(
                "a proxy benchmark cannot be relabelled as official"
            )
        if self.tier != RESEARCH_PROXY_TIER:
            raise ResearchProxyBoundaryError("benchmark proxy tier cannot be upgraded")
        weights = self.weights
        if weights.index.has_duplicates or len(weights) < 2:
            raise ResearchProxyBoundaryError(
                "benchmark proxy needs at least two unique symbols"
            )
        numeric = _require_finite_series(weights, label="benchmark proxy weights")
        if (numeric < 0.0).any() or not np.isclose(numeric.sum(), 1.0, atol=1e-10):
            raise ResearchProxyBoundaryError(
                "benchmark proxy weights must be non-negative and sum to one"
            )
        if _series_snapshot_hash(self.weights) != self.weights_hash:
            raise ResearchProxyBoundaryError("benchmark proxy weights hash mismatch")


def build_log_mcap_benchmark_proxy(
    log_mcap: pd.Series,
    *,
    as_of: pd.Timestamp | str,
    universe_member: pd.Series | None = None,
) -> ResearchBenchmarkProxy:
    """Build an explicitly non-official cap-weight membership proxy."""

    values = log_mcap.copy()
    values.index = values.index.astype(str)
    if values.index.has_duplicates:
        raise ResearchProxyBoundaryError("log_mcap index contains duplicate symbols")
    numeric = _require_finite_series(values, label="log_mcap")
    if universe_member is not None:
        member = universe_member.copy()
        member.index = member.index.astype(str)
        if not member.index.equals(numeric.index):
            raise ResearchProxyBoundaryError(
                "universe_member must exactly align with log_mcap"
            )
        if member.isna().any() or not member.isin([True, False, 0, 1]).all():
            raise ResearchProxyBoundaryError("universe_member is not strict boolean")
        numeric = numeric.loc[member.astype(bool)]
    if len(numeric) < 2:
        raise ResearchProxyBoundaryError("benchmark proxy universe is too small")
    scaled = np.exp(numeric - float(numeric.max()))
    weights = scaled / scaled.sum()
    weights.name = "benchmark_proxy_weight"
    source = numeric.rename("log_mcap").to_frame()
    return ResearchBenchmarkProxy(
        weights=weights,
        as_of=pd.Timestamp(as_of).normalize(),
        method="point_in_time_member_implied_market_cap_proxy_v1",
        input_hash=panel_snapshot_hash(source),
        weights_hash=_series_snapshot_hash(weights),
    )


@dataclass(frozen=True)
class ResearchProxyPortfolioResult:
    optimization: OptimizationResult
    benchmark: ResearchBenchmarkProxy
    provenance: Mapping[str, object]
    tier: str = RESEARCH_PROXY_TIER
    disclaimer: str = RESEARCH_PROXY_DISCLAIMER
    pnl_authorized: bool = False

    def __post_init__(self) -> None:
        if self.tier != RESEARCH_PROXY_TIER or self.pnl_authorized:
            raise ResearchProxyBoundaryError(
                "research-proxy target weights cannot authorize P&L or live capital"
            )

    def require_event_driven_pnl_authority(self) -> None:
        raise ResearchProxyBoundaryError(
            "research proxy cannot generate P&L: official historical benchmark "
            "weights and a complete sourced corporate-action ledger are required"
        )


def optimize_research_proxy_portfolio(
    optimizer: ConstrainedPortfolioOptimizer,
    *,
    family: FrozenDevelopmentCompositeFamily,
    alpha_snapshot: DevelopmentAlphaSnapshot,
    benchmark: ResearchBenchmarkProxy,
    previous_weights: pd.Series,
    controls: ResearchProxyControlSnapshot,
    portfolio_value: float,
    bootstrap: bool = False,
) -> ResearchProxyPortfolioResult:
    """Strict research-only wrapper around the long-only optimizer."""

    if not isinstance(bootstrap, bool):
        raise ResearchProxyBoundaryError("bootstrap must be an explicit boolean")
    if bootstrap:
        raise ResearchProxyBoundaryError(
            "bootstrap=True is not implemented: no first-date turnover exemption "
            "is authorized by this research-proxy interface"
        )
    family.verify_integrity()
    if (
        alpha_snapshot.composite_family_hash != family.composite_family_hash
        or alpha_snapshot.backend != family.primary_backend
        or alpha_snapshot.development_root_hash
        != family.family_commitment["development_root_hash"]
    ):
        raise ResearchProxyBoundaryError(
            "typed alpha snapshot is not the frozen family's primary output"
        )
    if _series_snapshot_hash(alpha_snapshot.values) != alpha_snapshot.values_hash:
        raise ResearchProxyBoundaryError("typed alpha snapshot mutated after creation")
    alpha = alpha_snapshot.values
    names = pd.Index(alpha.index.astype(str), name="symbol")
    if names.has_duplicates or len(names) < 2:
        raise ResearchProxyBoundaryError("alpha must contain unique symbols")
    if alpha_snapshot.sealed_rows_loaded != 0:
        raise ResearchProxyBoundaryError("portfolio alpha consumed sealed information")
    if not (
        alpha_snapshot.as_of == benchmark.as_of == controls.as_of
    ):
        raise ResearchProxyBoundaryError(
            "alpha, benchmark and every risk/liquidity control must share one as-of date"
        )
    alpha_values = alpha.copy()
    alpha_values.index = names
    alpha_values = _require_finite_series(alpha_values, label="alpha")

    benchmark_names = pd.Index(benchmark.weights.index.astype(str), name="symbol")
    if _series_snapshot_hash(benchmark.weights) != benchmark.weights_hash:
        raise ResearchProxyBoundaryError("benchmark proxy mutated after construction")
    if not names.equals(benchmark_names):
        raise ResearchProxyBoundaryError(
            "alpha universe must exactly equal the benchmark-proxy universe"
        )

    previous = previous_weights.copy()
    previous.index = previous.index.astype(str)
    if not previous.index.equals(names):
        raise ResearchProxyBoundaryError(
            "previous weights must exactly cover the optimization universe"
        )
    previous = _require_finite_series(previous, label="previous weights")
    if (previous < 0.0).any() or previous.sum() > 1.000001:
        raise ResearchProxyBoundaryError("previous weights are not valid long-only weights")
    cash_to_invest = max(
        0.0,
        float(optimizer.constraints.target_investment - previous.sum()),
    )
    if cash_to_invest > optimizer.constraints.max_one_way_turnover + 1e-12:
        raise ResearchProxyBoundaryError(
            "initial/cash bootstrap would exceed the ordinary turnover limit; "
            "this interface does not silently grant a first-date exemption"
        )

    covariance = controls.covariance
    if set(covariance.index.astype(str)) != set(names) or set(
        covariance.columns.astype(str)
    ) != set(names):
        raise ResearchProxyBoundaryError("covariance must exactly cover the universe")
    covariance_values = covariance.copy()
    covariance_values.index = covariance_values.index.astype(str)
    covariance_values.columns = covariance_values.columns.astype(str)
    covariance_values = covariance_values.loc[names, names]
    matrix = covariance_values.apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, rtol=1e-8, atol=1e-12
    ):
        raise ResearchProxyBoundaryError(
            "daily covariance must be finite, complete and symmetric"
        )

    industries = controls.industry.copy()
    industries.index = industries.index.astype(str)
    if not industries.index.equals(names):
        raise ResearchProxyBoundaryError("industry must exactly cover the universe")
    industries = industries.astype("string")
    if industries.isna().any() or (industries.str.strip() == "").any():
        raise ResearchProxyBoundaryError("industry must be complete and non-blank")

    styles = controls.style_exposures.copy()
    styles.index = styles.index.astype(str)
    if not styles.index.equals(names):
        raise ResearchProxyBoundaryError(
            "style exposures must exactly cover the universe"
        )
    missing_styles = set(REQUIRED_PORTFOLIO_STYLES).difference(styles.columns)
    if missing_styles:
        raise ResearchProxyBoundaryError(
            "style exposures omit required columns: " + ",".join(sorted(missing_styles))
        )
    styles = styles.loc[:, REQUIRED_PORTFOLIO_STYLES].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(styles.to_numpy(dtype=float)).all():
        raise ResearchProxyBoundaryError("style exposures must be finite and complete")
    missing_bounds = set(REQUIRED_PORTFOLIO_STYLES).difference(
        optimizer.constraints.style_active_bounds
    )
    if missing_bounds:
        raise ResearchProxyBoundaryError(
            "optimizer omits required style bounds: "
            + ",".join(sorted(missing_bounds))
        )

    adv = controls.adv20_dollars.copy()
    adv.index = adv.index.astype(str)
    if not adv.index.equals(names):
        raise ResearchProxyBoundaryError("ADV20 must exactly cover the universe")
    adv = _require_finite_series(adv, label="ADV20", positive=True)
    if not np.isfinite(float(portfolio_value)) or portfolio_value <= 0:
        raise ResearchProxyBoundaryError("portfolio value must be positive and finite")

    can_trade = controls.tradable.copy()
    can_trade.index = can_trade.index.astype(str)
    if not can_trade.index.equals(names):
        raise ResearchProxyBoundaryError("tradable must exactly cover the universe")
    if can_trade.isna().any() or not can_trade.isin([True, False, 0, 1]).all():
        raise ResearchProxyBoundaryError("tradable must be strict boolean")
    can_trade = can_trade.astype(bool)

    result = optimizer.optimize(
        alpha_values,
        benchmark.weights,
        previous,
        covariance_values,
        industries,
        style_exposures=styles,
        adv20_dollars=adv,
        portfolio_value=float(portfolio_value),
        tradable=can_trade,
    )
    provenance = {
        "tier": RESEARCH_PROXY_TIER,
        "disclaimer": RESEARCH_PROXY_DISCLAIMER,
        "alpha_backend": alpha_snapshot.backend,
        "alpha_as_of": str(alpha_snapshot.as_of.date()),
        "alpha_composite_family_hash": alpha_snapshot.composite_family_hash,
        "alpha_development_root_hash": alpha_snapshot.development_root_hash,
        "alpha_source_feature_hash": alpha_snapshot.source_feature_hash,
        "alpha_values_hash": alpha_snapshot.values_hash,
        "benchmark_method": benchmark.method,
        "benchmark_is_official": False,
        "benchmark_input_hash": benchmark.input_hash,
        "benchmark_weights_hash": benchmark.weights_hash,
        "benchmark_as_of": str(benchmark.as_of.date()),
        "covariance_frequency": "daily",
        "covariance_hash": panel_snapshot_hash(covariance_values),
        "style_hash": panel_snapshot_hash(styles),
        "industry_hash": _series_snapshot_hash(industries),
        "adv20_hash": _series_snapshot_hash(adv),
        "tradable_hash": _series_snapshot_hash(can_trade.astype(int)),
        "previous_weights_hash": _series_snapshot_hash(previous),
        "previous_risky_weight": float(previous.sum()),
        "previous_cash_weight": float(1.0 - previous.sum()),
        "portfolio_value": float(portfolio_value),
        "bootstrap_exemption": False,
        "pnl_authorized": False,
        "sealed_rows_loaded": 0,
        "sealed_results_consumed": False,
    }
    return ResearchProxyPortfolioResult(result, benchmark, provenance)


__all__ = [
    "CompositeFamilyPolicy",
    "DevelopmentAlphaSnapshot",
    "DevelopmentFeatureHandoff",
    "DevelopmentHandoffPolicy",
    "FROZEN_CHAMPION_COUNT",
    "FrozenChampionCommitment",
    "FrozenDevelopmentCompositeFamily",
    "FrozenDevelopmentCommitment",
    "RESEARCH_PROXY_DISCLAIMER",
    "RESEARCH_PROXY_TIER",
    "REQUIRED_PORTFOLIO_STYLES",
    "ResearchBenchmarkProxy",
    "ResearchProxyBoundaryError",
    "ResearchProxyControlSnapshot",
    "ResearchProxyPortfolioResult",
    "SealedCompositeClaimCapability",
    "build_development_handoff",
    "build_development_handoff_from_commitment",
    "build_log_mcap_benchmark_proxy",
    "dump_frozen_composite_family",
    "fit_frozen_development_composite_family",
    "load_frozen_composite_family",
    "optimize_research_proxy_portfolio",
]
