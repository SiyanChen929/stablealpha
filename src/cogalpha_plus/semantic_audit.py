"""Fail-closed economic-semantic review manifests for frozen Factor Cards.

This module deliberately does *not* infer, score, or approve semantic reviews.
The caller must supply one complete human/LLM review record for every frozen
Factor Card.  The five proposal file digests and their exact candidate-key set
are code-side trust anchors.  While those anchors remain trusted, changing
proposals and coherently re-sealing only the review manifest is rejected.  This
is not a signature: an actor who can rewrite the anchors, proposals, and
manifest can re-seal all local evidence, so an externally stored digest or
signature is needed to defend that stronger threat model.

Likewise, reviewer names and notes are explicit attestations, not authenticated
identity or proof that a review was performed independently.  That assurance
must come from the review workflow and any external signature it produces.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .manifest import canonical_json, hash_json


SEMANTIC_REVIEW_SCHEMA = "cogalpha-plus.semantic-review/v1"
SEMANTIC_REVIEW_SCHEMA_VERSION = 1
EXPECTED_CANDIDATE_COUNT = 176

PROPOSAL_PATHS = (
    "proposals/codex_seed/l1_l2.jsonl",
    "proposals/codex_seed/l3_l4.jsonl",
    "proposals/codex_seed/l5_l7.jsonl",
    "proposals/codex_injection/l1_l4.jsonl",
    "proposals/codex_injection/l5_l7.jsonl",
)

# These anchors are filled from the frozen, independently reviewed proposal
# library.  They must only change in the same reviewed source-code change as an
# intentional proposal-library revision.
FROZEN_PROPOSAL_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "proposals/codex_seed/l1_l2.jsonl": "283461e2eb7101cb448b19b1d4a924180e01075dfdf0eede1d3d5014338c164a",
        "proposals/codex_seed/l3_l4.jsonl": "d85ecc041272d18dd713fec13625540d875e3cafbd5901005bfd641e9e832091",
        "proposals/codex_seed/l5_l7.jsonl": "7ce9a02009fb327b961055674df680729f15110f2b2e6c2549aeb15fa6867543",
        "proposals/codex_injection/l1_l4.jsonl": "5e1f542e5448252575337b18ed2453e292f7781e962376438ffff440a5e7c624",
        "proposals/codex_injection/l5_l7.jsonl": "3ae938b8f4679acaf958c675be195a5335e4da473ad31abad4e972a921bd0f3c",
    }
)
FROZEN_CANDIDATE_KEYS = frozenset(
    {
        "L1MC_001_cycle_acceleration",
        "L1MC_002_participation_confirmed_cycle",
        "L1MC_003_distribution_with_fading_volume",
        "L1MC_004_directional_cycle_efficiency",
        "L1VR_001_recent_volatility_expansion",
        "L1VR_002_trend_under_compression",
        "L1VR_003_confirmed_range_breakout",
        "L1VR_004_intraday_stress_absorption",
        "L2CP_001_fragile_closing_markup",
        "L2CP_002_down_gap_range_stress",
        "L2CP_003_downside_acceleration_liquidity_fade",
        "L2CP_004_turbulent_path_withdrawal",
        "L2TR_001_downside_semivariance_share",
        "L2TR_002_recent_downside_stress_ratio",
        "L2TR_003_weak_recovery_asymmetry",
        "L2TR_004_repeated_down_gap_share",
        "codex_inject_l1_market_cycle_a01",
        "codex_inject_l1_market_cycle_b01",
        "codex_inject_l1_market_cycle_c01",
        "codex_inject_l1_market_cycle_d01",
        "codex_inject_l1_volatility_regime_a01",
        "codex_inject_l1_volatility_regime_b01",
        "codex_inject_l1_volatility_regime_c01",
        "codex_inject_l1_volatility_regime_d01",
        "codex_inject_l2_crash_predictor_a01",
        "codex_inject_l2_crash_predictor_b01",
        "codex_inject_l2_crash_predictor_c01",
        "codex_inject_l2_crash_predictor_d01",
        "codex_inject_l2_tail_risk_a01",
        "codex_inject_l2_tail_risk_b01",
        "codex_inject_l2_tail_risk_c01",
        "codex_inject_l2_tail_risk_d01",
        "codex_inject_l3_liquidity_a01",
        "codex_inject_l3_liquidity_b01",
        "codex_inject_l3_liquidity_c01",
        "codex_inject_l3_liquidity_d01",
        "codex_inject_l3_order_imbalance_a01",
        "codex_inject_l3_order_imbalance_b01",
        "codex_inject_l3_order_imbalance_c01",
        "codex_inject_l3_order_imbalance_d01",
        "codex_inject_l3_price_volume_coherence_a01",
        "codex_inject_l3_price_volume_coherence_b01",
        "codex_inject_l3_price_volume_coherence_c01",
        "codex_inject_l3_price_volume_coherence_d01",
        "codex_inject_l3_volume_structure_a01",
        "codex_inject_l3_volume_structure_b01",
        "codex_inject_l3_volume_structure_c01",
        "codex_inject_l3_volume_structure_d01",
        "codex_inject_l4_daily_trend_a01",
        "codex_inject_l4_daily_trend_b01",
        "codex_inject_l4_daily_trend_c01",
        "codex_inject_l4_daily_trend_d01",
        "codex_inject_l4_lag_response_a01",
        "codex_inject_l4_lag_response_b01",
        "codex_inject_l4_lag_response_c01",
        "codex_inject_l4_lag_response_d01",
        "codex_inject_l4_range_vol_a01",
        "codex_inject_l4_range_vol_b01",
        "codex_inject_l4_range_vol_c01",
        "codex_inject_l4_range_vol_d01",
        "codex_inject_l4_reversal_a01",
        "codex_inject_l4_reversal_b01",
        "codex_inject_l4_reversal_c01",
        "codex_inject_l4_reversal_d01",
        "codex_inject_l4_vol_asymmetry_a01",
        "codex_inject_l4_vol_asymmetry_b01",
        "codex_inject_l4_vol_asymmetry_c01",
        "codex_inject_l4_vol_asymmetry_d01",
        "codex_inject_l5_drawdown_a01",
        "codex_inject_l5_drawdown_b01",
        "codex_inject_l5_drawdown_c01",
        "codex_inject_l5_drawdown_d01",
        "codex_inject_l5_fractal_a01",
        "codex_inject_l5_fractal_b01",
        "codex_inject_l5_fractal_c01",
        "codex_inject_l5_fractal_d01",
        "codex_inject_l5_herding_a01",
        "codex_inject_l5_herding_b01",
        "codex_inject_l5_herding_c01",
        "codex_inject_l5_herding_d01",
        "codex_inject_l6_regime_a01",
        "codex_inject_l6_regime_b01",
        "codex_inject_l6_regime_c01",
        "codex_inject_l6_regime_d01",
        "codex_inject_l6_regime_e01",
        "codex_inject_l6_regime_f01",
        "codex_inject_l6_stability_a01",
        "codex_inject_l6_stability_b01",
        "codex_inject_l6_stability_c01",
        "codex_inject_l6_stability_d01",
        "codex_inject_l6_stability_e01",
        "codex_inject_l6_stability_f01",
        "codex_inject_l7_bar_a01",
        "codex_inject_l7_bar_b01",
        "codex_inject_l7_bar_c01",
        "codex_inject_l7_bar_d01",
        "codex_inject_l7_bar_e01",
        "codex_inject_l7_bar_f01",
        "codex_inject_l7_composite_a01",
        "codex_inject_l7_composite_b01",
        "codex_inject_l7_composite_c01",
        "codex_inject_l7_composite_d01",
        "codex_inject_l7_composite_e01",
        "codex_inject_l7_composite_f01",
        "codex_inject_l7_composite_g01",
        "codex_inject_l7_creative_a01",
        "codex_inject_l7_creative_b01",
        "codex_inject_l7_creative_c01",
        "codex_inject_l7_creative_d01",
        "codex_inject_l7_creative_e01",
        "codex_inject_l7_creative_f01",
        "codex_inject_l7_creative_g01",
        "codex_l3_liquidity_01",
        "codex_l3_liquidity_02",
        "codex_l3_liquidity_03",
        "codex_l3_liquidity_04",
        "codex_l3_order_imbalance_01",
        "codex_l3_order_imbalance_02",
        "codex_l3_order_imbalance_03",
        "codex_l3_order_imbalance_04",
        "codex_l3_price_volume_coherence_01",
        "codex_l3_price_volume_coherence_02",
        "codex_l3_price_volume_coherence_03",
        "codex_l3_price_volume_coherence_04",
        "codex_l3_volume_structure_01",
        "codex_l3_volume_structure_02",
        "codex_l3_volume_structure_03",
        "codex_l3_volume_structure_04",
        "codex_l4_daily_trend_01",
        "codex_l4_daily_trend_02",
        "codex_l4_daily_trend_03",
        "codex_l4_lag_response_01",
        "codex_l4_lag_response_02",
        "codex_l4_lag_response_03",
        "codex_l4_range_vol_01",
        "codex_l4_range_vol_02",
        "codex_l4_range_vol_03",
        "codex_l4_reversal_01",
        "codex_l4_reversal_02",
        "codex_l4_reversal_03",
        "codex_l4_vol_asymmetry_01",
        "codex_l4_vol_asymmetry_02",
        "codex_l4_vol_asymmetry_03",
        "codex_l4_vol_asymmetry_04",
        "codex_seed_l5_drawdown_01",
        "codex_seed_l5_drawdown_02",
        "codex_seed_l5_drawdown_03",
        "codex_seed_l5_drawdown_04",
        "codex_seed_l5_fractal_01",
        "codex_seed_l5_fractal_02",
        "codex_seed_l5_fractal_03",
        "codex_seed_l5_fractal_04",
        "codex_seed_l5_herding_01",
        "codex_seed_l5_herding_02",
        "codex_seed_l5_herding_03",
        "codex_seed_l5_herding_04",
        "codex_seed_l6_regime_gating_01",
        "codex_seed_l6_regime_gating_02",
        "codex_seed_l6_regime_gating_03",
        "codex_seed_l6_regime_gating_04",
        "codex_seed_l6_stability_01",
        "codex_seed_l6_stability_02",
        "codex_seed_l6_stability_03",
        "codex_seed_l6_stability_04",
        "codex_seed_l7_bar_shape_01",
        "codex_seed_l7_bar_shape_02",
        "codex_seed_l7_bar_shape_03",
        "codex_seed_l7_bar_shape_04",
        "codex_seed_l7_composite_01",
        "codex_seed_l7_composite_02",
        "codex_seed_l7_composite_03",
        "codex_seed_l7_composite_04",
        "codex_seed_l7_creative_01",
        "codex_seed_l7_creative_02",
        "codex_seed_l7_creative_03",
        "codex_seed_l7_creative_04",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_INPUT_KEYS = frozenset(
    {
        "candidate_key",
        "status",
        "reviewer",
        "note",
        "mechanism_ast_consistent",
        "direction_consistent",
        "repair_tags_consistent",
        "falsification_reviewed",
    }
)
_REVIEW_MANIFEST_KEYS = _REVIEW_INPUT_KEYS | {"review_hash"}
_PROPOSAL_RECORD_KEYS = frozenset(
    {
        "relative_path",
        "sha256",
        "card_count",
        "candidate_keys",
        "candidate_keys_hash",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "policy",
        "reviewer",
        "reviewed_at",
        "proposal_files",
        "proposal_snapshot_hash",
        "candidate_count",
        "candidate_keys",
        "candidate_keys_hash",
        "reviews",
        "review_set_hash",
        "approval_binding_hash",
    }
)
_ENVELOPE_KEYS = frozenset({"semantic_review_manifest", "aggregate_hash"})
_POLICY = "explicit-per-card-economic-semantic-review; no-auto-approval"


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    actual = set(value)
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if missing or extra:
        raise ValueError(
            f"{context} schema mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json_strict(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_object_pairs,
                parse_constant=reject_constant,
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load semantic review manifest {path}: {exc}") from exc


def _read_proposal_snapshot(path: Path) -> tuple[str, list[str]]:
    """Hash and parse one proposal from the same immutable byte snapshot."""

    try:
        with path.open("rb") as handle:
            payload = handle.read()
    except OSError as exc:
        raise ValueError(f"cannot read frozen proposal {path}: {exc}") from exc

    digest = sha256(payload).hexdigest()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"cannot decode frozen proposal {path} as UTF-8: {exc}") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    keys: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_object_pairs,
                parse_constant=reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"{path}:{line_number}: card must be a JSON object")
        key = record.get("candidate_key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{path}:{line_number}: candidate_key must be non-empty")
        keys.append(key)
    return digest, keys


def _validate_timestamp(reviewed_at: Any) -> str:
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ValueError("reviewed_at must be an explicit non-empty UTC timestamp")
    if not reviewed_at.endswith("Z"):
        raise ValueError("reviewed_at must use the UTC 'Z' suffix")
    try:
        parsed = datetime.fromisoformat(reviewed_at[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("reviewed_at must be a valid ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("reviewed_at must be UTC")
    return reviewed_at


def _validate_reviewer(reviewer: Any, *, context: str = "reviewer") -> str:
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return reviewer


def _assert_code_anchors_complete() -> None:
    if set(FROZEN_PROPOSAL_SHA256) != set(PROPOSAL_PATHS):
        raise RuntimeError(
            "semantic review proposal SHA anchors are incomplete; review cannot pass"
        )
    if len(FROZEN_CANDIDATE_KEYS) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError(
            "semantic review candidate-key anchor is incomplete; review cannot pass"
        )
    for relative_path, digest in FROZEN_PROPOSAL_SHA256.items():
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise RuntimeError(f"invalid frozen SHA-256 anchor for {relative_path}")


def _proposal_snapshot(project: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read proposals and verify them against the code-side frozen anchors."""

    _assert_code_anchors_complete()
    root = Path(project).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"project is not a directory: {root}")

    records: list[dict[str, Any]] = []
    all_keys: list[str] = []
    seen: set[str] = set()
    for relative_path in PROPOSAL_PATHS:
        path = root / relative_path
        actual_sha, file_keys = _read_proposal_snapshot(path)
        expected_sha = FROZEN_PROPOSAL_SHA256[relative_path]
        if actual_sha != expected_sha:
            raise ValueError(
                f"frozen proposal hash drift for {relative_path}: "
                f"expected {expected_sha}, found {actual_sha}"
            )
        duplicate = seen.intersection(file_keys)
        if len(set(file_keys)) != len(file_keys):
            raise ValueError(f"duplicate candidate_key within {relative_path}")
        if duplicate:
            raise ValueError(
                f"duplicate candidate_key across proposals: {sorted(duplicate)[0]}"
            )
        seen.update(file_keys)
        all_keys.extend(file_keys)
        ordered_file_keys = sorted(file_keys)
        records.append(
            {
                "relative_path": relative_path,
                "sha256": actual_sha,
                "card_count": len(file_keys),
                "candidate_keys": ordered_file_keys,
                "candidate_keys_hash": hash_json(ordered_file_keys),
            }
        )

    if len(all_keys) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_CANDIDATE_COUNT} proposal cards, found {len(all_keys)}"
        )
    actual_key_set = frozenset(all_keys)
    if actual_key_set != FROZEN_CANDIDATE_KEYS:
        missing = sorted(FROZEN_CANDIDATE_KEYS.difference(actual_key_set))
        extra = sorted(actual_key_set.difference(FROZEN_CANDIDATE_KEYS))
        raise ValueError(
            "frozen candidate_key set drift: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return records, sorted(all_keys)


def _review_body(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in sorted(_REVIEW_INPUT_KEYS)}


def _validate_review_input(
    raw: Any,
    *,
    expected_keys: frozenset[str],
    manifest_reviewer: str,
    context: str,
    require_hash: bool,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    expected_schema = _REVIEW_MANIFEST_KEYS if require_hash else _REVIEW_INPUT_KEYS
    _require_exact_keys(raw, expected_schema, context=context)

    candidate_key = raw["candidate_key"]
    if not isinstance(candidate_key, str) or candidate_key not in expected_keys:
        raise ValueError(f"{context} has an unknown candidate_key")
    if raw["status"] != "approved":
        raise ValueError(f"{context} status must be exactly 'approved'")
    reviewer = _validate_reviewer(raw["reviewer"], context=f"{context}.reviewer")
    if reviewer != manifest_reviewer:
        raise ValueError(f"{context}.reviewer must match manifest reviewer")
    note = raw["note"]
    if not isinstance(note, str) or not note.strip():
        raise ValueError(f"{context}.note must be a non-empty explicit review note")

    for field in (
        "mechanism_ast_consistent",
        "direction_consistent",
        "repair_tags_consistent",
        "falsification_reviewed",
    ):
        if raw[field] is not True:
            raise ValueError(f"{context}.{field} must be the boolean true")

    body = _review_body(raw)
    normalized = json.loads(canonical_json(body))
    review_hash = hash_json(normalized)
    if require_hash:
        recorded_hash = raw["review_hash"]
        if not isinstance(recorded_hash, str) or recorded_hash != review_hash:
            raise ValueError(f"{context}.review_hash mismatch")
    normalized["review_hash"] = review_hash
    return normalized


def _binding_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest["schema"],
        "schema_version": manifest["schema_version"],
        "policy": manifest["policy"],
        "reviewer": manifest["reviewer"],
        "reviewed_at": manifest["reviewed_at"],
        "proposal_snapshot_hash": manifest["proposal_snapshot_hash"],
        "candidate_keys_hash": manifest["candidate_keys_hash"],
        "review_set_hash": manifest["review_set_hash"],
    }


def build_semantic_review_manifest(
    project: str | Path,
    reviews: Sequence[Mapping[str, Any]],
    reviewer: str,
    reviewed_at: str,
) -> dict[str, Any]:
    """Build a fully approved manifest from 176 *explicit* per-card reviews.

    There are intentionally no review defaults and no auto-approval path.  Each
    input record must explicitly contain all four ``True`` attestations,
    ``status='approved'``, the same non-empty ``reviewer``, and a non-empty
    evidence note.  The function returns a JSON-compatible envelope; it does not
    write or install an approved manifest.
    """

    manifest_reviewer = _validate_reviewer(reviewer)
    timestamp = _validate_timestamp(reviewed_at)
    if isinstance(reviews, (str, bytes, Mapping)) or not isinstance(
        reviews, Sequence
    ):
        raise TypeError("reviews must be an explicit sequence of review records")
    if len(reviews) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_CANDIDATE_COUNT} reviews, found {len(reviews)}"
        )

    proposal_files, candidate_keys = _proposal_snapshot(project)
    expected_keys = frozenset(candidate_keys)
    normalized_reviews: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(reviews):
        review = _validate_review_input(
            raw,
            expected_keys=expected_keys,
            manifest_reviewer=manifest_reviewer,
            context=f"reviews[{index}]",
            require_hash=False,
        )
        key = review["candidate_key"]
        if key in seen:
            raise ValueError(f"duplicate review candidate_key: {key}")
        seen.add(key)
        normalized_reviews.append(review)
    if seen != expected_keys:
        missing = sorted(expected_keys.difference(seen))
        extra = sorted(seen.difference(expected_keys))
        raise ValueError(
            f"review candidate set mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    normalized_reviews.sort(key=lambda item: item["candidate_key"])

    manifest: dict[str, Any] = {
        "schema": SEMANTIC_REVIEW_SCHEMA,
        "schema_version": SEMANTIC_REVIEW_SCHEMA_VERSION,
        "policy": _POLICY,
        "reviewer": manifest_reviewer,
        "reviewed_at": timestamp,
        "proposal_files": proposal_files,
        "proposal_snapshot_hash": hash_json(proposal_files),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "candidate_keys": candidate_keys,
        "candidate_keys_hash": hash_json(candidate_keys),
        "reviews": normalized_reviews,
        "review_set_hash": hash_json(normalized_reviews),
        "approval_binding_hash": "",
    }
    manifest["approval_binding_hash"] = hash_json(_binding_payload(manifest))
    return {
        "semantic_review_manifest": manifest,
        "aggregate_hash": hash_json(manifest),
    }


def _validate_proposal_records(
    recorded: Any, actual: list[dict[str, Any]]
) -> None:
    if not isinstance(recorded, list) or len(recorded) != len(PROPOSAL_PATHS):
        raise ValueError("proposal_files must contain exactly the five frozen files")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(recorded):
        if not isinstance(item, Mapping):
            raise ValueError(f"proposal_files[{index}] must be a JSON object")
        _require_exact_keys(item, _PROPOSAL_RECORD_KEYS, context=f"proposal_files[{index}]")
        relative_path = item["relative_path"]
        sha = item["sha256"]
        card_count = item["card_count"]
        keys = item["candidate_keys"]
        key_hash = item["candidate_keys_hash"]
        if not isinstance(relative_path, str):
            raise ValueError(f"proposal_files[{index}].relative_path must be a string")
        if not isinstance(sha, str) or _SHA256_RE.fullmatch(sha) is None:
            raise ValueError(f"proposal_files[{index}].sha256 is invalid")
        if not isinstance(card_count, int) or isinstance(card_count, bool):
            raise ValueError(f"proposal_files[{index}].card_count must be an integer")
        if (
            not isinstance(keys, list)
            or any(not isinstance(key, str) or not key for key in keys)
            or len(keys) != len(set(keys))
            or keys != sorted(keys)
        ):
            raise ValueError(f"proposal_files[{index}].candidate_keys is invalid")
        if card_count != len(keys):
            raise ValueError(f"proposal_files[{index}].card_count mismatch")
        if key_hash != hash_json(keys):
            raise ValueError(f"proposal_files[{index}].candidate_keys_hash mismatch")
        normalized.append(dict(item))
    if normalized != actual:
        raise ValueError("proposal_files do not match the live frozen proposal snapshot")


def validate_semantic_review_manifest(
    project: str | Path, path: str | Path
) -> dict[str, Any]:
    """Validate and return a semantic-review manifest, or fail closed.

    Validation re-reads all five proposal files and checks the code-side SHA and
    candidate-key anchors before trusting any manifest-provided digest.  It then
    checks exact schemas, every per-card decision, nested hashes, and the
    canonical aggregate hash.
    """

    envelope = _load_json_strict(Path(path))
    if not isinstance(envelope, Mapping):
        raise ValueError("semantic review envelope must be a JSON object")
    _require_exact_keys(envelope, _ENVELOPE_KEYS, context="semantic review envelope")
    manifest = envelope["semantic_review_manifest"]
    if not isinstance(manifest, Mapping):
        raise ValueError("semantic_review_manifest must be a JSON object")
    _require_exact_keys(manifest, _MANIFEST_KEYS, context="semantic_review_manifest")

    aggregate_hash = envelope["aggregate_hash"]
    if not isinstance(aggregate_hash, str) or aggregate_hash != hash_json(manifest):
        raise ValueError("semantic review aggregate hash mismatch")
    if manifest["schema"] != SEMANTIC_REVIEW_SCHEMA:
        raise ValueError("unsupported semantic review schema")
    if (
        not isinstance(manifest["schema_version"], int)
        or isinstance(manifest["schema_version"], bool)
        or manifest["schema_version"] != SEMANTIC_REVIEW_SCHEMA_VERSION
    ):
        raise ValueError("unsupported semantic review schema version")
    if manifest["policy"] != _POLICY:
        raise ValueError("semantic review policy mismatch")
    manifest_reviewer = _validate_reviewer(manifest["reviewer"])
    _validate_timestamp(manifest["reviewed_at"])
    if (
        not isinstance(manifest["candidate_count"], int)
        or isinstance(manifest["candidate_count"], bool)
        or manifest["candidate_count"] != EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("semantic review candidate_count must be exactly 176")

    actual_proposals, actual_keys = _proposal_snapshot(project)
    _validate_proposal_records(manifest["proposal_files"], actual_proposals)
    if manifest["proposal_snapshot_hash"] != hash_json(actual_proposals):
        raise ValueError("proposal_snapshot_hash mismatch")
    keys = manifest["candidate_keys"]
    if (
        not isinstance(keys, list)
        or any(not isinstance(key, str) for key in keys)
        or len(keys) != len(set(keys))
        or keys != actual_keys
    ):
        raise ValueError("manifest candidate_keys do not match the exact frozen set")
    if manifest["candidate_keys_hash"] != hash_json(actual_keys):
        raise ValueError("candidate_keys_hash mismatch")

    reviews = manifest["reviews"]
    if not isinstance(reviews, list) or len(reviews) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("reviews must contain exactly 176 explicit records")
    normalized_reviews: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_keys = frozenset(actual_keys)
    for index, raw in enumerate(reviews):
        review = _validate_review_input(
            raw,
            expected_keys=expected_keys,
            manifest_reviewer=manifest_reviewer,
            context=f"reviews[{index}]",
            require_hash=True,
        )
        key = review["candidate_key"]
        if key in seen:
            raise ValueError(f"duplicate review candidate_key: {key}")
        seen.add(key)
        normalized_reviews.append(review)
    if seen != expected_keys:
        raise ValueError("review candidate set does not match the exact frozen set")
    if [review["candidate_key"] for review in normalized_reviews] != actual_keys:
        raise ValueError("reviews must be in canonical candidate_key order")
    if manifest["review_set_hash"] != hash_json(normalized_reviews):
        raise ValueError("review_set_hash mismatch")
    if manifest["approval_binding_hash"] != hash_json(_binding_payload(manifest)):
        raise ValueError("approval_binding_hash mismatch")

    # A canonical round-trip also rejects non-JSON scalar types should this
    # function ever be called with a non-file decoder in the future.
    normalized_manifest = json.loads(canonical_json(manifest))
    return normalized_manifest


__all__ = [
    "EXPECTED_CANDIDATE_COUNT",
    "FROZEN_CANDIDATE_KEYS",
    "FROZEN_PROPOSAL_SHA256",
    "PROPOSAL_PATHS",
    "SEMANTIC_REVIEW_SCHEMA",
    "SEMANTIC_REVIEW_SCHEMA_VERSION",
    "build_semantic_review_manifest",
    "validate_semantic_review_manifest",
]
