from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from types import MappingProxyType

import pytest

from cogalpha_plus.manifest import canonical_json, hash_json
import cogalpha_plus.semantic_audit as semantic_audit


REVIEWER = "independent-semantic-reviewer"
REVIEWED_AT = "2026-08-31T17:42:11Z"


def _write_fake_proposals(root: Path) -> list[str]:
    counts = (36, 36, 36, 34, 34)
    keys: list[str] = []
    offset = 0
    for relative_path, count in zip(semantic_audit.PROPOSAL_PATHS, counts, strict=True):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for number in range(offset, offset + count):
            key = f"semantic-fixture-{number:03d}"
            keys.append(key)
            rows.append(
                canonical_json(
                    {
                        "candidate_key": key,
                        "hypothesis": f"fixture mechanism {number}",
                    }
                )
            )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        offset += count
    assert len(keys) == semantic_audit.EXPECTED_CANDIDATE_COUNT
    return keys


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _install_fixture_anchors(
    root: Path, keys: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    anchors = {
        relative_path: _file_sha(root / relative_path)
        for relative_path in semantic_audit.PROPOSAL_PATHS
    }
    monkeypatch.setattr(
        semantic_audit, "FROZEN_PROPOSAL_SHA256", MappingProxyType(anchors)
    )
    monkeypatch.setattr(semantic_audit, "FROZEN_CANDIDATE_KEYS", frozenset(keys))


@pytest.fixture
def semantic_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[str]]:
    keys = _write_fake_proposals(tmp_path)
    _install_fixture_anchors(tmp_path, keys, monkeypatch)
    return tmp_path, keys


def _reviews(keys: list[str]) -> list[dict[str, object]]:
    return [
        {
            "candidate_key": key,
            "status": "approved",
            "reviewer": REVIEWER,
            "note": f"Independently reconciled mechanism, AST, direction, repair tags, and falsification for {key}.",
            "mechanism_ast_consistent": True,
            "direction_consistent": True,
            "repair_tags_consistent": True,
            "falsification_reviewed": True,
        }
        for key in keys
    ]


def _build(root: Path, keys: list[str]) -> dict[str, object]:
    return semantic_audit.build_semantic_review_manifest(
        root, _reviews(keys), REVIEWER, REVIEWED_AT
    )


def _save(root: Path, envelope: dict[str, object], name: str = "semantic.json") -> Path:
    path = root / name
    path.write_text(canonical_json(envelope) + "\n", encoding="utf-8")
    return path


def _reseal(envelope: dict[str, object]) -> None:
    manifest = envelope["semantic_review_manifest"]
    assert isinstance(manifest, dict)
    reviews = manifest["reviews"]
    assert isinstance(reviews, list)
    for raw in reviews:
        assert isinstance(raw, dict)
        body = {key: value for key, value in raw.items() if key != "review_hash"}
        raw["review_hash"] = hash_json(body)
    manifest["review_set_hash"] = hash_json(reviews)
    manifest["approval_binding_hash"] = hash_json(
        semantic_audit._binding_payload(manifest)
    )
    envelope["aggregate_hash"] = hash_json(manifest)


def test_explicit_review_manifest_round_trip_and_canonical_hash(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    envelope = _build(root, keys)
    path = _save(root, envelope)

    loaded = semantic_audit.validate_semantic_review_manifest(root, path)

    manifest = envelope["semantic_review_manifest"]
    assert envelope["aggregate_hash"] == hash_json(manifest)
    assert loaded["candidate_count"] == 176
    assert loaded["candidate_keys"] == sorted(keys)
    assert len(loaded["reviews"]) == 176
    assert all(review["status"] == "approved" for review in loaded["reviews"])
    assert all(review["review_hash"] for review in loaded["reviews"])


def test_builder_has_no_default_or_mapping_auto_approval_path(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    with pytest.raises(TypeError, match="explicit sequence"):
        semantic_audit.build_semantic_review_manifest(
            root,
            {key: {} for key in keys},  # type: ignore[arg-type]
            REVIEWER,
            REVIEWED_AT,
        )


@pytest.mark.parametrize("drop_count", [1, 2])
def test_builder_rejects_missing_review_cards(
    semantic_project: tuple[Path, list[str]], drop_count: int
) -> None:
    root, keys = semantic_project
    with pytest.raises(ValueError, match="exactly 176 reviews"):
        semantic_audit.build_semantic_review_manifest(
            root, _reviews(keys)[:-drop_count], REVIEWER, REVIEWED_AT
        )


def test_builder_rejects_extra_review_card(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    reviews = _reviews(keys)
    reviews.append(deepcopy(reviews[0]))
    with pytest.raises(ValueError, match="exactly 176 reviews"):
        semantic_audit.build_semantic_review_manifest(
            root, reviews, REVIEWER, REVIEWED_AT
        )


def test_builder_rejects_duplicate_review_key_even_at_exact_count(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    reviews = _reviews(keys)
    reviews[-1] = deepcopy(reviews[0])
    with pytest.raises(ValueError, match="duplicate review candidate_key"):
        semantic_audit.build_semantic_review_manifest(
            root, reviews, REVIEWER, REVIEWED_AT
        )


@pytest.mark.parametrize("status", ["pending", "rejected", "APPROVED", True])
def test_builder_rejects_every_nonapproved_status(
    semantic_project: tuple[Path, list[str]], status: object
) -> None:
    root, keys = semantic_project
    reviews = _reviews(keys)
    reviews[0]["status"] = status
    with pytest.raises(ValueError, match="status must be exactly 'approved'"):
        semantic_audit.build_semantic_review_manifest(
            root, reviews, REVIEWER, REVIEWED_AT
        )


@pytest.mark.parametrize(
    "field",
    [
        "mechanism_ast_consistent",
        "direction_consistent",
        "repair_tags_consistent",
        "falsification_reviewed",
    ],
)
def test_builder_rejects_missing_or_nontrue_attestation(
    semantic_project: tuple[Path, list[str]], field: str
) -> None:
    root, keys = semantic_project
    missing = _reviews(keys)
    del missing[0][field]
    with pytest.raises(ValueError, match="schema mismatch"):
        semantic_audit.build_semantic_review_manifest(
            root, missing, REVIEWER, REVIEWED_AT
        )

    false_value = _reviews(keys)
    false_value[0][field] = False
    with pytest.raises(ValueError, match=f"{field} must be the boolean true"):
        semantic_audit.build_semantic_review_manifest(
            root, false_value, REVIEWER, REVIEWED_AT
        )


@pytest.mark.parametrize("field,value", [("note", "  "), ("reviewer", "")])
def test_builder_rejects_empty_note_or_per_card_reviewer(
    semantic_project: tuple[Path, list[str]], field: str, value: str
) -> None:
    root, keys = semantic_project
    reviews = _reviews(keys)
    reviews[0][field] = value
    with pytest.raises(ValueError, match="non-empty"):
        semantic_audit.build_semantic_review_manifest(
            root, reviews, REVIEWER, REVIEWED_AT
        )


def test_builder_rejects_mismatched_per_card_reviewer(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    reviews = _reviews(keys)
    reviews[0]["reviewer"] = "different-reviewer"
    with pytest.raises(ValueError, match="must match manifest reviewer"):
        semantic_audit.build_semantic_review_manifest(
            root, reviews, REVIEWER, REVIEWED_AT
        )


@pytest.mark.parametrize("reviewed_at", ["", "2026-08-31", "2026-08-31T17:42:11+01:00"])
def test_builder_requires_explicit_utc_review_time(
    semantic_project: tuple[Path, list[str]], reviewed_at: str
) -> None:
    root, keys = semantic_project
    with pytest.raises(ValueError, match="reviewed_at"):
        semantic_audit.build_semantic_review_manifest(
            root, _reviews(keys), REVIEWER, reviewed_at
        )


def test_validator_rejects_status_change_after_all_public_hashes_are_resealed(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    envelope = _build(root, keys)
    manifest = envelope["semantic_review_manifest"]
    assert isinstance(manifest, dict)
    reviews = manifest["reviews"]
    assert isinstance(reviews, list) and isinstance(reviews[0], dict)
    reviews[0]["status"] = "pending"
    _reseal(envelope)

    with pytest.raises(ValueError, match="status must be exactly 'approved'"):
        semantic_audit.validate_semantic_review_manifest(root, _save(root, envelope))


def test_validator_rejects_boolean_deletion_after_reseal(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    envelope = _build(root, keys)
    manifest = envelope["semantic_review_manifest"]
    assert isinstance(manifest, dict)
    reviews = manifest["reviews"]
    assert isinstance(reviews, list) and isinstance(reviews[0], dict)
    del reviews[0]["direction_consistent"]
    _reseal(envelope)

    with pytest.raises(ValueError, match="schema mismatch"):
        semantic_audit.validate_semantic_review_manifest(root, _save(root, envelope))


def test_validator_rejects_duplicate_review_key_after_reseal(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    envelope = _build(root, keys)
    manifest = envelope["semantic_review_manifest"]
    assert isinstance(manifest, dict)
    reviews = manifest["reviews"]
    assert isinstance(reviews, list)
    reviews[-1] = deepcopy(reviews[0])
    _reseal(envelope)

    with pytest.raises(ValueError, match="duplicate review candidate_key"):
        semantic_audit.validate_semantic_review_manifest(root, _save(root, envelope))


def test_validator_rejects_manifest_candidate_key_loss_even_if_resealed(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    envelope = _build(root, keys)
    manifest = envelope["semantic_review_manifest"]
    assert isinstance(manifest, dict)
    candidate_keys = manifest["candidate_keys"]
    assert isinstance(candidate_keys, list)
    candidate_keys.pop()
    manifest["candidate_keys_hash"] = hash_json(candidate_keys)
    _reseal(envelope)

    with pytest.raises(ValueError, match="exact frozen set"):
        semantic_audit.validate_semantic_review_manifest(root, _save(root, envelope))


def test_validator_rejects_plain_manifest_tamper(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    envelope = _build(root, keys)
    manifest = envelope["semantic_review_manifest"]
    assert isinstance(manifest, dict)
    manifest["reviewer"] = "tampered-reviewer"

    with pytest.raises(ValueError, match="aggregate hash mismatch"):
        semantic_audit.validate_semantic_review_manifest(root, _save(root, envelope))


@pytest.mark.parametrize("invalid_version", [True, 1.0, "1"])
def test_validator_requires_literal_integer_schema_version(
    semantic_project: tuple[Path, list[str]], invalid_version: object
) -> None:
    root, keys = semantic_project
    envelope = _build(root, keys)
    manifest = envelope["semantic_review_manifest"]
    assert isinstance(manifest, dict)
    manifest["schema_version"] = invalid_version
    _reseal(envelope)

    with pytest.raises(ValueError, match="unsupported semantic review schema version"):
        semantic_audit.validate_semantic_review_manifest(root, _save(root, envelope))


def test_validator_rejects_proposal_change_with_same_candidate_keys(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    envelope = _build(root, keys)
    proposal = root / semantic_audit.PROPOSAL_PATHS[0]
    proposal.write_text(
        proposal.read_text(encoding="utf-8").replace(
            "fixture mechanism 0", "materially different mechanism 0", 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen proposal hash drift"):
        semantic_audit.validate_semantic_review_manifest(root, _save(root, envelope))


def test_coordinated_proposal_and_manifest_reseal_still_hits_code_anchor(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    envelope = _build(root, keys)
    relative_path = semantic_audit.PROPOSAL_PATHS[0]
    proposal = root / relative_path
    proposal.write_text(
        proposal.read_text(encoding="utf-8").replace(
            "fixture mechanism 0", "coordinated replacement mechanism 0", 1
        ),
        encoding="utf-8",
    )

    manifest = envelope["semantic_review_manifest"]
    assert isinstance(manifest, dict)
    proposal_files = manifest["proposal_files"]
    assert isinstance(proposal_files, list) and isinstance(proposal_files[0], dict)
    proposal_files[0]["sha256"] = _file_sha(proposal)
    manifest["proposal_snapshot_hash"] = hash_json(proposal_files)
    _reseal(envelope)

    with pytest.raises(ValueError, match="frozen proposal hash drift"):
        semantic_audit.validate_semantic_review_manifest(root, _save(root, envelope))


@pytest.mark.parametrize("delta", [-1, 1])
def test_proposal_card_count_must_remain_exactly_176(
    semantic_project: tuple[Path, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    delta: int,
) -> None:
    root, keys = semantic_project
    relative_path = semantic_audit.PROPOSAL_PATHS[-1]
    path = root / relative_path
    lines = path.read_text(encoding="utf-8").splitlines()
    if delta < 0:
        lines.pop()
    else:
        lines.append(
            canonical_json(
                {
                    "candidate_key": "semantic-fixture-extra",
                    "hypothesis": "extra",
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    anchors = dict(semantic_audit.FROZEN_PROPOSAL_SHA256)
    anchors[relative_path] = _file_sha(path)
    monkeypatch.setattr(
        semantic_audit, "FROZEN_PROPOSAL_SHA256", MappingProxyType(anchors)
    )

    with pytest.raises(ValueError, match="expected 176 proposal cards"):
        semantic_audit.build_semantic_review_manifest(
            root, _reviews(keys), REVIEWER, REVIEWED_AT
        )


def test_strict_json_loader_rejects_duplicate_envelope_key(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, keys = semantic_project
    envelope = _build(root, keys)
    encoded = canonical_json(envelope)
    path = root / "duplicate-key.json"
    path.write_text(
        encoded[:-1] + ',"aggregate_hash":"' + "0" * 64 + '"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        semantic_audit.validate_semantic_review_manifest(root, path)


def test_incomplete_code_anchors_fail_closed(
    semantic_project: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, keys = semantic_project
    monkeypatch.setattr(semantic_audit, "FROZEN_PROPOSAL_SHA256", MappingProxyType({}))
    with pytest.raises(RuntimeError, match="anchors are incomplete"):
        semantic_audit.build_semantic_review_manifest(
            root, _reviews(keys), REVIEWER, REVIEWED_AT
        )


def test_proposal_hash_and_keys_come_from_one_byte_snapshot(
    semantic_project: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, keys = semantic_project
    target = root / semantic_audit.PROPOSAL_PATHS[0]
    original = target.read_text(encoding="utf-8")
    replacement = original.replace(keys[0], "semantic-raced-key", 1)
    real_open = Path.open
    raced = False

    def racing_open(self: Path, *args: object, **kwargs: object):
        nonlocal raced
        handle = real_open(self, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == target and mode == "rb" and not raced:
            raced = True
            temporary = target.with_name(target.name + ".replacement")
            temporary.write_text(replacement, encoding="utf-8")
            os.replace(temporary, target)
        return handle

    monkeypatch.setattr(Path, "open", racing_open)

    # The already-open descriptor supplies both the digest and keys, so this
    # build is internally coherent even though the path is replaced mid-read.
    envelope = semantic_audit.build_semantic_review_manifest(
        root, _reviews(keys), REVIEWER, REVIEWED_AT
    )
    assert raced

    # A new operation sees the replacement and fails the frozen code anchor.
    with pytest.raises(ValueError, match="frozen proposal hash drift"):
        semantic_audit.validate_semantic_review_manifest(
            root, _save(root, envelope, "raced.json")
        )


def test_strict_manifest_json_rejects_nonstandard_numeric_constant(
    semantic_project: tuple[Path, list[str]],
) -> None:
    root, _ = semantic_project
    path = root / "nonstandard.json"
    path.write_text(
        '{"semantic_review_manifest":NaN,"aggregate_hash":"x"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        semantic_audit.validate_semantic_review_manifest(root, path)
