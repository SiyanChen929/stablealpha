from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import cogalpha_plus.semantic_audit as semantic_audit


PROJECT = Path(__file__).resolve().parents[1]


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"duplicate JSON object key in frozen proposal: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AssertionError(f"non-standard JSON constant: {value}")


def test_real_proposal_library_matches_frozen_semantic_review_anchors() -> None:
    assert tuple(semantic_audit.FROZEN_PROPOSAL_SHA256) == semantic_audit.PROPOSAL_PATHS

    candidate_keys: list[str] = []
    for relative_path in semantic_audit.PROPOSAL_PATHS:
        payload = (PROJECT / relative_path).read_bytes()
        assert sha256(payload).hexdigest() == semantic_audit.FROZEN_PROPOSAL_SHA256[
            relative_path
        ]
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_constant,
            )
            assert isinstance(record, dict), f"{relative_path}:{line_number}"
            candidate_key = record.get("candidate_key")
            assert isinstance(candidate_key, str) and candidate_key
            candidate_keys.append(candidate_key)

    assert len(candidate_keys) == semantic_audit.EXPECTED_CANDIDATE_COUNT == 176
    assert len(set(candidate_keys)) == len(candidate_keys)
    assert frozenset(candidate_keys) == semantic_audit.FROZEN_CANDIDATE_KEYS
