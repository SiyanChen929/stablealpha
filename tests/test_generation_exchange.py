import json

import pytest

from cogalpha_plus.generation_exchange import FileGenerationReviewer


def test_file_generation_exchange_is_hash_bound_and_immutable(tmp_path):
    root = tmp_path / "exchange"
    reviewer = FileGenerationReviewer(root, poll_seconds=0.001, timeout_seconds=1.0)
    request = {
        "schema": "codex_generation_request_v1",
        "generation": 1,
        "request_sha256": "a" * 64,
    }
    response = {
        "schema": "codex_generation_response_v1",
        "generation": 1,
        "request_sha256": "a" * 64,
        "reviewer_model": "codex-test",
        "decisions": [],
        "sealed_feedback_consumed": False,
    }
    response_path = root / "responses" / "generation_01.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    assert reviewer(1, request) == response
    assert (root / "requests" / "generation_01.json").is_file()
    transcript = json.loads(
        (root / "transcripts" / "generation_01.json").read_text()
    )
    assert transcript["request"] == request
    assert transcript["response"] == response

    changed = {**request, "request_sha256": "b" * 64}
    with pytest.raises(RuntimeError, match="immutable generation exchange conflict"):
        reviewer(1, changed)


def test_file_generation_exchange_times_out_without_response(tmp_path):
    reviewer = FileGenerationReviewer(
        tmp_path / "exchange", poll_seconds=0.001, timeout_seconds=0.005
    )
    with pytest.raises(TimeoutError, match="timed out"):
        reviewer(
            2,
            {
                "schema": "codex_generation_request_v1",
                "generation": 2,
                "request_sha256": "c" * 64,
            },
        )
