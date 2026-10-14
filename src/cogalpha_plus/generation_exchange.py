"""Hash-bound file exchange for mandatory per-generation LLM review."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class FileGenerationReviewer:
    """Pause a trusted run until LLM writes the matching reviewed response.

    Requests contain only the evolution engine's scalar allowlist. Responses
    are immutable once consumed and are copied into the transcript directory.
    The engine performs the semantic/schema validation after this transport
    returns, keeping the exchange layer deliberately data-blind.
    """

    root: Path
    poll_seconds: float = 2.0
    timeout_seconds: float = 86_400.0

    def __post_init__(self) -> None:
        if self.poll_seconds <= 0.0 or self.timeout_seconds <= 0.0:
            raise ValueError("generation exchange timing must be positive")
        root = Path(self.root).resolve()
        object.__setattr__(self, "root", root)
        for child in ("requests", "responses", "transcripts"):
            (root / child).mkdir(parents=True, exist_ok=True)

    def _path(self, child: str, generation: int) -> Path:
        return self.root / child / f"generation_{generation:02d}.json"

    @staticmethod
    def _write_exclusive(path: Path, text: str) -> None:
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(text)
                handle.write("\n")
        except FileExistsError:
            existing = path.read_text(encoding="utf-8").strip()
            if existing != text:
                raise RuntimeError(f"immutable generation exchange conflict: {path}")

    def __call__(
        self,
        generation: int,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        request_path = self._path("requests", generation)
        response_path = self._path("responses", generation)
        transcript_path = self._path("transcripts", generation)
        request_text = _canonical_json(request)
        self._write_exclusive(request_path, request_text)

        deadline = time.monotonic() + self.timeout_seconds
        while not response_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"LLM generation review timed out: {response_path}"
                )
            time.sleep(self.poll_seconds)
        raw_response = response_path.read_text(encoding="utf-8")
        try:
            response = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM generation response is not valid JSON: {response_path}"
            ) from exc
        if not isinstance(response, Mapping):
            raise ValueError("LLM generation response must be a JSON object")
        transcript = {
            "schema": "codex_generation_transcript_v1",
            "generation": generation,
            "request": dict(request),
            "response": dict(response),
        }
        self._write_exclusive(transcript_path, _canonical_json(transcript))
        return response


__all__ = ["FileGenerationReviewer"]
