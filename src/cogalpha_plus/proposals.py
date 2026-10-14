"""Strict interchange format between Codex research agents and Python gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dsl import canonical_json, expression_hash, validate_expression
from .evolution import Candidate, deterministic_candidate_id
from .validation import PRODUCTION_DSL_SPEC


REQUIRED_CARD_KEYS = frozenset(
    {
        "candidate_key",
        "agent_id",
        "hypothesis",
        "economic_mechanism",
        "expected_direction",
        "forecast_horizon",
        "expected_regime",
        "falsification_test",
        "mechanism_tags",
        "expression",
        "parameters_rationale",
    }
)


FACTOR_FIELD_INPUT_SOURCES: Mapping[str, str] = {
    "open": "pit_ohlcv",
    "high": "pit_ohlcv",
    "low": "pit_ohlcv",
    "close": "pit_ohlcv",
    "volume": "pit_ohlcv",
    "amount": "pit_amount",
    "vwap": "derived_price",
}


def build_agent_input_contract(
    catalog: Mapping[str, Any],
) -> dict[str, frozenset[str]]:
    """Resolve and validate the role-to-input-source contract in agents.yaml."""

    input_catalog = catalog.get("allowed_input_catalog")
    agents = catalog.get("agents")
    if not isinstance(input_catalog, Mapping) or not isinstance(agents, list):
        raise ValueError("agent catalog must define allowed_input_catalog and agents")
    for field, source in FACTOR_FIELD_INPUT_SOURCES.items():
        declaration = input_catalog.get(source)
        declared_fields = (
            declaration.get("fields") if isinstance(declaration, Mapping) else None
        )
        if (
            not isinstance(declared_fields, list)
            or field not in declared_fields
        ):
            raise ValueError(
                f"agent input catalog must map factor field {field!r} to {source!r}"
            )

    contract: dict[str, frozenset[str]] = {}
    for item in agents:
        if not isinstance(item, Mapping):
            raise ValueError("agent catalog entries must be mappings")
        agent_id = item.get("id")
        allowed_inputs = item.get("allowed_inputs")
        if (
            not isinstance(agent_id, str)
            or not agent_id
            or not isinstance(allowed_inputs, list)
            or any(not isinstance(value, str) for value in allowed_inputs)
        ):
            raise ValueError("each agent must declare id and allowed_inputs")
        if agent_id in contract:
            raise ValueError(f"duplicate agent input contract: {agent_id}")
        contract[agent_id] = frozenset(allowed_inputs)
    return contract


def validate_factor_card_input_contract(
    card: "FactorCard",
    agent_allowed_inputs: Mapping[str, Iterable[str]],
) -> None:
    """Fail closed when a card uses a field outside its role's source groups."""

    declared = agent_allowed_inputs.get(card.agent_id)
    if declared is None:
        raise ValueError(f"agent {card.agent_id!r} has no allowed_inputs contract")
    if isinstance(declared, (str, bytes)):
        raise TypeError("agent allowed_inputs must be a collection of source names")
    allowed = frozenset(declared)
    fields = validate_expression(card.expression, PRODUCTION_DSL_SPEC).fields
    violations = {
        field: FACTOR_FIELD_INPUT_SOURCES[field]
        for field in fields
        if FACTOR_FIELD_INPUT_SOURCES[field] not in allowed
    }
    if violations:
        details = ", ".join(
            f"field {field!r} requires {source!r}"
            for field, source in sorted(violations.items())
        )
        raise ValueError(
            f"agent {card.agent_id!r} input contract violation: {details}; "
            f"allowed_inputs={sorted(allowed)}"
        )


@dataclass(frozen=True)
class FactorCard:
    candidate_key: str
    agent_id: str
    hypothesis: str
    economic_mechanism: str
    expected_direction: int
    forecast_horizon: int
    expected_regime: str
    falsification_test: str
    mechanism_tags: tuple[str, ...]
    expression: Mapping[str, Any]
    parameters_rationale: str
    source_file: str = ""
    source_line: int = 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, source_file: str = "", source_line: int = 0) -> "FactorCard":
        missing = REQUIRED_CARD_KEYS.difference(payload)
        extra = set(payload).difference(REQUIRED_CARD_KEYS)
        if missing or extra:
            raise ValueError(f"factor card schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        direction = payload["expected_direction"]
        if direction not in {-1, 1}:
            raise ValueError("expected_direction must be +1 or -1")
        horizon = payload["forecast_horizon"]
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("forecast_horizon must be a positive integer")
        tags = payload["mechanism_tags"]
        if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError("mechanism_tags must be a non-empty string array")
        string_fields = (
            "candidate_key",
            "agent_id",
            "hypothesis",
            "economic_mechanism",
            "expected_regime",
            "falsification_test",
            "parameters_rationale",
        )
        for name in string_fields:
            if not isinstance(payload[name], str) or not payload[name].strip():
                raise ValueError(f"{name} must be a non-empty string")
        expression = payload["expression"]
        if not isinstance(expression, Mapping):
            raise ValueError("expression must be a JSON object")
        validate_expression(
            expression,
            PRODUCTION_DSL_SPEC,
        )
        return cls(
            candidate_key=payload["candidate_key"],
            agent_id=payload["agent_id"],
            hypothesis=payload["hypothesis"],
            economic_mechanism=payload["economic_mechanism"],
            expected_direction=direction,
            forecast_horizon=horizon,
            expected_regime=payload["expected_regime"],
            falsification_test=payload["falsification_test"],
            mechanism_tags=tuple(tags),
            expression=json.loads(canonical_json(expression, PRODUCTION_DSL_SPEC)),
            parameters_rationale=payload["parameters_rationale"],
            source_file=source_file,
            source_line=source_line,
        )

    def to_candidate(self) -> Candidate:
        canonical_expression = json.loads(
            canonical_json(self.expression, PRODUCTION_DSL_SPEC)
        )
        candidate = Candidate(
            expression=canonical_expression,
            agent_id=self.agent_id,
            hypothesis=self.hypothesis,
            mechanism_tags=self.mechanism_tags,
            metadata={
                "source": "codex_factor_card",
                "candidate_key": self.candidate_key,
                "economic_mechanism": self.economic_mechanism,
                "expected_direction": self.expected_direction,
                "forecast_horizon": self.forecast_horizon,
                "expected_regime": self.expected_regime,
                "falsification_test": self.falsification_test,
                "parameters_rationale": self.parameters_rationale,
                "source_file": self.source_file,
                "source_line": self.source_line,
            },
        )
        candidate.candidate_id = deterministic_candidate_id(candidate)
        return candidate


def load_factor_cards(
    paths: Sequence[str | Path],
    *,
    allowed_agent_ids: Iterable[str] | None = None,
    agent_allowed_inputs: Mapping[str, Iterable[str]] | None = None,
    expected_horizon: int | None = None,
    expected_count: int | None = None,
) -> list[FactorCard]:
    allowed = set(allowed_agent_ids) if allowed_agent_ids is not None else None
    cards: list[FactorCard] = []
    keys: set[str] = set()
    expressions: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    card = FactorCard.from_mapping(payload, source_file=str(path), source_line=line_number)
                except Exception as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if allowed is not None and card.agent_id not in allowed:
                    raise ValueError(f"{path}:{line_number}: unknown agent_id {card.agent_id!r}")
                if agent_allowed_inputs is not None:
                    try:
                        validate_factor_card_input_contract(
                            card,
                            agent_allowed_inputs,
                        )
                    except Exception as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if expected_horizon is not None and card.forecast_horizon != expected_horizon:
                    raise ValueError(
                        f"{path}:{line_number}: horizon {card.forecast_horizon} does not match frozen {expected_horizon}"
                    )
                if card.candidate_key in keys:
                    raise ValueError(f"duplicate candidate_key: {card.candidate_key}")
                digest = expression_hash(card.expression, PRODUCTION_DSL_SPEC)
                if digest in expressions:
                    raise ValueError(f"behavior-independent structural duplicate at {path}:{line_number}")
                keys.add(card.candidate_key)
                expressions.add(digest)
                cards.append(card)
    if expected_count is not None and len(cards) != expected_count:
        raise ValueError(f"expected {expected_count} factor cards, found {len(cards)}")
    return cards


def cards_to_candidates(cards: Sequence[FactorCard]) -> list[Candidate]:
    return [card.to_candidate() for card in cards]
