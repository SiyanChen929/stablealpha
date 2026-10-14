import json
from pathlib import Path

import pytest
import yaml

from cogalpha_plus.proposals import (
    build_agent_input_contract,
    cards_to_candidates,
    load_factor_cards,
)


def card(key, expression):
    return {
        "candidate_key": key,
        "agent_id": "L4_reversal",
        "hypothesis": "价格冲击在流动性恢复后回归",
        "economic_mechanism": "短期非信息冲击被后续流动性吸收",
        "expected_direction": 1,
        "forecast_horizon": 10,
        "expected_regime": "正常流动性",
        "falsification_test": "中性化和成本后失效则否证",
        "mechanism_tags": ["reversal", "liquidity"],
        "expression": expression,
        "parameters_rationale": "5日刻画短期冲击",
    }


def test_load_factor_cards_and_deterministic_candidate_ids(tmp_path):
    path = tmp_path / "cards.jsonl"
    payload = card("rev_001", {"op": "neg", "args": [{"op": "return", "args": [{"op": "field", "name": "close"}], "periods": 5}]})
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    cards = load_factor_cards([path], allowed_agent_ids={"L4_reversal"}, expected_horizon=10, expected_count=1)
    first = cards_to_candidates(cards)[0]
    second = cards_to_candidates(cards)[0]
    assert first.candidate_id == second.candidate_id
    assert first.metadata["expected_direction"] == 1

    opposite_path = tmp_path / "opposite.jsonl"
    opposite_payload = dict(payload)
    opposite_payload["expected_direction"] = -1
    opposite_path.write_text(
        json.dumps(opposite_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    opposite = cards_to_candidates(load_factor_cards([opposite_path]))[0]
    assert opposite.expression_hash == first.expression_hash
    assert opposite.candidate_id != first.candidate_id


def test_structural_duplicates_and_future_lag_are_rejected(tmp_path):
    path = tmp_path / "cards.jsonl"
    expr = {
        "op": "add",
        "args": [
            {
                "op": "return",
                "args": [{"op": "field", "name": "close"}],
                "periods": 2,
            },
            {
                "op": "return",
                "args": [{"op": "field", "name": "open"}],
                "periods": 2,
            },
        ],
    }
    path.write_text("\n".join(json.dumps(card(key, expr), ensure_ascii=False) for key in ("a", "b")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_factor_cards([path])
    bad = tmp_path / "bad.jsonl"
    bad_expr = {"op": "lag", "args": [{"op": "field", "name": "close"}], "periods": -1}
    bad.write_text(json.dumps(card("bad", bad_expr), ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="future"):
        load_factor_cards([bad])


def test_factor_cards_enable_production_scale_and_operator_order_gates(tmp_path):
    nominal = tmp_path / "nominal.jsonl"
    nominal.write_text(
        json.dumps(
            card(
                "nominal",
                {
                    "op": "cs_rank",
                    "args": [{"op": "field", "name": "close"}],
                },
            ),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not provably invariant"):
        load_factor_cards([nominal])

    time_over_cross_section = tmp_path / "time_over_cross_section.jsonl"
    time_over_cross_section.write_text(
        json.dumps(
            card(
                "time_over_cross_section",
                {
                    "op": "ts_mean",
                    "args": [
                        {
                            "op": "cs_rank",
                            "args": [
                                {
                                    "op": "return",
                                    "args": [{"op": "field", "name": "close"}],
                                    "periods": 1,
                                }
                            ],
                        }
                    ],
                    "window": 3,
                },
            ),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="may not wrap"):
        load_factor_cards([time_over_cross_section])


@pytest.mark.parametrize(
    ("expression", "allowed_inputs", "required_source"),
    [
        (
            {"op": "cs_rank", "args": [{"op": "field", "name": "amount"}]},
            {"pit_ohlcv", "derived_price"},
            "pit_amount",
        ),
        (
            {
                "op": "return",
                "args": [{"op": "field", "name": "vwap"}],
                "periods": 1,
            },
            {"pit_ohlcv", "pit_amount"},
            "derived_price",
        ),
        (
            {
                "op": "return",
                "args": [{"op": "field", "name": "close"}],
                "periods": 1,
            },
            {"derived_price", "pit_amount"},
            "pit_ohlcv",
        ),
    ],
)
def test_factor_card_role_input_contract_fails_closed_by_field_source(
    tmp_path,
    expression,
    allowed_inputs,
    required_source,
):
    payload = card("role_contract", expression)
    payload["agent_id"] = "test_role"
    path = tmp_path / "role_contract.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=required_source):
        load_factor_cards(
            [path],
            allowed_agent_ids={"test_role"},
            agent_allowed_inputs={"test_role": allowed_inputs},
        )


def test_agent_catalog_binds_every_production_factor_field_to_a_source():
    project = Path(__file__).resolve().parents[1]
    catalog = yaml.safe_load(
        (project / "configs" / "agents.yaml").read_text(encoding="utf-8")
    )
    contract = build_agent_input_contract(catalog)
    assert len(contract) == 21
    assert "pit_amount" not in contract["L7_bar_shape"]
    assert "pit_ohlcv" in contract["L7_bar_shape"]
    assert "derived_price" in contract["L7_bar_shape"]
