from pathlib import Path

from cogalpha_plus.cli import check_configuration


PROJECT = Path(__file__).resolve().parents[1]


def test_check_configuration_binds_seven_levels_budgets_and_codex_libraries():
    result = check_configuration(PROJECT)
    assert result["agents"] == 21
    assert result["levels"] == list(range(1, 8))
    assert result["core_budget_equal"] is True
    assert result["core_budget"] == {
        "initial_population": 80,
        "parent_pool": 32,
        "offspring_per_generation": 96,
        "generations": 24,
        "elite_carryover": 2,
        "final_factor_count": 20,
    }
    assert result["factor_cards"] == {
        "generation_zero": 80,
        "precommitted_adaptive_injection": 96,
        "total_codex_authored": 176,
        "all_21_roles_covered_in_each_library": True,
        "structural_overlap_between_libraries": 0,
        "role_input_field_contract_enforced": True,
    }
    assert result["economic_semantic_review"]["candidate_count"] == 176
    assert result["economic_semantic_review"]["aggregate_hash"] == (
        "946ac53db659db97515a5add596db4a7de229dc68d988401c61cd5ae2cad5bb9"
    )
