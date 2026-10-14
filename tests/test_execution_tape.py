import pandas as pd
import pytest

from cogalpha_plus.execution_tape import ExecutionTapeCoverageError, UnionExecutionTape


def _frame(rows):
    index = pd.MultiIndex.from_tuples(rows, names=["date", "symbol"])
    return pd.DataFrame({"close": 1.0}, index=index).sort_index()


def test_member_only_panel_is_rejected_as_execution_tape_after_index_removal():
    dates = pd.bdate_range("2024-01-02", periods=3)
    membership = _frame([(dates[0], "A"), (dates[1], "B"), (dates[2], "B")])
    membership["universe_member"] = True
    with pytest.raises(ExecutionTapeCoverageError, match="first missing date"):
        UnionExecutionTape.build(membership, research_membership=membership)


def test_full_historical_union_tape_survives_index_removal():
    dates = pd.bdate_range("2024-01-02", periods=3)
    membership = _frame([(dates[0], "A"), (dates[1], "B"), (dates[2], "B")])
    membership["universe_member"] = True
    tape = _frame([(date, symbol) for date in dates for symbol in ("A", "B")])
    result = UnionExecutionTape.build(tape, research_membership=membership)
    assert set(result.snapshot(dates[2], required_symbols={"A", "B"}).index) == {"A", "B"}


def test_explicit_terminal_event_ends_old_symbol_tape_requirement():
    dates = pd.bdate_range("2024-01-02", periods=3)
    membership = _frame([(dates[0], "A")])
    membership["universe_member"] = True
    tape = _frame([(dates[0], "A"), (dates[1], "B"), (dates[2], "B")])
    result = UnionExecutionTape.build(
        tape,
        research_membership=membership,
        terminal_dates={"A": dates[1]},
    )
    assert result.dates.equals(pd.DatetimeIndex(dates))
