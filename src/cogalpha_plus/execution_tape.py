"""Validated full-union market tape for event-driven execution.

The alpha universe and execution tape are different objects.  A security that
leaves an index can remain in the portfolio after a suspension, limit-down or
partial fill, so the tape must continue to carry it until an explicit terminal
corporate action removes the position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


class ExecutionTapeCoverageError(ValueError):
    """The execution tape cannot value or trade a potentially held security."""


def _validate_panel_index(panel: pd.DataFrame, *, label: str) -> None:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["date", "symbol"]:
        raise ValueError(f"{label} must use MultiIndex(date, symbol)")
    if not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError(f"{label} must be sorted and unique")
    if not pd.api.types.is_datetime64_any_dtype(panel.index.get_level_values("date")):
        raise ValueError(f"{label} date level must be datetime-like")


def _membership_series(membership: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(membership, pd.DataFrame):
        _validate_panel_index(membership, label="research membership")
        if "universe_member" not in membership.columns:
            raise ValueError("research membership frame requires universe_member")
        values = membership["universe_member"]
    elif isinstance(membership, pd.Series):
        frame = membership.to_frame("universe_member")
        _validate_panel_index(frame, label="research membership")
        values = membership
    else:
        raise TypeError("research membership must be a DataFrame or Series")
    if values.isna().any() or not values.dropna().isin([True, False, 0, 1]).all():
        raise ValueError("research membership contains invalid boolean values")
    return values.astype(bool)


@dataclass(frozen=True)
class UnionExecutionTape:
    panel: pd.DataFrame
    trading_dates: pd.DatetimeIndex

    @classmethod
    def build(
        cls,
        panel: pd.DataFrame,
        *,
        research_membership: pd.DataFrame | pd.Series | None = None,
        terminal_dates: Mapping[str, pd.Timestamp | str] | None = None,
    ) -> "UnionExecutionTape":
        _validate_panel_index(panel, label="execution tape")
        dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
        if dates.empty:
            raise ValueError("execution tape is empty")
        if research_membership is not None:
            membership = _membership_series(research_membership)
            member_index = membership.index[membership.to_numpy()]
            missing_member_rows = member_index.difference(panel.index)
            if len(missing_member_rows):
                raise ExecutionTapeCoverageError(
                    f"execution tape misses {len(missing_member_rows)} research-universe rows"
                )
            terminals = {
                str(symbol): pd.Timestamp(date).normalize()
                for symbol, date in (terminal_dates or {}).items()
            }
            date_set_by_symbol = {
                str(symbol): set(pd.DatetimeIndex(group.index.get_level_values("date")))
                for symbol, group in panel.groupby(level="symbol", sort=False)
            }
            member_frame = membership[membership].rename("member").reset_index()
            for symbol, group in member_frame.groupby("symbol", sort=False):
                name = str(symbol)
                first = pd.Timestamp(group["date"].min())
                terminal = terminals.get(name)
                if terminal is not None and terminal <= first:
                    raise ValueError(f"terminal date for {name!r} does not follow first membership")
                required = dates[dates >= first]
                if terminal is not None:
                    # A terminal event is applied before that day's snapshot.
                    required = required[required < terminal]
                available = date_set_by_symbol.get(name, set())
                missing = [date for date in required if date not in available]
                if missing:
                    raise ExecutionTapeCoverageError(
                        f"execution tape for {name!r} stops before holdings are guaranteed closed; "
                        f"first missing date {missing[0].date()}"
                    )
        return cls(panel.copy(deep=False), dates)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.trading_dates

    def snapshot(
        self,
        date: pd.Timestamp | str,
        *,
        required_symbols: set[str] | None = None,
    ) -> pd.DataFrame:
        day = pd.Timestamp(date).normalize()
        if day not in self.dates:
            raise ExecutionTapeCoverageError(f"execution date {day.date()} is absent from the tape")
        snapshot = self.panel.xs(day, level="date").copy()
        snapshot.index = snapshot.index.astype(str)
        required = required_symbols or set()
        missing = sorted(set(map(str, required)).difference(snapshot.index))
        if missing:
            raise ExecutionTapeCoverageError(
                f"execution tape is missing held/target symbols on {day.date()}: {missing}"
            )
        return snapshot


__all__ = ["ExecutionTapeCoverageError", "UnionExecutionTape"]
