"""Fail-closed point-in-time contracts for a US equity index universe.

This module contains no downloader and makes no claim about a particular data
vendor.  It defines the minimum deterministic objects a future S&P 500 adapter
must supply before factor discovery can read market data:

* permanent security-level identities with dated ticker aliases;
* announced, known and effective membership events;
* explicit multi-leg corporate actions, including cash in lieu;
* separate cross-sectional and execution-tape coverage checks.

Ticker strings are display aliases, never join keys.  A security that leaves
the index remains on the execution tape until an explicit terminal event.  The
daily event order is a frozen public constant because changing it changes
dividend entitlement, merger accounting and simulated fills.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


US_MARKET_TIMEZONE = "America/New_York"

# Opening corporate actions precede opening-effective index changes.  Orders
# use the raw cash VWAP, and the close is valued only after both earlier steps.
# For example, a position held overnight receives an ex-date entitlement
# before a same-day VWAP sale; a position first bought on the ex-date does not.
US_INDEX_EVENT_DAY_ORDER_V1: tuple[str, ...] = (
    "opening_corporate_actions",
    "effective_membership_and_benchmark",
    "raw_vwap_execution",
    "close_valuation",
)
US_INDEX_EVENT_DAY_ORDER_CONTRACT_V1: Mapping[str, object] = MappingProxyType(
    {
        "schema_version": 1,
        "market_timezone": US_MARKET_TIMEZONE,
        "steps": US_INDEX_EVENT_DAY_ORDER_V1,
    }
)
US_INDEX_EVENT_DAY_ORDER_SHA256 = hashlib.sha256(
    json.dumps(
        dict(US_INDEX_EVENT_DAY_ORDER_CONTRACT_V1),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class USIndexContractError(ValueError):
    """Base class for a fail-closed US index contract violation."""


class UnknownSecurityIdentifierError(USIndexContractError):
    """A ticker or permanent security identity is absent or ambiguous."""


class MembershipReplayError(USIndexContractError):
    """Membership events cannot be replayed deterministically."""


class USCorporateActionContractError(USIndexContractError):
    """Corporate-action terms are incomplete or internally inconsistent."""


class USIndexTapeCoverageError(USIndexContractError):
    """Research or execution tapes do not cover the required universe."""


def _nonblank(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise USIndexContractError(f"{name} must be non-blank")
    return text


def _aware_utc(value: object, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - pandas exception details vary.
        raise USIndexContractError(f"{name} is not a valid timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise USIndexContractError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _session_date(value: object, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - pandas exception details vary.
        raise USIndexContractError(f"{name} is not a valid session date") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is not None:
        raise USIndexContractError(f"{name} must be a timezone-naive session date")
    normalized = timestamp.normalize()
    if timestamp != normalized:
        raise USIndexContractError(f"{name} must not contain a time of day")
    return normalized


def _us_session_date(value: object, *, name: str) -> pd.Timestamp:
    timestamp = _aware_utc(value, name=name)
    local = timestamp.tz_convert(ZoneInfo(US_MARKET_TIMEZONE))
    return pd.Timestamp(local.date())


def _positive_finite(
    value: object,
    *,
    name: str,
    error_type: type[USIndexContractError] = USIndexContractError,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise error_type(f"{name} must be positive and finite") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise error_type(f"{name} must be positive and finite")
    return numeric


def _strict_bool_series(values: pd.Series, *, name: str) -> pd.Series:
    if values.isna().any():
        raise USIndexContractError(f"{name} contains missing values")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise USIndexContractError(f"{name} must contain only boolean or 0/1 values")
        return numeric.astype(bool)
    raise USIndexContractError(f"{name} must use a boolean or numeric 0/1 dtype")


def _validate_panel_index(index: pd.Index, *, name: str) -> pd.MultiIndex:
    if not isinstance(index, pd.MultiIndex) or tuple(index.names) != (
        "date",
        "security_id",
    ):
        raise USIndexContractError(
            f"{name} must use MultiIndex(date, security_id)"
        )
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise USIndexContractError(f"{name} must be sorted and unique")
    dates = pd.DatetimeIndex(index.get_level_values("date"))
    if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()):
        raise USIndexContractError(
            f"{name} dates must be valid timezone-naive session dates"
        )
    security_ids = pd.Index(index.get_level_values("security_id").astype(str))
    if (security_ids.str.strip() == "").any():
        raise USIndexContractError(f"{name} contains a blank security_id")
    return index


@dataclass(frozen=True)
class PermanentSecurity:
    """One provider-guaranteed permanent security-level identity.

    ``security_id`` identifies a listed share class, not merely an issuer.  An
    issuer identifier such as CIK therefore cannot replace it for companies
    with multiple index-eligible share classes.
    """

    security_id: str
    id_scheme: str
    source: str
    issuer_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "security_id", _nonblank(self.security_id, name="security_id")
        )
        object.__setattr__(
            self, "id_scheme", _nonblank(self.id_scheme, name="id_scheme")
        )
        object.__setattr__(self, "source", _nonblank(self.source, name="source"))
        if self.issuer_id is not None:
            object.__setattr__(
                self, "issuer_id", _nonblank(self.issuer_id, name="issuer_id")
            )


@dataclass(frozen=True)
class TickerAlias:
    """Canonical exchange ticker valid on ``[valid_from, valid_to_exclusive)``."""

    security_id: str
    ticker: str
    valid_from: pd.Timestamp | str
    valid_to_exclusive: pd.Timestamp | str | None
    known_at: pd.Timestamp | str
    source: str

    def __post_init__(self) -> None:
        security_id = _nonblank(self.security_id, name="security_id")
        ticker = _nonblank(self.ticker, name="ticker").upper()
        if any(character.isspace() for character in ticker):
            raise USIndexContractError("ticker must not contain whitespace")
        start = _session_date(self.valid_from, name="valid_from")
        end = (
            _session_date(self.valid_to_exclusive, name="valid_to_exclusive")
            if self.valid_to_exclusive is not None
            else None
        )
        if end is not None and end <= start:
            raise USIndexContractError("ticker alias interval must have positive length")
        known = _aware_utc(self.known_at, name="known_at")
        if _us_session_date(known, name="known_at") > start:
            raise USIndexContractError(
                "ticker alias was not known by its first valid session"
            )
        object.__setattr__(self, "security_id", security_id)
        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "valid_from", start)
        object.__setattr__(self, "valid_to_exclusive", end)
        object.__setattr__(self, "known_at", known)
        object.__setattr__(self, "source", _nonblank(self.source, name="source"))

    def active_on(self, session_date: pd.Timestamp | str) -> bool:
        day = _session_date(session_date, name="session_date")
        return day >= self.valid_from and (
            self.valid_to_exclusive is None or day < self.valid_to_exclusive
        )


class SecurityIdentityCatalog:
    """Validated permanent identities and non-overlapping dated aliases."""

    def __init__(
        self,
        securities: Iterable[PermanentSecurity],
        ticker_aliases: Iterable[TickerAlias],
    ) -> None:
        ordered_securities = tuple(sorted(securities, key=lambda item: item.security_id))
        if not ordered_securities:
            raise USIndexContractError("security catalog cannot be empty")
        ids = [item.security_id for item in ordered_securities]
        if len(ids) != len(set(ids)):
            raise USIndexContractError("security_id values must be unique")
        aliases = tuple(
            sorted(
                ticker_aliases,
                key=lambda item: (
                    item.valid_from,
                    item.ticker,
                    item.security_id,
                ),
            )
        )
        if not aliases:
            raise USIndexContractError("ticker alias catalog cannot be empty")
        unknown = sorted({item.security_id for item in aliases}.difference(ids))
        if unknown:
            raise UnknownSecurityIdentifierError(
                "ticker aliases reference unknown security_id values: "
                + ",".join(unknown)
            )
        self._validate_nonoverlap(aliases, key="ticker")
        self._validate_nonoverlap(aliases, key="security_id")
        self.securities = ordered_securities
        self.ticker_aliases = aliases
        self._security_ids = frozenset(ids)

    @staticmethod
    def _validate_nonoverlap(aliases: Sequence[TickerAlias], *, key: str) -> None:
        groups: dict[str, list[TickerAlias]] = {}
        for alias in aliases:
            groups.setdefault(str(getattr(alias, key)), []).append(alias)
        for value, group in groups.items():
            ordered = sorted(group, key=lambda item: item.valid_from)
            prior: TickerAlias | None = None
            for alias in ordered:
                if prior is not None and (
                    prior.valid_to_exclusive is None
                    or alias.valid_from < prior.valid_to_exclusive
                ):
                    raise USIndexContractError(
                        f"overlapping ticker alias intervals for {key}={value!r}"
                    )
                prior = alias

    @property
    def security_ids(self) -> frozenset[str]:
        return self._security_ids

    def require_security(self, security_id: object) -> str:
        value = _nonblank(security_id, name="security_id")
        if value not in self._security_ids:
            raise UnknownSecurityIdentifierError(
                f"unknown permanent security_id: {value!r}"
            )
        return value

    def resolve_ticker(
        self,
        ticker: object,
        session_date: pd.Timestamp | str,
        *,
        decision_at: pd.Timestamp | str,
    ) -> str:
        symbol = _nonblank(ticker, name="ticker").upper()
        decision = _aware_utc(decision_at, name="decision_at")
        candidates = [
            alias
            for alias in self.ticker_aliases
            if alias.ticker == symbol
            and alias.active_on(session_date)
            and alias.known_at <= decision
        ]
        if len(candidates) != 1:
            raise UnknownSecurityIdentifierError(
                f"ticker {symbol!r} resolves to {len(candidates)} securities"
            )
        return candidates[0].security_id

    def ticker_for(
        self,
        security_id: object,
        session_date: pd.Timestamp | str,
        *,
        decision_at: pd.Timestamp | str,
    ) -> str:
        identity = self.require_security(security_id)
        decision = _aware_utc(decision_at, name="decision_at")
        candidates = [
            alias
            for alias in self.ticker_aliases
            if alias.security_id == identity
            and alias.active_on(session_date)
            and alias.known_at <= decision
        ]
        if len(candidates) != 1:
            raise UnknownSecurityIdentifierError(
                f"security_id {identity!r} has {len(candidates)} active tickers"
            )
        return candidates[0].ticker


class MembershipAction(str, Enum):
    ADD = "ADD"
    REMOVE = "REMOVE"


@dataclass(frozen=True)
class MembershipSeed:
    """Audited constituent snapshot from which membership events are replayed."""

    index_id: str
    member_security_ids: tuple[str, ...]
    known_at: pd.Timestamp | str
    effective_at: pd.Timestamp | str
    source: str

    def __post_init__(self) -> None:
        index_id = _nonblank(self.index_id, name="index_id")
        members = tuple(
            _nonblank(value, name="member_security_id")
            for value in self.member_security_ids
        )
        if not members or len(members) != len(set(members)):
            raise MembershipReplayError(
                "membership seed must contain unique, non-empty members"
            )
        known = _aware_utc(self.known_at, name="known_at")
        effective = _aware_utc(self.effective_at, name="effective_at")
        if known > effective:
            raise MembershipReplayError(
                "membership seed must be known no later than its effective time"
            )
        object.__setattr__(self, "index_id", index_id)
        object.__setattr__(self, "member_security_ids", tuple(sorted(members)))
        object.__setattr__(self, "known_at", known)
        object.__setattr__(self, "effective_at", effective)
        object.__setattr__(self, "source", _nonblank(self.source, name="source"))


@dataclass(frozen=True)
class MembershipEvent:
    """One announced and effective ADD/REMOVE instruction."""

    event_id: str
    index_id: str
    security_id: str
    action: MembershipAction | str
    announcement_at: pd.Timestamp | str
    known_at: pd.Timestamp | str
    effective_at: pd.Timestamp | str
    source: str
    reason: str = ""

    def __post_init__(self) -> None:
        try:
            action = (
                self.action
                if isinstance(self.action, MembershipAction)
                else MembershipAction(str(self.action).upper())
            )
        except ValueError as exc:
            raise MembershipReplayError(
                f"unsupported membership action: {self.action!r}"
            ) from exc
        announcement = _aware_utc(self.announcement_at, name="announcement_at")
        known = _aware_utc(self.known_at, name="known_at")
        effective = _aware_utc(self.effective_at, name="effective_at")
        if not announcement <= known <= effective:
            raise MembershipReplayError(
                "membership event requires announcement_at <= known_at <= effective_at"
            )
        object.__setattr__(self, "event_id", _nonblank(self.event_id, name="event_id"))
        object.__setattr__(self, "index_id", _nonblank(self.index_id, name="index_id"))
        object.__setattr__(
            self, "security_id", _nonblank(self.security_id, name="security_id")
        )
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "announcement_at", announcement)
        object.__setattr__(self, "known_at", known)
        object.__setattr__(self, "effective_at", effective)
        object.__setattr__(self, "source", _nonblank(self.source, name="source"))
        object.__setattr__(self, "reason", str(self.reason).strip())


class USIndexMembershipTimeline:
    """Replay an initial snapshot and immutable membership events."""

    def __init__(
        self,
        identities: SecurityIdentityCatalog,
        seed: MembershipSeed,
        events: Iterable[MembershipEvent],
    ) -> None:
        for security_id in seed.member_security_ids:
            identities.require_security(security_id)
        ordered = tuple(
            sorted(
                events,
                key=lambda item: (
                    item.effective_at,
                    0 if item.action is MembershipAction.REMOVE else 1,
                    item.security_id,
                    item.event_id,
                ),
            )
        )
        ids = [event.event_id for event in ordered]
        if len(ids) != len(set(ids)):
            raise MembershipReplayError("membership event_id values must be unique")
        event_keys = [(event.effective_at, event.security_id) for event in ordered]
        if len(event_keys) != len(set(event_keys)):
            raise MembershipReplayError(
                "a security cannot have multiple membership actions at one effective time"
            )
        active = set(seed.member_security_ids)
        for event in ordered:
            if event.index_id != seed.index_id:
                raise MembershipReplayError("membership events mix index_id values")
            identities.require_security(event.security_id)
            if event.effective_at <= seed.effective_at:
                raise MembershipReplayError(
                    "membership event must follow the seed effective time"
                )
            if event.action is MembershipAction.ADD:
                if event.security_id in active:
                    raise MembershipReplayError(
                        f"ADD repeats active member {event.security_id!r}"
                    )
                active.add(event.security_id)
            else:
                if event.security_id not in active:
                    raise MembershipReplayError(
                        f"REMOVE references inactive member {event.security_id!r}"
                    )
                active.remove(event.security_id)
        self.identities = identities
        self.seed = seed
        self.events = ordered

    @property
    def historical_security_ids(self) -> tuple[str, ...]:
        values = set(self.seed.member_security_ids)
        values.update(event.security_id for event in self.events)
        return tuple(sorted(values))

    def members_at(self, decision_at: pd.Timestamp | str) -> frozenset[str]:
        decision = _aware_utc(decision_at, name="decision_at")
        if decision < self.seed.effective_at or decision < self.seed.known_at:
            raise MembershipReplayError(
                "decision time precedes the membership seed"
            )
        active = set(self.seed.member_security_ids)
        for event in self.events:
            if event.effective_at > decision:
                break
            if event.known_at > decision:  # Defense in depth if construction changes.
                raise MembershipReplayError(
                    f"effective event {event.event_id!r} was not known at decision time"
                )
            if event.action is MembershipAction.ADD:
                active.add(event.security_id)
            else:
                active.remove(event.security_id)
        return frozenset(active)

    def expand_membership(
        self,
        session_decision_times: Mapping[pd.Timestamp | str, pd.Timestamp | str],
        *,
        security_ids: Sequence[str] | None = None,
    ) -> pd.Series:
        """Return a complete date x security mask at the supplied decision times."""

        normalized: dict[pd.Timestamp, pd.Timestamp] = {}
        for raw_date, raw_decision in session_decision_times.items():
            date = _session_date(raw_date, name="session_decision_date")
            if date in normalized:
                raise MembershipReplayError("duplicate session decision date")
            decision = _aware_utc(raw_decision, name="session_decision_time")
            if _us_session_date(decision, name="session_decision_time") != date:
                raise MembershipReplayError(
                    "session decision time falls on a different US market date"
                )
            normalized[date] = decision
        dates = pd.DatetimeIndex(sorted(normalized), name="date")
        if dates.empty:
            raise MembershipReplayError("session decision calendar cannot be empty")
        ordered_decisions = [normalized[date] for date in dates]
        if any(
            right <= left
            for left, right in zip(
                ordered_decisions, ordered_decisions[1:], strict=False
            )
        ):
            raise MembershipReplayError(
                "session decision times must be strictly chronological"
            )
        selected = tuple(
            sorted(
                _nonblank(value, name="membership security_id")
                for value in (
                    self.historical_security_ids
                    if security_ids is None
                    else security_ids
                )
            )
        )
        if not selected or len(selected) != len(set(selected)):
            raise MembershipReplayError("membership expansion security_ids are invalid")
        for security_id in selected:
            self.identities.require_security(security_id)
        missing_history = sorted(set(self.historical_security_ids).difference(selected))
        if missing_history:
            raise MembershipReplayError(
                "membership expansion omits historical securities: "
                + ",".join(missing_history)
            )
        index = pd.MultiIndex.from_product(
            [dates, selected], names=["date", "security_id"]
        )
        values = np.zeros(len(index), dtype=bool)
        width = len(selected)
        selected_index = {security_id: offset for offset, security_id in enumerate(selected)}
        for date_position, date in enumerate(dates):
            for security_id in self.members_at(normalized[date]):
                values[date_position * width + selected_index[security_id]] = True
        return pd.Series(values, index=index, name="universe_member", dtype=bool)


class USCorporateActionKind(str, Enum):
    CASH_DISTRIBUTION = "cash_distribution"
    STOCK_DISTRIBUTION = "stock_distribution"
    MERGER = "merger"
    SPINOFF = "spinoff"
    SHARE_CLASS_CONVERSION = "share_class_conversion"


class USCorporateActionLegKind(str, Enum):
    CASH = "cash"
    SECURITY = "security"
    FRACTIONAL_CASH_IN_LIEU = "fractional_cash_in_lieu"


class FractionalTreatment(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    DELIVER_FRACTIONAL = "deliver_fractional"
    CASH_IN_LIEU = "cash_in_lieu"


@dataclass(frozen=True)
class USCorporateActionLeg:
    """One fully specified cash or security leg of a corporate action."""

    leg_id: str
    kind: USCorporateActionLegKind | str
    settlement_at: pd.Timestamp | str
    known_at: pd.Timestamp | str
    source: str
    cash_per_source_share: float | None = None
    destination_security_id: str | None = None
    shares_per_source_share: float | None = None
    currency: str | None = None
    fractional_treatment: FractionalTreatment | str = FractionalTreatment.NOT_APPLICABLE
    applies_to_leg_id: str | None = None
    cash_per_fractional_destination_share: float | None = None

    def __post_init__(self) -> None:
        try:
            kind = USCorporateActionLegKind(self.kind)
        except ValueError as exc:
            raise USCorporateActionContractError(
                f"unsupported corporate-action leg kind: {self.kind!r}"
            ) from exc
        try:
            fractional = FractionalTreatment(self.fractional_treatment)
        except ValueError as exc:
            raise USCorporateActionContractError(
                f"unsupported fractional treatment: {self.fractional_treatment!r}"
            ) from exc
        settlement = _aware_utc(self.settlement_at, name="settlement_at")
        known = _aware_utc(self.known_at, name="known_at")
        if known > settlement:
            raise USCorporateActionContractError(
                "corporate-action leg must be known by settlement"
            )
        cash = (
            _positive_finite(
                self.cash_per_source_share,
                name="cash_per_source_share",
                error_type=USCorporateActionContractError,
            )
            if self.cash_per_source_share is not None
            else None
        )
        ratio = (
            _positive_finite(
                self.shares_per_source_share,
                name="shares_per_source_share",
                error_type=USCorporateActionContractError,
            )
            if self.shares_per_source_share is not None
            else None
        )
        fractional_cash = (
            _positive_finite(
                self.cash_per_fractional_destination_share,
                name="cash_per_fractional_destination_share",
                error_type=USCorporateActionContractError,
            )
            if self.cash_per_fractional_destination_share is not None
            else None
        )
        destination = (
            _nonblank(self.destination_security_id, name="destination_security_id")
            if self.destination_security_id is not None
            else None
        )
        currency = str(self.currency or "").strip().upper() or None
        applies = (
            _nonblank(self.applies_to_leg_id, name="applies_to_leg_id")
            if self.applies_to_leg_id is not None
            else None
        )
        if currency is not None and (len(currency) != 3 or not currency.isalpha()):
            raise USCorporateActionContractError(
                "currency must be a three-letter alphabetic code"
            )

        if kind is USCorporateActionLegKind.CASH:
            if (
                cash is None
                or currency is None
                or destination is not None
                or ratio is not None
                or applies is not None
                or fractional_cash is not None
                or fractional is not FractionalTreatment.NOT_APPLICABLE
            ):
                raise USCorporateActionContractError(
                    "cash leg requires only cash_per_source_share and currency"
                )
        elif kind is USCorporateActionLegKind.SECURITY:
            if (
                destination is None
                or ratio is None
                or cash is not None
                or currency is not None
                or applies is not None
                or fractional_cash is not None
                or fractional is FractionalTreatment.NOT_APPLICABLE
            ):
                raise USCorporateActionContractError(
                    "security leg requires destination, share ratio and explicit fractional treatment"
                )
        else:
            if (
                destination is None
                or applies is None
                or fractional_cash is None
                or currency is None
                or cash is not None
                or ratio is not None
                or fractional is not FractionalTreatment.NOT_APPLICABLE
            ):
                raise USCorporateActionContractError(
                    "cash-in-lieu leg requires target leg, destination, cash price and currency"
                )

        object.__setattr__(self, "leg_id", _nonblank(self.leg_id, name="leg_id"))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "settlement_at", settlement)
        object.__setattr__(self, "known_at", known)
        object.__setattr__(self, "source", _nonblank(self.source, name="source"))
        object.__setattr__(self, "cash_per_source_share", cash)
        object.__setattr__(self, "destination_security_id", destination)
        object.__setattr__(self, "shares_per_source_share", ratio)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "fractional_treatment", fractional)
        object.__setattr__(self, "applies_to_leg_id", applies)
        object.__setattr__(
            self, "cash_per_fractional_destination_share", fractional_cash
        )


@dataclass(frozen=True)
class USCorporateActionEntitlement:
    cash_by_currency: Mapping[str, float]
    security_quantities: Mapping[str, float]
    source_security_terminates: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cash_by_currency", MappingProxyType(dict(self.cash_by_currency))
        )
        object.__setattr__(
            self,
            "security_quantities",
            MappingProxyType(dict(self.security_quantities)),
        )


@dataclass(frozen=True)
class USCorporateActionEvent:
    """A dated event with all economic legs needed for deterministic replay."""

    event_id: str
    kind: USCorporateActionKind | str
    source_security_id: str
    announcement_at: pd.Timestamp | str
    known_at: pd.Timestamp | str
    effective_at: pd.Timestamp | str
    source_security_terminates: bool
    source: str
    legs: tuple[USCorporateActionLeg, ...]
    effective_phase: str = "opening"

    def __post_init__(self) -> None:
        try:
            kind = USCorporateActionKind(self.kind)
        except ValueError as exc:
            raise USCorporateActionContractError(
                f"unsupported corporate-action kind: {self.kind!r}"
            ) from exc
        announcement = _aware_utc(self.announcement_at, name="announcement_at")
        known = _aware_utc(self.known_at, name="known_at")
        effective = _aware_utc(self.effective_at, name="effective_at")
        if not announcement <= known <= effective:
            raise USCorporateActionContractError(
                "corporate action requires announcement_at <= known_at <= effective_at"
            )
        if not isinstance(self.source_security_terminates, bool):
            raise USCorporateActionContractError(
                "source_security_terminates must be a strict boolean"
            )
        if str(self.effective_phase).strip().lower() != "opening":
            raise USCorporateActionContractError(
                "corporate actions must be normalized to the opening event phase"
            )
        terminal_kind = kind in {
            USCorporateActionKind.MERGER,
            USCorporateActionKind.SHARE_CLASS_CONVERSION,
        }
        if self.source_security_terminates is not terminal_kind:
            raise USCorporateActionContractError(
                f"source termination is inconsistent with {kind.value}"
            )
        legs = tuple(self.legs)
        if not legs:
            raise USCorporateActionContractError("corporate action cannot have no legs")
        leg_ids = [leg.leg_id for leg in legs]
        if len(leg_ids) != len(set(leg_ids)):
            raise USCorporateActionContractError("corporate-action leg_id values must be unique")
        for leg in legs:
            if leg.settlement_at < effective:
                raise USCorporateActionContractError(
                    f"leg {leg.leg_id!r} settles before effective_at"
                )
            if leg.known_at < announcement:
                raise USCorporateActionContractError(
                    f"leg {leg.leg_id!r} was known before the event announcement"
                )

        cash_legs = [leg for leg in legs if leg.kind is USCorporateActionLegKind.CASH]
        security_legs = [
            leg for leg in legs if leg.kind is USCorporateActionLegKind.SECURITY
        ]
        cil_legs = [
            leg
            for leg in legs
            if leg.kind is USCorporateActionLegKind.FRACTIONAL_CASH_IN_LIEU
        ]
        if kind is USCorporateActionKind.CASH_DISTRIBUTION:
            if not cash_legs or security_legs or cil_legs:
                raise USCorporateActionContractError(
                    "cash distribution requires cash legs only"
                )
        elif kind in {
            USCorporateActionKind.STOCK_DISTRIBUTION,
            USCorporateActionKind.SPINOFF,
        }:
            if not security_legs or cash_legs:
                raise USCorporateActionContractError(
                    f"{kind.value} requires security and optional cash-in-lieu legs"
                )
        elif kind is USCorporateActionKind.MERGER:
            if not cash_legs and not security_legs:
                raise USCorporateActionContractError(
                    "merger requires cash, stock, or mixed consideration"
                )
        elif kind is USCorporateActionKind.SHARE_CLASS_CONVERSION:
            if len(security_legs) != 1 or cash_legs:
                raise USCorporateActionContractError(
                    "share-class conversion requires one security leg"
                )

        security_by_id = {leg.leg_id: leg for leg in security_legs}
        cil_by_target: dict[str, USCorporateActionLeg] = {}
        for leg in cil_legs:
            target = security_by_id.get(str(leg.applies_to_leg_id))
            if target is None:
                raise USCorporateActionContractError(
                    f"cash-in-lieu leg {leg.leg_id!r} references no security leg"
                )
            if target.destination_security_id != leg.destination_security_id:
                raise USCorporateActionContractError(
                    f"cash-in-lieu leg {leg.leg_id!r} destination differs from target"
                )
            if target.leg_id in cil_by_target:
                raise USCorporateActionContractError(
                    f"security leg {target.leg_id!r} has multiple cash-in-lieu legs"
                )
            cil_by_target[target.leg_id] = leg
        for leg in security_legs:
            has_cil = leg.leg_id in cil_by_target
            expects_cil = leg.fractional_treatment is FractionalTreatment.CASH_IN_LIEU
            if has_cil != expects_cil:
                raise USCorporateActionContractError(
                    f"security leg {leg.leg_id!r} fractional terms are incomplete"
                )

        source_security_id = _nonblank(
            self.source_security_id, name="source_security_id"
        )
        if kind in {
            USCorporateActionKind.SPINOFF,
            USCorporateActionKind.MERGER,
            USCorporateActionKind.SHARE_CLASS_CONVERSION,
        } and any(
            leg.destination_security_id == source_security_id for leg in security_legs
        ):
            raise USCorporateActionContractError(
                f"{kind.value} destination must differ from source security"
            )
        object.__setattr__(self, "event_id", _nonblank(self.event_id, name="event_id"))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_security_id", source_security_id)
        object.__setattr__(self, "announcement_at", announcement)
        object.__setattr__(self, "known_at", known)
        object.__setattr__(self, "effective_at", effective)
        object.__setattr__(self, "source", _nonblank(self.source, name="source"))
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "effective_phase", "opening")

    def entitlements(self, source_shares: int) -> USCorporateActionEntitlement:
        """Calculate final cash and security entitlements for integer source shares."""

        try:
            quantity = int(source_shares)
        except (TypeError, ValueError, OverflowError) as exc:
            raise USCorporateActionContractError(
                "source_shares must be a non-negative integer"
            ) from exc
        if isinstance(source_shares, bool) or quantity != source_shares or quantity < 0:
            raise USCorporateActionContractError(
                "source_shares must be a non-negative integer"
            )
        cash: dict[str, float] = {}
        securities: dict[str, float] = {}
        cil_by_target = {
            str(leg.applies_to_leg_id): leg
            for leg in self.legs
            if leg.kind is USCorporateActionLegKind.FRACTIONAL_CASH_IN_LIEU
        }
        for leg in self.legs:
            if leg.kind is USCorporateActionLegKind.CASH:
                currency = str(leg.currency)
                cash[currency] = cash.get(currency, 0.0) + quantity * float(
                    leg.cash_per_source_share
                )
            elif leg.kind is USCorporateActionLegKind.SECURITY:
                exact = quantity * float(leg.shares_per_source_share)
                destination = str(leg.destination_security_id)
                if leg.fractional_treatment is FractionalTreatment.CASH_IN_LIEU:
                    whole = math.floor(exact + 1e-12)
                    fractional = max(exact - whole, 0.0)
                    securities[destination] = securities.get(destination, 0.0) + whole
                    cil = cil_by_target[leg.leg_id]
                    currency = str(cil.currency)
                    cash[currency] = cash.get(currency, 0.0) + fractional * float(
                        cil.cash_per_fractional_destination_share
                    )
                else:
                    securities[destination] = securities.get(destination, 0.0) + exact
        return USCorporateActionEntitlement(
            cash_by_currency={
                key: float(value) for key, value in sorted(cash.items())
            },
            security_quantities={
                key: float(value) for key, value in sorted(securities.items())
            },
            source_security_terminates=self.source_security_terminates,
        )


class USCorporateActionCatalog:
    """Identity-checked event catalog with unique terminal paths."""

    def __init__(
        self,
        identities: SecurityIdentityCatalog,
        events: Iterable[USCorporateActionEvent],
    ) -> None:
        ordered = tuple(
            sorted(
                events,
                key=lambda item: (
                    item.effective_at,
                    item.source_security_id,
                    item.event_id,
                ),
            )
        )
        ids = [event.event_id for event in ordered]
        if len(ids) != len(set(ids)):
            raise USCorporateActionContractError(
                "corporate-action event_id values must be unique"
            )
        terminal_sources: set[str] = set()
        for event in ordered:
            identities.require_security(event.source_security_id)
            for leg in event.legs:
                if leg.destination_security_id is not None:
                    identities.require_security(leg.destination_security_id)
            if event.source_security_terminates:
                if event.source_security_id in terminal_sources:
                    raise USCorporateActionContractError(
                        "a source security cannot have multiple terminal events"
                    )
                terminal_sources.add(event.source_security_id)
        self.identities = identities
        self.events = ordered

    def terminal_session_dates(self) -> dict[str, pd.Timestamp]:
        return {
            event.source_security_id: _us_session_date(
                event.effective_at, name="effective_at"
            )
            for event in self.events
            if event.source_security_terminates
        }


@dataclass(frozen=True)
class USIndexUniverseValidation:
    trading_sessions: int
    security_ids: int
    member_rows: int
    cross_sectional_rows: int
    execution_required_rows: int
    new_entrant_security_ids: tuple[str, ...]
    removed_security_ids: tuple[str, ...]
    terminal_security_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_sessions(values: Sequence[pd.Timestamp | str]) -> pd.DatetimeIndex:
    sessions = pd.DatetimeIndex(
        [_session_date(value, name="trading_session") for value in values],
        name="date",
    )
    if sessions.empty or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise USIndexContractError(
            "trading sessions must be non-empty, unique and strictly increasing"
        )
    return sessions


def validate_us_index_universe_views(
    identities: SecurityIdentityCatalog,
    membership_timeline: USIndexMembershipTimeline,
    membership: pd.Series,
    cross_sectional_mask: pd.Series,
    feature_tape: pd.DataFrame,
    execution_tape: pd.DataFrame,
    *,
    trading_sessions: Sequence[pd.Timestamp | str],
    session_decision_times: Mapping[
        pd.Timestamp | str, pd.Timestamp | str
    ],
    corporate_actions: USCorporateActionCatalog | None = None,
) -> USIndexUniverseValidation:
    """Validate entry masking and post-removal execution retention.

    ``membership`` and ``cross_sectional_mask`` must be a complete rectangular
    date x historical-security domain.  A new entrant may have pre-entry bars
    for rolling features, but the cross-sectional mask cannot open before the
    effective ADD.  A removed security remains required on every execution
    session until an explicit opening-effective terminal event.
    """

    _validate_panel_index(membership.index, name="membership")
    _validate_panel_index(cross_sectional_mask.index, name="cross_sectional_mask")
    _validate_panel_index(feature_tape.index, name="feature_tape")
    _validate_panel_index(execution_tape.index, name="execution_tape")
    if not membership.index.equals(cross_sectional_mask.index):
        raise USIndexContractError(
            "membership and cross-sectional mask indexes must match exactly"
        )
    member = _strict_bool_series(membership, name="membership")
    selected = _strict_bool_series(
        cross_sectional_mask, name="cross_sectional_mask"
    )
    if (selected & ~member).any():
        raise USIndexTapeCoverageError(
            "cross-sectional mask contains a pre-entry or post-removal security"
        )
    if "observed_bar" not in feature_tape.columns:
        raise USIndexTapeCoverageError(
            "feature tape requires explicit observed_bar"
        )
    observed = _strict_bool_series(
        feature_tape["observed_bar"], name="feature_tape.observed_bar"
    )
    selected_index = selected.index[selected.to_numpy(dtype=bool)]
    missing_selected = selected_index.difference(feature_tape.index)
    if len(missing_selected):
        raise USIndexTapeCoverageError(
            f"feature tape misses {len(missing_selected)} cross-sectional rows"
        )
    if not observed.reindex(selected_index).all():
        raise USIndexTapeCoverageError(
            "cross-sectional mask includes a non-observed feature row"
        )

    sessions = _normalize_sessions(trading_sessions)
    mask_dates = pd.DatetimeIndex(
        membership.index.get_level_values("date").unique(), name="date"
    )
    if not mask_dates.equals(sessions):
        raise USIndexContractError(
            "membership mask must cover the exact frozen trading calendar"
        )
    mask_security_ids = pd.Index(
        membership.index.get_level_values("security_id").unique().astype(str)
    )
    if not mask_security_ids.is_monotonic_increasing:
        raise USIndexContractError(
            "membership security_id domain must be sorted"
        )
    expected_index = pd.MultiIndex.from_product(
        [sessions, mask_security_ids], names=["date", "security_id"]
    )
    if not membership.index.equals(expected_index):
        raise USIndexContractError(
            "membership mask must be a complete date x security_id rectangle"
        )
    for security_id in mask_security_ids:
        identities.require_security(security_id)
    if membership_timeline.identities.security_ids != identities.security_ids:
        raise USIndexContractError(
            "membership timeline and universe validator use different identity catalogs"
        )
    expected_membership = membership_timeline.expand_membership(
        session_decision_times,
        security_ids=tuple(mask_security_ids.astype(str)),
    )
    if not expected_membership.index.equals(membership.index) or not expected_membership.equals(
        member.rename("universe_member")
    ):
        raise MembershipReplayError(
            "declared membership differs from ADD/REMOVE event replay"
        )
    feature_dates = pd.DatetimeIndex(
        feature_tape.index.get_level_values("date").unique()
    )
    if len(feature_dates.difference(sessions)):
        raise USIndexContractError("feature tape contains dates outside the calendar")
    execution_dates = pd.DatetimeIndex(
        execution_tape.index.get_level_values("date").unique(), name="date"
    )
    if not execution_dates.equals(sessions):
        raise USIndexTapeCoverageError(
            "execution tape must expose every frozen trading session"
        )
    if "valuation_known" not in execution_tape.columns:
        raise USIndexTapeCoverageError(
            "execution tape requires explicit valuation_known"
        )
    valuation_known = _strict_bool_series(
        execution_tape["valuation_known"],
        name="execution_tape.valuation_known",
    )

    normalized_terminals: dict[str, pd.Timestamp] = {}
    if corporate_actions is not None:
        if corporate_actions.identities.security_ids != identities.security_ids:
            raise USIndexContractError(
                "corporate-action and universe validators use different identity catalogs"
            )
        normalized_terminals = corporate_actions.terminal_session_dates()
    for security_id, date in normalized_terminals.items():
        if date not in sessions:
            raise USIndexContractError(
                f"terminal date for {security_id!r} is absent from the calendar"
            )

    member_frame = member.unstack("security_id").reindex(
        index=sessions, columns=mask_security_ids
    )
    ever_member = [
        str(security_id)
        for security_id in mask_security_ids
        if bool(member_frame[security_id].any())
    ]
    new_entrants: list[str] = []
    removed: list[str] = []
    required_pairs: list[tuple[pd.Timestamp, str]] = []
    for security_id in ever_member:
        history = member_frame[security_id].astype(bool)
        true_locations = np.flatnonzero(history.to_numpy(dtype=bool))
        first = sessions[int(true_locations[0])]
        if first > sessions[0]:
            new_entrants.append(security_id)
        transitions = history.astype(np.int8).diff().fillna(0)
        if (transitions < 0).any():
            removed.append(security_id)
        terminal = normalized_terminals.get(security_id)
        if terminal is not None and terminal <= first:
            raise USIndexContractError(
                f"terminal date for {security_id!r} must follow first membership"
            )
        required_dates = sessions[sessions >= first]
        if terminal is not None:
            # Corporate action is applied before that session's market snapshot.
            required_dates = required_dates[required_dates < terminal]
        required_pairs.extend((date, security_id) for date in required_dates)

    required_index = pd.MultiIndex.from_tuples(
        required_pairs, names=["date", "security_id"]
    ).sort_values()
    missing_execution = required_index.difference(execution_tape.index)
    if len(missing_execution):
        first = missing_execution[0]
        raise USIndexTapeCoverageError(
            "execution tape stops before a removed holding is guaranteed closed; "
            f"first missing {pd.Timestamp(first[0]).date()}|{first[1]}"
        )
    if len(required_index) and not valuation_known.reindex(required_index).all():
        raise USIndexTapeCoverageError(
            "execution tape contains an unknown valuation for a potentially held security"
        )
    unknown_terminals = sorted(set(normalized_terminals).difference(ever_member))
    if unknown_terminals:
        raise USIndexContractError(
            "terminal map contains never-member securities: "
            + ",".join(unknown_terminals)
        )
    return USIndexUniverseValidation(
        trading_sessions=len(sessions),
        security_ids=len(mask_security_ids),
        member_rows=int(member.sum()),
        cross_sectional_rows=int(selected.sum()),
        execution_required_rows=len(required_index),
        new_entrant_security_ids=tuple(sorted(new_entrants)),
        removed_security_ids=tuple(sorted(removed)),
        terminal_security_ids=tuple(sorted(normalized_terminals)),
    )


__all__ = [
    "FractionalTreatment",
    "MembershipAction",
    "MembershipEvent",
    "MembershipReplayError",
    "MembershipSeed",
    "PermanentSecurity",
    "SecurityIdentityCatalog",
    "TickerAlias",
    "USCorporateActionCatalog",
    "USCorporateActionContractError",
    "USCorporateActionEntitlement",
    "USCorporateActionEvent",
    "USCorporateActionKind",
    "USCorporateActionLeg",
    "USCorporateActionLegKind",
    "USIndexContractError",
    "USIndexMembershipTimeline",
    "USIndexTapeCoverageError",
    "USIndexUniverseValidation",
    "US_INDEX_EVENT_DAY_ORDER_SHA256",
    "US_INDEX_EVENT_DAY_ORDER_CONTRACT_V1",
    "US_INDEX_EVENT_DAY_ORDER_V1",
    "US_MARKET_TIMEZONE",
    "UnknownSecurityIdentifierError",
    "validate_us_index_universe_views",
]
