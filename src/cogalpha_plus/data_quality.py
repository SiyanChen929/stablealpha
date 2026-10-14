"""Explicit data-quality policy for free-data A-share research.

The quality labels in this module describe provenance, not whether a number
looks plausible.  In particular, a carefully calculated proxy remains a proxy
and must not silently pass a production-quality data gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


RESEARCH_PROXY_WATERMARK = (
    "RESEARCH_PROXY_ONLY: benchmark weights and/or risk fields are not official "
    "point-in-time index data; results are unsuitable for benchmark replication "
    "or capital deployment without licensed-data verification."
)


class AuditMode(str, Enum):
    """Quality gate selected before a data snapshot is evaluated."""

    STRICT = "strict"
    RESEARCH_PROXY = "research_proxy"

    @classmethod
    def coerce(cls, value: AuditMode | str) -> AuditMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown audit mode {value!r}; expected one of {choices}") from exc


class FieldQuality(str, Enum):
    """Provenance class for one field in a point-in-time panel."""

    OFFICIAL_POINT_IN_TIME = "official_point_in_time"
    OFFICIAL_HISTORICAL = "official_historical"
    DERIVED_FROM_OFFICIAL = "derived_from_official"
    RESEARCH_PROXY = "research_proxy"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class FieldAssessment:
    """One auditable field-level provenance assertion."""

    field: str
    quality: FieldQuality
    source: str
    note: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        if not self.field.strip():
            raise ValueError("field assessment requires a field name")
        if not self.source.strip():
            raise ValueError(f"field assessment {self.field!r} requires a source")


@dataclass(frozen=True)
class DataQualityAudit:
    """Fail-closed result of applying a pre-declared quality mode."""

    mode: AuditMode
    accepted: bool
    assessments: tuple[FieldAssessment, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    watermarks: tuple[str, ...]

    def require_accepted(self) -> DataQualityAudit:
        if not self.accepted:
            raise DataQualityError("data-quality audit failed: " + "; ".join(self.errors))
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "accepted": self.accepted,
            "assessments": [
                {
                    "field": item.field,
                    "quality": item.quality.value,
                    "source": item.source,
                    "note": item.note,
                    "required": item.required,
                }
                for item in self.assessments
            ],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "watermarks": list(self.watermarks),
        }


class DataQualityError(ValueError):
    """Raised when data would cross a stronger gate than its provenance allows."""


def hs300_default_assessments() -> tuple[FieldAssessment, ...]:
    """Return the declared provenance of the no-cost BaoStock research path.

    BaoStock supplies historical constituent snapshots but does not expose
    official historical CSI 300 constituent weights.  The free-data path can
    therefore support factor research, but not strict benchmark replication.
    """

    official_daily = "BaoStock query_history_k_data_plus(adjustflag=3)"
    return (
        FieldAssessment(
            "universe_member",
            FieldQuality.OFFICIAL_POINT_IN_TIME,
            "BaoStock query_hs300_stocks(date)",
            "Weekly snapshot expanded by backward-looking as-of join.",
        ),
        FieldAssessment("open_high_low_close", FieldQuality.OFFICIAL_HISTORICAL, official_daily),
        FieldAssessment("volume_amount", FieldQuality.OFFICIAL_HISTORICAL, official_daily),
        FieldAssessment("tradable", FieldQuality.OFFICIAL_POINT_IN_TIME, official_daily),
        FieldAssessment("is_st", FieldQuality.OFFICIAL_POINT_IN_TIME, official_daily),
        FieldAssessment(
            "industry",
            FieldQuality.OFFICIAL_POINT_IN_TIME,
            "BaoStock query_stock_industry(code, date)",
            "Weekly CSRC classification; not a licensed CITIC/CSI industry history.",
        ),
        FieldAssessment(
            "research_prices",
            FieldQuality.DERIVED_FROM_OFFICIAL,
            "BaoStock backward-adjusted OHLC (adjustflag=1)",
            "Kept physically separate from raw execution prices.",
        ),
        FieldAssessment(
            "vwap",
            FieldQuality.DERIVED_FROM_OFFICIAL,
            "raw daily amount / raw daily volume",
            "Daily average transaction price, not an exchange-provided intraday VWAP series.",
        ),
        FieldAssessment("adv20_dollars", FieldQuality.DERIVED_FROM_OFFICIAL, "20-session mean raw amount"),
        FieldAssessment("beta", FieldQuality.DERIVED_FROM_OFFICIAL, "rolling return covariance"),
        FieldAssessment("listed_days", FieldQuality.DERIVED_FROM_OFFICIAL, "BaoStock ipoDate"),
        FieldAssessment(
            "implied_float_mcap",
            FieldQuality.RESEARCH_PROXY,
            "raw close * volume / (turnover_percent / 100)",
            "Rounding and free-float-definition differences make this a proxy.",
        ),
        FieldAssessment(
            "benchmark_weight",
            FieldQuality.RESEARCH_PROXY,
            "implied-float-market-cap weights with equal-weight fallback",
            "BaoStock does not provide official historical CSI 300 weights.",
        ),
    )


def hs300_field_quality_map() -> dict[str, str]:
    """Column-level quality labels compatible with :mod:`data_contract`."""

    official_history = FieldQuality.OFFICIAL_HISTORICAL.value
    official_pit = FieldQuality.OFFICIAL_POINT_IN_TIME.value
    derived = FieldQuality.DERIVED_FROM_OFFICIAL.value
    proxy = FieldQuality.RESEARCH_PROXY.value
    return {
        "open": official_history,
        "high": official_history,
        "low": official_history,
        "close": official_history,
        "vwap": derived,
        "adj_open": derived,
        "adj_high": derived,
        "adj_low": derived,
        "adj_close": derived,
        "adj_vwap": derived,
        "volume": official_history,
        "amount": official_history,
        "adj_factor": derived,
        "tradable": official_pit,
        "limit_up": proxy,
        "limit_down": proxy,
        "limit_proxy_known": proxy,
        "research_close_limit_up_proxy": proxy,
        "research_close_limit_down_proxy": proxy,
        "research_close_limit_known_proxy": proxy,
        "is_st": official_pit,
        "listed_days": derived,
        "industry": official_pit,
        "log_mcap": proxy,
        "beta": derived,
        "universe_member": official_pit,
        "vendor_universe_member": official_pit,
        "research_eligible": derived,
        "benchmark_weight": proxy,
        "adv20_dollars": derived,
        "implied_float_shares": proxy,
        "implied_float_mcap": proxy,
    }


def audit_field_quality(
    mode: AuditMode | str,
    assessments: Iterable[FieldAssessment],
) -> DataQualityAudit:
    """Apply a strict or explicitly watermarked proxy quality policy."""

    selected = AuditMode.coerce(mode)
    items = tuple(assessments)
    fields = [item.field for item in items]
    if len(fields) != len(set(fields)):
        duplicates = sorted({field for field in fields if fields.count(field) > 1})
        raise ValueError(f"duplicate field assessments: {','.join(duplicates)}")

    errors: list[str] = []
    warnings: list[str] = []
    watermarks: list[str] = []
    by_field = {item.field: item for item in items}

    for item in items:
        if not item.required:
            continue
        if item.quality is FieldQuality.UNAVAILABLE:
            errors.append(f"required_field_unavailable:{item.field}")
        elif selected is AuditMode.STRICT and item.quality is FieldQuality.RESEARCH_PROXY:
            errors.append(f"strict_rejects_research_proxy:{item.field}")

    historical_weight = by_field.get("benchmark_weight")
    if selected is AuditMode.STRICT and (
        historical_weight is None
        or historical_weight.quality
        not in {
            FieldQuality.OFFICIAL_POINT_IN_TIME,
            FieldQuality.OFFICIAL_HISTORICAL,
        }
    ):
        errors.append("official_historical_benchmark_weight_unavailable")

    proxy_fields = [
        item.field
        for item in items
        if item.required and item.quality is FieldQuality.RESEARCH_PROXY
    ]
    if selected is AuditMode.RESEARCH_PROXY and proxy_fields:
        warnings.append("research_proxy_fields:" + ",".join(sorted(proxy_fields)))
        watermarks.append(RESEARCH_PROXY_WATERMARK)

    return DataQualityAudit(
        mode=selected,
        accepted=not errors,
        assessments=items,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        watermarks=tuple(dict.fromkeys(watermarks)),
    )


def audit_hs300_fields(
    mode: AuditMode | str,
    overrides: Mapping[str, FieldAssessment] | None = None,
) -> DataQualityAudit:
    """Audit the default free HS300 source, optionally replacing declarations."""

    declared = {item.field: item for item in hs300_default_assessments()}
    if overrides:
        declared.update(overrides)
    return audit_field_quality(mode, declared.values())


__all__ = [
    "AuditMode",
    "DataQualityAudit",
    "DataQualityError",
    "FieldAssessment",
    "FieldQuality",
    "RESEARCH_PROXY_WATERMARK",
    "audit_field_quality",
    "audit_hs300_fields",
    "hs300_default_assessments",
    "hs300_field_quality_map",
]
