"""Deterministic adapters and transformations for external market data."""

from .hs300 import (
    HS300PriceViews,
    MembershipSnapshotAudit,
    attach_proxy_benchmark_weights,
    derive_adv20_dollars,
    derive_hs300_daily_features,
    derive_implied_float_mcap,
    derive_listed_days,
    derive_rolling_beta,
    derive_vwap,
    expand_membership_snapshots,
    normalize_baostock_daily_fields,
    separate_price_views,
    validate_membership_snapshots,
)
from .hs300_build import (
    HS300BuildConfig,
    HS300BuildResult,
    build_hs300_research_panel,
    derive_conservative_one_price_limits,
)
from .hs300_audit import (
    AuditFinding,
    AuditSeverity,
    HS300AuditConfig,
    HS300AuditReport,
    audit_hs300_research_panel,
)

__all__ = [
    "HS300PriceViews",
    "HS300BuildConfig",
    "HS300BuildResult",
    "HS300AuditConfig",
    "HS300AuditReport",
    "MembershipSnapshotAudit",
    "AuditFinding",
    "AuditSeverity",
    "attach_proxy_benchmark_weights",
    "derive_adv20_dollars",
    "derive_conservative_one_price_limits",
    "derive_hs300_daily_features",
    "derive_implied_float_mcap",
    "derive_listed_days",
    "derive_rolling_beta",
    "derive_vwap",
    "expand_membership_snapshots",
    "normalize_baostock_daily_fields",
    "separate_price_views",
    "validate_membership_snapshots",
    "build_hs300_research_panel",
    "audit_hs300_research_panel",
]
