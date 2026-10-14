# Real-data pipeline

## Selected first source: historical CSI300 via BaoStock

The zero-cost first track is now pinned to BaoStock 0.9.3 and the historical
CSI300 constituent domain from 2010-01-01 through 2025-12-31. The adapter has
been live-tested against old constituents and delisted securities. It provides
dated constituent snapshots, raw and backward-adjusted daily prices, turnover,
trade status, ST state, security master records, the CSI300 index series and
dated CSRC industry classifications.

This track is deliberately classified `research_proxy`, because BaoStock does
not supply official historical CSI300 constituent weights and daily bars cannot
prove whether a limit-price order was actually fillable. Implied float market
capitalization, benchmark weights and price-limit locks remain labelled proxy
fields. `strict` validation rejects this dataset.

The downstream optimizer does not upgrade that classification. Solver status
alone is never treated as evidence of feasibility: the returned weights are
rechecked at a fixed one-part-per-million NAV tolerance for investment and cash,
long-only and single-name bounds, benchmark-relative bounds, industry and style
exposures, turnover including the cash leg, tracking error, ADV participation
and untradable-position freezes. `optimal_inaccurate` is accepted only when the
same audit passes; otherwise optimization fails closed. This numerical control
does not make proxy benchmark weights, proxy tradability or an incomplete
corporate-action history capital-grade.

Raw downloads live under `data/baostock_hs300/raw` as immutable request objects
plus atomic state files. Normalized catalogs and their SHA-256 manifests live
under `data/baostock_hs300/normalized`. A failed or logged-out request never
commits a successful object; rerunning the same CLI command resumes from the
cache.

The code now fails closed on the data needed for a portfolio-grade A-share
experiment. A provider adapter must produce one sorted `(date, symbol)` panel
matching `configs/data_contract_a_share.yaml` and a metadata dictionary.

Required metadata example:

```json
{
  "vendor": "provider and dataset version",
  "extracted_at": "ISO-8601 timestamp",
  "calendar": "SSE/SZSE point-in-time calendar version",
  "corporate_action_policy": "how announcement timing and adjustments work",
  "includes_delisted": true,
  "point_in_time_universe": true,
  "execution_price_basis": "raw_unadjusted"
}
```

Before loading Qlib, call `require_valid_daily_panel`. Qlib may consume a
validated snapshot; it must not become the source of truth for historical
constituents, delistings, ST status or price-limit tradability unless those fields
have been independently verified.

The next implementation decision depends on the actual source:

1. a user-supplied Wind/Choice/Tushare or broker export;
2. an existing Parquet/CSV database;
3. the implemented BaoStock research-only track, acceptable for factor
   plumbing and paper trading but not automatically accepted for live capital.

Do not start a 2,384-candidate run before this contract passes. Search quality
cannot repair survivor bias or revised data.
