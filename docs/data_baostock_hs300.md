# BaoStock historical CSI300 data track

## Evidence checked on 2026-08-29

- Package: `baostock==0.9.3` from
  [PyPI](https://pypi.org/project/baostock/).
- Historical constituents: `query_hs300_stocks(date)`. Live probes returned
  exactly 300 names for dates in 2010, 2011, 2015, 2020 and 2025; the response
  includes a provider `updateDate` no later than the requested date.
- Daily bars: `query_history_k_data_plus`, with `adjustflag=3` for raw cash
  prices and `adjustflag=1` for backward-adjusted research prices. The official
  [adjustment-factor note](https://www.baostock.com/helpdocs/pdf/BaoStock%E5%A4%8D%E6%9D%83%E5%9B%A0%E5%AD%90%E7%AE%80%E4%BB%8B.pdf)
  documents the factor calculation.
- Delisted history: live probes recovered `sh.600005` (武钢股份) through its
  2017-02-14 out-date and `sz.000562` (宏源证券) through 2015-01-26.
- Historical industry: `query_stock_industry(date)` returns dated CSRC
  classifications. It is not a licensed CSI/CITIC/Barra industry history.

BaoStock's access page currently limits a client to 50,000 requests per day and
disallows concurrent connections. The downloader is intentionally serial and
the complete plan is far below that request ceiling.

## What is direct, derived and proxy

| Field family | Quality | Construction |
|---|---|---|
| Historical membership | provider point-in-time | Native weekly versions, backward as-of expanded |
| Raw OHLCV/amount/status/ST | provider historical | BaoStock daily bars, `adjustflag=3` |
| Adjusted research prices | deterministic derived | Raw price × backward factor |
| Daily transaction-average price | deterministic derived | amount / volume; suspended-day close is valuation-only |
| ADV20 | deterministic derived | trailing 20-session amount |
| Industry | provider point-in-time | dated CSRC classification |
| Beta | research estimate | rolling adjusted return covariance to `sh.000300` |
| Float capitalization | research proxy | close × volume ÷ turnover rate |
| Benchmark weights | research proxy | normalized implied float capitalization, equal-weight fallback |
| Limit-up/down lock | research proxy | board/date/ST rule plus one-price daily bar |

The last three proxy families must never be described as Barra exposures,
official CSI300 weights or verified order-book tradability.

## Temporal rules

- 2010 is used only as feature warm-up; the historical research interval begins
  in 2011.
- The frozen dataset has two physical views. The continuous tape contains every
  cached bar for the 790-name historical constituent union; the eligible panel
  contains only the 300 names that belong to the index on each date. Lag,
  rolling and forward-label operators run on the continuous tape, so an exit
  and later re-entry cannot turn a months-old quote into a one-day lag.
  `cs_rank` and `cs_zscore` receive the explicit daily eligibility mask and are
  therefore computed only across that day's 300 members. Nonmembers remain in
  the tape for temporal history but are invisible to cross-sectional nodes.
- Signals are formed after the day-`t` close and may first execute in session
  `t+1`.
- The primary ten-session forecast label uses adjusted VWAP so an ex-date is
  not mistaken for a predictive price collapse. It is explicitly a forecast
  endpoint return, not a claim that both endpoints filled. Beside it, every
  candidate reports an immediate-execution diagnostic that marks suspension,
  limit-up entry, limit-down exit and unknown limit status as missing and
  discloses the missing-endpoint share. It does not filter on future index
  membership. Delayed fills, T+1 and costs are evaluated only by the event
  backtester. Cash execution continues to use raw VWAP.
- If a sourced symbol exchange removes the source security inside a label
  window, the bridge values the successor at the same endpoint field as that
  label: raw open for Track A and raw VWAP for the research-proxy path. The
  bridge accepts the complete tape, extracts the canonical raw field itself,
  and verifies `adj_price == raw_price * adj_factor`; an adjusted-only Series
  is rejected rather than silently treated as raw.
- BaoStock `adjustflag=1` is an IPO-anchored backward-adjusted path. Its
  documented event multiplier is previous raw close divided by the current
  raw `preclose`; an event changes the factor on that date and later, never on
  earlier dates. `adjustflag=2` is forbidden because its current-price anchor
  rewrites the historical absolute scale when a later action occurs.
- Every direct `adjustflag=1` pair is checked against the raw-`preclose`
  recurrence. The production DSL first proves symbol-scale degree zero
  statically and forbids a time-series operator from wrapping a masked
  cross-sectional descendant. It then recomputes every surviving candidate
  after each security's adjusted OHLC/VWAP is multiplied by a different stable
  constant in `[0.2, 5.0]`. A candidate whose values change is rejected:
  arbitrary IPO or vendor price anchors are not Alpha.
- A snapshot is joined only when `vendor_update_date <= signal_date`.
- No future fill, backward fill or current-constituent reconstruction is
  permitted.

### Adjustment evidence and the faster event-table path

The causal interpretation was checked against the captured cache on
2026-08-30, without issuing additional provider requests:

- For `sh.600000`, the 20 overlapping rows from a query ending 2010-01-31 and
  one ending 2025-12-31 had exactly identical backward-adjusted
  OHLC/`preclose` values. A future query endpoint did not rescale the prefix.
- Across the first 22 completed raw/backward pairs, 80,441 rows and 323 factor
  changes, reconstructing five price fields from raw `preclose` had maximum
  relative error `6.29e-7` (vendor decimal rounding).

This is an algebraic and as-of consistency audit of the frozen provider
snapshot. It does not prove that every historical value is identical to the
vintage published on that old date; vintage archives remain a separate
licensed-data requirement.

The module also supports a lower-latency reconstruction from raw bars plus the
documented `query_adjust_factor` event fields (`dividOperateDate`,
`backAdjustFactor`). It applies the last event with
`dividOperateDate <= price_date` and requires an event factor on or before the
first raw row for every symbol. Future event rows therefore cannot affect an
earlier prefix.

This event-table route is not yet the reference downloader. Before replacing
direct `adjustflag=1` capture, one complete cached event response must be
cross-checked against direct vendor prices for all OHLC/`preclose` rows and the
factor endpoint must be included in the content-addressed manifest. Until that
test passes, the slower direct pair remains the fail-closed choice.

After the price and adjustment-factor manifests are complete, run that gate
offline with:

```bash
PYTHONPATH=src python -m cogalpha_plus.cli \
  data adjustment-reconstruction-audit
```

The command performs no provider calls and writes no files. It recomputes every
request key and cache-object hash, requires one raw, one direct backward-adjusted
and one adjustment-factor request for every historical-union symbol, then
compares `open`, `high`, `low`, `close` and `preclose` on the exact same
date-symbol index. Its registered limits are `rtol=2e-6`, `atol=2e-5`, with CLI
overrides restricted to tighter values. JSON output is valid only at 100%
symbol, row and cell coverage with zero out-of-tolerance cells. A failed or
incomplete report retains direct `adjustflag=1` as a required source; a valid
report only authorizes a separately pre-registered v2 data contract.

## Remaining production blockers

1. Official historical daily constituent weights.
2. Official daily limit prices or order-book/auction data.
3. A complete, auditable corporate-action event feed. The runtime now has a
   typed cash/share ledger and fails when a held adjustment factor changes
   without an event; BaoStock's factor alone cannot decompose dividends,
   distributions, rights decisions, exchanges and taxes.
4. Licensed point-in-time risk exposures and a calibrated market-impact model.
5. Broker reconciliation and forward paper/live evidence.

Until those are supplied, every output carries the `research_proxy` watermark.

## Frozen-snapshot audit

After `data build`, run the independent offline gate:

```bash
PYTHONPATH=src python -m cogalpha_plus.cli data audit
```

The command does not call BaoStock. It reconstructs the panel from its frozen
partitions, verifies SHA-256 provenance through the normalized and raw cache
layers, independently reconstructs the continuous union tape and emits a
fail/warn JSON report. It requires the tape's member mask to equal the frozen
300-name panel exactly and checks that each security has no missing market
sessions between its first and last cached bar. Structural gaps, contract breaks,
future-dated joins, calendar gaps, missing delisted cache pairs and hard return
outliers fail the gate. Natural suspensions, unknown daily-bar limit proxies,
equal-weight fallbacks and review-threshold return outliers are reported as
warnings with counts and samples. CLI threshold overrides may only tighten the
frozen defaults; attempts to loosen them fail before any data is read.
