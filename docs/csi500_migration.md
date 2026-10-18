# Migrating StableAlpha to this machine (CSI500 factor mining)

## Status

| Step | State |
|---|---|
| Python 3.12 venv in `.venv` with core + `dev,data-baostock,figures` + LightGBM | done |
| `stablealpha check-config --project .` | passes (176 cards, 21 roles, review hash `946ac53d…`) |
| `pytest` | 494 passed, 1 skipped (skip needs the local BaoStock CSI300 panel) |
| `stablealpha demo --budget tiny --reveal-synthetic-sealed` | runs end to end (~25 s) |
| CSI500 panel | **pending: data not yet on this machine** |
| Codex/Claude handoff (`AGENTS.md`, `CLAUDE.md`, `tasks/`) | done |

Reproduce from a fresh clone:

```bash
bash scripts/setup_env.sh            # add --with-qlib for pyqlib, --skip-tests to skip pytest
source .venv/bin/activate
stablealpha demo --output runs/demo --budget tiny --reveal-synthetic-sealed
```

## Compatibility fixes made during migration

- `pandas` pinned to `<3`. pandas 3.0 changes the default string dtype, which
  breaks the BaoStock normalization code written against pandas 2.x.
- `pytest` gets `pythonpath = ["."]` so `tests/test_portfolio_replay.py` can
  import `scripts.*` without `python -m pytest`.
- `tests/test_hs300_derived.py`: the fixture's `np.where(cond, 1.0, "")` no
  longer promotes to a string array on current numpy; it now writes `"1.0"`
  directly, which is the array older numpy produced.
- LightGBM is mandatory for Track A and the research proxy (no fallback), so
  the setup script always installs it.

## CSI500 data intake

The code never trusts a provider export blindly: any panel must pass
`configs/data_contract_a_share.yaml` (`require_valid_daily_panel`) before a
search starts. Put the data under `$STABLEALPHA_DATA_HOME` (default `./data`,
git-ignored, never committed), e.g. `data/csi500/raw/`.

The contract expects one sorted, unique `(date, symbol)` daily panel with:

| Group | Fields |
|---|---|
| prices (raw, unadjusted) | `open, high, low, close, vwap` |
| trading | `volume, amount, tradable, limit_up, limit_down` |
| corporate actions | `adj_factor` |
| eligibility | `is_st, listed_days, universe_member` (point-in-time CSI500 membership) |
| classification | `industry` (dated, not today's mapping backfilled) |
| risk | `log_mcap, beta` |
| benchmark | `benchmark_weight` |
| capacity | `adv20_dollars` |

plus metadata: `vendor, extracted_at, calendar, corporate_action_policy,
includes_delisted=true, point_in_time_universe=true,
execution_price_basis=raw_unadjusted`.

Fields that can be derived if missing (as the BaoStock HS300 builder already
does): `vwap = amount/volume`, `listed_days`, `adv20_dollars`, rolling `beta`
to `sh.000905`, rule-based `limit_up/limit_down`, implied-float `log_mcap`.
Fields that cannot be invented: historical CSI500 membership and delisted
names; without them the run has survivor bias.

Next step once the files are here: an adapter `data_sources/csi500.py` that maps
the export to this contract, plus a CSI500 data config modeled on
`configs/data_baostock_hs300.yaml` (benchmark `sh.000905`).
