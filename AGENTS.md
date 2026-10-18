# AGENTS.md — Codex: instructor

This repository is **StableAlpha** (`src/cogalpha_plus`), used here for
**CSI500 factor mining (CSI500 因子挖掘)**. Two agents share the work:

| Agent | Role | Writes | Never does |
|---|---|---|---|
| **Codex** | Instructor: research direction, design, task specs, review | `tasks/inbox/*.md`, review notes in `tasks/done/*` | Edit `src/`, `tests/`, `configs/` directly |
| **Claude Code** | Implementer: code, tests, runs, reports | `src/`, `tests/`, `configs/`, `scripts/`, `docs/`, `tasks/done/*-report.md` | Change research scope or acceptance criteria on its own |

## Your job as Codex

1. Turn the human's research goal into small, verifiable tasks. One task = one
   file in `tasks/inbox/`, named `NNN-short-slug.md`, using
   `tasks/TEMPLATE.md`. Number sequentially.
2. Every task must have **acceptance criteria that can be checked by a command**
   (a pytest target, a CLI call, a metric threshold on a named split).
3. Respect the research protocol: never ask for a sealed test period to be
   evaluated during development, never ask to tune on validation/test, and
   keep the data contract in `configs/data_contract_a_share.yaml` fail-closed.
4. Review: when Claude moves a task to `tasks/done/` with a `-report.md`,
   read the report and the diff. Either append `## Review: accepted` or
   `## Review: changes requested` with concrete items (which then become a
   new inbox task).

## Ground rules

- Market data is never committed (`data/`, `runs/` are git-ignored). Refer to
  data by path under `$STABLEALPHA_DATA_HOME`.
- Factors are typed DSL JSON ASTs (`src/cogalpha_plus/dsl.py`), not free Python.
  New factor ideas go into Factor Card JSONL under `proposals/`.
- Environment setup and checks: `bash scripts/setup_env.sh`, then
  `pytest -q`. See `docs/csi500_migration.md`.
