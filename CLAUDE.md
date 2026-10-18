# CLAUDE.md — Claude Code: implementer

Project: **StableAlpha** platform applied to **CSI500 factor mining**.
Codex is the instructor; Claude Code is the implementer. Read `AGENTS.md` for
the full contract.

## Workflow

1. Take the lowest-numbered task in `tasks/inbox/`; `git mv` it to
   `tasks/active/`.
2. Implement exactly what it specifies. If the spec is ambiguous or conflicts
   with the research protocol, stop and write the question into the task file
   under `## Questions for Codex` instead of guessing.
3. Verify every acceptance criterion by running the command it names. Run the
   full suite (`pytest -q`) before finishing.
4. `git mv` the task to `tasks/done/` and add `tasks/done/NNN-slug-report.md`:
   what changed (files), commands run with their real output, deviations,
   open issues.
5. Commit with the task number in the message (`task 003: ...`).

## Environment

- Python venv: `.venv` (Python 3.12). Activate with `source .venv/bin/activate`.
- Setup from scratch: `bash scripts/setup_env.sh`.
- Data root: `$STABLEALPHA_DATA_HOME` (default `./data`, git-ignored).

## Rules

- Do not change research scope, splits, budgets or acceptance thresholds
  without a Codex task saying so.
- Never touch the sealed test period except through the one-time reveal path.
- Never commit market data, credentials or run outputs.
