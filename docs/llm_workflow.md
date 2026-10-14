# LLM proposal layer and evolution workflow

The language model is the proposal layer of the search. It covers the 21
logical research roles serially or in a few parallel batches; "21 agents"
denotes 21 distinct research mandates and memories, not 21 model processes
running at once.

The model supplied a frozen 80-card seed library and a separate 96-card
contingent repair library under `proposals/` (the `codex_seed` and
`codex_injection` folders and the `codex_*` identifiers in the code name this
LLM proposal layer). Each card states a falsifiable economic mechanism for one
of the 21 roles. The deterministic runtime validates every JSON AST, evaluates
only the permitted development split, uses sanitized feedback to drive typed
mutation/crossover, and selects eight still-unused repair cards every second
generation. No local model and no external LLM API are needed during a run. A
later review can propose a newly frozen library, but it cannot interact with
the sealed period or alter an in-flight experiment.

The loop follows standard practice for iterative research automation: define
machine-readable evaluations, score every iteration, retain a running log,
make focused changes and use explicit stopping rules.

## Initial batch

The frozen 10-day seed batch contains exactly 80 cards across every ID in
`configs/agents.yaml`. Validate every expression and reject the whole batch on:

- a missing Agent role;
- duplicate `candidate_key` or canonical expression hash;
- an unknown field/operator, lookahead, more than 48 nodes, or a window over 252;
- a direction or horizon chosen after seeing validation performance.

## Evolution batches

The Plus allocation is exact:

- normal generation: 24 deterministic mutations, 24 semantic mutations,
  24 mechanism crossovers and 24 crossover-then-mutations;
- injection generation: 22 of each plus 8 fresh factors;
- repeat for 24 generations with injection every second generation.

Every crossover response must name both parents, its economic composition rule,
and a falsification test. The validator calculates a separate equal-weight
parent control. Cosmetic scalar rewrites are structural duplicates, not new
offspring.

Anti-bloat compaction is typed rather than syntactic. Each retained subtree must
keep its adjusted-price scale degree and its cross-sectional stage; a
time-series operator can never be moved above a cross-sectional descendant.
The nonlinear child and its equal-weight linear control embed the same two
direction-oriented compacted parent branches. Their source hashes, embedded
branch hashes and node counts are recorded in the child metadata. Under the
frozen 80/32/96/24 budget, the static production audit covers all 2,384
candidates plus all 1,104 crossover controls; the audit is structural and is
not evidence of market efficacy.

Runtime candidate IDs use the `candidate_behavior_v2` identity: expression and
control AST hashes, declared direction, ordered parents, operator, normalized
mechanism tags and role lineage, adaptive cause/fix directive, and Factor Card
source are bound together. Floating validation metrics, explanatory prose and
display text are intentionally excluded. Opposite directions or different
controls therefore cannot share an ID. At the end of development, fewer passed
candidates than `final_factors` is a terminal error with eligible/rejection
counts; failed candidates are never used to fill the frozen family.

At runtime, each Plus child also carries a machine-readable
`adaptive_feedback_v1` record. The controller chooses exactly one cause and one
repair from rejection, coverage, turnover, style retention, worst-window
stability, novelty, or incremental value versus the simple parent control.
Failed-factor memory persists across all 24 generations, while an independent
role-coverage map keeps the original 21 research islands identifiable through
multi-parent descendants. Fresh injections target missing/failed roles and the
corresponding predeclared repair class, draw without replacement from the
96-card library, and remain inside the fixed 96-child budget. The runtime does not
call a model live on every generation; the precommitted conditional library is
the reproducible no-API substitute.

## Sealed data

Historical folds already inspected by researchers are strict replays, not truly
unseen evidence. After code and prompts freeze, a new future shadow period is the
only genuine sample out of sample. Once revealed, that period is retired forever.
