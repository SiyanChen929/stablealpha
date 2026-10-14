# CogAlpha configuration profiles

This directory defines the research roles and search profiles; it does not contain
runtime implementation or model prompts.

## Files

- `configs/agents.yaml` is the canonical seven-level, 21-agent catalog. The names
  and level assignments follow the official CogAlpha prompt repository. The local
  fields turn each role into an auditable research contract.
- `configs/faithful.yaml` freezes the published CogAlpha search shape as the
  reproduction baseline.
- `configs/plus.yaml` keeps that search budget and adds the safety, diversity, and
  validation mechanisms proposed for CogAlpha-Plus.
- `reviews/semantic_review_176.json` is the frozen economic-semantic approval
  envelope. It contains one explicit evidence note and four literal `true`
  attestations for every card; it is validated against code-side hashes of all
  five proposal files before a real search can load factors or market data.

## Required agent fields

Every agent has:

- `id`: stable machine identifier;
- `level`: the integer hierarchy level;
- `name`: the upstream role name;
- `mandate`: the mechanism boundary for the role;
- `allowed_inputs`: references into `allowed_input_catalog`;
- `hypothesis_templates`: patterns for generating falsifiable factor cards;
- `crossover_tags`: typed matching hints for mechanism-aware crossover;
- `failure_modes`: domain-specific rejection checks;
- `upstream_template`: the matching public CogAlpha prompt filename.

The hierarchy is a set of specialized research islands, not seven sequential data
processing stages. Every agent receives point-in-time data and may evolve its own
factor family. Pools are merged only under the configured selection and
deduplication policy.

The role input contract is executable. Raw `open/high/low/close/volume` require
`pit_ohlcv`, `amount` requires `pit_amount`, and `vwap` requires
`derived_price`. A card that uses a field outside its role's `allowed_inputs`
fails in proposal validation, `check-config`, and the real seed/injection
loaders; the catalog is not merely descriptive.

## Economic-semantic approval gate

The five Factor Card files are frozen by full-byte SHA-256 and an exact
176-candidate-key set in `semantic_audit.py`. The approved manifest must contain
exactly 176 distinct reviews, all with status `approved`, the same explicit
reviewer attestation, a non-empty card-specific note, and literal `true` values
for mechanism/AST consistency, direction consistency, repair-tag consistency,
and falsification review. Strict JSON, nested canonical hashes and the aggregate
hash are rechecked on every configuration check and real development run.

The current manifest aggregate is
`946ac53db659db97515a5add596db4a7de229dc68d988401c61cd5ae2cad5bb9`.
This is tamper-evident within the local code trust domain, not an authenticated
reviewer signature. Defending against an actor who can rewrite the proposals,
code anchors and manifest together requires retaining this digest externally.

## Faithful profile

The baseline preserves the reference search shape:

| Setting | Value |
| --- | ---: |
| Initial population | 80 |
| Parent pool | 32 |
| Offspring per generation | 96 |
| Generations | 24 |
| Subcycles | 3 x 8 |
| Fresh injection | Every 2 generations |
| Elite carryover | 2 |
| Final factors | 20 |

Some details needed for bitwise reproduction were not released publicly. They are
listed under `reproducibility.unresolved_public_details`; they must not be silently
invented and then described as faithful.

## Plus profile

The Plus profile holds the population and evaluation budget constant so an
improvement cannot be attributed merely to more trials. Runtime-enabled fields
are resolved and hashed before research; roadmap fields are explicitly marked
disabled. It adds:

1. a typed DSL and normalized AST;
2. production-canonical structural deduplication, a behavioral-novelty soft
   search score, and a hard development-only behavioral-correlation gate for
   the final family (semantic and residual-correlation parent gates remain
   roadmap);
3. deterministic, semantic, and mechanism-aware variation;
4. scalar fitness with original-role representatives, tag crowding and novelty
   as the currently implemented quality-diversity heuristic (not NSGA-II or
   MAP-Elites);
5. a purged train/inner-validation split, one-time sealed replay,
   dependence-aware inference, and a complete trial ledger; expanding
   walk-forward refits remain roadmap;
6. explicit personal-capacity execution rules without relaxing tradability;
   these and the ensemble settings belong to downstream portfolio modules and
   are not invoked by HS300 factor discovery.

The final-family gate greedily scans passed candidates in descending
development fitness. It first considers the configured leading pool and then,
only if needed, expands to the remaining passed archive. A candidate is kept
only when its maximum absolute mean daily cross-sectional Spearman correlation
to every selected factor is at most `0.85`. Both this threshold and the leading
pool size are resolved from `configs/plus.yaml` into the hashed run
configuration; the sealed period is not loaded or consulted.

The DSL currently enforces a production field allowlist, causal operator rules
and a price-field scale-degree proof. It does not claim a general per-field
availability type system or physical unit type system.

After each generation, the controller copies only six scalar score fields and
the exact diagnostic keys `rejection`, `coverage`, and `train_raw_rank_ic`.
It converts the highest-priority weakness into one fixed repair: shorter
lookback for coverage, causal persistence for turnover/stability, robust
cross-sectional representation for poor style retention, complementary
recombination for low novelty, nonlinear interaction for a negative control
delta, or conservative structural repair for a rejected expression. Every
offspring stores that observation, cause, fix, source candidate, and target role
in its lineage. Unknown diagnostics are neither enumerated nor copied.

The memory keeps all 21 original roles separate even after crossover. Parent
selection retains the best viable representative of each available role before
filling diversity slots. On injection generations, the eight reserved slots
first target roles absent from the current parent pool, then roles with retained
failures and the fewest prior injections. The injected expression is selected
without replacement from that role's frozen LLM repair-card library;
feedback chooses a pinned failure class, not an unconstrained formula generator.
The scheduler receives only the static set of roles with unused cards, so it
cannot request an exhausted role and silently relabel another role's factor;
role and repair matches are counted separately in every generation audit.

On injection generations, eight of the 96 offspring slots are reserved for fresh
mechanisms. They are not added on top of the budget. The four evolutionary modes
receive 22 slots each, so the total remains 96.

For a real Plus run, all six `fixed_thresholds` are parsed fail-closed into an
`AdaptiveFeedbackPolicy`. Missing/extra keys, booleans, non-finite values and
invalid ranges abort before search. That exact policy object is passed to
`AdaptiveFeedbackMemory` and serialized into the resolved experiment config,
so changing one threshold changes the RunManifest config hash and the sealed
identity. The faithful profile does not instantiate the Plus controller.

Family-wise testing is performed only at the one-time sealed reveal, using the
dependence-preserving block-bootstrap max-T result together with Holm step-down
adjusted p-values; FDR is not the implemented method. The listed comparison
ablations are a planned external protocol and are not automatically run by
factor discovery.

## Sealed reporting contract

Both profiles freeze the same closed `ensemble.reporting_contract` before any
market partition is opened. `composite:lightgbm` on the standard adjusted-VWAP
neutral endpoint is the only result eligible for the primary claim. The joint
20-factor + 3-composite family remains a 23-series HAC/Holm/max-T family for
conservative multiplicity control; it is explicitly not a model-selection or
winner pool. The 20 component factors, equal-weight and Ridge model controls,
three pre-registered neutral-IC contrasts, the composite subfamily, and all
immediate-executable endpoints are pre-registered secondary diagnostics.

The sealed JSON repeats these roles on each record, sets primary eligibility
explicitly, and includes both a canonical contract hash and a hash binding the
unique primary record. Sealed outcomes cannot reassign the primary backend or
promote a better-looking secondary result.

## Leakage boundary

Generation and evolution may consume only training and inner-validation summaries.
Outer and sealed-test results never become adaptive feedback. A factor direction
must be fixed from training data, and the sealed test is revealed only once after
the run manifest, prompts, factors, and model settings are frozen. For real
HS300 research this is a process boundary, not an in-memory flag: development
loads market data only through the post-validation purge and emits a
content-hashed `sealed_reveal_manifest.json`. A separate `hs300-reveal` command
accepts only that artifact. It atomically retires the bound snapshot/split in a
persistent registry before loading any replay row, then verifies the exact
canonical champion AST/hash family and source-data identity. The development
command has no real-data reveal option.
The development artifact stores each complete Candidate identity (including
generation, parents, operator, mechanism tags, control AST and identity-
critical metadata) and reveal reconstructs it to recompute both its
deterministic ID and predeclared direction. The final development ledger event
binds that family to the RunManifest, summary, factor registry and exact
champion evaluation/registry records. Reveal verifies their SHA-256 identities,
the full ledger chain and fixed run-relative locations before making the
one-time claim. Scoring reads the frozen artifact rather than a mutable
in-memory candidate. The final
`label_horizon + entry_delay` sessions form an unscored outcome tail so the
reported sealed end is always a signal date with a fully realized label.

These commitments detect partial or coordinated edits that leave any frozen
development evidence behind. They cannot defeat a same-permission local
attacker who rewrites every file and rebuilds the entire hash chain: that
threat requires an external signature, trusted timestamp, or remotely retained
append-only digest.

## Upstream references

- Paper: <https://aclanthology.org/2026.acl-long.538/>
- Official prompt repository: <https://github.com/uwFengyuan/CogAlpha_Prompt>
- Official seven-level mapping:
  <https://github.com/uwFengyuan/CogAlpha_Prompt/blob/main/prompts/seven_level_agent_hierarchy/README.md>
