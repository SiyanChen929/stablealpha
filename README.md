# StableAlpha
### When does better factor search improve an investment decision?

I study how gains from LLM-guided factor discovery travel through a fixed investment pipeline: **factor training score → joint return prediction → holdings → return after costs**. This project grew from factor-research work during a mentored research experience and continued into an independent working paper.

**Author:** Siyan (Diego) Chen, University of California, Santa Barbara.

## Research in one minute

| Question | Design | Finding |
|---|---|---|
| Does improving a formula's training score improve the joint forecast? | Matched starting libraries; search policies compared at fixed proposal budgets | Repair-directed search increased training scores while lowering joint forecast accuracy in the reported replication. |
| What does extra search effort buy? | Empty-library search with fixed checkpoints and a common predictor/account | Twelve further price-volume proposals improved the best training score by about 17%; no reliable net-return change was established. |
| Is the gain from new information or formula composition? | Equal-dimension direct-field Ridge control and controls with matched information | No return advantage over direct-field Ridge was established. With common raw fields, formula composition retained a small forecasting advantage; the reported return contrast was inside the prespecified one-percentage-point band. |

These findings are conditional on the manuscript's CSI500 cohort, information vintage, learner, trading rule, and comparison. The paper records prior development exposure and distinguishes it from frozen later-period evaluation. Overall investment status: **NOT VALIDATED**.

## Read the paper

**The Marginal Yield of LLM-Guided Alpha Search: Search Effort, Information and Investment Value** — October 2026 preprint. [Zenodo / DOI](https://doi.org/10.5281/zenodo.23379471).

- [Public preprint on Zenodo](https://zenodo.org/records/23379471) · [DOI: 10.5281/zenodo.23379471](https://doi.org/10.5281/zenodo.23379471)
- [Design and interpretation](docs/research-overview.md)
- [Software provenance and contribution](docs/provenance.md)

## Inspect the implementation

The public software is a curated snapshot of the project's **earlier CogAlpha-Plus engineering layer**. It contains the safe factor-expression language, proposal cards, evolutionary operators, deterministic scoring, validation contracts, registries, and execution/accounting modules. The later paper experiments used additional frozen implementations and protocols. This snapshot demonstrates the engineering layer; it is not an end-to-end reproducer of the paper's tables.

Start with [the factor language](src/cogalpha_plus/dsl.py), [candidate evolution](src/cogalpha_plus/evolution.py), [scoring](src/cogalpha_plus/metrics.py), and [the trial registry](src/cogalpha_plus/registry.py).

```bash
python -m pip install -e '.[dev]'
python scripts/synthetic_demo.py
python -m pytest -q
```

The demo uses seeded synthetic prices, runs without credentials or network access, and demonstrates formula validation and calculation. Its values have no empirical investment interpretation. Market data, private run archives, fitted models, and API credentials are excluded.

## Contribution

I set the research questions and specifications, reviewed protocols and outputs, made continuation decisions, and interpreted the findings. AI coding assistants supported implementation, diagnostics, and manuscript preparation. The architecture adapts ideas from CogAlpha; Alpha158 is a Qlib baseline. Detailed attribution and the distinction between engineering and paper experiments appear in [provenance](docs/provenance.md).
