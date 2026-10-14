# Reproduce the paper figures

This self-contained package contains ten small CSV tables and the three plotting scripts used to assemble the manuscript figures. It requires no market-data cache, model fitting, LLM access, credentials, or network access after package installation.

## Run

Use Python 3.10 or later in an environment containing Matplotlib, NumPy, ReportLab and pypdf:

```sh
python -m pip install -r requirements.txt
python run_all.py          # from the repository root: python paper/figure_reproduction/run_all.py
```

The command works from any current directory. All inputs resolve relative to this package, and all generated figures go to `output/`. Input fingerprints are checked before plotting.

For the optional overview PNG preview, install Poppler so `pdftoppm` is available on `PATH`. The six required vector PDFs and editable SVGs are produced even if Poppler is absent. The other five PNG previews are produced by Matplotlib directly.

## Outputs

| File stem | Figure |
|---|---|
| `research_overview` | Three experimental comparisons with observed training and return budget curves |
| `controlled_search_transmission` | Four policies: paired training, prediction and return effects |
| `from_zero_budget_curve` | Discovery from empty learned-factor libraries, including initialization |
| `from_zero_capacity_cost` | Matched-capacity Ridge comparison and complete generation-token cost |
| `composition_beyond_raw_inputs` | Formula-composition effects when both models receive common raw inputs |
| `historical_frontier` | Historical best selection score and cumulative output tokens |

All figures are 18 cm wide and use vector marks, embedded PDF fonts and editable SVG text. Arial is used when available; the overview falls back to Matplotlib's bundled DejaVu Sans. Matplotlib data figures also include DejaVu Sans in their font fallback list. No image-generation model is used.

## Data and statistical interpretation

`data/from_zero/` contains the 80 saved budget outcomes, group means, the single fixed direct-field Ridge outcome, saved repetition contrasts and intervals, and observed token accounts. `data/continuation/` contains saved effects and intervals for eight new repetitions, with six matched starting states averaged within each repetition. `data/common_inputs/effects.csv` contains period-specific and pooled common-input contrasts with their saved HAC intervals.

The scripts assemble existing estimates. They do not recalculate factor values, repeat statistical tests, select favourable checkpoints, or refit models. Search-variation and market-calendar intervals retain their distinct definitions in the accompanying generated caption files. A direct-field Ridge model remains a single fixed comparator rather than an artificially replicated control.

`DATA_MANIFEST.json` records source-relative provenance and input fingerprints. These internal identifiers do not appear in figure labels. Font rendering may differ slightly across systems, while values, scales and vector content remain determined by the supplied tables.

`data/historical/SEARCH_GENERATION.csv` preserves the historical partial-rank selection scores and recorded output totals. Its score is defined separately from the ordinary IC in the new controlled experiments. The overview also exports an alternative matrix layout as `overview_variant_B`.
