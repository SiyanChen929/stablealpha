"""Reproduce the six publication figures from the supplied small tables."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import tempfile

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'DATA_MANIFEST.json').read_text())
for item in manifest['files']:
    path=ROOT/item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'], f'Changed input table: {item["path"]}'

with tempfile.TemporaryDirectory(prefix='stablealpha-figure-fonts-') as cache:
    os.environ.setdefault('MPLCONFIGDIR',cache)
    for script in ['make_from_zero_figures.py','make_control_figures.py','make_overview.py']:
        runpy.run_path(str(ROOT/'scripts'/script),run_name='__main__')

names=['research_overview','controlled_search_transmission','from_zero_budget_curve',
       'from_zero_capacity_cost','composition_beyond_raw_inputs','historical_frontier']
for name in names:
    for extension in ['pdf','svg']:
        assert (ROOT/'output'/f'{name}.{extension}').is_file()
print('Complete: six vector PDFs and six editable SVGs in output/.')
