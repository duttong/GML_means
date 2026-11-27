#!/usr/bin/env python
"""Refresh the data embedded in docs/index.html from the annual mean CSV files.

Run this after gml_annualmeans.py has regenerated gml_annual_means/data_files:

    python docs/update_web_data.py

It reads the Jan-Dec ("jul") yearly files for the gases shown on the page and
rewrites the single `const DATA = {...};` line in index.html.
"""
import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / 'gml_annual_means' / 'data_files'
PAGE = HERE / 'index.html'

# Gases shown in the page's gas picker (must match the GASES list in index.html).
GASES = ['CFC11', 'CFC12', 'CFC113', 'CCl4', 'CH3CCl3', 'H1211', 'HCFC22', 'HCFC141b', 'HFC134a']
COLUMNS = (('Global', 'G'), ('HN', 'HN'), ('LN', 'LN'), ('LS', 'LS'), ('HS', 'HS'))


def num(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, 2)


def load(gas):
    path = DATA_DIR / f'{gas}_jul_yearly.csv'
    with path.open() as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith('#')))
    # drop leading/trailing years without a global mean
    while rows and num(rows[0]['Global']) is None:
        rows.pop(0)
    while rows and num(rows[-1]['Global']) is None:
        rows.pop()
    out = {'y': [int(r['year']) for r in rows]}
    for col, key in COLUMNS:
        out[key] = [num(r[col]) for r in rows]
    return out


def main():
    data = {gas: load(gas) for gas in GASES}
    blob = json.dumps(data, separators=(',', ':'))
    html = PAGE.read_text()
    html, n = re.subn(r'^const DATA = .*;$', lambda _: f'const DATA = {blob};', html, count=1, flags=re.M)
    if n != 1:
        raise SystemExit('Could not find the "const DATA = ...;" line in index.html')
    PAGE.write_text(html)
    for gas in GASES:
        years = data[gas]['y']
        print(f'{gas:9s} {years[0]}-{years[-1]}')
    print(f'Updated {PAGE}')


if __name__ == '__main__':
    main()
