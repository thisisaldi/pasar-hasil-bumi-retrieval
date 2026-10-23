# Two-tower: Pasar Hasil Bumi

Retrieval model: given a user (and optionally their current cart), rank the 45 products.
Cart items are masked at ranking time (a basket never repeats a product).

## Setup / run
    uv venv .venv --python 3.11          # TF 2.15 needs Python <= 3.11
    uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
    .venv/Scripts/python.exe -m pytest
    .venv/Scripts/python.exe -m twotower.train --tag mine [--cart --drop 0.2 --dim 64 ...]

Data: `dataset/` (not in the repo - production exports; place the CSVs there) (`prod.users.csv`, `prod.products.csv`, `transactions.csv`). Each run prints a table, appends to
`reports/runs.csv`, writes `reports/<tag>.json` (`--save` stores weights in `artifacts/<tag>`).

## Layout
- `src/twotower/data.py` load, time split 70/10/20, cart features
- `src/twotower/metrics.py` full-catalog hit@k + baselines (random, popularity, personal frequency)
- `src/twotower/model.py` towers + TFRS model; `train.py` CLI, early stopping on val hit
- `notebooks/` original Colab notebook (kept for reference); `reports/` run logs

## Results (test, cart-masked hit@k)
![training curves](reports/final_curves.png)

| | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| popularity | 0.098 | 0.236 | 0.358 | 0.584 |
| personal frequency | 0.381 | 0.769 | 0.860 | 0.912 |
| two-tower (default) | 0.387 | 0.755 | 0.841 | 0.892 |

DAT (`--dat`, mean of 3 seeds): 0.382 / 0.755 / 0.850 / 0.903 vs plain 0.380 / 0.740 / 0.838 / 0.903.
Small, consistent gain at @3/@5; seed std is ~0.01, so treat as suggestive. Still below personal frequency.

Data is synthetic (80% of picks from 3-6 per-user favorites), so a user-id two-tower can at best
learn "personal frequency". Validate on real orders before claiming more.
