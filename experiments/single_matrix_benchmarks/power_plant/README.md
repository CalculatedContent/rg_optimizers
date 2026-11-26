# Combined Cycle Power Plant — next single-matrix experiment

Prepared, not launched. UCI source: https://archive.ics.uci.edu/dataset/294/combined%2Bcycle%2Bpower%2Bplant ; Tüfekci (2014), https://doi.org/10.1016/j.ijepes.2014.02.027.

## Data and reference
Pinned author benchmark files from Gal and Ghahramani's repository, commit 6eb4497628d12b0f300f4b4f6bdc386bebad565c, under UCI_Datasets/power-plant/data. The checked-in source files were retrieved directly from that commit. Git blob identities are in plan.json. The upstream dataset matrix has 9568 rows, four numerical inputs, one power output (MW). The published outer split 0 has 8611 train and 957 test rows with complete, non-overlapping row-index coverage. There are 41 repeated full rows in the whole dataset and 11 test rows exactly duplicate an outer-train row. Keep the historical outer split for parity; disclose this potential optimism, and propose a separately reviewed duplicate-grouped sensitivity study. Do not infer time-blocked generalization.

Gal and Ghahramani's updated public results report MC-dropout RMSE 4.01 ± 0.04 MW (standard error across 20 outer splits), not a single fixed-fold score and not an AdamW baseline. https://github.com/yaringal/DropoutUncertaintyExps/tree/6eb4497628d12b0f300f4b4f6bdc386bebad565c. An exact literature reproduction would require the full protocol, model and all 20 splits. Our first study uses only fold 0 (never pick a fold based on performance), deterministic inner validation within outer train, then refits on all outer training rows with frozen hyperparameters. Report five-seed CI conditional on this split. Literature performance is a reference band, not a guaranteed attainable optimum.

## Budget and comparison
Architecture 4 → 784 → 784 → 1. FC2 only: 784×784. Fixed batch 256, clipping 1, no normalization layers; train-only z-score inputs and targets. No dropout in primary. All runs exactly 20k updates maximum. Development: two seeds on six SGDM settings (LR .003/.01/.03 × momentum .9/.99) and three LR settings (.0003/.001/.003) for each AdamW track (decay 0 and .001). Total 24. Select separately per track by mean fixed-20k validation RMSE, then five confirmation seeds per track (15). Primary total 39 runs, up to 780000 updates; optional paired dropout extension 15 additional runs only after review, up to 300000 more updates. No simultaneous datasets, maximum five standard CPU workers and 45-minute job timeout when a live workflow is installed. No optimizer-only causal claim from nonzero-decay comparison. No new AdamW search on California Housing. No restart of old runs.

## Reproduction (no training by default)
Python 3.12.9 and pinned requirements in parent folder. From repo root:
```bash
python -m pip install -r experiments/single_matrix_benchmarks/requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python experiments/single_matrix_benchmarks/prepare_power.py
python experiments/single_matrix_benchmarks/run_power.py
```
The first Python command prepares and audits data (skip it if the committed prepared directory exists), the second prints the frozen run matrix. A separate no-training preflight workflow compiles the scripts, checks exact data hashes and writes the dry-run plan. No training is started by these scripts unless the explicit execution flag is provided. Only after user review, install the staged workflow template or explicitly run a single approved case:
```bash
python experiments/single_matrix_benchmarks/run_power.py --execute-training --phase grid --case s0 --seed 3370
```
Run the remaining frozen development cases, then `audit_power.py select`. Confirm `--phase confirm --case <track> --seed <fresh-seed>`. Run `audit_power.py summarize` after complete confirmation. Optional dropout phase requires both explicit execution and extension flags and the matched archived no-dropout counterpart. All full checkpoints and logs need archival, not merely temporary artifacts. The prepared code is not yet certified by a real training run. [Primary GitHub Actions workflow template](workflow-template.json) is intentionally inert. Installing it at `.github/workflows/single-matrix-power-plant.yml` on this isolated branch would start 39 fresh primary runs, so do not install it before review. It excludes the optional dropout extension. No numerical training smoke test or real training has yet been performed.
