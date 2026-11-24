# Five-seed modified AdamW on Shakespeare GPT

This experiment completes five **GitHub Actions** confirmation runs, seeds
**1340–1344**, of the previously selected AdamW magnitude graft. Every run
receives **20,000 updates**. The reference rate is frozen at **0.4**; there is
no new hyperparameter selection, alpha-based selection, or early stopping.

## Frozen experiment

The source is bundled in `frozen_source/`, from
[rg_optimizers revision b306247d](https://github.com/CalculatedContent/rg_optimizers/tree/b306247d1b0078a18b1539c68ba14bb64d41a079).
`plan.json` records the cohort, recipe, source identity, measurement cadence,
and decision to repeat seed 1340. The launchers verify the recorded package
versions, Python version, source hashes, and corpus hash before training.

| Component | Setting |
|---|---|
| Model | Character GPT; 4 blocks, 4 heads, width 128, context 64, 804,096 parameters |
| Weights | Tied token/output embedding; learned positions; no biases or dropout |
| Data | Bundled `data/input.txt`; 65 characters; contiguous 90%/5%/5% train/validation/test |
| Minibatch | 12 random windows × 64 next-character targets |
| Native AdamW | Peak rate 0.001; betas (0.9, 0.99); epsilon 1e-8; zero decay |
| Graft reference | SGDM peak 0.4; momentum 0.9; reference/native rate ratio 400 |
| Graft operation | Keep each transformer matrix's AdamW direction; match its update norm to the same-gradient-history SGDM reference |
| Other parameters | Embeddings and normalization retain native AdamW updates |
| Schedule | 100 warmup updates; cosine decay to one tenth of peak at 17,000; floor through 20,000 |
| Numerics | CPU float32; one numerical thread per worker; global gradient-norm clip 1 |

The original frozen runner defaults to reference rate **0.1**. Use this
directory's `run_additional_seed.py`, which calls the unchanged study worker
with reference rate **0.4**, rather than invoking that default runner directly.

## Why seed 1340 is repeated

`existing_seed1340/` retains the historical completed run and its design.
A separate 500-update replay matched that design, source and package versions,
and initial spectral fits, but did not reproduce the subsequent trajectory
exactly. Validation loss differed by −0.0052965 and the weighted alpha by
+0.0287218 at update 500; the specific numerical cause is not established.
`compatibility.json` records this comparison.

The local replay was continued while the execution path was checked. The
primary background cohort now consists of **five fresh GitHub Actions runs**
in the pinned environment, with execution/build metadata retained for every
seed. Local partial runs are kept separately and are excluded, as is the
historical seed. Duplicate versions of a seed are never counted as replicates.
Archived native AdamW and SGDM controls in `data/original_comparison_audit.json`
remain historical-runtime comparators. Matching their seed labels does not
isolate an optimizer effect from possible runtime differences.

## Background execution and archives

The dedicated workflow is `.github/workflows/shakespeare-graft-five-seeds-20261124.yml`
on branch `shakespeare-graft-five-seeds-20261124`. It starts only when that
workflow file changes on that branch. Five independent jobs train seeds
1340–1344 with the frozen recipe and archive each result, diagnostic log,
execution metadata, and the 2,000/20,000-update checkpoint ZIP. A final job
requires all five results, calculates the statistics, and commits `analysis/`
and a completed `cohort_status.json`. Archived results are stored alongside
this README. The commands below reproduce the same experiment locally.

## Environment and preflight

Use **Python 3.12.14** in a dedicated environment. From this directory:

```bash
python3 -m zipfile -e experiment_inputs.zip .
python3.12 -m venv .venv
. .venv/bin/activate
python --version
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
python -m pip install -r requirements.txt
mkdir -p logs
python run_additional_seed.py --preflight
```

The recorded CPU wheel reports `torch==2.14.1+cpu`. The other pinned numerical
packages are NumPy 2.3.5, SciPy 1.17.0, WeightWatcher 0.7.7, and powerlaw 2.0.0.
Use this directory's requirements, not the older portable requirements under
`frozen_source/`, which specify a different NumPy range. If `python --version`
does not report 3.12.14, select that interpreter before creating the environment.

The committed `experiment_inputs.zip` contains the corpus, historical control
audit, and separate historical seed-1340 result/design. Extract it once before
preflight. It contains no newly measured endpoints.

Preflight checks shared initialization and minibatch state across the optimizer
arms, exact local checkpoint replay, and all 18 spectral fits. Its receipt is
`preflight.json`. It verifies the local execution path, not numerical identity
with every other CPU or library build.

## Launch or resume

For five concurrent CPU workers, activate the same environment in two terminals
and run these foreground processes from this directory:

```bash
# Terminal 1: starts seeds 1341–1344.
python -u supervise.py
```

```bash
# Terminal 2: starts seed 1340 and monitors the entire five-seed cohort.
python -u complete_cohort.py
```

Alternatively, run the seeds individually, sequentially or in separate
terminals according to available resources:

```bash
python -u run_additional_seed.py --seed 1340
python -u run_additional_seed.py --seed 1341
python -u run_additional_seed.py --seed 1342
python -u run_additional_seed.py --seed 1343
python -u run_additional_seed.py --seed 1344
python analyze_results.py
```

Repeat the same command after an interruption to resume the corresponding
checkpoint. Run each seed only once at a time; file locks reject duplicate
workers. Keep the frozen design and environment unchanged when resuming.
The five-seed monitor's `cohort_status.json` is the authoritative cohort status;
`status.json` covers only the four-worker supervisor. An interim analysis can
remain incomplete while seed 1340 is still running.

## Measurements and retained outputs

Each seed writes to:

```text
runs/seed_<seed>/confirmation/rate_0.4/graft/seed_<seed>/
```

`last.pt` is refreshed every 250 updates. The launcher also preserves
`checkpoint_2000.pt` and `checkpoint_20000.pt`. Full validation metrics and
all 18 individual WeightWatcher fits are recorded at initialization and
every 500 updates: **41 measurements per completed seed**. Full train/test
metrics are evaluated at the fixed endpoint and validation-selected checkpoint.
`result.json` contains the completed history; `history.json`, `progress.json`,
worker states, and `logs/` retain readable progress and diagnostics.

Run `python analyze_results.py` after completion. It validates all five results
and produces the loss, perplexity, accuracy, individual layer alpha, and
confidence-weighted, tail-adjusted model averages. For each model:

```text
w_l = sqrt(num_pl_spikes_l) / (D_l + 0.02)
alpha_bar = sum(w_l * alpha_l) / sum(w_l)
```

All 18 unique matrices are required; the tied embedding is counted once.
The fitted `alpha` field is used, not WeightWatcher's `alpha_weighted` field.
Seed-level averages then receive equal weight. Outputs include sample SD,
SEM, and a 95% Student-t interval for the seed mean, separately from layer
variation and fit-quality weights.

`analysis/summary.json` identifies a complete result by
`complete_five_seeds: true` and `n_completed: 5`. Without all five results, the
analysis writes explicitly partial outputs and exits with status 2. Use
`python analyze_results.py --allow-partial` only for progress inspection.
See [ANALYSIS_README.md](ANALYSIS_README.md) for every output file, the historical
seed comparison, and the interpretation of comparisons with archived controls.

