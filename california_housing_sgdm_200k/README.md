# California Housing: unchanged SGDM continued to 200,000 updates

This experiment extends the **same five original SGDM trajectories** to 200,000
total optimizer updates. It tests training duration while holding the model,
optimizer, preprocessing, and data split fixed. The user authorized this horizon
extension on 2026-11-26 after reviewing the earlier alpha/performance plots.

This is an additional continuation phase. It does not repeat the existing grid,
reinitialize a model, replace a completed result, or tune settings against the
test set or the measured alpha.

## Exact lineage and fixed design

1. Original published-MLP cohort: workflow
   [45753214421](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45753214421),
   source `67a0bf203ab0f6b34565e37b4f510f123f669876`, archived results
   `84b0219dc87c7caaefdc7245e455e53c300b8dfc`.
2. Existing 20,000-to-100,000 continuation: workflow
   [45758895218](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45758895218),
   frozen source `f89e5e00a3b9ad09c906b272fbc1d627ccecd714`, results under
   `california_housing_sgdm_search/results/2026-11-26_sgdm_long_search/continuations`.
3. This phase starts from each retained **full 100,000-update checkpoint** and
   performs exactly 100,000 additional updates. Preparation pins the parent
   archive commit and records each file's SHA-256 and Git blob identity.

The five seeds remain **2340, 2341, 2342, 2343, and 2344**. The fixed configuration
is SGD with learning rate **0.03**, momentum **0.9**, zero weight decay, zero
dampening, no Nesterov, and a constant learning rate. The model remains the
623,281-parameter 8→784→784→1 MLP with ReLU and dropout
`0.3690073279296028`, batch size 256, and gradient clipping at norm 1.

The data remain the exact prepared arrays and manifest from the original cohort:
13,209 training, 3,303 validation, and 4,128 test examples. Preparation restores
the previously fitted preprocessing; it never changes the split or refits a
transform. The source and dependencies are frozen by `SOURCE_SHA256.json`.
Python 3.12.9, PyTorch 2.7.1+cpu, float32 arithmetic, and one numerical thread per
worker match the previous CPU experiment. Runtime metadata record the actual
CPU and packages. Restoring exact state does not claim bitwise equivalence across
different CPU kernels or hardware.

## State continuity and checkpoints

The loader verifies the 100,000-update parent run, launch source, seed, optimizer,
model, data identity, and numerical-runtime settings before updating any weight.
It restores the model, momentum buffers, all RNG states, minibatch generator,
current permutation and position, partial-epoch accumulators, and minibatch
history digest. Fingerprints before and after restoration must match.

Full checkpoints are retained at **100,000, 125,000, 150,000, 175,000, and 200,000**
updates, in addition to resumable progress and the optional validation-selected
weights. Earlier 2,000/20,000/50,000/100,000 files remain in their existing archive.
A job retry resumes its retained state; a validated completed run is reused.

## Measurements and interpretation

The primary endpoint is the fixed **200,000-update model**. The analysis compares
20,000, 100,000, and 200,000 updates within the same five seeds. It reports equal
seed-weighted means, sample standard deviations, and two-sided 95% Student-t
confidence intervals. Paired differences use the five within-seed differences.

Validation is measured each epoch; full train/validation/test metrics and spectra
are recorded every 5,000 updates. RMSE and MAE use target units of $100,000; lower
is better. They are regression errors, not classification accuracy or percentage
errors. Alpha and test performance at a plotted point must identify the same
weights. The optional best-validation model is selected only within
**[100,000, 200,000]**, separately from the fixed-budget endpoint. Test performance
and alpha do not choose a checkpoint or hyperparameter.

The spectral average uses ordinary fitted alpha with weights
`sqrt(num_pl_spikes)/(D + 0.02)`, without an additional matrix-size factor. The
784×784 hidden matrix is this model's only eligible spectral matrix, so the
weighted model average equals its alpha. Confidence intervals describe variation
across five seeds, not variation across layers. No result is forced toward 2.

The earlier native and modified AdamW runs end at 20,000 updates. A comparison
with 200,000-update SGDM has unequal compute and does not isolate an optimizer
effect. The purpose of this phase is to observe long-horizon behavior, including
any benefit, plateau, overfitting, or spectral drift.

## GitHub execution and storage

The workflow `.github/workflows/california-housing-sgdm-200k.yml` shares the
existing workflow's concurrency group. It queues behind that workflow and then
runs at most five standard CPU jobs, each with a 90-minute limit. It adds no
simultaneous workers while the existing 20-worker grid is active.

Each seed archives its JSON, logs, full checkpoints, and execution provenance in
`california_housing_sgdm_200k/results/2026-11-26_sgdm_200k`. The final analysis
archives tables, plots, source identities, and audit results in the same
repository. Lightweight Actions artifacts expire after seven days; a failed job
also uploads its full recovery state. Completed experiment files are not deleted.

Preparation claims one unique workflow cohort. Starting a second workflow against
that cohort is rejected; rerun only a failed job in the original workflow. Missing
or invalid parent checkpoints stop before training and do not trigger a restart.

## Reproduction without a notebook

Start with this branch of `CalculatedContent/rg_optimizers`, use Python 3.12.9,
and install the pinned CPU requirements:

```bash
python -m pip install -r california_housing_sgdm_200k/requirements.txt
python -m california_housing_sgdm_200k.verify
python -m pytest -q --import-mode=importlib california_housing_sgdm_200k/tests
```

The GitHub workflow initializes and archives the effective plan, parent snapshot,
verified prepared data, and full parent states before any worker starts. Within
that workflow's recorded environment, this command runs or resumes one seed:

```bash
python -m california_housing_sgdm_200k.run_one \
  --root california_housing_sgdm_200k/results/2026-11-26_sgdm_200k --seed 2340
```

Use each seed in `plan.json`, then analyze the complete set:

```bash
python -m california_housing_sgdm_200k.analysis \
  --root california_housing_sgdm_200k/results/2026-11-26_sgdm_200k
```

Official cohort records require the real GitHub workflow environment. For a
separate local reproduction, restore the archived inputs, use the same Python
and package versions, and run the training engine into a separate output
directory. Do not set GitHub identity variables to impersonate the official run.
For example, from the repository root:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0
python - <<'PY'
import json
from pathlib import Path
source = Path('california_housing_sgdm_200k/results/2026-11-26_sgdm_200k')
output = Path('reproductions/california_sgdm_200k')
output.mkdir(parents=True, exist_ok=True)
plan = json.loads((source / 'plan.json').read_text())
(output / 'config.json').write_text(json.dumps(plan['continuation_config']))
PY
python -m california_housing_sgdm_200k.long_train \
  --plan california_housing_sgdm_200k/results/2026-11-26_sgdm_200k/plan.json \
  --config reproductions/california_sgdm_200k/config.json \
  --data-dir california_housing_sgdm_200k/results/2026-11-26_sgdm_200k/data \
  --parent-checkpoint california_housing_sgdm_200k/results/2026-11-26_sgdm_200k/parent/seed_2340/checkpoint100k.pt \
  --output reproductions/california_sgdm_200k/seed_2340 \
  --seed 2340 --phase continuation
```

The engine records the local runtime separately; the official cohort analyzer
rejects mixing that reproduction with the GitHub results. The original
model/data recipe is documented in
`california_housing_benchmark/README.md`; the prior grid and 100,000-update
extension are documented in `california_housing_sgdm_search/README.md`.
