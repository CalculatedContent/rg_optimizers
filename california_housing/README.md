# California Housing: SGDM, AdamW, and modified AdamW

This CPU experiment tests whether the optimizer behavior studied in MNIST and
Shakespeare extends to ordinary regression. It compares three small neural
networks with the same architecture, split, initialization and minibatch order
within each seed. The primary GitHub cohort contains **five seeds per optimizer**,
with measurements at **2,000 and 20,000 updates**.

**Status and results:** [current campaign](results/2026-11-26_three_optimizers/README.md).
The workflow commits results and resumable checkpoints to this branch. Actions
artifacts provide an additional copy; they are not the durable archive.

## What is fixed before training

| Setting | Value |
|---|---|
| Network | `8 → 128 → 128 → 128 → 128 → 1`; four ReLU hidden layers; biases; no dropout |
| Trainable parameters | 50,817 |
| Initialization | Native PyTorch `Linear.reset_parameters`; `torch.manual_seed(seed)` |
| Data | California Housing, 20,640 rows, eight inputs; target in units of $100,000 |
| Split | One fixed random 80/10/10 split: 16,512 train / 2,064 validation / 2,064 test |
| Split RNG | NumPy `default_rng(20261126)` / PCG64; archived row indices |
| Scaling | Population mean/std, fitted on training rows only, for inputs and target |
| Objective | Mean squared error of standardized targets |
| Minibatches | 128, with replacement; dedicated CPU generator seeded `seed + 1000000` |
| Development seed | 1339; excluded from confirmation statistics |
| Confirmation seeds | 1340, 1341, 1342, 1343, 1344 |
| Runtime | CPU float32; one intra-op, inter-op and numerical-library thread |
| Schedule | 100-update linear warmup; cosine decay to 10% peak at update 20,000 |
| Gradient clipping | Global norm 1, before any optimizer update |
| Weight decay | Zero in all three arms |
| Measurements | Initialization plus every 500 updates: 41 records per confirmation run |
| Saved states | Full model/optimizer/RNG at 2,000 and 20,000; resume state every 500 |

The fixed random split tests this task's held-out examples. It does not establish
geographical extrapolation to a different region or a modern housing-price
forecast. RMSE and MAE are reported after reversing target scaling; multiply
either by 100,000 for its dollar-valued unit. Smaller errors are better.

## Optimizer identity

**SGDM:** momentum 0.9, zero dampening, no Nesterov, no weight decay.

**Native AdamW:** betas `(0.9, 0.99)`, epsilon `1e-8`, no weight decay,
`foreach=False`, `fused=False`.

**Modified AdamW:** the existing per-matrix magnitude graft from
[`short_test_adamw_graft/graft.py`](../short_test_adamw_graft/graft.py), adapted
only to the MLP parameter names. All five `fc*.weight` matrices receive the
graft; all biases retain the ordinary AdamW update. The native Adam moments and
bias corrections remain. With fixed beta1, `m/(1-beta1)` is the uncorrected
momentum buffer on this run's own clipped gradient history. The graft keeps the
Adam update direction and sets its Frobenius norm to the reference learning
rate times that momentum-buffer norm. It never uses another run's gradients,
fits alpha during the update, or projects weights toward a spectral target.

This is the same intervention as the MNIST/Shakespeare magnitude graft, not
one of the separate trace-log projection experiments in `optimizers/`.
All five MLP matrices are grafted, following the original MNIST matrix policy.

## Calibration is separate from confirmation

The nine predetermined development runs each use seed 1339 and the first 2,000
updates of the **same 20,000-update schedule** used for confirmation:

| Optimizer | Three candidates |
|---|---|
| SGDM peak LR | 0.003, 0.01, 0.03 |
| Native AdamW peak LR | 0.0003, 0.001, 0.003 |
| Modified AdamW reference peak LR | 0.003, 0.01, 0.03; native Adam peak fixed at 0.001 |

Thus the modified reference ratios are 3, 10 and 30. Shakespeare's ratio 400
is not transferred to this new task. Select the lowest full validation RMSE at
update 2,000, breaking exact ties toward the smaller candidate rate. Calibration
does not evaluate the test split or fit alpha. Record all candidates, including
failed candidates; an arm with no valid candidate blocks confirmation. There is
no grid expansion or confirmation-seed retuning.

`frozen_selection.json` records the selected settings, source/data/plan hashes,
development results and GitHub launch provenance before any confirmation run.
These are **separately calibrated optimizer recipes**. If native AdamW selects a
rate other than 0.001, its base learning rate differs from the modified arm's
fixed base rate. The comparison therefore evaluates the recipes and does not
isolate a graft-only change at identical native learning rates. The limited
grid does not establish globally optimal hyperparameters.

## Spectra and uncertainty

The primary spectral average uses exactly `fc2.weight`, `fc3.weight` and
`fc4.weight`, each 128×128. The predeclared dimensional rule is a unique 2D
matrix with minimum dimension at least 64. `fc1.weight` (128×8), `fc5.weight`
(1×128), and all biases are excluded by dimensionality, not by their measured
alpha. Graft targets and spectral targets are different lists.

Each eligible matrix uses the unmodified WeightWatcher 0.7.7 `analyze()` call,
with powerlaw 2.0.0. Its ordinary fitted alpha receives weight
`sqrt(num_pl_spikes)/(D + 0.02)`. The model statistic is the weighted sum divided
by the weight sum, without an extra matrix-size factor. It is **not**
WeightWatcher's `alpha_weighted`. All three eligible fits must be valid; an
invalid fit yields a missing complete average and a recorded diagnostic.

Every record preserves all WeightWatcher fit fields, fitted tail size, KS
distance, cutoff and full squared singular values of a float64 copy of each
weight matrix. Those saved eigenvalues are unnormalized diagnostic values;
they do not replace WeightWatcher's fit. Fitting preserves all training RNGs.
Measurement updates also retain gradient/update norms and direction cosines
against the optimizer's own momentum reference, for later mechanism analysis.

Seed means receive equal weight. Summary tables report the mean, sample SD,
and two-sided 95% Student-t interval across the five seeds (four degrees of
freedom). Layer dispersion is separate. Plot bands are pointwise seed intervals,
not simultaneous confidence bands. Paired modified-minus-native performance
differences are evaluated at both fixed budgets. The design permits improvement,
no improvement, or deterioration; no outcome is assumed in advance.

## Reproduce from a fresh CPU environment

GitHub pins Python 3.12.14. Use that patch version and the pinned packages for the closest local reproduction. From this repository's root:

```bash
python3.12 -m venv .venv-california
. .venv-california/bin/activate
python -m pip install -r california_housing/requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
python -m california_housing.verify
python -m california_housing.data --data-dir california_housing/.cache/data
python -m california_housing.calibrate --plan california_housing/plan.json --data-dir california_housing/.cache/data --root california_housing/.cache/local_reproduction
for seed in 1340 1341 1342 1343 1344; do
  python -m california_housing.run_seed --plan california_housing/plan.json --data-dir california_housing/.cache/data --root california_housing/.cache/local_reproduction --seed "$seed"
done
python -m california_housing.analyze --root california_housing/.cache/local_reproduction --local-reproduction
```

A local reproduction is a separate cohort. It must never be placed into the
primary GitHub results directory or mixed with primary statistics. The analysis
checks runtime provenance and reports whether it is a local reproduction.

The official loader downloads
`https://ndownloader.figshare.com/files/5976036` and verifies SHA256
`aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`.
`manifest.json` additionally records decoded array hashes, split hashes,
training scalers, feature names and the prepared archive hash. Calibration
archives the exact split indices and manifest. The original data remain
available through the documented download rather than a notebook.

## Resume one run

```bash
python -m california_housing.train \
  --plan california_housing/plan.json \
  --config california_housing/.cache/local_reproduction/configs/modified_adamw.json \
  --data-dir california_housing/.cache/data \
  --output california_housing/.cache/local_reproduction/runs/seed_1340/modified_adamw \
  --seed 1340 --phase confirmation
```

The trainer resumes `resume.pt`, including optimizer moments and all RNGs.
It rejects mismatched source, numerical runtime, settings, or data. A completed
matching run is validated and skipped. A nonblocking output lock rejects
concurrent duplicate workers. `--no-resume` never overwrites an existing run.
Matching package versions do not guarantee identical arithmetic on another
CPU/kernel; CPU and runner details remain part of the recorded provenance.

## GitHub execution and archive

The workflow runs the development screen first, then five parallel seed jobs.
Each seed job runs the three optimizers sequentially and publishes after each
optimizer. A final analysis job audits all 15 complete runs, both saved endpoint
checkpoints, pairing fingerprints, source/frozen-setting hashes and GitHub
provenance before declaring `complete_five_seeds=true`.

The dedicated `LAUNCH.json` push triggers training. Result commits do not
retrigger it. Workflow concurrency and trainer locks prevent overlapping work.
For a failed job, inspect its log and retained artifact, then **re-run only that
failed job of the original workflow**. It restores durable progress first,
validates the identity, and preserves completed runs. Do not create a new
workflow run merely to recover a failed seed. The publication helper checks
that updates never replace a more advanced or completed state with older work.
`publish restore` is a preparation operation for a fresh job, before training.

The branch archive contains JSON/CSV summaries, raw fit records, figures,
full `.pt` checkpoints, and reproduction code. It does not depend on the
90-day retention of the additional Actions artifact copies.

## Sources

- [Official California Housing loader and data description](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [Established magnitude-graft implementation](../short_test_adamw_graft/graft.py)
- [Original MNIST protocol](../experiments/adamw_modified/2026-11-13_mnist_mlp_experiment01/frozen/Protocol.json)
- [Frozen plan](plan.json) and [source manifest](SOURCE_SHA256.json)
