# RG optimizer reference baselines

This directory contains the **unmodified control experiments** used to evaluate
RG-motivated optimizer extensions. No ECS correction, trace-log projection,
WW-PGD retraction, spectral-flow subtraction, or other RG intervention is
active in these runs.

## Current status

| Item | Status |
|---|---|
| Data, architecture, initialization, and schedule review | Complete |
| Notebook, restart, RNG, and statistical review | Complete |
| Bounded executable tests | Passing at the final technical-audit merge |
| Source-backed center configurations | Committed |
| Validation-only hyperparameter qualification | Must be run before declaring a frozen “best baseline” |
| Long-horizon target-hardware campaigns | Empirical runs; not replaced by CI |

The audit and qualification documents are:

- [`BASELINE_RECIPE_AUDIT.md`](BASELINE_RECIPE_AUDIT.md)
- [`BASELINE_EXECUTION_REVIEW.md`](BASELINE_EXECUTION_REVIEW.md)
- [`FINAL_TECHNICAL_AUDIT.md`](FINAL_TECHNICAL_AUDIT.md)
- [`FINAL_BASELINE_QUALIFICATION.md`](FINAL_BASELINE_QUALIFICATION.md)

## Baseline suite

| Baseline | Dataset / corpus | Model | Optimizer controls | Entry point |
|---|---|---|---|---|
| **MNIST / MLP3** | 55k optimization / 5k validation; official 10k test monitoring-only | `784 → 512 → 512 → 10`, ReLU | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`notebooks/MNIST_MLP3_Baseline_Comparison.ipynb`](notebooks/MNIST_MLP3_Baseline_Comparison.ipynb) |
| **CIFAR-10 / small ViT** | 45k optimization / 5k validation; official 10k test monitoring-only | 4×4 patches, width 192, 6 blocks, 3 heads | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb`](notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb) |
| **One-head nanoGPT / FineWeb-Edu** | Pinned `sample-10BT`; exact document-disjoint 80M / 1M / 1M GPT-2-BPE splits | 1 block, 1 head, width 128, context 256 | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`nanogpt_one_head/README.md`](nanogpt_one_head/README.md) |
| **nanochat d12** | Native pinned nanochat data/tokenizer pipeline | 12 layers, width 768, context 2048 | Native nanochat Muon + AdamW | [`notebooks/NanoChat_D12_Reference_Baseline.ipynb`](notebooks/NanoChat_D12_Reference_Baseline.ipynb) |
| **nanochat mac_d4** | Separately cached reduced nanochat preparation | 4 layers, width 256, context 512 | Same pinned upstream optimizer mathematics | Same notebook; auto-selected on MPS/CPU |

`nanochat d12` and `nanochat mac_d4` are separate baseline versions. Never
report a `mac_d4` result as d12.

## Scientific contract

1. A complete training run is the unit of replication.
2. Optimizer arms share the same architecture, split identities, seed set,
   evaluation probes, training budget, and measurement cadence.
3. Optimizer-specific learning rates are allowed because SGD, AdamW, and Muon
   have different update geometries.
4. Validation data select hyperparameters and `checkpoint_best.pt`.
5. Protected-test metrics never alter optimization, scheduling, stopping, or
   checkpoint selection.
6. Three-seed uncertainty is a two-sided 95% Student-t interval across complete
   runs. Layers, blocks, matrices, checkpoints, and fit points are not extra
   replicates.
7. Strict WeightWatcher measurements use direct outputs from
   `analyze(ERG=True, randomize=True, ...)`. No fallback alpha, proxy
   `num_traps`, or fabricated `ERG_gap` is allowed.
8. “Best baseline” means the validation winner in the preregistered bounded
   neighborhood for the exact architecture, data, initialization, budget,
   optimizer implementation, and runtime policy.

## Environment and persistent paths

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[experiment]'
```

When installing from the repository root, use:

```bash
python -m pip install -e './baseline[experiment]'
```

Set persistent locations before running long jobs:

```bash
export RG_BASELINE_DATA_DIR="$HOME/rg-optimizer-data"
export RG_BASELINE_RUN_ROOT="$HOME/rg-optimizer-runs"
```

The isolated one-head nanoGPT suite uses:

```bash
export RG_NANOGPT_ONE_HEAD_ROOT="$HOME/rg-nanogpt-one-head"
```

Do not put long-running results in `/tmp`.

## 1. MNIST / MLP3

Run these notebooks in order:

```text
notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb
notebooks/MNIST_MLP3_AdamW_Baseline.ipynb
notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb
notebooks/MNIST_MLP3_Baseline_Comparison.ipynb
```

### Fixed protocol

```text
architecture:         784 → 512 → 512 → 10
hidden activation:    ReLU
optimization split:   55,000
validation split:      5,000
test split:           10,000, monitoring-only
batch size:              128
epochs:                   30
seeds:             1337, 2027, 31415
gradient clipping:       1.0
```

Initialization:

```text
fc1, fc2:  Kaiming uniform, fan-in, ReLU gain
fc3:       Xavier uniform
biases:    zero
```

Learning-rate schedules are updated before every optimizer step:

| Optimizer | Peak LR | Floor | Warm-up | Other settings |
|---|---:|---:|---:|---|
| SGD + Nesterov | 0.05 | 5e-4 | 2 epochs | momentum 0.90, matrix WD 1e-4 |
| AdamW | 1e-3 | 1e-5 | 1 epoch | betas (0.90, 0.999), matrix WD 1e-2 |
| Muon matrices | 0.02 | 0.002 | 2 epochs | momentum 0.95, Nesterov, 5 Newton–Schulz steps, WD 0.01 |
| Auxiliary AdamW | 3e-4 | 3e-5 | 2 epochs | betas (0.90, 0.95), matrix WD 0.01 |

Muon acts on `fc1.weight` and `fc2.weight`; `fc3.weight` and all biases use
auxiliary AdamW. The historical output key `sgd_momentum_muon` remains only for
backward compatibility.

Every seed persists latest, validation-best, final, and per-epoch checkpoints,
including model, optimizer, data-generator, Python/NumPy/Torch/CUDA/MPS RNG
state, best-validation state, and a protocol fingerprint.

## 2. CIFAR-10 / small ViT

Run:

```text
notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb
```

The notebook imports the final public runtime from `rg_baselines.vit_final`.

### Fixed protocol

```text
optimization split:       45,000
validation split:           5,000
test split:                10,000, monitoring-only
patch size:                   4×4
embedding width:              192
transformer blocks:             6
attention heads:                3
MLP ratio:                       4
LayerNorm epsilon:           1e-6
epochs:                        300
batch size:                    128
stochastic depth:             0.10
gradient clipping:             1.0
```

Regularization and augmentation:

```text
RandAugment:       2 operations, magnitude 9
color jitter:      0.30
random erasing:    0.25
mixup alpha:       0.80
CutMix alpha:      1.00
label smoothing:   0.10
dropout:           0.0
```

The patch projection uses fan-in Conv2d initialization. Transformer matrices use
truncated-normal initialization.

| Optimizer | Warm-up start | Peak LR | Floor | Warm-up |
|---|---:|---:|---:|---:|
| SGD + Nesterov | 1e-3 | 0.10 | 0.001 | 5 epochs |
| AdamW | 1e-6 | 1.25e-4 | 1e-5 | 5 epochs |
| Muon matrices | 2e-4 | 0.02 | 0.002 | 5 epochs |
| Auxiliary AdamW | 3e-6 | 3e-4 | 3e-5 | 5 epochs |

The schedule is:

```text
explicit low-LR warm-up
→ cosine decay
→ final 10 epochs at the nonzero LR floor
```

The hardened runner saves accelerator RNG state, isolates randomized
WeightWatcher calls from training RNG, repairs the valid epoch-zero-best edge
case, and writes final plus validation-selected test summaries. Each physical
matrix receives its own curve and its uncertainty interval uses exactly three
complete runs.

## 3. One-head nanoGPT / FineWeb-Edu

See [`nanogpt_one_head/README.md`](nanogpt_one_head/README.md) for the complete
MacBook workflow.

### Fixed protocol

```text
dataset:                  HuggingFaceFW/fineweb-edu / sample-10BT
pinned revision:          593b3a867298afb8ce42625a270ef20ddcad28f9
tokenizer:                GPT-2 BPE
training corpus:          80,000,000 tokens
validation corpus:         1,000,000 tokens
test corpus:               1,000,000 tokens
model:                    1 block, 1 head, width 128, context 256
optimizer steps:          9,766
reporting / WW cadence:   0, 0.125, ..., 1.0 corpus-equivalent budget
validation probe:         64 common fixed batches
test probe:               64 common fixed batches
BLEU probe:               64 common fixed continuations
seeds:                    1337, 2027, 4099
```

| Optimizer | Peak LR | Floor | Warm-up |
|---|---:|---:|---:|
| SGD + Nesterov | 0.05 | 0.005 | 10% of updates |
| AdamW | 6e-4 | 6e-5 | 1% of updates |
| Muon matrices | 0.02 | 0.002 | 5% of updates |
| Auxiliary AdamW | 3e-4 | 3e-5 | 5% of updates |

The sampler draws random contiguous windows from the fixed 80M-token training
corpus. All optimizer arms and training seeds use the same evaluation probes.
Prepared data are accepted only when dataset identity, pinned revision, token
counts, byte counts, and SHA-256 hashes match the reference configuration.

The suite records train/validation/test next-token loss, accuracy, perplexity,
fixed-continuation BLEU, learning-rate trajectories, and direct six-matrix
WeightWatcher metrics. The LR in a checkpoint row is the LR used by the update
that produced that checkpoint.

Mac workflow:

```bash
cd nanogpt_one_head
bash scripts/setup_mac.sh
bash scripts/prepare_data.sh
bash scripts/smoke_test.sh

caffeinate -dimsu bash scripts/run_all_baselines.sh \
  2>&1 | tee "$RG_NANOGPT_ONE_HEAD_ROOT/run_all.log"
```

## 4. nanochat

Run:

```text
notebooks/NanoChat_D12_Reference_Baseline.ipynb
```

The canonical source is pinned to:

```text
92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
```

### Profiles

| Profile | Device role | Architecture | Compile policy |
|---|---|---|---|
| `d12` | CUDA/server reference | 12 layers, width 768, context 2048 | Model and fused optimizer kernels compiled |
| `mac_d4` | Apple-MPS development reference | 4 layers, width 256, context 512 | Same mathematics in eager mode; isolated unsupported MPS operations may fall back to CPU |

`RG_NANOCHAT_PROFILE=auto` selects d12 on CUDA and mac_d4 on MPS/CPU. Override
with `RG_NANOCHAT_PROFILE=d12` or `RG_NANOCHAT_PROFILE=mac`.

The wrapper preserves upstream initialization, Muon/AdamW parameter grouping,
model-size-derived token horizon and total batch, separate learning rates,
40-step warm-up, warmdown, momentum schedule, and weight-decay schedule. It
fingerprints the pinned commit, profile, device, process count, compile policy,
and fallback policy. Checkpoint resume requires model, metadata, and every
optimizer shard.

Before preparing the large corpus, the notebook runs a real pinned upstream GPT
forward/backward pass and native combined Muon/AdamW optimizer step on the
selected device.

## Output contract

Applicable runs persist:

```text
manifest / protocol fingerprint
per-epoch or per-checkpoint performance CSVs
raw and summarized WeightWatcher CSVs
checkpoint_latest
checkpoint_best selected by validation loss
checkpoint_final
periodic analysis checkpoints
test_results.json
run_complete.json
plots with individual runs, means, and 95% Student-t intervals
```

The test set is monitored throughout where required for trajectory plots, but it
is never read by qualification ranking or checkpoint selection.

## WeightWatcher contract

The baseline environment pins:

```text
weightwatcher==0.7.7
```

Strict runs call:

```python
watcher.analyze(
    ERG=True,
    randomize=True,
    ...
)
```

Required direct outputs include:

```text
alpha
ERG_gap
num_traps
```

A missing or non-finite required value fails the reference run. No substitute
value is generated.

## Qualification: promoting a center to the frozen baseline

The committed configurations are source-backed centers, not claims of an
unbounded global optimum. The qualification process is:

1. Run the preregistered bounded neighborhood with validation data only.
2. Select a small finalist set.
3. Run each finalist with the full three-seed protocol.
4. Rank by mean best-validation loss.
5. Freeze the complete winner, source commit, data identity, leaderboard, and
   evidence paths in a lock file.
6. Only after freezing, interpret protected-test comparisons.

Candidate grids, validation ranking, complete-seed enforcement, Student-t
aggregation, and lock-file writing are implemented in
[`rg_baselines/qualification.py`](rg_baselines/qualification.py).

Every lock records:

```text
protected_test_used_for_selection: false
```

## Automated validation

```bash
PYTHONPATH=. python -m unittest discover -s tests -v

cd nanogpt_one_head
PYTHONPATH=src pytest -q
```

GitHub Actions additionally compiles all Python sources, parses notebook code
cells, runs a real pinned WeightWatcher integration, and executes a pinned
nanochat CPU model/optimizer preflight. These bounded checks do not replace the
full three-seed long-horizon campaigns or the required target-MPS preflight.
