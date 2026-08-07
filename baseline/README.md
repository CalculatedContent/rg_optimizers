# RG optimizer baselines

This folder contains **unmodified optimizer/model baselines** for the RG-optimizer
experiments. No trace-log projection, self-consistent ECS correction, WW-PGD
retraction, spectral-flow subtraction, or other RG intervention is applied.

The baseline suite currently includes:

| Baseline | Dataset / corpus | Model | Purpose |
|---|---|---|---|
| MNIST / MLP3 | MNIST | 784 -> 512 -> 512 -> 10 MLP | Fast dense-network optimizer debugging and spectral diagnostics |
| CIFAR-10 / small ViT | CIFAR-10 | 4x4 patches, width 192, 6 transformer blocks, 3 heads | Vision-transformer optimizer comparison under fixed architecture/data |
| nanochat d12 | nanochat miniseries corpus | 12-layer, 768-wide, 2048-context decoder transformer | Modern small-LLM reference baseline using nanochat's tuned initialization, scaling rules, hybrid Muon+AdamW optimizer, and schedules |

The MNIST notebooks are:

1. `notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb`
2. `notebooks/MNIST_MLP3_AdamW_Baseline.ipynb`
3. `notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`
4. `notebooks/MNIST_MLP3_Baseline_Comparison.ipynb`

The additional architecture baselines are:

5. `notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb`
6. `notebooks/NanoChat_D12_Reference_Baseline.ipynb`

## Shared persistent run directory

All baseline notebooks resolve the same run root:

```text
RG_BASELINE_RUN_ROOT, when set
otherwise: baseline/runs/
```

For a clone at `/tmp/rg_optimizers`, the default is therefore
`/tmp/rg_optimizers/baseline/runs`. A different shared local directory can be
selected before starting Jupyter:

```bash
export RG_BASELINE_RUN_ROOT=/tmp/rg_optimizers_baseline_runs
```

The MNIST/CIFAR data directory can likewise be overridden with
`RG_BASELINE_DATA_DIR`; its default remains `baseline/data/`.

## MNIST / MLP3 baseline

All MNIST runs use the same architecture and preprocessing:

```text
784 -> 512 -> 512 -> 10
ReLU after fc1 and fc2
MNIST normalized by mean 0.1307 and std 0.3081
```

### Independent replicates and error bars

Each optimizer notebook runs three independent complete training trajectories:

```python
SEEDS = (1337, 2027, 31415)
```

The unit of replication is a full model-training run. It is **not** a minibatch,
test example, layer, or WeightWatcher fit point.

Every aggregate curve reports the mean and the two-sided 95% Student-t
confidence interval,

$$
\bar{x}
\pm
t_{0.975,n-1}\frac{s}{\sqrt{n}},
\qquad n=3.
$$

The plots show faint individual-seed trajectories behind the mean, a shaded
confidence band, and capped error bars. Summary CSV files record `n`, sample
standard deviation, standard error, Student-t critical value, interval
half-width, lower/upper bounds, minimum, and maximum.

### Baseline definitions

#### SGD + momentum

```python
torch.optim.SGD(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    dampening=0.0,
    nesterov=False,
    weight_decay=1e-4,
)
```

#### AdamW

```python
torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,
)
```

#### SGD + momentum + Muon

Muon is applied to `fc1.weight` and `fc2.weight`. It first forms a momentum
update and replaces that matrix update by its approximate polar factor using
five quintic Newton--Schulz iterations. The final classifier matrix
`fc3.weight` and all biases use ordinary SGD + momentum. This is intentionally
an **SGD-auxiliary** Muon baseline rather than `MuonWithAuxAdam`.

The Muon notebook prints and asserts the parameter-to-optimizer assignment
before training.

### MNIST metrics

Epoch zero is evaluated as well. For every seed and epoch, the framework records
train/test loss and accuracy, online train metrics, gradient norms, parameter
norm, global step, training/evaluation time, and WeightWatcher time.

For every layer and epoch, the spectral CSV contains WeightWatcher `alpha`,
`detX_num`, `num_pl_spikes`, `ERG_gap`, midpoint retained rank, midpoint
trace-log, geometric-mean diagnostics, effective-rank measures, energy
fractions, condition number, and normalization audits.

`alpha`, `detX_num`, `num_pl_spikes`, and `ERG_gap` come directly from
`watcher.analyze(ERG=True)`. The code rejects a missing boundary, inconsistent
gap, incomplete epoch/layer, or inconsistent midpoint rather than substituting
a silent fallback.

### MNIST checkpoint and result persistence

Every optimizer notebook sets `save_epoch_checkpoints=True`. Each seed folder
contains both a final state and a complete model/optimizer state after every
training epoch. The comparison notebook validates all persisted artifacts and
produces mean/95% CI trajectories, layerwise WeightWatcher comparisons,
final-epoch tables, convergence summaries, paired seed-level differences, and a
reproducibility manifest.

## CIFAR-10 / small ViT baseline

Run:

```text
notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb
```

This notebook trains the same small Vision Transformer from scratch on CIFAR-10
with three optimizer baselines: SGD + Nesterov momentum, AdamW, and Muon with an
auxiliary optimizer for parameters that should not receive Muon updates.

The committed reference architecture is:

```text
input: 32x32 RGB
patch size: 4x4
embedding width: 192
transformer blocks: 6
attention heads: 3
MLP ratio: 4
training budget: 120 epochs
replicates: 3 seeds
```

Training uses random crop, horizontal flip, RandAugment, mixup, label smoothing,
gradient clipping, linear warmup, and cosine decay. It persists train/test
metrics, checkpoints, WeightWatcher diagnostics, aggregate CSVs, and 95%
confidence intervals.

## nanochat d12 reference baseline

Run:

```text
notebooks/NanoChat_D12_Reference_Baseline.ipynb
```

This is the modern small-LLM reference baseline. It deliberately **does not
reimplement nanochat training inside RG Optimizers**. Instead,
`rg_baselines/nanochat_reference.py` checks out a pinned upstream nanochat
revision and runs nanochat's own training code so that the initialization,
architecture, optimizer grouping, scaling rules, and schedules remain the
reference implementation.

Pinned upstream revision:

```text
karpathy/nanochat
commit 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
```

The only runtime source patch replaces nanochat's hard-coded seed with an
environment-controlled seed so that three independent full training replicates
can be run without changing the optimization recipe.

### Reference model scale

The committed baseline uses nanochat **d12**:

```text
layers: 12
model width: 768
context length: 2048
training target: 12 tokens per scaling parameter
replicates: seeds 17, 29, 43
```

nanochat computes the appropriate batch size and training horizon from its own
scaling rules rather than using an arbitrary fixed step count.

### Initialization and optimizer recipe

The baseline preserves nanochat's native initialization and parameter grouping.
Transformer matrix parameters use **Muon**, while embeddings, unembedding,
learned scalars, and other non-Muon parameters use their native AdamW groups.
The upstream reference values before nanochat's model/batch scaling rules are:

```text
embedding LR:     0.30
unembedding LR:   0.008
matrix LR:        0.020
scalar LR:        0.50
Muon weight decay: 0.28
```

The schedule is also left upstream and intact:

```text
LR warmup:             40 steps
main phase:            plateau
warmdown:              final 65% of training
final LR fraction:     0.05
Muon momentum:         scheduled warmup/warmdown
Muon weight decay:     cosine decay toward zero
```

This matters for optimizer experiments: the nanochat reference should be a
strong baseline, not a generic GPT trained with guessed AdamW hyperparameters.

### nanochat data and evaluation

The notebook uses nanochat's own miniseries data/tokenizer preparation and runs
the upstream base-training evaluation path. It saves validation results and
checkpoints periodically and runs the nanochat CORE evaluation at the end.

The RG wrapper parses the upstream logs into tidy CSV files for comparison with
other experiments. WeightWatcher analysis is performed **offline on saved
checkpoints** so spectral diagnostics do not perturb timed nanochat training.
This produces layerwise spectral measurements suitable for comparison against
RG optimizer runs, including available WeightWatcher `alpha`, ERG, and
correlation-trap diagnostics.

Typical nanochat outputs live below:

```text
runs/nanochat_d12/
  seed_17/
  seed_29/
  seed_43/
  performance_all_runs.csv
  final_summary_95ci.csv
  weightwatcher/
```

The exact checkpoint/log substructure follows the pinned nanochat revision.

## Recommended benchmark suite

For serious optimizer comparisons, use the baselines as a progression in model
complexity:

```text
1. MNIST / MLP3
   - cheapest debugging baseline
   - dense matrices and detailed per-epoch spectral diagnostics

2. CIFAR-10 / small ViT
   - image classification
   - transformer architecture with a modest compute budget

3. nanochat d12
   - modern autoregressive language-model training
   - strong native initialization and tuned hybrid Muon+AdamW recipe
   - scaling-law-derived batch/training horizon
```

A new RG optimizer should be compared against the appropriate strong baseline
for each model rather than against one universal set of optimizer
hyperparameters.

## Run order

For the MNIST suite, run:

```text
1. MNIST_MLP3_SGD_Momentum_Baseline.ipynb
2. MNIST_MLP3_AdamW_Baseline.ipynb
3. MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb
4. MNIST_MLP3_Baseline_Comparison.ipynb
```

The CIFAR-10 ViT and nanochat notebooks are independent architecture baselines
and can be run separately once their respective data/runtime requirements are
available.

## Tests

```bash
cd baseline
PYTHONPATH=. python -m unittest discover -s tests -v
```
