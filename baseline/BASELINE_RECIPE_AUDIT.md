# Baseline recipe audit

This document records the August 2026 audit of every control experiment under
`baseline/`. The objective is a **strong, reproducible reference recipe**, not
a claim that a finite hyperparameter grid proves a globally optimal setting.
Any future tuning must use training/validation data only; the protected test set
must not select hyperparameters, schedules, checkpoints, or stopping times.

## Audit summary

| Baseline | Data audit | Optimizer/schedule audit | Status after this revision |
|---|---|---|---|
| MNIST / MLP3 | Canonical MNIST train/test files and standard normalization | Fixed learning rates were replaced by optimizer-specific warm-up/cosine schedules; the Muon arm now uses the reference auxiliary-AdamW partition | **Corrected** |
| CIFAR-10 / small ViT | Canonical CIFAR-10 with a fixed 45k/5k training/validation split and protected test set | Upgraded to the full DeiT-style augmentation/regularization stack, optimizer-specific schedules and nonzero floors, validation-selected checkpoints, and restartable runs | **Corrected** |
| One-head nanoGPT / FineWeb-Edu | Pinned FineWeb-Edu `sample-10BT`, exact document-disjoint 10M/1M/1M GPT-2-BPE splits, file hashes | AdamW follows nanoGPT; Muon follows the reference hidden-matrix/auxiliary-AdamW split; every optimizer has warm-up and cosine decay | **Accepted** |
| nanochat d12 | Pinned upstream nanochat data/tokenizer pipeline | Runs the pinned upstream d12 recipe unchanged, including native initialization, Muon/AdamW groups, scaling rules, warm-up, warmdown, momentum, and weight-decay schedules | **Accepted as upstream reference** |

## 1. MNIST / MLP3

### Data

The baseline uses the official torchvision MNIST training and test sets with
the conventional normalization:

```text
mean = 0.1307
std  = 0.3081
```

The dataset is appropriate for a cheap dense-network control. It is not meant
to establish broad architectural generality.

### Corrected optimizer recipes

All three runs now use 30 epochs, batch size 128, gradient clipping at 1.0, a
short linear warm-up, and cosine decay to a nonzero floor.

| Optimizer | Peak LR | Floor | Warm-up | Other settings |
|---|---:|---:|---:|---|
| SGD + Nesterov | 0.05 | 5e-4 | 2 epochs | momentum 0.90, matrix-only weight decay 1e-4 |
| AdamW | 1e-3 | 1e-5 | 1 epoch | betas (0.90, 0.999), matrix-only weight decay 1e-2 |
| Muon matrices | 0.02 | 0.002 | 2 epochs | momentum 0.95, Nesterov, 5 Newton-Schulz steps, weight decay 0.01 |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 2 epochs | betas (0.90, 0.95), weight decay 0.01 on auxiliary matrices only |

The historical result key `sgd_momentum_muon` remains in file paths so old
baseline stores stay discoverable, but the corrected implementation is
**Muon + auxiliary AdamW**, not Muon + auxiliary SGD.

The Muon partition follows the reference implementation: hidden 2-D matrices
receive Muon; classifier/head parameters, embeddings where present, gains, and
biases receive AdamW.

Primary references:

- nanoGPT training defaults:
  <https://github.com/karpathy/nanoGPT/blob/master/train.py>
- reference Muon implementation and parameter partition:
  <https://github.com/KellerJordan/Muon>

## 2. CIFAR-10 / small ViT

### Data and selection policy

The official CIFAR-10 training set is split once with a fixed split seed:

```text
optimization/train: 45,000 examples
validation:          5,000 examples
test:               10,000 examples
```

All optimizer arms and seeds use the same train/validation identities.
Validation loss selects `checkpoint_best.pt`. Test loss and accuracy may be
plotted by epoch, but they are explicitly monitoring-only and never influence
training or selection.

### Corrected model/training recipe

The model remains the intended small ViT:

```text
patches: 4x4
width: 192
blocks: 6
heads: 3
MLP ratio: 4
```

The strong reference now uses 300 epochs and the full DeiT-style regularization
stack:

```text
dropout: 0.0
stochastic depth: 0.10
RandAugment: 2 operations, magnitude 9
color jitter: 0.30
random erasing: 0.25
mixup alpha: 0.80
CutMix alpha: 1.00
label smoothing: 0.10
gradient clipping: 1.0
```

The augmentation stack matters for ViT/Muon comparisons: weak regularization
can change both generalization and gradient spectral structure, so an optimizer
comparison should not be built on an under-regularized ViT recipe.

### Corrected optimizer schedules

| Optimizer | Peak LR | Floor | Warm-up | Other settings |
|---|---:|---:|---:|---|
| SGD + Nesterov | 0.10 | 0.001 | 5 epochs | momentum 0.90, weight decay 5e-4 |
| AdamW | 1.25e-4 | 1e-5 | 5 epochs | betas (0.90, 0.999), weight decay 0.05 |
| Muon matrices | 0.02 | 0.002 | 5 epochs | momentum 0.95, Nesterov, 5 Newton-Schulz steps, weight decay 0.01 |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 5 epochs | betas (0.90, 0.95), weight decay 0.01 |

The AdamW peak is the DeiT reference LR `5e-4` linearly scaled from effective
batch 512 to the committed effective batch 128:

```text
5e-4 * 128 / 512 = 1.25e-4
```

The implementation now saves exact restart state every epoch, including model,
optimizer, data-loader generator, Python/NumPy/Torch RNG state, protocol
fingerprint, and best validation loss.

Primary references:

- DeiT training recipe:
  <https://github.com/facebookresearch/deit/blob/main/main.py>
- Muon reference implementation:
  <https://github.com/KellerJordan/Muon>
- recent optimizer/recipe interaction study for Muon in ViTs:
  <https://arxiv.org/abs/2605.24770>

## 3. One-head nanoGPT / FineWeb-Edu

### Data

This is not a Tiny Shakespeare demonstration. The preparation script streams a
pinned revision of:

```text
HuggingFaceFW/fineweb-edu
configuration: sample-10BT
revision: 593b3a867298afb8ce42625a270ef20ddcad28f9
```

It creates exact, document-disjoint splits:

```text
train:      10,000,000 tokens
validation:  1,000,000 tokens
test:        1,000,000 tokens
```

The GPT-2 BPE tokenizer is retained deliberately for comparability and simple
decoding of continuation BLEU. The 50,257-token embedding is large relative to
a one-block transformer; therefore parameter-count scaling claims must exclude
or separately report embedding parameters. This does not invalidate the
optimizer control because Muon is applied only to the six hidden transformer
matrices and every arm shares the identical embedding.

### Accepted optimizer recipes

The committed recipes are already aligned with the primary references:

| Optimizer | Peak LR | Floor | Warm-up |
|---|---:|---:|---:|
| SGD + Nesterov | 0.05 | 0.005 | 10% |
| AdamW | 6e-4 | 6e-5 | 1% |
| Muon matrices | 0.02 | 0.002 | 5% |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 5% |

AdamW uses `(0.9, 0.95)` betas, weight decay `0.1`, gradient clipping at `1.0`,
and nanoGPT initialization. Muon uses the reference hidden-matrix partition,
momentum `0.95`, Nesterov, five Newton-Schulz steps, and auxiliary AdamW.

The run is restartable and uses validation loss for best-checkpoint selection.
The test set is monitoring-only.

Primary references:

- nanoGPT: <https://github.com/karpathy/nanoGPT>
- Muon: <https://github.com/KellerJordan/Muon>

## 4. nanochat d12

The d12 control is intentionally different from the home-computer one-head
baseline. It is the **native upstream nanochat reference**, pinned to:

```text
92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
```

The wrapper does not replace upstream architecture or optimization logic. It
preserves:

```text
depth: 12
width: 768
context: 2048
target data ratio: 12 tokens per scaling parameter
embedding LR: 0.30
unembedding LR: 0.008
matrix LR: 0.020
scalar LR: 0.50
cautious Muon weight decay: 0.28
warm-up: 40 steps
warmdown: final 65% of training
final LR fraction: 0.05
```

This recipe is accepted because nanochat performs its own depth/batch transfer,
parameter grouping, initialization, LR/momentum/weight-decay scheduling, data
packing, and tokenizer preparation. Updating the pinned nanochat commit creates
a new baseline version and requires a new audit.

Primary reference: <https://github.com/karpathy/nanochat>

## WeightWatcher contract

Every baseline that runs in this repository must preserve the raw output of:

```python
watcher.analyze(
    ERG=True,
    randomize=True,
    ...
)
```

The baseline must not invent a fallback `alpha`, proxy `num_traps`, or
synthesized `ERG_gap`. Missing or unsupported required measurements fail
visibly in strict reference runs. WeightWatcher analysis may run on CPU copies
to keep unsupported SVD/RMT operations off Apple MPS.

## What “optimized” means here

The committed defaults are now source-backed, internally consistent reference
recipes. They still require actual target-hardware runs before any performance
claim. A genuine hyperparameter optimum is empirical and must be selected on
validation data with a preregistered search; it cannot be inferred solely from
literature defaults. The test set remains untouched by that search.
