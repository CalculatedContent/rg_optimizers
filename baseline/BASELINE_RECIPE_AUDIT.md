# Baseline recipe audit

This document records the August 2026 audit of every control experiment under
`baseline/`. The objective is a **strong, reproducible reference recipe**, not
a claim that a finite hyperparameter grid proves a globally optimal setting.
Any tuning must use training/validation data only; the protected test set must
not select hyperparameters, schedules, checkpoints, or stopping times.

## Audit summary

| Baseline | Data audit | Optimizer / initialization / schedule audit | Status after final technical pass |
|---|---|---|---|
| MNIST / MLP3 | Official MNIST, standard normalization, deterministic 55k/5k optimization/validation split, protected test | Explicit Kaiming-ReLU/Xavier initialization, optimizer-specific update-level warm-up/cosine schedules, correct Muon auxiliary-AdamW partition | **Corrected and qualified for validation search** |
| CIFAR-10 / small ViT | Official CIFAR-10 with fixed 45k/5k optimization/validation split and protected test | DeiT-style augmentation, LayerNorm eps 1e-6, fan-in patch initialization, explicit warm-up start, cosine plus cooldown, optimizer-specific schedules, restartable validation-selected runs | **Corrected and qualified for validation search** |
| One-head nanoGPT / FineWeb-Edu | Pinned FineWeb-Edu `sample-10BT`, exact document-disjoint 80M/1M/1M GPT-2-BPE splits, byte/hash verification | nanoGPT initialization and AdamW, reference Muon partition, update-level warm-up/cosine schedules, common fixed probes, approximately 80M-token budget | **Corrected and qualified for validation search** |
| nanochat d12 | Pinned upstream nanochat data/tokenizer pipeline | Native d12 initialization, Muon/AdamW groups, scaling rules, warm-up, warmdown, momentum, and weight-decay schedules | **Accepted as pinned CUDA reference** |
| nanochat mac_d4 | Separately cached reduced upstream preparation | Same pinned model and optimizer mathematics, eager non-CUDA runtime, device preflight and locked compile/fallback policy | **Accepted as separate development profile** |

## 1. MNIST / MLP3

### Data and initialization

The baseline uses the official torchvision MNIST training and test sets with
the conventional normalization:

```text
mean = 0.1307
std  = 0.3081
```

The official training set is split once into 55,000 optimization examples and
5,000 validation examples. The official 10,000-example test set is
monitoring-only.

The MLP initialization is explicit rather than inherited from the generic
`nn.Linear` default:

```text
fc1, fc2: Kaiming uniform, fan-in, ReLU gain
fc3:      Xavier uniform
biases:   zero
```

The dataset/model pair is appropriate for a cheap dense-network control. It is
not meant to establish broad architectural generality.

### Optimizer recipes

All three runs use 30 epochs, batch size 128, gradient clipping at 1.0, a short
update-level linear warm-up, and cosine decay to a nonzero floor.

| Optimizer | Peak LR | Floor | Warm-up | Other settings |
|---|---:|---:|---:|---|
| SGD + Nesterov | 0.05 | 5e-4 | 2 epochs | momentum 0.90, matrix-only weight decay 1e-4 |
| AdamW | 1e-3 | 1e-5 | 1 epoch | betas (0.90, 0.999), matrix-only weight decay 1e-2 |
| Muon matrices | 0.02 | 0.002 | 2 epochs | momentum 0.95, Nesterov, 5 Newton-Schulz steps, weight decay 0.01 |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 2 epochs | betas (0.90, 0.95), weight decay 0.01 on auxiliary matrices only |

The historical result key `sgd_momentum_muon` remains in file paths so old
stores stay discoverable, but the implementation is **Muon + auxiliary AdamW**.
Muon acts on the two hidden matrices; the classifier and biases use AdamW.

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

### Final model/training recipe

The selected small ViT remains:

```text
patches: 4x4
width: 192
blocks: 6
heads: 3
MLP ratio: 4
LayerNorm epsilon: 1e-6
patch projection: Conv2d fan-in initialization
```

The reference uses 300 epochs and the full DeiT-style regularization stack:

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

Weak regularization can change both generalization and gradient spectral
structure, so an optimizer comparison should not be built on an
under-regularized ViT.

### Final optimizer schedules

All profiles use explicit low-LR warm-up, cosine decay, and a final ten-epoch
cooldown at the nonzero floor.

| Optimizer | Warm-up start | Peak LR | Floor | Warm-up | Other settings |
|---|---:|---:|---:|---:|---|
| SGD + Nesterov | 1e-3 | 0.10 | 0.001 | 5 epochs | momentum 0.90, weight decay 5e-4 |
| AdamW | 1e-6 | 1.25e-4 | 1e-5 | 5 epochs | betas (0.90, 0.999), weight decay 0.05 |
| Muon matrices | 2e-4 | 0.02 | 0.002 | 5 epochs | momentum 0.95, Nesterov, 5 Newton-Schulz steps, weight decay 0.01 |
| Muon auxiliary AdamW | 3e-6 | 3e-4 | 3e-5 | 5 epochs | betas (0.90, 0.95), weight decay 0.01 |

The AdamW center is the DeiT reference LR `5e-4` linearly scaled from effective
batch 512 to the committed effective batch 128:

```text
5e-4 * 128 / 512 = 1.25e-4
```

The qualification grid also tests larger AdamW peaks through the unscaled
reference value. Checkpoints include model, optimizer, data-loader generator,
Python/NumPy/Torch/CUDA/MPS RNG state, protocol fingerprint, and best validation
loss.

Primary references:

- DeiT training recipe:
  <https://github.com/facebookresearch/deit/blob/main/main.py>
- Muon reference implementation:
  <https://github.com/KellerJordan/Muon>

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
train:      80,000,000 tokens
validation:  1,000,000 tokens
test:        1,000,000 tokens
```

The GPT-2 BPE tokenizer is retained for comparability and simple decoding of the
continuation BLEU probe. The 50,257-token tied embedding/head is large relative
to a one-block transformer and is therefore explicitly included or excluded in
any parameter-scaling statement. Muon is applied only to the six hidden
transformer matrices; every optimizer arm shares the identical embedding.

The approximately 80M sampled-token budget is close to 12 tokens per scaling
parameter when the tied embedding/head is counted once. Training draws random
contiguous windows from an 80M-token corpus, so the budget is corpus-equivalent
rather than an exact sequential pass.

### Optimizer recipes

| Optimizer | Peak LR | Floor | Warm-up |
|---|---:|---:|---:|
| SGD + Nesterov | 0.05 | 0.005 | 10% |
| AdamW | 6e-4 | 6e-5 | 1% |
| Muon matrices | 0.02 | 0.002 | 5% |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 5% |

AdamW uses `(0.9, 0.95)` betas, weight decay `0.1`, gradient clipping at `1.0`,
and nanoGPT initialization. Muon uses the reference hidden-matrix partition,
momentum `0.95`, Nesterov, five Newton-Schulz steps, and auxiliary AdamW.

Validation/test probes use 64 common fixed batches across every optimizer and
training seed. BLEU uses 64 common held-out continuations. LR logs identify the
update that produced each checkpoint. The run is restartable, validation loss
selects the best checkpoint, and the test set is monitoring-only.

Primary references:

- nanoGPT: <https://github.com/karpathy/nanoGPT>
- Muon: <https://github.com/KellerJordan/Muon>

## 4. nanochat

The d12 control is the **native upstream nanochat reference**, pinned to:

```text
92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
```

The wrapper does not replace upstream architecture or optimization mathematics.
It preserves:

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

CUDA keeps compilation for the model and both fused optimizer kernels. MPS/CPU
uses the same model and optimizer mathematics in eager mode, because those
compiled paths are not part of the portable non-CUDA contract. MPS fallback is
explicitly recorded. A real pinned model/optimizer preflight must pass on the
target device before a long run.

This recipe is accepted because nanochat performs its own depth/batch transfer,
parameter grouping, initialization, LR/momentum/weight-decay scheduling, data
packing, and tokenizer preparation. Updating the pinned nanochat commit or
runtime policy creates a new baseline version and requires a new audit.

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
to keep unsupported SVD/RMT operations off Apple MPS. Randomized analysis must
restore all available training RNG streams.

## What “optimized” means here

The committed defaults are source-backed, internally consistent center
candidates. They still require actual target-hardware validation qualification
before any “best baseline” claim. A genuine hyperparameter optimum is empirical
and must be selected on validation data with the preregistered bounded search in
`FINAL_BASELINE_QUALIFICATION.md`; it cannot be inferred solely from literature
defaults. The protected test set remains untouched by that search.
