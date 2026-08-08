# RG optimizer baselines

This directory contains the **unmodified reference experiments** used to test
RG-motivated optimizer extensions. These controls do not apply trace-log
projection, ECS correction, WW-PGD retraction, spectral-flow subtraction, or
any other RG intervention.

The complete data/hyperparameter/schedule review is recorded in
[`BASELINE_RECIPE_AUDIT.md`](BASELINE_RECIPE_AUDIT.md).

## Baseline suite

| Baseline | Dataset / corpus | Reference model | Optimizer controls | Main entry point |
|---|---|---|---|---|
| **MNIST / MLP3** | MNIST | `784 -> 512 -> 512 -> 10` MLP | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`notebooks/MNIST_MLP3_Baseline_Comparison.ipynb`](notebooks/MNIST_MLP3_Baseline_Comparison.ipynb) |
| **CIFAR-10 / small ViT** | CIFAR-10 with fixed 45k/5k train/validation split | 4x4 patches, width 192, 6 blocks, 3 heads | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb`](notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb) |
| **One-head nanoGPT / FineWeb-Edu** | Pinned FineWeb-Edu `sample-10BT`, document-disjoint 10M/1M/1M GPT-2-BPE splits | 1 block, 1 attention head, width 128, context 256 | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`nanogpt_one_head/README.md`](nanogpt_one_head/README.md) |
| **nanochat d12** | Native nanochat miniseries corpus/tokenizer pipeline | 12 layers, width 768, context 2048 | Native nanochat Muon + AdamW recipe | [`notebooks/NanoChat_D12_Reference_Baseline.ipynb`](notebooks/NanoChat_D12_Reference_Baseline.ipynb) |

## Shared scientific contract

Unless an experiment-specific README says otherwise:

- the unit of replication is a complete training run;
- all optimizer arms share the same architecture, data identities, seed set,
  evaluation probes, and training budget;
- optimizer-specific learning rates and schedules are allowed because SGD,
  AdamW, and Muon have different update geometries;
- validation data select hyperparameters and best checkpoints;
- test measurements are monitoring-only and never alter optimization,
  scheduling, early stopping, or checkpoint selection;
- WeightWatcher values are stored as returned, with no fallback alpha, proxy
  correlation-trap count, or fabricated ERG gap;
- three-seed summaries use two-sided 95% Student-t intervals;
- “optimized” means a strong source-backed reference recipe. A claim of a true
  optimum still requires an explicit validation-only search on the target
  hardware.

## Persistent output roots

MNIST, ViT, and nanochat use `RG_BASELINE_RUN_ROOT`; their shared data/cache root
can be changed with `RG_BASELINE_DATA_DIR`. The isolated one-head nanoGPT suite
uses `RG_NANOGPT_ONE_HEAD_ROOT`. For long MacBook runs, use persistent
directories under `$HOME` rather than `/tmp`.

## 1. MNIST / MLP3

Run the three optimizer notebooks, then the comparison notebook:

1. [`MNIST_MLP3_SGD_Momentum_Baseline.ipynb`](notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb)
2. [`MNIST_MLP3_AdamW_Baseline.ipynb`](notebooks/MNIST_MLP3_AdamW_Baseline.ipynb)
3. [`MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`](notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb)
4. [`MNIST_MLP3_Baseline_Comparison.ipynb`](notebooks/MNIST_MLP3_Baseline_Comparison.ipynb)

All runs use 30 epochs, batch size 128, three seeds, gradient clipping, complete
train/test evaluation, and WeightWatcher analysis at epoch zero and every
epoch.

| Optimizer | Peak LR | Floor | Warm-up | Other settings |
|---|---:|---:|---:|---|
| SGD + Nesterov | 0.05 | 5e-4 | 2 epochs | momentum 0.90, matrix-only WD 1e-4 |
| AdamW | 1e-3 | 1e-5 | 1 epoch | betas (0.90, 0.999), matrix-only WD 1e-2 |
| Muon matrices | 0.02 | 0.002 | 2 epochs | momentum 0.95, Nesterov, 5 NS steps, WD 0.01 |
| Auxiliary AdamW | 3e-4 | 3e-5 | 2 epochs | betas (0.90, 0.95), WD 0.01 on auxiliary matrices |

Every profile uses linear warm-up followed by cosine decay to the listed
nonzero floor. The historical result-directory key `sgd_momentum_muon` is
retained for compatibility, but the corrected implementation is **Muon +
auxiliary AdamW**.

## 2. CIFAR-10 / small ViT

Run `notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb`.

The official 50,000-example training set is deterministically divided into
45,000 optimization examples and 5,000 validation examples. The official test
set is protected from selection.

```text
patch size: 4x4
width: 192
blocks: 6
heads: 3
MLP ratio: 4
epochs: 300
replicates: 3 seeds
```

The recipe includes dropout 0, stochastic depth 0.10, RandAugment, color
jitter, random erasing, mixup 0.8, CutMix 1.0, label smoothing 0.1, gradient
clipping, and optimizer-specific warm-up/cosine schedules.

| Optimizer | Peak LR | Floor | Warm-up |
|---|---:|---:|---:|
| SGD + Nesterov | 0.10 | 0.001 | 5 epochs |
| AdamW | 1.25e-4 | 1e-5 | 5 epochs |
| Muon matrices | 0.02 | 0.002 | 5 epochs |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 5 epochs |

Validation loss selects the best checkpoint. The notebook plots
train/validation/test metrics, LR schedules, and layer-resolved WeightWatcher
alpha, ERG gap, and `num_traps`. `checkpoint_latest.pt` contains full restart
state.

## 3. One-head nanoGPT / FineWeb-Edu

See [`nanogpt_one_head/README.md`](nanogpt_one_head/README.md).

This MacBook-scale language baseline uses a pinned FineWeb-Edu corpus rather
than Tiny Shakespeare:

```text
train:      10,000,000 tokens
validation:  1,000,000 tokens
test:        1,000,000 tokens
split policy: document-disjoint
tokenizer: GPT-2 BPE
model: 1 block, 1 head, width 128, context 256
```

Its existing source-backed SGD, AdamW, and Muon schedules passed the audit. The
suite is restartable, selects best checkpoints on validation loss, and records
loss, perplexity, next-token accuracy, continuation BLEU, LR, gradient/update
norms, MPS memory, and per-matrix WeightWatcher metrics.

## 4. nanochat d12

Run `notebooks/NanoChat_D12_Reference_Baseline.ipynb`.

This is the native upstream nanochat reference pinned to commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`. The wrapper preserves upstream
architecture, initialization, tokenizer/data preparation, parameter groups,
scaling rules, and schedules. WeightWatcher runs offline on saved checkpoints.

## WeightWatcher contract

Strict reference runs call:

```python
watcher.analyze(ERG=True, randomize=True, ...)
```

Required direct outputs include layer `alpha`, `ERG_gap`, and randomized-MP
`num_traps`. Missing values fail visibly; no fallback or proxy is substituted.

## Recommended ladder

```text
MNIST / MLP3
  -> CIFAR-10 / small ViT
  -> One-head nanoGPT / FineWeb-Edu
  -> nanochat d12
```

## Tests

```bash
cd baseline
PYTHONPATH=. python -m unittest discover -s tests -v

cd nanogpt_one_head
PYTHONPATH=src pytest -q
```
