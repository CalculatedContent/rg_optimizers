# RG optimizer baselines

This directory contains the **unmodified reference experiments** used to test
RG-motivated optimizer extensions. These controls do not apply trace-log
projection, ECS correction, WW-PGD retraction, spectral-flow subtraction, or
any other RG intervention.

## Baseline suite

| Baseline | Dataset / corpus | Reference model | Optimizer controls | Main entry point |
|---|---|---|---|---|
| **MNIST / MLP3** | MNIST | `784 -> 512 -> 512 -> 10` MLP | SGD + momentum, AdamW, SGD + momentum + Muon | [`notebooks/MNIST_MLP3_Baseline_Comparison.ipynb`](notebooks/MNIST_MLP3_Baseline_Comparison.ipynb) |
| **CIFAR-10 / small ViT** | CIFAR-10 | 4x4 patches, width 192, 6 transformer blocks, 3 heads | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb`](notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb) |
| **One-head nanoGPT / FineWeb-Edu** | Pinned FineWeb-Edu `sample-10BT`, document-disjoint GPT-2-BPE splits | **1 transformer block, 1 attention head**, width 128, context 256 | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`nanogpt_one_head/README.md`](nanogpt_one_head/README.md) |
| **nanochat d12** | nanochat miniseries corpus | 12 layers, width 768, context 2048 | Native nanochat Muon + AdamW recipe | [`notebooks/NanoChat_D12_Reference_Baseline.ipynb`](notebooks/NanoChat_D12_Reference_Baseline.ipynb) |

The one-head nanoGPT row is the smallest realistic language-model control in
this repository. It is intentionally separate from nanochat: the one-head suite
is cheap enough for optimizer debugging and three-seed comparisons on a
MacBook, while nanochat d12 is the stronger modern language-model reference.

## Shared experimental conventions

Unless an experiment-specific README says otherwise:

- the unit of replication is a complete training run;
- optimizer comparisons use independent seeds and matched architecture, data,
  evaluation probes, and training budgets;
- optimizer-specific learning rates and schedules are allowed because SGD,
  AdamW, and Muon have different update geometries;
- test measurements are monitoring-only and are never used for optimizer
  updates, early stopping, learning-rate changes, or checkpoint selection;
- WeightWatcher values are retained as returned rather than replaced with
  fallback alpha values, proxy trap counts, or fabricated ERG gaps;
- trajectory plots show individual runs plus across-seed uncertainty, and final
  summaries use two-sided 95% Student-t intervals when three seeds are present.

## Persistent output roots

The original MNIST, ViT, and nanochat notebooks use:

```text
RG_BASELINE_RUN_ROOT, when set
otherwise: baseline/runs/
```

Their shared data/cache root can be changed with:

```text
RG_BASELINE_DATA_DIR
```

The one-head nanoGPT suite is intentionally isolated and uses:

```text
RG_NANOGPT_ONE_HEAD_ROOT, when set
otherwise: baseline/nanogpt_one_head/runs/
```

A persistent location under `$HOME` is recommended for long MacBook runs:

```bash
export RG_BASELINE_RUN_ROOT="$HOME/rg-optimizer-baselines"
export RG_NANOGPT_ONE_HEAD_ROOT="$HOME/rg-nanogpt-one-head"
```

---

## 1. MNIST / MLP3

The matched MLP3 notebooks are:

1. [`MNIST_MLP3_SGD_Momentum_Baseline.ipynb`](notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb)
2. [`MNIST_MLP3_AdamW_Baseline.ipynb`](notebooks/MNIST_MLP3_AdamW_Baseline.ipynb)
3. [`MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`](notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb)
4. [`MNIST_MLP3_Baseline_Comparison.ipynb`](notebooks/MNIST_MLP3_Baseline_Comparison.ipynb)

All runs use:

```text
784 -> 512 -> 512 -> 10
ReLU after fc1 and fc2
MNIST normalization: mean 0.1307, std 0.3081
seeds: 1337, 2027, 31415
```

The notebooks record train/test loss and accuracy, gradient and parameter
norms, complete epoch checkpoints, layerwise WeightWatcher alpha and ERG
metrics, midpoint trace-log diagnostics, effective-rank measures, energy
fractions, and reproducibility manifests. Run the three optimizer notebooks
first, then the comparison notebook.

---

## 2. CIFAR-10 / small ViT

Run:

```text
notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb
```

The committed reference model is:

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

The notebook trains the identical model with SGD + Nesterov, AdamW, and Muon +
auxiliary AdamW. It uses optimizer-specific warm-up/cosine schedules, random
crop, horizontal flip, RandAugment, mixup, label smoothing, gradient clipping,
checkpoint persistence, WeightWatcher diagnostics, aggregate CSV files, and
95% confidence intervals.

---

## 3. One-head nanoGPT / FineWeb-Edu

The complete runbook is:

[`nanogpt_one_head/README.md`](nanogpt_one_head/README.md)

This experiment was adapted from the restart, data, measurement, and multi-seed
conventions in `CalculatedContent/nanogpt-experiments`, but is implemented as a
clean baseline inside this repository.

### Reference protocol

| Component | Value |
|---|---:|
| Dataset | `HuggingFaceFW/fineweb-edu`, `sample-10BT` |
| Dataset revision | `593b3a867298afb8ce42625a270ef20ddcad28f9` |
| Tokenizer | GPT-2 BPE, vocabulary 50,257 |
| Training split | 10,000,000 tokens |
| Validation split | 1,000,000 tokens |
| Protected test split | 1,000,000 tokens |
| Split construction | Document-disjoint |
| Transformer blocks | **1** |
| Attention heads | **1** |
| Embedding width | 128 |
| Context length | 256 |
| Dropout | 0.0 |
| Token embedding / LM head | Tied |
| Target horizon | 5 passes over the fixed training split |
| Seeds | 1337, 2027, 4099 |
| Preferred device | Apple MPS |

It is deliberately not a Tiny Shakespeare demonstration. FineWeb-Edu provides
a realistic language distribution while the one-block, one-head architecture
keeps the experiment small enough for repeated optimizer work on a MacBook.

### Optimizer reference profiles

| Optimizer | Peak LR | LR floor | Warm-up | Schedule | Other settings |
|---|---:|---:|---:|---|---|
| SGD + Nesterov | 0.05 | 0.005 | 10% | Cosine | momentum 0.90, weight decay 0.01 |
| AdamW | 6e-4 | 6e-5 | 1% | Cosine | betas (0.90, 0.95), weight decay 0.10 |
| Muon matrices | 0.02 | 0.002 | 5% | Cosine | momentum 0.95, Nesterov, 5 Newton-Schulz steps, weight decay 0.01 |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 5% | Same progress as Muon | betas (0.90, 0.95), weight decay 0.01 |

The AdamW values follow the canonical nanoGPT pretraining recipe. Muon is
restricted to hidden two-dimensional transformer matrices; embeddings, the
tied head, normalization parameters, and other auxiliary parameters use
AdamW. These are strong preregistered reference settings, not a claim that a
finite sweep proves a globally optimal point.

### WeightWatcher contract

At initialization and every nominal epoch, the suite copies the six transformer
matrices to CPU and calls:

```python
watcher.analyze(
    ERG=True,
    randomize=True,
    plot=False,
    min_evals=20,
)
```

The monitored matrices are:

```text
W_Q, W_K, W_V, W_O, W_MLP_IN, W_MLP_OUT
```

The raw and summarized outputs retain direct per-matrix `alpha`, `ERG_gap`, and
`num_traps`, along with `detX_num`, spike counts, fit distance, rank, norm, and
entropy fields. The strict reference configuration fails visibly when required
WeightWatcher outputs are unavailable; it does not invent substitutes.

### Metrics, notebooks, and error bars

The suite records per epoch:

```text
train / validation / test cross-entropy
train / validation / test perplexity
train / validation / test next-token accuracy
fixed held-out continuation BLEU
generalization gaps
learning rates and gradient norms
weight and update norms
throughput and MPS memory
per-matrix alpha, ERG_gap, and num_traps
```

BLEU is a deterministic held-out continuation-overlap diagnostic, not a
translation benchmark and not a replacement for loss or perplexity.

The notebooks are:

1. [`01_sgd_momentum_baseline.ipynb`](nanogpt_one_head/notebooks/01_sgd_momentum_baseline.ipynb)
2. [`02_adamw_baseline.ipynb`](nanogpt_one_head/notebooks/02_adamw_baseline.ipynb)
3. [`03_muon_baseline.ipynb`](nanogpt_one_head/notebooks/03_muon_baseline.ipynb)
4. [`04_compare_baselines.ipynb`](nanogpt_one_head/notebooks/04_compare_baselines.ipynb)

They plot individual seed trajectories, across-seed means, and two-sided 95%
Student-t confidence intervals. Layer plots use one fixed color-blind-safe map
for all six matrices.

### Restart and MacBook execution

Each run writes latest, best, final, and model-only epoch checkpoints.
`checkpoint_latest.pt` includes model and optimizer state, sampling RNG, Python,
NumPy and Torch RNG state, elapsed time, and a protocol fingerprint. Compatible
interrupted runs resume automatically; incompatible data, model, optimizer, or
configuration identities are rejected.

Run the complete suite with:

```bash
cd baseline/nanogpt_one_head
bash scripts/setup_mac.sh
bash scripts/prepare_data.sh
bash scripts/smoke_test.sh

export RG_NANOGPT_ONE_HEAD_ROOT="$HOME/rg-nanogpt-one-head"
caffeinate -dimsu bash scripts/run_all_baselines.sh \
  2>&1 | tee "$RG_NANOGPT_ONE_HEAD_ROOT/run_all.log"
```

`--device auto` selects Apple MPS when available. WeightWatcher analyzes CPU
copies so its SVD/RMT path does not depend on MPS linear-algebra support.

---

## 4. nanochat d12 reference

Run:

```text
notebooks/NanoChat_D12_Reference_Baseline.ipynb
```

The notebook pins upstream nanochat commit:

```text
92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
```

It preserves nanochat's native d12 architecture, initialization, Muon/AdamW
parameter partitioning, separate embedding/unembedding/matrix/scalar learning
rates, depth- and batch-aware scaling rules, 40-step warm-up, long linear
warmdown, Muon momentum schedule, and cosine-decayed cautious weight decay.

The reference scale is:

```text
layers: 12
width: 768
context: 2048
training target: 12 tokens per scaling parameter
seeds: 17, 29, 43
```

The wrapper adds reproducible independent seeds, periodic checkpoints and
validation, final CORE evaluation, tidy CSV logs, and offline WeightWatcher
analysis so spectral diagnostics do not perturb timed training.

---

## Recommended benchmark ladder

Use the suite as a progression rather than treating one model as sufficient:

```text
1. MNIST / MLP3
   fast dense-network and spectral debugging

2. One-head nanoGPT / FineWeb-Edu
   cheapest realistic autoregressive language-model control

3. CIFAR-10 / small ViT
   transformer optimization on image data

4. nanochat d12
   stronger modern language-model reference
```

A new RG optimizer should be compared against the appropriate strong control
for each architecture, using the same seeds, data identity, evaluation policy,
and training budget.

## Tests

For the original baseline package:

```bash
cd baseline
PYTHONPATH=. python -m unittest discover -s tests -v
```

For the one-head nanoGPT package:

```bash
cd baseline/nanogpt_one_head
PYTHONPATH=src pytest -q
```
