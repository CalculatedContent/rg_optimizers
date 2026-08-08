# One-head nanoGPT optimizer baselines

This experiment is the smallest realistic language-model baseline in
`CalculatedContent/rg_optimizers`. It was adapted from the restart, data,
measurement, and multi-seed conventions in
`CalculatedContent/nanogpt-experiments`, but is isolated here so RG optimizer
variants can use it as a clean control.

It trains the same **one-block, one-attention-head nanoGPT** with:

1. **SGD + Nesterov momentum**;
2. **AdamW**;
3. **Muon on hidden transformer matrices + auxiliary AdamW**.

The experiment does **not** use Tiny Shakespeare. It streams a pinned revision
of FineWeb-Edu, creates exact document-disjoint train/validation/test splits,
and tokenizes them with GPT-2 BPE.

## Reference protocol

| Component | Value |
|---|---:|
| Dataset | `HuggingFaceFW/fineweb-edu`, `sample-10BT` |
| Dataset revision | `593b3a867298afb8ce42625a270ef20ddcad28f9` |
| Tokenizer | GPT-2 BPE, vocabulary 50,257 |
| Training split | 10,000,000 tokens |
| Validation split | 1,000,000 tokens |
| Test split | 1,000,000 tokens |
| Split construction | Document-disjoint |
| Transformer blocks | **1** |
| Attention heads | **1** |
| Embedding width | 128 |
| Context length | 256 |
| MLP width | 512 |
| Dropout | 0.0 |
| Bias | false |
| Token embedding / LM head | tied |
| Micro-batch | 4 sequences |
| Gradient accumulation | 8 |
| Tokens per optimizer step | 8,192 |
| Target training horizon | 5 passes over the fixed train split |
| Optimizer steps | 6,104 |
| Seeds | 1337, 2027, 4099 |
| Preferred device | Apple MPS |
| Numerical precision | float32 |

The model uses nanoGPT-style `N(0, 0.02)` initialization and scales the two
residual-output matrices by `1/sqrt(2 * n_layer)`. With one block, WeightWatcher
sees exactly six matrices:

```text
W_Q, W_K, W_V, W_O, W_MLP_IN, W_MLP_OUT
```

The GPT-2 token embedding is large relative to the one-block transformer.
Parameter-scaling claims must separately report or exclude embedding parameters.
This does not invalidate the matched optimizer comparison because every arm
shares the identical embedding and Muon acts only on the six hidden matrices.

## Corpus integrity

The preparation step writes exact split sizes and records, for every token file:

- dataset name, configuration, split, and pinned revision;
- tokenizer and dtype;
- document-disjoint status;
- exact byte count; and
- SHA-256.

Every training run verifies all of those fields before opening the memory maps.
A modified, truncated, old hashless, or otherwise corrupt cache fails visibly;
it is never silently reused. Recreate such a cache explicitly with the data
preparation command and `--force`.

## Optimizer profiles

The profiles are intentionally optimizer-specific. Forcing the same numerical
learning rate across SGD, AdamW, and Muon would not be a meaningful control.

| Optimizer | Peak LR | LR floor | Warm-up | Schedule | Decay / momentum |
|---|---:|---:|---:|---|---|
| SGD + Nesterov | 0.05 | 0.005 | 10% | step-level linear warm-up + cosine decay | momentum 0.90, weight decay 0.01 |
| AdamW | 6e-4 | 6e-5 | 1% | step-level linear warm-up + cosine decay | betas (0.90, 0.95), weight decay 0.10 |
| Muon matrices | 0.02 | 0.002 | 5% | step-level linear warm-up + cosine decay | momentum 0.95, Nesterov, 5 Newton-Schulz steps, weight decay 0.01 |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 5% | same progress as Muon | betas (0.90, 0.95), weight decay 0.01 |

AdamW follows the canonical nanoGPT pretraining values. Muon follows the
reference partition: only hidden two-dimensional transformer matrices use
Muon; embeddings, tied output parameters, normalization gains, and other
non-Muon parameters use AdamW. These are strong preregistered reference
settings, not a claim that a finite grid search proves global optimality.

## WeightWatcher contract

At epoch zero and every nominal epoch, the code copies the six transformer
matrices to CPU and calls:

```python
watcher.analyze(
    ERG=True,
    randomize=True,
    plot=False,
    min_evals=20,
)
```

The raw result is saved. Required direct outputs include:

- per-matrix `alpha`;
- per-matrix `ERG_gap`;
- per-matrix `num_traps` from randomized MP diagnostics;
- `detX_num`, `num_pl_spikes`, fit distance `D`, stable rank, MP soft rank,
  spectral/log norms, entropy, and all other returned WeightWatcher columns.

There is no fallback alpha, proxy trap count, or synthesized ERG gap. A
WeightWatcher version that does not return `alpha`, `ERG_gap`, and `num_traps`
fails visibly.

## Metrics and plots

Each nominal epoch records:

```text
train / validation / test cross-entropy
train / validation / test perplexity
train / validation / test next-token top-1 accuracy
fixed-continuation test BLEU
validation and test generalization gaps
primary and auxiliary learning rates
gradient norms
weight norm and update-to-weight ratio
MPS memory usage
```

BLEU is a deterministic secondary diagnostic. For 16 fixed held-out test
segments, the model receives a 64-token prompt and greedily predicts the next
32 tokens. Corpus BLEU compares those continuations with the exact held-out
continuations. It is **not** a translation benchmark and does not replace
cross-entropy or perplexity.

The notebooks plot individual seed trajectories, the across-seed mean, and a
two-sided **95% Student-t confidence interval**. Matrix plots use one invariant
color map for `W_Q`, `W_K`, `W_V`, `W_O`, `W_MLP_IN`, and `W_MLP_OUT`.

## Checkpoints and restart behavior

Every run writes:

```text
results/<optimizer>/seed_<seed>/
  manifest.json
  metrics.csv
  epoch_metrics.csv
  checkpoint_latest.pt
  checkpoint_best.pt
  checkpoint_final.pt
  epoch_checkpoints/
  spectral/
    layers.csv
    summary.csv
    raw/weightwatcher_step_*.csv
  test_results.json
  run_complete.json
```

`checkpoint_latest.pt` contains the model, optimizer state, data-sampling
generator, Python/NumPy/Torch RNG state, CUDA RNG state where applicable, MPS
RNG state where the installed PyTorch exposes it, elapsed time, validation-best
state, and a protocol fingerprint. A mismatched config, verified data identity,
optimizer, or seed is rejected rather than silently resumed. Completed runs are
skipped.

Test measurements are monitoring-only. Validation loss selects
`checkpoint_best.pt`; test loss, test accuracy, test perplexity, and BLEU never
change optimizer updates, schedules, stopping, or checkpoint selection.

## MacBook MPS workflow

From the repository root:

```bash
cd baseline/nanogpt_one_head
bash scripts/setup_mac.sh
bash scripts/prepare_data.sh
bash scripts/smoke_test.sh

export RG_NANOGPT_ONE_HEAD_ROOT="$HOME/rg-nanogpt-one-head"
caffeinate -dimsu bash scripts/run_all_baselines.sh \
  2>&1 | tee "$RG_NANOGPT_ONE_HEAD_ROOT/run_all.log"
```

The setup script reports whether MPS is built and available. When available,
`--device auto` selects it. Unsupported individual MPS operations may use
PyTorch CPU fallback because `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the
scripts. WeightWatcher always runs on CPU copies of the six matrices.

The first data-preparation run requires internet access. Later runs reuse only a
fully verified corpus under:

```text
$RG_NANOGPT_ONE_HEAD_ROOT/data
```

## Notebook order

```text
notebooks/01_sgd_momentum_baseline.ipynb
notebooks/02_adamw_baseline.ipynb
notebooks/03_muon_baseline.ipynb
notebooks/04_compare_baselines.ipynb
```

The first three notebooks run or resume their optimizer's three seeds. The
comparison notebook requires all nine completed runs and produces optimizer
overlays plus final and validation-selected 95% confidence-interval tables.

Launch Jupyter with:

```bash
.venv-one-head/bin/jupyter lab notebooks
```

## Smaller development runs

The committed `configs/reference.yaml` defines the scientific reference.
Temporary pilots belong in a separate YAML file and a separate output root.

Run one optimizer:

```bash
.venv-one-head/bin/python -m rg_nanogpt_one_head.training \
  --config configs/reference.yaml \
  --optimizer adamw \
  --device auto
```

Run one seed:

```bash
.venv-one-head/bin/python -m rg_nanogpt_one_head.training \
  --config configs/reference.yaml \
  --optimizer muon \
  --seeds 1337 \
  --device auto
```

## Validation

```bash
bash scripts/smoke_test.sh
```

The test suite checks the architecture, all optimizer update paths, exact
split-writing and cache corruption detection, checkpoint round-tripping,
Student-t intervals, direct `ERG_gap`/`num_traps` handling, tiny CPU training,
and notebook structure. The same tests run in the repository's baseline CI.
