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
| Protocol version | 3 |
| Dataset | `HuggingFaceFW/fineweb-edu`, `sample-10BT` |
| Dataset revision | `593b3a867298afb8ce42625a270ef20ddcad28f9` |
| Tokenizer | GPT-2 BPE, vocabulary 50,257 |
| Training split | **80,000,000 tokens** |
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
| Target training budget | **about 80M sampled tokens** |
| Optimizer steps | **9,766** |
| Reporting / WW checkpoints | 0, 0.125, ..., 1.0 corpus-equivalent budget |
| Validation probe | 64 common fixed batches |
| Test probe | 64 common fixed batches |
| BLEU probe | 64 common held-out continuations |
| Seeds | 1337, 2027, 4099 |
| Preferred device | Apple MPS |
| Numerical precision | float32 |

The model has roughly 6.6 million scaling parameters when the tied vocabulary
embedding/head is counted once. An approximately 80M-token budget is therefore
close to a 12-tokens-per-parameter reference regime. The training sampler draws
random contiguous windows from an 80M-token corpus, so `1.0` is a
**corpus-equivalent token budget**, not a claim that every token is visited
exactly once. This is stronger than repeatedly sampling an undersized 10M-token
corpus to manufacture the same processed-token count.

Validation loss selects the best checkpoint, so the full horizon provides room
to converge without forcing the final checkpoint to be reported.

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
A modified, truncated, old hashless, undersized, or otherwise incompatible
cache fails visibly; it is never silently reused. Recreate such a cache
explicitly with the data-preparation command and `--force`.

## Optimizer profiles

The profiles are intentionally optimizer-specific. Forcing the same numerical
learning rate across SGD, AdamW, and Muon would not be a meaningful control.

| Optimizer | Peak LR | LR floor | Warm-up | Schedule | Decay / momentum |
|---|---:|---:|---:|---|---|
| SGD + Nesterov | 0.05 | 0.005 | 10% | update-level linear warm-up + cosine decay | momentum 0.90, weight decay 0.01 |
| AdamW | 6e-4 | 6e-5 | 1% | update-level linear warm-up + cosine decay | betas (0.90, 0.95), weight decay 0.10 |
| Muon matrices | 0.02 | 0.002 | 5% | update-level linear warm-up + cosine decay | momentum 0.95, Nesterov, 5 Newton-Schulz steps, weight decay 0.01 |
| Muon auxiliary AdamW | 3e-4 | 3e-5 | 5% | same progress as Muon | betas (0.90, 0.95), weight decay 0.01 |

AdamW follows the canonical nanoGPT pretraining values. Muon follows the
reference partition: only hidden two-dimensional transformer matrices use
Muon; embeddings, tied output parameters, normalization gains, and other
non-Muon parameters use AdamW.

The CSV `primary_lr` and `auxiliary_lr` values are the learning rates used by
the update that produced the recorded checkpoint. At step zero they are zero,
because no optimizer update has yet occurred. The console also displays the LR
prepared for the next update, preventing an off-by-one interpretation of the
warm-up curve.

These settings are the committed centers of the bounded validation search in
[`../FINAL_BASELINE_QUALIFICATION.md`](../FINAL_BASELINE_QUALIFICATION.md). They
become the frozen best baseline only after validation qualification; protected
test metrics never choose a candidate.

## Common evaluation probes

All optimizer arms and all training seeds use the exact same preregistered probe
windows:

```text
train probe seed:       21001
validation probe seed:  22001
test probe seed:        23001
BLEU probe seed:        24001
```

This removes evaluation-sample drift from paired optimizer comparisons. A model
seed changes initialization and training batches; it does not change the
validation or test examples on which that run is measured.

## WeightWatcher contract

At the initial state and at eight evenly spaced corpus-equivalent checkpoints,
the code copies the six transformer matrices to CPU and calls:

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
WeightWatcher version that does not return finite `alpha`, `ERG_gap`, and
`num_traps` fails visibly. Python, NumPy, CPU Torch, CUDA, and MPS RNG state are
restored after randomized analysis.

## Metrics and plots

Each nominal reporting checkpoint records:

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

BLEU is a deterministic secondary diagnostic. For 64 fixed held-out test
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

## Conda / local workflow

Use the currently activated conda environment. Do **not** create a project venv
and do not run setup wrapper scripts.

From the repository root:

```bash
cd baseline/nanogpt_one_head

# Install this package and any missing dependencies into the active conda env.
python -m pip install -e .

# Keep all corpus caches and experiment outputs outside the git checkout.
export RG_NANOGPT_ONE_HEAD_ROOT="$HOME/rg-nanogpt-one-head"

# Allow unsupported individual MPS operations to fall back to CPU when needed.
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Verify the active Python and accelerator before a long run:

```bash
which python
python -c "import torch; print(torch.__version__); print('MPS built:', torch.backends.mps.is_built()); print('MPS available:', torch.backends.mps.is_available())"
```

### Prepare the pinned FineWeb-Edu corpus

The first preparation requires internet access. Later runs reuse only the fully
verified cache under `$RG_NANOGPT_ONE_HEAD_ROOT/data`.

```bash
rg-onehead-prepare --config configs/reference.yaml
```

To deliberately replace an incompatible or stale cache:

```bash
rg-onehead-prepare --config configs/reference.yaml --force
```

### Run the three reference baselines directly

Each command runs or resumes all three canonical seeds.

```bash
python -m rg_nanogpt_one_head.training \
  --config configs/reference.yaml \
  --optimizer sgd_momentum \
  --device auto

python -m rg_nanogpt_one_head.training \
  --config configs/reference.yaml \
  --optimizer adamw \
  --device auto

python -m rg_nanogpt_one_head.training \
  --config configs/reference.yaml \
  --optimizer muon \
  --device auto
```

On an Apple Silicon Mac, `--device auto` selects MPS when available.
WeightWatcher measurements are performed on CPU copies of the six matrices.

### Build the comparison output notebook

After all nine runs are complete:

```bash
cd notebooks
papermill 04_compare_baselines.ipynb 04_compare_baselines.out.ipynb
```

The notebook requires all three optimizers × all three seeds and produces
optimizer overlays plus final and validation-selected 95% confidence-interval
tables.

## Notebook order

```text
notebooks/01_sgd_momentum_baseline.ipynb
notebooks/02_adamw_baseline.ipynb
notebooks/03_muon_baseline.ipynb
notebooks/04_compare_baselines.ipynb
```

The first three notebooks are interactive entry points for the same training
runtime. The direct Python commands above are the simplest path for unattended
local runs.

To launch Jupyter from the active conda environment:

```bash
jupyter lab notebooks
```

## Smaller development runs

The committed `configs/reference.yaml` defines the scientific reference.
Temporary pilots belong in a separate YAML file and a separate output root.

Run one optimizer:

```bash
python -m rg_nanogpt_one_head.training \
  --config configs/reference.yaml \
  --optimizer adamw \
  --device auto
```

Run one seed:

```bash
python -m rg_nanogpt_one_head.training \
  --config configs/reference.yaml \
  --optimizer muon \
  --seeds 1337 \
  --device auto
```

## Validation

No shell wrapper is required. From `baseline/nanogpt_one_head` run:

```bash
python -m pytest -q tests
```

The test suite checks the architecture, 80M-token corpus contract, all optimizer
update paths, common probe identity, exact split-writing and cache corruption
detection, checkpoint round-tripping, LR logging semantics, Student-t
intervals, direct `ERG_gap`/`num_traps` handling, tiny CPU training, and
notebook structure. The same tests run in the repository's baseline CI.
