# NGB v4 nanoGPT baselines

`baseline/ngb` is a separate experiment family for properly tuned small-language-model optimizer baselines. It does **not** reuse the v3 one-head result directories and does not alter the checked-in v3 protocol or outputs.

All data, checkpoints, plots, and results default to:

```text
/tmp/rg-ngb
```

No shell wrappers or project virtual environments are required. Use the currently activated conda environment.

## Protocols

| Configuration | Architecture | Horizon | Purpose |
|---|---|---:|---|
| `configs/v4_one_head.yaml` | 1 block, 1 head, width 128, context 256 | 2 corpus-equivalent epochs | Direct tuned successor to the v3 diagnostic model |
| `configs/v4_small_4x4.yaml` | 4 blocks, 4 heads, width 128, context 256 | 2 corpus-equivalent epochs | Distinct small but substantially more expressive language-model baseline |

Both protocols use the same pinned, document-disjoint FineWeb-Edu 80M / 1M / 1M GPT-2-BPE corpus contract. The prepared corpus can be shared, while protocol fingerprints and result directories remain separate.

## Tuned optimizer centers

The v4 centers respond directly to the v3 instability observed in the checked-in comparison.

### One-head v4

| Optimizer | Peak LR | Floor | Warm-up | Decay |
|---|---:|---:|---:|---:|
| SGD + Nesterov | `5e-2` | `5e-4` | 5% | `1e-2` |
| AdamW | `3e-4` | `1e-5` | 2.5% | `1e-1` |
| Muon matrices | `1e-2` | `2e-4` | 5% | `2e-2` |
| Muon auxiliary AdamW | `3e-4` | `1e-5` | 5% | `1e-1` |

### Small 4x4 v4

The 4x4 architecture uses the same adaptive centers. Its SGD center is reduced to `3e-2` because four residual blocks produce a materially different optimization geometry.

These are preregistered v4 centers, not a claim that a broad hyperparameter search has already been completed. Validation loss remains the only checkpoint-selection and qualification objective.

## Install in the active conda environment

From the repository root:

```bash
cd baseline/ngb
python -m pip install -e ../nanogpt_one_head
python -m pip install -e '.[dev]'
```

The first editable install supplies shared, already-tested data, evaluation, checkpoint, runtime, and optimizer primitives. NGB owns its generalized model, protocol, spectral inventory, completion validation, result directories, and comparison logic.

## Prepare the corpus

```bash
export RG_NGB_ROOT="/tmp/rg-ngb"
export PYTORCH_ENABLE_MPS_FALLBACK=1

ngb-prepare --config configs/v4_one_head.yaml
```

The 4x4 protocol uses the same verified token files, so no second corpus download is required.

## Run the tuned one-head v4 experiment

```bash
ngb-train \
  --config configs/v4_one_head.yaml \
  --optimizer all \
  --device auto
```

## Run the distinct 4-layer / 4-head experiment

```bash
ngb-train \
  --config configs/v4_small_4x4.yaml \
  --optimizer all \
  --device auto
```

The default canonical qualification seeds are `1337, 2027, 4099`. An explicit matched expansion can be run with, for example:

```bash
ngb-train \
  --config configs/v4_one_head.yaml \
  --optimizer all \
  --seeds 1337,2027,4099,5003,6007,7013,8017,9011 \
  --device auto
```

Completed compatible runs are verified and reused. Stale or protocol-incompatible directories fail visibly rather than being silently accepted.

## Notebooks

```text
notebooks/01_v4_one_head_train.ipynb
notebooks/02_v4_one_head_compare.ipynb
notebooks/11_v4_small_4x4_train.ipynb
notebooks/12_v4_small_4x4_compare.ipynb
```

Run with the active conda kernel:

```bash
papermill notebooks/01_v4_one_head_train.ipynb notebooks/01_v4_one_head_train.out.ipynb -k python3
papermill notebooks/02_v4_one_head_compare.ipynb notebooks/02_v4_one_head_compare.out.ipynb -k python3
```

The comparison notebooks discover the complete seed intersection shared by all three optimizers. They report run-level 95% Student-t intervals, matched-seed differences, validation-selected and final metrics, best-step drift, clipping frequency, maximum update-to-weight ratio, and WeightWatcher trajectories. Perplexity intervals are obtained by exponentiating the loss-space confidence interval, so they cannot have impossible negative lower bounds.

## Validation

```bash
python -m pytest -q tests
```

The tests cover both architectures, exact protocol settings, dynamic matrix inventories, `/tmp` path isolation, matched-seed discovery, transformed perplexity intervals, optimizer partitioning, notebook syntax, and a tiny CPU training path.
