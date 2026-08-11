# NGB v4 — nanoGPT baselines

`baseline/ngb` is the version-4 nanoGPT experiment family. It is separate from
`baseline/nanogpt_one_head`, so the existing v3 one-epoch runs and checked-in
comparison notebook remain intact.

NGB contains two preregistered architectures:

1. **v4 one-head control** — one block, one attention head, width 128.
2. **v4 small 4×4 model** — four blocks, four attention heads, width 128.

Both use the same pinned, document-disjoint FineWeb-Edu corpus and the same
fixed train/validation/test/BLEU probes. Test metrics remain monitoring-only.
Validation cross-entropy selects `checkpoint_best.pt`.

## Why v4 exists

The v3 experiment used one corpus-equivalent epoch and optimizer profiles
transferred almost directly from much larger nanoGPT/Muon training regimes.
The checked-in comparison showed:

- SGD was reproducible but still improving near the one-epoch boundary.
- AdamW and Muon reached better validation-selected checkpoints but exhibited
  large seed-dependent final-checkpoint drift.
- The adaptive profiles used a high peak learning rate relative to the
  8,192-token effective batch.
- The one-head model was 96.55% tied token embedding / output head by parameter
  count, so Muon's auxiliary AdamW controlled most of the model.

NGB v4 therefore changes both the horizon and the optimizer centers rather than
merely doubling the old schedule.

## Protocols

| Field | v4 one-head | v4 small 4×4 |
|---|---:|---:|
| Blocks | 1 | 4 |
| Attention heads | 1 | 4 |
| Embedding width | 128 | 128 |
| Context length | 256 | 256 |
| Parameters | 6,662,656 | 7,253,248 |
| Muon matrix parameters | 196,608 | 786,432 |
| Training tokens | about 160M | about 160M |
| Corpus-equivalent epochs | 2.0 | 2.0 |
| Reporting interval | 0.25 epoch | 0.25 epoch |
| Default seeds | 1337, 2027, 4099 | 1337, 2027, 4099 |

The 4×4 architecture has only about 9% more total parameters than the one-head
control because both models are dominated by the tied GPT-2 vocabulary
embedding. It nevertheless has four times as many hidden transformer matrices
and materially greater contextual capacity.

## Tuned optimizer centers

### One-head v4

| Optimizer | Peak LR | Floor | Warm-up | Weight decay |
|---|---:|---:|---:|---:|
| SGD + Nesterov | 0.05 | 5e-4 | 5% | 0.01 |
| AdamW | 3e-4 | 1e-5 | 2.5% | 0.10 |
| Muon matrices | 0.01 | 1e-4 | 2.5% | 0.02 |
| Muon auxiliary AdamW | 2e-4 | 1e-5 | 2.5% | 0.10 |

### Small 4×4 v4

The adaptive profiles are the same. The SGD peak is reduced to `0.03`, with a
`3e-4` floor, for the deeper residual stack.

These are conservative v4 center profiles, not a claim that a validation-only
grid search has already proved global optimality. The protocol is designed to
train stably enough that the full two-epoch trajectory is scientifically
interpretable.

## Storage

No NGB command defaults to a home directory.

```text
/tmp/rg-ngb/
  results/
    v4_one_head/
    v4_small_4x4/
  plots/
    v4_one_head/
    v4_small_4x4/
```

The prepared v3 corpus can be reused directly:

```text
/tmp/rg-nanogpt-one-head/data
```

Set a different explicit `/tmp` path through `RG_NGB_DATA_ROOT` when needed.

## Install in the active conda environment

From the repository root:

```bash
cd /tmp/rg_optimizers/baseline/ngb
python -m pip install -e ../nanogpt_one_head
export PYTORCH_ENABLE_MPS_FALLBACK=1
export RG_NGB_ROOT=/tmp/rg-ngb
export RG_NGB_DATA_ROOT=/tmp/rg-nanogpt-one-head/data
```

Prepare or verify the shared corpus:

```bash
python -m rg_nanogpt_one_head.data \
  --config configs/v4_one_head.yaml \
  --output-dir /tmp/rg-nanogpt-one-head/data
```

## Run v4 one-head

```bash
python -m rg_nanogpt_one_head.training \
  --config configs/v4_one_head.yaml \
  --optimizer all \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-ngb/results/v4_one_head \
  --device auto
```

## Run the small 4×4 model

```bash
python -m rg_nanogpt_one_head.training \
  --config configs/v4_small_4x4.yaml \
  --optimizer all \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-ngb/results/v4_small_4x4 \
  --device auto
```

## Eight matched seeds

After the canonical three, add five more matched seeds with:

```bash
python -m rg_nanogpt_one_head.training \
  --config configs/v4_one_head.yaml \
  --optimizer all \
  --seeds 5003,6007,7013,8017,9011 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-ngb/results/v4_one_head \
  --device auto
```

Use the same seed list and corresponding result root for `v4_small_4x4.yaml`.

## Notebooks

```text
notebooks/01_run_v4_one_head.ipynb
notebooks/02_compare_v4_one_head.ipynb
notebooks/03_run_v4_small_4x4.ipynb
notebooks/04_compare_v4_small_4x4.ipynb
notebooks/05_compare_v4_architectures.ipynb
notebooks/06_compare_v3_v4_one_head.ipynb
```

The comparison notebooks discover the intersection of completed seeds across
all three optimizers. They therefore use three, eight, or any later matched
seed count without editing the notebook.

They report:

- final and validation-selected metrics with run-level 95% Student-t intervals;
- perplexity intervals obtained by exponentiating the loss-space interval;
- matched-seed paired optimizer contrasts;
- best validation step and final-minus-best validation-loss drift;
- maximum update-to-weight ratio and evaluation-snapshot clipping rate;
- optimizer configuration tables;
- optimizer-level and block-resolved WeightWatcher trajectories.

Block-resolved plots compute uncertainty across seeds for each matrix. Blocks
are never treated as additional statistical replicates.

Run all NGB notebooks with the active conda kernel:

```bash
cd /tmp/rg_optimizers/baseline/ngb

for nb in \
  01_run_v4_one_head \
  02_compare_v4_one_head \
  03_run_v4_small_4x4 \
  04_compare_v4_small_4x4 \
  05_compare_v4_architectures \
  06_compare_v3_v4_one_head
do
  papermill \
    "notebooks/${nb}.ipynb" \
    "notebooks/${nb}.out.ipynb" \
    -k python3
done
```

The architecture-comparison notebook must run after both three-optimizer suites
are complete.

## Validation

```bash
PYTHONPATH=../nanogpt_one_head/src \
  python -m pytest -q tests
```
