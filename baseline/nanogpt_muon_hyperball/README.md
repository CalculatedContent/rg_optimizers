# nanoGPT Muon-HyperBall baseline

This is a **separate** one-block, one-head nanoGPT experiment family for a
matched comparison of:

1. ordinary Muon + auxiliary AdamW; and
2. Muon + a relative Frobenius HyperBall projection + auxiliary AdamW.

It does not modify the existing `baseline/nanogpt_one_head` controls or their
results.

## Why this experiment exists

The one-epoch Muon runs produced highly variable layer-wise power-law exponents.
A naïvely stretched ten-epoch learning-rate schedule was also unstable: because
the original five-percent warmup was interpreted as five percent of the full
ten-epoch horizon, Muon's matrix LR kept increasing until epoch 0.5 and the run
became non-finite before the first half epoch completed.

This baseline separates the **training horizon** from the **LR-schedule
horizon**:

- train for ten corpus-equivalent epochs;
- reproduce the original 488-step Muon warmup;
- finish the original cosine decay at epoch 1;
- hold the configured LR floors through epochs 1–10.

That preserves the known one-epoch trajectory and then gives the weights a long
low-LR relaxation period in which alpha can stabilize.

## HyperBall update

For each hidden transformer matrix, ordinary Muon first proposes the complete
next value, including multiplicative matrix weight decay:

\[
W_t^\star = \operatorname{MuonStep}(W_t),
\qquad
\Delta W_t = W_t^\star-W_t .
\]

HyperBall then projects that displacement into a relative Frobenius ball:

\[
\Delta W_t^{\mathrm{HB}}
=
\Delta W_t
\min\left(
1,
\frac{\rho\lVert W_t\rVert_F}
{\lVert\Delta W_t\rVert_F+\epsilon}
\right).
\]

The reference radius is `rho = 0.01`, so each hidden matrix moves by at most one
percent of its current Frobenius norm per optimizer step. The projection is
radial: it does not rotate Muon's update and does not use alpha, ERG gap,
trace-log, validation loss, or any test measurement.

Muon acts on:

```text
W_Q, W_K, W_V, W_O, W_MLP_IN, W_MLP_OUT
```

The tied embedding/head, normalization gains, and all remaining parameters use
the same auxiliary AdamW path in both arms.

## Reference protocol

- pinned FineWeb-Edu 80M / 1M / 1M token train/validation/test split;
- one transformer block, one attention head, width 128, context 256;
- seed `1337`;
- ten corpus-equivalent training epochs;
- WeightWatcher every quarter epoch;
- matrix LR `0.02 -> 0.002` over the first epoch, then floor;
- auxiliary LR `3e-4 -> 3e-5` over the first epoch, then floor;
- validation loss selects `checkpoint_best.pt`;
- fixed test probes are monitoring-only.

The default output root is:

```text
/tmp/rg-nanogpt-muon-hyperball
```

## Install

From the repository root:

```bash
cd baseline/nanogpt_muon_hyperball
python -m pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Reuse the already verified corpus:

```text
/tmp/rg-nanogpt-one-head/data
```

## Run the ordinary long-Muon control

```bash
python -u -m rg_nanogpt_muon_hyperball.training \
  --config configs/reference.yaml \
  --optimizer muon \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-muon-hyperball/results \
  --device auto
```

## Run Muon-HyperBall

```bash
python -u -m rg_nanogpt_muon_hyperball.training \
  --config configs/reference.yaml \
  --optimizer muon_hyperball \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-muon-hyperball/results \
  --device auto
```

For an unattended MacBook run:

```bash
mkdir -p /tmp/rg-nanogpt-muon-hyperball/logs

nohup caffeinate -i \
python -u -m rg_nanogpt_muon_hyperball.training \
  --config configs/reference.yaml \
  --optimizer muon_hyperball \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-muon-hyperball/results \
  --device auto \
  > /tmp/rg-nanogpt-muon-hyperball/logs/muon_hyperball_seed_1337.log 2>&1 &

echo $!
```

Watch it with:

```bash
tail -f /tmp/rg-nanogpt-muon-hyperball/logs/muon_hyperball_seed_1337.log
```

Do not resume the failed stretched-schedule run. It already contains non-finite
weights. This experiment uses a new result namespace and a different protocol
fingerprint.

## Recorded HyperBall diagnostics

Each evaluation row records:

```text
hyperball_relative_radius
hyperball_matrix_updates_since_eval
hyperball_active_fraction
hyperball_mean_scale
hyperball_min_scale
hyperball_mean_radius
hyperball_max_proposed_update_to_weight_ratio
hyperball_max_applied_update_to_weight_ratio
hyperball_max_proposed_update_norm
hyperball_max_applied_update_norm
```

The training loop aborts on non-finite train or validation metrics **before**
calling WeightWatcher, so a numerical optimizer failure is reported directly
rather than surfacing later as `numpy.linalg.LinAlgError: SVD did not converge`.

## Analysis

Run:

```bash
jupyter lab notebooks
```

The comparison notebook plots task metrics, per-layer alpha and `D`, ERG gap,
traps, and HyperBall activation statistics.

## Radius ablation

Use separate config files and separate result roots for:

```text
rho = 0.005
rho = 0.010  # reference
rho = 0.020
```

Select only on validation loss. Test metrics remain protected monitoring
measurements.

## Tests

```bash
python -m pytest -q tests
```

The tests cover the Frobenius cap, direction preservation, equivalence to
ordinary Muon at effectively infinite radius, optimizer partitioning, schedule
horizon, configuration validation, and notebook syntax.
