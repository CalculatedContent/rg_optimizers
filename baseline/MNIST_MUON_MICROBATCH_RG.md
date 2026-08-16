# MNIST MLP3 Muon microbatch RG capture

This opt-in experiment trains the existing `784-512-512-10` MNIST MLP3 with
the baseline's exact Muon-on-hidden-layers plus auxiliary-AdamW recipe. The
baseline has no gradient accumulation, so each DataLoader minibatch is one
optimizer microbatch.

## Install

```bash
cd baseline
python -m pip install -e '.[experiment]'
```

## Run

A bounded first test that saves the three weight matrices after every update:

```bash
rg-mnist-muon-microbatch \
  --data-dir ./data \
  --output-dir ./results/mnist_mlp3_muon_microbatch_500 \
  --max-steps 500 \
  --capture-every 1 \
  --checkpoint-dtype float32 \
  --overwrite
```

The full 30-epoch baseline has about 12,900 optimizer microbatches. Saving all
three matrices in float32 at every step is roughly 32 GiB before container
overhead, so the runner refuses captures above 8 GiB unless explicitly enabled:

```bash
rg-mnist-muon-microbatch \
  --data-dir ./data \
  --output-dir ./results/mnist_mlp3_muon_microbatch_full \
  --capture-every 1 \
  --checkpoint-dtype float32 \
  --allow-large-capture \
  --overwrite
```

To reduce storage, use `--checkpoint-dtype float16`, increase
`--capture-every`, or set `--max-capture-step` while allowing training to
continue.

## Artifacts

```text
<run>/
  manifest.json
  training_metrics.csv
  final_state.pt
  microbatch_checkpoints/
    manifest.json
    checkpoint_index.csv
    frames/
      step_0000000.pt
      step_0000001.pt
      ...
```

Each frame stores `fc1.weight`, `fc2.weight`, and `fc3.weight` only.

## Original pseudoinverse analysis

Open:

```text
notebooks/MNIST_MLP3_Muon_Microbatch_RG_ESD.ipynb
```

This exploratory notebook computes the ordinary weight ESD and the supported
pseudoinverse relative-flow spectrum. The latter is complete for square
full-rank matrices but mixes core deformation with subspace overlap for
rectangular matrices.

## Gauge-aligned rectangular analysis

Open:

```text
notebooks/MNIST_MLP3_Muon_Rectangular_RG_ESD.ipynb
```

or run:

```bash
rg-mnist-muon-rectangular-analysis \
  --run-dir ./results/mnist_mlp3_muon_microbatch_500 \
  --step-stride 1
```

For a wide full-row-rank matrix such as `fc1.weight`, write

```text
W_t = B_t V_t^T,
V_t^T V_t = I.
```

The row-space bases at successive steps are aligned by orthogonal Procrustes.
The analysis then reports two independent spectra:

1. Aligned square-core flow:
   `abs(log(sigma(B_t_aligned B_{t-1}^{-1})^2))`.
2. Grassmann angular flow: the squared principal angles `theta_i^2` between
   successive row spaces.

For `fc1.weight` (`512 x 784`), two 512-dimensional row spaces must intersect
in at least 240 dimensions, so there are at most 272 nontrivial angular modes.
The implementation removes those dimension-forced zero angles before fitting.

For square full-rank `fc2.weight`, the angular sector vanishes and the aligned
core operator reduces numerically to `W_t W_{t-1}^{-1}`. This gives a direct
control showing that the rectangular construction agrees with the original
square relative Jacobian.

The analysis writes power-law fits, tail sizes, tail fractions, KS distances,
condition numbers, principal-angle diagnostics, ESD archives, and alpha-versus-
step plots.

`powerlaw` 2.0 uses a built-in upper bound of `alpha = 3` for its power-law
model. Both notebooks explicitly expand the fitting range to
`1.01 <= alpha <= 10` and mark fits that reach the expanded boundary.
