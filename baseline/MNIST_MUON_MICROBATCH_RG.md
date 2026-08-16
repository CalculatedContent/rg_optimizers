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

## Notebook

Open:

```text
notebooks/MNIST_MLP3_Muon_Microbatch_RG_ESD.ipynb
```

The notebook computes and power-law-fits three spectra per layer:

1. Weight ESD: `sigma(W_t)^2`.
2. Relative-flow ESD: `sigma(J_t)^2`, with the supported square map formed from
   successive checkpoints and a pseudoinverse.
3. Log-flow deviations: `abs(log(sigma(J_t)^2))`, which drops the trivial
   identity/orthogonal mode at one.

It writes `microbatch_powerlaw_fits.csv` and
`microbatch_esd_spectra.npz`, then plots fitted alpha versus optimizer step with
an `alpha = 2` reference line.

The relative-flow construction is basis dependent. This experiment tests that
proposal numerically; it does not establish a basis-invariant quotient.
