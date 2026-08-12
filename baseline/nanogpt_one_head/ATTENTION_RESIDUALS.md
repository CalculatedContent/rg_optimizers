# Attention Residuals baseline

This experiment adds **Full Attention Residuals (AttnRes)** as a controlled architectural baseline inside the existing one-head nanoGPT suite.

## What changes

Standard residual routing uses the immediately preceding state with a fixed additive path. Full AttnRes instead forms each sublayer input by content-dependent attention over all residual states available earlier in depth. Each routing point has a learned pseudo-query vector; prior residual states are RMS-normalized to form routing keys, while the original states remain the values.

The implementation provides one router before attention and one router before the MLP. The existing causal self-attention computation itself is unchanged.

## What does not change

The controlled comparison keeps the existing long-Muon baseline fixed:

- FineWeb-Edu sample and pinned revision
- GPT-2 tokenizer
- 80M training tokens, 1M validation tokens, 1M test tokens
- context length 256
- one transformer block
- one attention head
- embedding width 128
- zero dropout, bias disabled, tied token/output embeddings
- batch size 4 with 8 gradient-accumulation steps
- fixed evaluation probes and BLEU protocol
- WeightWatcher ERG/randomization settings
- separated `W_Q`, `W_K`, `W_V`, `W_O`, `W_MLP_IN`, and `W_MLP_OUT` matrices

AttnRes pseudo-query vectors are one-dimensional parameter vectors. They are deliberately excluded from `transformer_matrix_items()`, so WeightWatcher continues to analyze the same six matrices per block. Under Muon, the six 2-D hidden matrices remain in the Muon group while the AttnRes queries enter the auxiliary AdamW group.

## Why keep one block first?

Full AttnRes becomes more expressive as depth increases. In a one-block transformer, the attention router initially has only the embedding residual state available, and the MLP router can select between the embedding state and the post-attention state. This is intentionally a conservative first experiment: it preserves the architecture, parameter scale, data exposure, and token budget of the current baseline and therefore isolates the effect of residual routing as cleanly as possible.

The model implementation itself is depth-ready and keeps one head at every depth. A later multi-block study can test the larger routing advantage without changing the AttnRes implementation.

## Experiment 1: exact matched long-Muon comparison

Use the existing control:

```bash
rg-onehead-train \
  --config configs/muon_10epochs.yaml \
  --optimizer muon \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-long-muon-standard/results \
  --device auto \
  --no-resume
```

Run AttnRes with every training hyperparameter unchanged:

```bash
rg-onehead-train \
  --config configs/attnres_muon_10epochs.yaml \
  --optimizer muon \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-long-muon-attnres/results \
  --device auto \
  --no-resume
```

Use different results roots because run directories are keyed by optimizer and seed; the protocol fingerprint will correctly distinguish the model configurations but should not be forced to collide in one directory.

## Experiment 2: conservative full-horizon cosine pair

The existing ten-epoch Muon repair uses a one-epoch cosine schedule followed by nine epochs at the LR floor. That is the correct matched historical control, but it is not the only plausible schedule for studying long-horizon convergence.

The new `*_longcosine.yaml` pair is a deliberately conservative candidate, not an empirically proven optimum. It uses the same schedule for both residual architectures:

```text
training horizon:       10 epochs
LR schedule horizon:    10 epochs
Muon matrix peak LR:    0.0100
Muon matrix floor LR:   0.0005
auxiliary peak LR:      0.0003
auxiliary floor LR:     0.00001
warmup:                 1% of the 10-epoch schedule
```

Run the standard-residual control:

```bash
rg-onehead-train \
  --config configs/muon_10epochs_longcosine.yaml \
  --optimizer muon \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-longcosine-standard/results \
  --device auto \
  --no-resume
```

Run the matched AttnRes model:

```bash
rg-onehead-train \
  --config configs/attnres_muon_10epochs_longcosine.yaml \
  --optimizer muon \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-longcosine-attnres/results \
  --device auto \
  --no-resume
```

Do not interpret the long-cosine pair as an AttnRes gain unless it is compared against its matching standard-residual long-cosine control.

## Primary comparison metrics

For convergence speed, compare at matched optimizer steps and matched tokens:

- train, validation, and test loss
- validation and test next-token accuracy
- perplexity
- BLEU probe
- step/token count to fixed validation-loss thresholds
- best validation loss and the step at which it occurs

For the RG/WeightWatcher analysis, retain the existing per-matrix metrics for all six matrices and compare trajectories of alpha, randomization distance, ERG quantities, and spectral diagnostics. AttnRes routing weights can additionally be inspected with `model.attention_residual_weights()` without altering the WeightWatcher matrix set.

The key causal question is not whether the AttnRes run ends with a better number after ten epochs, but whether it reaches the same validation-loss or accuracy threshold in fewer matched tokens while preserving or improving out-of-sample behavior.
