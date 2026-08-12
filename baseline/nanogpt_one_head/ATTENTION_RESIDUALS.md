# Attention Residuals baseline

This experiment adds **Full Attention Residuals (AttnRes)** as a controlled architectural baseline inside the existing one-head nanoGPT suite.

## What changes

Standard residual routing accumulates sublayer outputs through a fixed additive path. Full AttnRes replaces that accumulation: before each attention or MLP sublayer, it forms the sublayer input by content-dependent attention over the embedding and all preceding sublayer outputs. Each routing point has a learned pseudo-query vector; prior outputs are RMS-normalized to form routing keys, while the original outputs remain the values. The selected mixture is passed through the normal pre-norm sublayer, and only that sublayer output is appended as the next depth value.

The implementation provides one router before attention and one router before the MLP. The existing causal self-attention computation itself is unchanged.

## What does not change

Every matched comparison keeps the existing one-head nanoGPT protocol fixed except for residual routing:

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

AttnRes pseudo-query vectors are 1-D parameter tensors of length `n_embd`. They are deliberately excluded from `transformer_matrix_items()`, so WeightWatcher continues to analyze the same six matrices per block. Under Muon, the six 2-D hidden matrices remain in the Muon group while the AttnRes queries enter the auxiliary AdamW group.

## Command-line setup

From `baseline/nanogpt_one_head`:

```bash
python -m pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Prepare the pinned corpus once and reuse it for every standard/AttnRes comparison:

```bash
rg-onehead-prepare --config configs/reference.yaml
```

All scientific runs use the same command-line trainer:

```text
rg-onehead-train
```

This preserves the existing checkpointing, restart, evaluation, WeightWatcher, MPS/CUDA/CPU/TPU device selection, and result layout.

## Short run: exact one-epoch reference comparison

This is the quickest clean comparison and uses all three canonical seeds by default.

Standard residual control:

```bash
rg-onehead-train \
  --config configs/reference.yaml \
  --optimizer muon \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-short-standard/results \
  --device auto \
  --no-resume
```

AttnRes, with the same data, model dimensions, seeds, token budget, Muon hyperparameters, schedule, and probes:

```bash
rg-onehead-train \
  --config configs/attnres_reference.yaml \
  --optimizer muon \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-short-attnres/results \
  --device auto \
  --no-resume
```

For a fast pilot before committing to all three seeds, append `--seeds 1337` to both commands.

## Long run A: exact matched historical 10-epoch comparison

This preserves the existing validated one-epoch Muon warmup/cosine horizon and then holds the LR floors through epoch 10. It is the correct historical comparison because it changes only residual routing.

Standard residual control:

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

Matched AttnRes:

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

## Long run B: convergence-oriented full-horizon cosine pair

The historical 10-epoch protocol spends epochs 1--10 at the LR floor. For a long run intended to continue optimizing rather than primarily observe late-horizon behavior, the `*_longcosine.yaml` pair uses a conservative learning-rate schedule across all ten epochs:

```text
training horizon:       10 epochs
LR schedule horizon:    10 epochs
Muon matrix peak LR:    0.0100
Muon matrix floor LR:   0.0005
auxiliary peak LR:      0.0003
auxiliary floor LR:     0.00001
warmup:                 1% of the 10-epoch schedule
```

The lower Muon peak relative to the one-epoch reference reduces long-horizon instability risk, while the nonzero floor prevents the optimization from becoming effectively frozen. This is a convergence-oriented candidate rather than a claim that its hyperparameters are globally optimal; actual convergence must be established from validation trajectories.

Standard residual control:

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

Matched AttnRes:

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

Do not attribute a gain to AttnRes unless it appears against the matching standard-residual config at the same token/step budget.

## Why keep one block first?

Full AttnRes becomes more expressive as depth increases. In a one-block transformer, the attention router initially has only the embedding output available, and the MLP router can select between the embedding and the attention-sublayer output. This is intentionally conservative: it preserves the architecture, data exposure, token budget, and all six transformer matrices of the current baseline and therefore isolates the effect of residual routing as cleanly as possible. The only added trainable parameters are two length-128 pseudo-query vectors.

The model implementation itself is depth-ready and keeps one head at every depth. A later multi-block study can test the larger routing advantage without changing the AttnRes implementation.

## Primary comparison metrics

For convergence speed, compare at matched optimizer steps and matched tokens:

- train, validation, and test loss
- validation and test next-token accuracy
- perplexity
- BLEU probe
- step/token count to fixed validation-loss thresholds
- best validation loss and the step at which it occurs

For the RG/WeightWatcher analysis, retain the existing per-matrix metrics for all six matrices and compare trajectories of alpha, randomization distance, ERG quantities, and spectral diagnostics. AttnRes routing weights can additionally be inspected with `model.attention_residual_weights()` without altering the WeightWatcher matrix set.

The key causal question is not whether the AttnRes run ends with a better number after a fixed horizon, but whether it reaches the same validation-loss or accuracy threshold in fewer matched tokens while preserving or improving out-of-sample behavior.
