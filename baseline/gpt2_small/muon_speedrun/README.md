# Upstream GPT-2 Small: MuonClip and AdamW on TPU

Both optimizers instantiate the **unchanged upstream `GPT` class** vendored from
karpathy/llm.c commit `7ecd8906afe6ed7a2b2cdb731c042f26d525b820`. The source is
byte-identical to upstream and its SHA256 is checked before import. There are
no replacement model layers or rewritten forward methods. QKV stays packed.

The model is GPT-2 Small: **12 blocks, 12 heads, width 768, MLP 3072, context 1024,
vocabulary 50257, 124,439,808 parameters**. It has learned positions, LayerNorm,
GPT-2 GELU, biases, standard residuals and tied token/output embeddings.
[All stored weight matrix dimensions](../STOCK_ARCHITECTURE.md).

The [pinned benchmark configuration](BENCHMARK.md) uses the original GPT-2/FineWeb
speedrun baseline: 19,560 updates, 524,288 tokens per update, 700 warmup updates,
cosine decay to zero, global gradient clipping at 1.0, and full 10,485,760-token
validation every 250 updates and at exit. Single-run initialization defaults to
the upstream seed 42. Deadline interruptions do not count as completed runs.

## Experiment folder and results

[Experiment README and audited configuration](../../experiments/stock_gpt2_fineweb_muonclip_adamw/README.md).
New single runs save beneath `/mnt/disks/rg-data/gpt2small/stock_gpt2_fineweb_muonclip_adamw/runs/`;
paired suites use `suites/`. Cloud backup mirrors this hierarchy under
`gs://tpu-builders-504820-ww-continuous8/gpt2small/stock_gpt2_fineweb_muonclip_adamw/`.
Each `launch.json` records the concrete paths. Downloaded results belong in the
experiment folder's `results/` directory. Historical results retain their locations.

## Checkout and run

Use a clean checkout of the published commit, the preserved data disk and the
existing environment at `/mnt/disks/rg-data/continuous8/venv`. Specify a **live**
v5e-8 node; the old October 4 allocation has expired.

```bash
python3 baseline/gpt2_small/muon_speedrun/cloudshell.py start --node YOUR_TPU_NODE --optimizer muon_clip --hours 12
# Run the AdamW control separately on the idle TPU:
python3 baseline/gpt2_small/muon_speedrun/cloudshell.py start --node YOUR_TPU_NODE --optimizer adamw --hours 12
```

Add `--here` from the TPU terminal. The default optimizer is `muon_clip`. The
optional `muon` choice retains plain Muon; it is a different optimizer.
`--replace-current` and `--replace-longrun` explicitly stop the respective recorded
service while preserving files. By default active training blocks another launch.
The launcher checks the live lease and guest identity. It never allocates or
extends a TPU. Twelve hours is a cap including setup and backup, not a runtime ETA.

Before training, the worker verifies the complete pinned corpus and runs:

1. Flash-attention forward/backward parity at 12 heads × 64 dimensions.
2. A disposable **full-sized model and selected optimizer update**, with the actual
   1024 context, selected microbatch, eight-chip partition and accumulation to the
   524,288-token global batch. It checks finite loss, gradient norm, updated weights
   and embedding/output tying, then saves `MODEL_PREFLIGHT.json`.
3. Fresh training from initialization. TPU training refuses a missing or mismatched
   preflight identity. A failed preflight stops the launch and preserves its log.

The hardware adapter selects the TPU attention kernel and BF16 autocast outside
the upstream model definition. TPU numerical/reduction order and timing differ
from the CUDA record; CPU tests do not establish TPU memory fit or convergence.

## Optimizers and observations

AdamW uses LR 0.0006, betas (0.9,0.95), epsilon 1e-8, decay 0.1 on matrices and
zero decay on biases/LayerNorm. MuonClip uses LR 0.02, Nesterov momentum 0.95,
five Newton–Schulz iterations, RMS scaling `0.2*sqrt(max(rows,columns))`, and
hidden-matrix decay 0.1. Its auxiliary parameters use the same AdamW recipe.
MuonClip applies per-head QK clipping at threshold 100 with equal Q/K scaling.
It observes exact causal attention-logit maxima across accumulation microbatches
through hooks that return `None`. Q/K weights and biases are rescaled only in the
optimizer step; V is not rescaled. The architecture and forward outputs are untouched.

MuonClip operates on 48 stored block matrices, including packed QKV. WeightWatcher
extracts separate Q/K/V views **from saved CPU checkpoints** for the existing 72
spectral traces. Measurement does not split trainable parameters. Raw and clipped
alpha, validation NLL/perplexity and top-1 token error stay paired by checkpoint.
Spectral fitting runs in a separate CPU process, never in the training graph.

Atomic latest/best/target checkpoints include model, optimizer state, data cursor,
RNG and manifest. Immutable spectral snapshots remain on disk; small results and
full checkpoints receive verified final cloud backup. No automatic resume/restart.

`python3 baseline/gpt2_small/speedrun.py plan` shows the
[three-seed MuonClip/AdamW comparison](REPEATED_SEEDS.md). Historical six-head
modified-model and plain-Muon results remain separately identified and cannot
enter the new comparison. The old 25k entry point remains historical.
