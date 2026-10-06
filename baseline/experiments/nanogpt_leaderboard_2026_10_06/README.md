# Current nanoGPT leaderboard implementation — pinned October 6, 2026

This is a separate experiment using the current accepted **modded-nanogpt
track-1 record**, as retrieved on October 6, 2026. It keeps the complete upstream
training implementation unchanged, including the optimizer and CUDA kernels.

The [stock GPT-2 MuonClip/AdamW experiment](../stock_gpt2_fineweb_muonclip_adamw/README.md)
remains a separate model, protocol, launcher and results directory.

## Exact reference

| Item | Pinned value |
|---|---|
| Upstream repository | [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| Source commit | [`4ea6b937337a4889b8cfe3f38a93d120048d8f71`](https://github.com/KellerJordan/modded-nanogpt/tree/4ea6b937337a4889b8cfe3f38a93d120048d8f71), September 28, 2026 |
| Latest accepted track-1 record | #92, August 30, 2026, [ANVIL2 / PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360) |
| Published timed result | 0.665 minutes = 39.9 seconds on 8 H100 GPUs |
| Target | FineWeb validation cross-entropy ≤3.28 over 10,485,760 tokens |
| Entry point | `vendor/train_gpt.py` plus the full `vendor/track_1_short/` package |
| Training | 1,194 updates, 328,663,040 tokens, five batch/sequence/window stages |
| Optimizer | Upstream ANVIL stack, auxiliary Adam, and sparse row-wise Adam |
| Hardware | One node with 8 NVIDIA H100 80GB GPUs; CUDA/NCCL |

The 39.9 seconds is the published benchmark's timed region. It excludes the
roughly seven-minute first compilation/warmup and final validation forward pass;
installation, downloads, and optional weight export also take additional time.
It is not a runtime estimate for a TPU or a guarantee for our checkout.

`upstream.json` records the reference and computed schedule. `upstream_git_files.json`
contains Git blob IDs and SHA256 hashes for all 58 vendored files. They were verified
byte for byte against the pinned upstream commit. The original MIT license and
README are retained in `vendor/`. Future records require a new explicit pin;
this folder does not follow a moving branch at launch.

## This latest model is no longer a 124M GPT-2

The current trainer constructs an 11-block, 6-head, width-768 model with attention
only in selected blocks and mixed active QK/V head widths. It also maintains a
separate **84,602,880 × 768 hashed n-gram table: 64,975,011,840 trainable values**,
about 130 GB in BF16, sharded across the eight GPUs. This table is outside
`model.parameters()` and `model.state_dict()`; quoting only the dense model count
would conceal most of the learned storage.

The accumulated upstream changes now go beyond rotary embeddings, QK-norm,
ReLU², value embeddings, long/short windows, YaRN, smear/gates and FP8. They include
sampled softmax, sparse n-gram updates, mixed-width/depth-reduced attention, ANVIL,
tail weight averaging and CUDA-graph capture. Some earlier features have evolved
again; the code at the pinned commit is authoritative, not a hand-recreated list.
See the preserved [upstream README](vendor/README.md) and [model](vendor/track_1_short/model/gpt.py).

**This implementation cannot run unchanged on our TPU.** It depends on Triton,
CUDA graphs, FP8 GPU kernels, NCCL and patched FlashAttention-3. A TPU adaptation
would be a separate port requiring numerical and throughput validation. No TPU
port or optimizer substitution is included in this reference experiment.

## Inspect and prepare

From the repository root, these two commands require only Python's standard library:

```bash
python3 baseline/experiments/nanogpt_leaderboard_2026_10_06/experiment.py plan
python3 baseline/experiments/nanogpt_leaderboard_2026_10_06/experiment.py verify
```

On the H100 host, use the [upstream dependency list](vendor/requirements.txt) and
[Dockerfile](vendor/Dockerfile): PyTorch **2.10 cu128**, a **CUDA 13 runtime**, and
the pinned `devenpzak/flash-attn3-12864` kernel. Do not replace this stack with a
nightly build. The upstream Dockerfile installs dependencies but does not copy
the source: mount this repository into the container when running it.

Example inside the correctly configured environment (run from the repository root):

```bash
python baseline/experiments/nanogpt_leaderboard_2026_10_06/experiment.py check
python baseline/experiments/nanogpt_leaderboard_2026_10_06/experiment.py prepare --data-root /data/nanogpt-leaderboard
python baseline/experiments/nanogpt_leaderboard_2026_10_06/experiment.py run --data-root /data/nanogpt-leaderboard --results-root /results/nanogpt-leaderboard
```

The unchanged downloader fetches nine training shards (900M tokens) plus validation.
Its dataset revision is not pinned upstream; the wrapper records SHA256 hashes of
the actual data files in each run manifest. The default upstream initialization is
unseeded. The wrapper refuses inherited `TRAIN_SEED` or `NUM_SCHEDULED_ITERATIONS`
overrides so a shell setting cannot silently change the reference recipe.

Each run gets an immutable copy of the verified sources and its own logs/results.
The wrapper starts exactly eight local GPU workers. It does not allocate hardware,
stop existing TPU jobs, resume training, or change the upstream schedule. GPU memory,
kernel compatibility and convergence have **not** been verified on our hardware.

## Saving final weights for WeightWatcher

Upstream defaults to **no checkpoint**. Its optional rank-0 checkpoint also omits
the sparse n-gram table. To retain the learned tensors, explicitly select our
post-run export mode:

```bash
python baseline/experiments/nanogpt_leaderboard_2026_10_06/experiment.py run --data-root /data/nanogpt-leaderboard --results-root /results/nanogpt-leaderboard --save-weights
```

This mode leaves upstream files, model forwards and optimizer updates unchanged.
An external adapter exports after final validation, immediately before process-group
shutdown, outside the timed region. It is recorded as `upstream-plus-postrun-weight-export`,
so its end-to-end time is distinguishable from the default run. It requires at least
160 GiB of free disk space; the sparse weights alone occupy about 121 GiB.

- `weights/rank-00/model.pt`: dense `model` state dict, final step and inference buffers;
  these are the final evaluated, tail-averaged dense weights.
- `weights/rank-XX/ngram-*.pt`: CPU BF16 sparse-table chunks with global row offsets;
  all eight ranks are required for the complete learned table.
- `weights/WEIGHTS_COMPLETE.json`: completed rank exports, coverage and file hashes.

These are actual CPU-readable tensors, suitable for later WeightWatcher analysis;
no training data or TPU is needed for spectral analysis. The dense model uses packed
parameter banks with inactive/padded rows, so per-layer analysis must use the pinned
bank mapping and active widths. Do not treat a whole 3D/4D bank as one ordinary
linear layer or attempt a dense SVD of the entire 65B table. Automatic WeightWatcher
fitting is not injected into the leaderboard run.

The export is for weight analysis, **not an optimizer-resume checkpoint**. GPU execution
of the adapter remains unverified; CPU tests cover its shutdown hook, tensor export,
hash checks and row coverage. An interrupted run may have only partial outputs.
Use durable storage and archive the results; this runner does not automatically
upload to the stock experiment's TPU/GCS paths. See [`results/`](results/README.md).
