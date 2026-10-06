# Stock GPT-2 Small / FineWeb: MuonClip versus AdamW

This is the dedicated experiment folder for the **original GPT-2/FineWeb
nanoGPT speedrun baseline**, with MuonClip as an explicit optimizer substitution.
The shared implementation remains in [`../../gpt2_small/muon_speedrun`](../../gpt2_small/muon_speedrun).
[`configuration.json`](configuration.json) records the audited configuration;
[`results/`](results/) is reserved for downloaded experimental outputs.

## Architecture and recipe audit — 2026-10-06

The model source was compared byte for byte against `karpathy/llm.c` commit
`7ecd8906afe6ed7a2b2cdb731c042f26d525b820`. Both optimizer arms instantiate that
same upstream `GPT` class. Its SHA256 is checked before import. No layers or
forward methods are rewritten. Packed QKV stays packed during training.

| Setting | Fixed value |
|---|---|
| Model | GPT-2 Small, 124,439,808 unique parameters |
| Blocks / attention heads / width | 12 / 12 / 768 |
| MLP width / context / vocabulary | 3072 / 1024 / 50257 |
| Positions / normalization / activation | Learned / affine LayerNorm / GPT-2 GELU |
| Embeddings / biases | Tied token-output weights / enabled |
| Data | `kjj0/fineweb10B-gpt2`, revision `889765ea1f903759787add96995d81171b632d0c` |
| Updates / global tokens per update | 19,560 / 524,288 |
| Warmup / schedule / global gradient clipping | 700 updates / cosine to zero / norm 1.0 |
| AdamW | LR 0.0006, betas 0.9/0.95, epsilon 1e-8, matrix decay 0.1 |
| MuonClip | LR 0.02, momentum 0.95, five NS steps, RMS scale 0.2, decay 0.1, QK threshold 100 |
| Validation and spectra | Every 250 updates and final; validation uses 10,485,760 tokens |
| Default single run | Seed 42, microbatch 64 sequences, eight accumulation steps |
| Paired comparison | Seeds 1337, 1338, 1339 for both optimizers |

This pins the original baseline, not the continually changing modified-model
speedrun leaderboard. The TPU attention adapter, BF16 execution, sequential Python
shard order, and measurement overhead differ from the published CUDA run. See
[the exact benchmark and differences](../../gpt2_small/muon_speedrun/BENCHMARK.md).
A code audit establishes the configuration; **no live TPU execution of this
configuration has been verified here**. Each launch requires a successful full-model
forward/backward/optimizer preflight on the selected TPU before fresh training.

## Weight matrix sizes

Shapes are stored PyTorch `[output, input]` dimensions; each block row repeats
for all 12 blocks. Bias and LayerNorm vectors are not matrices.

| Weight | Shape |
|---|---|
| Token embedding | 50,257 × 768 |
| Position embedding | 1,024 × 768 |
| Each block: packed QKV | 2,304 × 768 |
| Each block: attention output | 768 × 768 |
| Each block: MLP input | 3,072 × 768 |
| Each block: MLP output | 768 × 3,072 |
| Output head | 50,257 × 768; shared with token embedding |

There are 48 block matrices, 50 unique matrix parameters, and 51 named matrix
entries including the shared output head. [Every named matrix](../../gpt2_small/stock_weight_matrices.csv).

## Launch from the repository root

Use a clean checkout of the merged commit and a live v5e-8 allocation with the
preserved disk and environment. Specify the current node; the old node expired.

```bash
python3 baseline/gpt2_small/speedrun.py plan
python3 baseline/gpt2_small/muon_speedrun/cloudshell.py start --node YOUR_TPU_NODE --optimizer muon_clip --hours 12
# After that run has finished, start the control on the idle TPU:
python3 baseline/gpt2_small/muon_speedrun/cloudshell.py start --node YOUR_TPU_NODE --optimizer adamw --hours 12
```

Add `--here` when running in the TPU terminal. Twelve hours is a cap including
setup, preflight, tracking, and backup, not a completion estimate. The six-run
[paired suite](../../gpt2_small/muon_speedrun/REPEATED_SEEDS.md) requires 72h45m
remaining. An interrupted or deadline-stopped run is incomplete; reaching the
loss target early does not satisfy the default full-budget requirement.

## Dedicated output locations

New launches automatically create unique dated directories:

- TPU: `/mnt/disks/rg-data/gpt2small/stock_gpt2_fineweb_muonclip_adamw/runs/<run-id>/`
- Paired suite: the same experiment root under `suites/<suite-id>/`, with each seed/optimizer run below it.
- Cloud: `gs://tpu-builders-504820-ww-continuous8/gpt2small/stock_gpt2_fineweb_muonclip_adamw/`, mirroring the TPU hierarchy.
- Downloaded results: this folder's `results/runs/<run-id>/` or `results/suites/<suite-id>/`.

`launch.json` records the exact output/cloud path and commit. Each run retains
configuration, source hash and matrix inventory in `manifest.json`; preflight
proof; validation metrics; raw/clipped WeightWatcher tables; full-state
checkpoints; and completion/backup status. Immutable spectral weight snapshots
stay on the TPU disk; scientific tables and full checkpoints are backed up.
Only completed 19,560-update runs with final full validation, completed tracking,
and verified backup qualify for the paired report.

The global launch lock and latest-run pointers remain in `/mnt/disks/rg-data/gpt2small/`
so historical and current launchers still prevent competing jobs. Existing results
are retained in their original locations; the new folder does not move old data.
