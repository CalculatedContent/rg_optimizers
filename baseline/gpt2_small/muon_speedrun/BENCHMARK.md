# Pinned GPT-2 / FineWeb baseline for the next Muon and AdamW runs

The default is the **original GPT-2 124M / FineWeb baseline behind the nanoGPT
speedrun**, pinned to the October 13, 2024 llm.c reference. It is not the evolving
KellerJordan modified-model leaderboard architecture. There is no universal
stock-GPT-2 Muon recipe: the Muon arm below is an explicitly identified optimizer
substitution. This TPU port is not claimed to reproduce H100 wall-clock records.

| Setting | Default in both new experiment arms |
|---|---|
| Model | Stock GPT-2 Small, 12 blocks, 12 heads, width 768, MLP 3072 |
| Context / vocabulary | 1024 / 50257 |
| Positions / norm / activation | Learned absolute / LayerNorm / GPT-2 tanh GELU |
| Biases / head / dropout | Enabled / tied token embedding / 0 |
| Unique parameters | 124,439,808 |
| Data | GPT-2-tokenized FineWeb10B, revision 889765ea1f903759787add96995d81171b632d0c |
| Global batch | 524,288 tokens, microbatch 64 sequences, 8 accumulation passes |
| Training budget | 19,560 updates / 10,255,073,280 tokens |
| Schedule | 700-update linear warmup, cosine decay to zero at update 19,560 |
| AdamW | Peak LR 0.0006, betas (0.9, 0.95), epsilon 1e-8 |
| AdamW decay | 0.1 for matrices, zero for biases and LayerNorm |
| Gradient clipping | Global L2 norm 1.0, after accumulation and before optimizer update |
| Validation | First 10,485,760 validation tokens every 250 updates and at exit |
| Stop condition | Full training budget by default; deadline interruptions are incomplete |

Muon replaces AdamW on the 72 transformer projection matrices only. It uses peak
LR 0.04, momentum 0.85→0.95 over 500 updates, five Newton–Schulz iterations and
zero hidden-matrix decay. It shares the baseline warmup/cosine schedule and global
clipping. Embeddings, biases and LayerNorm use auxiliary AdamW at 0.0006.
These Muon settings have not been tuned or validated for convergence on this model.

The architecture is numerically checked against the repository's independently
pinned upstream packed-QKV GPT-2 implementation. Q/K/V are stored separately for
per-projection Muon and WeightWatcher measurements; orthogonalizing them separately
is not the same Muon update as orthogonalizing one packed matrix. AdamW is elementwise.
TPU SPMD, BF16 arithmetic, microbatch reduction order, split-QKV initialization draw
order, diagnostic overhead and token loading differ from the CUDA record. The
loader follows the upstream **Python** sequential shard order, discards incomplete
microbatch tails and wraps at the corpus end; it is not the C loader's shuffled
order. All 103 training shards plus validation are verified before training.
These differences are recorded in `protocol` rather than hidden behind a
claim of exact benchmark reproduction.

`benchmark_config.py` is the single source for the schedule, budget and protocol
identifier. Plans, manifests and comparison reports carry
`llmc-gpt2-124m-fineweb10b-2024-10-13-tpu-v1`. Reports exclude mismatched protocols;
old 3,000-update plans cannot start new runs. Old jobs keep their pinned source.

For one new Muon run, use a clean checkout of merged `main` and a live allocation:

```bash
python3 baseline/gpt2_small/muon_speedrun/cloudshell.py start --node YOUR_TPU_NODE --optimizer muon --hours 12
```

Use `--optimizer adamw` for the control, or `--here` from the TPU terminal. The
launcher checks the live lease and refuses insufficient time or a mismatched guest.
The default 12-hour cap includes setup, tracking and backup; it is not an ETA or
a guarantee that 19,560 steps will finish. It never allocates or extends a TPU.
The six-run suite reserves six 12-hour caps and requires **72h45m** remaining;
a 48-hour allocation cannot run that whole suite under these caps.

References:

- [Pinned llm.c model](https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/train_gpt2.py)
- [Pinned reproduction launcher](https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/scripts/run_gpt2_124M.sh)
- [October 13 published 19,560-update reference](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2024-10-13_llmc)
- [Architecture and every weight-matrix dimension](../STOCK_ARCHITECTURE.md)
