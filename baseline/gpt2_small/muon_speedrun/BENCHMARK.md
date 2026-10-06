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

Both arms load the byte-identical upstream GPT class directly, with packed QKV,
its unchanged layers, forward methods and initialization algorithm. Default seed
42 is upstream's seed; repeated-seed runs explicitly reseed the same initializer.
`stock_model.py` verifies source SHA256
`757d0cea0d48cbc4c7d7d70371f955d49cdf3a7cfb4c87701a720c2fe0905c34` before import.

MuonClip is an explicit optimizer substitution: peak LR 0.02, Nesterov momentum
0.95, five Newton–Schulz steps, RMS scale `0.2*sqrt(max(rows,columns))`, hidden
weight decay 0.1, and per-head QK clipping threshold 100 with balance 0.5. It uses
48 packed block matrices; auxiliary embeddings/biases/LayerNorm use AdamW 0.0006.
It shares the baseline warmup, cosine schedule and global clipping. The optional
plain `muon` arm retains LR 0.04, its 500-step momentum ramp and zero hidden decay.
These optimizer substitutions are not claims of a published GPT-2 MuonClip record.

The architecture itself is unchanged. Hardware adaptation is explicit: TPU SPMD,
BF16 autocast and checked flash attention replace the CUDA execution environment.
The loader follows the upstream Python sequential shard order, discards incomplete
microbatch tails and wraps at corpus end; it differs from the C loader's shuffled
order. All 103 training shards plus validation are hash-verified before training.
WeightWatcher adds measurement overhead. Loss convergence and H100 wall-clock
records are not guaranteed by matching model definitions and core hyperparameters.

`benchmark_config.py` is the single source for the schedule, budget and protocol
identifier. Plans, manifests and comparison reports carry
`llmc-gpt2-124m-fineweb10b-2024-10-13-upstream-tpu-v2`. Reports exclude mismatched protocols;
old 3,000-update plans cannot start new runs. Old jobs keep their pinned source.

For one new MuonClip run, use a clean checkout of merged `main` and a live allocation:

```bash
python3 baseline/gpt2_small/muon_speedrun/cloudshell.py start --node YOUR_TPU_NODE --optimizer muon_clip --hours 12
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
