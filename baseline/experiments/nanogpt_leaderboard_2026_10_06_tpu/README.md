# ANVIL2 nanoGPT leaderboard — Google TPU port

This folder implements a PyTorch/XLA BF16 port for the existing **v5litepod-8 (v5e-8), eight chips on one host**. Model, optimizer, host-backed n-gram table, training loop, validation and weight export are implemented. **Actual TPU execution, full-size memory fit and convergence remain unqualified.** `run` requires a successful on-device preflight; CPU tests cannot satisfy that gate.

The sibling [CUDA reference](../nanogpt_leaderboard_2026_10_06) remains byte-verified at upstream `4ea6b937337a4889b8cfe3f38a93d120048d8f71`. This is the evolved ANVIL2 leaderboard model, **not stock GPT-2 or a MuonClip/AdamW comparison**. No CUDA reference files are changed.

## Fixed experiment

| Item | Value |
|---|---|
| Dense model | 11 blocks, width 768, six heads; 302,792,179 parameters |
| N-gram table | 84,602,880 × 768 BF16; 64,975,011,840 values; 121.025 GiB |
| Optimizers | ANVIL twin rails and six-map cascade; auxiliary cautious Adam; sparse row-wise Adam |
| Steps / training tokens | 1,194 / 328,663,040 |
| Stage boundaries | 0, 320, 681, 1107, 1174, 1194 |
| Global batch tokens | 131072, 262144, 393216, 327680, 131072 |
| Final evaluation | 10,485,760 FineWeb tokens; NLL ≤ 3.28; perplexity ≤ 26.576 |

The table is **not shrunk**. Eight CPU row owners hold it in host RAM, exchange requested rows through Gloo, and upload bounded BF16 caches to their TPU chips. Dense gradients use XLA collectives. One PJRT process runs per chip (MPMD); SPMD is rejected. Dense optimizer state is replicated. The launcher requires **192 GiB available host RAM** and **160 GiB free export disk**; these guards do not prove HBM fit. `capacity.py` separately explains why the all-HBM approach fails on this machine.

## Run on the TPU VM

Use Python 3.11 or 3.12 and `python3-venv` on the existing v5e-8 VM. No H100 Docker image, CUDA wheel or FlashAttention install is needed.

```bash
cd baseline/experiments/nanogpt_leaderboard_2026_10_06_tpu
bash setup_tpu.sh
source .venv/bin/activate
export PJRT_DEVICE=TPU
python experiment.py verify
python experiment.py check
python experiment.py prepare --data-root /data/nanogpt-leaderboard
python experiment.py preflight --data-root /data/nanogpt-leaderboard
```

Preflight allocates the full host table, dense model, optimizer and tail-average buffers. Its 30 training steps span all five stages, sampled-softmax variants, optimizer cadence changes and embedding untie. It also evaluates one full-size validation batch. It writes `PREFLIGHT_COMPLETE.json` only after all ranks finish. Compilation can take substantial time. This is not a convergence test.

Use the receipt path printed by preflight:

```bash
python experiment.py run --data-root /data/nanogpt-leaderboard \
  --preflight-receipt results/<preflight-run>/PREFLIGHT_COMPLETE.json
```

Omitting the receipt runs a fresh preflight automatically. Source fingerprints, data hashes and package versions must match the receipt. `--results-root /path/with/space` selects another output disk. Each invocation creates a unique results directory. There is no resume support: failed training must restart.

## Results and WeightWatcher

`run_manifest.json` records source fingerprints, all ten data-shard SHA256s, seed, package versions and completion status. Per-rank hardware reports record device attributes. `metrics.jsonl` records loss, stages, token counts and wall times. `FINAL_RESULT.json` records final NLL/perplexity and whether the full evaluation reached 3.28. Generated results are ignored by Git.

After tail averaging and final validation, the unchanged reference exporter writes `weights/rank-00/model.pt` and every rank's `ngram-*.pt` chunks, including global row offsets and hashes. `weights/WEIGHTS_COMPLETE.json` appears only after complete row coverage and checksums pass. These are CPU-readable analysis weights for WeightWatcher, not resume checkpoints. Process n-gram chunks individually rather than loading the whole table.

## Verification and limitations

The full dense topology passed a 16-token forward/backward smoke test on CPU and through XLA's CPU backend. Component tests cover attention masks/gradients, rotary indexing, sparse Adam, MTP terminal gradients, optimizer cadence, schedule and model shapes. XLA CPU tests also exercise the ANVIL recurrence, embedding untie and tail averaging. A two-process Gloo test is included; local socket restrictions may skip it, but CI treats that failure as an error. Neither CPU backend verifies the full TPU workload.

```bash
python -m pip install pytest
python -m pytest -q tests
python experiment.py cpu-smoke
```

See [PORTING.md](PORTING.md) for numerical differences. BF16 replaces FP8; ordinary XLA operations replace CUDA kernels. Long packed sequences create large compiled graphs. Compilation time, HBM peak, host-transfer throughput and convergence must be measured on the TPU. **No defensible time-to-3.28 estimate is available yet. The 39.9-second H100 record is not a TPU estimate.**
