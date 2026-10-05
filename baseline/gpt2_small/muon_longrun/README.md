# Fresh 25,000-update Muon trajectory

This extends the successful `muon-speedrun-muon-20261005-030026` recipe from a
new seed-1337 initialization. It never loads that speedrun's trained checkpoint.
The original run directory remains the read-only reference. The experiment
does **not** stop when validation NLL reaches 3.28.

## Exact reference and intentional changes

The reference is the modified `2024-11-10_UNetDoubleLr` transformer, **not stock
12-head GPT-2 and not MuonClip**: 162,201,642 parameters, 12 blocks, 6 heads,
width 768, context 1,024, vocab 50,304. QK normalization is present; QK clipping
and gradient clipping are absent. The benchmark dataset is pinned GPT-2-tokenized
FineWeb (`kjj0/fineweb10B-gpt2`, revision
`889765ea1f903759787add96995d81171b632d0c`), not FineWeb-Edu.

The launcher compares the actual reference manifest and hashes of the model,
optimizer, runtime, data implementation/manifest and Pallas installer. Those
files are imported unchanged. Numerical settings remain:

| Setting | Value |
|---|---|
| Hardware | One v5litepod-8 host, eight-chip SPMD |
| Attention | Verified TPU flash; no math fallback |
| Global microbatch | 64 sequences, eight per chip |
| Gradient accumulation | 8 microbatches |
| Effective batch | 524,288 tokens/update |
| Precision | BF16 activations, embedding/scalars; FP32 linear weights |
| Muon peak LR | 0.04 |
| Muon momentum | 0.85 to 0.95 over first 500 updates, then 0.95 |
| Newton–Schulz | 5 iterations; original coefficients and BF16 operation order |
| Auxiliary Adam peak LRs | Embedding 0.6; head 0.008; scalars 0.04 |
| Auxiliary Adam | betas (0.9, 0.95), eps 1e-8; foreach/fused false; TPU capturable |
| Weight decay / clipping | 0 / none |
| Initialization seed | 1337 |

Only the run horizon/schedule, measurement cadence, gradient-norm logging,
full-state recovery, and orchestration change. The no-warmup scheduler is
`min(1, max(0, (25000 - update_index) / 7500))`, where `update_index` is zero
based. Completed update 17,500 is immediately before cooldown; the next update
uses index 17,500. The last applied factor is `1/7500`; the next is zero.
The separate 500-update momentum ramp is unchanged. At step 3,000 the LR
factor remains **1**, whereas the short reference had finished cooldown.

The old full validation at step 3,000 was NLL **3.28082160949707**, perplexity
**26.5976**. That narrowly missed a strict `<=3.28` threshold. The new step-3,000
result is compared with it in `COMPARISON_3000.json`; the schedules intentionally
differ after index 2,100, so identical validation loss is not expected.

## Corpus and duration

25,000 updates process **13,107,200,000 token presentations**. All 103 pinned
training shards contain **10,255,324,043 tokens**, slightly fewer usable after
the unchanged per-shard batch truncation. The stream makes one full sequential
pass and repeats about 28%; this is not 13.1B distinct tokens. Epoch is
`tokens_seen / usable_tokens_per_full_corpus_pass`, recorded with its denominator.

The existing benchmark cache is reused. Missing pinned shards are downloaded
before training; the older FineWeb/Edu corpus is never deleted or reformatted.
At the measured reference rate, expect roughly **10–11 hours**, plus any unusual
setup/measurement overhead. This is an estimate, not a completion guarantee.
The service has a 12-hour cap, with the final 30 minutes reserved for tracker
drain/backup; the trainer saves at its earlier deadline.

## Start, inspect, stop

From a clean checkout on your authenticated **Mac terminal or Cloud Shell**:

```bash
python3 baseline/gpt2_small/muon_longrun/launch.py start
python3 baseline/gpt2_small/muon_longrun/launch.py status
python3 baseline/gpt2_small/muon_longrun/launch.py metrics
python3 baseline/gpt2_small/muon_longrun/launch.py stop
```

`start` describes the live node and its linked queued resource and prints queue
creation, node creation, maximum duration, explicit termination timestamp and
remaining hours. It requires **more than 12.5 hours**, an already-mounted data
disk with 45 GiB free, a healthy reference manifest, and an idle TPU. It never
creates/deletes an allocation or stops another running experiment. It refuses
to guess expiration from queue submission time. SSH retry is idempotent.

The launcher prints the commit, directory, service, training deadline and cloud
prefix. The record is `/mnt/disks/rg-data/gpt2small/MUON_LONG25K_LATEST.json`.
Output is under `/mnt/disks/rg-data/gpt2small/muon-long25k-s1337-<UTC timestamp>`.
`status` reports the systemd PID, latest metrics, checkpoint, and tracker state;
`metrics` also tails the scalar file. `stop` requests a checkpoint after the
current update, then drains tracking and performs backup. There is no automatic
restart. Do not delete the allocation before the stop/backup finishes.

## Measurement plan

| Output | Cadence |
|---|---|
| Train NLL, global FP32 gradient L2 norm, all LRs, phase, tokens/s, time, epoch | Every 10 updates (also first 5) |
| Full validation NLL, perplexity, top-1 token error and accuracy | Every 500 updates and every spectral snapshot |
| WeightWatcher | 0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000; then every 1000; plus 17500/final |
| Full checkpoints | Every 2500; also initial, 3000, 17500, final/clean stop |

Every validation uses the **same 10,485,760 benchmark tokens** and the unchanged
evaluator. These are validation measurements, not a separate held-out test set.
Step-0 validation and CPU WeightWatcher must complete before the first update.

All 72 hidden matrices are retained: Q, K, V, O, MLP_IN, MLP_OUT for each block.
`alpha_raw` comes only from `raw_alpha`; `alpha_clip_xmax` comes only from
WeightWatcher's clipped `alpha`. Fitting imposes no alpha=2 constraint. All
native scalar outputs are retained, including randomized/null and ERG statistics,
`alpha_weighted`, `log_alpha_norm`, matrix rank and fit diagnostics. The weighted
metrics retain WW's native clipped-alpha definition and are labelled accordingly.

Zero-initialized matrices have rank 0 and explicit unavailable/degenerate fits;
they are not discarded or assigned invented alphas. Thus 72 rows at step 0
does not imply 72 successful power-law fits. Across-matrix standard deviations
are not error bars across independent seeds.

After step 0, the CPU tracker operates on immutable weight snapshots, independently
of training RNG and the data stream. Completed JSON/CSV measurements and training
metrics are flushed to the persistent disk. Transfer/enqueue overhead and CPU WW
time are recorded separately. Above 10% measured foreground overhead, or a >10%
median-update slowdown while the CPU tracker is active, later sampling drops to
every 2,000 updates. This second criterion is a conservative contention proxy,
**not proof** that WW caused the slowdown. Early/milestone/final measurements
remain, and scientific quantities do not change. The tracker drains at exit.

## Startup and recovery

The existing eight-chip flash forward/backward gate runs first. The trainer then
verifies its initial full validation and 72-row WW measurement. It checks finite
loss and sampled gradient norm while progressing. At update 2 it captures full
state, restores **its own step-0 initialization**, replays its first two updates,
and requires exact tensor/state equality. This is an in-process recovery test,
not a new training process and not a restore from the short reference. It stops
if the comparison fails and writes `RESUME_PARITY.json` only on success.

After 100 updates it records steady timing relative to the reference, refuses a
>50% slowdown, and continues the **same process**. `STARTUP.json` is the evidence;
an allocated/active service alone does not mean those checks passed.

Checkpoints atomically contain model, both optimizer states, scheduler, completed
step, token count, CPU/Python/NumPy/XLA RNG, exact shard/offset/cycle, corpus identity
and numerical source hashes. Keep the latest two rolling checkpoints plus
permanent 0, 3000, 10000, 17500, 25000 milestones. Final cloud backup copies these
retained checkpoint files and the small scientific outputs with CRC verification.
Intermediate checkpoints/results are immediately safe on the mounted disk;
cloud backup is an exit operation, not a per-update claim.

For an **explicit recovery**, use the same commit and a compatible existing TPU:

```bash
python3 baseline/gpt2_small/muon_longrun/launch.py recover \
  --checkpoint /mnt/disks/rg-data/gpt2small/<long-run>/checkpoints/step_0010000.pt
```

Recovery writes a new directory and rejects changed scheduler, data order,
numerical source, precision/runtime identity or model shape. A 3,000-step speedrun
checkpoint has a different schema and is rejected. No recovery happens implicitly.
CPU tests establish exact state replay across a shard boundary. The live TPU
startup gate establishes initial two-update replay only if it passes; it does
not prove that every later interruption or software/hardware change is bitwise
reproducible. Retain `RESUME_PARITY.json` and the pinned environment with the data.

## Local verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=baseline/gpt2_small/src:baseline/nanogpt_one_head/src \
python3 -m pytest baseline/gpt2_small/tests/test_speedrun30.py \
  baseline/gpt2_small/tests/test_muon_speedrun.py \
  baseline/gpt2_small/tests/test_muon_longrun.py -q
```

Tests cover upstream model/optimizer parity, schedule boundaries, unchanged
sequential sampling, corpus budget, exact full-state CPU recovery, retention,
lease rejection, and real WeightWatcher raw/clipped/null fields with zero matrices.
