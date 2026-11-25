# Stock GPT-2 / FineWeb: TPU SGD + momentum experiment

Prepared on **2026-11-25** for a later Google TPU run. This is a separate experiment
beside `stock_gpt2_sgd_momentum` and `stock_gpt2_fineweb_muonclip_adamw`.

**Start here when returning:** read [LAUNCH.md](LAUNCH.md), then generate the search
plan with [TUNING.md](TUNING.md). A future request can simply say:

> Give me the run instructions for `baseline/experiments/stock_gpt2_sgdm_tpu`.
> Start with the documented hyperparameter search on our Google TPU.

The folder contains a trainer, verified data preparation, an immutable staged
SGDM search, checkpoint/resume support, an independent CPU spectral worker,
detached supervision, optional verified cloud backup, and report/plot export.
**No TPU training was launched while preparing this folder.** CPU tests and a
synthetic smoke run validate software paths; they are not FineWeb benchmark results.
The actual TPU kernel/shape preflight is required at a future launch.

## What the experiment measures

| Item | Fixed protocol |
|---|---|
| Model | Stock upstream GPT-2 Small: 12 blocks, 12 heads, width 768, 124,439,808 unique parameters |
| Vocabulary / context | GPT-2 vocabulary 50,257; context length 1,024 |
| Dataset | Original FineWeb `sample-10BT`, pinned GPT-2-tokenized shards; not Shakespeare or FineWeb-Edu |
| Optimizer | `torch.optim.SGD`, unnormalized momentum, dampening 0, Nesterov off |
| Parameters updated | All trainable parameters; tied embedding/output head updated once |
| Gradient handling | Mean-token loss; normalize accumulation; global L2 clip 1.0 once before momentum |
| Weight decay | Coupled SGD decay on matrices/embeddings only; proposed candidates use 0 |
| TPU target | One existing single-host `v5litepod-8`, eight addressable chips, PyTorch/XLA SPMD |
| Numerics | FP32 weights/momentum, BF16 TPU autocast; checked causal flash attention |
| Full training budget | 10,255,073,280 processed tokens |
| Final benchmark validation | Exact first 10,485,760 validation targets, offset 0 |
| Benchmark target | Mean validation NLL ≤ **3.28 nats/token**, equivalent to perplexity ≤ **26.5758** |

The original reference reached approximately 3.28 with **AdamW**. This new trainer
uses SGDM throughout. We have not established an exact target-reaching SGDM recipe
for this stock model/dataset/protocol. The supplied settings are **search candidates**,
not a claimed optimum. [SOURCES.md](SOURCES.md) separates published evidence,
historical settings, and our proposed choices.

Loss and perplexity describe the same objective here: `perplexity = exp(NLL)`.
The target is a reproducible benchmark threshold, not proof of the smallest
possible loss. A full-budget run can finish without reaching it; status files
report that outcome explicitly.

## Hyperparameter selection

The proposed grid combines momentum **0.98**, peak learning rates
**0.01, 0.03, 0.1, 0.3, 1.0**, global batches **16 or 64 sequences**, and weight
decay **0**. A historical hyperparameter control uses momentum **0.9**, LR **0.3**,
weight decay **3e-5**, and batch **512 sequences**. Every candidate uses a global
microbatch of 16 sequences, distributed over the eight chips; accumulation
produces its configured global batch.

The proposed schedule is 10% warmup followed by cosine decay to zero. The control
retains the historical warmup fraction (700/19,560). Each training horizon is
hashed into its configuration. Promotion starts from the same seeded initialization
with a new horizon; it does not continue a shortened cosine run whose LR has
already decayed to zero.

| Stage | Candidates | Tokens per candidate | Selection |
|---|---:|---:|---|
| Screen | 11 | 134,217,728 | Promote 4 by final tuning-window NLL |
| Refine | 4 | 1,073,741,824 | Promote 1 by final tuning-window NLL |
| Full confirmation | 1 | 10,255,073,280 | Evaluate the official benchmark target |

The planned seed-42 search processes **16,026,435,584 training tokens** if every
candidate completes. This is more work than one full run, and its runtime must
be measured on the selected TPU; a GPU record or the old large-batch run does not
establish throughput for small-batch SGDM. Optional confirmation seeds 43 and 44
each require another full run. The `confirm --seeds 42 43 44` command in the tuning
guide creates three fresh runs, including a new seed-42 repeat; use `--seeds 43 44`
when retaining an already completed matching full seed-42 run as the first replicate.
A short-budget ranking can discard a candidate
that would eventually be better; the search provides an evidence-based starting
point, not a global optimality certificate.

Tuning uses a reserved, disjoint validation window: offset **10,485,760**, length
**1,048,576** targets. The official prefix is reserved for the finalist. The
selector rejects wrong budgets, changed configurations, incomplete evaluations,
missing spectral checkpoints, and nonfinite results. Numerical divergence is
recorded and excluded rather than promoted as a missing value.

All candidates see the same continuous, sequential training-token prefix. The
loader crosses shard boundaries without discarding microbatch tails. This is
deliberately batch independent, and differs from the historical C loader's
shuffle and the old Python loader's dropped tails. These are protocol differences
from the published CUDA run; bitwise reproduction is not claimed.

## Layer measurements

At initialization, each spectral interval, and the final checkpoint, a separate
CPU worker analyzes an immutable copy of the weights. A clean pause also records
its stopping checkpoint. Every snapshot is paired with validation at the **same
optimizer step and token count**.

For each of 12 blocks, packed QKV is split into **Q, K, V**, with **O, MLP input,
and MLP output**, giving **72 transformer matrices**. The tied token embedding/output
head is measured once, plus the positional embedding: **74 matrix rows** with
the default configuration. The canonical model average includes the 72 block
matrices; embedding results remain separately identified. Bias and LayerNorm
vectors are updated by SGDM but are not matrices with these spectral-tail fits.

The worker preserves every native WeightWatcher result for two independent fits:
the original spectrum and `clip_xmax` (upper-tail/finger adjustment). Each alpha
uses the **KS statistic and tail count from that same fit**. For both versions,
outputs include:

- Arithmetic mean alpha, confidence-weighted alpha, tail-weighted alpha, and the
  combined confidence/tail-weighted mean.
- Per-matrix, per-block, and per-role results; fit status, KS distance, exact
  fitted tail count, xmin/xmax, valid counts and effective number of weighted layers.
- Native ERG gap and ERG/randomized-spectrum fields, plus all other native
  WeightWatcher fields available in the pinned version.

The combined mean is defined explicitly:

\[
w_\ell = \frac{\sqrt{n_{\mathrm{tail},\ell}}}{D_{\mathrm{KS},\ell}+0.02},
\qquad
\bar\alpha_{\mathrm{confidence,tail}} =
\frac{\sum_\ell w_\ell\alpha_\ell}{\sum_\ell w_\ell}.
\]

These confidence weights are fit-quality heuristics, not confidence probabilities
or statistical confidence intervals. Invalid or unavailable fits remain explicit;
they are excluded from aggregates with their coverage counts reported. Native
WeightWatcher 0.7.7 defines `ERG_gap = detX_num - num_pl_spikes`; the exact fitted
tail count used in weighting is stored separately. See [METRICS.md](METRICS.md)
for the definitions, column names, fitting caveats, and reconstruction rules.

Tracking is passive: no alpha-based LR controller, optimizer replacement,
layer-specific update scaling, or hidden AdamW is used. The bounded queue allows
four pending snapshots. Training waits when the CPU worker falls behind, so
required measurements are not silently skipped. Processed weight snapshots are
pruned only after their result and CSV exports are committed; measurements,
receipts, and latest/best/previous resumable checkpoints are retained. The first
official target-reaching checkpoint is retained separately when one exists.

## Commands to use later

The following commands create a plan or print instructions. They do not allocate
a TPU or begin training. Run them from this folder in the existing repository:

```bash
python tune.py plan --output /persistent/sgdm/search
python launch.py plan \
  --config /persistent/sgdm/search/screen/configs/sgdm-b0016-lr0p03-m0p98.json \
  --data-dir /persistent/sgdm/fineweb \
  --run-dir /persistent/sgdm/runs/screen/sgdm-b0016-lr0p03-m0p98 \
  --device tpu
```

`/persistent/sgdm` is an example path. Replace it with the actual persistent
mounted volume on the TPU VM. The model, data hashes, dependency versions,
output mount, space, topology, flash-attention derivatives, optimizer update,
and native spectral capabilities are checked before a real launch.

[LAUNCH.md](LAUNCH.md) contains venv setup, data preparation, node/project/zone
selection, exact `preflight` and `start` commands, optional Cloud Shell SSH,
monitoring, stopping/resuming, backup, and recovery. [TUNING.md](TUNING.md)
contains the complete sequence of candidate runs and promotion commands.
The default action of `launch.py` is `plan`; actual execution requires `start`.

After a run has finished and its metrics have drained:

```bash
python report.py --run-dir /persistent/sgdm/runs/full/SELECTED_CANDIDATE --plots
```

This creates `reports/summary.json`, final per-layer/group JSON, and PNG/PDF plots
of validation loss, perplexity, mean/weighted alpha, and ERG gap against tokens
(or `--axis step`). A missing final spectral result is reported as missing; an
older snapshot is not substituted as the final measurement. Synthetic smoke
plots are prominently labeled as non-benchmark data.

## Checkpoints and trustworthy completion

`checkpoints/latest.pt`, `best.pt`, `previous.pt`, and (after the first qualifying
hit) `first_target.pt` include model weights,
SGD momentum, Python/NumPy/Torch/XLA RNG state, exact data cursor, immutable config
hash, step/token counts, next LR factor, validation, and timing state. Their JSON
sidecars contain SHA256 integrity hashes. Writes replace files atomically.

Resume requires the original config hash, training/metric source hashes, and
device kind. Changing LR, momentum, global batch, seed, schedule, or budget starts
a new run. A soft session stop can pause without changing the scientific horizon.
`SIGINT`/`SIGTERM` finish the current optimizer update and attempt final validation
and checkpointing; a hard VM kill can recover only the last committed checkpoint.
No code can guarantee a final save after power loss or `SIGKILL`.

The first official target-reaching evaluation, best validation, latest validation,
and their checkpoints are distinguished. A later loss regression does not erase
the first hit. Active-loop timing includes validation/checkpoint/queue waits and
excludes preflight, data preparation, model setup, and downtime; it is labeled
explicitly, rather than presented as an official leaderboard timing.

| File | Meaning |
|---|---|
| `TRAINING_STATUS.json` / `status.json` | Current/terminal training state, endpoint target flag, first target hit, best validation, timing |
| `TRACKING_STATUS.json` | Live required snapshot completion, failures, pending jobs, fit warnings and coverage |
| `BACKUP_STATUS.json` | Independent cloud copy success/failure or not requested |
| `LAUNCHER_STATUS.json` | Process supervision and final independent outcomes |
| `manifest.json` / `data_receipts.json` | Actual model, code, software, device and verified data identities |
| `metrics.jsonl` | Logged training updates and exact validation events |
| `tracking/measurements/*.json` | Complete immutable per-snapshot layer results and native fields |
| `tracking/*.csv` | Reconstructed summaries and layer/group tables for analysis |

`target_met` means the latest complete official validation meets 3.28;
`target_ever_met` retains any earlier qualifying evaluation. Full training budget,
complete diagnostics, and successful backup are separate facts. A training job
finishing does not automatically mean it met the loss target or saved its cloud
backup successfully. The optional backup destination is write/read/delete-tested
before starting; uploads do not delete remote objects or alter IAM.

## Folder map and validation

| Entry | Purpose |
|---|---|
| [LAUNCH.md](LAUNCH.md) | Hardware/environment setup and operational runbook |
| [TUNING.md](TUNING.md) | Candidate grid, stage execution, selection and limitations |
| [METRICS.md](METRICS.md) | Exact alpha/ERG definitions, scopes, schemas, quality flags |
| [SOURCES.md](SOURCES.md) | Primary sources and evidence versus proposal |
| [VALIDATION.md](VALIDATION.md) | Checks actually executed during preparation and remaining hardware checks |
| `config.py`, `configs/`, `tune.py` | Immutable configurations and audited selection |
| `prepare_data.py`, `dataset.py` | Verified shard preparation and continuous resumable stream |
| `train.py`, `checkpoint.py` | Plain SGDM, exact validation, finite-budget training and recovery |
| `metrics.py`, `report.py` | Passive full-matrix diagnostics and post-run summaries/plots |
| `preflight.py`, `launch.py` | Future readiness checks, explicit launch, supervision and backup |
| `requirements-cpu.txt`, `requirements-tpu.txt` | Isolated pinned environments |
| `tests/` | Numerical, protocol, checkpoint, metric and launch boundary checks |

The trainer reuses the existing hash-verified upstream model and established
runtime in `baseline/gpt2_small/`; it does not replace those shared files. Keep
the full repository checkout, including the retained upstream source/license.
There are no production result files in this folder: results belong in the
explicit persistent run directory supplied at launch.
