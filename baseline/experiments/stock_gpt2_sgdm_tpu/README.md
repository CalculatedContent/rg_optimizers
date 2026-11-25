# Stock GPT-2 / FineWeb: TPU SGD + momentum experiment

Prepared on **2026-11-25** for a later Google TPU run. This is a separate experiment
beside `stock_gpt2_sgd_momentum` and `stock_gpt2_fineweb_muonclip_adamw`.

**One command on the prepared TPU VM runs the complete campaign:**

```bash
python3 baseline/experiments/stock_gpt2_sgdm_tpu/run_speedrun.py start
```

The entry point creates its isolated pinned environment, verifies GitHub writing
and the FineWeb data, runs the TPU preflight, trains and selects candidates,
attempts the official loss endpoint, publishes every completed layer measurement,
and generates final reports. The default persistent work directory is
`/mnt/disks/rg-data/gpt2small/stock_gpt2_sgdm_tpu`; use `--work-dir /absolute/path`
once if the mounted volume is elsewhere. The existing VM, persistent disk and
GitHub write credentials must be available. See [CAMPAIGN.md](CAMPAIGN.md).

A future request can simply say:

> Give me the run instructions for `baseline/experiments/stock_gpt2_sgdm_tpu`.
> Start the automated SGD-with-momentum campaign on our Google TPU.

The campaign automatically executes the immutable staged search; no separate
candidate launches, promotion commands, metric uploads or plot commands are
needed. Completed measurements are committed incrementally to the dedicated
[`sgdm-tpu-results` branch](https://github.com/CalculatedContent/rg_optimizers/tree/sgdm-tpu-results)
under this experiment's `results/<campaign-id>/` directory. That branch and its
first campaign namespace are created and verified at the future start; no
production result is implied by this prepared code.

**No TPU training was launched while preparing this folder.** CPU tests and a
synthetic smoke run validate software paths; they are not FineWeb benchmark results.
The actual TPU kernel/shape/compilation preflight runs automatically at launch.

## What the experiment measures

| Item | Fixed protocol |
|---|---|
| Model | Stock upstream GPT-2 Small: 12 blocks, 12 heads, width 768, 124,439,808 unique parameters |
| Vocabulary / context | GPT-2 vocabulary 50,257; context length 1,024 |
| Dataset | Original FineWeb `sample-10BT`, pinned GPT-2-tokenized shards; not Shakespeare or FineWeb-Edu |
| Optimizer | Conventional SGD equations and state: unnormalized momentum, dampening 0, Nesterov off; tensor-LR implementation for XLA |
| Parameters updated | All trainable parameters; tied embedding/output head updated once |
| Gradient handling | Mean-token loss; normalize accumulation; global L2 clip 1.0 once before momentum |
| Weight decay | Coupled SGD decay on matrices/embeddings only; proposed candidates use 0 |
| TPU target | One existing single-host `v5litepod-8`, eight addressable chips, PyTorch/XLA SPMD |
| Numerics | FP32 weights/momentum, BF16 TPU autocast; checked causal flash attention |
| First full training horizon | Up to 10,255,073,280 processed tokens; stop on the first qualifying evaluation |
| Automatic fallback | One fresh 20,510,146,560-token horizon if the first finalist misses; also stops at the target |
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

`optim.py` implements the same conventional PyTorch SGD momentum and coupled-decay
updates, with tensor multiplication for the changing learning rate. This avoids
the host scalar extraction in PyTorch 2.6's scalar `add_` argument. Numerical parity
and checkpoint interchange are tested against native SGD. The TPU preflight
requires stable compilation over changed-LR updates after warmup. The experiment
reproduces the stock GPT-2/FineWeb quality target; current record-setting modified
nanoGPT architectures and CUDA timings are different protocols.

This campaign reports single-seed attainment of the shared quality threshold.
An official speed-record submission additionally needs timing on the benchmark's
hardware and repeated-run statistical evidence under its current rules. Optional
confirmation seeds are documented below; a TPU target hit alone is not a certified
leaderboard record.

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
| Full confirmation | 1 | Up to 10,255,073,280 | Stop at a complete official validation NLL ≤3.28 |
| Conditional fallback | 1 fresh run | Up to 20,510,146,560 | Run only if the first full finalist misses; same target |

The initial seed-42 search processes at most **16,026,435,584 training tokens**.
The default policy allows **36,536,582,144 tokens** including the fallback, whose
longer schedule starts from the same seed rather than resuming a decayed-to-zero
schedule. The fallback cycles through the verified training corpus when needed.
Use `--max-full-budget-multiplier 1` at the initial start to disable it.
This is more work than one full run, and its runtime must
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

The campaign publishes the full native measurement JSON, per-step layer/group
CSV, and a cumulative summary as each snapshot completes. It verifies Git blob
hashes before recording remote success. Final aggregate tables and report JSON
are published after the worker drains. Binary checkpoints remain on the persistent
volume, with optional GCS backup. GitHub receives the analysis data, not the large
weight tensors. Publication failures trigger bounded retries and a clean pause;
they cannot silently turn into a successful benchmark. See [PUBLISHING.md](PUBLISHING.md).

## Start, inspect, or resume later

From the repository root on the existing TPU VM:

```bash
python3 baseline/experiments/stock_gpt2_sgdm_tpu/run_speedrun.py plan
python3 baseline/experiments/stock_gpt2_sgdm_tpu/run_speedrun.py start
```

`plan` is an optional read-only preview. `start` does the work; it does not allocate
a TPU or alter cloud access. The controller detaches from the SSH session and
records the campaign directory automatically. A single `status` action reads its
progress, and `resume` restores the last campaign after an interruption:

```bash
python3 baseline/experiments/stock_gpt2_sgdm_tpu/run_speedrun.py status
python3 baseline/experiments/stock_gpt2_sgdm_tpu/run_speedrun.py resume
```

Use the same `--work-dir` override for these commands if one was used initially.
An ordinary successful campaign needs neither monitoring nor manual stage
promotion. Operational failures remain explicit and retain recoverable files.
[CAMPAIGN.md](CAMPAIGN.md) describes the controller; [LAUNCH.md](LAUNCH.md) retains
the lower-level hardware and individual-run reference, and [TUNING.md](TUNING.md)
documents the scientific selection protocol.

Reports are automatic. Each completed run creates `reports/summary.json`, final
per-layer/group JSON, and PNG/PDF plots
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
| `CAMPAIGN_STATUS.json` / `FINAL_REPORT.json` | Automatic search progress, selected configurations, target result, measured endpoints and finite budget outcome |
| `PUBLISH_STATUS.json` / `FINAL_PUBLICATION.json` | Verified incremental/final GitHub publication, commit IDs, backlog or error |
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
| [CAMPAIGN.md](CAMPAIGN.md), `run_speedrun.py`, `campaign.py` | One entry point, automatic promotion, target stop, fallback and recovery |
| [PUBLISHING.md](PUBLISHING.md), `publish_metrics.py` | Incremental full-layer GitHub publication and independent remote verification |
| [LAUNCH.md](LAUNCH.md) | Hardware/environment setup and operational runbook |
| [TUNING.md](TUNING.md) | Candidate grid, stage execution, selection and limitations |
| [METRICS.md](METRICS.md) | Exact alpha/ERG definitions, scopes, schemas, quality flags |
| [SOURCES.md](SOURCES.md) | Primary sources and evidence versus proposal |
| [VALIDATION.md](VALIDATION.md) | Checks actually executed during preparation and remaining hardware checks |
| `config.py`, `configs/`, `tune.py` | Immutable configurations and audited selection |
| `prepare_data.py`, `dataset.py` | Verified shard preparation and continuous resumable stream |
| `train.py`, `optim.py`, `checkpoint.py` | Plain SGDM equations with dynamic tensor LR, exact validation, finite-budget training and recovery |
| `metrics.py`, `report.py` | Passive full-matrix diagnostics and post-run summaries/plots |
| `preflight.py`, `launch.py` | Future readiness checks, explicit launch, supervision and backup |
| `requirements-cpu.txt`, `requirements-tpu.txt` | Isolated pinned environments |
| `tests/` | Numerical, protocol, checkpoint, metric and launch boundary checks |

The trainer reuses the existing hash-verified upstream model and established
runtime in `baseline/gpt2_small/`; it does not replace those shared files. Keep
the full repository checkout, including the retained upstream source/license.
There are no production results from this setup work. Future run outputs live in
the persistent campaign directory, with verified metric copies on the results
branch. A campaign is successful only after the full official target is reached,
required layer diagnostics finish, and final GitHub publication is confirmed.
