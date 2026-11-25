# Automatic AdamW/SGDM magnitude-graft FineWeb campaign

`run_speedrun.py start` prepares the environment and starts the complete finite
search, finalist training, layer measurement, GitHub publication and final
verification workflow. There are no separate candidate launch commands or
manual promotion decisions in this workflow.

The production experiment uses the pinned stock GPT-2 small model, GPT-2
tokenization and FineWeb data described in [README.md](README.md). It trains
with the prior magnitude-only graft: AdamW directions and moments, scaled to
same-history SGDM update norms for physical transformer matrices. Its official
target is validation negative log-likelihood **at most 3.28 nats/token**, equivalent to perplexity
**at most `exp(3.28)`, approximately 26.5758**, evaluated on the exact first
10,485,760 benchmark validation tokens. A shorter validation, a synthetic test
or a low training loss cannot qualify.

The workflow searches a declared set of hyperparameters and reports the best
tested configuration. It does not certify a globally optimal learning rate,
reference-LR ratio, loss or perplexity, and it cannot promise in advance that
this optimizer will reach the reference loss within the bounded compute budget.
A missed target is an explicit, fully archived outcome.

This is a single-seed TPU quality-target experiment. The official speed-record
rules also require timing on comparable benchmark hardware and repeated-run
statistical evidence. The controller's `successful` flag certifies the recorded
target, diagnostic and publication conditions of this campaign, not acceptance
as an official leaderboard record.

## One command on the prepared TPU VM

Use an existing single-host, eight-chip TPU VM with its persistent disk mounted,
a clean complete checkout of the committed experiment, and GitHub write access
already configured on that machine. From the repository root:

```bash
python baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py start
```

The default work directory is
`/mnt/disks/rg-data/gpt2small/stock_gpt2_adamw_sgdm_tpu`. Set it once if the actual
persistent mount differs:

```bash
python baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py start \
  --work-dir /your/persistent/mount/stock_gpt2_adamw_sgdm_tpu
```

The entry point creates an isolated pinned Python environment, makes a unique
campaign ID, and records the latest campaign under the work directory. The
campaign checks the source and dependencies, proves GitHub write/read access,
downloads and validates the required data, checks TPU topology and the actual
optimizer kernel, then launches runs sequentially. The local supervisor is
detached, so closing the SSH session does not end training. No cloud resource
allocation, permission change or automatic merge into the source branch occurs.

GitHub authentication may come from `GH_TOKEN`, `GITHUB_TOKEN`, or an existing
`gh` login. The token must have repository Contents read/write access. Credentials
are not written to result files or command arguments. The defaults are:

| Setting | Default |
|---|---|
| Repository | `CalculatedContent/rg_optimizers` |
| Result branch | `adamw-sgdm-tpu-results` |
| Campaign prefix | `baseline/experiments/stock_gpt2_adamw_sgdm_tpu/results/<campaign_id>` |
| Optimizer identity | `adamw_sgdm_graft_v1` |
| Generated campaign-name prefix | `adamw-sgdm-` |
| Production device | TPU |
| Seed | 42 |
| Metric worker CPU threads | 2 |
| Initial free-space minimum | 150 GiB |
| Per-run free-space minimum | 30 GiB |
| Optional checkpoint backup | Disabled unless `--gcs-uri` is supplied |

The 150 GiB initial check reserves space for Adam's two moment tensors, retained
checkpoint generations across up to 14 runs, the corpus, dependencies and layer
artifacts. Each later run still requires 30 GiB free before launch. These are
startup/headroom checks; the actual retained artifact sizes determine disk use.

The result branch is created from the pinned source commit if it does not yet
exist. Existing unrelated branch content is preserved. See [PUBLISHING.md](PUBLISHING.md)
for the exact archive and permission-check behavior, and [LAUNCH.md](LAUNCH.md)
for VM preparation and operational requirements.

This folder is an explicitly recorded sister fork of the prepared SGDM engine.
`REUSE_PROVENANCE.json` records the inherited source commit and file hashes.
The original SGDM folder is unchanged. Package-qualified internal imports,
separate optimizer/configuration identities, the distinct result prefix and the
distinct default work directory prevent the two campaigns from sharing runtime
modules, checkpoints or result namespaces accidentally.

## Exact finite search and stopping policy

Nine candidates combine native Adam peak learning rates `0.0003`, `0.001` and
`0.003` with reference-LR ratios `100`, `400` and `1000`. The reference LR is the
ratio times the current native Adam LR. The earlier Shakespeare pair
`0.001 × 400` is included as a proposed FineWeb setting, without a transferred
convergence or optimality claim.

All candidates use global batch 64, microbatch 16, context length 1,024,
`beta1=0.9`, `beta2=0.99`, Adam epsilon `1e-8`, norm denominator floor `1e-30`,
zero weight decay, clipping at 1.0, 10% warmup and cosine decay. Grafting applies
to four physical matrices per transformer block, treating packed QKV jointly;
embeddings, normalization vectors and biases retain native AdamW behavior.
See [TUNING.md](TUNING.md) for the exact equations. Full configuration JSON is
saved before execution.

| Stage | Candidates | Maximum tokens per candidate | Selection or stopping rule |
|---|---:|---:|---|
| Screen | 9 | 134,217,728 | Complete equal token horizons; promote the three lowest valid endpoint tuning losses |
| Refine | 3 | 1,073,741,824 | Fresh runs at equal token horizons; promote the lowest valid endpoint tuning loss |
| Full finalist | 1 | 10,255,073,280 | Fresh run; stop at the first qualifying official benchmark evaluation |
| Extended finalist, only if needed | 1 | 20,510,146,560 | One fresh run of the same selected settings with a new longer schedule; stop at the first qualifying official evaluation |

Screen and refine use a disjoint 1,048,576-token tuning window after the official
validation prefix. They are never stopped early for reaching the benchmark
threshold. Failed/nonfinite candidates cannot win promotion. Each completed
candidate must have complete required diagnostics before selection can use it.

The default maximum sum of planned training horizons is therefore:

```text
9 × 134,217,728 + 3 × 1,073,741,824
  + 10,255,073,280 + 20,510,146,560
= 35,194,404,864 training tokens.
```

This is at most 14 planned training runs, including the optional extended
attempt. Evaluation, WeightWatcher analysis and checkpoint I/O add work beyond
those training tokens. Replaying work lost after a hard interruption can add
compute; the token total is not an accounting limit on hardware time.

Use `--max-full-budget-multiplier 1` on a new campaign to disable the extended
attempt. Its maximum planned total becomes 14,684,258,304 tokens. This choice is
saved immutably and cannot be changed by resuming the same campaign. The default
multiplier of 2 authorizes one doubled-horizon fresh attempt, not an unbounded
series of retries or ever-longer training.

The full and extended configurations use `stop_at_target: true`. The first
exact official evaluation meeting the threshold is saved as the endpoint, with
matching model checkpoint and required layer snapshot. The campaign waits for
diagnostics and verified GitHub publication before calling that endpoint a
successful benchmark result. A saved best loss or target flag from another step
cannot substitute for the actual endpoint.

## Layer measurements and incremental persistence

Every run measures initialization, configured spectral intervals and the terminal
training endpoint. Full and extended runs use intervals of 131,072,000 training
tokens, plus the exact final/early-target endpoint. Every measurement retains
its step, token count, run identity, scientific configuration hash and paired
validation result.

The collector saves all native WeightWatcher fields for raw and upper-tail
adjusted fits, plus the matrix/group/model summaries needed for average layer
alpha, confidence-weighted tail-adjusted alpha, ERG gap and coverage analysis.
Transformer matrix averages use the 72 logical transformer matrices, with Q/K/V
measured separately even though the optimizer grafts packed QKV jointly. This
is distinct from the 48 physical matrices receiving the graft. The two embeddings
are also measured and identified separately. Invalid fits remain explicit with
their reasons and coverage counts. A complete diagnostic inventory is distinct
from all fits being statistically valid. See [METRICS.md](METRICS.md) for formulas,
which values enter each average, and the meaning of the fit-quality weights.

The campaign polls for completed measurements and publishes every newly
committed snapshot, or refreshes publication after 30 seconds by default. Each
published measurement contains the complete native JSON and per-step layer/group
CSV, with a remote commit and verified file hashes. Final publication also
includes aggregate CSV tables, summary JSON and generated PNG/PDF plots. Training
status, selected settings, promotion decisions and the campaign report are
published alongside the measurements.

GitHub stores the scientific measurements and reports. Resumable model/optimizer
checkpoints stay on the persistent disk; optional GCS backup is independent.
CPU tensor snapshots may be pruned only after their diagnostic output is safely
committed locally. They are not needed to republish the resulting measurement.

## Status, resume and failure handling

Use the same work directory to inspect or resume the latest campaign:

```bash
python baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py status
python baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py resume
```

Add `--work-dir` when using a nondefault mount, and `--campaign-id` to choose an
older campaign. No candidate-specific command is required. The saved manifest
determines the device, source commit, data paths, candidate settings, horizons,
publication destination and finite recovery policy.

| Event | Automatic behavior |
|---|---|
| Candidate reaches its equal-budget screen/refine endpoint | Verify endpoint and complete layer inventory; publish and consider it for promotion |
| Full finalist reaches the official target | Save the target endpoint; drain diagnostics; publish and verify before success |
| Full finalist misses its budget | Launch the one fresh doubled-horizon attempt if enabled |
| Candidate diverges/nonfinite loss | Archive the failure and all already-requested diagnostics; exclude it from selection |
| Training pauses with a resumable checkpoint | Resume the identical configuration/horizon, within the restart limit |
| Recoverable launcher/training failure with a checkpoint | Preserve records and restart, at most two restarts per planned run by default |
| Diagnostics remain incomplete after training ends | Make one explicit retry of the failed/pending diagnostic work; fail visibly if it remains incomplete |
| GitHub publication fails | Request a graceful pause, then attempt publication at most five times, 10 seconds apart by default |
| GitHub publication recovers | Continue from the saved checkpoint and unchanged schedule |
| Publication retries are exhausted | Leave an explicit incomplete campaign with local results available for one-command resume |
| No valid candidates remain or source/identity checks fail | Stop with an explicit error and preserve the evidence |

A pause is graceful: the active run must finish its current safe checkpoint and
diagnostic drain. A hard interruption without a resumable checkpoint is reported
as an operational failure rather than silently deleting the run or fabricating a
restart. The supervisor verifies process identity before signaling a recorded
PID, so a stale PID is not treated as the old run.

Resume rechecks the optimizer family, exact source file hashes and GitHub write
permission.
Every hashed source file must belong to the pinned source commit, and source
hashes are checked again before each fresh or resumed run is launched.
Explicitly changing saved scientific or operational options is refused; create
a new campaign for changed settings. Already completed runs are audited and
republished idempotently without training them again. A resumed run retains the
same learning-rate horizon and optimizer state. The optional doubled-horizon
attempt has a new configuration and fresh initialization, preserving the
interpretation of its longer learning-rate schedule.

The bootstrap strips inherited Python package paths before creating/installing
the venv or starting its controller, while preserving device and authentication
settings. An explicit higher `--min-free-gib` on resume adds a stricter startup
space check; it does not weaken or change the saved per-run minimum.

## Files and final interpretation

Within `<work-dir>/campaigns/<campaign_id>/`:

| File or directory | Purpose |
|---|---|
| `CAMPAIGN.json` | Immutable campaign identity, source hashes, paths, policy and planned maximum training tokens |
| `CAMPAIGN_STATUS.json` | Current state, active run, attempts, audited run records and success/publication flags |
| `search/` | Immutable stage configurations, promotion rankings and chosen finalist settings |
| `runs/<stage>/<candidate>/` | Checkpoints, native measurements, CSV tables, per-run reports and publication receipts |
| `logs/campaign.log` | Detached supervisor output |
| `FINAL_REPORT.json` | Actual target outcome, selected settings, best official validation, endpoint metrics, layer summaries and every run record |
| `FINAL_PUBLICATION.json` | Final remotely verified campaign publication receipt |

Production `successful: true` requires all three: an exact official target
endpoint, complete required layer measurement coverage, and verified remote
publication of the runs and final report. `workflow_complete: true` alone is not
a benchmark success. The final outcome `budget_exhausted_target_not_met` means
the authorized finite campaign completed without reaching 3.28; all measured
losses, perplexities, settings and layer records remain available for analysis.
The final report includes both the best valid full-run endpoint and the best
observed benchmark validation, retaining their precise steps and token counts.

## Preview and synthetic verification

The top-level preview is read-only: it does not install dependencies, allocate
hardware, download data, publish files or start training.

```bash
python baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py plan
```

For a small real CPU workflow test, the explicit smoke mode runs two tiny
synthetic candidates followed by their selected finalist. It uses two and four
optimizer updates, respectively, and cannot qualify as a FineWeb benchmark:

```bash
python baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py start \
  --work-dir /absolute/local/path/adamw-sgdm-smoke --smoke --no-publish --min-free-gib 0
```

Omit `--no-publish` to test the actual GitHub publication path using an authorized
dedicated smoke campaign namespace. Production publication cannot be disabled.
Smoke completion is always labeled `smoke_completed_not_a_benchmark`, with
`benchmark_target_met: false` and `successful: false`.

The lower-level controller is also available inside the prepared environment:

```bash
python campaign.py plan \
  --campaign-dir /persistent/adamw-sgdm/campaigns/ID \
  --data-dir /persistent/adamw-sgdm/fineweb
python campaign.py start --campaign-dir /persistent/adamw-sgdm/campaigns/ID
python campaign.py status --campaign-dir /persistent/adamw-sgdm/campaigns/ID
python campaign.py resume --campaign-dir /persistent/adamw-sgdm/campaigns/ID
```

Unlike the top-level read-only preview, lower-level `campaign.py plan` writes the
immutable campaign and first-stage configuration files. It still starts no
training and makes no publication call. For a partial synthetic test checkout,
`--source-sha` can supply an explicit full source SHA; production requires the
actual complete clean checkout.

Preparation verification includes orchestration contract tests for automatic
9-to-3-to-1 promotion, early official target handling, the one bounded longer
attempt, missed-target outcomes, publication gates/retries, source/settings
immutability, resumed-launch state, process identity and explicit synthetic
classification. Those simulated endpoint tests verify the controller, not graft
convergence. A real small trainer/WeightWatcher integration is separately labeled
as synthetic; a future production TPU campaign is still needed to measure the
actual benchmark result and runtime.
