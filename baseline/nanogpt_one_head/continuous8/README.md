# Continuous eight-chip MuonClip experiment

This is a fresh single-host v5e-8 experiment, not a continuation of the earlier
one-layer run. The older extension changed the learning-rate phase and reset
phase diagnostics. That alone does not establish a broken checkpoint restore;
this experiment removes deliberate process/phase restarts from the scientific run.

## Protocol

The default budget is **one machine for six hours**. The alternative is two
independent machines for four hours each; they are different seeds, not multi-host
training of a single model.

- Seed 1337 by default; the two-machine option adds seed 2027. Each machine has
  one Python training process, no automatic retry or resume.
- NanoGPT: 12 layers, 12 heads, width 768, context 256, tied GPT-2 embeddings,
  no dropout, approximately 124M parameters. MuonClip with the existing QK
  clipping implementation; no alpha-based adaptive learning-rate controller.
- FineWeb-Edu `sample-10BT`, revision pinned in the YAML. Exactly 5B train,
  10M validation and 10M test tokens, split at document boundaries.
- Global microbatch 32 sequences, four accumulation steps: 32,768 tokens/update.
  Weights are replicated; batch is sharded across eight chips on ONE host.
- Fixed maximum 1,000,000 updates (32.768B token presentations, 6.5536 corpus
  equivalents). Learning rate warms up for 2,000 updates to 2e-4, follows its
  preregistered cosine to 2e-5 by update 100,000, then stays at that floor.
  No schedule is rebuilt or extended after launch. This is a new model/data
  regime, not a controlled replication of the old model.
- Google Cloud enforces a six-hour allocation limit (four hours per machine for
  the two-machine option). Setup, downloads, compilation and diagnostics consume
  allocation time. A stop is requested 30 minutes before that limit, measured
  conservatively from VM boot; it saves after the next optimizer update.
  Checkpoints also save every 500 updates. Actual throughput is measured on
  hardware. A failure may interrupt training; automatic restart remains disabled.

## Fixed measurements

512 distinct eligible documents per split, one 257-token window per document,
256 teacher-forced next-token predictions per window. Seeds, document IDs,
absolute offsets, window SHA256 and corpus hashes are recorded once and checked.
The probe consumes no training RNG. This uses the same *definition* of token
error as the earlier document audit, but a larger, new probe/corpus; its numerical
values must not be presented as an exact replication of yesterday's probe.

`token_error.csv` records top-1 token error percentages every 500 updates.
`alpha_token_error.csv` pairs errors and WeightWatcher measurements at exactly
one model state every 1,000 updates, including a model tensor hash and probe hash.
All 72 Q/K/V/O/MLP_IN/MLP_OUT matrices are retained in `spectral/layers.csv`.
Raw and clipped alpha remain separate. An aggregate is NaN if the expected
matrix count is not present; changing subsets must not create a spurious trend.
`alpha_token_error.png` plots raw mean/min alpha against token error with ordinary
regression lines, step coloring, no detrending, and no step-zero point. Repeated
checkpoints are dependent observations: Pearson r is descriptive, not causal.
The single-seed option does not provide across-seed error bars. Two seeds give
only a limited estimate of seed variability; compare matched training steps.

The test split is repeatedly monitored and is not used for checkpoint selection
or an adaptive training controller. Validation loss selects the best checkpoint.
It is a monitored test set, not an untouched final confirmation set.

## Storage and failure behavior

A dedicated 200 GB persistent disk is attached to each host and mounted at
`/mnt/disks/rg-data`. GCS is the durable experiment archive:
`gs://tpu-builders-504820-ww-continuous8/runs/ww-continuous8-pilot-20261002-sSEED/`.
The launcher requests the TPU immediately after resource checks. Data preparation
runs on the TPU VM CPU, using 16 encoding threads and the attached persistent
disk; Cloud Shell is only the submission client. Ordered parallel encoding
preserves the serial writer's token bytes and document-disjoint splits. The corpus
is uploaded and SHA256-verified before scientific training starts. This work
counts against the four/six-hour allocation. No additional CPU VM is created.
Each seed archives its corpus under a separate prefix; corpus hashes must match
before comparing independent seeds. Code commit,
resolved dependency versions, preflight report, fixed probes, metrics, spectra,
plots and logs are recorded. Full checkpoints include optimizer buffers/counters,
RNG/sampler state, learning-rate/config identity and monitoring state.

Checkpoint files upload synchronously with CRC32C checking. A receipt containing
SHA256, generation, size and identity is written only after successful upload.
`LATEST_RESUMABLE.json` only points to a full-state checkpoint with the required
resume diagnostics (or initialization). All checkpoint artifacts are retained;
large archives incur storage costs. Backup failure stops the run after retaining
the local checkpoint, rather than silently running without durable protection.
Abrupt hardware loss may lose work since the last uploaded checkpoint.

No systemd restart, continuation supervisor, reboot startup replay, or implicit
scientific directory reuse is permitted. A later manual recovery would be a
separately identified resumed run and is not exposed by this launcher. Recoverable
checkpoints do not constitute a guarantee that any future environment reproduces
TPU updates bit-for-bit.

Before the scientific process, the existing TPU preflight checks global gradients,
QK clipping, CPU/TPU evaluation agreement and restored optimizer/sampler updates,
then benchmarks the proposed large model shape. That separate disposable test
may restore a checkpoint; the scientific run always starts from step zero.
CPU tests additionally check exact continued-vs-restored weights, optimizer hashes,
sampler state, LR scheduling and metrics. Actual TPU validation must pass on the
allocated hardware before the launch script starts scientific training.

## Launch and monitor

From a clean checkout of `codex/continuous-muonclip-8`, on Cloud Shell:

```bash
bash baseline/nanogpt_one_head/continuous8/run.sh launch --machines 1 --hours 6
```

For two independent four-hour seeds, use `--machines 2 --hours 4`.

The optional `--delete-old-experiments` flag deletes earlier `ww-long-`, `ww-mem-`, `ww-mem2-` and
`ww-v6e16-` and the earlier seven-day continuous-run queues/nodes in us-west4-a and us-east5-a, their discovered attached
data disks plus `ww-full-data-20260929`, and snapshots of those disks. The exact
inventory is saved to `~/continuous8-cleanup.json`. Other project resources,
unrelated buckets and local/downloaded files are not deleted. Inventory or
permission failures stop the operation. Re-running launch does not duplicate an
existing pilot queue or restart its training. If only one of two queue submissions
succeeds, the successful request is retained and status must be checked; there is
no automatic second attempt. Disk creation or quota errors can leave a dedicated
disk that must be inspected before retrying.

```bash
python3 baseline/nanogpt_one_head/continuous8/cloudshell.py status
```

The queue may wait for capacity or fail quota validation. A startup script fetches
the exact launch commit and starts setup via a systemd service with `Restart=no`.
A persistent claim blocks reruns after reboot. The TPU uses a dedicated service
account with object-admin permission on this experiment bucket, not TPU-admin.
Consequently it cannot delete itself: early completion/failure leaves the TPU
allocated until explicit deletion or the requested four/six-hour expiry. Review status promptly.
At the published v5e Flex-start rate of $0.60/chip-hour, eight chips cost $4.80/hour,
or **$28.80 for one six-hour machine**, **$38.40 for two four-hour machines**,
plus disk/bucket/network charges. Storage is retained after TPU expiry and
continues to incur charges until explicitly deleted. Credits and
remaining balance must be checked in the project's billing account; this script
does not assert that sufficient credits remain.

## How far will the run get?

`BENCHMARK_PROJECTION.json` reports measured training-only throughput and an
optimistic upper estimate based on the remaining allocation. It excludes
monitoring and uploads. `results/progress.json`, updated at evaluation, projects
from observed throughput including training-time evaluation/diagnostics/I/O.
Neither is a guarantee. With 32,768 token presentations/update, divide processed
tokens by 32,768 for update count, or by 5B for corpus-equivalent passes. The
sampler draws random windows; token presentations are not a count of unique
tokens visited. A larger corpus reduces repeated sampling but does not guarantee
better test accuracy or that alpha will fall below two within six hours.

## Launch visibility and failure diagnosis

The launcher records each phase and any exception in `~/continuous8-launch.json`.
The `run.sh` wrapper also saves the terminal transcript in
`~/continuous8-launch.log`. These small files live in Cloud Shell HOME; the large
corpus and training job do not. After the queued request is successfully submitted,
Cloud Shell may disconnect without stopping setup or training on the TPU VM.

`cloudshell.py status` explicitly reports `NO TPU REQUEST` when none exists.
`cloudshell.py check` performs only read-only local/cloud diagnostics. VM setup
phase is recorded in `SETUP_STATUS.json`; `WORKER_STATUS.json` records worker
exit, including failures before training; `CONTINUOUS_STATUS.json` records the
scientific process outcome. If dependency installation fails before cloud upload
libraries are available, inspect the persistent disk's startup/run log over SSH.

The earlier launcher did long preparation in Cloud Shell `/tmp`. No request/data
was found after one attempt; the initial error was not retained, so its exact
cause is unknown. Cloud Shell VM disposal can lose `/tmp`, and the old empty
status output did not distinguish preparation from failure. This version moves
the long work off Cloud Shell and makes those states explicit.
