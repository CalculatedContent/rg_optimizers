# Continuous eight-chip MuonClip experiment

This is a fresh single-host v5e-8 experiment, not a continuation of the earlier
one-layer run. The older extension changed the learning-rate phase and reset
phase diagnostics. That alone does not establish a broken checkpoint restore;
this experiment removes deliberate process/phase restarts from the scientific run.

## Protocol

- One seed (1337), one Python training process, no automatic retry or resume.
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
- Stop request after 144 training wall-hours, honored at the next full
  checkpoint. Actual achievable update count is unknown until benchmarked.
  Cloud allocation is capped at seven days, including setup. Hardware can fail;
  no promise of uninterrupted infrastructure or a particular correlation is made.

## Fixed measurements

512 distinct eligible documents per split, one 257-token window per document,
256 teacher-forced next-token predictions per window. Seeds, document IDs,
absolute offsets, window SHA256 and corpus hashes are recorded once and checked.
The probe consumes no training RNG. This uses the same *definition* of token
error as the earlier document audit, but a larger, new probe/corpus; its numerical
values must not be presented as an exact replication of yesterday's probe.

`token_error.csv` records top-1 token error percentages every 500 updates.
`alpha_token_error.csv` pairs errors and WeightWatcher measurements at exactly
one model state every 2,000 updates, including a model tensor hash and probe hash.
All 72 Q/K/V/O/MLP_IN/MLP_OUT matrices are retained in `spectral/layers.csv`.
Raw and clipped alpha remain separate. An aggregate is NaN if the expected
matrix count is not present; changing subsets must not create a spurious trend.
`alpha_token_error.png` plots raw mean/min alpha against token error with ordinary
regression lines, step coloring, no detrending, and no step-zero point. Repeated
checkpoints are dependent observations: Pearson r is descriptive, not causal.
One seed does not provide across-seed error bars.

The test split is repeatedly monitored and is not used for checkpoint selection
or an adaptive training controller. Validation loss selects the best checkpoint.
It is a monitored test set, not an untouched final confirmation set.

## Storage and failure behavior

A dedicated 500 GB persistent disk is attached to the single host and mounted at
`/mnt/disks/rg-data`. GCS is the durable experiment archive:
`gs://tpu-builders-504820-ww-continuous8/runs/ww-continuous8-20261002/`.
The pinned dataset is uploaded before scientific training begins. Code commit,
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
python3 baseline/nanogpt_one_head/continuous8/cloudshell.py launch --delete-old-experiments
```

The explicit cleanup flag deletes earlier `ww-long-`, `ww-mem-`, `ww-mem2-` and
`ww-v6e16-` queues/nodes in us-west4-a and us-east5-a, their discovered attached
data disks plus `ww-full-data-20260929`, and snapshots of those disks. The exact
inventory is saved to `~/continuous8-cleanup.json`. Other project resources,
unrelated buckets and local/downloaded files are not deleted. Inventory or
permission failures stop the operation. Re-running launch does not duplicate an
existing continuous8 queue or restart its training.

```bash
python3 baseline/nanogpt_one_head/continuous8/cloudshell.py status
```

The queue may wait for capacity or fail quota validation. A startup script fetches
the exact launch commit and starts setup via a systemd service with `Restart=no`.
A persistent claim blocks reruns after reboot. The TPU uses a dedicated service
account with object-admin permission on this experiment bucket, not TPU-admin.
Consequently it cannot delete itself: early completion/failure leaves the TPU
allocated until explicit deletion or the seven-day expiry. Review status promptly.
At the published v5e Flex-start rate of $0.60/chip-hour, eight chips cost $4.80/hour,
or $806.40 for the seven-day cap, plus disk/bucket/network charges. Credits and
remaining balance must be checked in the project's billing account; this script
does not assert that sufficient credits remain.
