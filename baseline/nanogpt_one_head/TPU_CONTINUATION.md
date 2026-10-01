# Continuing MuonClip for an extended period

`rg_nanogpt_one_head.muonclip_continue` can run an explicit number of additional
updates or keep launching successive segments until stopped. It imports a full
training checkpoint into a new run identity. The original experiment remains
intact. Do not edit the original run's duration and try to bypass its fingerprint.

## Behavior

- Preserves model weights, Muon momentum, Adam moments and per-parameter update
  counters, RNG state, and the training sampler. The new segment resets only
  logging intervals and its own best-validation selection. Its step zero is the
  imported trained model, not a new random initialization.
- Defaults to the parent's minimum learning rate, held constant: **2e-5** for
  `muonclip_tpu_spmd_long.yaml`. There is no repeated LR ramp. `--learning-rate`
  can specify a different positive constant for an explicitly new series.
  This is a conservative continuation setting, not a claim of optimality.
- Defaults to one million additional updates per segment, with no overall cap
  under `--forever`. The global counter includes all parent updates; the first
  extension of the 2,150,000-update baseline starts at global step 2,150,000.
- Inherits four-chip SPMD, precision, batch size, gradient clipping, optimizer
  algorithm, dataset, and probe seeds. Single-host/four-chip support is retained;
  this change does not add multi-host training or change the model architecture.
- Inherits full rolling checkpoints every 500 updates, train/validation
  evaluation every 1,000, and WeightWatcher/epoch snapshots every 10,000.
  Segment endpoints are measured too. Raw and clipped alpha remain separate.
- Adds fixed test-probe loss, perplexity, bits/token, top-1 and top-5 token
  accuracy every 10,000 updates by default (`--test-interval-steps`). For the
  current configuration, each probe contains 8 x 32 x 256 = **65,536 token
  predictions**, using the same test windows across all segments. This is not
  evaluation of every token in the million-token test split.
- Validation loss alone selects each segment's best checkpoint. Test outcomes
  do not drive automatic checkpoint selection or an LR controller. Once humans
  monitor test results to choose further experiments, the test set is no longer
  an untouched final holdout. BLEU remains an endpoint diagnostic.
- Keeps all CSV histories, spectral outputs, manifests, configs, test reports,
  and lineage. By default, keeps full and epoch checkpoints for the latest
  **three completed segments plus the active segment**. Older segment checkpoint
  files are removed only after completion validation; their archival marker is
  `checkpoints_pruned.json`. The original parent run is never pruned.
- Pauses at a saved checkpoint on a stop request or below **5 GiB free disk**.
  Metrics and logs still grow over time, so storage is not literally unlimited.
  Exit code 75 means a deliberate pause; it does not consume the retry budget.
- Uses the existing progress-aware worker recovery and a writer lock inherited
  by workers. A second driver cannot write the same series while either the
  first driver or its worker holds the lock. Status records after host loss may
  be stale; resuming acquires the lock and checks the durable state.

## Install after the current baseline finishes

From **Cloud Shell**, enter the current TPU VM:

```bash
gcloud compute tpus tpu-vm ssh ww-long-20260930-232752-node \
  --project=tpu-builders-504820 --zone=us-west4-a
```

Run the following **inside the TPU VM**. Use the existing pinned dependencies;
this code adds no external packages. Keep the current source checkout available
for the original experiment. Do not reinstall or update dependencies mid-run.

```bash
set -e
source "$HOME/.config/rg_optimizers/tpu_env.sh"
mountpoint -q /mnt/disks/rg-data
export RG_PARENT=/mnt/disks/rg-data/muonclip-spmd-long/muon_clip/seed_1337
test -s "$RG_PARENT/run_complete.json"
test -s "$RG_PARENT/checkpoint_final.pt"
cd /mnt/disks/rg-data
git clone --branch codex/tpu-long-continuation \
  https://github.com/CalculatedContent/rg_optimizers.git rg_optimizers_continuation
cd rg_optimizers_continuation/baseline/nanogpt_one_head
export PYTHONPATH="$PWD/src"
git rev-parse HEAD > /mnt/disks/rg-data/continuation-source-commit.txt
python3 -m pip freeze > /mnt/disks/rg-data/continuation-environment.txt
export RG_DATA=/mnt/disks/rg-data/rg-nanogpt-one-head/data
export RG_SERIES=/mnt/disks/rg-data/muonclip-extended
```

Confirm the previous supervisor/worker exited before using the chips. The
completion marker is written near the end; `pgrep -af rg_nanogpt_one_head` can
show whether a worker still owns the TPU. Separate series roots have separate
locks; they do not arbitrate TPU ownership across independent experiments.

First rerun the four-chip numerical/resume acceptance check, then a short
continuation from the actual trained checkpoint. These commands use the TPU
and should run sequentially after the previous experiment has exited:

```bash
python3 -m rg_nanogpt_one_head.tpu_spmd_check --backend tpu --chips 4 \
  --output /mnt/disks/rg-data/continuation-spmd-check.json

python3 -u -m rg_nanogpt_one_head.muonclip_continue start \
  --series-root /mnt/disks/rg-data/muonclip-continuation-smoke \
  --from-checkpoint "$RG_PARENT/checkpoint_final.pt" \
  --data-root "$RG_DATA" --device tpu \
  --additional-steps 20 --segment-steps 20 --test-interval-steps 10
```

The smoke command returns zero only after its full completion audit succeeds.
Its worker log is in `segments/segment_000001/launch.log` under the smoke root.
If it fails, inspect that log before launching the long series. The original
checkpoint is unchanged by the smoke test.

## Start, monitor, pause, and resume

Start the extended series from the original final checkpoint:

```bash
python3 -u -m rg_nanogpt_one_head.muonclip_continue start \
  --series-root "$RG_SERIES" \
  --from-checkpoint "$RG_PARENT/checkpoint_final.pt" \
  --data-root "$RG_DATA" --device tpu --forever \
  --segment-steps 1000000 --test-interval-steps 10000 \
  --keep-segments 3 --min-free-disk-gb 5 --background
```

`--background` detaches the driver from SSH. The command prints its PID and
driver-log path (`/mnt/disks/rg-data/muonclip-extended.driver.log`). The child
performs startup validation; check the log and status after launching. Detailed
training logs are in each segment's `launch.log`. To run a finite extension,
replace `--forever` with `--additional-steps 1000000`.

```bash
python3 -m rg_nanogpt_one_head.muonclip_continue status --series-root "$RG_SERIES"
python3 -m rg_nanogpt_one_head.monitor --series-root "$RG_SERIES" \
  --interval 60 --no-clear
```

The monitor joins the baseline and all extensions using cumulative steps and
epochs. It reports latest validation accuracy, latest measured test accuracy,
recent test history, and layer alphas. Blank test entries at intermediate
validation steps mean no test probe ran at that step. In the recent history
table, accuracy is a fraction; the headline uses percent.

Export the full retained trajectory for later plotting:

```bash
python3 -m rg_nanogpt_one_head.monitor --series-root "$RG_SERIES" --once \
  --export-metrics /mnt/disks/rg-data/muonclip-extended-history.csv
```

Request a deliberate pause, then wait for `status` to report `paused` and the
worker to exit before detaching the disk or replacing the TPU:

```bash
python3 -m rg_nanogpt_one_head.muonclip_continue stop --series-root "$RG_SERIES"
python3 -m rg_nanogpt_one_head.muonclip_continue status --series-root "$RG_SERIES"
```

Resume later using the **same checkout/commit, dependencies, mount path and
four-chip configuration**. Ordinary in-segment resumes retain strict runtime
and protocol checks. The resume command removes the stop request and continues
the active segment from its full checkpoint:

```bash
source "$HOME/.config/rg_optimizers/tpu_env.sh"
cd /mnt/disks/rg-data/rg_optimizers_continuation/baseline/nanogpt_one_head
export PYTHONPATH="$PWD/src"
export RG_SERIES=/mnt/disks/rg-data/muonclip-extended
python3 -u -m rg_nanogpt_one_head.muonclip_continue resume \
  --series-root "$RG_SERIES" --background
```

Resume uses the options recorded in `series.json`; it does not reschedule a
running series from new CLI hyperparameters. For a deliberate LR change, pause
and start a **new series root** from the saved full checkpoint, specifying
`--learning-rate`. Preserve both roots. Model-only epoch snapshots cannot
restart optimization; full `checkpoint_latest.pt` and `checkpoint_final.pt` can.

## TPU allocation lifetime

The software's `--forever` option does not extend the TPU allocation. The
existing runbook requested 72 hours. Check its actual termination timestamp
from **Cloud Shell**:

```bash
gcloud alpha compute tpus queued-resources describe ww-long-20260930-232752 \
  --project=tpu-builders-504820 --zone=us-west4-a --format=yaml
```

[Google's Flex-start documentation](https://docs.cloud.google.com/tpu/docs/request-using-flex-start)
states that VMs are deleted at the requested duration and that requests can be
up to seven days. A longer experiment must span allocations. Preserve the data
disk, mount it at the same path on the replacement VM, reproduce the pinned
environment, then use `resume`. The supervisor does not provision cloud VMs or
reattach disks. An abrupt termination can lose updates after the last durable
checkpoint; it does not require restarting the experiment from initialization.

No alpha-targeting feedback controller is enabled. Continued training may
change accuracy and spectra, but cannot guarantee every fitted alpha reaches 2.
