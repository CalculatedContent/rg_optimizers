# Prepare now; launch on your later TPU allocation

This directory prepares **plain SGD with momentum on stock GPT-2 Small and pinned FineWeb10B**, with a staged parameter search and passive layer measurements. It is a proposed experiment. No configuration here has been demonstrated to reach validation NLL 3.28.

## Recommended: launch the whole campaign

Use the one-command entry point from the repository root on the existing TPU VM:

```bash
python3 baseline/experiments/stock_gpt2_sgdm_tpu/run_speedrun.py start
```

It creates an isolated pinned environment, verifies data, runs the hardware and
spectral checks, executes the 11 → 4 → 1 search, stops the full finalist at the
official target, attempts one bounded longer horizon if needed, saves measurements
to GitHub as they finish, and generates reports. Candidate-by-candidate commands
below are an operational reference; they are not required for this workflow.
See [CAMPAIGN.md](CAMPAIGN.md) for the automatic budget and resume policy.

The default work directory is
`/mnt/disks/rg-data/gpt2small/stock_gpt2_sgdm_tpu`. Override it once with
`--work-dir /absolute/persistent/path` if the actual mount differs. The entry point
checks the mount and 100 GiB of initial free space before installing packages.
GitHub authentication with contents-write access must already be available through
`GH_TOKEN`, `GITHUB_TOKEN`, or `gh auth token` on that VM. Do not paste tokens into
commands, repository files or chat. [PUBLISHING.md](PUBLISHING.md) describes the
automatic write/readback check and dedicated results branch. A ChatGPT connector
session does not supply credentials to the TPU host.

`run_speedrun.py plan` only previews the work. `status` reads the latest campaign;
`resume` continues that same saved campaign after an interruption. These use the
same work directory. Optional `--gcs-uri gs://BUCKET/unique-campaign` adds verified
checkpoint backup, with separate child paths for each run.

## Lower-level individual-run reference

`launch.py` defaults to **plan**. Planning reads the selected configuration and prints the exact training/tracking commands; it starts no process. `preflight` checks the supplied environment and runs disposable kernel/capability checks. **Only the explicit `start` action launches detached training.** These low-level commands do not install packages. The higher-level `run_speedrun.py start` installs its own isolated dependencies. Neither entry point allocates a TPU, starts a stopped node, or modifies IAM. A campaign can pause only its own supervised run.

## 1. Prepare an existing single-host TPU and a separate environment

Use an existing **v5litepod-8** TPU VM with all eight chips addressable from worker0. Choose the project, zone, node, checkout, and persistent disk paths for the allocation you actually obtain. No previous project's lease, service account, bucket, or mount is assumed.

Keep run outputs and the prepared data on a persistent mounted volume. Preflight rejects a TPU output path on the root filesystem or a temporary filesystem. Verify that the volume survives the lifecycle of your allocation. The full token corpus uses about 20.7 GB. Run preflight requires 30 GiB free after data preparation; a volume with 100 GiB free allows room for data, three resumable checkpoints plus a conditional first-target checkpoint, logs, and the bounded spectral queue. The metrics worker removes a snapshot only after its measurement is committed; it does not delete the training checkpoints.

For manual individual-run work, create a **new** Python3.10–3.12 environment. The recommended campaign entry point performs this step automatically. These are commands to run later, not completed setup:

```bash
python3.12 -m venv /your/persistent/sgdm-venv
/your/persistent/sgdm-venv/bin/python -m pip install --upgrade pip
/your/persistent/sgdm-venv/bin/python -m pip install -r baseline/experiments/stock_gpt2_sgdm_tpu/requirements-tpu.txt
```

The files pin PyTorch2.6.0, PyTorch/XLA2.6.0, libtpu0.0.7.1, JAX/JAXLIB0.4.38, and WeightWatcher0.7.7. The TPU/Pallas versions follow [PyTorch/XLA v2.6.0's own package configuration](https://github.com/pytorch/xla/blob/v2.6.0/setup.py). These are reproduction pins, not a claim to use the latest runtime. Use the CPU requirements file on a machine without a TPU. Do not overlay these requirements on a shared environment with a different libtpu installation.

Set your paths; none of these assignments creates cloud resources:

```bash
SGDM_REPO=/absolute/path/to/rg_optimizers
SGDM_EXPERIMENT="$SGDM_REPO/baseline/experiments/stock_gpt2_sgdm_tpu"
SGDM_PYTHON=/your/persistent/sgdm-venv/bin/python
SGDM_DATA=/your/persistent/fineweb10B-gpt2
SGDM_CAMPAIGN=/your/persistent/sgdm-search-v1
SGDM_RUNS=/your/persistent/sgdm-runs-v1
```

Always invoke the launcher with the selected environment's interpreter. Remote launch has a separate `--python` option identifying that interpreter on the TPU VM.

## 2. Run the small CPU smoke check

The checked-in smoke configuration is deliberately tiny, uses synthetic tokens, and cannot qualify for the GPT-2/FineWeb target. It verifies execution and checkpoint handling without downloading20GB of data.

```bash
"$SGDM_PYTHON" "$SGDM_EXPERIMENT/launch.py" plan \
  --config "$SGDM_EXPERIMENT/configs/smoke.json" --data-dir "$SGDM_DATA" \
  --run-dir "$SGDM_RUNS/cpu-smoke" --device cpu --skip-spectral --min-free-gib 0.1

"$SGDM_PYTHON" "$SGDM_EXPERIMENT/launch.py" start \
  --config "$SGDM_EXPERIMENT/configs/smoke.json" --data-dir "$SGDM_DATA" \
  --run-dir "$SGDM_RUNS/cpu-smoke" --device cpu --skip-spectral --min-free-gib 0.1
```

For a smoke check that includes the real WeightWatcher worker, omit `--skip-spectral` and choose a fresh output path. Separately, `python metrics.py --check` verifies the native raw/clipped fit and ERG fields on small CPU matrices. Production launches require spectral tracking and cannot use `--skip-spectral`.

## 3. Create the parameter screen and prepare its data

```bash
"$SGDM_PYTHON" "$SGDM_EXPERIMENT/tune.py" plan --output "$SGDM_CAMPAIGN" --seed 42
"$SGDM_PYTHON" "$SGDM_EXPERIMENT/prepare_data.py" \
  --data-dir "$SGDM_DATA" --budget-tokens 134217728
```

The screen contains the declared learning-rate/batch candidates and the historical large-batch control. Configurations and stage manifests carry hashes. Each candidate receives an equal token budget and a reserved tuning validation window that is disjoint from the benchmark prefix. The planner creates files, not training jobs.

Choose one candidate from `screen/stage.json`, for example:

```bash
SGDM_CONFIG="$SGDM_CAMPAIGN/screen/configs/sgdm-b0016-lr0p03-m0p98.json"
SGDM_RUN="$SGDM_RUNS/screen/sgdm-b0016-lr0p03-m0p98"
```

This example is a candidate, not a preselected winner.

## 4. Inspect readiness, then explicitly launch that candidate

```bash
"$SGDM_PYTHON" "$SGDM_EXPERIMENT/launch.py" plan \
  --config "$SGDM_CONFIG" --data-dir "$SGDM_DATA" --run-dir "$SGDM_RUN" --device tpu

"$SGDM_PYTHON" "$SGDM_EXPERIMENT/launch.py" preflight \
  --config "$SGDM_CONFIG" --data-dir "$SGDM_DATA" --run-dir "$SGDM_RUN" --device tpu

"$SGDM_PYTHON" "$SGDM_EXPERIMENT/launch.py" start \
  --config "$SGDM_CONFIG" --data-dir "$SGDM_DATA" --run-dir "$SGDM_RUN" --device tpu
```

Start repeats the mandatory readiness checks and writes `PREFLIGHT.json` before spawning the trainer. The checks verify the pinned model source and dataset manifest, SHA256 and binary headers of the required token shards, installed runtime versions, writable storage/free space, all eight observed TPU devices, attention outputs/derivatives, one disposable full-model SGDM update at the configured microbatch shape, and the native WeightWatcher/ERG fields. A passed kernel probe does not establish convergence, full-run throughput, or numerical identity to the CUDA record.

The individual-run launcher refuses an existing output directory unless `--resume` explicitly requests the same saved checkpoint/configuration. It never accepts a `--replace` option. Use one candidate at a time on this TPU. The campaign controller applies this policy automatically, waits for training and tracking, publishes results, and then launches the next candidate. Neither path extends a cloud allocation.

The supervisor is detached, so disconnecting SSH does not intentionally terminate it. Node preemption, node deletion, lease expiry, or reboot can still terminate all processes. There is no automatic restart.

### Optional Cloud Shell entry

The same checked-out code and environment must already exist on the TPU VM. Supply your actual node, project, and zone; the launcher performs a read-only node description, requires READY/healthy single-host v5litepod-8, and uses `--worker=0`.

```bash
python3 baseline/experiments/stock_gpt2_sgdm_tpu/launch.py plan \
  --node YOUR_EXISTING_NODE --project YOUR_PROJECT --zone YOUR_ZONE \
  --remote-repo /absolute/checkout/on/tpu --python /persistent/sgdm-venv/bin/python \
  --config /persistent/sgdm-search-v1/screen/configs/sgdm-b0016-lr0p03-m0p98.json \
  --data-dir /persistent/fineweb10B-gpt2 \
  --run-dir /persistent/sgdm-runs-v1/screen/sgdm-b0016-lr0p03-m0p98 --device tpu
```

After inspecting the plan, use `preflight`, then `start` with the same supplied arguments. The paths in this command are **remote** paths. The launcher does not copy a checkout, download packages, or presume that your Cloud Shell identity and the TPU VM identity have the same permissions.

### Optional backup to a bucket you control

Add `--gcs-uri gs://YOUR_BUCKET/unique-campaign/unique-run` to the start command. Use a unique prefix per run. Without this option the launcher performs no cloud storage operation.

Only explicit start performs the small write/read/delete probe, before training begins. A403 or another failed access check prevents that start; correct the credentials/permissions outside this launcher, or explicitly choose to launch without cloud backup. An occupied prefix is refused for a new run. Resume may use the same prefix only when its saved run/configuration identity matches.

During training, non-deleting backups run independently at the requested `--backup-interval` (default 600 s). They include the latest/best/previous checkpoints and, when the official benchmark target is first met, `first_target.pt` with its sidecar. They also include the terminal marker and scientific JSON/JSONL/CSV/log outputs; pending matrix snapshots and XLA caches are excluded. Every checkpoint is staged through a stable inode and checked against its sidecar SHA256/configuration before upload. If publication races with staging, both checkpoint files are skipped and the existing remote copy is retained until another cycle succeeds. A backup failure is recorded separately and never rewrites successful training as a failed optimizer run. Inspect `BACKUP_STATUS.json`; a valid local checkpoint is not evidence that its remote copy succeeded.

## 5. Inspect, pause, and resume

Important outputs are separate:

| File | Meaning |
|---|---|
| `TRAINING_STATUS.json` / `status.json` | Training step, processed tokens, validation outcome, and whether training finished |
| `TRACKING_STATUS.json` | Expected/completed/failed spectral measurements and layer coverage |
| `BACKUP_STATUS.json` | Last independent cloud upload outcome |
| `LAUNCHER_STATUS.json` | Process outcomes and pointers to the independent status records |
| `logs/training.log`, `logs/tracking.log`, `logs/backup.log` | Separate diagnostic logs |
| `checkpoints/latest.pt` and `latest.json` | Resumable training state and checksum/identity metadata |

`training_process_succeeded` only means that the process exited cleanly. Read the training status to distinguish a pause, a spent budget, and a verified target-reaching evaluation. Tracking must also finish before a result is eligible for parameter selection. If the metrics process exits while training is running, the supervisor requests a graceful trainer pause rather than allowing its bounded snapshot queue to stall indefinitely.

To request a graceful pause, send SIGTERM to the supervisor PID recorded in this run's `LAUNCH.json`. Verify the PID and run path first. Wait for the status and latest checkpoint; avoid killing a checkpoint write. To resume after the old supervisor has exited, repeat the same `start` command with `--resume`. The original config hash, data state, optimizer momentum, RNG states, and original schedule horizon remain fixed. Changing the horizon or promoting a candidate requires a fresh run, not resume.

Invoke local `preflight` and `start` with the environment's own Python, as in the commands above. If `--python` is supplied, it must name that same interpreter; the launcher refuses to validate one environment and run another. Virtual-environment Python symlinks are preserved when starting children.

If the latest checkpoint fails its SHA256 check, stop and preserve the run directory, including both latest files, `previous.pt`/`previous.json`, and the tracking receipts/results. An interruption can occur between replacing a checkpoint and replacing its sidecar. The trainer refuses this mismatched pair and does not automatically overwrite it or roll back. A recovery must first validate the previous checkpoint's own checksum, run/configuration/source identities, and data cursor, then inspect any measurements or log events after that step. Replaying an already recorded later step with different weights or validation is rejected by the immutable snapshot check. Do not simply copy `previous.pt` over `latest.pt`; retain the evidence and review a recovery plan before publishing a verified pair or resuming a rollback.

If training ended but measurements remain, the immutable pending snapshots remain available:

```bash
"$SGDM_PYTHON" "$SGDM_EXPERIMENT/metrics.py" --run-dir "$SGDM_RUN" --watch --threads 2
```

The CPU worker sees the run directory's `TRAINING_DONE`, drains queued snapshots, and exits. A missing/failed measurement is recorded explicitly; no fabricated fit replaces it.

## 6. Promote completed candidates, then run full confirmation

After all screen runs and their metrics have completed:

```bash
"$SGDM_PYTHON" "$SGDM_EXPERIMENT/tune.py" select \
  --stage-dir "$SGDM_CAMPAIGN/screen" --results-dir "$SGDM_RUNS/screen" \
  --output "$SGDM_CAMPAIGN/refine"
```

Prepare the next budget with `prepare_data.py --budget-tokens 1073741824`, then use the same plan/preflight/start entry for each selected refinement configuration, fresh initialization and a new run directory. After refinement completes:

```bash
"$SGDM_PYTHON" "$SGDM_EXPERIMENT/tune.py" select \
  --stage-dir "$SGDM_CAMPAIGN/refine" --results-dir "$SGDM_RUNS/refine" \
  --output "$SGDM_CAMPAIGN/full"

"$SGDM_PYTHON" "$SGDM_EXPERIMENT/prepare_data.py" \
  --data-dir "$SGDM_DATA" --budget-tokens 10255073280
```

The resulting `full/winner.json` freezes the selected settings. Launch it fresh with the same entry point and a new full-run directory. Full confirmation evaluates the original10,485,760-token benchmark prefix; the goal is NLL≤3.28, corresponding to perplexity≤26.5758. Completing the10.255B-token budget without reaching that threshold must be reported as such.

`tune.py confirm --config .../full/winner.json --output .../confirmation --seeds 42 43 44` prepares independent seed configurations. It launches nothing. Report every confirmation seed; do not select a favorable seed after seeing results. The short-stage ranking may change at a longer budget, so the selected candidate is the best within the declared search—not a proven global optimum.

## Validation status of this preparation

CPU/unit checks validate configuration, data identity, launch boundaries, checkpoint/resume handling, native SGD equivalence, and the spectral worker. **This new launcher has not been executed on a TPU.** The observed device, attention, full-model update, stable changing-LR compilation, credentials, persistent storage, and data checks run automatically on the future allocation before production training starts. No TPU throughput or completion-time promise is implied.
