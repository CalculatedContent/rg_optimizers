# Launch and recover the modified AdamW TPU campaign

## Prepared VM entry point

From a clean complete checkout of `CalculatedContent/rg_optimizers` on the existing
single-host `v5litepod-8` TPU VM:

```bash
python3 baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py start
```

The bootstrap chooses Python 3.10–3.12, creates its own pinned virtual environment,
prepares verified FineWeb data, and starts a detached campaign controller. It strips
inherited Python package paths before running environment/controller subprocesses.
The command neither creates a TPU nor changes cloud IAM or GitHub permissions.

The default persistent work directory is
`/mnt/disks/rg-data/gpt2small/stock_gpt2_adamw_sgdm_tpu`. Override it with
`--work-dir /absolute/persistent/path` if the actual mount differs. The launch
checks a persistent non-root mount and 150 GiB of initial free space; each run
also checks its saved 30 GiB free-space threshold after data preparation. The
dataset is about 20.7 GB. A checkpoint contains FP32 weights and both Adam moments
(about 1.39 GiB before small metadata). Up to three retained generations across
14 runs can occupy about 58 GiB. The startup allowance also covers dependencies,
the bounded diagnostic queue, exports and the later free-space checks.

GitHub contents-write access must be available on the VM through `GH_TOKEN`,
`GITHUB_TOKEN`, or an existing `gh auth token` identity. Credentials stay out of
arguments and saved results. The default results branch is
`adamw-sgdm-tpu-results`, with a unique subtree for each campaign. A fresh write
and readback check runs before expensive work and on resume. A connector session
in another application does not supply credentials to the TPU host.

The normal campaign automatically performs the search, selection, full finalist,
optional bounded longer attempt, native WeightWatcher measurements, incremental
publication and final plots. [CAMPAIGN.md](CAMPAIGN.md) records its exact finite
budget and recovery policy. No candidate-by-candidate launch is required.

## Preview, status and resume

```bash
python3 baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py plan
python3 baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py status
python3 baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py resume
```

`plan` is a read-only preview and performs no install, download, publication or
training. `status` and `resume` resolve the saved latest campaign. Use the same
`--work-dir` override on all commands. `--campaign-id` selects an older campaign.
Resume preserves the original optimizer, LR/reference ratio, beta values, batch,
source hashes, seed and schedule horizon. Changed settings require a fresh campaign.

Completed runs are audited and reused during campaign recovery. A completed
campaign is a no-op. The controller verifies its own process identity before
signaling a recorded supervisor. SIGINT/SIGTERM request a safe pause; a hard VM
kill can recover only from the last committed checkpoint. Persistent publication
or diagnostic failure remains an explicit incomplete outcome with local evidence.

## Optional checkpoint backup

Add `--gcs-uri gs://YOUR_BUCKET/unique-campaign` to the initial start. Each run
receives a separate child prefix. The existing launcher checks write/read/delete
access before starting and verifies copied checkpoint/sidecar pairs. It does not
alter IAM or remove unrelated remote objects. Omit this option to retain
checkpoints on the persistent VM disk; GitHub still receives required metric data.

## CPU verification only

A synthetic smoke campaign runs two tiny candidates and one selected finalist:

```bash
python3 baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py start \
  --smoke --no-publish --min-free-gib 0 --work-dir /absolute/local/adamw-graft-smoke
```

`--no-publish` is permitted only for smoke. Omit it to verify live GitHub publication
using a separate smoke campaign ID. Synthetic runs cannot qualify for the official
loss target. The default CPU path is for verification; full FineWeb training uses TPU.

## Automatic readiness checks

Before production training, the preflight verifies the pinned model/data hashes,
package versions, all eight addressable TPU chips, persistent storage, attention
forward/backward, complete native WeightWatcher capability, and disposable full-model
updates with the actual graft optimizer. It rejects optimizer host scalar extraction
and continuing XLA compilation after warmup while learning rates/counters change.
CPU tests of this gate use simulated telemetry; actual TPU behavior remains a
future launch check. Runtime estimates require measured throughput on that hardware.

The individual-run `launch.py`, `preflight.py`, `train.py`, `metrics.py` and
`report.py` commands remain available for targeted investigation. Use their `--help`
inside the experiment's prepared environment. [PUBLISHING.md](PUBLISHING.md)
describes standalone archive recovery. The normal workflow uses the single entry
point above.
