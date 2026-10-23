# Four-chip hardware qualification and full-study runtime estimate

The full experiment is described in [TPU_FULL_MEMORIZATION.md](TPU_FULL_MEMORIZATION.md).
The code and CPU test suite alone do not prove that a particular TPU VM works.
Run the following qualification on the actual, otherwise idle four-chip v5e VM.
It does not provision a VM, stop other workloads, or change installed packages.

## What the qualification actually checks

- Four independent MuonClip processes concurrently, isolated to chip IDs 0,1,2,3.
- Every worker must report a single TPU device and world size one, through the
  same checks used by the production sweep.
- The existing full FineWeb 80M/1M/1M corpus is validated by identity and hashes.
  The one-head/128-width/256-context/50,257-vocabulary model, FP32, 8,192-token
  batch, full 39,063-step schedule, canary construction and injection remain intact.
- Four high-load (10%) tasks are run to step 500 and stopped only after the
  reference engine's finite atomic restart checkpoint has been saved.
- Four NEW processes load and validate those step-500 checkpoints, then continue
  to step 2500. The actual restored step is audited, not inferred from a filename.
- Complete 40-canary evaluations and all six raw/clipped WeightWatcher rows must
  exist at both initialization and the first trained spectral state (step 2441).
- All outputs, per-chip logs and pre-resume archives remain in a new uniquely
  named qualification directory, separate from the production study.

This is a **partial hardware/throughput acceptance test**, not a completed
memorization experiment. It must not be counted among the 25 completed runs.
There is deliberately no `run_complete.json` in a successful qualification prefix.
A successful prefix also cannot guarantee that no later numerical instability
will occur; production finite-state checks and bounded checkpoint retries remain.

## Run the check

Use a separate clean checkout containing PR #161 and the already activated,
matching PyTorch/PyTorch-XLA environment. Do not update an active checkout.
Set the following variables to existing, verified locations. The disk mount is
an example: simply creating that directory does not make it durable storage.

```bash
CODE="$(git rev-parse --show-toplevel)"
DATA="/mnt/disks/rg-data/fineweb-corpus"
PARENT="/mnt/disks/rg-data"
BLOCK="v5e4-fineweb-memorization-fp32"
LAUNCH="$CODE/baseline/nanogpt_one_head/tpu_full_memorization.sh"

# DATA must already contain verified full-size train.bin, val.bin, test.bin,
# and meta.json. The 'prepare' command can download/tokenize it once if absent.
bash "$LAUNCH" qualify --code "$CODE" --data-root "$DATA" \
  --output-parent "$PARENT" --hardware-block "$BLOCK"
```

Wait for `ALL FOUR CHIPS PASSED`. The program prints a unique output directory
and writes `qualification_report.json`, including each chip's timing and a
projection for the entire default 25-run MuonClip campaign. If any phase fails,
read the exact printed chip log; do not launch production or change the protocol
to hide the failure. `--allow-ephemeral` is only for an explicitly disposable
check; the full experiment should use durable storage.

## What the timing numbers mean

The production grid has 25 jobs assigned to four chips as **7/6/6/6**.
For roughly equal end-to-end replicate durations, wall time is about seven
replicate durations, not 25 and not exactly 25/4. The program uses each chip's
own concurrent second-phase timing rather than assuming perfect scaling:

```text
seconds_per_step[chip] = wall_seconds(step 500 -> 2500) / 2000
projected_queue_hours[chip] = jobs_on_chip * 39063 * seconds_per_step[chip] / 3600
projected_sweep_hours = max(projected_queue_hours)
```

The measured segments include actual training, train/validation probes, canary
evaluation, a trained WeightWatcher call, checkpoint writes, fresh-process
startup, restore and pre-resume archival. They do not include corpus preparation,
TPU queueing, future failures/retries, or all final test/report work. They are
prefix-based projections, not benchmarked full-campaign completion times.
Later constant-LR graphs may be faster than the changing-LR prefix. Load and seed
also affect runtime; the reported extra 25% is a planning allowance, not a
statistical confidence interval or guaranteed upper bound.

Conditional examples BEFORE any hardware timing:

| End-to-end duration of one full replicate while four are concurrent | 25-run sweep, before other overhead |
|---:|---:|
| 30 minutes | about 3.5 hours |
| 1 hour | about 7 hours |
| 2 hours | about 14 hours |
| 4.25 hours | about 30 hours |

These rows are scenarios, not a claim about achieved TPU throughput. The earlier
98-step TPU test includes a different workload/cold-start mix and must not be
used as a full-memorization speed guarantee. Likewise the 1,320-to-33 reduction
in canary forward-call count is not a 40x training speedup.

## Start the full experiment only after inspecting a passing report

```bash
ROOT="$PARENT/fineweb-memorization-tpu-v1"
bash "$LAUNCH" plan --code "$CODE" --root "$ROOT" --data-root "$DATA" \
  --hardware-block "$BLOCK"
# Continue only if the plan validates and the hardware check above passed.
bash "$LAUNCH" run --code "$CODE" --root "$ROOT" --data-root "$DATA" \
  --hardware-block "$BLOCK"
```

The production command retains the full five loads, five seeds, MuonClip, and
39,063 steps per run. Use the same root/options to resume. Mac results remain a
separate hardware block and are not resumed or pooled as TPU replicas.
