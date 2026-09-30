# High-dose canaries: raw-alpha / stronger-recall pilot

This is a NEW five-seed experiment, not a continuation of the 25-run additional-bank load sweep. The existing model, optimizers, training engine, corpus, and historical configurations are unchanged. The new entry point reuses the qualified four-independent-chip executor, including its exact run identity checks, atomic finite checkpoints, pre-resume archives, and bounded retries.

## Registered protocol

- Same pinned FineWeb-Edu corpus: 80M train, 1M validation, 1M test tokens; GPT-2 vocabulary 50,257.
- Same one-block, one-head, width-128 model; context 256; FP32; MuonClip plus auxiliary AdamW; unchanged learning rates, clipping, and weight decay.
- Five seeds: 1337, 2027, 4099, 31415, 271828. Four simultaneous independent single-chip runs, then the fifth on chip 0 (queue lengths 2/1/1/1).
- Every run contains doses **0, 64, 128, 256, 512, 1024**, with eight independent random sequences per dose: **48 tracked canaries**.
- A dose is the exact number of presentations of each canary over the acquisition window, NOT percent load and NOT presentations per epoch. All 48 sequences are scored in one fixed-shape batch.
- 15,872 total tracked presentations; 625,024 acquisition sequence slots; about 2.54% replacement during acquisition. No extra untracked random bank. Dose-0 sequences are never injected.
- Injections use the existing scheduler for the first 19,532 of 39,063 updates (approximately epochs 0-2); epochs 2-4 measure retention. Every seed still sees 8,192 token positions per update.
- Spectral states every 0.125 nominal epoch: 33 states including initialization. All six hidden matrices retain **alpha_raw**, alpha_clip_xmax, fit diagnostics, ERG and randomized-null diagnostics. **alpha_raw is the primary variable; clipped alpha is secondary only.** Existing console output prints both.

The target is to see whether increasing exact exposure produces substantial teacher-forced and free-running recall and whether that co-occurs with changes in raw alpha. Literal recall and a raw-alpha crossing are hypotheses, not guaranteed outcomes.

## Interpretation and analysis

Primary behavioral measures are mean suffix NLL, teacher-forced token accuracy, free-running token accuracy and exact 32-token continuation recall, relative to never-injected dose-0 canaries in the SAME model. Preserve absolute NLL/recall as well as contrasts. Primary spectral summaries are each matrix's raw alpha and the minimum VALID raw alpha across the six matrices, retaining the identity of the minimizing matrix. Do not treat failed-fit sentinels as small alpha. Exclude initialization from association plots but preserve it for baseline changes.

Match spectral and behavioral measurements by exact optimizer step. Separate acquisition from retention, report seed-level effects, and control for common training-time trends. Checkpoint and layer rows are not independent replications.

All dose groups coexist in each model: there is one spectrum per matrix/checkpoint, NOT a separate spectrum for each dose. This design tests within-model dose response and whether strong recall can occur without raw alpha below 2. It does not by itself identify a causal between-model dose -> spectrum effect. A separate randomized between-run dose intervention would be needed for that claim. Do not pool these results with the prior load sweep as the same protocol; the changed dose inventory also changes which random sequences are assigned to the dose labels.

## Before starting on the existing TPU

Use the existing installed TPU environment; do not reinstall PyTorch or update the actively running checkout. Clone this branch into a SEPARATE directory on the persistent disk. The default launcher requires all 25 old jobs to have matching saved completion receipts. If it says 24/25, wait and rerun start later; it does not stop that last old job.

The original corpus is reused read-only from:

`/mnt/disks/rg-data/rg-nanogpt-one-head/data`

The old run root is read only for the completion gate:

`/mnt/disks/rg-data/fineweb-memorization-tpu-v1`

New results and logs:

```text
/mnt/disks/rg-data/fineweb-highdose-rawalpha-v1/
  tpu_sweep_plan.json
  logs/task_*.log
  receipts/task_*.json
  status/task_*.json
  load_highdose/results/muon_clip/seed_*/
    metrics.csv
    random_canary_manifest.json
    random_canary_metrics.csv
    spectral/layers.csv
    checkpoint_initial.pt
    checkpoint_latest.pt
    checkpoint_best.pt
    checkpoint_final.pt
    run_complete.json
  highdose_exit.json
/mnt/disks/rg-data/fineweb-highdose-rawalpha-v1.control/
  plan-*.log
  run-*.log
  latest_log
```

## Commands inside the TPU VM (not Cloud Shell)

Use a new, clean checkout pinned to the tested high-dose commit; the following path is separate from rg_optimizers_full:

```bash
CODE=/mnt/disks/rg-data/rg_optimizers_highdose
# After cloning/checking out the exact high-dose commit:
bash "$CODE/baseline/nanogpt_one_head/tpu_high_dose.sh" start
```

`start` validates the old completion receipts, current source, exact protocol, verified corpus and persistent placement. It saves the verbose plan to disk, prints a short summary, and launches detached tmux session **ww_highdose** with persistent output logging from the first process message. It does not create a nested interactive tmux client. A session with the same name is preserved, never killed. Worker/seed progress is printed every 60 seconds.

```bash
bash "$CODE/baseline/nanogpt_one_head/tpu_high_dose.sh" status
bash "$CODE/baseline/nanogpt_one_head/tpu_high_dose.sh" tail
```

After confirming four workers are advancing, logging out of SSH is safe. Ctrl-C when following `tail` stops only that log viewer; do not send Ctrl-C to the training pane. To inspect the pane, `tmux attach -d -t ww_highdose`; detach with Ctrl-b then d.

A failed/interrupted high-dose run is resumed by the same start command and the same frozen source/options after its prior session exits. The inherited executor checks identity and archives partial state before rollback. There is no overwrite option. Changing machines or Python/package versions can correctly fail identity validation; do not bypass it.

Defaults may be overridden BEFORE first launch using RG_HIGH_DOSE_ROOT, RG_HIGH_DOSE_DATA, RG_HIGH_DOSE_PREVIOUS, and RG_HIGH_DOSE_BLOCK. The wrapper expects the persistent volume mounted at /mnt/disks/rg-data. Do not change options in an already registered root.

## Timing and lifetime

At previous measured throughput, two seed-runtime waves suggest roughly 2.5-3 hours; allow about 3-4 hours initially for the doubled spectral cadence and 48-canary evaluations. This is a planning estimate, not a measured high-dose benchmark.

The code DOES NOT extend, restart, create, delete or shut down the TPU. Its pre-existing Flex-Start termination time still applies; verify enough time remains before launching. tmux survives an SSH disconnect, not VM termination. Persistent data survive independently of the TPU node as long as the data disk is retained.

## Validation

The high-dose test suite covers configuration invariants, five-seed/four-chip task construction, all five exact exposure schedules, no injected dose-0 controls, read-only old-completion checks, inherited worker dispatch, and 48-canary batched/scalar metric parity on a small CPU model. The shared executor has already been exercised on the user's four-chip TPU by the prior experiment. The NEW high-dose scientific runs have not been executed by these CPU tests.

```bash
PYTHONPATH=baseline/nanogpt_one_head/src python3 -m pytest -q baseline/nanogpt_one_head/tests/test_tpu_high_dose_sweep.py
```
