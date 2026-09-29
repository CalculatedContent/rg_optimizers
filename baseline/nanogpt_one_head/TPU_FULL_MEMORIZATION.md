# Full FineWeb memorization sweep on four TPU chips

This is an **additive full-study entry point**, not the four-device ordinary-Muon
quick test. No historical runner, model, optimizer, configuration, or result is
modified. The default campaign runs **MuonClip, five loads, five matched seeds**.

## Exactly which experiment runs

| Quantity | Full study |
|---|---|
| Corpus | FineWeb-Edu `sample-10BT`, revision `593b3a867298afb8ce42625a270ef20ddcad28f9` |
| Prepared splits | 80,000,000 train / 1,000,000 validation / 1,000,000 test tokens |
| Tokenizer | GPT-2 BPE; 50,257 tokens |
| Model | One block, one head, width 128, context 256; tied language head |
| Optimizer | Existing **MuonClip + auxiliary AdamW**, not ordinary Muon |
| Global batch for **each independent run** | 4 sequences x 8 accumulation x 256 tokens = 8,192 tokens/update |
| Horizon | 39,063 updates; 320,004,096 sampled token positions; approximately four corpus-equivalent epochs |
| Additional-bank loads | 0%, 0.1%, 0.5%, 2%, 10% |
| Matched seeds | 1337, 2027, 4099, 31415, 271828 |
| Precision and schedule | Existing FP32 protocol and one-epoch warmup/cosine, then LR floor; no hyperparameter retuning |
| Tracked canaries | Doses 0, 1, 4, 16, 64; eight sequences per dose; same contents/schedule within a matched seed |
| Acquisition | Injections during the first half of training; later training measures retention/forgetting |
| WeightWatcher | All six matrices at 17 permanent states, with raw and `clip_xmax` alpha, ERG and randomization retained |

**0% additional-bank load is not zero canary exposure.** Every run includes the
tracked exposed probes and the never-injected dose-0 probes. Additional-bank
load is the fraction of *acquisition-window sequence slots* replaced by the
extra bank, excluding the tracked-canary presentations. Consequently a 10%
setting occupies about 5% of full-horizon slots, plus tracked probes, because
acquisition ends halfway through training. It is not 10% of the stored corpus.

The base `train.bin`, `val.bin`, `test.bin`, and `meta.json` are identical across
loads. The existing `RandomCanaryExperiment.inject` changes selected training
batches at runtime; no alternative corpus is generated for each load.

## Parallelism: independent runs, not a changed global batch

A `v5litepod-4` is used as four independent single-chip workers. Up to four
seed/load jobs run concurrently. Each process receives its own
`TPU_VISIBLE_CHIPS` and single-process topology **before XLA imports**. A worker
refuses to train unless it sees exactly one XLA device and world size one.

This is intentionally separate from PR #160's data-parallel smoke runner:
there is no cross-rank gradient averaging, no need to invent distributed
QK-Clip semantics, and no multiplication of a run's global batch. Each task
uses the existing dedicated MuonClip extension and reference resume engine.

The full plan contains 25 jobs. Chip assignments are persisted and balanced
(7/6/6/6 jobs), with seed-major ordering so multiple loads become visible early.
Optional `--optimizers muon_clip,adamw` makes 50 jobs. An explicit longer
`--seeds` list registers a new plan; never change the seed list in an existing
root. The `--chips` option is for deliberately selected free chips, not for
joining an already-running distributed job.

## Faster canary evaluation, same measurements

The new worker opts into `canary_eval_batched.py`. The historical evaluator is
unchanged outside that worker. By default all 40 canaries are scored together.
Teacher-forced NLL uses only suffix logits. Greedy decoding uses a fixed padded
sequence shape, a tensor position index, and projects only the selected final
position into the vocabulary. It uses `torch.no_grad()` on XLA.

This reduces evaluator forward calls from 40 x (1 + 32) = 1,320 to **33** per
evaluation. It is a reduction in call count, **not a promised 40x end-to-end
speedup**. Evaluation still reports the same per-canary NLL, teacher accuracy,
free-running token accuracy, and exact match. Padding is strictly causal.
`canary_evaluation_timing.jsonl` records actual evaluation time.

The evaluator version and batch size are fingerprinted. CPU tests compare it
with the original scalar scorer, including a GPT-2-vocabulary parity check.
TPU throughput, numerical parity, and restart behavior still need a real-device
qualification; passing CPU tests is not a TPU performance benchmark.

## Before launching

Use a **separate clean checkout**, pinned to the commit containing these files.
Do not pull, switch branches, reinstall packages, or change precision in a
checkout/environment that is running another experiment. Keep the Mac campaign
untouched; TPU results form a new hardware block, not extra interchangeable Mac
seeds. Do not resume a Mac checkpoint as a TPU replicate.

Run on the TPU VM after installing a matching PyTorch/PyTorch-XLA pair using the
existing TPU setup instructions. This launcher does not provision or tear down
TPUs, kill unrelated jobs, or change installed packages. Confirm the chips are
free: a four-device smoke job and this four-chip campaign cannot own the same
chips concurrently. TPU runtime resource locking remains enabled.

Use a genuinely durable mounted volume below `/mnt/disks`, `/mnt/hyperdisk`, or
`/mnt/persistent`. The launcher rejects a normal boot-disk or `/tmp` root by
default. A directory created with `mkdir` is **not** a mounted durable volume.
`--allow-ephemeral` is only an explicit disposable-validation escape hatch.
A Flex-Start VM can expire; preserve checkpoints on the durable volume and keep
an independent backup. The code does not upload to Cloud Storage automatically.

Allow ample disk for 25 model/optimizer checkpoint sets and recovery archives;
50 GB free is a practical starting budget, not an enforced guarantee.

## Full-study commands

The example assumes an existing mounted volume at `/mnt/disks/rg-data`.
Set `CODE` to the **new, clean** checkout; do not use an actively running tree.
Use one stable, descriptive hardware-block label for the actual homogeneous TPU
slice. These variables are examples of locations, not a claim the disk exists.

```bash
CODE="$(git rev-parse --show-toplevel)"
ROOT="/mnt/disks/rg-data/fineweb-memorization-tpu-v1"
DATA="$ROOT/data"
BLOCK="v5e4-fineweb-memorization-fp32"
LAUNCH="$CODE/baseline/nanogpt_one_head/tpu_full_memorization.sh"

# Download/tokenize ONCE; or set DATA to an existing verified full corpus.
# A 4M-token smoke corpus is rejected, not silently substituted.
bash "$LAUNCH" prepare --code "$CODE" --data-root "$DATA"

# Validate source, full configs, corpus hashes, placement, and the 25-job plan.
# This command does not start training.
bash "$LAUNCH" plan --code "$CODE" --root "$ROOT" --data-root "$DATA" \
  --hardware-block "$BLOCK"

# Foreground run, or execute this command inside a tmux session.
# Repeating the SAME command resumes/reuses the SAME registered campaign.
bash "$LAUNCH" run --code "$CODE" --root "$ROOT" --data-root "$DATA" \
  --hardware-block "$BLOCK"
```

Only proceed from `prepare` to `plan` to `run` when the preceding command
succeeds. For detached execution after successful preparation/planning:

```bash
tmux new-session -d -s fineweb-memorization \
  "bash '$LAUNCH' run --code '$CODE' --root '$ROOT' --data-root '$DATA' --hardware-block '$BLOCK'"

bash "$LAUNCH" status --root "$ROOT"
tmux attach -t fineweb-memorization
```

`status` reports saved status and the last metric step; it is not a liveness
probe for a process after a VM disappears. Read the current attempt log printed
by the supervisor to inspect each task. Final completion requires the reference
engine's validated `run_complete.json`, not simply exit of a notebook.

## Resume and preservation contract

- There is **no `--overwrite` option**. A nonempty historical/Mac root is refused.
- Source commit, execution-code hashes, configs, chips, seeds, hardware block,
  shared corpus and evaluation batch size are frozen in `tpu_sweep_plan.json`.
- Existing completed tasks are passed back to the reference engine for identity
  and checkpoint validation before reuse. Completion is not inferred from a
  file name alone.
- Partial tasks resume through the existing finite atomic-checkpoint engine.
  Before that engine rolls diagnostic files back to the checkpoint, the worker
  copies the entire previous run into a new `recovery/task_NNN/<uuid>/` archive.
- Replayed canary rows are replaced at/after the resume step rather than appended
  as duplicate observations. The pre-resume archive preserves the old rows.
- Each retry has a new log. Retries are bounded (two by default). Exhaustion
  stops new jobs; other active jobs are allowed to finish.
- Locks protect the sweep, chip queues, and individual tasks. A duplicate
  invocation is refused, not forced. Interrupting the supervisor signals only
  its own children. Last finite checkpoints remain available.
- Rolling `checkpoint_latest.pt`, manifests and status files are naturally
  updated within a running experiment. "Preserve" does not mean disabling
  atomic checkpoint updates; it means never deleting/replacing an old run to
  start a different experiment.
- Resume on the original device/environment. Hardware or package changes can
  correctly fail native identity checks; never bypass those checks.

## Outputs and analysis

```text
ROOT/
  tpu_sweep_plan.json
  logs/task_NNN_<uuid>.log
  status/task_NNN.json
  receipts/task_NNN.json
  recovery/task_NNN/<uuid>/...
  load_0pct/results/muon_clip/seed_1337/
    manifest.json
    metrics.csv
    epoch_metrics.csv
    random_canary_manifest.json
    random_canary_metrics.csv
    canary_evaluation_timing.jsonl
    muonclip_qk.csv
    checkpoint_initial.pt
    checkpoint_latest.pt
    checkpoint_best.pt
    checkpoint_final.pt
    spectral/layers.csv
    test_results.json
    run_complete.json
  load_0.1pct/...
  load_0.5pct/...
  load_2pct/...
  load_10pct/...
```

Use within-run `NLL(dose 0) - NLL(dose d)` and recall differences as the
memorization measurements; 0% additional-bank load is only one experimental
condition. Compare loads at exact matched steps and paired seeds. Report
uncertainty across seeds, not layers or checkpoints. Do not mix partial latest
values with completed epoch-4 values. Keep raw and clipped alpha separate and
exclude failed-fit sentinels; an alpha threshold alone is not memorization.

The original frozen `baseline.yaml` report builder is not a load-sweep report.
These outputs retain the known CSV schemas for direct sweep analysis. The
additional random bank is not itself the tracked canary probe set, so probe
memorization does not directly measure all information retained from that bank.

## Validation

```bash
PYTHONPATH="$CODE/baseline/nanogpt_one_head/src" python -m pytest -q \
  "$CODE/baseline/nanogpt_one_head/tests/test_tpu_memorization_sweep.py"
```

Test coverage includes the complete task grid, chip-isolation environment,
rejection of smoke/altered configs, immutable plans, source/output separation,
locking, reference-scorer parity, fixed-shape padding, unchanged model/RNG and
canary inventories, and replay-row handling. Full real-TPU training and measured
speedup are not claimed by this documentation.
