# Muon speedrun recipe for one eight-chip TPU

This ports the published **2024-11-10 UNetDoubleLr** recipe, which reached
**3.2753 validation NLL after 3,000 updates / 1,572,864,000 training tokens**.
The record reports 7.23 minutes of training on eight H100 GPUs. That timing
excludes validation and the first ten updates and is **not a TPU prediction**.
This is an established portable recipe, not the latest CUDA speed record and not
a claim of optimality on v5e. TPU convergence and performance require a live run.

## Start from Cloud Shell

Use a clean checkout of the published commit:

```bash
python3 baseline/gpt2_small/muon_speedrun/cloudshell.py start
```

The launcher uses the existing `ww-gpt2-validation-48h-20261004-s1337-node` in
`tpu-builders-504820/us-west4-a`. It creates no TPU, preserves the shared environment,
and refuses concurrent training. Repeating start while active prints status.
Every launch starts from the same seed and initialization, without resume.
To stop the current Muon speedrun and start fresh with tracking, use
`start --replace-current --optimizer muon --microbatch 64 --attention flash --hours 3`.
This stops only the service recorded in `MUON_SPEEDRUN_LATEST.json`; the TPU,
FineWeb cache, and prior files remain.

The worker stops at the first full validation with **NLL <= 3.28**, after the
published 3,000-update schedule, or before its deadline, whichever comes first.
It reports `schedule_complete_target_not_met` explicitly if all updates finish
above target. It never keeps training at a zero learning rate or declares
success merely because the process exited cleanly.

Default allocation budget: **three hours maximum**, including data preparation,
attention verification, compilation, training and final cloud backup. This is a
safety cap, not an ETA. It is also bounded by the existing TPU allocation expiry.
`--hours 1` requests a one-hour cap; a cap can truncate the schedule without
meeting the target. The TPU allocation itself remains active when the job ends.

```bash
python3 baseline/gpt2_small/muon_speedrun/cloudshell.py status
```

## The complete recipe

| Setting | Value |
|---|---|
| Architecture | 12 blocks, width 768, 6 heads of width 128, vocabulary 50,304 |
| Model changes | RoPE, RMSNorm/QK normalization, squared ReLU, zero output projections, value residuals, learned input/UNet skip weights, untied output head, logit soft cap 30 |
| Parameters | About 162M total; this is a modified transformer, not standard GPT-2 Small |
| Context / global batch | 1,024 tokens / 524,288 tokens per update |
| Default TPU microbatch | 64 sequences globally, eight accumulation passes; 8 sequences per chip |
| Schedule | 3,000 updates, zero warmup, constant LR through 2,100 then 900-update linear decay |
| Hidden matrices | Muon LR 0.04; five quintic Newton–Schulz iterations |
| Muon momentum | Linear ramp 0.85 to 0.95 over the first 500 updates; Nesterov |
| Embeddings / head / scalars | Adam LR 0.6 / 0.008 / 0.04; betas 0.9, 0.95; epsilon 1e-8 |
| Weight decay / global gradient clipping | None, matching the record |
| Precision | BF16 embedding/scalars; FP32 linear weights, BF16 linear compute; BF16 Newton–Schulz |
| Validation | Same pinned GPT-2-tokenized FineWeb file; first 10,485,760 tokens every 125 updates and at stop |
| Target | Full validation NLL <= 3.28, equivalent to perplexity <= exp(3.28) |

All model and optimizer settings come from the source record. The global batch
stays fixed when microbatch size changes. `--microbatch 32` uses sixteen accumulation
passes for additional memory headroom. The first 128-sequence/math-attention run
failed at update zero: 16.89 GiB required versus 15.75 GiB available per chip.
The default is now 64 and flash attention is required. No automatic
microbatch change or restart can silently alter a run. The selected microbatch
is a starting point, not the result of a TPU tuning sweep.

## TPU implementation

One XLA SPMD process partitions batches across eight chips. Global gradients are
replicated before optimizer work. Muon groups matrices by shape, pads groups to a
multiple of eight, and shards the matrix index during Newton–Schulz, then gathers
the resulting updates for replicated weights. It uses the exact record's
coefficients, Nesterov convention and rectangular scaling. It is **Muon, not
MuonClip**. Learning rate and momentum are device tensors to avoid compiling a
new graph solely because their Python values change.

The worker installs JAX and jaxlib **0.4.38**, the exact Pallas versions specified
by [PyTorch/XLA 2.6 setup.py](https://github.com/pytorch/xla/blob/v2.6.0/setup.py),
plus pinned ml-dtypes/opt-einsum into a run-local `pallas-deps` overlay. It does
not upgrade torch, torch_xla, libtpu, NumPy, SciPy or the shared venv. It verifies
the imports and records versions before starting the TPU check.

The worker tests PyTorch/XLA 2.6 TPU flash attention at the selected batch size,
sequence length 1,024 and head dimension 128. The check compares outputs and Q/K/V gradients
against mathematical attention with BF16 relative-L2 tolerance 0.03. It supplies
`sm_scale=1/sqrt(head_dim)` and the SPMD batch partition explicitly. The isolated
check is capped at five minutes. With default `--attention flash`, an unavailable
or failing kernel stops the job. The legacy `auto` option also requires a pass;
there is no implicit fallback to a memory-heavier attention implementation.
`--attention math` remains an explicit diagnostic option with microbatch <=64.

The 16 necessary training shards plus validation are SHA256-verified and prepared
before training, reusing `/mnt/disks/rg-data/benchmark-fineweb10B-889765ea`. This
avoids synchronous network downloads at training shard boundaries. Existing
FineWeb-Edu is a different corpus and is retained separately. All timings include
end-to-end training overhead; published GPU training-only timing is labelled.

WeightWatcher runs in a separate CPU process on immutable snapshots made from
the weights already transferred for checkpointing. There are no spectral fits,
gradient scans or additional per-matrix TPU reads in the training loop. Scalar
loss checks remain. An isolated flash check does not prove
complete TPU optimizer/model parity. Source CUDA compilation, random seed,
microbatch reduction order, rotary-buffer calculation and shard-boundary order
can differ; these are recorded rather than represented as exact reproduction.

## Paired WeightWatcher and token-error tracking

Tracking is enabled for new runs. The model, optimizers, hyperparameters, data,
seed, global batch, microbatch, flash attention, 3,000-update schedule, target,
evaluation/checkpoint cadence and three-hour cap remain unchanged. Additional
measurement work can add wall time within that cap.

Every existing validation point (125 updates and final stop) now counts top-1
prediction errors from the **same capped logits and the same benchmark tokens**
as validation NLL. `val_token_error = val_error_count / evaluation_tokens` is a
fraction, not a percentage. This is teacher-forced validation token error, not
free-generation accuracy or a separate test set. Partial evaluation is explicitly
flagged and cannot satisfy the target gate.

The same saved weights are queued for WeightWatcher 0.7.7 on CPU: Q, K, V, O,
MLP_IN and MLP_OUT in each of the twelve blocks (72 matrices). Embeddings,
output vocabulary head and scalar parameters are excluded. Fits use `ERG=True`,
`randomize=True`, `fix_fingers="clip_xmax"`, `max_fingers=10`, `min_evals=20`.
`raw_alpha` is recorded as `alpha_raw`; `alpha` as `alpha_clip_xmax`. Missing or
failed fits stay unavailable; clipped alpha is never substituted for raw alpha.
The package's other scalar outputs, including available trap/finger counts,
are retained. Randomization happens only in the separate CPU process.

- `tracking/layers.csv`: per-matrix raw/clipped alpha, fit status and paired
  validation loss, perplexity and token error, keyed by run and exact update.
- `tracking/summary.csv`: mean/minimum alpha, valid-fit counts, counts below two,
  and sample standard deviation **across matrices**, which is not a seed error bar.
- `tracking/measurements/<step>.json`: immutable results with snapshot SHA256,
  WeightWatcher version, diagnostic seed and fit options.
- `tracking/snapshots/<step>.pt`: immutable CPU transformer weights and paired
  validation metadata retained on the mounted disk for later analysis. These
  spectral snapshots are not full training-state checkpoints and are not uploaded
  by the final backup. Full latest/best/target checkpoints retain their existing
  cloud backup behavior.
- `TRACKING_STATUS.json`: completed, pending and failed measurements. A fit-process
  failure or deadline backlog is explicitly reported; snapshots remain recoverable.

Small tracking tables and JSON results are uploaded first during the existing
final cloud backup. The CPU worker drains its queue within the existing time cap;
there is no extension of the TPU allocation or automatic training restart.

## Checkpoints and results

Runs live at `/mnt/disks/rg-data/gpt2small/muon-speedrun-<optimizer>-<timestamp>`.
`MUON_SPEEDRUN_LATEST.json` identifies the latest run and service.

- `checkpoint_latest.pt`: atomic full-state save at initialization, updates 1 and 5, every 125
  updates, and normal stop. Includes model, optimizers, RNG, data cursor, recipe
  and next-step schedule. No automatic resume. TPU resume parity is not yet tested.
- `checkpoint_best.pt`: best fully evaluated checkpoint; a hard link protects it
  when the latest checkpoint is replaced, without duplicating local storage.
- `checkpoint_target.pt`: saved only after a qualifying full validation.
- `metrics.jsonl`, `latest_validation.json`, `manifest.json`, `status.json`:
  measurements, pinned provenance and explicit target outcome.
- Final cloud backup to the matching prefix under
  `gs://tpu-builders-504820-ww-continuous8/gpt2small/`, using object permissions
  with checksum verification. `CLOUD_BACKUP_VERIFIED.json` confirms completion.
  Backups are final, not continuous; intermediate checkpoints remain on the disk.

A stalled trainer with no recorded progress for 15 minutes is stopped. All
subprocesses and final backup share the hard allocation budget. Abrupt failure
preserves the previous atomic checkpoint; incomplete updates are not labelled
saved. Detailed failure and attention-check records remain with the run.

## Optional Adam comparison

`--optimizer adam` runs the same model, seed, token stream, batch and 3,000-update
schedule. Auxiliary Adam groups stay identical; hidden matrices use Adam with
LR 0.0006, betas 0.9/0.95 and zero weight decay. This is an **untuned control**,
not an optimized Adam speedrun or a promise of matching the target. It must run
sequentially on this TPU. Compare at equal tokens and end-to-end wall time; do
not attribute all differences from the earlier standard GPT-2 run to Muon.

## Pinned source and local verification

Source: [KellerJordan/modded-nanogpt record](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_1_short/2024-11-10_UNetDoubleLr/c87bb826-797b-4f37-98c7-d3a5dad2de74.txt).
Record author: Brendan Hogan Rappazzo. The log blob is
`4ef6d69736c2e49ccc1ce4ca98f35128ea7adb8d`.
`vendor/record_source.py` contains the original executable source extracted from
that reproducible log; `vendor/LICENSE` preserves the MIT license. Do not execute
the vendor file on TPU: its launcher targets CUDA/DDP.

`reference_val.json` preserves the 25 published validation observations. Tests
compare the port's forward/backward math to original source definitions, batched
Muon updates/state restoration to the original Newton–Schulz function, full
mixed-dtype optimizer learning on a tiny CPU model, full-validation target gates,
checkpoint retention and duplicate-launch prevention. These tests cannot certify
TPU performance or convergence before the live run.
