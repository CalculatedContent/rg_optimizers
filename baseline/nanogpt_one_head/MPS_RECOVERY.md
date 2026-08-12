# MPS recovery and finite-checkpoint policy

## Failure mode

Apple's Metal backend can occasionally report a command-buffer recovery such as:

```text
Discarded (victim of GPU error/recovery)
kIOGPUCommandBufferCallbackErrorInnocentVictim
```

The message is emitted by the asynchronous MPS runtime. A Python training loop
may continue temporarily even though a queued GPU operation was discarded. If a
subsequent update consumes corrupted state, training loss and validation loss
can become `NaN`. WeightWatcher then fails later while attempting an SVD of a
non-finite matrix. The SVD exception is downstream; it is not the original
failure.

## Repository behavior

The ordinary `rg-onehead-train` and opt-in `rg-onehead-muonclip` launchers now
isolate every MPS optimizer/seed run in a fresh subprocess. This prevents a
sequence of long replicates from sharing one Metal command queue and one
long-lived MPS allocator state.

For a command such as:

```bash
rg-onehead-train \
  --config configs/reference.yaml \
  --optimizer muon \
  --seeds 1337,2027,4099 \
  --device auto
```

an Apple-Silicon machine runs three sequential worker processes. The scientific
protocol is unchanged: every worker receives the same config, seed, data root,
results root, batch size, evaluation probes, optimizer, and LR schedule.

If an isolated MPS worker exits nonzero, the supervisor waits briefly for Metal
to reset and makes one fresh-process resume attempt from
`checkpoint_latest.pt`. The checkpoint includes model state, optimizer state,
training-generator state, Python/NumPy/Torch RNG state, and MPS RNG state.

The default is:

```text
initial worker + one fresh-process resume attempt
```

Change it with:

```bash
--mps-retries 0   # no automatic restart
--mps-retries 2   # at most two restarts
```

For debugging only, the old same-process behavior can be requested with:

```bash
--no-mps-isolation
```

## Finite-state gate

Before replacing any training checkpoint, the code now:

1. synchronizes the accelerator;
2. copies the complete model and optimizer state to CPU;
3. verifies that every floating-point or complex tensor is finite;
4. writes to a temporary file;
5. atomically replaces the target checkpoint only after validation succeeds.

A contaminated update therefore cannot overwrite the last verified
`checkpoint_latest.pt`. Loading also applies the same finite-state validation,
so a legacy checkpoint containing `NaN` or `Inf` is rejected explicitly.

The training loop already checks finite train/validation metrics and model
parameters before WeightWatcher. The expected failure is now a direct
`FloatingPointError`, not the misleading downstream NumPy SVD error.

## What is not automatic

The supervisor does not silently switch from MPS to CPU or TPU. A device change
would alter numerical execution and should be an explicit experimental choice.
If the same step fails again after a fresh-process resume, stop the MPS run and
restart that seed from scratch on CPU or TPU, or investigate the local macOS and
PyTorch MPS versions.

## Existing runs

An already-running Python process is not changed by pulling this commit. To use
MPS worker isolation and the finite-checkpoint gate, stop the old launcher,
pull and reinstall the package, and start or resume with the normal command.
Do not resume from a checkpoint that the updated loader identifies as
contaminated.
