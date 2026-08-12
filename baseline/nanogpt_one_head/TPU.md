# TPU/XLA execution and storage

The existing one-head nanoGPT package can run unchanged from the command line
on Apple MPS, CUDA, TPU/XLA, or CPU. `--device auto` selects the accelerator in
this order:

```text
TPU/XLA -> CUDA -> Apple MPS -> CPU
```

The implementation is intentionally **single-process**. On a multi-chip TPU
slice it uses one XLA device. This preserves the reference batch size, gradient
accumulation, optimizer-step count, and learning-rate schedule. A future
multi-device protocol must explicitly define gradient reduction, global batch
size, data sampling, checkpoint ownership, and WeightWatcher ownership; the
current runner refuses a multi-process XLA launch rather than silently changing
the experiment.

## Install on a TPU VM

Use a Python/PyTorch combination supported by the installed PyTorch/XLA release,
then install the TPU extra from this directory:

```bash
cd baseline/nanogpt_one_head
python -m pip install -e '.[tpu]'
```

PyTorch and PyTorch/XLA must have matching major/minor versions. The runner
checks this before training and reports a direct error if they do not match.
The reference protocol is float32; `XLA_USE_BF16` and `XLA_DOWNCAST_BF16` must
not be enabled.

## Persistent TPU storage

On macOS and ordinary CPU/CUDA machines the historical default remains:

```text
/tmp/rg-nanogpt-one-head
```

On a TPU VM, `--device auto` refuses to place a real run on the boot disk or
`/tmp` by default. Attach and mount durable storage first. The preferred layout
is a writable Hyperdisk mounted below `/mnt/disks`, for example:

```text
/mnt/disks/rg-data
```

With that mount present, no path environment variable is required. The runner
automatically resolves:

```text
/mnt/disks/rg-data/rg-nanogpt-one-head/data
/mnt/disks/rg-data/rg-nanogpt-one-head/results
/mnt/disks/rg-data/rg-nanogpt-one-head/plots
```

For a nonstandard mount path, set either:

```bash
export RG_TPU_PERSISTENT_ROOT=/your/mounted/volume
```

which appends `rg-nanogpt-one-head`, or set the exact experiment root:

```bash
export RG_NANOGPT_ONE_HEAD_TPU_ROOT=/your/mounted/volume/rg-nanogpt-one-head
```

The existing general overrides still take precedence:

```text
RG_NANOGPT_ONE_HEAD_ROOT
RG_NANOGPT_ONE_HEAD_DATA_ROOT
RG_NANOGPT_ONE_HEAD_RESULTS_ROOT
RG_NANOGPT_ONE_HEAD_PLOTS_ROOT
```

A general root override on TPU is checked against the mounted filesystems. Two
explicit per-purpose data/results paths are treated as an intentional advanced
override, which permits organization-specific bind mounts.

Only for a disposable smoke test, without any expectation that checkpoints or
data survive VM deletion, opt into ephemeral storage explicitly:

```bash
export RG_NANOGPT_ALLOW_EPHEMERAL_TPU_STORAGE=1
```

Do not set that variable for a long run.

## Inspect automatic detection

Run this before training:

```bash
rg-onehead-env --device auto
```

It prints JSON containing the selected accelerator, XLA runtime information,
and resolved data/results/plot roots. On a TPU VM, the output must show:

```text
accelerator: tpu
pjrt_device: TPU
xla_process_count: 1
```

and the roots must point to the attached durable volume.

## Smoke test

After the pinned corpus is available on the persistent volume, run the short
non-scientific integration test:

```bash
rg-onehead-train \
  --config configs/tpu_smoke.yaml \
  --optimizer muon \
  --device auto \
  --no-resume
```

The smoke test exercises model transfer, Muon, auxiliary AdamW, XLA step
boundaries, evaluation, CPU BLEU, CPU WeightWatcher, portable checkpoints, and
persistent path resolution. It is not an optimizer result and should never be
included in the scientific tables.

## Scientific runs

The same commands used on the Mac are used on TPU:

```bash
rg-onehead-train \
  --config configs/reference.yaml \
  --optimizer muon \
  --device auto
```

and for the corrected long-horizon ordinary-Muon run:

```bash
rg-onehead-train \
  --config configs/muon_10epochs.yaml \
  --optimizer muon \
  --device auto \
  --no-resume
```

The long-Muon schedule remains 488 warmup steps, cosine decay through step
9,766, and the LR floors through step 97,657. Device detection and storage
selection do not change the model, optimizer, batch size, seed, token budget,
checkpoint selection policy, or WeightWatcher definition.

## Accelerator-specific behavior

- Training, validation loss, and next-token accuracy run on the selected device.
- PyTorch/XLA receives an explicit graph-execution boundary after every
  optimizer step.
- Checkpoints always contain CPU tensors, so a run can be resumed or analyzed
  on a different accelerator.
- WeightWatcher always receives CPU copies of only the six hidden transformer
  matrices.
- Greedy BLEU is monitoring-only. On TPU it runs on a CPU snapshot to avoid
  compiling a different XLA graph for every generated sequence length.
- Randomized WeightWatcher measurements restore Python, NumPy, CPU Torch, and
  accelerator RNG state before training resumes.
