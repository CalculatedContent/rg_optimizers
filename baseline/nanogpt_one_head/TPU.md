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

## TPU Builders v5e Flex-Start quick path

The following is a known-good path for a one-hour disposable v5e session. Run
the provisioning commands from Google Cloud Shell, not from inside a TPU VM.

### 1. Request the TPU

```bash
gcloud alpha compute tpus queued-resources create tpu-v5e-request \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-4 \
  --runtime-version=v2-alpha-tpuv5-lite \
  --node-id=tpu-v5e-node \
  --provisioning-model=flex-start \
  --max-run-duration=1h \
  --valid-until-duration=30m \
  --labels=purpose=flex-start
```

Check the request until its state is `ACTIVE`:

```bash
gcloud alpha compute tpus queued-resources describe tpu-v5e-request \
  --zone=us-west4-a \
  --format='value(state.state)'
```

`--valid-until-duration=30m` is the capacity-acquisition window.
`--max-run-duration=1h` starts after provisioning and automatically terminates
the TPU after at most one hour.

SSH into the active TPU VM:

```bash
gcloud compute tpus tpu-vm ssh tpu-v5e-node \
  --project=YOUR_PROJECT_ID \
  --zone=us-west4-a
```

To clean up early from Cloud Shell:

```bash
gcloud alpha compute tpus queued-resources delete tpu-v5e-request \
  --zone=us-west4-a \
  --force \
  --quiet
```

### 2. Clone and bootstrap the TPU VM

Inside the TPU VM:

```bash
cd /tmp
git clone https://github.com/CalculatedContent/rg_optimizers.git
cd rg_optimizers/baseline/nanogpt_one_head
bash setup_tpu_v5e.sh --ephemeral
```

The script resolves the current user's login directory dynamically and records
the required environment below that directory. To inspect the exact path:

```bash
USER_ROOT="$(getent passwd "$(id -u)" | cut -d: -f6)"
ENV_FILE="${USER_ROOT}/.config/rg_optimizers/tpu_env.sh"
printf '%s\n' "$ENV_FILE"
```

It also adds an idempotent source line to the current user's `.bashrc`, so
future SSH shells load the TPU environment automatically. For the shell that
launched the setup script, load it explicitly after the script returns:

```bash
USER_ROOT="$(getent passwd "$(id -u)" | cut -d: -f6)"
source "${USER_ROOT}/.config/rg_optimizers/tpu_env.sh"
```

For a disposable setup plus a small, non-scientific AdamW throughput test:

```bash
bash setup_tpu_v5e.sh --ephemeral --run-quick-smoke
```

That optional quick smoke creates a reduced 4M/100k/100k-token corpus, disables
WeightWatcher, and runs approximately 49 optimizer steps. It is only a TPU/XLA
compatibility and throughput check; it must not appear in scientific result
tables.

For a real run, mount durable storage and use:

```bash
bash setup_tpu_v5e.sh \
  --persistent-root /mnt/disks/rg-data
```

### What the setup script fixes and verifies

The stock TPU VM may contain an old packaging toolchain and an incompatible
PyTorch/PyTorch-XLA pair. The reusable script therefore:

1. upgrades user-level `pip`, `setuptools`, and `wheel` before installing this
   package, preventing the erroneous `UNKNOWN-0.0.0` build observed with the
   stock toolchain;
2. removes a stale `UNKNOWN` package if one exists;
3. installs matching `torch==2.6.0` and `torch_xla[tpu]==2.6.0` binaries from
   the TPU wheel indexes when the installed major/minor versions do not match;
4. installs this package without editable mode;
5. exports `PJRT_DEVICE=TPU` and the TPU provenance label
   `TPU_ACCELERATOR_TYPE=v5litepod-4`;
6. unsets `XLA_USE_BF16` and `XLA_DOWNCAST_BF16`, preserving the float32
   reference protocol;
7. configures either persistent or explicitly ephemeral storage; and
8. verifies both the raw XLA devices and `rg-onehead-env --device auto`.

The expected direct XLA check on a `v5litepod-4` is:

```text
torch: 2.6.0+cu124
torch_xla: 2.6.0
TPU devices: ['xla:0', 'xla:1', 'xla:2', 'xla:3']
```

The baseline still uses only `xla:0` in its current single-process protocol.

## Manual installation fallback

The setup script is preferred. The equivalent core installation sequence is:

```bash
cd baseline/nanogpt_one_head

python3 -m pip install --user --upgrade pip setuptools wheel
python3 -m pip uninstall -y torch torch_xla torchvision
python3 -m pip install --user \
  'torch==2.6.0' \
  'torch_xla[tpu]==2.6.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
python3 -m pip install --user .
```

PyTorch and PyTorch/XLA must have matching major/minor versions. A mismatch can
surface as an `_XLAC` import failure with an undefined PyTorch symbol. The
runner also checks the versions before training and reports a direct error.

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
and resolved data/results/plot roots. On a v5e TPU VM, the output should include:

```text
accelerator: tpu
device: xla:0
pjrt_device: TPU
tpu_accelerator_type: v5litepod-4
xla_addressable_device_count: 4
xla_process_count: 1
```

For a scientific run, the roots must point to the attached durable volume.

## Full integration smoke test

After the pinned corpus is available on the persistent volume, run the committed
integration test:

```bash
rg-onehead-train \
  --config configs/tpu_smoke.yaml \
  --optimizer muon \
  --device auto \
  --no-resume
```

This smoke test exercises model transfer, Muon, auxiliary AdamW, XLA step
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
- Greedy BLEU is a post-training secondary audit. On TPU it runs on a CPU
  snapshot to avoid compiling a different XLA graph for every generated
  sequence length.
- Randomized WeightWatcher measurements restore Python, NumPy, CPU Torch, and
  accelerator RNG state before training resumes.
