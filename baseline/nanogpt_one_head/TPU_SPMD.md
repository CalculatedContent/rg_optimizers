# One MuonClip run across four TPU chips

For extending a trained checkpoint, periodic test accuracy, bounded checkpoint
retention, and open-ended training, see [TPU_CONTINUATION.md](TPU_CONTINUATION.md).

This is single-host XLA SPMD data parallelism for the v5e-4, with one Python
process and one checkpoint/WeightWatcher owner. It is not a four-run sweep.
Multi-host TPU slices are rejected. The old single-chip configs remain opt-out.

## Provision from Cloud Shell

If you already submitted the request from chat, reuse its variables and skip
creation. Otherwise:

```bash
export RG_PROJECT=YOUR_PROJECT_ID
export RG_ZONE=us-west4-a
export RG_REQUEST="ww-long-$(date -u +%Y%m%d-%H%M%S)"
export RG_NODE="${RG_REQUEST}-node"
declare -p RG_PROJECT RG_ZONE RG_REQUEST RG_NODE > "$HOME/ww-long-session.env"
gcloud alpha compute tpus queued-resources create "$RG_REQUEST" \
  --project="$RG_PROJECT" --zone="$RG_ZONE" --node-id="$RG_NODE" \
  --accelerator-type=v5litepod-4 --runtime-version=v2-alpha-tpuv5-lite \
  --provisioning-model=flex-start --max-run-duration=72h \
  --valid-until-duration=1h --labels=purpose=muonclip-longrun

gcloud alpha compute tpus queued-resources describe "$RG_REQUEST" \
  --project="$RG_PROJECT" --zone="$RG_ZONE" --format='yaml(state)'
```

Wait for `ACTIVE`. The acquisition window is one hour; the run limit is 72
hours after provisioning. Flex-start supports up to seven days. To restore
variables in another Cloud Shell session, source `~/ww-long-session.env`.

## Durable storage

List existing disks first; an existing data disk can be reused only when it is
available for attachment. Do not detach a disk from an active experiment.

```bash
gcloud compute disks list --project="$RG_PROJECT" \
  --filter="zone:($RG_ZONE)" --format='table(name,sizeGb,type.basename(),users)'
```

For a NEW dedicated disk (skip creation if deliberately reusing an existing
one, and set `RG_DISK` to that disk's name):

```bash
export RG_DISK="${RG_REQUEST}-data"
gcloud compute disks create "$RG_DISK" --project="$RG_PROJECT" \
  --zone="$RG_ZONE" --size=100GB --type=pd-balanced

gcloud alpha compute tpus tpu-vm attach-disk "$RG_NODE" \
  --project="$RG_PROJECT" --zone="$RG_ZONE" --disk="$RG_DISK" --mode=read-write

declare -p RG_PROJECT RG_ZONE RG_REQUEST RG_NODE RG_DISK > "$HOME/ww-long-session.env"
gcloud compute tpus tpu-vm ssh "$RG_NODE" --project="$RG_PROJECT" --zone="$RG_ZONE"
```

Inside the VM, inspect `lsblk -f` and `ls -l /dev/disk/by-id/`. Identify the
attached data filesystem by its size and UUID. The guest device alias may be
`google-persistent-disk-1`; it need not match the Cloud disk resource name.
Mount the existing filesystem using the UUID shown by `lsblk -f`:

```bash
lsblk -f
ls -l /dev/disk/by-id/
export RG_DATA_UUID=YOUR_EXISTING_DATA_FILESYSTEM_UUID
sudo mkdir -p /mnt/disks/rg-data
sudo mount "UUID=$RG_DATA_UUID" /mnt/disks/rg-data
```

For a **new blank disk only**, set `RG_DEVICE` to its verified device path and
format it with `sudo mkfs.ext4 -m 0 "$RG_DEVICE"`, then read its UUID with
`lsblk -f` and mount as above. Never format a reused data disk. If it has a
partition, use the UUID of the filesystem-bearing partition. Then:

```bash
sudo chown "$(id -u):$(id -g)" /mnt/disks/rg-data
findmnt /mnt/disks/rg-data
```

The data disk survives TPU deletion and continues to incur storage charges.
The script below deliberately requires this mount for the long experiment.

## Install the branch inside the TPU VM

```bash
mountpoint -q /mnt/disks/rg-data
cd /mnt/disks/rg-data
git clone --branch codex/tpu-spmd-muonclip \
  https://github.com/CalculatedContent/rg_optimizers.git rg_optimizers_spmd
cd rg_optimizers_spmd/baseline/nanogpt_one_head
unset TPU_VISIBLE_CHIPS TPU_PROCESS_BOUNDS TPU_CHIPS_PER_PROCESS_BOUNDS
unset XLA_USE_SPMD XLA_USE_BF16 XLA_DOWNCAST_BF16
bash setup_tpu_v5e.sh --persistent-root /mnt/disks/rg-data
source "$HOME/.config/rg_optimizers/tpu_env.sh"
```

The config enables SPMD before XLA device creation. Do not use `torchrun`,
`xmp.spawn`, or one process per chip. Do not run the older per-chip sweep on
these same chips at the same time.

## Acceptance check and measured speed

This check downloads no data. It uses synthetic tokens only to test numerical
correctness, global QK clipping, evaluation and full-state checkpoint resume.
The optional throughput benchmark uses the real model shape and global batch.

```bash
python3 -m rg_nanogpt_one_head.tpu_spmd_check --backend tpu --chips 4 \
  --benchmark-config configs/muonclip_tpu_spmd_long.yaml --benchmark-steps 30 \
  --output /mnt/disks/rg-data/spmd-four-chip-check.json
```

Compare one chip using the SAME global batch and model shape, in a fresh process:

```bash
TPU_VISIBLE_CHIPS=0 TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
python3 -m rg_nanogpt_one_head.tpu_spmd_check --backend tpu --chips 1 \
  --benchmark-config configs/muonclip_tpu_spmd_long.yaml --benchmark-steps 30 \
  --output /mnt/disks/rg-data/spmd-one-chip-check.json
```

A correctness check must pass before continuing. Benchmark timing excludes the
first five warm-up updates and excludes WeightWatcher/checkpoint/evaluation I/O.
Do not interpret CPU-XLA throughput as TPU performance. Four chips need not be
faster for this small model.

## FineWeb smoke, then long run

Set `RG_DATA` to an existing compatible 80M/1M/1M-token cache or let the trainer
prepare it at a new path. The example reuses the existing one-head cache.
Keep smoke and long results separate.

```bash
export RG_DATA=/mnt/disks/rg-data/rg-nanogpt-one-head/data
python3 -m rg_nanogpt_one_head.muonclip --device tpu --optimizer muon_clip \
  --config configs/muonclip_tpu_spmd_smoke.yaml --seeds 1337 \
  --data-root "$RG_DATA" --results-root /mnt/disks/rg-data/muonclip-spmd-smoke
```

For the long run, use `tmux new -s muonclip-long` (install tmux if absent), then:

```bash
python3 -u -m rg_nanogpt_one_head.muonclip_resilient --device tpu \
  --config configs/muonclip_tpu_spmd_long.yaml --seed 1337 \
  --data-root "$RG_DATA" \
  --results-root /mnt/disks/rg-data/muonclip-spmd-long \
  --max-no-progress-failures 3
```

Detach with Ctrl-b then d. Reattach with `tmux attach -t muonclip-long`.
Run the same command to resume after replacing a VM and mounting the disk.
Keep the source revision, config, dependencies, and four-chip topology fixed;
resume checks reject a changed protocol. The supervisor handles worker process
failures, not provisioning/replacing an expired VM.

## Protocol and monitoring

- Global microbatch: 32 sequences, context 256, accumulation 1 = 8,192 tokens
  per update. On four chips each contributes eight sequences / 2,048 tokens.
- Replicated weights and gradients; global mean-loss gradients before clipping,
  momentum, and Muon Newton–Schulz. Never average independent Muon updates.
- QK maxima reduce over the global batch and every accumulation microbatch;
  replicated per-head maxima drive identical clipping.
- Explicit FP32 baseline; BF16 is not enabled or claimed validated.
- Both SPMD configs request `matmul_precision: highest`. The runtime sets XLA's
  separate native precision control as well as PyTorch's setting. PyTorch/XLA
  2.6 does not inherit this setting from `torch.set_float32_matmul_precision`.
  The check verifies the emitted HLO precision and a precision-sensitive matrix
  product before comparing gradients. Numerical tolerances are unchanged.
- 2,150,000 updates, peak LR 2e-4, 2,000 warm-up updates, full-horizon cosine to
  2e-5. This is a starting protocol, not an assertion of optimality.
- Full optimizer/model/RNG/sampler checkpoint every 500 updates; existing
  finite-state validation and atomic replacement preserve the last good state.
- Train/validation loss, perplexity and accuracy every 1,000 updates; held-out
  test report at completion. CPU WeightWatcher and permanent model snapshots
  every 10,000 updates plus endpoints. Raw and clipped alpha stay separate.
- WeightWatcher remains synchronous and runs once per snapshot. This commit
  provides multi-chip support; adaptive per-layer LR/backtracking is not enabled.
- The model remains one block/head, width 128, GPT-2 tokenizer, clean FineWeb-Edu.
  Synthetic acceptance tokens are not experimental memorization data.

Live outputs are under `muonclip-spmd-long/muon_clip/seed_1337/`: `metrics.csv`,
`muonclip_qk.csv`, spectral outputs and `checkpoint_latest.pt`.

## Stop the allocation early (Cloud Shell)

After training has stopped and its durable checkpoint is verified:

```bash
source "$HOME/ww-long-session.env"
gcloud alpha compute tpus queued-resources delete "$RG_REQUEST" \
  --project="$RG_PROJECT" --zone="$RG_ZONE" --force
```

## References

- https://docs.pytorch.org/xla/release/r2.6/perf/spmd_basic.html
- https://docs.pytorch.org/xla/master/tutorials/precision_tutorial.html
- https://docs.cloud.google.com/tpu/docs/request-using-flex-start
- https://docs.cloud.google.com/tpu/docs/attach-durable-block-storage
