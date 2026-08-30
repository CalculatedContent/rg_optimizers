# Large MuonClip nanoGPT long run — 2026-08-30

This is a single-seed, MuonClip-only scaling run intended for Charles's
16-GiB Apple M2 Pro. It does **not** run AdamW or any other comparison arm.

## Scale point

| Quantity | Value |
|---|---:|
| Transformer blocks | 6 |
| Attention heads per block | 8 |
| Embedding width | 384 |
| Context length | 512 |
| Trainable parameters | 30,117,120 |
| Unique training tokens | 512,000,000 |
| Validation / test tokens | 4,000,000 / 4,000,000 |
| Processed training tokens | 512,000,000 |
| Tokens per parameter | 17.0 |
| Effective optimizer batch | 8,192 tokens |
| Optimizer updates | 62,500 |
| Warmup updates | 1,000 |
| Permanent WeightWatcher states | 26 |
| WeightWatcher matrices per state | 36 |

This is near a Chinchilla-style token budget in ratio, but it is not presented
as a scaling-law measurement: there is only one model size, one optimizer, and
one seed. The purpose is to obtain a serious long trajectory without repeatedly
cycling over the same small corpus.

The peak MuonClip learning rate remains the empirically exercised `2e-4`.
Warmup is lengthened to 1,000 updates and cosine decay spans the full fresh-data
pass, ending at `1e-5`. Weight decay is `0.1`, the RMS update scale is `0.2`,
gradient clipping is `1.0`, and QK-Clip retains the tested threshold of `100`.

Expected M2 Pro wall time is roughly **4–6 days**, based on the measured
four-head run and the increase in block count, width, and context. The first
few evaluation rows provide a machine-specific ETA; `status` reports it.

## Exact protocol

The frozen YAML is [`configs/muonclip_long_mps.yaml`](configs/muonclip_long_mps.yaml).
FineWeb-Edu is streamed from pinned revision
`593b3a867298afb8ce42625a270ef20ddcad28f9`. Train, validation, and test are
document-disjoint. The test split remains untouched until the final and
validation-selected checkpoint audit.

## Mac setup and preflight

Use the Conda Python that already passed the earlier campaign's dependency
check:

```bash
cd /tmp/rg_optimizers

git switch main
git pull --ff-only origin main

CONDA_PY="/Users/charleshmartin/opt/anaconda3/envs/ww_prod310/bin/python"

"$CONDA_PY" -m pip install -e baseline/nanogpt_one_head

cd baseline/experiments/nanogpt_muonclip_large_2026_08_30

export RG_NANOGPT_LARGE_EXPERIMENT_ROOT="/Users/charleshmartin/rg_runs/nanogpt_muonclip_large_20260830"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

"$CONDA_PY" scripts/run_experiment.py doctor --device auto --smoke-step
```

The smoke step instantiates the full 30.1M-parameter model on MPS and performs
one real MuonClip forward/backward/update before any 512M-token download.

## Start the detached long run

The following creates one detached `tmux` session. It prepares the larger
dataset and then starts or resumes the single MuonClip seed. No AdamW command is
present.

```bash
tmux new-session -d -s muonclip-large \
  "/usr/bin/caffeinate -dimsu '$CONDA_PY' scripts/run_experiment.py prepare && /usr/bin/caffeinate -dimsu '$CONDA_PY' scripts/run_experiment.py run --device auto --mps-retries 20"
```

Detach/closing the Cloud or Terminal window does not stop a process inside
`tmux`. To watch the live terminal and detach again, use:

```bash
tmux attach -t muonclip-large
```

Press `Control-b`, release both keys, then press `d`.

## Check progress at any time

Run these from the experiment directory with the same three exported variables
shown above:

```bash
"$CONDA_PY" scripts/run_experiment.py status
```

For the last terminal output:

```bash
tmux capture-pane -p -t muonclip-large -S -40
```

For the durable combined training log:

```bash
tail -n 40 "$RG_NANOGPT_LARGE_EXPERIMENT_ROOT/logs/train.log"
```

## Generate a live report without stopping training

```bash
"$CONDA_PY" scripts/run_experiment.py report --open
```

The report is regenerated atomically at:

```text
$RG_NANOGPT_LARGE_EXPERIMENT_ROOT/live_report/report.html
```

It includes train/validation loss, perplexity, optimizer diagnostics,
throughput, MPS memory, QK-Clip activity, and per-block curves for all six
matrix types for raw alpha, clip_xmax alpha, ERG gap, random distance, and trap
count. It uses only already-completed CSV rows and does not touch the model or
checkpoint.

## Resume after interruption or reboot

Re-export the variables, return to this directory, and run the same command:

```bash
/usr/bin/caffeinate -dimsu "$CONDA_PY" scripts/run_experiment.py run \
  --device auto \
  --mps-retries 20
```

The verified `checkpoint_latest.pt` includes the model, both optimizer states,
RNG state, training generator, and resume diagnostics. The launcher resumes by
default. Do not pass `--overwrite` unless intentionally discarding the run.
