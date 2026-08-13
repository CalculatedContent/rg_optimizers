# Fifty-epoch ordinary Muon versus MuonClip + RMS

This matched pair is designed to answer one specific question:

> Does RMS-matched MuonClip produce more stable layerwise spectral exponents
> than ordinary Muon during very long one-head nanoGPT training?

The two committed configs are:

```text
configs/muon_50epochs.yaml
configs/muonclip_50epochs.yaml
```

They use the same:

```text
FineWeb-Edu revision
80M / 1M / 1M train-validation-test token splits
one block, one head, width 128, context 256
seed 1337
batch size 4 and gradient accumulation 8
50 corpus-equivalent epochs
quarter-epoch WeightWatcher measurements
fixed train/validation/test/BLEU probes
ERG=True, randomize=True, strict=True
```

Each run therefore has:

```text
training steps:       488,282
LR-schedule steps:      9,766
spectral checkpoints:     201  (epoch 0 through 50 every 0.25 epoch)
```

The optimizer-specific schedules are intentionally different because MuonClip
uses Moonshot-style RMS-matched update scaling:

| Optimizer | Peak LR | Floor LR | Warmup | Schedule horizon | Long tail |
|---|---:|---:|---:|---:|---:|
| ordinary Muon matrices | 0.02 | 0.002 | 488 steps | 1 epoch | 0.002 |
| ordinary Muon auxiliary AdamW | 3e-4 | 3e-5 | 488 steps | 1 epoch | 3e-5 |
| MuonClip + RMS matrices | 2e-4 | 2e-5 | 500 steps | 1 epoch | 2e-5 |
| MuonClip auxiliary AdamW | 2e-4 | 2e-5 | 500 steps | 1 epoch | 2e-5 |

MuonClip additionally keeps:

```text
RMS update coefficient: 0.20 * sqrt(max(n, m))
weight decay:            0.10
QK-Clip threshold:       100
Q/K balance:             0.50 / 0.50
momentum:                0.95
Nesterov:                false
```

## Run MuonClip + RMS

Do not run ordinary Muon and MuonClip concurrently on the same Apple GPU. The
MPS launcher isolates a run in a fresh worker process and permits one resume
attempt from the last finite atomic checkpoint after a transient Metal failure.

```bash
cd baseline/nanogpt_one_head
python -m pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1

RUNROOT=/tmp/rg-nanogpt-long-muonclip-50ep
rm -rf "$RUNROOT"
mkdir -p "$RUNROOT/logs"

nohup caffeinate -i \
rg-onehead-muonclip \
  --config configs/muonclip_50epochs.yaml \
  --optimizer muon_clip \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root "$RUNROOT/results" \
  --device auto \
  --no-resume \
  > "$RUNROOT/logs/muonclip_seed_1337.log" 2>&1 &

echo $! | tee "$RUNROOT/muonclip_seed_1337.pid"
```

Watch training:

```bash
tail -f /tmp/rg-nanogpt-long-muonclip-50ep/logs/muonclip_seed_1337.log
```

Watch alpha, fit D, randomized distance, ERG gap, and traps:

```bash
rg-onehead-monitor \
  --results-root /tmp/rg-nanogpt-long-muonclip-50ep/results \
  --optimizer muon_clip \
  --seed 1337
```

Inspect QK-Clip activation:

```bash
column -s, -t \
  < /tmp/rg-nanogpt-long-muonclip-50ep/results/muon_clip/seed_1337/muonclip_qk.csv \
  | tail -20
```

## Matched ordinary-Muon control

The committed control command is:

```bash
RUNROOT=/tmp/rg-nanogpt-long-muon-50ep
rm -rf "$RUNROOT"
mkdir -p "$RUNROOT/logs"

nohup caffeinate -i \
rg-onehead-train \
  --config configs/muon_50epochs.yaml \
  --optimizer muon \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root "$RUNROOT/results" \
  --device auto \
  --no-resume \
  > "$RUNROOT/logs/muon_seed_1337.log" 2>&1 &
```

An already-running ordinary-Muon 50-epoch run does not need to be restarted
merely for analysis, provided it used the same architecture, data, seed,
quarter-epoch WeightWatcher cadence, and one-epoch LR horizon.

## Compare layerwise alpha at matched epochs

This command joins the two `spectral/layers.csv` files on exact quarter epoch
and matrix name. It prints the latest common epoch and recent median
trajectories.

```bash
MUON=/tmp/rg-nanogpt-long-muon-50ep/results/muon/seed_1337
CLIP=/tmp/rg-nanogpt-long-muonclip-50ep/results/muon_clip/seed_1337

python - "$MUON" "$CLIP" <<'PY'
from pathlib import Path
import sys
import pandas as pd

muon_path = Path(sys.argv[1]) / "spectral" / "layers.csv"
clip_path = Path(sys.argv[2]) / "spectral" / "layers.csv"


def load(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("step", "epoch", "alpha", "D", "rand_distance", "ERG_gap"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["epoch", "matrix_name"])
    frame["epoch_key"] = frame["epoch"].round(6)
    keep = [
        "epoch_key", "matrix_name", "alpha", "D", "rand_distance", "ERG_gap"
    ]
    frame = frame[keep].drop_duplicates(
        ["epoch_key", "matrix_name"], keep="last"
    )
    return frame.rename(
        columns={
            "alpha": f"alpha_{label}",
            "D": f"D_{label}",
            "rand_distance": f"rand_distance_{label}",
            "ERG_gap": f"ERG_gap_{label}",
        }
    )


muon = load(muon_path, "muon")
clip = load(clip_path, "clip")
joined = muon.merge(clip, on=["epoch_key", "matrix_name"], how="inner")
if joined.empty:
    raise RuntimeError("The two runs have no common completed spectral epoch")

joined["delta_alpha_clip_minus_muon"] = (
    joined["alpha_clip"] - joined["alpha_muon"]
)
joined["abs_ERG_gap_muon"] = joined["ERG_gap_muon"].abs()
joined["abs_ERG_gap_clip"] = joined["ERG_gap_clip"].abs()

latest_epoch = joined["epoch_key"].max()
latest = joined[joined["epoch_key"] == latest_epoch].sort_values("matrix_name")

print(f"LATEST COMMON EPOCH: {latest_epoch:.2f}\n")
print(
    latest[
        [
            "matrix_name",
            "alpha_muon",
            "alpha_clip",
            "delta_alpha_clip_minus_muon",
            "D_muon",
            "D_clip",
            "rand_distance_muon",
            "rand_distance_clip",
            "abs_ERG_gap_muon",
            "abs_ERG_gap_clip",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
)

recent = (
    joined.groupby("epoch_key", as_index=False)
    .agg(
        alpha_muon_median=("alpha_muon", "median"),
        alpha_clip_median=("alpha_clip", "median"),
        D_muon_median=("D_muon", "median"),
        D_clip_median=("D_clip", "median"),
        rand_muon_median=("rand_distance_muon", "median"),
        rand_clip_median=("rand_distance_clip", "median"),
        abs_erg_muon_median=("abs_ERG_gap_muon", "median"),
        abs_erg_clip_median=("abs_ERG_gap_clip", "median"),
    )
    .sort_values("epoch_key")
    .tail(12)
)

print("\nRECENT MATCHED CHECKPOINT MEDIANS")
print(recent.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
PY
```

The primary comparison is not merely whose final mean alpha is lower. Examine:

```text
per-layer alpha trajectories
checkpoint-to-checkpoint alpha variation
fit D at the same checkpoints
rand_distance
absolute ERG gap
train/validation collapse frequency
```

MuonClip is spectrally more stable only if the individual layer trajectories
settle while the fits remain credible and the training dynamics avoid the
catastrophic excursions seen in the ordinary-Muon long tail.
