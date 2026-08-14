# MuonClip microbatch spectral walk

This opt-in diagnostic captures the initial model, every accumulation
microbatch, and every resulting MuonClip optimizer update for the first ten
effective batches.

```text
config:   configs/muonclip_microbatch10.yaml
trainer:  rg-onehead-muonclip-walk
movie:    rg-onehead-muonclip-movie
seed:     2027
```

## Exact checkpoint count

The committed run uses:

```text
10 effective optimizer batches
8 microbatches per effective batch
1 initialization checkpoint
```

Therefore it writes:

```text
1 + (10 * 8) + 10 = 91 WeightWatcher checkpoints
```

It also writes `step_traces/step_0000000.pt`, an explicit initialization
trace, so step zero is visible in both the WeightWatcher checkpoint inventory
and the trace inventory.

The code has a hard upper bound of 500 WeightWatcher checkpoints. The full
configuration is rejected before training if its requested count exceeds the
configured or hard cap.

## Semantics

```text
ww_step_0000000.pt            initialized weights before any batch
ww_microbatch_0000001.pt      after backward for microbatch 1, before update
...
ww_microbatch_0000008.pt      after backward for microbatch 8, before update
ww_step_0000001.pt            after MuonClip optimizer update 1
```

Gradient accumulation does not change the weights between microbatches. Thus:

```text
weight ESD:       stationary within one accumulation group; moves at ww_step_N
accumulated-gradient ESD: can move after every ww_microbatch_N checkpoint
```

Each microbatch checkpoint contains both sources and can be loaded with:

```python
load_weightwatcher_checkpoint(path, source="weights")
load_weightwatcher_checkpoint(path, source="accumulated_gradients")
```

Microbatch capture is controlled by:

```yaml
walk_capture_microbatches: true
```

Set it to `false` for a long-horizon run. Post-update checkpoints and the
explicit `ww_step_0000000.pt` initialization remain available.

## Run the ten-effective-batch experiment

Do not run it concurrently with another MPS training process.

```bash
cd /private/tmp/rg_optimizers
git switch main
git pull --ff-only

cd baseline/nanogpt_one_head
python -m pip install -e .

export PYTORCH_ENABLE_MPS_FALLBACK=1
export RG_MUONCLIP_WALK_ROOT=/tmp/rg-nanogpt-muonclip-walk

RUNROOT="/tmp/rg-nanogpt-muonclip-microbatch10-seed2027-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUNROOT/logs"

nohup caffeinate -i \
  rg-onehead-muonclip-walk \
  --config configs/muonclip_microbatch10.yaml \
  --optimizer muon_clip \
  --seeds 2027 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root "$RUNROOT/results" \
  --device auto \
  --no-resume \
  > "$RUNROOT/logs/muonclip_microbatch10_seed2027.log" 2>&1 &

echo $! | tee "$RUNROOT/muonclip_microbatch10_seed2027.pid"
```

Resolve the exact capture directory:

```bash
RUN_DIR="$RUNROOT/results/muon_clip/seed_2027"

while [ ! -f "$RUN_DIR/muonclip_walk_location.json" ]; do
  sleep 1
done

WALK_DIR="$(
python - "$RUN_DIR/muonclip_walk_location.json" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads(Path(sys.argv[1]).read_text())["capture_dir"])
PY
)"
export WALK_DIR

echo "$WALK_DIR"
```

Verify initialization and all 91 matrix checkpoints:

```bash
ls -l "$WALK_DIR/weightwatcher_checkpoints/ww_step_0000000.pt"

find "$WALK_DIR/weightwatcher_checkpoints" \
  -name 'ww_*.pt' | wc -l
```

## Make the smooth weight-ESD movie

The movie tool analyzes one matrix at a time. Its WeightWatcher call is:

```python
savedir = str(native_dir)
watcher.analyze(
    plot=True,
    savefig=savedir,
    min_evals=20,
    randomize=False,
    ERG=False,
)
```

For the query matrix in transformer block 0:

```bash
rg-onehead-muonclip-movie \
  --walk-dir "$WALK_DIR" \
  --matrix L00_W_Q \
  --cadence microbatch \
  --source weights \
  --first-effective-batch 1 \
  --last-effective-batch 10 \
  --max-checkpoints 500 \
  --fps 30 \
  --frames-per-transition 8
```

This timeline includes initialization, all 80 microbatch states, and all 10
post-update states. The smooth in-between frames are visual interpolation; the
91 integer snapshots are the actual saved matrices.

## Make the accumulated-gradient ESD movie

This is the quantity that actually evolves after every backward microbatch:

```bash
rg-onehead-muonclip-movie \
  --walk-dir "$WALK_DIR" \
  --matrix L00_W_Q \
  --cadence microbatch \
  --source accumulated_gradients \
  --first-effective-batch 1 \
  --last-effective-batch 10 \
  --max-checkpoints 500 \
  --fps 30 \
  --frames-per-transition 8
```

The outputs are written under:

```text
$WALK_DIR/diagnostics/esd_movie_<matrix>_<source>_<cadence>/
```
