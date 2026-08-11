# Long-horizon ordinary Muon

This protocol studies whether the six hidden-matrix WeightWatcher exponents stabilize only after substantially longer Muon training. It uses **ordinary Muon**, not HyperBall or any RG correction.

## Why a separate LR horizon is required

The one-epoch reference has 9,766 optimizer steps and a five-percent Muon warmup. Simply changing `training.target_epochs` from `1.0` to `10.0` also stretches that warmup from about 488 steps to about 4,883 steps. In the failed pilot, Muon's matrix LR was still rising near epoch 0.5 and the model became non-finite around step 4,250.

The repaired protocol separates the two horizons:

```text
training horizon:  97,657 steps (10 corpus-equivalent epochs)
LR horizon:         9,766 steps (the validated one-epoch schedule)
warmup:                488 steps
post-schedule:      hold the matrix and auxiliary AdamW LR floors
```

Thus the schedule is:

```text
steps 0--488:       linear warmup
steps 488--9766:    cosine decay
steps 9766--97657:  matrix LR 0.002; auxiliary LR 0.00003
```

## Run one seed from scratch

From `baseline/nanogpt_one_head`:

```bash
python -m pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1

RUNROOT=/tmp/rg-nanogpt-long-muon-v2
rm -rf "$RUNROOT"
mkdir -p "$RUNROOT/logs"

nohup caffeinate -i \
rg-onehead-train \
  --config configs/muon_10epochs.yaml \
  --optimizer muon \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root "$RUNROOT/results" \
  --device auto \
  --no-resume \
  > "$RUNROOT/logs/muon_seed_1337.log" 2>&1 &

echo $! | tee "$RUNROOT/muon_seed_1337.pid"
```

Watch the run:

```bash
tail -f /tmp/rg-nanogpt-long-muon-v2/logs/muon_seed_1337.log
```

At step zero, the next matrix LR should be approximately `4.10e-05`, not `4.10e-06`. The latter indicates that the ten-epoch warmup has been stretched incorrectly.

## Verify the schedule without training

```bash
python - <<'PY'
from rg_nanogpt_one_head.config import (
    load_config,
    lr_schedule_steps,
    max_steps,
    optimizer_profile,
    warmup_steps,
)

cfg = load_config('configs/muon_10epochs.yaml')
profile = optimizer_profile(cfg, 'muon')
training_steps = max_steps(cfg)
schedule_steps = lr_schedule_steps(cfg, profile)
warmup = warmup_steps(profile, schedule_steps)

print('training steps:', training_steps)
print('schedule steps:', schedule_steps)
print('warmup steps:', warmup)

assert training_steps == 97657
assert schedule_steps == 9766
assert warmup == 488
PY
```

The runtime aborts explicitly on non-finite metrics or parameters before a contaminated checkpoint or WeightWatcher SVD is written.
