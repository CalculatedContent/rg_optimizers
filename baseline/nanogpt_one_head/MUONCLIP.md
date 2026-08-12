# MuonClip nanoGPT baseline

This baseline adds Moonshot AI's **MuonClip** optimizer to the existing
one-block, one-head FineWeb-Edu experiment without changing the historical
`rg-onehead-train` command or the committed SGD, AdamW, and ordinary-Muon
protocols.

MuonClip combines:

1. Muon momentum followed by Newton--Schulz orthogonalization;
2. decoupled weight decay;
3. RMS-matched matrix updates;
4. per-head QK-Clip.

For a hidden matrix `W` with shape `n x m`, this implementation applies

```text
M_t = momentum * M_(t-1) + G_t
O_t = NewtonSchulz(M_t) * 0.2 * sqrt(max(n, m))
W_t = W_(t-1) - lr * (O_t + weight_decay * W_(t-1))
```

For each attention head, the maximum finite causal pre-softmax logit is
measured over every gradient-accumulation micro-batch. With threshold `tau`,

```text
gamma_h = min(1, tau / max_logit_h)
```

and regular multi-head attention uses the balanced scaling

```text
W_Q[h] *= gamma_h ** 0.5
W_K[h] *= gamma_h ** 0.5
```

The committed reference value is `tau = 100`, matching the Kimi K2 report.
The compact nanoGPT baseline has only one head, so it is possible that QK-Clip
never activates. That is a valid experimental result and is recorded explicitly
rather than inferred from the loss curve.

## Why a dedicated launcher?

MuonClip is opt-in. The historical package and result tables retain exactly the
three original optimizer arms. The dedicated launcher installs the MuonClip
extension in the current process and then delegates to the same data,
checkpoint, evaluation, WeightWatcher, MPS/CUDA/TPU, and long-horizon code.

```bash
rg-onehead-muonclip --help
```

## One-epoch reference run

```bash
cd baseline/nanogpt_one_head
python -m pip install -e .

rg-onehead-muonclip \
  --config configs/muonclip_reference.yaml \
  --optimizer muon_clip \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-muonclip/results \
  --device auto \
  --no-resume
```

The reference profile uses:

```text
base LR:                2e-4
minimum LR:             2e-5
warmup:                 500 of 9,766 steps
weight decay:           0.1
momentum:               0.95
Newton--Schulz steps:   5
RMS scale:              0.2 * sqrt(max(n, m))
QK-Clip threshold:      100
Q/K balance:            0.5 / 0.5
auxiliary optimizer:    AdamW with the same base LR and weight decay
```

The Kimi K2 production recipe used a much larger model, a 15.5-trillion-token
WSD schedule, and distributed bfloat16 training. This compact baseline preserves
the repository's matched warmup-plus-cosine protocol so optimizer comparisons
remain interpretable; it is not presented as a literal reproduction of K2
pretraining.

## Ten-epoch spectral-relaxation run

```bash
rg-onehead-muonclip \
  --config configs/muonclip_10epochs.yaml \
  --optimizer muon_clip \
  --seeds 1337 \
  --data-root /tmp/rg-nanogpt-one-head/data \
  --results-root /tmp/rg-nanogpt-muonclip-long/results \
  --device auto \
  --no-resume
```

This follows the same long-horizon convention as ordinary Muon:

```text
training steps:     97,657
LR schedule steps:   9,766
warmup steps:          500
post-epoch-1 LR:      2e-5
```

It is a long-time spectral-relaxation experiment, not the K2 WSD schedule.

## Diagnostics

The usual files remain unchanged:

```text
metrics.csv
spectral/layers.csv
spectral/summary.csv
checkpoint_latest.pt
checkpoint_best.pt
checkpoint_final.pt
```

MuonClip additionally writes:

```text
muonclip_qk.csv
```

at the configured diagnostic interval. Its columns are:

```text
step
threshold
steps_in_interval
head_observations
active_heads
active_fraction
mean_max_logit
max_logit
mean_gamma
min_gamma
```

This file is monitoring-only. It does not select checkpoints or tune the
threshold. The existing WeightWatcher monitor can be used for alpha, fit `D`,
`rand_distance`, ERG gap, and traps:

```bash
rg-onehead-monitor \
  --results-root /tmp/rg-nanogpt-muonclip/results \
  --optimizer muon_clip \
  --seed 1337
```

## TPU

The same launcher uses the automatic accelerator and persistent-storage logic:

```bash
rg-onehead-muonclip \
  --config configs/muonclip_reference.yaml \
  --optimizer muon_clip \
  --device auto \
  --no-resume
```

Run the existing TPU smoke test first. The MuonClip-specific path should then be
qualified with a short one-epoch seed before starting the ten-epoch run.
