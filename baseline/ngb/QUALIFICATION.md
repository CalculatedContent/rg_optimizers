# NGB v4 qualification plan

The committed v4 optimizer profiles are conservative center candidates derived
from the v3 three-seed trajectories. They are not declared globally optimal.
Protected test metrics and WeightWatcher diagnostics must not select a profile.

## Stage 1 — instability screen

Use seeds `1337,2027`, which exposed the largest v3 adaptive-optimizer
bifurcation. Run temporary 0.5-epoch configurations in separate result roots.

Candidate neighborhoods:

| Optimizer | Screen |
|---|---|
| SGD + Nesterov | peak LR `{0.03, 0.05}`; floor `1%` of peak; decay `0.01` |
| AdamW | peak LR `{2e-4, 3e-4}`; decay `{0.05, 0.10}`; floor `1e-5` |
| Muon matrices | peak LR `{0.005, 0.01}`; decay `{0.01, 0.02}`; floor `{1e-4, 2e-4}` |
| Muon auxiliary AdamW | peak LR `{2e-4, 3e-4}`; decay `{0.01, 0.10}`; floor `1e-5` |

Reject a candidate if a run becomes nonfinite, if final validation loss exceeds
its observed minimum by more than `0.25`, or if update-to-weight trajectories
show discontinuous excursions.

## Stage 2 — two-epoch qualification

Promote surviving candidates to seeds `1337,2027,4099` and two
corpus-equivalent epochs. Rank strictly by:

1. mean best validation cross-entropy;
2. worst-seed best validation cross-entropy;
3. seed standard deviation;
4. final-minus-best validation-loss drift.

Do not use alpha, ERG gap, trap counts, BLEU, test loss, or test accuracy to
select the optimizer profile.

## Stage 3 — frozen matched replication

After locking one profile per optimizer, run the same eight seeds for every arm:

```text
1337, 2027, 4099, 5003, 6007, 7013, 8017, 9011
```

Report final and validation-selected task metrics with run-level 95% Student-t
intervals and matched-seed paired contrasts. Perplexity intervals must be
obtained by exponentiating the corresponding loss-space interval.
