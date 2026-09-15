# Stronger example-specific likelihood memorization test

This analysis strengthens the replay-versus-fresh result from the completed AdamW verbatim run without retraining the model.

`membership_stats.py` consumes a completed `audit_checkpoints.py` output directory. It uses the same fixed early-training examples and the same fixed fresh examples at initialization and at every saved checkpoint.

For each example it computes

`NLL improvement = NLL(step 0) - NLL(step t)`.

The primary membership effect is

`mean improvement(seen early) - mean improvement(fresh)`.

A positive value means examples actually encountered in training improved more than matched fresh examples after correcting each example for its own initialization difficulty.

The script also reports:

- baseline-corrected ROC-AUC for classifying seen versus fresh examples;
- raw-NLL ROC-AUC for comparison;
- 95% nonparametric bootstrap confidence intervals over examples;
- one-sided permutation p-values;
- Holm correction across saved checkpoints;
- a global max-statistic permutation test that corrects for selecting the strongest saved checkpoint after looking across the whole trajectory.

These are within-model uncertainty and randomization checks. They do **not** turn examples or checkpoints into independent training replications. A general optimizer/model claim still requires independently trained seeds.

## Recommended high-power rerun of the existing checkpoints

No training is performed. Increase the fixed background cohorts from 64 to 512 examples:

```bash
RUN="/private/tmp/nanogpt_memorization_20260914_222611/full/repository/verbatim/adamw/seed_1337"
AUDIT="/private/tmp/nanogpt_membership_audit_$(date +%Y%m%d_%H%M%S)"

caffeinate -dimsu python -u audit_checkpoints.py \
  --run-dir "$RUN" \
  --device mps \
  --background-examples 512 \
  --audit-seed 20260915 \
  --output "$AUDIT"

python membership_stats.py \
  --audit "$AUDIT" \
  --bootstrap 5000 \
  --permutations 20000

cat "$AUDIT/membership/membership_report.md"
```

The audit reconstructs the completed training stream and verifies that the fresh inputs were never used in training. The membership analysis then removes baseline cohort difficulty by using each example's step-0 score.

## Outputs

- `membership/membership_stats.csv`: per-checkpoint effect sizes, AUCs, confidence intervals, raw permutation p-values, and Holm-adjusted p-values.
- `membership/membership_metadata.json`: source audit and global max-statistic results.
- `membership/membership_report.md`: final checkpoint and strongest-checkpoint summary plus interpretation limits.

## Interpretation

Evidence for example-specific likelihood memory is strongest when all of the following agree:

1. baseline-corrected membership effect is positive;
2. its bootstrap interval excludes zero;
3. corrected membership AUC is above 0.5;
4. permutation tests are small after checkpoint correction;
5. the global max-statistic test remains significant;
6. the pattern is stable under a larger cohort and later under independent training seeds.

This test addresses likelihood/membership memorization. It is separate from exact sequence extraction, which the original canary experiment did not detect.
