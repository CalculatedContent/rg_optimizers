# High-alpha Muon control for the membership-memorization study

Goal: compare the completed AdamW seed-1337 run, which developed alpha below 2, against a Muon run that is qualified **in advance** to keep both `alpha_clip_xmax` and `alpha_raw` at or above 2.0 at every saved checkpoint.

This is deliberately a two-stage protocol so the Muon hyperparameter is not tuned on the seed-1337 comparison itself.

## 1. Development-only Muon pilot

Use seed 2027 to select the largest pre-registered hidden-matrix LR scale that passes the alpha gate. The candidate list is fixed, descending:

`1.0, 0.5, 0.25, 0.125, 0.0625`.

Only Muon's hidden-matrix learning-rate pair is scaled. Auxiliary AdamW settings, momentum, weight decay, Newton-Schulz steps, data, model, schedule geometry, WeightWatcher settings, and objective remain unchanged.

```bash
DEV_ROOT="/private/tmp/nanogpt_muon_alpha_dev_$(date +%Y%m%d_%H%M%S)"

caffeinate -dimsu python -u muon_alpha_control.py pilot \
  --device mps \
  --root "$DEV_ROOT"

cat "$DEV_ROOT/muon_alpha_lock.json"
```

Selection rule: choose the **largest** listed scale whose completed pilot has both raw alpha and clip_xmax alpha >= 2.0 for all six monitored matrices at every saved spectral checkpoint. If no candidate passes, the script exits without qualifying a control. Do not tune the threshold or candidate list on seed 1337.

The plain repository Muon profile is scale 1.0. Therefore, if ordinary Muon already stays above the gate, it is selected and no lower-learning-rate variant is tried.

## 2. Locked full Muon run on seed 1337

After the pilot produces a lock:

```bash
MUON_ROOT="/private/tmp/nanogpt_muon_high_alpha_$(date +%Y%m%d_%H%M%S)"

caffeinate -dimsu python -u muon_alpha_control.py full \
  --lock "$DEV_ROOT/muon_alpha_lock.json" \
  --device mps \
  --root "$MUON_ROOT"
```

The full run uses verbatim condition, seed 1337, and the locked scale. It then checks every saved raw and clipped alpha again. The run is a valid **high-alpha control only if** `full_alpha_gate.json` says `"passed": true`.

A pilot pass does not guarantee the full run will pass. If the full run crosses below 2, report that outcome; do not silently select another scale on seed 1337.

## 3. Run the same 512-example checkpoint membership audit

Find the full Muon run path from the command output. It has the form:

`$MUON_ROOT/full/muon_alpha_s<SCALE>/verbatim/muon/seed_1337`

Then:

```bash
MUON_RUN=$(find "$MUON_ROOT/full" -path '*/verbatim/muon/seed_1337' -type d | head -1)
MUON_AUDIT="/private/tmp/nanogpt_muon_membership_audit_$(date +%Y%m%d_%H%M%S)"

caffeinate -dimsu python -u audit_muon_checkpoints.py \
  --run-dir "$MUON_RUN" \
  --device mps \
  --background-examples 512 \
  --audit-seed 20260915 \
  --output "$MUON_AUDIT"

python membership_stats.py \
  --audit "$MUON_AUDIT" \
  --bootstrap 5000 \
  --permutations 20000

cat "$MUON_AUDIT/membership/membership_report.md"
```

Use the same audit seed and cohort size for AdamW and Muon.

## 4. AdamW high-power membership analysis

Re-run the existing AdamW checkpoints with 512 examples if not already done:

```bash
ADAMW_RUN="/private/tmp/nanogpt_memorization_20260914_222611/full/repository/verbatim/adamw/seed_1337"
ADAMW_AUDIT="/private/tmp/nanogpt_adamw_membership_audit_$(date +%Y%m%d_%H%M%S)"

caffeinate -dimsu python -u audit_checkpoints.py \
  --run-dir "$ADAMW_RUN" \
  --device mps \
  --background-examples 512 \
  --audit-seed 20260915 \
  --output "$ADAMW_AUDIT"

python membership_stats.py \
  --audit "$ADAMW_AUDIT" \
  --bootstrap 5000 \
  --permutations 20000
```

## 5. Compare the matched optimizer trajectories

Only do this if the Muon full alpha gate passed:

```bash
COMPARE="/private/tmp/nanogpt_membership_optimizer_compare_$(date +%Y%m%d_%H%M%S)"

python compare_membership.py \
  --adamw "$ADAMW_AUDIT/membership" \
  --muon "$MUON_AUDIT/membership" \
  --muon-gate "$MUON_ROOT/full_alpha_gate.json" \
  --output "$COMPARE"

cat "$COMPARE/comparison_report.md"
```

The comparison script verifies the two source runs have the same condition, stage, seed, data identity, initialization, model, and device. It refuses a Muon arm that failed the spectral gate.

## Interpretation

The target contrast is:

- AdamW: observed alpha < 2 plus any baseline-corrected example-specific likelihood membership effect.
- Muon: alpha >= 2 throughout plus the same membership analysis.

If AdamW shows a reproducible membership effect while the qualified high-alpha Muon control does not, that is evidence of an association between the low-alpha training regime and example-specific likelihood memorization in this controlled setup. One seed is still not an optimizer-level proof. The next step after a clear paired result is independent seed replication with the Muon scale kept locked.
