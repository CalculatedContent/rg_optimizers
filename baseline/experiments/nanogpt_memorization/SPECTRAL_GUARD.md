# Adaptive spectral guard: Muon and SGD momentum

`spectral_guard_train.py` turns WeightWatcher alpha into an explicit training-time control signal.

The trainer operates in fixed windows. Before each window it snapshots the model, optimizer state, RNG state, and accepted exposure counts. At the end of the window it runs the same pinned WeightWatcher analysis on all six transformer matrices.

A window is accepted only when **both** `alpha_raw` and `alpha_clip_xmax` are at least `--target-alpha` for every measured matrix. Otherwise the window is rolled back and retried.

For each offending matrix the controller:

1. lowers that matrix's learning-rate multiplier;
2. increases a matrix-specific L2 anchor gradient toward the last accepted spectral checkpoint;
3. for `sgd_momentum`, also increases dropout on the transformer block containing the offending matrix.

A violation below `--alpha-floor` gets the stronger LR backoff. A violation between the floor and the target gets the milder backoff.

Dropout is block/activation dropout, not a direct operation on one weight matrix. In the current one-block nanoGPT, all six monitored matrices belong to block 0, so a dropout intervention on any offending matrix changes the attention/MLP dropout for that block.

## Scientific interpretation

This procedure explicitly optimizes against the WeightWatcher diagnostic. Therefore it is an **intervention experiment**, not an unbiased comparison of ordinary Muon, SGD, and AdamW. A useful question is whether actively maintaining alpha above 2 suppresses the example-specific membership/overfitting effect. Do not use a successful guard run as evidence that unmodified Muon or SGD naturally maintains alpha above 2.

Tune controller hyperparameters on a development seed (for example 2027), then freeze those hyperparameters before using another seed for comparison. The controller itself may still react to alpha on the comparison seed; what must stay fixed is the policy (threshold, backoff factors, anchor strengths, dropout schedule/caps, guard interval).

## Adaptive Muon development run

Based on the fixed-LR development runs already performed, start Muon at 0.25 of the repository matrix LR (`0.005` instead of `0.02`) and let the guard reduce individual matrix LRs only when necessary:

```bash
ROOT="/private/tmp/nanogpt_muon_spectral_guard_$(date +%Y%m%d_%H%M%S)"

caffeinate -dimsu python -u spectral_guard_train.py \
  --optimizer muon \
  --stage pilot \
  --seed 2027 \
  --device mps \
  --root "$ROOT" \
  --target-alpha 2.0001 \
  --alpha-floor 1.95 \
  --guard-interval 125 \
  --initial-lr-scale 0.25 \
  --mild-backoff 0.75 \
  --hard-backoff 0.5 \
  --anchor-start 0.5 \
  --anchor-growth 2 \
  --max-anchor 32 \
  --max-retries 8
```

Accepted windows print `ACCEPT ... min_alpha=...`. Failed trial windows print `ROLLBACK ...`, restore the last accepted state, adapt only the offending matrices, and retry.

The accepted spectra are under `spectral/`; rejected trial spectra are retained under `spectral_rejected/`; every control decision is recorded in `control_events.jsonl`.

## Adaptive SGD + momentum + dropout development run

This uses the repository's `sgd_momentum` profile for every trainable parameter. The six transformer matrices receive individually controllable LR multipliers. When a matrix violates the alpha gate, the whole block's attention/MLP dropout is raised because dropout operates on activations rather than a single matrix.

```bash
ROOT="/private/tmp/nanogpt_sgd_spectral_guard_$(date +%Y%m%d_%H%M%S)"

caffeinate -dimsu python -u spectral_guard_train.py \
  --optimizer sgd_momentum \
  --stage pilot \
  --seed 2027 \
  --device mps \
  --root "$ROOT" \
  --target-alpha 2.0001 \
  --alpha-floor 1.95 \
  --guard-interval 125 \
  --initial-lr-scale 1.0 \
  --mild-backoff 0.75 \
  --hard-backoff 0.5 \
  --anchor-start 0.5 \
  --anchor-growth 2 \
  --max-anchor 32 \
  --dropout-step 0.025 \
  --max-dropout 0.25 \
  --max-retries 8
```

A hard violation (`alpha < 1.95`) increases block dropout by two dropout steps on retry; a mild violation increases it by one step. Dropout is capped at 0.25. The controller fails rather than silently relaxing the alpha target.

## Resume after interruption

The safe restart point is always the latest **accepted** spectral checkpoint. An interrupted or rejected window is replayed from that checkpoint:

```bash
python -u spectral_guard_train.py <same arguments> --resume
```

Do not change controller arguments when resuming; the manifest check rejects a changed policy.

## Extensive overfitting/membership audit

After a complete spectral-guard run, use the same 512-example audit and membership statistics used for AdamW:

```bash
RUN="<complete spectral-guard run directory>"
AUDIT="/private/tmp/nanogpt_guard_audit_$(date +%Y%m%d_%H%M%S)"

caffeinate -dimsu python -u audit_spectral_guard.py \
  --run-dir "$RUN" \
  --device mps \
  --background-examples 512 \
  --audit-seed 20260915 \
  --output "$AUDIT"

python membership_stats.py \
  --audit "$AUDIT" \
  --bootstrap 5000 \
  --permutations 20000

cat "$AUDIT/report.md"
cat "$AUDIT/membership/membership_report.md"
```

The audit remains read-only. It tests replay-vs-fresh NLL, baseline-corrected example-specific membership effects, ROC-AUC, bootstrap intervals, permutation tests, canary extraction, partial extraction, hint-assisted continuation, prefix dependence, rule perturbations, and position sensitivity.

## Important limit

`--guard-interval 125` means the claim is: **every measured accepted spectral checkpoint passed the alpha gate**. WeightWatcher is not run after every optimizer update, so the code cannot claim that alpha never crossed below 2 between measurements. Use a shorter guard interval if that distinction matters enough to justify the extra WeightWatcher cost.
