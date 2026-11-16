# Fashion-MNIST: scale-equivariance audit and matched follow-up

Status on 2026-11-16: PREPARATION AUTHORIZED; TRAINING NOT LAUNCHED.

Charles explicitly confirmed “yes go” after clarification: continue preparing and launch the next experiment after verification. Text updates requested during the next several hours. Historical runs/results remain immutable; GPT stays on hold.

## Scientific question

Does the magnitude-only graft reduce AdamW's tendency toward alpha below 2, and improve held-out predictive performance? An alpha above 2 is not automatically a functional failure. Keep accuracy, cross-entropy, calibration, sub-2 deficit and symmetric distance as distinct outcomes. Do not infer overfitting from alpha alone or assume this hypothesis is already proved.

## First work: prerequisites

1. Review primary Fashion-MNIST literature and published dense-MLP benchmarks. Distinguish comparable dense networks from CNN/ensemble/augmentation results; no CNNs. Record actual source links, architecture, split, schedule, augmentation and accuracy.
2. Derive the proposed rescaling explicitly. Distinguish scale invariance of a loss, covariance/equivariance of an optimizer trajectory, and any additional RG assumptions. Include weights, inputs or adjacent layers as relevant, gradients, momentum, second moments, bias correction, epsilon, learning rates, biases and decay. Do not claim that SGDM magnitudes alone guarantee the required property.
3. Numerically test nonzero optimizer states under the declared transformation, with intentionally incorrect transformation controls. Preserve Adam direction/moments and exact native bias updates in the graft, using SGDM history on the graft's own trajectory. A dimensionless multiplier is allowed only if the same demonstrated transformation law survives; no interpolating in an unverified Adam magnitude.
4. Inspect current WeightWatcher source/documentation for the requested small-matrix/FC3 correction. Record its actual API and assumptions; test on controlled spectra. Save all three layer spectra and alpha estimates, including support/uncertainty. Do not invent a correction or call ten eigenvalues a supported ordinary tail fit.
5. Write and freeze the concrete protocol and source hashes before training. Record unresolved theoretical limitations candidly. If the required scale property cannot be established for the actual implementation, report the blocker instead of launching a falsely labeled invariant experiment.

## Bounded execution

First launch one development pilot with the three primary arms: native AdamW, exact magnitude-only graft, SGDM. Use the existing dense 784→256→256→10 Fashion-MNIST baseline initially to preserve comparability. Review literature before freezing one schedule/rate configuration; record all differences from the prior experiment.

If numerical/functional pilot checks pass and measured throughput fits the budget, run one fixed matched comparison with five fresh paired seeds (32001–32005), the same initialization, split, minibatch order and augmentation realization within each seed, and the same fixed endpoint across arms. Pilot seed 32999 is excluded from inference. No alpha-based selection or success gate. No test-based tuning. Freeze endpoint and rates before the pilot; retain raw validation curves and all failures. Check restored optimizer/RNG state, replay, graft magnitude/direction/moments/biases, and common initialization/order. Save logits, per-epoch losses and all-layer spectra.

Use a persistent GitHub Actions workflow with checkpoint/artifact preservation, bounded concurrency, a four-hour training cap and five-hour total cap from launch. Estimate runtime from the pilot rather than promising an unsupported finish time. Do not duplicate workflows or rerun completed trajectories. Layer ablations, weaker invariant graft constants, width changes and augmentation studies should be documented as proposed follow-ups rather than silently expanding this first bounded pass.

## Analysis and reporting

Keep this subfolder separate from every prior experiment. Save executable code, pinned protocol, literature review, invariance audit, FC3 estimator audit, run status, checkpoints/receipts, model register, results and plots. Update status with actual run ID only after a successful launch. Publish using a fresh GitHub head lease.

Five seeds are the independent replicates; report paired seed-level 95% intervals. Report all-layer alpha and sub-2 deficit alongside symmetric distance, accuracy, CE and calibration. Any new one-sided metric is post-hoc relative to historical work and prospective only for this newly frozen protocol. Keep the original weighted analysis unchanged. Reused Fashion test data are not untouched external confirmation.

## Existing evidence

- [General comparison](../../../comparisons/README.md)
- Parent run 43793781330 completed successfully; no continuation of that completed run is authorized by this plan.
- Previous published analysis commit: 97f4cfc8a3738d045e3485437ff98e5c8cd1a770.
