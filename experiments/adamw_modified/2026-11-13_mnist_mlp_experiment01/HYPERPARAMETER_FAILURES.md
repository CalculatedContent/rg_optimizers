# Hyperparameter-setting failures

**Failed setting: magnitude-only graft, batch128, LR multiplier2.** All original seeds 31001–31005 and the separately authorized fresh seed31006 collapsed to 11.35% test accuracy. These outcomes remain failures in the record; they are excluded from the stable-training comparison rather than interpreted as evidence about useful spectral learning.

Peak native Adam rates were [0.002, 0.002, 0.002]. Peak SGDM target layer rates were [0.08, 0.4, 1.6]. Momentum was 0.9; Adam β₂=0.999, ε=1e−8; five-epoch warmup and cosine schedule to 10% peak. The shared manual decay convention is unchanged. For grafted matrices, the Adam rate controls the native update before magnitude normalization; the SGDM rates set the target magnitude. Adam bias updates retain the native Adam rates.

The new seed reproduced the failure without a source or parameter change. All 256 FC2 ReLUs are inactive on every test input in each collapsed final checkpoint; logits are constant across inputs and the predicted class is always digit1. Matrix-gradient norms become zero. The original checkpoint diagnostics and fresh-seed verification directly establish this failure mode.

The seed31006 sampled logs show an update surge in epoch28. Compared with epoch27, the largest sampled FC2 update norm rises from approximately1.55 to59.19, and the output-layer update norm from2.92 to191.97. By the final sampled batch of epoch28 all three matrix-gradient norms are zero. These are sampled observations, not identification of the exact initiating minibatch or proof that one particular layer caused the event.

Batch128 at LR×0.5 and ×1 worked for all original seeds and all arms. Native AdamW and SGDM also remained functional at batch128/LR×2. This is therefore recorded as optimizer-specific hyperparameter instability at the tested combination, not as a failure of every batch128 run. Lower-rate success supports excessive update size as a working explanation; the causal trigger remains unresolved.

## Reporting policy

- Main stable analysis: exclude the entire batch128/LR×2 setting for all three arms, preserving a balanced eight-setting comparison with five paired seeds (120 runs).
- Failures: six graft runs, separately listed with complete measurements.
- Successful controls at the failed setting: ten original controls and two fresh-seed controls are retained outside the stable subset. They are not labeled failures.
- The subset is post hoc. Original full-grid statistics and source data remain available; no outcome is overwritten or silently removed.
- All settings that trained successfully remain in the stable subset, including settings where α is farther from2 or the graft does not improve accuracy.

[Failure rows](results/hyperparameter_failures.csv) · [All run dispositions](results/run_disposition.csv) · [Original failure diagnostics](verification/Low_Accuracy_Diagnostics.json) · [Fresh-seed diagnostics](replications/2026-11-14_batch128_seed31006/verification/Replication_Verification.json) · [Original full-grid results](FULL_GRID_RESULTS.md)

## Explicit model labels

[Model status](MODEL_STATUS.md) and [the model register](results/model_status.csv) mark these six outcomes `FAILED`. The 12 successful controls at this setting remain `USABLE`; their exclusion from the shared stable subset is a comparison-design decision, not a failure label. `Complete` in historical execution logs means training reached its endpoint, not that the resulting classifier is usable.
