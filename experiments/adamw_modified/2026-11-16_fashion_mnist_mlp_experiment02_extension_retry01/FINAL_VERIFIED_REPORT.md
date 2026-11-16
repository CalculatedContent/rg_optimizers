# Verified Fashion-MNIST continuation RETRY01

All 120 models reached the fixed epoch60 endpoint. All120 passed numerical verification and the predeclared functional criterion; no failed or unsupported outcomes were removed. The graft improves accuracy relative to AdamW, but worsens cross-entropy and average per-model distance to alpha=2. Extending training did not rescue the proposed joint alpha-alignment/performance prediction in this protocol.

[Published report and plots](RESULTS.md) · [Downloaded-bundle checks](verification/Downloaded_Artifact_Verification.json) · [Archive receipts](verification/Artifact_Receipts.json) · [Recomputed intervals](verification/recomputed_overall_effects.csv)

## Epoch60 means

Distance is the mean of each model’s absolute distance, not the absolute distance of the mean alpha. The distinction matters because models can fall on both sides of2.

| Arm | FC1 alpha | FC2 alpha | FC1 mean distance | FC2 mean distance | Test accuracy (%) | Test CE (nat) |
|---|---:|---:|---:|---:|---:|---:|
| adamw | 1.81322 | 1.95360 | 0.18944 | 0.09556 | 89.24375 | 0.33910 |
| magnitude_graft | 1.92909 | 2.13296 | 0.25577 | 0.14950 | 89.66325 | 0.43362 |
| sgdm | 2.19190 | 2.01528 | 0.29544 | 0.10488 | 89.48425 | 0.34996 |

## Paired optimizer contrasts at epoch60

Eight settings averaged within each seed; n=5 independent paired seeds. Each entry is mean difference [95% Student-t interval]. Intervals are descriptive and not adjusted for multiple comparisons. Positive accuracy is favorable; positive distance or CE is unfavorable.

| Metric | Graft minus AdamW | SGDM minus AdamW |
|---|---:|---:|
| alpha_fc1 | +0.11587 [+0.07703, +0.15471] | +0.37868 [+0.34686, +0.41050] |
| alpha_fc2 | +0.17936 [+0.16439, +0.19433] | +0.06168 [+0.03670, +0.08665] |
| distance_fc1 | +0.06632 [+0.04263, +0.09002] | +0.10600 [+0.07661, +0.13539] |
| distance_fc2 | +0.05394 [+0.02762, +0.08025] | +0.00932 [-0.00805, +0.02670] |
| mean_abs_alpha_minus_2 | +0.06013 [+0.03699, +0.08327] | +0.05766 [+0.04710, +0.06823] |
| test_accuracy_pct | +0.41950 [+0.27635, +0.56265] | +0.24050 [+0.13012, +0.35088] |
| test_loss | +0.09452 [+0.08849, +0.10054] | +0.01086 [+0.00584, +0.01588] |

## Changes from epoch20 to60

| Arm | Accuracy change (pp) | Test CE change (nat) | Mean hidden distance change |
|---|---:|---:|---:|
| adamw | +0.30775 [+0.26431, +0.35119] | +0.01699 [+0.01527, +0.01872] | -0.01046 [-0.02829, +0.00738] |
| magnitude_graft | +0.05250 [-0.05003, +0.15503] | +0.09371 [+0.08966, +0.09775] | +0.02074 [+0.00078, +0.04070] |
| sgdm | +0.20550 [+0.07667, +0.33433] | +0.03198 [+0.02936, +0.03460] | +0.00071 [-0.00694, +0.00835] |

Graft accuracy did not demonstrably increase during continuation: its paired interval includes zero. Relative to AdamW, the graft accuracy advantage shrank by0.25525pp [−0.35822,−0.15228], its CE penalty increased by0.07671nat [0.07290,0.08052], and its mean hidden distance penalty increased by0.03120 [0.00703,0.05537]. Full per-layer within-arm changes and SGDM effect changes are retained in the linked tables.

## Verification and limits

- Exact source runner/protocol/report hashes match the launch manifest. All120 saved epoch20 checkpoint hashes and hyperparameters passed restoration checks; no optimizer reset. Five parent seed artifact IDs and all600 checkpoint-file hashes were checked in the downloaded bundles (including duplicate last/final files).
- All120 current-runtime preflight replays and all120 epoch21→22 continuation replays passed exact checks. Historical cross-runtime replay was not bitwise exact: maximum loss discrepancy1.1920928955078125e-7 under the frozen1e-6 bound; maximum statistics discrepancy was0.019994149 of its declared bound. No tolerance was changed.
- All9 deliberately incorrect input/state/rate controls were rejected. Shared initialization and every recorded minibatch-order fingerprint matched across arms. Final restored raw spectra matched exactly; independently recomputed test CE from saved logits differed by at most3.003e-8nat.
- Recorded graft magnitudes include115800 nonzero matrix updates with maximum formula-level relative discrepancy1.1920928955078125e-7. The independent resumed15-update contract retained Adam moments and bias updates exactly, with minimum Adam-direction cosine0.9999999936. Actual float32 weight-subtraction norm error was at most2.66915e-6 relative.
- All240 final hidden-layer fits meet the declared support criteria. All120 FC3 fits remain exploratory/unsupported because the layer has only10 eigenvalues. Functional quality and spectral alignment are separate labels.
- All four published plots were inspected. Raw validation CE rises during much of continuation, especially for the graft; the evidence does not establish a failed implementation or prove the original learning rates optimal.
- This continuation was chosen after seeing epoch20 results and reuses the same seeds and test set. It is an exploratory follow-up, not independent confirmation of the theory. No further training was launched; GPT remains on hold.

## Exploratory loss-tail diagnostic

Computed only from the saved fixed-epoch60 test logits, without training or endpoint selection. Each model’s95th/99th loss quantiles and worst100 of10000 examples were calculated, then settings averaged within seeds. Tail sets can contain different examples in different models. This is the tail of per-example prediction loss, not the weight spectrum.

| Metric | Graft minus AdamW | Paired95% interval |
|---|---:|---:|
| loss_q95 | +0.53000 | [+0.43688, +0.62313] |
| loss_q99 | +3.29792 | [+3.07354, +3.52229] |
| worst1pct_mean_loss | +5.55917 | [+5.41549, +5.70286] |
| worst1pct_share_of_total_loss | +0.07538 | [+0.06977, +0.08099] |
| confident_errors_gt90pct | +148.82500 | [+139.97501, +157.67499] |

Worst1% mean loss: AdamW7.8075nat, graft13.3667nat, SGDM8.8374nat. The worst1% account for22.81%,30.35%,25.10% of total loss respectively (means of model-level fractions). The graft has on average386.8 versus238.0 errors with confidence above90% per10000 test examples. These measurements substantiate worse high-loss behavior alongside better classification accuracy.

[Model-level loss-tail data](verification/exploratory_loss_tail_models.csv) · [Paired loss-tail effects](verification/exploratory_loss_tail_effects.csv) · [Reproducible read-only audit](verification/verify_downloaded.py)

The audit script expects itself to be beside the downloaded/extracted artifact folders, with the six outer ZIPs extracted and each inner seed ZIP still present. It performs no training.
