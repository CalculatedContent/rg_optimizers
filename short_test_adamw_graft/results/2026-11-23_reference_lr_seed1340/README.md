# TinyShakespeare reference learning-rate tuning

The selected alternative momentum-SGD reference peak learning rate was **0.4** for the AdamW graft. The original graft used 0.1. Both used native AdamW peak learning rate 0.001. The table compares all four optimizers on confirmation seed 1340. Loss is cross-entropy in nats per character, and accuracy is the fraction of correct next-character predictions.

## Calibration

On calibration seed 2027, the two alternatives with the lowest validation loss at step 4,000 advanced alongside the original rate. The alternative with the lowest validation loss at step 20,000 qualified for confirmation, with ties going to the lower rate. The original rate served as a calibration comparator. Calibration used only the training and validation splits.

| Reference peak LR | Step | Validation loss | Outcome |
| ---: | ---: | ---: | :--- |
| 0.05 | 4,000 | 1.810202 | Stopped |
| 0.1 | 4,000 | 1.760204 | Advanced |
| 0.2 | 4,000 | 1.710886 | Advanced |
| 0.4 | 4,000 | 1.716785 | Advanced |
| 0.8 | 4,000 | 1.763216 | Stopped |
| 0.1 | 20,000 | 1.480594 | Original comparator |
| 0.2 | 20,000 | 1.479674 | Finalist |
| 0.4 | 20,000 | 1.474755 | Selected |

## Confirmation

The fixed endpoint follows 20,000 updates. Each optimizer's validation-selected row uses the checkpoint with its lowest full validation loss.

| Optimizer | Checkpoint | Step | Validation loss | Test loss | Test accuracy | Weighted α |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tuned graft | Fixed endpoint | 20,000 | 1.476417 | 1.634099 | 52.79% | 3.0587 |
| Original graft | Fixed endpoint | 20,000 | 1.481386 | 1.642861 | 52.65% | 2.8752 |
| Native AdamW | Fixed endpoint | 20,000 | 1.460256 | 1.617580 | 52.92% | 3.2888 |
| SGD with momentum | Fixed endpoint | 20,000 | 1.516208 | 1.695734 | 51.48% | 2.2624 |
| Tuned graft | Validation-selected | 19,000 | 1.473438 | 1.628925 | 53.04% | 3.1502 |
| Original graft | Validation-selected | 20,000 | 1.481386 | 1.642861 | 52.65% | 2.8752 |
| Native AdamW | Validation-selected | 16,000 | 1.456495 | 1.604805 | 52.97% | 3.2246 |
| SGD with momentum | Validation-selected | 20,000 | 1.516208 | 1.695734 | 51.48% | 2.2624 |

## Paired test-loss differences

| Checkpoint | Tuned − native AdamW | Tuned − original graft | Tuned − momentum SGD |
| --- | ---: | ---: | ---: |
| Fixed endpoint | +0.016519 | -0.008762 | -0.061635 |
| Validation-selected | +0.024120 | -0.013936 | -0.066809 |

Negative differences indicate lower loss for the tuned graft. The comparison covers one confirmation seed. Across-seed variability remains unmeasured.

Each weighted α averages the exponents from all 18 unique matrices, using weights √tail / (D + 0.02). Here tail counts eigenvalues in the fitted power-law tail and D is the Kolmogorov–Smirnov distance.

## Reproduction

From the repository root, use the [experiment environment](../../README.md) and run the driver with a fresh output directory.

```bash
python short_test/download_data.py --output short_test_adamw_graft/data/input.txt
python short_test_adamw_graft/reference_lr_study.py --device cpu --data short_test_adamw_graft/data/input.txt --output runs/reference_lr_study
```

[Full-precision comparison](comparison.csv) · [Calibration scores](calibration.csv) · [Endpoint layer fits](endpoint_layers.csv) · [Tuned result, calibration records and designs](raw_tuned_results.zip) · [Archived original graft and controls](../2026-11-23_seed1340/raw_results.zip)

[Download the trained checkpoint](trained_checkpoint.zip). Reloading its endpoint and validation-selected weights reproduced both reported test losses.
