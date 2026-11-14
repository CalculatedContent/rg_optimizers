# Experiment 01: magnitude-only AdamW graft

**Complete. Main presentation: 120 runs across eight shared stable settings, five paired seeds, three optimizers.** The batch128/LR×2 graft collapse is reported separately as a hyperparameter-setting failure. No new training was performed for this cleanup.

The graft keeps Adam's adaptive update direction, moments, bias correction and bias updates. Only matrix-update magnitudes are replaced with SGDM target magnitudes computed along the graft's own gradient history. MNIST; ReLU MLP 784→256→256→10; 120 epochs; WeightWatcher 0.7.7 default fits.

## Stable-setting results

Across the eight shared stable settings, graft minus AdamW gives:

- FC1 α: **+0.324 [0.261, 0.387]**; FC2 α: **+0.177 [0.165, 0.190]**.
- Mean hidden distance |α−2|: **−0.056 [−0.085, −0.026]**, closer to 2.
- Test accuracy: **+0.091 percentage points [−0.014, 0.196]**; an overall accuracy advantage is unresolved.

Intervals use five paired seed averages, each averaged over the same eight settings. This stable subset was selected after observing failure, so these are descriptive, conditional results; they do not replace the original prespecified analysis. No selection uses α or treatment benefit.

![Stable settings](plots/Stable_Setting_Comparison.png)

The prespecified central comparison, batch256/LR×1, is unchanged:

| Optimizer | FC1 α | FC2 α | Mean hidden \|α−2\| | Test accuracy |
|---|---:|---:|---:|---:|
| AdamW | 1.783 | 1.762 | 0.228 | 98.220% |
| Magnitude graft | 2.075 | 1.910 | 0.083 | 98.344% |
| SGD + momentum | 2.067 | 1.900 | 0.083 | 98.446% |

At that central setting, the graft's paired accuracy change is +0.124 pp [0.034, 0.214]; its distance change is −0.145 [−0.174, −0.116].

## Working ranges and failures

Observed working settings for all arms: batch128 at LR×0.5/×1; batch256 and batch512 at LR×0.5/×1/×2. These are tested working settings, not a complete stability boundary or proof of optimal tuning. AdamW's late validation loss increases despite strong accuracy.

**Batch128/LR×2 is a failed graft setting: six of six tested seeds collapsed.** Its native AdamW and SGDM controls remained functional. Classify this as optimizer-specific hyperparameter instability, not as an ordinary low-accuracy spectral observation. The initiating cause of the update surge has not been isolated.

For a balanced stable comparison, exclude this entire setting from all arms, not just the failed grafts. Of 138 total runs, 120 enter the main subset, six are documented failures, and 12 successful controls remain available outside this subset. The extra seed is a separate replication and is not pooled into the original five-seed intervals.

## Reports and evidence

- [Stable-setting report and all included settings](RESULTS.md)
- [Hyperparameter failure report](HYPERPARAMETER_FAILURES.md)
- [Run disposition: all 138 outcomes](results/run_disposition.csv)
- [Original 135-run full-grid results](FULL_GRID_RESULTS.md) and [original narrative](FULL_GRID_ANALYSIS.md)
- [Fresh-seed replication](replications/2026-11-14_batch128_seed31006/README.md)
- [Original verification](verification/Final_Verification.json), [frozen protocol](frozen/Protocol.json), and [saved checkpoints](ARTIFACTS.md)

Original raw data, all original plots, all 690 checkpoints across the original pass and replication, training source and protocol are preserved. The clean report is rebuilt by `execution/build_stable_review.py` from original and replication endpoint CSVs; it runs no training. FC3 remains exploratory because its matrix has ten eigenvalues. No further experiment is running.
