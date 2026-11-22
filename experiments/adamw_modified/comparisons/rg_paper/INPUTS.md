# Data dictionary and reproduction scope

The paper uses 540 digit checkpoints, 341 original Fashion checkpoints and 289 Fashion continuation checkpoints. The continuation reaches epoch 120 for 49 of 120 runs. A checkpoint contains FC1, FC2 and FC3. The independent unit for the paired comparisons is the seed.

| Input | What it contains | What can be recomputed |
|---|---|---|
| `inputs/spectra_*.npz` | Original Gram eigenvalues from the 1,170 trained checkpoints. Keys are seed, batch, learning-rate multiplier, optimizer and epoch within each named cohort. Each row concatenates 256 FC1, 256 FC2 and 10 FC3 eigenvalues. | All 3,510 tail cutoffs, α, KS distances, weights and signed ERG counts. |
| `optimizer_checkpoints.csv` | Stored test accuracy (percent) and layer fits, keyed by cohort, optimizer, seed, batch, rate, epoch and layer. | Six panel winners, α averages, table of matched seeds and ERG sign counts. |
| `validation_endpoints.csv` | Validation accuracy (%) for every completed epoch-120 checkpoint. | Winners selected on validation within each of the 3 sweeps. |
| `inputs/calibration_counts.csv` | For each run and epoch: test-image count and number correct, plus count, number correct and sum of top probabilities for predictions ≥99% confident. | Accuracy and absolute calibration gap at epochs 60 and 120 for 292 matched run pairs. |
| `inputs/fashion_scores_{seed}.npz` | Archived test and validation logits, ordered test labels, and hidden-layer spectra at epochs 60 and 120 for unaugmented Fashion seeds 33001 and 33002. | Per-image losses, hardest-tenth attribution, validation-fitted logit rescaling and top-8 trace shares. |
| `confidence_run_pairs.csv` | Original paired summaries, including epoch-specific gaps. | Cross-checks for the calibration counts. Other descriptive columns are archived summaries. |
| `muon_30_epochs.csv`, `Muon_500_Reported.csv` | Separate short-run mean/range across 3 seeds, and the long-run notebook’s printed checkpoints. | Figure 18 and the equal-weight Muon average and across-layer SD. Figure 17 uses the original image. |

**How α is fitted.** For the 1,170 plotted checkpoints, sort positive eigenvalues and fit each candidate tail. With k retained values and L = Σ log(λ/λmin), α = 1 + k/L for FC1/FC2 and α = 1 + (k−1)/L for FC3. Select the minimum KS distance using the empirical CDF 0, 1/k, …, (k−1)/k. FC3 considers k = 8, 9, 10. This reproduces every stored cutoff. The separately named small-N routine in the paired follow-up runs uses its own tail-size penalty. A layer’s weight in the paper average is √k/(D + 0.02), with D the fitted KS distance.

**How the ERG gap is counted.** Normalize each layer’s eigenvalues by their arithmetic average. Starting with the largest, add their logarithms until the first negative sum and include that crossing eigenvalue. The ERG gap is this retained count minus the fitted tail count, in eigenvalues.

**Selection and uncertainty.** Test-selected green triangles and validation-selected winners answer different questions. Figure 15 omits 17 diagnosed collapsed checkpoints and places 19 valid high-α checkpoints beyond the displayed x-range. All 1,170 remain in the CSV. Muon’s ± measures spread across 3 layers. Paired confidence intervals on the central digit comparisons use 5 independent seeds.

**What a download contains.** SHA256SUMS identifies the fixed replay inputs. The one-command replay needs these files and the listed Python packages. The optional archive audit in the README starts a step earlier, from historical seed ZIPs. A fresh model-training run or a complete inspection of all per-image scores requires the original datasets, environments and checkpoints. The long-run Muon notebook did not publish a complete epoch-by-epoch numerical table. The saved image preserves the trajectory, and the numeric verification uses the checkpoints it printed.

Archive owners can regenerate the small replay bundle with `python export_inputs.py ARCHIVES EVALUATED_SCORES LABEL_DIRECTORY OUTPUT`. The first directory contains the original seed ZIPs. The second holds evaluated arrays named `{cohort}__seed_{seed}__{setting}__{optimizer}__{epoch}.npy`. The label directory holds `digits-y.npy` and `fashion-y.npy` in test order. The ordinary replay reads the bundled inputs directly.
