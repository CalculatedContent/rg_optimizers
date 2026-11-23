# TinyShakespeare magnitude graft, seed 1340

The graft completed all 20,000 updates on Linux CPU with the frozen Shakespeare recipe. The AdamW peak learning rate was 0.001 and the momentum-SGD reference peak was 0.1. The same-seed controls come from the archived October comparison. The primary outcome is test cross-entropy in nats per character at the fixed endpoint. A secondary analysis uses the checkpoint with the lowest full validation loss.

| Optimizer | Checkpoint | Step | Validation loss | Test loss | Confidence-weighted α |
| --- | --- | ---: | ---: | ---: | ---: |
| AdamW with graft | Fixed endpoint | 20,000 | 1.481386 | 1.642861 | 2.8752 |
| Native AdamW | Fixed endpoint | 20,000 | 1.460256 | 1.617580 | 3.2888 |
| SGD with momentum | Fixed endpoint | 20,000 | 1.516208 | 1.695734 | 2.2624 |
| AdamW with graft | Validation-selected | 20,000 | 1.481386 | 1.642861 | 2.8752 |
| Native AdamW | Validation-selected | 16,000 | 1.456495 | 1.604805 | 3.2246 |
| SGD with momentum | Validation-selected | 20,000 | 1.516208 | 1.695734 | 2.2624 |

At the fixed endpoint, graft minus native AdamW test loss was **+0.025281**. Graft minus SGD with momentum was **-0.052873**. Across-seed uncertainty requires further seeds.

The trajectory covers initialization and all 40 validation checkpoints. The layer table includes α, KS distance, tail count, fit weight, and fit status for each of the 18 physical matrices. The weighted mean uses the square root of the tail count divided by the KS distance plus 0.02.

[Comparison](comparison.csv) · [Trajectory](trajectory.csv) · [Per-layer fits](layers.csv) · [Full raw graft and control outputs](raw_results.zip) · [Frozen design](design.json) · [Archived control provenance](control_provenance.json)

To reproduce this specific run, follow the [experiment setup](../../README.md) and pass `--arm graft --seed 1340` to the training command. The remaining four confirmation seeds were held for review.
