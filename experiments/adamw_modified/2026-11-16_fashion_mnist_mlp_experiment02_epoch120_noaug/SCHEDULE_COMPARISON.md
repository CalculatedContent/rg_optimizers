# Existing AdamW cohorts compared at the same epoch

All rows below use the shared B256/LR×1 setting. Five seeds per cohort; validation data, not a newly evaluated test endpoint. The original cohort uses seeds31001–31005 and the augmented cohort32001–32005, so this is not a paired-seed intervention comparison.

| Quantity at epoch60 | Original, no augmentation | Newer, translation augmentation |
|---|---:|---:|
| Optimizer updates |12900|12900|
| Training examples processed |3300000|3300000|
| Adam LR |0.000010000|0.000580709|
| Fraction of peak LR |0.010000|0.580709|
| Cumulative nominal Adam LR |2.785325|10.636795579|
| Validation accuracy |90.456%|91.060%|
| Validation CE |0.282512620|0.250518757|
| Mean FC1 alpha |1.713819540|1.784923547|
| Mean FC2 alpha |1.925109652|2.042305726|

The exact per-model figures are in comparison_epoch60.csv. The original LR schedule reached0.1 of peak at20 and0.01 at60. The augmented schedule runs from five-epoch warmup through a single120epoch cosine, ending at0.1 of peak. Thus both epoch60 rows have the same number of updates but different LR exposure, augmentation and seeds. The higher validation accuracy cannot be attributed to duration or augmentation alone.

Sources: original results/trajectories.csv in the sibling extension_retry01 folder; newer results/histories.csv in extension_retry01/follow_up/2026-11-16_scale_equivariance. No weights or results were changed and no new test evaluation was performed for this comparison.
