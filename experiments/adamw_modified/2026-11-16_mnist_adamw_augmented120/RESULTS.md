# Native AdamW augmented MNIST, 120 epochs

Completed 2/2 fixed 120-epoch trajectories; numerical verification: PASS.

Two independent seeds, augmentation ON only. Same model, initialization rule, minibatch order rule, augmentation recipe and learning-rate schedule as the Fashion augmented arm, using the original digit-MNIST images and split. Native AdamW only. Historical sample-normalized decoupled decay retained; no graft. Five-epoch warmup to 0.001; cosine to 0.0001 at epoch 120.

Every outcome is retained. Missing endpoints are not evidence for convergence or against it. Two seeds are exploratory: show individuals and ranges; epochs/layers are not replicates. Test metrics are computed from prespecified saved checkpoints after each120-epoch trajectory finishes; no checkpoint selection. The same test set was used historically.

Alpha rising, alpha crossing2, smaller absolute distance to2, and performance improvements are different outcomes. FC3 has only10correlated eigenvalues; both estimators remain exploratory. These finite trajectories cannot establish asymptotic or RG convergence.

## Snapshot at epochs0,20,60,120 (all121epochs in CSV)

| seed | augmentation | epoch | alpha_fc1 | alpha_fc2 | alpha_fc3_small_n | hidden_symmetric_distance | hidden_sub2_deficit | test_accuracy_pct | test_ce |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33001 | True | 0 | 6.13451 | 4.9257 | 6.00921 | 3.53011 | 0 | 10.63 | 2.30169 |
| 33001 | True | 20 | 1.68433 | 1.85372 | 2.79823 | 0.230978 | 0.230978 | 98.42 | 0.047368 |
| 33001 | True | 60 | 1.64936 | 1.80771 | 3.26446 | 0.271462 | 0.271462 | 98.88 | 0.034282 |
| 33001 | True | 120 | 1.64552 | 1.86208 | 3.58958 | 0.246201 | 0.246201 | 99.09 | 0.0296557 |
| 33002 | True | 0 | 23.6263 | 5.9849 | 5.03543 | 12.8056 | 0 | 10.51 | 2.30374 |
| 33002 | True | 20 | 1.6971 | 1.83793 | 2.86899 | 0.232482 | 0.232482 | 98.63 | 0.0411896 |
| 33002 | True | 60 | 1.65903 | 1.79673 | 3.1676 | 0.27212 | 0.27212 | 98.8 | 0.0361198 |
| 33002 | True | 120 | 1.64554 | 1.78425 | 3.57549 | 0.285106 | 0.285106 | 99.1 | 0.0304522 |

Within-run changes at 20→60, 60→120 and 20→120, fit support, weight norms, calibration, exact schedule exposure, model status and verification are in results/. Raw states and training histories are preserved in the checkpoint artifacts. There is no unaugmented control in this experiment. Training duration and learning-rate annealing co-vary, so this test cannot isolate their effects.
