# Native AdamW paired120 Fashion-MNIST

Completed 4/4 fixed120-epoch trajectories; numerical verification: PASS.

Two paired seeds, augmentation off/on, same initialization, order, rate schedule and update counts. Native AdamW only. Historical sample-normalized decoupled decay retained; no graft. Five-epoch warmup to0.001; cosine to0.0001 at120.

Every outcome is retained. Missing endpoints are not evidence for convergence or against it. Two seeds are exploratory: show individuals and ranges; epochs/layers are not replicates. Test metrics are computed from prespecified saved checkpoints after each120-epoch trajectory finishes; no checkpoint selection. The same test set was used historically.

Alpha rising, alpha crossing2, smaller absolute distance to2, and performance improvements are different outcomes. FC3 has only10correlated eigenvalues; both estimators remain exploratory. These finite trajectories cannot establish asymptotic or RG convergence.

## Snapshot at epochs0,20,60,120 (all121epochs in CSV)

| seed | augmentation | epoch | alpha_fc1 | alpha_fc2 | alpha_fc3_small_n | hidden_symmetric_distance | hidden_sub2_deficit | test_accuracy_pct | test_ce |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33001 | False | 0 | 6.13451 | 4.9257 | 6.00921 | 3.53011 | 0 | 9.89 | 2.30332 |
| 33001 | False | 20 | 1.64564 | 1.89462 | 2.52886 | 0.229868 | 0.229868 | 88.64 | 0.336157 |
| 33001 | False | 60 | 2.69053 | 1.93849 | 3.54661 | 0.376022 | 0.0307555 | 89.62 | 0.557153 |
| 33001 | False | 120 | 2.53205 | 1.98071 | 3.00505 | 0.275673 | 0.00964637 | 89.86 | 0.911453 |
| 33001 | True | 0 | 6.13451 | 4.9257 | 6.00921 | 3.53011 | 0 | 9.89 | 2.30332 |
| 33001 | True | 20 | 1.62194 | 1.87909 | 2.806 | 0.249482 | 0.249482 | 88.85 | 0.315873 |
| 33001 | True | 60 | 1.60737 | 2.05057 | 3.54235 | 0.2216 | 0.196314 | 90.15 | 0.286158 |
| 33001 | True | 120 | 2.70842 | 2.20029 | 4.65741 | 0.454351 | 0 | 91.25 | 0.29058 |
| 33002 | False | 0 | 23.6263 | 5.9849 | 5.03543 | 12.8056 | 0 | 10.35 | 2.30687 |
| 33002 | False | 20 | 1.65498 | 1.8615 | 2.47223 | 0.241759 | 0.241759 | 88.92 | 0.333245 |
| 33002 | False | 60 | 2.00534 | 1.93687 | 3.17361 | 0.0342357 | 0.0315658 | 90.01 | 0.559539 |
| 33002 | False | 120 | 2.09041 | 1.97169 | 2.77654 | 0.0593594 | 0.0141539 | 89.81 | 0.909599 |
| 33002 | True | 0 | 23.6263 | 5.9849 | 5.03543 | 12.8056 | 0 | 10.35 | 2.30687 |
| 33002 | True | 20 | 1.62511 | 1.86121 | 2.85717 | 0.256837 | 0.256837 | 88.74 | 0.32173 |
| 33002 | True | 60 | 2.28709 | 1.9769 | 3.16488 | 0.155096 | 0.0115498 | 89.92 | 0.322345 |
| 33002 | True | 120 | 2.27291 | 2.18637 | 3.72718 | 0.229638 | 0 | 90.74 | 0.301219 |

Detailed paired changes, support, weight norms, calibration, exact schedule exposure, model status and verification are in results/. Raw states and training histories are preserved in the checkpoint artifacts.
