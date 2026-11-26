# Interpretation: optimizer-state diagnostics on MAGIC
2026-11-27. Post hoc diagnostic, not a new training experiment.
Source: [immutable raw audit archive ab1b7a6](https://github.com/CalculatedContent/rg_optimizers/tree/ab1b7a686bb23e4edc0b0faaf4d5ee75064b6bb4/experiments/single_matrix_benchmarks/magic_scale_invariant/optimizer_step_audit), [workflow 45902105886](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45902105886). All 48 endpoint states and 288 one-step trials completed; no errors. All original model and optimizer-state hashes checked by the diagnostic. All updates on disposable copies. No test access.

## 1. Architectural symmetry versus optimizer covariance
The raw gradient of the normalized FC2 matrix is tangent to its Frobenius sphere to numerical precision, but the actual SGDM and AdamW updates are not exactly tangent. This distinguishes an instantaneous gradient identity from a full-state optimizer map. The following are means of the five selected confirmation endpoint states; one actual next minibatch reconstructed from each saved sampler state, not an average over training.

| Model | LR | FC2 angle (rad) | Mean abs gradient radial cosine | Mean abs update radial cosine | Mean global clip factor | Clipped batches |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary SGDM | .003 | .00029336 | .0023764 | .0111904 | .83532 | 3/5 |
| Ordinary AdamW | .0001 | .00044230 | .0021302 | .0457430 | .77597 | 4/5 |
| Normalized SGDM | .1 | .00053242 | 4.72e-8 | .0062210 | 1.00000 | 0/5 |
| Normalized AdamW | .0001 | .00065698 | 7.62e-9 | .0056545 | .25124 | 5/5 |

The selected normalized AdamW has about 1.23 times the local angular motion of selected normalized SGDM despite 1000x smaller raw LR. The SGDM angular values vary widely across seeds (.0000293–.0015765), so do not interpret this ratio as a precise optimizer speed estimate. At matched AdamW LR .0001 the normalized/ordinary angular ratio is 1.49 across five confirmation seeds; paired difference +.00021468 ± .00015929 rad (nominal 95% paired t interval). Clipping factor difference -.52473 ± .21435 and signed radial cosine difference -.04129 ± .00755. At the final state the normalized AdamW preconditioner p99/p01 ratio averages 209,264 (ordinary 245,078), a highly anisotropic diagonal metric. These ratios are descriptive of the preconditioner, not of the realized update spectrum.

Matched-LR development (two seeds) normalized/ordinary mean endpoint angle ratios: AdamW .0001 1.82, .0003 1.76, .001 26.34; SGDM .003 .886, .01 1.81, .03 3.58, .1 .171. The extreme .001 AdamW ratio has a small ordinary denominator. The nonmonotonic SGDM result illustrates why nominal LR and endpoint local motion cannot stand in for cumulative angular training time.

## 2. Direct radial rescaling intervention
Scale only raw FC2 by b=1/4 or 4, leaving effective model, other parameters and gains unchanged, and transform stored momentum/moments consistently. Compare delta_b/b with the unscaled delta. Compensate FC2 LR by b^2 for SGDM, b for AdamW; transform Adam epsilon by 1/b. On these power-of-two rescalings logits agree and raw gradient covariance is verified numerically.

| Optimizer | b | Unclipped, uncompensated median relative error | Unclipped, compensated median error | Clipped, compensated median error | Clipped, compensated max error |
|---|---:|---:|---:|---:|---:|
| SGDM | .25 | 15.0000 | 0 | 0 | .5003 |
| SGDM | 4 | .9375 | 0 | 0 | .7028 |
| AdamW | .25 | 3.0001 | ~0 | .2662 | .3865 |
| AdamW | 4 | .7500 | 0 | .0854 | .1275 |

The uncompensated errors match |1/b^2-1| for SGDM and approximately |1/b-1| for AdamW (epsilon held fixed). The compensated unclipped updates agree to numerical precision. Global clipping changes the factor multiplying FC2 gradients when FC2 alone is rescaled, invalidating otherwise correct state transformations. Zero SGDM median under clipping is not proof of invariance: many endpoint batches were unclipped. Restricting to trials where either reference or counterfactual clipping is active, mean relative errors are .3492 (6/13) and .5492 (4/13) for SGDM, and .2704 (11/11) and .07894 (10/11) for AdamW at b=.25 and 4 respectively. This is direct evidence of a symmetry-breaking implementation mechanism, not a proof that clipping is the cause of the observed alpha trajectories.

## 3. Mechanistic interpretation
Let g_W be the raw gradient of L(bW)=L(W), so g_{bW}=g_W/b. For momentum, scaling the velocity and LR by 1/b and b^2 gives delta_{bW}=b delta_W. For Adam, with consistently transformed state, first and second moments scale 1/b and 1/b^2. Transforming epsilon by 1/b makes the normalized update direction invariant and LR by b makes delta covariant. This is only a local single-layer transformation under the stated assumptions, not global dynamical scale invariance. The gradient clipping factor c=min(1,C/||g_all||) does not generally remain unchanged under a layerwise rescaling. It also scales the newly computed gradient relative to historical buffers. Therefore it obstructs both covariance laws. The Adam preconditioner has a different scale dimension, and its adaptive normalization can maintain meaningful angular motion at very small LR.

This establishes a concrete mechanism distinguishing the optimizers, and a concrete obstruction to both idealized flows. It does not establish why MAGIC selects a near-2 AdamW tail, that preconditioning is causal for marginality, or that AdamW has no overfitting. Prior matched results show both ordinary and normalized AdamW approach 2 from above at .0001; at higher rates they go below 2. Prior calibration analysis shows gain/logit growth contributes to overconfidence. The present one-step audit cannot resolve causal effects of clipping, gain or cumulative angular motion on alpha or generalization.

## 4. Next controlled test
The existing authorized fixed-gain comparison remains the next training intervention: freeze gains at matched initialization, keep exact data, split, seeds, architecture, LR grid, update cap and global clip rule, no WD or spectral penalty. Reuse trainable-gain and ordinary controls. Track actual angular motion and clipping incidence over training if feasible; compare alpha, full spectral shape, accuracy, CE and calibration. This isolates the gain-freezing intervention, not clipping. A later separate clip-disabled or scale-covariant clipping control and angular-motion-matched preconditioner ablation would be necessary to causally attribute the marginality effect to clipping or Adam normalization; not executed or authorized here. No training replay, test-set selection, or new datasets.

All numerical evidence is in [step_diagnostics.csv](results/step_diagnostics.csv) and [summary.json](results/summary.json). The prior paired-spectrum [AdamW investigation](../adamw_investigation/RESULTS.md) and [consolidated report](../SCALE_INVARIANCE_REPORT.md) are unchanged.
