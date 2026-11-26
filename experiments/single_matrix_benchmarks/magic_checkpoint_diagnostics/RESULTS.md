# MAGIC checkpoint diagnostic audit: complete

## Finding

The original development experiment does not show SGDM failing to optimize while AdamW succeeds. At the prescribed 20,000-step endpoint, low-LR SGDM has the best mean validation accuracy and CE of the 12 configurations, and it beats every AdamW endpoint within each of the two matched seeds. High-LR SGDM and AdamW both fit the training set much more closely while producing increasingly confident validation errors. Cross-fitted temperature scaling removes much of the excess validation CE, but it does not erase the remaining deterioration in prediction quality.

The spectral evidence does not establish an independent RG fixed point. Ordinary alpha near two is compatible with poor validation CE; displacement alpha is not universally near two; small final spectral drift or close ECS/PL ranks are not sufficient for good validation performance. Exact compensated ReLU gauge invariance holds. Neither PL nor ECS truncation substantially repairs the high-CE endpoints.

This is the completed requested retrospective diagnostic audit. It identifies confidence inflation and residual generalization deterioration in the archived runs. It does not uniquely identify a causal optimizer/RG mechanism.

## 1. Completion, provenance, and the software failure

- Original failed workflow: [45841037613](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45841037613), job [139982420008](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45841037613/job/139982420008).
- Analysis-only repair: [37bf3cf4562237c2d336bc1c02dfc4bc5c414f3b](https://github.com/CalculatedContent/rg_optimizers/commit/37bf3cf4562237c2d336bc1c02dfc4bc5c414f3b).
- Repaired analysis launch: [2c41bb327e46c966f1c37e27579469b5dcdbdce0](https://github.com/CalculatedContent/rg_optimizers/commit/2c41bb327e46c966f1c37e27579469b5dcdbdce0).
- Successful workflow: [45846514114](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45846514114), job [140001482779](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45846514114/job/140001482779). Every step succeeded; completed 2026-11-26 15:13:16 UTC.
- Immutable original archive: [09dbaa4e095ff21fc8b1e130798d247b75c15913](https://github.com/CalculatedContent/rg_optimizers/tree/09dbaa4e095ff21fc8b1e130798d247b75c15913).
- Permanent complete diagnostic results: [a7262f96f1a64ffffcc45b921884bc4284fd9b7f](https://github.com/CalculatedContent/rg_optimizers/commit/a7262f96f1a64ffffcc45b921884bc4284fd9b7f).

The [permanent summary](https://github.com/CalculatedContent/rg_optimizers/blob/a7262f96f1a64ffffcc45b921884bc4284fd9b7f/experiments/single_matrix_benchmarks/magic_checkpoint_diagnostics/results/summary.json) records **24 runs, 144 checkpoint records, zero new neural-training updates, zero test evaluations, zero analysis errors, and zero failed final displacement fits**. All 24 JSON files, 144 trajectory rows, 24 endpoint rows, and case-summary means agree in the independent saved-record audit. The 144 checkpoint files/records contain **122 distinct weight states and expected checkpoint hashes** because every configuration shares its seed-matched initialization.

The executed diagnostic checks the prepared-data SHA-256, plan and manifest, each checkpoint file SHA-256, embedded and spectral weight hashes, all final resume-file hashes, and restored weights after the inference interventions. Independently, every file/tree SHA in the original `magic` directory is identical between the source archive and the result commit. The [archive comparison](https://github.com/CalculatedContent/rg_optimizers/compare/09dbaa4e095ff21fc8b1e130798d247b75c15913...a7262f96f1a64ffffcc45b921884bc4284fd9b7f) confines changes to the diagnostic folder and its workflow. The extended grid and 40k continuation were not modified or launched by this audit.

### Corrected failure diagnosis

The original exception combined CE, accuracy and AUC checks but printed only the CE values. The previously reported CE differences of about 1e-8 to 1e-7 were already below the existing 3e-6 tolerance. The actual mismatch was between **AUC computed from native float32 softmax probabilities in the archive** and **AUC computed from logit margins in the diagnostic**. Saturated or rounded probabilities create ties, erasing ranking distinctions that the logit margins retain.

The repaired evaluator replays the original CE, accuracy and probability-AUC computation for fidelity checks while preserving logit AUC and all other diagnostic definitions. No tolerance values, scientific settings, checkpoints, folds, fit bounds, optimizer hyperparameters or neural weights were changed. The repair also restores the original evaluator's deterministic-algorithm setting.

The old AUC predicate fails on **53 validation checkpoint rows in exactly the original 18 failed runs**. Their first failing steps match all 18 original errors. The maximum logit-versus-archived-probability AUC difference is **0.00477656**. Replayed validation errors are at most CE **2.3842e-7**, accuracy **0**, AUC **4.5827e-7**, within the original thresholds. This confirms the failure mechanism instead of merely suppressing the assertion. Small last-digit changes versus the six partial records are numerical; metric definitions are preserved.

## 2. Endpoint comparison and trajectories

Every table below uses the two development seeds, 3470 and 3471. Means are descriptive; the same validation split is reused. CE is per example, in nats. SGDM has zero weight decay throughout. The primary analyzed matrix is FC2 in a 10→784→784→2 ReLU MLP; this is not a model with only one trained matrix.

| Case / settings | Train CE | Validation accuracy | Raw validation CE | Cross-fitted CE | Mean T | Ordinary alpha |
| --- | --- | --- | --- | --- | --- | --- |
| s0: SGDM 0.003, momentum 0.9 | 0.191260 | 86.965% | 0.340601 | 0.321693 | 1.452 | 4.4407 |
| s1: SGDM 0.01, momentum 0.9 | 0.069533 | 86.084% | 0.520497 | 0.347860 | 2.874 | 2.6151 |
| s2: SGDM 0.03, momentum 0.9 | 0.004072 | 85.624% | 0.994796 | 0.358118 | 6.243 | 2.0179 |
| s3: SGDM 0.1, momentum 0.9 | 0.001161 | 85.861% | 1.237912 | 0.349897 | 7.881 | 1.7359 |
| s4: SGDM 0.03, momentum 0.8 | 0.035964 | 85.177% | 0.689905 | 0.359161 | 4.145 | 2.3252 |
| s5: SGDM 0.03, momentum 0.95 | 0.001116 | 85.848% | 1.229951 | 0.349509 | 7.850 | 1.8520 |
| a0: AdamW 0.0001, WD 0 | 0.075776 | 85.690% | 0.511256 | 0.358679 | 2.806 | 2.0272 |
| a1: AdamW 0.0003, WD 0 | 0.009213 | 84.836% | 1.139945 | 0.364809 | 7.364 | 1.8119 |
| a2: AdamW 0.001, WD 0 | 0.000679 | 85.585% | 1.425980 | 0.356602 | 9.103 | 1.8531 |
| a3: AdamW 0.0001, WD 0.001 | 0.076003 | 85.611% | 0.514114 | 0.359093 | 2.822 | 2.0236 |
| a4: AdamW 0.0003, WD 0.001 | 0.005193 | 85.414% | 1.113842 | 0.366202 | 7.057 | 2.0153 |
| a5: AdamW 0.001, WD 0.001 | 0.000869 | 85.348% | 1.431455 | 0.359513 | 9.086 | 2.0125 |

At 20k, s0 exceeds a0 by **1.2746 percentage points** in mean accuracy and has raw CE lower by **0.17066**. Its accuracy advantage is **1.9711 points** for seed 3470 and **0.5782 points** for seed 3471. These are paired observations, not a population-level significance claim.

The saved trajectories distinguish fitting the training objective from generalization:

- **s0, SGDM .003:** training CE falls .30561→.19126 from 2k→20k, while validation accuracy improves 85.900%→86.965%. Raw validation CE reaches its best saved mean at 5k (.32397), then rises to .34060. Cross-fitted CE reaches .31738 at 15k and .32169 at 20k.
- **s2, SGDM .03:** training CE falls .24419→.00407, but validation CE rises .33099→.99480. Cross-fitted CE also rises .31855→.35812. Accuracy peaks at 86.938% at 5k and is 85.624% at 20k.
- **s3, SGDM .1:** training CE falls .21893→.00116, while validation CE rises .36216→1.23791. The cross-fitted change is .32883→.34990; raw CE inflation is much larger than the residual calibrated deterioration.
- **a0, AdamW .0001:** training CE falls .26362→.07578, raw validation CE rises .31838→.51126, and cross-fitted CE rises .31559→.35868. Accuracy peaks at 86.781% at 5k and is 85.690% at 20k.
- **a2, AdamW .001:** training CE falls .18632→.00068, raw validation CE rises .36983→1.42598, and cross-fitted CE rises .32930→.35660.

All eleven cases other than s0 achieve their best saved mean raw validation CE at 2k. This is retrospective trajectory evidence. It does not replace the original fixed-20k selection rule or establish a new selected model. In particular, the best saved mean accuracy anywhere is a5 at 2k (87.175%); the best fixed-20k endpoint is s0. These are different comparisons.

## 3. Signed margins, confidence, and CE decomposition

For binary logit gap z and label y, signed margin is m=(2y−1)z and CE is log(1+exp(−m)). Incorrect predictions have nonpositive margins, with larger negative magnitude yielding larger CE. The contribution column divides the total loss on incorrect examples by the count of **all** validation examples; conditional CE divides only by the incorrect count. The final percentage is the ratio of mean contributions to mean total CE, not a pooled quantile or a mean of seed-level percentages.

| Case | Mean signed margin on errors | Mean confidence on errors | Conditional CE on errors | Errors' CE contribution | Errors' share of total CE |
| --- | --- | --- | --- | --- | --- |
| s0 | -1.7402 | 78.05% | 2.0083 | 0.2618 | 76.86% |
| s1 | -3.2373 | 84.54% | 3.4254 | 0.4759 | 91.43% |
| s2 | -6.6612 | 90.80% | 6.7714 | 0.9734 | 97.85% |
| s3 | -8.5694 | 93.33% | 8.6484 | 1.2228 | 98.78% |
| s4 | -4.2568 | 86.45% | 4.4209 | 0.6549 | 94.92% |
| s5 | -8.4916 | 92.53% | 8.5811 | 1.2144 | 98.74% |
| a0 | -2.9675 | 82.27% | 3.1847 | 0.4553 | 89.05% |
| a1 | -7.2910 | 91.04% | 7.3981 | 1.1207 | 98.31% |
| a2 | -9.7135 | 93.66% | 9.7893 | 1.4116 | 98.99% |
| a3 | -2.9836 | 82.32% | 3.1998 | 0.4600 | 89.48% |
| a4 | -7.3945 | 91.50% | 7.4964 | 1.0933 | 98.16% |
| a5 | -9.5942 | 93.27% | 9.6746 | 1.4174 | 99.02% |

At s0, errors average about **−1.74** signed margin and **78.05%** confidence. At s3 these become **−8.57** and **93.33%**, with errors accounting for **98.78%** of CE. At a2 they become **−9.71** and **93.66%**, accounting for about **98.99%** of CE. The median error margins are −1.391/−1.296 for s0, −5.597/−6.000 for s3, and −6.247/−6.599 for a2 (seeds 3470/3471). The 1st-percentile error margins reach approximately −40 for s3 and −45 to −48 for a2. Full signed-margin quantiles are in the per-run JSON files.

### Cross-fitted temperature calibration

The same two duplicate-grouped validation folds are used throughout. A positive scalar temperature is fit on one half and evaluated on the other; the halves are then swapped and held-out CE is pooled. Neural parameters stay unchanged. This is statistical fitting of a calibration scalar, not neural training and not an independent test evaluation.

Raw→cross-fitted endpoint CE is **.34060→.32169** for s0, **.99480→.35812** for s2, **1.23791→.34990** for s3, **.51126→.35868** for a0, and **1.42598→.35660** for a2. The reduction is roughly 5.55%, 64.0%, 71.7%, 29.8%, and 75.0%, respectively (ratio of means). This is direct evidence that confidence rescaling accounts for much of the large raw-CE differences.

It is not the entire explanation: calibrated endpoint CE for these stronger-fitting regimes remains worse than s0, and often worse than their own early checkpoints. Positive temperatures preserve predicted classes; calibration cannot fix wrong signs. Because the two folds use different temperatures, the code appropriately reports cross-fitted CE rather than asserting preservation of pooled cross-fold AUC rankings. The temperature bounds are exp(−5) to exp(5); initial checkpoints can hit the upper bound and should not be interpreted as informative physical temperatures.

## 4. Initialization displacement, spectra, and fit quality

Relative displacement and cosine refer to FC2 and its own seed-matched W0. Spectral drift is the mean absolute difference of log sorted eigenvalues normalized by their respective full-spectrum means. The endpoint value compares **15k with 20k**, not initialization with 20k.

| Case | Relative displacement | Cosine with initialization | Stable rank | Top-10 energy | PL / ECS ranks | ECS minus PL | 15k to 20k drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s0 | 0.3231 | 0.9529 | 26.653 | 10.71% | 93.5 / 551.5 | 458 | 0.03221 |
| s1 | 0.7262 | 0.8217 | 10.642 | 23.97% | 30 / 475 | 445 | 0.09178 |
| s2 | 1.4097 | 0.6156 | 16.909 | 32.05% | 273 / 353 | 80 | 0.06750 |
| s3 | 2.4170 | 0.4251 | 13.710 | 35.89% | 323 / 277.5 | -45.5 | 0.13064 |
| s4 | 0.9727 | 0.7397 | 11.658 | 28.30% | 222 / 424.5 | 202.5 | 0.11374 |
| s5 | 1.8656 | 0.5130 | 16.427 | 33.97% | 301.5 / 305.5 | 4 | 0.02587 |
| a0 | 1.6594 | 0.5396 | 5.361 | 40.41% | 237.5 / 326 | 88.5 | 0.16233 |
| a1 | 3.4760 | 0.3089 | 6.782 | 40.96% | 159.5 / 279 | 119.5 | 0.09094 |
| a2 | 7.1042 | 0.1613 | 10.085 | 37.64% | 123 / 280 | 157 | 0.07698 |
| a3 | 1.6608 | 0.5386 | 5.355 | 40.32% | 236.5 / 325.5 | 89 | 0.16379 |
| a4 | 3.4520 | 0.3094 | 6.627 | 41.21% | 126 / 278.5 | 152.5 | 0.08936 |
| a5 | 7.1615 | 0.1574 | 10.119 | 37.74% | 85.5 / 281.5 | 196 | 0.08075 |

s0 remains strongly aligned with initialization (cosine .953, displacement .323); AdamW endpoints move substantially farther. Its full-weight alpha 4.44 and displacement alpha near 2 are consistent with sensitivity to the initialization background. They do not prove a simple additive masking mechanism: WᵀW contains both initialization/displacement spectra and their cross terms.

All final displacement fits returned algorithmic success, but their selected tails differ markedly:

| Case | Delta alpha (seed 3470 / 3471) | Tail counts | KS D |
| --- | --- | --- | --- |
| s0 | 2.0343 / 2.0905 | 27 / 23 | 0.04004 / 0.04594 |
| s1 | 2.1187 / 2.2976 | 32 / 20 | 0.05540 / 0.04939 |
| s2 | 1.4843 / 1.5232 | 219 / 182 | 0.06344 / 0.06542 |
| s3 | 2.2971 / 1.4332 | 33 / 311 | 0.07609 / 0.07631 |
| s4 | 2.3600 / 2.3662 | 22 / 20 | 0.06173 / 0.05074 |
| s5 | 1.4747 / 1.4533 | 237 / 268 | 0.06816 / 0.07041 |
| a0 | 2.2699 / 2.2623 | 20 / 31 | 0.05461 / 0.03915 |
| a1 | 2.2132 / 2.6309 | 37 / 13 | 0.06300 / 0.06668 |
| a2 | 2.0869 / 2.3440 | 60 / 32 | 0.07019 / 0.07297 |
| a3 | 2.2369 / 2.3235 | 31 / 18 | 0.05260 / 0.04522 |
| a4 | 2.2233 / 2.3098 | 40 / 30 | 0.05005 / 0.06943 |
| a5 | 2.0710 / 2.1468 | 64 / 55 | 0.07232 / 0.06403 |

The complete results reject a universal displacement-alpha≈2 interpretation. s2 has alpha 1.484/1.523 and s5 1.475/1.453. For s3, the two fits select alpha **2.297 versus 1.433** and tail counts **33 versus 311**; their mean hides a large cutoff difference. Tail counts across all displacement fits range from **13 to 311**, and KS D is not a formal power-law goodness-of-fit p-value.

The original ordinary fits have no recorded failures across 144 checkpoints. Thirty early/initial records have alpha>6; these are the original fitting protocol, not a fix-fingers rerun. No endpoint alpha exceeds six. The repeated initialization spectra are not independent observations.

### Contrary evidence for simple spectral explanations

Lower final spectral drift is not sufficient for better validation CE: s5 drift .02587 is below s0 .03221, yet raw CE is 1.22995 versus .34060 and calibrated CE .34951 versus .32169.

Increasing top-mode concentration is not a universal mechanism of worsening CE. In high-LR AdamW a2, mean top-10 energy **falls from 49.72% at 2k to 37.64% at 20k**, while stable rank rises **7.43→10.08** and validation CE deteriorates. SGDM and AdamW require separate trajectory descriptions. The drift statistic is also sensitive to very small eigenvalues, and the initial checkpoint intervals are unequal.

## 5. Layer-scale gradients, exact ReLU gauge, and next-update geometry

The compensated architectural transformation is (W_l,b_l,W_(l+1))→(cW_l,cb_l,W_(l+1)/c), c>0. Its gradient identity is <W_l,g_l>+<b_l,g_b>−<W_(l+1),g_(l+1)>=0. This does not require any individual layer radial gradient to vanish.

At every one of the 24 endpoints, c=.5 and c=2 preserve the **binary logit gap exactly**, with maximum recorded change zero. Across all 144 checkpoints, the largest compensated gradient-identity residual is **3.76568e-8** (FC1); FC2's maximum is **1.77033e-8**. The factors are exact binary scales. The field called `max_abs_logit_difference` compares the binary gap, not both output coordinates individually.

Every endpoint has positive FC2 −<W,g>, ranging .002183–.066493. A plain FC2 weight rescaling therefore is not a flat direction. The uncompensated intervention keeps the bias fixed and can change ReLU gates; it is not equivalent to temperature scaling.

The hypothetical update uses the archived optimizer state and reconstructs the next minibatch from saved permutation/offset/RNG. SGDM combines saved momentum with the next clipped gradient; AdamW uses its saved moments, bias corrections and decoupled decay. **No update is applied.** The decomposition is rho=<W,u>/||W||² and u_perp=u−rho W.

| Case | FC2 -<W,g> | FC2 update / weight norm | FC2 fractional radial update | Cosine with negative next-batch gradient |
| --- | --- | --- | --- | --- |
| s0 | 0.035526 | 0.00030349 | 0.00000345 | 0.64083 |
| s1 | 0.044217 | 0.00051907 | 0.00001112 | 0.30356 |
| s2 | 0.009922 | 0.00033434 | 0.00000534 | 0.52430 |
| s3 | 0.004376 | 0.00018307 | 0.00000276 | 0.02740 |
| s4 | 0.034328 | 0.00088412 | 0.00001236 | 0.67282 |
| s5 | 0.005744 | 0.00013857 | 0.00000284 | 0.15992 |
| a0 | 0.060162 | 0.00046342 | 0.00002320 | 0.31716 |
| a1 | 0.014352 | 0.00066960 | 0.00001565 | 0.42272 |
| a2 | 0.002385 | 0.00008995 | 0.00000226 | 0.39903 |
| a3 | 0.059771 | 0.00045389 | 0.00002274 | 0.31184 |
| a4 | 0.012457 | 0.00045489 | 0.00001443 | 0.46863 |
| a5 | 0.002731 | 0.00021813 | 0.00000389 | 0.30239 |

The FC2 shape component accounts for at least **99.6286% of hypothetical update squared norm** at every endpoint. This is an orthogonal decomposition in parameter coordinates, not a complete gauge-invariant functional decomposition. FC3 has a larger radial component in some configurations.

The next-update/negative-gradient cosine is negative for s3 seed 3471 (−.07817) and s5 seed 3471 (−.02242), illustrating saved momentum's contribution. One next minibatch is insufficient to diagnose long-run instability or establish optimizer equivariance. Gradient quantities are evaluated on one fixed 1,024-example training subset; the analytic update uses the actual next minibatch. Double-precision norm accumulation also means this is an analytic floating-point reconstruction, not a bitwise test of an executed optimizer step.

These checks show that the architectural gauge identity holds. They do not establish that either optimizer's dynamics respects the compensated transformation, or that optimizer symmetry breaking caused the generalization differences. `beta_weight_only` is a local loss-gradient quantity, not an independently demonstrated RG beta function.

## 6. PL versus ECS inference interventions

Only FC2 is replaced by its rank-k truncated SVD; other layers and biases are retained. No fine-tuning, rescaling, neural update, or test inference is performed. PL k is the archived WW fit tail count. ECS k minimizes the absolute retained mean log(lambda/full-spectrum mean), over k>=8.

| Case | Original accuracy | PL accuracy | ECS accuracy | Original CE | PL CE | ECS CE |
| --- | --- | --- | --- | --- | --- | --- |
| s0 | 86.965% | 86.597% | 86.912% | 0.340601 | 0.351713 | 0.340735 |
| s1 | 86.084% | 84.625% | 86.097% | 0.520497 | 0.591174 | 0.519976 |
| s2 | 85.624% | 85.598% | 85.664% | 0.994796 | 0.992267 | 0.994015 |
| s3 | 85.861% | 85.782% | 85.848% | 1.237912 | 1.236827 | 1.235921 |
| s4 | 85.177% | 84.993% | 85.217% | 0.689905 | 0.694450 | 0.691932 |
| s5 | 85.848% | 85.887% | 85.861% | 1.229951 | 1.229295 | 1.229285 |
| a0 | 85.690% | 85.230% | 85.414% | 0.511256 | 0.514974 | 0.514746 |
| a1 | 84.836% | 85.059% | 84.915% | 1.139945 | 1.131119 | 1.136257 |
| a2 | 85.585% | 85.283% | 85.493% | 1.425980 | 1.434325 | 1.427078 |
| a3 | 85.611% | 85.335% | 85.466% | 0.514114 | 0.517782 | 0.516467 |
| a4 | 85.414% | 79.566% | 85.414% | 1.113842 | 1.559683 | 1.111396 |
| a5 | 85.348% | 85.664% | 85.243% | 1.431455 | 1.467507 | 1.431707 |

ECS generally preserves the original predictions closely, but it usually retains more modes. This is not a rank-matched comparison, so better preservation does not isolate a causal advantage of its trace-log criterion.

Neither truncation substantially repairs the high raw CE of s2/s3/s5 or high-rate AdamW. This is contrary evidence to the simple explanation that removing discarded small modes would directly solve the confidence problem.

The largest PL failure is **a4, seed 3470**: PL rank 35 drops accuracy **85.4665%→73.8765%** and raises CE **1.13299→2.02887**. ECS rank 279 preserves accuracy **85.4402%** and CE **1.13372**. The other seed uses PL rank 217 and is largely preserved. The case-mean ordinary alpha 2.015 hides seed values 2.286 and 1.744 and this cutoff variability.

## 7. What these data do and do not establish about RG

Ordinary alpha is a tail-fit exponent. It is not by itself an independent RG fixed-point test. Here, s2 has ordinary alpha 2.0179 but retained PL mean log(lambda/full mean) .26813 and an ECS−PL rank gap of 80. a0 has alpha 2.0272, retained mean log .32463 and gap 88.5. The normalizations and cutoff criteria differ, so alpha≈2 does not imply this trace-log condition.

Conversely, s5 has a small rank gap (4), PL mean log .01577 and relatively small final spectral drift, yet poor raw validation CE. ECS's near-zero retained trace-log is largely its selection criterion, not independent evidence that training found a fixed point. A single exponent, cutoff agreement, or a late interval of spectral stability cannot establish invariance under a separately specified RG transformation.

The defensible conclusions are: the archived higher-rate and later checkpoints exhibit confidence inflation and residual validation deterioration; initial-reference and spectral-tail choices materially affect alpha; compensated architectural symmetry holds; and the specified inference SVD interventions mostly preserve rather than repair the endpoint probability behavior. Associations between these spectral observations and generalization do not uniquely identify the optimizer's causal mechanism.

## 8. Limitations and reproduction

- Two development seeds, one grouped validation split and one architecture; no independent test evaluations were added. Retrospective comparisons over many settings/checkpoints are exploratory.
- Stored checkpoints are sparse. Best saved checkpoints are not guaranteed global optima, and this audit does not change model selection.
- No extra training, continuation, architecture change, revised learning rate, or alternative spectral-fit parameter was used. The extended grid and 40k continuation remain outside this audit.
- Algorithmic WW fit success is not proof of a power law. Small tails and the large s3 seed discrepancy limit interpretation. Failed final delta fits: **0/24**; failed original ordinary fits: **0/144**.
- Gauge invariance of the network/loss is distinct from optimizer equivariance; one analytic next step does not establish a causal training mechanism.
- Temperature calibration is inference/post-processing with fitted scalar parameters; truncated SVD changes the inference function. Neither is neural training, and no held-out test results are claimed.

Primary permanent evidence: [summary](https://github.com/CalculatedContent/rg_optimizers/blob/a7262f96f1a64ffffcc45b921884bc4284fd9b7f/experiments/single_matrix_benchmarks/magic_checkpoint_diagnostics/results/summary.json), [endpoints](https://github.com/CalculatedContent/rg_optimizers/blob/a7262f96f1a64ffffcc45b921884bc4284fd9b7f/experiments/single_matrix_benchmarks/magic_checkpoint_diagnostics/results/endpoints.csv), [trajectory](https://github.com/CalculatedContent/rg_optimizers/blob/a7262f96f1a64ffffcc45b921884bc4284fd9b7f/experiments/single_matrix_benchmarks/magic_checkpoint_diagnostics/results/trajectory.csv), [per-run JSON directory](https://github.com/CalculatedContent/rg_optimizers/tree/a7262f96f1a64ffffcc45b921884bc4284fd9b7f/experiments/single_matrix_benchmarks/magic_checkpoint_diagnostics/results), [original plan](https://github.com/CalculatedContent/rg_optimizers/blob/09dbaa4e095ff21fc8b1e130798d247b75c15913/experiments/single_matrix_benchmarks/magic/plan.json), [original evaluator](https://github.com/CalculatedContent/rg_optimizers/blob/09dbaa4e095ff21fc8b1e130798d247b75c15913/experiments/single_matrix_benchmarks/magic/run.py), and [executed diagnostic source](https://github.com/CalculatedContent/rg_optimizers/blob/2c41bb327e46c966f1c37e27579469b5dcdbdce0/experiments/single_matrix_benchmarks/magic_checkpoint_diagnostics/diagnose.py).

`diagnose.py` reproduces the checkpoint inference and analytic diagnostics with the pinned original requirements. To independently verify the saved JSON/CSV consistency and regenerate the derived tables, from a checkout with the original MAGIC grid records run:

```bash
python experiments/single_matrix_benchmarks/magic_checkpoint_diagnostics/audit_saved_results.py
```

This stdlib-only audit does not load a neural framework or evaluate a model. It produces `results/audit_verification.json`, `results/case_means.csv`, and `results/trajectory_means.csv`. It checks every saved scalar field, case mean, source weight-hash association and the exact original failure-step pattern. Binary checkpoint verification is performed by the successful diagnostic workflow, not by this saved-record script.
