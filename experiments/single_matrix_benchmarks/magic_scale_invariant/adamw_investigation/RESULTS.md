# Why does AdamW approach marginality on MAGIC?
2026-11-27 — post hoc analysis of already archived training runs; no new neural training or test evaluation.

## Evidence and method
Read the [ordinary archive](https://github.com/CalculatedContent/rg_optimizers/tree/09dbaa4e095ff21fc8b1e130798d247b75c15913/experiments/single_matrix_benchmarks/magic), [normalized archive](https://github.com/CalculatedContent/rg_optimizers/tree/e6ec5fb97bc3db0130937dd49e710399dbc1dbec/experiments/single_matrix_benchmarks/magic_scale_invariant), and [cross-fitted calibration](https://github.com/CalculatedContent/rg_optimizers/tree/c4eda173b511cebd4d3809624c3e9fbba4884b5f/experiments/single_matrix_benchmarks/magic_scale_invariant/calibration). Both architectures use the same split, dimensions, initialization protocol, seeds, constant LR, clipping, batch size and zero decay for comparisons below. For the endpoint comparison, read all ten complete AdamW confirmation result.json files (five identical seeds per architecture, LR .0001, 20k). Recompute shape statistics from the actual 784 archived Gram eigenvalues for FC2. [Per-seed table, source hashes and fixed-cutoff results](paired_spectral_endpoints.csv); [reproduction script](reproduce.py). Confidence intervals are nominal paired Student-t seed intervals (n=5, df=4), not independent test-population uncertainty or adjusted multiple comparisons.

## 1. Matched learning-rate development results
Two-seed means at 20k. O = ordinary; N = independently radially invariant with learned scalar gains.

| Optimizer, LR | O alpha | N alpha | O val accuracy % | N val accuracy % | O val CE | N val CE |
|---|---:|---:|---:|---:|---:|---:|
| AdamW .0001 | 2.0272 | 2.0515 | 85.69 | 85.34 | .5113 | .6409 |
| AdamW .0003 | 1.8119 | 1.6874 | 84.84 | 84.63 | 1.1399 | 1.8743 |
| AdamW .001 | 1.8531 | 1.8920 | 85.58 | 85.06 | 1.4260 | 1.7776 |
| SGDM .003 | 4.4407 | 8.2045 | 86.96 | 85.32 | .3406 | .6736 |
| SGDM .01 | 2.6151 | 2.9990 | 86.08 | 83.88 | .5205 | 1.6927 |
| SGDM .03 | 2.0179 | 1.9310 | 85.62 | 84.66 | .9948 | 1.9595 |
| SGDM .1 | 1.7359 | 1.6814 | 85.86 | 85.45 | 1.2379 | 2.2547 |

AdamW near-marginality occurs in both architectures at low LR; faster AdamW also overshoots. At matched LR, ordinary SGDM already approaches 2 without normalization at .03, but with worsening CE. The normalized SGDM optimum by accuracy is .1, unlike ordinary SGDM .003, so comparing their selected endpoints confounds architecture with LR. The alpha>6 fits at low LR and initialization deserve separate cutoff/finger sensitivity analysis; this investigation does not certify those fits.

## 2. AdamW confirmation trajectory
Five-seed means at LR .0001. No intermediate confirmation test access.

| Step | O alpha | N alpha | O tail | N tail | N FC2 raw norm | N FC2 effective norm | N FC2 gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.945 | 9.945 | 47.0 | 47.0 | 16.171 | 16.171 | .578 |
| 2000 | 3.404 | 3.022 | 159.0 | 139.2 | 15.933 | 18.454 | .659 |
| 5000 | 2.582 | 2.163 | 86.6 | 24.2 | 16.639 | 21.931 | .783 |
| 10000 | 2.301 | 2.157 | 58.4 | 53.2 | 17.517 | 29.928 | 1.069 |
| 15000 | 2.213 | 2.196 | 212.4 | 153.2 | 18.122 | 39.778 | 1.421 |
| 20000 | 2.072 | 2.076 | 231.6 | 234.4 | 18.612 | 50.130 | 1.790 |

Near-2 fit appears earlier in the normalized architecture, but fitted tails at 5k are much shorter and cutoff changes complicate interpreting alpha trajectories as continuous RG beta functions. Endpoint tails are substantial. The normalized raw norm increases modestly while effective scale grows about 3.1x due to the gain. This shows radial symmetry does not fix functional scale.

## 3. Is it the same spectrum?
Let p_i=lambda_i/sum_j(lambda_j) for sorted Gram eigenvalues. The spectral entropy effective rank exp(-sum p_i log p_i), participation ratio 1/sum p_i^2, and top-k fractions are invariant under uniform positive scaling. They distinguish shape even when fitted alpha agrees.

| Metric | Ordinary | Normalized | Paired N−O ± 95% CI |
|---|---:|---:|---:|
| Alpha | 2.07151 | 2.07628 | +.00477 ± .01007 |
| KS D | .02022 | .03212 | +.01190 ± .00336 |
| Tail count | 231.6 | 234.4 | +2.8 ± 11.2 |
| Tail energy fraction | .87543 | .88956 | +.01413 ± .00858 |
| Top eigenvalue energy fraction | .14795 | .15744 | +.00948 ± .00749 |
| Top ten energy fraction | .37012 | .40397 | +.03385 ± .00405 |
| Entropy effective rank | 144.64 | 125.56 | -19.08 ± 1.50 |
| Participation ratio | 32.23 | 27.76 | -4.47 ± 1.67 |
| Raw Gram trace | 895.70 | 346.41 | -549.29 ± 17.52 |
| Test accuracy (%) | 86.567 | 85.174 | -1.393 ± 1.529 pp |
| Test CE | .39951 | .48852 | +.08900 ± .02140 |
| Test AUC | .91732 | .91028 | -.00704 ± .00209 |
| Train accuracy (%) | 95.835 | 95.924 | +.089 ± .640 pp |

Sorted trace-normalized spectral L1 distance averages .07826. Equal fitted alpha does not imply equal shape: normalized AdamW concentrates more energy in leading modes, has worse fit D and lower effective rank. This is a reproducible association, not proof that concentration causes worse generalization. Neither final alpha is exactly 2: individual seed CIs at the selected cutoff exclude 2, though systematic model-fit/cutoff uncertainty is not covered by these intervals.

### Fixed-tail sensitivity (continuous Pareto MLE, not a new WW fit)
Use the largest k eigenvalues and set xmin to the smallest retained value. Alpha=1+k/sum log(lambda_i/xmin). Results (five-seed means):

| k | O alpha | N alpha | Paired N−O ± 95% CI |
|---:|---:|---:|---:|
| 78 | 2.0746 | 1.9917 | -.0829 ± .0318 |
| 157 | 2.1006 | 2.1283 | +.0277 ± .0112 |
| 235 | 2.0669 | 2.0744 | +.0075 ± .0140 |
| 314 | 1.9595 | 1.9413 | -.0182 ± .0106 |
| 392 | 1.8275 | 1.7957 | -.0318 ± .0064 |

Near-2 behavior is not exclusively a consequence of the automatic WW cutoff, but neither spectrum has a constant exponent across all cutoffs. The archived WW alpha is reproduced numerically from its selected xmin and tail. At that cutoff the mean log(lambda/(e*xmin)) is -.06664 (O) and -.07085 (N). Its proximity to zero is algebraically equivalent to alpha near 2 under the same MLE, not independent evidence of the dynamical trace-log condition. The latter requires a Jacobian or flow calculation.

## 4. Does AdamW avoid overfitting?
No, though it is less extreme than normalized SGDM .1. In confirmation the ordinary and normalized train–test gaps are 9.27 and 10.75 percentage points. In development, normalized AdamW .0001 reaches 86.87% validation accuracy at 5k (alpha 2.176, raw CE .3267, cross-fitted CE .3185), then drops to 85.34% at 20k (alpha 2.0515, raw CE .6409, calibrated CE .3639). Ordinary AdamW at 5k reaches 86.78% with alpha 2.344, CE .3289; at 20k it reaches 85.69%, alpha 2.027, CE .5113. Overtraining and calibration degradation persist even as alpha approaches 2. At 20k the normalized model has mean output gain 1.911 and fitted T 3.70. Gain growth explains a plausible part of CE inflation but not its entire cause. Earlier results are exploratory development checkpoints, not a revised selection rule or new confirmation result.

## 5. What mechanism is consistent with the data?
At zero decay AdamW is adaptive Adam. For a normalized layer write A=g sqrt(n) W/||W||. Differentiation gives grad_W L(bW)=b^{-1} grad_W L(W) with gain and all other variables fixed. Under consistent transformation of the Adam moment history, m scales b^{-1}, v scales b^{-2}, so m/(sqrt(v)+epsilon) is approximately invariant when epsilon is negligible, instead of scaling as b^{-1}. For isolated unclipped updates radial equivariance consequently needs eta→b eta (and epsilon→epsilon/b for exact covariance), versus eta→b^2 eta for plain SGD. Neither transformation is a proof of actual full-network covariance under shared LR, learned gains, biases, clipping or momentum. In particular, the preconditioned update need not be tangent even if the raw gradient is. The relevant angular step scales differently: nominal eta/r for Adam under those assumptions versus eta/r^2 for SGD. Actual steps depend on state, gradient magnitudes and clipping. AdamW's elementwise normalization could make its spectral dynamics less sensitive to the radial reparameterization, but the present experiment establishes robustness of the *endpoint exponent*, not the cause. Other hypotheses include the effective optimization time, gain/feature growth and the implicit bias of the adaptive preconditioner. Do not infer an RG fixed point or volume preservation solely from this observation.

## 6. Next decisive intervention (proposed, not run)
1. Compare matched ordinary, normalized learned-gain and normalized fixed-gain MLPs at the same LRs, seeds and update budget; include validation-tuned low-LR SGDM and AdamW separately. Fixing gains changes capacity as well as optimization. A separate gain-only regularization arm can distinguish effects.
2. On frozen checkpoints compute actual one-step optimizer update vectors, radial component <W,delta W>/(||W||||delta W||), angular step ||P_tangent(delta W)||/||W||, gain steps, clipping factors and state-derived preconditioner anisotropy. Compare rescaled copies with correctly transformed momentum buffers and Adam epsilon/moments. Test isolated-layer covariance versus full-network covariance explicitly. This is diagnostic on copies, not resumed training.
3. Compare AdamW to a matched AdamW-with-SGDM-update variant; match angular step or cumulative angular motion where feasible, not merely raw LR. Preserve exact split, report full spectra and cutoff sensitivity, fit diagnostics, validation accuracy, CE, AUC, calibration and gains. Do not select by alpha. Pre-register comparisons; obtain independent evaluation before broad generalization claims.

The defensible conclusion is that low-LR AdamW exhibits robust near-marginal spectral tails on MAGIC without architectural normalization; normalization changes the spectral concentration and calibration, not the final fitted exponent. This is an informative counterexample to a universal claim that AdamW must approach marginality from below. No new training was launched and no existing files were overwritten.
