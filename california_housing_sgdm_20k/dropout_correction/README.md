# Dropout correction: reproducible California Housing spectral study

**Status:** completed post-training audits; future multi-probability validation prepared but not launched. No new training was performed for the correction audits. All measured models are fixed 20,000-update SGDM checkpoints.

## Equation and interpretation

Let training dropout be $p$, keep probability $q=1-p$, and let $X=W^T W$. For independent input-neuron dropout with a Bernoulli diagonal mask $M$ and inverted-dropout scaling, the normalized expected Gram operator is
$$
\mathcal D_p(X)=(1-p)X+p\operatorname{diag}(X).
$$
It leaves diagonal entries intact and attenuates off-diagonal correlations by $q$. Its composition law is $\mathcal D_{p_1}\mathcal D_{p_2}=\mathcal D_{p_1+p_2-p_1p_2}$.

Our candidate diagnostic is **elementwise zero masking** with a fixed probability:
$$
\boxed{r(p)=1-\sqrt{1-p}},\qquad \widetilde W_{ij}=W_{ij}B_{ij},\quad B_{ij}\sim\mathrm{Bernoulli}(\sqrt{1-p}).
$$
Fit the ordinary power-law exponent to the positive squared singular values of each masked copy and average across masks within each model, then equally across independently trained seeds. Do not fit $r$, tune to 2, mutate the model, or apply this transformation to FC1 (no input dropout). FC1 has eight singular values and uses the shipped WeightWatcher small-N fit with `min_evals=8`; FC3 is not power-law measurable.

The square root follows exactly for **independent variance-preserving replacement** with centered, independent replacements whose column second moments equal those of the original weights: its off-diagonal expected Gram terms are multiplied by $(1-r)^2=q$, and diagonal expectations are preserved. This motivates, but does not prove, the same $r$ for elementwise *zero masking*. Zero masking instead has normalized expected Gram operator $\mathcal D_r$, not $\mathcal D_p$. Neither method is an exact inverse of training with activation dropout. The measured exponent depends on the full distribution, not just the expected Gram matrix.

## Measured results (not interpolated)

Original training dropout $p=0.3690073279296028$; fixed candidate $r=0.20564952818645776$. Seeds 3340–3344, five immutable 20k checkpoints; 20 independent diagnostic masks/noise realizations per seed in the final formula audit (100 total per stochastic method). SD is the sample SD of the five seed means, not 100 independent models.

| Method | Mean alpha | SD across seeds | Mean KS D | Mean tail size |
|---|---:|---:|---:|---:|
| Original FC2 | 1.690720 | 0.004079 | 0.05935 | 86.00 |
| Square-root entry masking | **2.002959** | 0.006731 | 0.02396 | 230.55 |
| Full-p entry masking | 2.126283 | 0.011486 | 0.01469 | 153.98 |
| Full-p expected Gram | 2.123473 | 0.149918 | 0.04308 | 216.60 |
| Fractional expected Gram | 2.151270 | 0.015535 | 0.04628 | 626.80 |
| Square-root variance-preserving replacement | 2.212362 | 0.006113 | 0.02519 | 198.69 |
| Full-p variance-matched additive noise | 2.261948 | 0.005734 | 0.03148 | 209.91 |
| Actual input-neuron masking | 1.746680 | 0.040993 | 0.05515 | 152.24 |
| Actual output-neuron masking | 1.746458 | 0.032353 | 0.05462 | 153.05 |
| Two-sided neuron masking | 1.784391 | 0.013404 | 0.04841 | 196.65 |

Square-root entry masking seed means: 2.001455, 1.995453, 2.003479, 2.000636, 2.013773. FC1 ordinary small-N mean alpha = 2.340561, uncorrected. Test RMSE of the original optimized SGDM checkpoint cohort = 0.49205 (units of $100,000); diagnostic masking is not applied to inference or performance evaluation.

Earlier exploratory sweep: 20% zero masking gave alpha 1.997840; 9.4% isotropic correction gave 2.001766 but reselected almost the entire spectrum (mean tail 759.2 versus original 86). Those strengths were selected after seeing alpha and are not independent evidence. The final formula was motivated by that same cohort, so fixing the expression before the second analysis is not independent training-data validation.

**Negative controls:** square-root variance-preserving replacement, which has the exact expected-Gram derivation, did not yield 2. Actual neuron dropout did not yield 2 either. In 30 paired development endpoints with valid archived spectra, uncorrected alpha's mean absolute error against the no-dropout control was 0.43824, versus 0.47436 for the isotropic half-channel correction and 0.51824 for the isotropic full-p correction. In 12 controls within 0.25 of 2 the corresponding errors were 0.15940, 0.24470, 0.31001. These are spectrum-only alternatives, not tests of elementwise masking. Four failed fits are preserved in the original grid and are not converted to alpha values. Only one nonzero training probability was measured. **The square-root zero-mask rule remains an empirical candidate, not a validated universal dropout correction.**

## Reproduce the completed analysis

From the repository root on the audit branch (or a checkout of this commit), use Python 3.12.9. The reproduction scripts are extracted from the successful workflow steps with only their output directories changed to avoid overwriting the committed archives.

```bash
python -m pip install -r california_housing_benchmark/requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python california_housing_sgdm_20k/dropout_correction/reproduce_scan.py
python california_housing_sgdm_20k/dropout_correction/reproduce_formula.py
```

The scripts verify the 20k checkpoint identities and reproduce the original ordinary WeightWatcher fit before transformation. The main FC2 fit is a numerical implementation of the same ordinary KS-minimizing Pareto MLE; it matched all five stored alpha, D, and tail sizes. FC1 calls WeightWatcher itself. The source and dependencies are pinned; different numerical libraries/hardware may vary slightly. The scripts use existing committed weights only; never retrain missing checkpoints.

- [Original SGDM 20k archive](../results/2026-11-26_sgdm_20k/)
- [Exploratory scan summary](../dropout_spectral_audit/summary.json), [raw fits](../dropout_spectral_audit/raw_fits.csv), [workflow 45781671831](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45781671831)
- [Fixed formula summary](../dropout_formula_audit/summary.json), [confirmation fits](../dropout_formula_audit/confirmation_fits.csv), [development spectrum controls](../dropout_formula_audit/development_spectrum_comparison.csv), [workflow 45809094083](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45809094083)
- Immutable completed audit commit: `1e2ec0a2c8d246154f46bbf8a6e38f75440ca1a3`; original model archive: `95cf2d49400c258588398383a9a041e5603e5837`.

## Next experiment (not launched)

See [future_plan.json](future_plan.json) and [run_future.py](run_future.py). We propose one fixed SGDM setting (LR .01, momentum .99, WD .001, from the prior no-dropout validation winner, with only two previous calibration seeds) and five dropout probabilities 0, .1, .2, .3690073279296028, .5, with five fresh paired seeds. That gives 25 new runs at exactly 20k updates, only if separately authorized. No hyperparameter retuning or AdamW runs; no reopening or resuming superseded cohorts. Maximum five CPU workers, 45-minute timeout per job if later scheduled; the future runner itself is a local single-run launcher and does not enforce an external wall clock timeout. No Actions workflow is installed for this future training plan.

The default command only prints the frozen matrix of runs:
```bash
python california_housing_sgdm_20k/dropout_correction/run_future.py
```

Do not pass the explicit execution flag until the study is authorized. The prepared analyzer compares uncorrected, full-p mask and square-root mask exponents against the same-seed no-dropout control, not a forced target of 2. It reports all rates and failures, test RMSE/MAE at the matching 20k endpoint, and paired errors. The analyzer requires all 25 checkpoints; no training is triggered by it. These newly prepared scripts have not been executed on fresh models.

```bash
python california_housing_sgdm_20k/dropout_correction/analyze_future.py
```
