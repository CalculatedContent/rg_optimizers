# MAGIC: fixed-gain SGDM five-seed confirmation — scientific interpretation

**Status:** Complete, 26 November 2026 (UTC). [Workflow 45910723862](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45910723862), attempt 1, authorized launch `496df6a0461f077c056c877572943172f3789461`, [permanent training archive](https://github.com/CalculatedContent/rg_optimizers/tree/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/results) `9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1`. All five jobs, gate, archive and final audit passed. No training was replayed.

## Protocol and provenance

The user-directed choice LR .03, momentum .9, fixed gains, zero weight decay/dropout, batch 256, clip 1, 10–784–784–2 MLP, 20,000 updates each, seeds 3472–3476, was made after two development seeds gave FC2 alpha 1.8775 and validation accuracy 86.87%. It was not the validation accuracy winner. The outer-train refit uses 15,216 examples and the same previously accessed 3,804-example test set, evaluated only at step 20,000. This is new training-seed evidence but not a new test population or a preregistered optimizer-wide comparison. No alpha-based filtering of confirmation results.

[Preflight](https://github.com/CalculatedContent/rg_optimizers/blob/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/results/preflight/preflight.json) verified zero training, exact 100,000-update cap, data SHA256 `c42ba2691997b884dae5d738bcb2b5bb5a5b07fbca0f0c83afae12ce76812d73`, plan SHA256 `ab5706db0f6a2416b0563e978a8530a762e1ff24c6c9433a034ee0764356d0cc`, manifest SHA256 `26cc2d30485301f72087e88a59c0b044f374b4987abe1e176a308a32e2cc2473`. FP32 maximum logit rescale error 4.17e-7 and gradient radial residual 1.92e-8; FP64 7.77e-16 and 8.10e-18. All 30 archived checkpoints and five resume files passed SHA256/state-hash checks in the workflow; all 30 records contain 784 eigenvalues. The independent archive read confirmed exact source/plan/manifest in all five JSONs, six checkpoint files and a resume file per seed, endpoint-only test, unchanged FC1/FC2/FC3 gains in each trajectory and initial full-state SHA256 equal to the matched learned-gain AdamW record in every seed. Four initial eigenvalue lists were exactly equal, the remaining list (3472) differed by at most 9.10e-15 in absolute value (max eigenvalue 1.311), consistent with numerical roundoff. This is a metadata/readback and workflow audit, not a second independent binary hash computation.

## Endpoint results

All means and 95% Student t intervals refer to five training seeds (df=4); they do not quantify test-set sampling uncertainty. Accuracy in percent, CE in natural-log units.

| Arm | Test accuracy (%) | Test CE | AUC | FC2 alpha | KS D | Tail n |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-gain SGDM | 87.608 ± .299 | .30721 ± .00406 | .92915 ± .00143 | 1.87830 ± .04596 | .04202 ± .00726 | 28.8 ± 8.8 |
| Ordinary AdamW .0001 | 86.567 ± .428 | .39951 ± .01032 | .91732 ± .00259 | 2.07151 ± .01487 | .02022 ± .00089 | 231.6 ± 6.8 |
| Learned-gain AdamW .0001 | 85.174 ± 1.865 | .48852 ± .02837 | .91028 ± .00246 | 2.07628 ± .00768 | .03212 ± .00342 | 234.4 ± 5.0 |

| Paired fixed SGDM minus control | Accuracy (pp) | CE | AUC | Alpha | D | Tail n |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary AdamW | +1.041 ± .362 | -.09230 ± .00718 | +.01183 ± .00195 | -.19321 ± .05981 | +.02180 ± .00780 | -202.8 ± 10.0 |
| Learned-gain AdamW | +2.434 ± 1.785 | -.18130 ± .02600 | +.01887 ± .00151 | -.19799 ± .05332 | +.00990 ± .00829 | -205.6 ± 10.6 |

Fixed SGDM wins on accuracy, CE and AUC for every matched seed against both archived controls. Its mean Brier score .09276 ± .00173, 15-bin ECE .02320 ± .00558, train accuracy 90.222% and train CE .24105. No post-hoc test temperature fitting; no control Brier/ECE comparison claimed. CE and AUC gains support improvements beyond confidence rescaling alone, but architecture and optimizer are jointly changed. Ordinary AdamW is actually closer to alpha=2 by absolute distance, and has far broader fitted support and lower D. The fixed SGDM alpha interval [1.8323,1.9243] excludes 2 under the seed-only t model; near marginality is an approximation, not exact convergence or equivalence.

## Spectral and optimization dynamics

| Updates | Mean raw FC2 alpha | Mean D | Mean fitted tail n | Train CE |
|---:|---:|---:|---:|---:|
| 0 | 9.9450 | .12938 | 47.0 | .70176 |
| 2,000 | 3.9442 | .09859 | 137.6 | .29173 |
| 5,000 | 3.2665 | .09243 | 175.0 | .27302 |
| 10,000 | 2.0019 | .07302 | 18.0 | .25789 |
| 15,000 | 1.9284 | .05744 | 22.6 | .24910 |
| 20,000 | 1.8783 | .04202 | 28.8 | .24105 |

All seeds cross or approach alpha=2 near 10k, then drift below it by 20k. All final tails are 23–41 of 784 eigenvalues, versus 222–240 for the controls. The sharp tail contraction between 5k and 10k means changing fit support may contribute to the apparent transition; no evidence of a fixed point from six time samples. Zero raw fit failures does not validate a power law at initialization (all five initial exponents exceed 6) or at later short tails. No fix-fingers sensitivity result is asserted; original raw fits remain intact. A separate spectral-only sensitivity analysis using archived eigenvalues would be appropriate, not training replay.

There were five clipped updates out of 100,000 (seed counts 1,0,3,0,1); endpoint sampled update angles 3.71e-4–5.69e-4 rad and radial cosines .00397–.00792. The largest relative radial gradient residual across all stored train probes and layers was 2.99e-7. Thus architectural scale symmetry is numerically well supported and clipping is rare here, but momentum updates are not strictly tangent and the raw norm can change. This is not a proof of optimizer scale equivariance or the ERG trace-log condition.

## Conclusion and limitations

The requested existence example is supported: on MAGIC, fixed-gain SGDM at .03 achieves alpha near 2 (1.878) and beats two archived AdamW baselines on matched-seed test accuracy, CE, AUC. The two-seed development alpha replicated almost exactly in five new training seeds. The stronger claim that alpha=2 uniquely optimizes generalization, or that SGDM is intrinsically superior at identical architecture, is not supported: AdamW is closer to 2, fixed SGDM tails are much shorter, and the experiment changes gain constraints as well as optimizer. Fixed gain imposes a constraint on effective matrix norms, not merely a reparameterization. No new test access beyond the authorized endpoints, no extra training or optimizer selection. No claim of statistical independence across reused test examples, no multiplicity-adjusted or population-level inference.

Permanent supporting data: [raw endpoints](https://github.com/CalculatedContent/rg_optimizers/blob/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/results/analysis/fixed_20k.csv), [controls and source hashes](https://github.com/CalculatedContent/rg_optimizers/blob/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/results/analysis/controls.csv), [paired effects](https://github.com/CalculatedContent/rg_optimizers/blob/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/results/analysis/paired_differences.csv), [statistics](https://github.com/CalculatedContent/rg_optimizers/blob/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/results/analysis/statistics.csv), [trajectory CSV](https://github.com/CalculatedContent/rg_optimizers/blob/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/results/analysis/trajectories.csv), [alpha plot](https://github.com/CalculatedContent/rg_optimizers/blob/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/results/analysis/alpha_trajectory.png), [machine-readable summary](https://github.com/CalculatedContent/rg_optimizers/blob/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/results/analysis/summary.json), [plan](https://github.com/CalculatedContent/rg_optimizers/blob/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/plan.json), [audit source](https://github.com/CalculatedContent/rg_optimizers/blob/9b1397dcf2ccf463ecb2b2fdd25a244d1e29c9d1/experiments/single_matrix_benchmarks/magic_fixed_gain_sgdm_confirm/audit.py). Control paths and original records are under `magic/results/confirm/adamw_wd0-{seed}/result.json` and `magic_scale_invariant/results/confirm/adamw_wd0-{seed}/result.json` at the same pinned archive commit. Checkpoint hashes and eigenvalues in each seed's result JSON.
