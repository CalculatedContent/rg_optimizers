# Independent audit: three Shakespeare GPT optimizer cohorts

Audit date: 2026-11-24. No training was launched. Only the five GitHub Actions results for seeds 1340–1344 in each named primary cohort enter this analysis.

## Provenance and completeness

- Modified AdamW: successful run 45375423863; frozen result commit `e0440e5a42dc6ba7f19d2d170f1a0d7551a16410`.
- Native AdamW: successful run 45393623223; frozen result commit `163980048ab7381e02b97a8620b25ddefa52b613`.
- SGDM: successful run 45386899611; frozen result commit `23ba932c01d63653985c28ef9df5ef3804ee4297`.
- All 15 runs completed 20,000 updates. Each run has 41 measurements (0,500,…,20,000), with 18 unique successful matrix fits at every measurement.
- Total: 615 model measurements; 11,070 matrix fits.
- The downloaded 116 files match their pinned Git blob SHAs. `source_manifest.json` additionally records SHA-256, byte count, repository path, and source URL.
- The native AdamW result commit contains all 96 modified-experiment blobs and all 35 SGDM-experiment blobs unchanged from their respective result commits.
- `checkpoint_inventory.json` establishes 15 ZIP archives by pinned Git-tree paths, sizes, and blob SHAs. The ZIP binaries were not downloaded during this audit. This audit does not claim independently recomputed checkpoint-binary hashes.
- Historical seed 1340, historical AdamW/SGDM cohorts, and local partial workers are excluded from primary statistics.

## Estimator and uncertainty

Each model alpha uses the ordinary fitted alpha from each of 18 unique matrices with weight `sqrt(num_pl_spikes)/(D+0.02)`. The tied token/output embedding appears once. No extra matrix-size weight is used. WeightWatcher's distinct `alpha_weighted` field is not used.

Five seed model averages receive equal weights. Standard deviation is sample SD with denominator 4. The 95% confidence interval is mean ± 2.7764451052×SD/√5 (Student-t, 4 degrees of freedom). Trajectory bands are pointwise intervals, not simultaneous bands. Matrix/layer dispersion is reported separately and is not the confidence interval.

Validation perplexity is per character. Each seed's perplexity is `exp(validation CE)`; the five perplexities, not the mean CE exponentiated, are averaged.

## Verified endpoints

All ± values below are 95% Student-t confidence half-widths across 5 seeds.

| Optimizer | Updates | Model-average alpha | Character validation perplexity |
|---|---:|---:|---:|
| SGDM |2,000|2.204359±0.073953|7.543061±0.080721|
| Native AdamW |2,000|1.640901±0.006738|6.642490±0.074168|
| Modified AdamW |2,000|1.915077±0.046388|6.317157±0.156422|
| SGDM |20,000|2.213622±0.086760|4.609264±0.049570|
| Native AdamW |20,000|3.247494±0.393972|4.341996±0.033591|
| Modified AdamW |20,000|3.131151±0.375255|4.374646±0.021568|

Modified minus native AdamW validation perplexity is −0.325333±0.121508 at 2,000 updates and +0.032650±0.025099 at 20,000. Modified has lower perplexity for every paired seed at 2,000; native has lower perplexity for every paired seed at 20,000. Both optimizers' endpoint alpha confidence intervals lie above 2 at 20,000.

## Trajectory and layer interpretation

Native AdamW falls below 2 early and later rises above 2; it does not remain below 2 throughout training. Its measured minimum mean alpha is 1.614671 at 2,500 updates. Modified AdamW reaches a minimum 1.851392 at 2,500 and rises above 3 by the final checkpoint.

SGDM remains near the marginal region after early relaxation. Its mean alpha ranges 1.974114–2.214505 from 2,000 through 20,000 updates. It dips near 2 mid-run rather than staying flat at 2.2. Its 2,000 and 20,000 confidence intervals exclude exactly 2.

Individual layer fits remain heterogeneous. At 20,000, 40/90 SGDM matrix fits lie below 2, while some exceed 13. Thus collective model-average proximity does not mean every layer approaches 2.

## Runtime and replay limits

Recorded training_seconds range 590.3–1425.2 seconds per seed (9.8–23.8min), for the update timer alone. Evaluation, spectral fitting, workflow setup, and archiving add overhead. These measurements describe the actual hosted CPU runs, not a guarantee for every computer.

The modified metadata includes heterogeneous Intel and AMD CPU models. Native/SGDM replay metadata records runner image and software/runtime settings, but not detailed CPU model. Matching version strings do not guarantee identical numerical kernels. Initial matrix-fit spectra match for each seed across all three arms; some initial validation CEs differ by approximately 10^−8 across runtimes.

## Recompute

From the source bundle directory:

```bash
python download_cohorts.py
python recompute_statistics.py
```

The download script uses fixed commit SHAs and verifies Git blob hashes. The statistics script uses Python's standard library, checks completeness/provenance/fit counts, reconstructs the requested estimator, verifies stored primary analyses, and regenerates the CSV/JSON outputs. It never launches training.

Useful outputs: `trajectory.csv`, `per_seed.csv`, `endpoint_summary.csv`, `matrix_fits.csv`, `matrix_endpoint_summary.csv`, `family_per_seed.csv`, `family_summary.csv`, and `verified_summary.json`. Family averages first apply the same fit weights within each family for each seed, then average the five seed family values; they do not replace the full 18-matrix model estimator.

