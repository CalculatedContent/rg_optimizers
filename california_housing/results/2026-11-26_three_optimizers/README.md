# California Housing: primary GitHub cohort

This directory is the durable archive for the authorized comparison of SGDM,
native AdamW, and modified AdamW on the 50,817-parameter California Housing MLP.

- **Source and instructions:** [experiment README](../../README.md)
- **Design:** [frozen plan](../../plan.json)
- **Primary seeds:** 1340–1344, all three optimizers, 20,000 updates each.
- **Primary budgets:** 2,000 and 20,000 updates.
- **Development only:** seed 1339, nine fixed calibration candidates.

The workflow will create:

| Path | Contents |
|---|---|
| `calibration/` | All development runs, validation choices, exact split indices and data provenance |
| `frozen_selection.json`, `configs/` | Frozen optimizer settings, plan/source/data hashes and launch provenance |
| `runs/seed_<seed>/<optimizer>/` | 41 measurements, raw layer fits, runtime metadata and resumable checkpoints |
| `analysis/RESULTS.md` | Human-readable comparison at 2,000 and 20,000 updates |
| `analysis/summary.json` | Mean, sample SD, 95% Student-t CI, paired effects and audit status |
| `analysis/figures/` | Average-alpha and prediction-error trajectories with seed confidence bands |

Results are complete only when the final analysis reports
`complete_five_seeds=true`, five complete seed triplets and 15 complete runs.
No training results are claimed by this initial setup file.
