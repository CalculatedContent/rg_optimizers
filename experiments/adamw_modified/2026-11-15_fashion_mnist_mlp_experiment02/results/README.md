# Results: pending

No Fashion-MNIST models have been trained. `STATUS.json` records 120 planned runs and zero completed runs. Usable/failed model counts remain null until measured; zero completed is not zero failures established by experiment.

When executed, save endpoints, per-layer spectra and fit quality, learning trajectories, sampled gradient/update statistics, saved test logits, seed summaries and paired differences here. Preserve run IDs from `configs/run_matrix.csv`.

The main analysis includes every planned outcome. Keep a model-status register separate from execution completion: USABLE for a trained classifier without observed collapse; FAILED for documented collapse or nonfinite training; INCOMPLETE for a stopped run. Alpha proximity is an outcome, not the usability criterion. Follow-up replications are separate cohorts and must not be silently pooled into the five-seed intervals.

Use the central batch256/LR×1 comparison and the eight-setting mean declared in the protocol. For the latter, first average differences within each seed; then compute descriptive paired 95% Student t intervals across the five seeds. If new Fashion-MNIST failures motivate a stable-only sensitivity analysis, label that subset post hoc and match settings across all three arms.
