# Published California Housing MLP reproduction

Status: prepared for a fresh GitHub cohort; final results require all 15 confirmation runs and the analysis job to succeed.

- [Protocol and reproduction instructions](../../README.md)
- [Frozen design](../../plan.json)
- Published native AdamW target: **0.4935855 RMSE**, sample SD **0.0110033** over 10 seeds, in $100,000 units.
- New confirmation seeds: **2340–2344**, all three optimizers.
- Primary: best validation checkpoint before the first patience stop.
- Secondary: 2,000 and 20,000 updates.
- Original exploratory cohort remains under `california_housing/results/2026-11-26_three_optimizers`.

The workflow commits exact data provenance, development metrics, frozen settings, per-seed results, checkpoints, and `analysis/RESULTS.md` here.
