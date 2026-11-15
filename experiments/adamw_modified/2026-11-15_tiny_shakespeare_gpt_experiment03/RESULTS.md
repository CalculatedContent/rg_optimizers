# Experiment03: GPT results

0/45 confirmatory runs completed and verified. Pilot runs are excluded.

The magnitude-only graft retains Adam direction/moments/bias updates on each run. It changes only the magnitudes of the eight physical attention/MLP matrix updates. Embeddings and tied output head retain native Adam updates in the graft arm.

Five seeds are the independent replicates. Across three learning-rate settings, differences are averaged within each seed before paired 95% t intervals. All outcomes remain visible; unsupported fits are flagged. Positive alpha movement is not necessarily closer to2. Lower test loss is better.

[Every model](results/model_status.csv) · [Per-matrix spectra](results/layer_spectra.csv) · [Paired effects](results/paired_effects.csv)

**Incomplete pass: no completed five-seed conclusion is claimed.**
