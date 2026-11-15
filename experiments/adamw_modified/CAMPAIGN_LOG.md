# Campaign log

| Date | Work | Status |
|---|---|---|
| 2026-11-13 | Experiment 01, MNIST MLP | 8 runs completed initially; 6 interrupted runs resumed and finished. All 14 verified. 121 remain unstarted. |
| 2026-11-14 | Experiment 01 completed | Reused 14 completed runs; trained the remaining 121. All 135 verified, 675 checkpoints preserved, eight plots and full paired five-seed statistics prepared. Graft benefits the central setting but collapses at batch 128 / LR ×2 in all five seeds. |
| 2026-11-14 | Batch-128 fresh-seed replication | Seed 31006, LR ×2: all three 120-epoch runs verified. AdamW 98.52%, graft 11.35%, SGDM 98.01%. Graft collapse reproduced; 15 new checkpoints saved. Original primary analysis unchanged. |
| 2026-11-14 | Presentation cleanup | Main report uses 120 balanced stable-setting runs. Six graft collapses are recorded as batch128/LR×2 hyperparameter failures; 12 successful controls at that setting remain outside the shared subset. Original full-grid analysis and all raw outcomes preserved. No training. |
| 2026-11-15 | Model-status check-in | Labeled all 138 models: 132 usable, 6 confirmed failures. Added a per-setting status plot, central-versus-stable paired-effect plot and concise summary; recomputed intervals match saved results. Original evidence preserved; no training. |
| Subsequent days | Further experiments | Awaiting review of the completed batch-128 replication; no further run started. |


