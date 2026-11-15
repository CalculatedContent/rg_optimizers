# Campaign log

| Date | Work | Status |
|---|---|---|
| 2026-11-13 | Experiment 01, MNIST MLP | 8 runs completed initially; 6 interrupted runs resumed and finished. All 14 verified. 121 remain unstarted. |
| 2026-11-14 | Experiment 01 completed | Reused 14 completed runs; trained the remaining 121. All 135 verified, 675 checkpoints preserved, eight plots and full paired five-seed statistics prepared. Graft benefits the central setting but collapses at batch 128 / LR ×2 in all five seeds. |
| 2026-11-14 | Batch-128 fresh-seed replication | Seed 31006, LR ×2: all three 120-epoch runs verified. AdamW 98.52%, graft 11.35%, SGDM 98.01%. Graft collapse reproduced; 15 new checkpoints saved. Original primary analysis unchanged. |
| 2026-11-14 | Presentation cleanup | Main report uses 120 balanced stable-setting runs. Six graft collapses are recorded as batch128/LR×2 hyperparameter failures; 12 successful controls at that setting remain outside the shared subset. Original full-grid analysis and all raw outcomes preserved. No training. |
| 2026-11-15 | Model-status check-in | Labeled all 138 models: 132 usable, 6 confirmed failures. Added a per-setting status plot, central-versus-stable paired-effect plot and concise summary; recomputed intervals match saved results. Original evidence preserved; no training. |
| 2026-11-15 | Experiment 02 Fashion-MNIST setup | Created a sibling folder and 120-run plan using the same MLP, exact graft and eight shared settings. Added run matrix, reference source, setup/data validator and separate follow-up space. No data staged or training launched; Experiment 01 unchanged. |
| 2026-11-15 | Experiment 02 execution | Explicitly authorized and started at 09:57:34 UTC. Fashion-MNIST checksums and preflight passed; one fixed 120-run pass with incremental checkpoint preservation and automatic verification. Final results pending. |
| Subsequent days | Further experiments | Awaiting review of Experiment 02; no additional run started. |
| 2026-11-15 | Fashion-MNIST runtime interruption | Last update around10:22UTC; 56 completed endpoints, 6 partial runs, 58 unstarted. Two complete seed archives plus available third-seed states saved. No restart; no completed five-seed result claimed. |
| 2026-11-15 | GPT Experiment03 pilot launch | Three cases launched in GitHub Actions run43705017007; numerical preflight passed. Fixed45-run expansion requires all pilot verification/learning/throughput checks; separate96-hour budget. |
