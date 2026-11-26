# Power Plant fixed-20k results

Complete: True; development 24/24; confirmation 15/15; optional dropout 0/15.

| Phase | Track | Metric | Mean | SD | 95% CI half width |
|---|---|---|---:|---:|---:|
| confirm | sgdm_wd0 | test_rmse | 3.995622924047749 | 0.04860167427014895 | 0.060346949193718115 |
| confirm | sgdm_wd0 | test_mae | 2.926493754072584 | 0.06643435445751432 | 0.08248914617386885 |
| confirm | sgdm_wd0 | alpha | 13.04067471644 | 1.413579836915192 | 1.755191192085737 |
| confirm | sgdm_wd0 | corrected_alpha | 13.04067471644 | 1.413579836915192 | 1.755191192085737 |
| confirm | adamw_wd0 | test_rmse | 3.716504785775678 | 0.06194400707176272 | 0.07691364348554827 |
| confirm | adamw_wd0 | test_mae | 2.6034635524772853 | 0.06527117332573751 | 0.08104486603915052 |
| confirm | adamw_wd0 | alpha | 2.7299332768 | 0.09523462378259502 | 0.11824940373342219 |
| confirm | adamw_wd0 | corrected_alpha | 2.7299332768 | 0.09523462378259502 | 0.11824940373342219 |
| confirm | adamw_wd001 | test_rmse | 3.6975038210288678 | 0.0744876765127644 | 0.09248866623582117 |
| confirm | adamw_wd001 | test_mae | 2.589676403398988 | 0.0851510454697327 | 0.10572898756926286 |
| confirm | adamw_wd001 | alpha | 2.75577599786 | 0.06829018174305838 | 0.08479346010121495 |
| confirm | adamw_wd001 | corrected_alpha | 2.75577599786 | 0.06829018174305838 | 0.08479346010121495 |

Published outer split 0 only; 11 identical full test rows overlap outer train. Literature 20-split MC-dropout aggregate is context, not same estimand.

Audit errors: []
