# Accelerated MAGIC SGDM fixed-20k sweep

Complete: True; grid 16/16; confirmation 5/5; failed fits 0.

Selection: fixed-20k validation accuracy, then CE, never alpha/test. One development seed. Reused test is exploratory.

Winner: {'sgdm': {'id': 'g32', 'track': 'sgdm', 'optimizer': 'sgdm', 'lr': 0.024, 'momentum': 0.9, 'weight_decay': 0.001}}

| Metric | Mean | 95% CI half-width | n |
|---|---:|---:|---:|
| alpha | 1.3227953 | 0.0063103501815998594 | 5 |
| D | 0.031787461 | 0.0043487284920459305 | 5 |
| tail | 502.8 | 16.76016444024709 | 5 |
| spectral_norm | 3.0084966 | 0.08048723604211558 | 5 |
| frobenius_norm | 5.7417839 | 0.03583677781980963 | 5 |
| train_accuracy | 0.90138013 | 0.006588007795446109 | 5 |
| train_ce | 0.24098764 | 0.007544934301918294 | 5 |
| train_auc | 0.95829932 | 0.001366294410218744 | 5 |
| train_gamma_at_fpr_0.01 | 0.46581483 | 0.03307986176625725 | 5 |
| train_gamma_at_fpr_0.02 | 0.56438495 | 0.009326590444525891 | 5 |
| train_gamma_at_fpr_0.05 | 0.7340432 | 0.009897320328028428 | 5 |
| train_gamma_at_fpr_0.1 | 0.86674779 | 0.008147205634561627 | 5 |
| train_gamma_at_fpr_0.2 | 0.95986208 | 0.003672580691728102 | 5 |
| test_accuracy | 0.87691903 | 0.0037812887158576764 | 5 |
| test_ce | 0.3041796 | 0.0073006734662203225 | 5 |
| test_auc | 0.93055255 | 0.0027901447950078907 | 5 |
| test_gamma_at_fpr_0.01 | 0.34334278 | 0.012177589890041824 | 5 |
| test_gamma_at_fpr_0.02 | 0.41974909 | 0.016120601128946775 | 5 |
| test_gamma_at_fpr_0.05 | 0.57288547 | 0.018151115388062623 | 5 |
| test_gamma_at_fpr_0.1 | 0.76139215 | 0.014721460563158088 | 5 |
| test_gamma_at_fpr_0.2 | 0.90942938 | 0.007956251641498175 | 5 |
| test_ce_gap | 0.063191956 | 0.0040013873991440714 | 5 |
| test_accuracy_gap | 0.024461094 | 0.005185581945916044 | 5 |

Original five-seed SGDM test accuracy 87.62%, alpha 4.439, CE .31890, AUC .92642; same-split XGBoost 88.22%.

Reused baseline files are referenced by path and SHA-256 in summary.json at immutable primary archive 09dbaa4e095ff21fc8b1e130798d247b75c15913.

Audit errors: []
Failed training: []

[Endpoint CSV](fixed_20k.csv) · [All trajectories](trajectories.csv) · [Statistics](statistics.csv) · [Grid accuracy vs alpha](grid_validation_accuracy_vs_alpha.png) · [Grid CE vs alpha](grid_validation_ce_vs_alpha.png)
