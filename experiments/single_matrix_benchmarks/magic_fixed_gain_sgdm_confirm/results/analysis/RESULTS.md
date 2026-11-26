# MAGIC fixed-gain SGDM five-seed exploratory confirmation

Complete: True; completed 5/5; checkpoints 30/30; errors: []

Configuration chosen by user after observing development alpha; not the validation accuracy winner. Fixed gains, LR .03, momentum .9, zero decay, 20k updates. Same original outer-train/test split and confirmation seeds. Test reused; this is exploratory. AdamW controls are archived models with different parameterization (ordinary or learned gain), not a matched fixed-gain AdamW confirmation.

| Arm | Metric | n | Mean | 95% CI half-width |
|---|---|---:|---:|---:|
| fixed_gain_sgdm | alpha | 5 | 1.87829595666 | 0.04596121478511914 |
| fixed_gain_sgdm | D | 5 | 0.0420205519 | 0.007263149487101458 |
| fixed_gain_sgdm | tail | 5 | 28.8 | 8.797432586763891 |
| fixed_gain_sgdm | test_accuracy | 5 | 0.8760778128286014 | 0.00298803437968883 |
| fixed_gain_sgdm | test_ce | 5 | 0.30721129179000856 | 0.004058359605062807 |
| fixed_gain_sgdm | test_auc | 5 | 0.929148565975974 | 0.0014321661139683347 |
| fixed_gain_sgdm | test_brier | 5 | 0.09275517436656569 | 0.0017340479983190167 |
| fixed_gain_sgdm | test_ece15 | 5 | 0.023195435656973774 | 0.005577564823622739 |
| fixed_gain_sgdm | train_accuracy | 5 | 0.9022213459516298 | 0.0016160596396494716 |
| fixed_gain_sgdm | train_ce | 5 | 0.2410549610853195 | 0.0033686525465174704 |
| ordinary_adamw | alpha | 5 | 2.07150678846 | 0.014874852447710836 |
| ordinary_adamw | D | 5 | 0.02021956728 | 0.0008861180008713401 |
| ordinary_adamw | tail | 5 | 231.6 | 6.83479358710938 |
| ordinary_adamw | test_accuracy | 5 | 0.8656677181913774 | 0.004280828252397639 |
| ordinary_adamw | test_ce | 5 | 0.3995123326778412 | 0.010321057251305709 |
| ordinary_adamw | test_auc | 5 | 0.9173185242891054 | 0.0025931821740046122 |
| learned_gain_adamw | alpha | 5 | 2.07628151832 | 0.007682128097748358 |
| learned_gain_adamw | D | 5 | 0.0321188124 | 0.0034187889868514914 |
| learned_gain_adamw | tail | 5 | 234.4 | 5.013002154028703 |
| learned_gain_adamw | test_accuracy | 5 | 0.8517350157728706 | 0.01864827460666784 |
| learned_gain_adamw | test_ce | 5 | 0.48851515650749205 | 0.028369674553255186 |
| learned_gain_adamw | test_auc | 5 | 0.9102830948530334 | 0.0024585580750627113 |
| fixed_sgdm_minus_ordinary_adamw | alpha | 5 | -0.19321083179999995 | 0.05980869549471646 |
| fixed_sgdm_minus_ordinary_adamw | D | 5 | 0.02180098462 | 0.00780066096942073 |
| fixed_sgdm_minus_ordinary_adamw | tail | 5 | -202.8 | 10.026004308057406 |
| fixed_sgdm_minus_ordinary_adamw | test_accuracy | 5 | 0.010410094637223977 | 0.0036171187273219153 |
| fixed_sgdm_minus_ordinary_adamw | test_ce | 5 | -0.09230104088783264 | 0.007176746939394127 |
| fixed_sgdm_minus_ordinary_adamw | test_auc | 5 | 0.01183004168686852 | 0.0019508818970651837 |
| fixed_sgdm_minus_learned_gain_adamw | alpha | 5 | -0.19798556165999998 | 0.05331893113269442 |
| fixed_sgdm_minus_learned_gain_adamw | D | 5 | 0.009901739500000001 | 0.008290977092262009 |
| fixed_sgdm_minus_learned_gain_adamw | tail | 5 | -205.6 | 10.630558367701676 |
| fixed_sgdm_minus_learned_gain_adamw | test_accuracy | 5 | 0.024342797055730836 | 0.017850782841724273 |
| fixed_sgdm_minus_learned_gain_adamw | test_ce | 5 | -0.18130386471748353 | 0.025995962880844842 |
| fixed_sgdm_minus_learned_gain_adamw | test_auc | 5 | 0.01886547112294055 | 0.0015094001453634437 |

Intervals are across training seeds, not independent test-population uncertainty. Paired differences use matching seeds. No alpha-based selection among new runs. Report failures and short tails without excluding them silently.

[Endpoints](fixed_20k.csv) · [Trajectories](trajectories.csv) · [Paired differences](paired_differences.csv) · [Alpha plot](alpha_trajectory.png)
