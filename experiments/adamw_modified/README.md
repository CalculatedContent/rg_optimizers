# Modified AdamW experiments

This campaign studies a **magnitude-only** modification: retain Adam's adaptive matrix-update direction and moment history, and replace its Frobenius magnitude with the SGDM target magnitude computed from that same run's gradient history. Bias updates remain AdamW. Controls are native AdamW and SGDM with matched decay.

## Experiments

| Folder | Dataset / model | Status |
|---|---|---|
| [2026-11-13_mnist_mlp_experiment01](2026-11-13_mnist_mlp_experiment01/) | MNIST, 784→256→256→10 ReLU MLP | 135 original runs + 3 replication runs verified; 132 usable models and 6 failed models; explicit model labels, paired-effect plots, and separate hyperparameter-failure report |
| [2026-11-15_fashion_mnist_mlp_experiment02](2026-11-15_fashion_mnist_mlp_experiment02/) | Fashion-MNIST, 784→256→256→10 ReLU MLP | Historical interrupted/superseded passes; recovery stopped and available outcomes preserved. Completed clean pass is in the parallel folder below. |
| [2026-11-15_fashion_mnist_mlp_experiment02_clean](2026-11-15_fashion_mnist_mlp_experiment02_clean/) | Fashion-MNIST, 784→256→256→10 ReLU MLP | Complete: 120/120 fixed-epoch-20 runs, five paired seeds; verified results, checkpoint links and comparison with MNIST. No new training. |
| [2026-11-16_fashion_mnist_mlp_experiment02_extension](2026-11-16_fashion_mnist_mlp_experiment02_extension/) | Fashion-MNIST, same 120 MLP models | Failed before continuation: 0/120 extended; parent replay loss equality failed. Parent checkpoints preserved; no rerun. |
| [2026-11-16_fashion_mnist_mlp_experiment02_extension_retry01](2026-11-16_fashion_mnist_mlp_experiment02_extension_retry01/) | Fashion-MNIST, same120 saved models | Complete and verified: 120/120 fixed-epoch60 continuations; all120 functional, five seeds. Graft accuracy +0.41950pp versus AdamW, but higher CE and worse mean alpha distance. Checkpoints, loss-tail diagnostic and final audit saved; awaiting review. |
| [2026-11-15_tiny_shakespeare_gpt_experiment03](2026-11-15_tiny_shakespeare_gpt_experiment03/) | Tiny Shakespeare, two-block GPT | On hold: finish Fashion-MNIST first; pilot controls preserved, graft cancelled,45-run expansion cancelled |

## Organization for the coming days

### Weighted analysis and general comparison

The same baseline-severity weighting is now applied to both datasets, with separate dataset reports and a third, reusable comparison folder:

| Analysis | Contents |
|---|---|
| [Original MNIST](2026-11-13_mnist_mlp_experiment01/post_analysis/weighted_alignment/CAMPAIGN_REPORT.md) | Five-seed weighted FC1/FC2 alignment, accuracy and CE; shared eight-setting comparison and separate full-grid failure sensitivity |
| [Fashion-MNIST](2026-11-16_fashion_mnist_mlp_experiment02_extension_retry01/post_analysis/weighted_alignment/CAMPAIGN_REPORT.md) | The same calculations at epochs 20 and 60, retaining their dependence and all outcomes |
| [General comparison](comparisons/README.md) | Cross-experiment plots, numerical tables, source hashes, reusable code/tests and a registry for future completed experiments |

On the shared settings, the graft's combined weighted alignment improvement is +0.07998 [0.06628, 0.09369] for MNIST, versus −0.00952 [−0.03221, 0.01317] for Fashion at epoch 60. Positive means closer to alpha=2. Performance outcomes are reported separately; schedules differ and the new weighting is post-hoc. All original data and failure records remain unchanged.

### Future experiment organization

Keep each experiment in a dated child folder: `YYYY-MM-DD_dataset_architecture_experimentNN/`. Continue the same experiment in its existing folder; do not duplicate completed runs. Store immutable source/config snapshots in `frozen/`, small numerical results in `results/`, and audit evidence in `verification/`. Record large artifact locations in `ARTIFACTS.md`.

Use [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) for future experiments over the next five days. Maintain [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md) as work actually completes. Future work is planned, not already running.

Run one experiment at a time with a 3–5 hour budget, preserve checkpoints, verify results, and review before starting the next experiment. Use dense layers or transformers; no CNNs. Treat seeds as replicates and preserve all preset hyperparameter outcomes. Present shared stable settings separately from documented failed settings, and label any post-hoc selection. Report upward alpha shifts separately from distance to alpha=2 and accuracy.



[Experiment 01 model status and concise results](2026-11-13_mnist_mlp_experiment01/MODEL_STATUS.md) include the complete 138-model register, setting-level status plot and paired α-alignment/accuracy plot.

[MNIST versus completed Fashion-MNIST analysis](2026-11-15_fashion_mnist_mlp_experiment02_clean/COMPARISON.md) uses the eight common settings and five independent seeds. MNIST used 120 epochs; Fashion used 20 epochs. The dataset and schedule changes are explicitly separated from within-experiment optimizer effects.

## Active scale-audited Fashion follow-up

[New experiment folder](2026-11-16_fashion_mnist_mlp_experiment02_extension_retry01/follow_up/2026-11-16_scale_equivariance/) · [Running workflow](https://github.com/CalculatedContent/rg_optimizers/actions/runs/43839661129).

Launched16 November2026: three pilot arms, followed conditionally by fifteen fresh paired-seed runs. Same dense MLP and exact magnitude-only kernel; fixed120 epochs and matched mild translations. Remote source/numerical preflight passed; training step is active. All three alpha estimates are retained, including the audited small-N FC3 estimate and native counterpart. The scale claim is conditional coordinate covariance with distinct matrix/bias rate transformations; full RG invariance is unproved. No historical results are overwritten.

### Completed scale-audited Fashion follow-up (16 November 2026)

[Results and plots](2026-11-16_fashion_mnist_mlp_experiment02_extension_retry01/follow_up/2026-11-16_scale_equivariance/RESULTS.md): 15/15 fixed-120 endpoints verified; all five paired seeds retained. Graft minus AdamW accuracy −0.344 pp [−0.561,−0.127], CE +0.06115 [0.05681,0.06550], hidden distance +0.59458 [0.39875,0.79042]. Publication recovered from saved artifacts without rerunning training. Extra durable backup pending after upload preparation failed; GitHub checkpoint/audit artifacts preserved for 90 days. GPT remains held.

Completion note (16 November): training and report publication are finished. A second attempt with refreshed backup helpers failed at upload preparation; no durable backup receipt exists. Verified checkpoint and audit archives remain in GitHub Actions until 14 February 2027. Monitoring is paused; no additional training authorized.

### Authorized unaugmented epoch60-to120 continuation (16 November 2026)

[New continuation folder](2026-11-16_fashion_mnist_mlp_experiment02_epoch120_noaug/): same120 saved epoch60 models, five original seeds and eight settings; no augmentation or optimizer reset. Hold the actual terminal LR at1% of peak through fixed epoch120. Explicit step/LR/decay accounting and paired within-model endpoint analysis. New workflow submitted with full120-state numerical preflight before training; completion not yet claimed. At equal epoch60, prior augmented/unaugmented runs had the same12900 B256 updates but58.07-fold different current LR, so that historical comparison does not isolate duration or augmentation. Historical results remain unchanged; GPT held.

Continuation status: [run43861743291](https://github.com/CalculatedContent/rg_optimizers/actions/runs/43861743291), launch commitda3e871798c9574d7228a39aac1d808862c76bf1. All120 restored parents passed numerical preflight; exact two-copy current-runtime replay120/120,9 wrong controls rejected,15 resumed graft matrix checks passed. Historical replay differs by at most2.98e-8 loss, within the frozen bounds. Training step is active; final results pending. [Verified launch receipt](2026-11-16_fashion_mnist_mlp_experiment02_epoch120_noaug/verification/Launch_Preflight_Receipt.json).

### Native AdamW paired120 diagnostic — queued after the active continuation

[Protocol and code](2026-11-16_fashion_mnist_adamw_paired120/README.md): two paired seeds33001/33002 × augmentation off/on; native AdamW only, fixed120epochs, identical LR/update count, every-epoch spectra/checkpoints/train-validation-test curves. The user's latest correction supersedes the unlaunched1000-epoch draft. Shared concurrency queues this pass after run43861743291; no historical experiment is changed. Two seeds are exploratory. All outcomes are retained.

**User-directed priority update,16Nov2026:** the unaugmented continuation43861743291 was cancelled with49/120 endpoints retained. Its partial results and all available states are preserved. [NativeAdamW paired120 run43864280628](https://github.com/CalculatedContent/rg_optimizers/actions/runs/43864280628) has started; its frozen protocol records all epochs0–120. See each folder's verification receipt. No continuation restart is authorized.

### Native AdamW augmented digit-MNIST replication (16 November 2026)

The Fashion native-AdamW paired120 diagnostic completed all four trajectories, with numerical verification and plots published. The sub-2 hidden-layer deficit faded by epoch120, while distance to2 and performance had different trajectories. Large checkpoint artifacts remain preserved on GitHub; additional durable backup is being handled separately.

[New MNIST protocol and code](2026-11-16_mnist_adamw_augmented120/README.md): two independent seeds33001/33002, augmentation ON only, native AdamW, same MLP/B256/LR schedule as the Fashion augmented arm, fixed120epochs. Every epoch records spectra, full metrics and checkpoints; every minibatch records updates. Submitted with source/data/numerical preflight before training. No endpoint selection, tuning or automatic extension; all historical results preserved. Final results are pending.

MNIST launch verified: [run43872338938](https://github.com/CalculatedContent/rg_optimizers/actions/runs/43872338938), commitfffbf2427d81ed4fb22426677033f74f8618c976. Exact original digit-MNIST data/split hashes, native AdamW reference30parameter checks, smoke replay, three wrong-state/input/rate controls, deterministic augmentation and100small-N estimator checks passed remotely. Two main120epoch runs have entered training. [Launch/preflight evidence](2026-11-16_mnist_adamw_augmented120/verification/Launch_Preflight_Receipt.json). Final outcomes remain pending; hourly text monitoring is active.
