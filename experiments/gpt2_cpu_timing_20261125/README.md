# Full GPT-2 CPU runtime estimate on GitHub Actions

Measured November 25, 2026. **One complete stock GPT-2 124M / 10.255-billion-token run projects to about 333–664 days of equivalent continuous compute on the two standard GitHub CPU runners measured here, including the specified validation schedule.** That is approximately 11–22 months per optimizer and seed for this FP32, eager PyTorch implementation. It is an extrapolation from actual full-model timing, not a completed training run or a prediction of time to a target loss.

## Measured evidence

[Successful two-runner workflow](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45579460692). The executed code is pinned at commit `044b2a8133f50f4f12e097014c2c8619ad3bdf0e`.

Both jobs used `ubuntu-24.04`, **four assigned vCPUs**, four PyTorch threads, approximately 16 GB RAM, Python 3.12.16, PyTorch 2.6.0+cpu, NumPy 2.2.6 and FP32. No GPU, compilation, lower precision, or reduced model was used. The physical processor name does not indicate the number of cores assigned to the VM.

| Runner | Host processor | Selected microbatch | Forward/backward throughput | Training incl. AdamW updates | Validation | Total compute |
|---|---|---:|---:|---:|---:|---:|
| 1 | Intel Xeon 6975P-C | 2 sequences | 366 tokens/s | 324.4 days | 8.7 days | **333.1 days** |
| 2 | AMD EPYC 9V74 | 4 sequences | 184 tokens/s | 645.7 days | 18.0 days | **663.7 days** |

These select the fastest measured median among microbatches 1, 2 and 4 on each runner. All configurations and individual samples are retained in [replica-1.json](replica-1.json), [replica-2.json](replica-2.json) and [projection.json](projection.json). Using microbatch 2 on both runners gives 333.1 and 669.0 days; the batching choice does not change the conclusion. Across all three microbatch medians the ranges are 333–345 days for runner 1 and 664–701 days for runner 2. The interval between the two observed machines is not a statistical confidence interval or a guaranteed future runtime bound.

## Workload and measurement

The benchmark imports the repository's hash-verified stock GPT-2 source: 124,439,808 unique parameters, 12 layers, 12 attention heads, width 768, 50,257-token vocabulary, context 1,024, tied embeddings and PyTorch causal scaled-dot-product attention. It performs the real full-vocabulary cross-entropy forward and backward computations with synthetic token IDs. A corpus download is unnecessary for timing these dense operations.

For each microbatch, one full-shape warmup is excluded and five accumulated forward/backward passes are timed. Gradients are already allocated, the loss is divided by the actual accumulation count, and AdamW's real moment tensors remain resident. This represents the repeated microbatches inside one 524,288-token update. Three validation passes are separately timed after an excluded warmup; they include cross-entropy and token argmax. Gradient clipping, optimizer work, and gradient zeroing are charged once per full update, not once per microbatch. AdamW, SGDM and the portable AdamW-SGDM graft each have a separately warmed optimizer timing.

The optimizer-only stage uses fixed dense nonzero gradients. AdamW uses betas 0.9/0.95, LR 0.0006 and matrix weight decay 0.1; SGDM uses momentum 0.9, LR 0.3 and zero decay; the graft uses betas 0.9/0.95, LR 0.0006, ratio 400, zero decay and the 48 transformer block matrices. These are timing configurations, not a controlled convergence comparison or tuned GPT-2 recommendations.

At the fixed token budget, the three optimizer timing projections differ by less than 0.1 day on either machine: forward and backward computation dominates. This says nothing about differences in the number of tokens needed to learn.

## Extrapolation

The full recipe is 19,560 updates × 524,288 tokens = **10,255,073,280 training-token presentations**. There are 78 full validation events at multiples of 250 updates plus the final evaluation: **79 × 10,485,760 = 828,375,040 validation-token presentations**.

For microbatch B and context 1,024:

```
training_seconds = (10,255,073,280 / (B * 1,024)) * median_forward_backward_seconds
                   + 19,560 * (median_clip_and_optimizer_seconds + zero_grad_seconds)
validation_seconds = (828,375,040 / (B * 1,024)) * median_validation_seconds
total_compute_days = (training_seconds + validation_seconds) / 86,400
```

Recompute from the saved records:

```bash
python experiments/gpt2_cpu_timing_20261125/analyze.py experiments/gpt2_cpu_timing_20261125/replica-1.json experiments/gpt2_cpu_timing_20261125/replica-2.json --output projection.json
```

## Practical limits

This is equivalent continuous compute time, excluding data I/O, WeightWatcher/SVD, checkpoint writes and transfers, package installation, queues and repeated job startup. Actual elapsed completion time would include those costs. The synthetic timing probe does not demonstrate convergence to loss 3.28. A different implementation, numerical precision or future runner can have different performance; no assumption of a speedup from untested optimizations is included.

[GitHub's standard public-repository runner specification](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) lists four CPUs and 16 GB RAM for `ubuntu-24.04`. [GitHub's Actions limits](https://docs.github.com/en/actions/reference/limits) cap each hosted job at six hours. Therefore this estimate does not imply a single uninterrupted run is possible on GitHub Actions. Only the short timing probe was executed.

Raw measurements, the projection code and this report are committed here rather than depending on expiring workflow artifacts.
