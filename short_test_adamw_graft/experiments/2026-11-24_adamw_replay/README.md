# TinyShakespeare: three-optimizer reproduction

This branch retains the prior SGDM and graft experimental records; native AdamW is a fresh replay, not pooled with historical controls.

- Native AdamW: short_test_adamw_graft/experiments/2026-11-24_adamw_replay; workflow .github/workflows/shakespeare-adamw-replay-20261124.yml.
- SGDM: short_test_adamw_graft/experiments/2026-11-24_sgdm_replay; workflow .github/workflows/shakespeare-sgdm-replay-20261124.yml; original branch shakespeare-sgdm-replay-20261124, run 45386899611.
- Modified AdamW: short_test_adamw_graft/experiments/2026-11-24_five_seeds; workflow .github/workflows/shakespeare-graft-five-seeds-20261124.yml; original branch shakespeare-graft-five-seeds-20261124, run 45375423863.

## Reproduce locally

Use a fresh clone of this branch and Linux x86-64 CPU, Python 3.12.14, torch 2.14.1 CPU and the frozen requirements. From repository root:

```bash
SOURCE=short_test_adamw_graft/experiments/2026-11-24_five_seeds
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
python -m pip install -r "$SOURCE/requirements.txt"
python -m zipfile -e "$SOURCE/experiment_inputs.zip" "$SOURCE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SHORT_GPT_DEVICE=cpu
```

Use separate clean working copies for fresh reruns; preserve archived results before removing or moving existing runs. For each seed 1340–1344 invoke the native or SGDM replay.py with the seed argument, then invoke it with analyze. For the graft use the original branch and its frozen workflow and README (run_additional_seed.py --seed SEED; see original workflow for compatibility and archive steps). Do not overwrite published records. The GitHub workflows define the exact setup and commands.

All arms use 20,000 updates, warmup 100, cosine end 17,000, LR floor 0.1, batch 12, clipping 1, contiguous 90/5/5 split. Native AdamW peak LR .001, betas (.9,.99), weight decay 0; SGDM peak LR .1, momentum .9, weight decay 0; graft calibrated reference peak LR .4. Preserve 2k and 20k weight/optimizer/RNG checkpoints, raw fits every 500 steps and runtime metadata. Checkpoint archives reside in GitHub, not ChatGPT storage.

## Analysis

Compute model alpha separately per seed using frozen extract_spectrum, then arithmetic mean over five seeds and 95% Student-t half-width 2.77644511*s/sqrt(5). Layer dispersion is separate, not a seed CI. Report validation perplexity (exp CE per character), accuracy and alpha at every 500 steps. Compare 2k and 20k replay results to historical native AdamW reference means (6.643 and 4.339 validation perplexity, 3.243 final alpha); historical rounded values are descriptive, not an equivalence test. Retain final test metrics in result.json. Do not claim reproduction until all five jobs and analysis succeed.
