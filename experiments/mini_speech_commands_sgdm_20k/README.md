# Mini Speech Commands: paired SGDM and dropout

Independent CPU experiment on Google's eight-class, 8,000-clip Mini Speech Commands dataset (TensorFlow tutorial: https://www.tensorflow.org/tutorials/audio/simple_audio; Google Speech Commands, Warden 2018, CC BY). We use a new fixed speaker-disjoint split, not the tutorial's random clip split, and a small MLP on deterministic log-mel features, not the tutorial CNN. No external model or pretrained weights.

## Frozen design

- Input: 32 log-mel bands × 32 time bins, 1,024 standardized inputs, training-only normalization.
- Architecture: 1024 → 128 → 128 → 128 → 8, ReLU after each hidden layer; dropout after each hidden activation, not on the input. FC1–FC3 are the three primary eligible matrices. FC4 has eight eigenvalues; it is fitted using the shipped WeightWatcher small-N estimator and reported both individually and in a secondary model average.
- SGDM only, constant LR 0.01/0.03/0.1 × momentum 0.90/0.99 × dropout 0/0.3690073279296028. Weight decay zero, zero dampening, no Nesterov, batch 128, gradient clipping 1. The user requested paired dropout comparison; zero weight decay isolates explicit dropout from weight decay.
- Twelve configurations × two development seeds (24 runs); choose a **common** LR/momentum pair by lowest equal-weight mean fixed-20k validation cross entropy over its two dropout rates and two seeds. No test data accessed in development training, no alpha-based selection, no retuning.
- Five new paired confirmation seeds at each dropout probability (ten runs). Every run exactly 20,000 updates; no early stopping, no long-horizon continuation. Total maximum 680,000 updates, max 12 standard CPU grid workers, max 10 confirmation workers, 45-minute job timeout. One numerical thread per job.
- Metrics at initialization, 2k, 5k, 10k, 15k, and 20k: train/validation CE and accuracy; confirmation only test CE and accuracy; ordinary WW layer fits, D, tail size, status, raw and corrected alpha, per-seed model averages. Keep checkpoint weights at all measured steps and full final optimizer/RNG checkpoint. Diagnostics do not alter model or training RNG.
- Primary alpha average over FC1–FC3 with weights sqrt(tail)/(D+0.02), no size factor; secondary over FC1–FC4. Averages are null if any required layer fails. For dropout-corrected fits, aggregate separately for each mask realization, then average 20 realizations within each seed. No dropout: raw and corrected identical. Report equal-seed means, sample SD and 95% Student-t CI on confirmation seeds (not treating masks as new models).

## Dropout formula

For training dropout $p$:
$$
r(p)=1-\sqrt{1-p},\qquad \widetilde W_{ij}=W_{ij}B_{ij},\quad B_{ij}\sim\mathrm{Bernoulli}(\sqrt{1-p}).
$$
Apply only to copies of layers receiving dropout input (FC2–FC4); FC1 remains unchanged. For $p=0.3690073279296028$, $r=0.20564952818645776$. No parameter search for the correction. Use direct WeightWatcher 0.7.7 analyze(min_evals=8, randomize=False, plot=False) for original and every masked matrix. The prior California Housing result for this fixed masking rule was 2.002959 ± 0.006731 SD, but this is an empirical candidate. The expected-Gram derivation for variance-preserving replacement does not establish an exact zero-mask inversion. See [prior derivation and negative controls](https://github.com/CalculatedContent/rg_optimizers/blob/f09758902997320e9397317bd45a88d7fc3869cd/california_housing_sgdm_20k/dropout_correction/README.md). The present paired controls test whether the correction generalizes; no guarantee alpha will equal 2.

## Reproduce

Python 3.12.9, pinned CPU requirements. From repo root:
```bash
python -m pip install -r experiments/mini_speech_commands_sgdm_20k/requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python experiments/mini_speech_commands_sgdm_20k/run.py --smoke
python experiments/mini_speech_commands_sgdm_20k/prepare.py  # only if prepared archive is absent
python experiments/mini_speech_commands_sgdm_20k/run.py --phase grid --case sgdm_00 --seed 3358
```

Run all frozen development cases/seeds before selection. Use `analysis.py select` and then `run.py --phase confirm --dropout-index 0 --seed 3360` (both dropout indices, all five confirmation seeds); `analysis.py summarize` audits hashes and writes figures/tables. Existing run folders are refused rather than overwritten. Use a clean checkout or restore archived checkpoints instead of replaying a completed run. The workflow archives prepared features, exact split and dataset SHA256, histories, checkpoint hashes, per-run results, frozen selection, tables/plots, and incomplete artifacts if a job fails. Training runs are independent from all California Housing and previous experiments. No new AdamW training or long-horizon workflows.

Current branch is intentionally isolated. The workflow is triggered by a push of its own YAML file only; publication commits do not retrigger training.
