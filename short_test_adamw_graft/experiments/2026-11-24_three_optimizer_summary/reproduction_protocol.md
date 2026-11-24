# Shakespeare GPT: reproduction protocol (v12)

## What this package reproduces

The primary comparison contains three independent GitHub Actions cohorts: native AdamW, modified AdamW (a matrix-magnitude graft), and SGD with momentum. Each cohort contains seeds 1340–1344 and the same 20,000-update design. The 2,000-update observations are saved checkpoints on that long schedule; they are not separately tuned short runs. Every published trajectory has 41 measurements (0, 500, …, 20,000), each with 18 unique matrix fits. This gives 15 runs, 615 model observations and 11,070 matrix fits.

The repository is <https://github.com/CalculatedContent/rg_optimizers>. Immutable result revisions are:

| Arm | Revision | Workflow run |
|---|---|---|
| Modified AdamW | [e0440e5a42dc](https://github.com/CalculatedContent/rg_optimizers/commit/e0440e5a42dc6ba7f19d2d170f1a0d7551a16410) | 45375423863 |
| SGD with momentum | [23ba932c01d6](https://github.com/CalculatedContent/rg_optimizers/commit/23ba932c01d63653985c28ef9df5ef3804ee4297) | 45386899611 |
| Native AdamW | [163980048ab7](https://github.com/CalculatedContent/rg_optimizers/commit/163980048ab7381e02b97a8620b25ddefa52b613) | 45393623223 |

The corresponding experiment directories under `short_test_adamw_graft/experiments/` are `2026-11-24_five_seeds`, `2026-11-24_sgdm_replay`, and `2026-11-24_adamw_replay`. The native replay revision also contains the identical archived modified and SGDM records. The source bundle retains the exact frozen training files plus a new convenience launcher; it contains no model weights. The convenience launcher did not produce the published cohort. It calls the unchanged archived training functions and adds independent output paths and guarded retention of the intermediate checkpoint.

## Architecture and corpus

The model uses the unmodified nanoGPT model definition from commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`: <https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py>. Four pre-normalized transformer blocks contain four causal attention heads each, width 128, MLP expansion width 512 with GELU, and learned positions for a 64-character context. Linear and LayerNorm biases are disabled; dropout is zero. Token embeddings and output weights are tied. The model has exactly 804,096 trainable parameters, including position embeddings: 8,320 token/output parameters, 8,192 position parameters, 786,432 transformer-matrix parameters, and 1,152 LayerNorm scale parameters. The model constructor’s default printed count subtracts the position embeddings; use the unique trainable-parameter count instead.

The corpus is the 1,115,394-character Tiny Shakespeare text from <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>. Its SHA256 is `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. Sorting the 65 unique Unicode characters defines the character-to-integer map. This is character tokenization, with no subword tokenizer. The first 1,003,854 characters form training; the next 55,770 form validation; the last 55,770 form test. Boundaries use `int(.90*N)` and `int(.95*N)`. The vocabulary is fixed from the corpus, but validation/test text never enters gradient updates.

## Randomness, minibatches, and numerical execution

For seed s, Python `random`, NumPy and PyTorch CPU initialization use s. Each model initializes on CPU using the unchanged nanoGPT constructor: normal standard deviation 0.02 for linear/embedding weights, unit LayerNorm scales, and residual-projection standard deviation `0.02/sqrt(2*n_layer)`. A distinct CPU `torch.Generator` with seed s+1 draws each training batch. It draws 12 start indices uniformly with replacement using `torch.randint(len(train)-64, (12,), generator=rng)`; the inputs span 64 characters and the targets are shifted by one. Thus each update scores 768 sampled next-character targets. These are sampled windows, not disjoint epochs. Matching seeds pair initial weights and minibatch indices across arms.

All runs use Linux x86-64 CPUs, float32 model/gradient/optimizer tensors, one numerical thread, and no GPU, autocast, gradient accumulation or compilation. Before imports, set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to 1, and `SHORT_GPT_DEVICE=cpu`. The code additionally calls `torch.set_num_threads(1)`. AdamW and SGD use `foreach=False`; AdamW also uses `fused=False`. Global gradient norm clipping at 1.0 occurs before each optimizer update. WeightWatcher diagnostics restore Python, NumPy and global PyTorch RNG states after fitting, so fitting does not perturb the training stream.

## Frozen optimizer recipes and schedule

Native AdamW uses peak learning rate 0.001, betas (0.9, 0.99), epsilon 1e-8, zero weight decay and default AMSGrad=false. SGDM uses peak rate 0.1, momentum 0.9, zero weight decay, zero dampening and no Nesterov correction. All parameter groups share each arm’s scheduled rate. The name AdamW identifies the implementation even though the selected recipe has zero decoupled decay.

Modified AdamW retains the same Adam moments and per-matrix Adam direction. For each of the 16 transformer matrices it rescales the proposed Adam update to the Frobenius norm of a momentum-SGD reference with peak rate 0.4 and momentum 0.9, computed from that model’s own clipped gradient history. Its reference/native learning-rate ratio remains 400 at every step. With zero initial momentum, Adam’s uncorrected first moment divided by `(1-beta1)` equals the corresponding unnormalized SGDM momentum buffer. The implementation uses that identity, not the gradients from an independently trained SGDM model. The norm denominator is clamped below by 1e-30. Embeddings, tied output weights and LayerNorm vectors retain native AdamW updates. No alpha value feeds back into training. The frozen base runner’s *default* graft reference is 0.1; this cohort uses the selected 0.4 override. Do not run the base graft default to reproduce this result.

The schedule uses zero-based update index k = 0,…,19,999. For peak rate p, it is `p*(k+1)/101` for k<100; `0.1*p + 0.5*(p-0.1*p)*(1+cos(pi*(k-100)/16900))` for 100<=k<17000; and `0.1*p` thereafter. At the 2,000-update checkpoint, the last applied AdamW rate is 0.0009722512092515008, SGDM rate 0.09722512092515007, and graft reference rate 0.3889004837006003. At 20,000 updates the rates are 0.0001, 0.01, and 0.04 respectively. Changing the cosine end to 2,000 would define a different experiment.

## Calibration and confirmation

The rates were fixed before the five fresh confirmation runs. The original base-optimizer calibration used seed 2027. AdamW candidates crossed learning rates {0.0003, 0.001, 0.003} with decay {0, 0.1}; SGDM crossed rates {0.01, 0.03, 0.1} with momentum {0.9, 0.98}. Every candidate received 4,000 updates. The two best per optimizer continued unchanged to 20,000. Ranking used the lowest full validation cross-entropy from update 500 onward, then endpoint validation loss and candidate ID for ties. Calibration did not inspect test metrics or alpha.

The later graft study used the same calibration seed 2027 and reference-rate candidates {0.05, 0.1, 0.2, 0.4, 0.8}. It advanced the original 0.1 rate and the two alternative rates with lowest validation loss at 4,000 updates to 20,000. It selected the best alternative by endpoint validation loss, breaking ties toward the lower rate; this selected 0.4. The rate remained fixed for seeds 1340–1344. The bound on the search means these recipes are selected configurations, not proven global optima. The test partition had been exposed in an earlier discovery pilot; no confirmation test or alpha result selected new hyperparameters. Reproducing the present confirmation experiment does not require repeating the calibration or selecting another rate.

## Measurements and uncertainty

Every 500 updates, including initialization, the runner evaluates the entire validation split and fits all 18 unique matrices. Full evaluation scores every next-character target once, resets context at each non-overlapping 64-character block, includes the final partial block, and starts positions at zero in each block. Validation/test each have 55,769 scored targets; training has 1,003,853. Cross-entropy uses natural logarithms and is reported in nats per character. Character perplexity is exp(cross-entropy) **within each seed**, followed by the seed mean. Top-1 next-character accuracy is supplementary. A separate training diagnostic uses 20 sampled batches from a fresh generator seeded 10001. Full train/test evaluation occurs at 20,000 updates and at the checkpoint with lowest full validation loss; 2,000-step test metrics were not recorded.

Each spectrum is fit by unmodified WeightWatcher 0.7.7 `WeightWatcher(single_linear_matrix).analyze()` defaults. Q, K and V remain packed in their 384x128 projection matrix. The four blocks contribute four matrices each: packed QKV 384x128; attention output 128x128; MLP expansion 512x128; MLP contraction 128x512. The token/output matrix is 65x128 and the position matrix 64x128. Count the tied output/token matrix once and exclude LayerNorm vectors. All 18 fits must succeed.

For matrix l, use the ordinary fitted `alpha`, tail count `num_pl_spikes`, and Kolmogorov–Smirnov distance `D`. Compute `w_l=sqrt(num_pl_spikes_l)/(D_l+0.02)` and `alpha_bar=sum(w_l*alpha_l)/sum(w_l)`. Do not use WeightWatcher’s `alpha_weighted`, do not multiply by matrix size, and do not pool matrices across seeds. Seed model averages receive equal weights. Sample SD uses denominator n-1=4; SEM is SD/sqrt(5); the 95% Student-t interval for each seed mean is mean ± 2.7764451051977934*SEM. Trajectory bands are pointwise intervals, not simultaneous confidence bands or layer-wise error bars. Layer dispersion describes matrix heterogeneity separately. These seed intervals condition on the fixed corpus and recipe; they do not estimate uncertainty across datasets or calibrated uncertainty of individual tail fits.

## Run from the source bundle, without a notebook

From the unpacked source-bundle root, create a dedicated **Python 3.12.14** environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -c 'import platform; assert platform.python_version()=="3.12.14"'
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
python -m pip install -r data/protocol/modified/requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 SHORT_GPT_DEVICE=cpu
python data/protocol/reproduce.py --verify-only
python data/protocol/reproduce.py --preflight
```

The CPU wheel reports 2.14.1+cpu. Other numerical pins are NumPy 2.3.5, SciPy 1.17.0, WeightWatcher 0.7.7 and powerlaw 2.0.0; figures use Matplotlib 3.10.8 and pandas 2.2.3. The old portable requirements under frozen_source specify a different NumPy range and must not replace these pins. Preflight checks shared initialization/minibatch state, exact local checkpoint replay, and 18 successful fits. It performs only three updates per arm as a numerical contract check.

Run the fixed recipe for all 15 confirmation reruns, or use one arm/seed first:

```bash
for arm in sgdm adamw graft; do
  for seed in 1340 1341 1342 1343 1344; do
    python -u data/protocol/reproduce.py --arm "$arm" --seed "$seed" \
      --output independent_runs
  done
done
python data/protocol/summarize_reproductions.py independent_runs
```

These commands are sequential and require no scheduler. To parallelize, run up to five distinct seed commands for one arm in separate processes, with one numerical thread each and adequate available CPU/memory. The original GitHub workflows used matrix jobs with `max-parallel: 5`; inspect the archived YAML files in `data/protocol/` for their exact environment and archive steps. No new jobs were launched to prepare this paper revision.

To stop deliberately at 2,000 and continue later, use the same output directory and seed:

```bash
python -u data/protocol/reproduce.py --arm adamw --seed 1340 \
  --output independent_runs --until-step 2000
python -u data/protocol/reproduce.py --arm adamw --seed 1340 \
  --output independent_runs --until-step 20000
```

After an interruption, reissue the same command. `last.pt` is saved atomically every 250 updates with weights, optimizer states, RNG states, history, best-validation state, accumulated training time and an arm/seed/design identity. Resume rejects changed identity. The convenience launcher checks the actual stored step before naming or copying a retained checkpoint, and refuses to reconstruct a missing 2,000-step checkpoint from later weights. This guard also avoids the original native/SGDM replay wrapper’s unconditional staging copy when resumed past 2,000. Preserve existing results; use a new output root for every independent replication. Results appear under `independent_runs/<arm>/seed_<seed>/state/<arm>/seed_<seed>/result.json`; each seed folder contains `checkpoint_2000.pt`, `checkpoint_20000.pt`, their ZIP, hashes and local execution metadata. The summary command requires all 15 complete runs and writes seed records, all 41 trajectory points and fixed-budget summaries to `independent_runs/analysis/`.

## Archives and provenance

The published frozen training revision is `b306247d1b0078a18b1539c68ba14bb64d41a079`. `data/protocol/modified/frozen_source/source_manifest.json` contains SHA256 for all 14 bundled source files. The vendored model SHA256 is `7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c`. The appendix prints the core source hashes; `source_provenance.json` and the paper audit’s `data/source_manifest.json` retain hashes and immutable repository links. The corpus and historical comparison audit came from the committed `experiment_inputs.zip` (556,484 bytes; SHA256 `6bce83603324ec6f0bd4e4847c32bd4041ba513a45517ca7cb39a35c17d8fda3`). The local source files and this input archive matched their GitHub identities before use.

Modified checkpoint ZIPs are under `2026-11-24_five_seeds/archives/seed_<seed>_checkpoints.zip`; native and SGDM ZIPs are under each experiment’s `runs/seed_<seed>/checkpoints.zip`. Each contains checkpoints at 2,000 and 20,000. The archiving code checked ZIP integrity when it created the archives. Modified execution metadata additionally records checkpoint step/identity checks, byte counts and SHA256; native replay has `checkpoint_hashes.json`. The current audit checked all 15 archives’ presence, Git blob identities and sizes at pinned revisions; it did not redownload and rehash all checkpoint binaries. `data/checkpoint_inventory.json` makes that verification scope explicit. Raw result JSONs and source files *were* checked against pinned Git blobs. This distinction preserves the evidence without overstating what the paper audit re-executed.

All three primary cohorts ran in GitHub Actions Ubuntu 24.04 with the pinned interpreter/packages. Recorded image versions include `20261116.133.1` and `20261116.376.1`; the jobs did not all use an identical image. Modified runs include richer CPU/build provenance and use heterogeneous Intel/AMD hosts; native/SGDM metadata does not record the CPU model. Do not infer identical kernels or hardware from matching version strings. Accumulated training-update times span about 590–1,425 seconds (9.8–23.8 minutes) across the 15 jobs. This timer excludes evaluation, spectral fitting, setup and archiving; end-to-end runtime and runtime on another CPU will differ.

A prior 500-update local replay matched code, package versions, settings and initialization spectra but differed from historical seed 1340 by validation CE -0.0052965 and weighted alpha +0.0287218. The specific numerical cause remains unproven. That discrepancy motivated five fresh GitHub modified runs. The historical seed, superseded recovery workflow 45373421406 and partial exploratory local workers do not enter primary statistics. The historical control table remains a descriptive comparison across recorded runtimes. In particular, the new native endpoint alpha mean is close to the historical mean while its across-seed SD is larger; agreement of means does not imply identical trajectories.

Karpathy’s contextual 2,000-update report gives approximate loss 1.88, implying character perplexity exp(1.88)≈6.55. The archived audit does not identify that loss split explicitly. The reference uses a 90/10 split, sampled evaluation, weight decay 0.1 and cosine decay ending at 2,000; the present experiment uses 90/5/5, full validation, selected zero decay and the fixed 20,000-update schedule. It is a contextual baseline, not a controlled performance comparison. Historical native/SGDM alpha at 2,000 was unrecorded and is never filled from the new cohort.
