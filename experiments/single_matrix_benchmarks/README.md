# WeightWatcher single-matrix benchmark suite

**Status: prepared, not launched.** One training cohort at a time. Speech Commands completed separately as a multi-matrix control; California Housing is the first existing single-matrix experiment. No new training is authorized by this commit. The historical California cap and canceled cohorts remain untouched.

## Scientific goal
A single large FC2 matrix in a common d → 784 → 784 → k ReLU MLP, trained with SGDM and AdamW on a fixed real-world benchmark protocol. Evaluate the RG marginality hypothesis, including failures; never select models using alpha. Keep classification CE/accuracy distinct from regression RMSE/MAE in original units. A low performance error and alpha near 2 are not imposed as success conditions.

The main no-dropout comparison uses SGDM with zero weight decay and AdamW with zero weight decay (Adam-equivalent); the separate native AdamW reference uses nonzero decoupled decay 0.001. Compare identical data/splits and architecture. Do not attribute differences between regularized and unregularized tracks solely to optimizer. Dropout is a distinct, optional paired extension. For each affected FC2 use fixed r(p)=1-sqrt(1-p), 20 independent masks per checkpoint, report raw and corrected alpha with fit failures, and compare to the paired zero-dropout model rather than target 2. FC2 is the only primary matrix, even if side matrices are measurable. No forced exponent, no new weight-decay or dropout tuning to reach alpha 2.

## Ten candidates

| ID | Data | Rows | Inputs | Outputs | Status |
|---|---|---:|---:|---:|---|
| california_housing | [California Housing](https://github.com/CalculatedContent/rg_optimizers/tree/95cf2d49400c258588398383a9a041e5603e5837/california_housing_sgdm_20k/results/2026-11-26_sgdm_20k) | 20640 | 8 | 1 | existing_study |
| power_plant | [Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined%2Bcycle%2Bpower%2Bplant) | 9568 | 4 | 1 | prepared_not_launched |
| magic | [MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic%2Bgamma%2Btelescope) | 19020 | 10 | 2 | candidate_requires_data_and_reference_audit |
| dry_bean | [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry%2Bbean%2Bdataset) | 13611 | 16 | 7 | candidate_requires_data_and_reference_audit |
| protein | [Protein tertiary structure](https://archive.ics.uci.edu/dataset/265/physicochemical%2Bproperties%2Bof%2Bprotein%2Btertiary%2Bst) | 45730 | 9 | 1 | candidate_requires_data_and_reference_audit |
| kin8nm | [Kin8nm](https://github.com/yaringal/DropoutUncertaintyExps/tree/6eb4497628d12b0f300f4b4f6bdc386bebad565c/UCI_Datasets/kin8nm) | 8192 | 8 | 1 | candidate_requires_data_and_reference_audit |
| concrete | [Concrete compressive strength](https://archive.ics.uci.edu/dataset/165/concrete%2Bcompressive%2Bstrength) | 1030 | 8 | 1 | candidate_requires_data_and_reference_audit |
| wine_red | [Wine Quality Red](https://archive.ics.uci.edu/dataset/186/wine%2Bquality) | 1599 | 11 | 1 | candidate_requires_data_and_reference_audit |
| letter | [Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter%2Brecognition) | 20000 | 16 | 26 | candidate_requires_data_and_reference_audit |
| satimage | [Statlog Landsat Satellite](https://archive.ics.uci.edu/dataset/146/statlog%2Blandsat%2Bsatellite) | 6435 | 36 | 6 | candidate_requires_data_and_reference_audit |

The six candidates beyond the first four are proposals; their baseline/split provenance must be audited before execution. Mini Speech Commands is an additional independent multi-matrix study, not counted as a single-matrix benchmark. [Speech Commands SGDM archive](https://github.com/CalculatedContent/rg_optimizers/tree/5860c8fea053cf564b25de3a0d3b5890b573ae8a/experiments/mini_speech_commands_sgdm_20k/results); an AdamW comparison remains a separate planned extension. [California Housing dropout correction](https://github.com/CalculatedContent/rg_optimizers/blob/f09758902997320e9397317bd45a88d7fc3869cd/california_housing_sgdm_20k/dropout_correction/README.md).

## Benchmark acceptance gates

1. Pin authoritative data source, original dataset shape, feature/target schema, license, split indices/hashes, duplicates and group/time leakage checks. Preserve exact published folds when feasible. Never optimize splits against test performance.
2. Record literature baselines with method, metric, exact protocol, units, uncertainty, split comparability and source. Do not claim a global optimum or compare incompatible folds/ensembles as equivalent.
3. Freeze elementary development search, numerical environment, update ceiling, seeds, validation-only selection, and resource limits. Complete all settings or explicitly document failures. Preserve partial records.
4. Train the chosen track configurations on five independent seeds; evaluate test only after frozen selection. Record same-checkpoint weights, WW fit details, performance, spectra, source/data hashes and error bars. Save checkpoints, tables, plots and reproduction instructions to GitHub.
5. Compare the paired dropout extension only after review of the no-dropout experiment. An empirical correction can fail and must not be re-tuned on test data.

## First staged dataset
[Power Plant plan](power_plant/README.md). The first adapter, source data, trainer and analysis code are committed. They are prepared but have not executed a real training job. No training Actions workflow is installed. Other datasets receive their own verified adapters, not a blind reuse of this regression adapter.

References: [Gal and Ghahramani uncertainty benchmark and 20 outer splits](https://github.com/yaringal/DropoutUncertaintyExps/tree/6eb4497628d12b0f300f4b4f6bdc386bebad565c); [Grinsztajn et al., NeurIPS 2022 tabular benchmark](https://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract.html); [OpenML-CTR23 split protocols](https://www.openml.org/search?id=353&study_type=task&type=study). Use the relevant task version and folds when selecting later datasets, not headline best scores.
