# Sources and provenance for the modified AdamW FineWeb experiment

Prepared 2026-11-25. Source evidence, transferred settings and new experimental
choices are distinguished below. No FineWeb optimum or target-reaching modified
AdamW result is inferred from the existing Shakespeare runs.

## Exact existing modification

The canonical GPT graft is
[`short_test_adamw_graft/graft.py`](https://github.com/CalculatedContent/rg_optimizers/blob/98891bed61a816277b7a92f296e44173704f5910/short_test_adamw_graft/graft.py),
Git blob `e353cd7a91625b6152882a925bdc16929ab2ae14`, SHA256
`e6c9fcf0e750e0efda8b716410623eff5e3f590751d5a345482e97a4b10c45a3`.
Its existing [tests](https://github.com/CalculatedContent/rg_optimizers/blob/98891bed61a816277b7a92f296e44173704f5910/short_test_adamw_graft/test_graft.py)
verify retained adaptive direction and SGDM norm matching.

The modification keeps Adam moments/bias correction and uses the uncorrected
SGDM reference `m/(1-beta1)`. It matches one Frobenius norm per physical transformer
weight; packed QKV remains joint. Embeddings, normalization parameters, biases
and the tied head retain native AdamW equations. Decoupled decay stays separate.
The local optimizer ports those equations to tensor LR/step/bias-correction
operations suitable for XLA; the CPU wrapper's `.item()` calls are not copied.

The latest [Shakespeare reference-rate study](https://github.com/CalculatedContent/rg_optimizers/blob/98891bed61a816277b7a92f296e44173704f5910/short_test_adamw_graft/results/2026-11-23_reference_lr_seed1340/README.md)
records native LR 0.001 and selected reference LR 0.4, ratio 400, with beta1 0.9,
beta2 0.99, epsilon 1e-8 and zero weight decay. Ratio 100 was the earlier comparator.
These are Shakespeare observations, not FineWeb hyperparameter guarantees.

The earlier [MNIST implementation](https://github.com/CalculatedContent/rg_optimizers/blob/98891bed61a816277b7a92f296e44173704f5910/experiments/adamw_modified/2026-11-13_mnist_mlp_experiment01/frozen/study.py)
uses the same magnitude-only idea with experiment-specific layer reference rates,
beta2 and decay matching. Those MNIST-specific choices are not silently imported
into the GPT experiment.

## Benchmark model, data and endpoint

- [Modded NanoGPT benchmark](https://github.com/KellerJordan/modded-nanogpt): the
  GPT-2-small quality threshold is FineWeb mean validation cross-entropy ≤3.28,
  evaluated over the specified first 10,485,760 validation tokens. The corresponding
  perplexity is `exp(3.28)`. Current modified record architectures and CUDA timings
  differ from this stock GPT-2 TPU experiment.
- The repository's pinned upstream model source at
  `baseline/gpt2_small/speedrun30/vendor/llmc_train_gpt2.py` has SHA256
  `757d0cea0d48cbc4c7d7d70371f955d49cdf3a7cfb4c87701a720c2fe0905c34`.
  Its source/license remain in the full checkout; the 124,439,808-parameter tied
  architecture is unchanged.
- The tokenized data manifest at `baseline/gpt2_small/speedrun30/data_manifest.json`
  has SHA256 `028141b267e9a6c99af8beea42bed58fe1d7418bdf034d630f1792aeb6a20e19`.
  It pins `kjj0/fineweb10B-gpt2` revision
  `889765ea1f903759787add96995d81171b632d0c` and every shard checksum.
- [Current record rules](https://github.com/KellerJordan/modded-nanogpt#rules)
  additionally require comparable-hardware timing and repeated-run statistical
  evidence for a new speed record. A single-seed TPU target hit is a recorded
  experiment result, not a certified leaderboard submission.

## Inherited engine and new choices

The sister folder derives from `baseline/experiments/stock_gpt2_sgdm_tpu` at source
commit `98891bed61a816277b7a92f296e44173704f5910`.
[REUSE_PROVENANCE.json](REUSE_PROVENANCE.json) records inherited file SHA256 values.
Package-relative imports, separate optimizer/configuration identities and separate
result destinations prevent accidental mixing with the SGDM experiment. The old
SGDM files are unchanged.

The following are explicit new design choices: native-LR grid (0.0003,0.001,0.003),
reference-ratio grid (100,400,1000), fixed batch64/microbatch16, 9→3→1 selection,
10% warmup/cosine horizons, one optional fresh twice-budget attempt, and a maximum
planned total of 35,194,404,864 training tokens. They define a finite search;
none is attributed to a published optimal FineWeb graft recipe.

The pinned WeightWatcher metrics and formulas, exact validation pairing,
continuous data prefix, checkpoint recovery and incremental archive contract
are retained from the SGDM experiment. Results must still be measured and archived
for this optimizer separately.

## Runtime and publication interfaces

- [PyTorch 2.6 Adam implementation](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/adam.py)
  and [AdamW](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/adamw.py)
  define the native moment, bias-correction and decoupled-decay conventions used
  for the adaptive direction and nongrafted parameters.
- [PyTorch/XLA 2.6 metrics](https://github.com/pytorch/xla/blob/v2.6.0/torch_xla/debug/metrics.py)
  exposes `CompileTime` sample counts for the actual hardware preflight.
- [GitHub references](https://docs.github.com/en/rest/git/refs),
  [trees](https://docs.github.com/en/rest/git/trees) and
  [blobs](https://docs.github.com/en/rest/git/blobs) provide incremental commit
  publication, non-force branch updates and remote blob verification.

[VALIDATION.md](VALIDATION.md) states which local checks ran and which hardware
and convergence checks remain for the future TPU session.
