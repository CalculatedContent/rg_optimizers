# Preparation validation — 2026-11-25

**155 tests passed, 0 failed, 0 skipped** for this modified-AdamW
sister experiment on the pinned CPU stack, in **239.501 seconds**.
Both experiments also passed **312 non-spectral-integration
checks together** in 18.787 seconds. All 49 original SGDM files
match their recorded source hashes.

These are software and small synthetic-training checks. They do not establish
FineWeb convergence, optimal hyperparameters, TPU throughput or TPU execution.
No production TPU training or GCS operation was performed. Result publication
was tested with an in-memory Git object service, not a live results branch.

## Executed checks

| Area | Tests | Evidence covered |
|---|---:|---|
| One-command entry point | 14 | Isolated pinned environment, 150 GiB startup/30 GiB run gates, resume identity, sanitized child Python environment |
| Automatic campaign | 18 | 9→3→1 selection, verified target, one bounded fresh 2× attempt, honest unsuccessful outcomes, retry and source/publication gates |
| Automatic native integration | 1 | Actual two candidates, automatic selection and fresh finalist, incremental/final archive through simulated GitHub transport |
| Configuration and selection | 17 | Nine native-LR/reference-ratio candidates, immutable optimizer contract, equal budgets, disjoint holdouts, complete diagnostics |
| Launch and backup | 17 | No overwrite, verified resume, remote paths, interpreter identity, cloud-failure boundaries, checkpoint-pair races |
| Spectral measurements | 14 | 72/74 production matrix inventory, independent raw/clipped fits, exact fitted-tail weights, ERG formula, atomic exports and queue pruning |
| Modified AdamW and TPU gates | 40 | Canonical graft/native AdamW/native SGDM comparisons, changed/zero tensor LRs, decay, missing gradients, counters, scope, state restore and simulated XLA telemetry |
| Incremental publisher | 23 | Full native fields, immutable steps, remote Git hashes, concurrency, stale/partial output rejection, bounded publication backlog |
| Single-run native integration | 1 | Actual detached supervisor, trainer, native WeightWatcher worker and JSON/CSV/PNG/PDF generation |
| Training and recovery | 10 | Continuous token prefix, FP32/FP64 mean-gradient accumulation, exact validation slice, target guards, bit-exact CPU weights/moments/counters resume |

The separate combined run collects both folders in one Python process, including
their optimizer, trainer, configuration and publication tests. The native
WeightWatcher integration tests are excluded only from that additional coexistence
run; both new native integrations are included in the complete 155-test suite.
This avoids treating isolated imports as evidence that the folders coexist.

## Optimizer equivalence and recovery

The pinned canonical graft source matches SHA256
`e6c9fcf0e750e0efda8b716410623eff5e3f590751d5a345482e97a4b10c45a3`.
Comparisons cover FP32 and FP64, ratios 100/400/1000, changing and zero learning
rates, missing gradients, zero and nonzero decay. The independent native-SGDM
comparison verifies the target norm, while native AdamW verifies the adaptive
direction and nongrafted updates. Both moments and per-parameter counters match
the canonical implementation. Scope tests verify 48 physical block matrices and
one shared embedding/head parameter.

Checkpoint continuation is bit-exact on CPU for model tensors, Adam first/second
moments, counters, data position, RNG state, next LR factor and validation NLL.
Configuration changes, contract changes and corrupted checkpoint bytes are rejected.
This does not claim bitwise TPU behavior.

Mean-gradient accumulation is checked in both FP32 and FP64. FP32 reductions in
different batch shapes can perturb a near-zero embedding gradient; Adam's epsilon
denominator amplifies that difference. The test checks gradients and moments
directly and allows at most 1e-6 absolute weight roundoff in FP32, with 1e-12 in
FP64. Canonical-optimizer parity has its own tighter per-step tolerances.

The port uses tensor learning rates, tensor step counters and tensor bias
corrections. Its CPU dispatch test records no host scalar extraction. State
relocation is exercised with CPU/meta devices. The preflight's recompilation and
scalar-extraction refusal cases use explicitly simulated XLA telemetry. **The
actual five-update full-model TPU preflight has not run.** It must pass on the
target host before production training; CPU tests cannot certify TPU lowering,
graph reuse, memory use or runtime performance.

## Real automatic CPU campaign

The campaign used a one-block, width-24, two-head model, context 32, vocabulary 64,
and synthetic tokens. It ran two candidates and selected a fresh finalist.

| Run | Training tokens | Measured steps | Verified final artifacts |
|---|---:|---|---:|
| `screen/graft-smoke-lr0001` | 64 | 0, 2 | 27 |
| `screen/graft-smoke-lr0003` | 64 | 0, 2 | 27 |
| `full/graft-smoke-lr0001` | 128 | 0, 4 | 27 |

All three runs used `TensorLrAdamWSGDMGraft`, with four physical grafted matrices.
Each of the six snapshots retained six logical transformer matrices and two
embeddings, including every native raw and `clip_xmax` field. Every run published
initialization incrementally before its final report. Measurement JSON matched
the committed local bytes. Final archives contained per-step and aggregate CSV,
JSON reports and PNG/PDF plots. All declared snapshots completed with zero
pending/failed jobs; validity flags are recorded in the machine-readable evidence.
The finalist PNG was visually inspected: its four loss/perplexity/alpha/ERG panels
were readable and its title prominently labels the synthetic smoke setting.

The optimizer, training, native WeightWatcher fits, selection, publisher and
report generator were real. Only the GitHub transport was a test double. No live
results branch was created by the test. A completed-campaign resume issued no
training or publication request. The outcome was explicitly
`smoke_completed_not_a_benchmark`, with target attainment and success both false.

## Real single-run CPU integration

The separate two-block smoke model used width 32, context 32 and vocabulary 256.
Two optimizer updates consumed 128 synthetic tokens. Its eight physical grafted
matrices produced 12 logical transformer-matrix rows plus two embeddings per
snapshot, at steps 0 and 2. Training and tracking both exited 0, all required
snapshots completed and committed exports allowed temporary tensor snapshots to
be pruned. PNG and PDF reports were generated. Incomplete or mixed-run final
reports were rejected by explicit follow-up assertions.

Synthetic smoke data is categorically ineligible for the FineWeb target. None of
these curves is a production NanoGPT loss, perplexity or layer-metric result.

## Environment

| Dependency | Observed version |
|---|---|
| torch | 2.6.0+cpu |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| pandas | 2.2.3 |
| weightwatcher | 0.7.7 |
| powerlaw | 1.5 |
| matplotlib | 3.9.4 |
| pytest | 8.3.5 |
| Python | 3.12.14 |

The machine-readable [validation_evidence.json](validation_evidence.json) records
the test counts, actual run completion, metric inventory and hashes of the tested
source and requirements. [SOURCES.md](SOURCES.md) records the pinned canonical
optimizer and model/data provenance separately from the new search choices.

## Reproduce

From the complete repository checkout, in an isolated environment installed from
`requirements-cpu.txt`:

```bash
python -m pytest -q baseline/experiments/stock_gpt2_adamw_sgdm_tpu/tests
```

The integration tests keep the invoking process alive until their detached
workers finish. This also supports environments that terminate descendants when
a tool-command sandbox exits.

## Future hardware and production evidence

The one-command launch verifies the existing TPU/node topology, pinned numerical
environment, persistent disk and free space, all data checksums, causal attention,
finite full-model modified-AdamW updates, changed-LR graph reuse, native spectral
capability and live GitHub write/readback permission. An optional GCS destination
has a separate write/read/delete preflight.

The finite 9→3→1 search and optional fresh 2× finalist target NLL ≤3.28 on exactly
10,485,760 official validation targets. Final success also requires every declared
measurement and verified incremental/final publication. The campaign can report
an unsuccessful bounded search; it never certifies a global optimum from these
candidates or treats budget exhaustion as target attainment.

Repository-wide legacy CI is separate from this scoped validation. The unchanged
SGDM sibling's [validation record](../stock_gpt2_sgdm_tpu/VALIDATION.md) documents
the previously observed legacy schedule/configuration mismatches. This folder
does not alter those legacy files or claim that every repository-wide job passes.
