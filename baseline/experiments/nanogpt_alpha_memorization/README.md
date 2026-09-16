# Alpha and memorization: ten paired nanoGPT runs

**Run five AdamW seeds and five spectrally guarded Muon seeds on one Mac MPS device.**
All five measurement views from the memorization checklist are evaluated in each
training run. No 80-run campaign, hand-written loop, shell option changes, or
manual result-directory substitutions are needed.

## Run

From your existing checkout, with your research Python environment activated:

```bash
cd /tmp/rg_optimizers_memorization
git switch main
git pull --ff-only origin main
cd baseline/experiments/nanogpt_alpha_memorization
caffeinate -dimsu python run_study.py run
```

The command prints a timestamped `/tmp/nanogpt_alpha_memorization_YYYYMMDD_HHMMSS`
results directory, performs two actual model/optimizer/WeightWatcher smoke runs,
then queues exactly **10 scientific runs sequentially**. A smoke failure stops
before the campaign. An individual scientific failure is preserved and reported;
other independent seeds are still attempted. Nothing silently retries a failed
scientific run with changed hyperparameters. A nonzero final exit status means
at least one requested run failed, not that all ten succeeded.

Dependencies are those of the existing `baseline/nanogpt_one_head[dev]` package,
including **WeightWatcher 0.7.7**, PyTorch, NumPy, pandas, SciPy, matplotlib,
nbformat and nbclient. Do not upgrade a running experiment's environment. For a
new environment only, from repository root:

```bash
python -m pip install -e './baseline/nanogpt_one_head[dev]'
```

Preview the queue with `python run_study.py plan`. To deliberately resume the
same study, use `python run_study.py run --resume`; complete seeds are skipped,
partial seeds use the last accepted checkpoint, and fingerprints must match.
No original memorization result or source file is deleted. Copy results out of
`/tmp` before an OS cleanup or reboot. Runtime on MPS has not been benchmarked;
do not assume all ten will finish overnight.

## Analyze and export for review

```bash
python run_study.py report
python run_study.py export
```

`report` reads the saved results without retraining. It produces `report/summary.md`,
CSV tables and PNG/SVG plots. It works on partial studies and prints the complete
run count rather than silently discarding failures.

`export` also executes the two **analysis-only** notebooks and creates a new
`review/<study_timestamp>_<export_timestamp>/` directory in this source folder.
It includes executed notebooks with embedded figures, PNG/SVG files, CSV tables,
protocol, and run manifests. It excludes checkpoints, arbitrary local files and
credentials. It does not run the launcher notebook or push anything to GitHub.
These are ordinary files ready for a review PR:

```bash
git add review/
git commit -m "Add measured alpha-memorization results for review"
```

An exact study can be selected using `--root /tmp/the_printed_directory` for
run/resume/report/export. `notebooks/00_Run_All_10.ipynb` contains a single
notebook shell cell for the launcher. `01_Memorization_Results.ipynb` and
`02_Spectral_Boundary.ipynb` can be opened normally in Jupyter.

## What is compared

Five fixed training seeds: **1337, 2027, 4099, 31415, 271828**. Both arms use the
same seed-specific initialization, examples, and presentation order.

- **AdamW:** ordinary source-backed AdamW, with no alpha objective. Entry into
  alpha below two is an observed result, never forced by relabeling, reseeding,
  extending only successful runs, or selecting favorable checkpoints.
- **Muon guarded:** the source Muon update on hidden transformer matrices, plus
  its auxiliary AdamW. Every 100 accepted updates the controller measures all
  **eight trainable 2-D tensors**, including token and position embeddings. The
  tied token embedding/output head is counted once. A window is accepted only
  if **both raw and clip_xmax alpha are at least 2.05 for every matrix**.
  Otherwise it restores the model, optimizer state, RNG and accepted exposure
  counts, halves the offending matrices' LR multipliers, and retries the same
  examples. All rejected spectra and attempted-update counts are retained.
  After eight retries it fails rather than relax the threshold.

This is **guarded Muon**, not a claim that plain Muon naturally keeps alpha above
two. The guard changes effective step sizes and compute cost. Differences from
AdamW are recipe/intervention differences, not by themselves proof that alpha
causes overfitting. A direct AdamW+guard and matched nonspectral regularization
ablation would be appropriate follow-on work, not silently added to these ten.

**Scope of "all alpha > 2":** all measured trainable matrices at accepted
measurement boundaries. We do not claim no crossing happened between them.
`--guard-every 1` requests measurement after every update, with much greater
cost. It changes the protocol and must be used consistently for both arms from
a fresh root. Undefined/nonfinite fits fail the gate; they are not set to 2.05.
If initialization fails the gate, that seed remains an explicit failure; there
is no search for a seed whose spectra pass.

## Model and objective

This is a **new versioned synthetic experiment**, not a resume or silent change
to the earlier FineWeb or `nanogpt_memorization` runs.

| Component | Value |
|---|---|
| nanoGPT | 1 block, 1 head, width 128, context capacity 256 |
| Synthetic vocabulary | **512** (explicit reduction from the 50,257-token parent vocabulary) |
| Objective | Mean conditional suffix loss per example; prompts are not targets |
| Effective batch | 32 examples; equal-example weighting across suffix lengths |
| Fixed training horizon | **10,000 accepted updates**, canaries withdrawn after 5,000 |
| AdamW | Source LR 0.0006 to 0.00006, betas (0.9,0.95), epsilon 1e-8, decay 0.1 |
| Muon | Source matrix LR 0.02 to 0.002, momentum 0.95, 5 Newton-Schulz steps; auxiliary AdamW unchanged |
| Source LR schedule | 9,766-update warmup/cosine geometry, then nonzero floor |
| Attention | Explicit FP32 scaled matmul, causal mask, softmax on CPU/MPS/CUDA |
| Spectra | Every 100 accepted updates, plus initialization/final |
| Behavior | Every 500 updates, plus initialization/withdrawal/final |
| Exhaustive exposure and prefix search | Every 2,500 updates, plus initialization/final |

`protocol.json` pins parent model/optimizer/recipe Git blob identities. Source
profiles are starting settings, **not a qualified optimum on the new objective**.
`--steps` is an explicit new-protocol override, not a hidden adaptive horizon.
The vocabulary reduction and suffix-only output projection avoid calculating
large unused softmaxes at padding/prompt positions. There is no left-padding
in training. Changing these features means old and new losses are not directly
comparable. The attention equation is unchanged, but its floating-point path is
explicit rather than fused; this is not proof that fused MPS caused the old crash.

## One training distribution, all requested measurements

A fixed modular-addition problem modulo 31 provides 480 training operand pairs,
240 validation pairs and 241 test pairs. Exactly 25% of training-pair labels are
replaced once with fixed independent random labels. A random replacement may
coincide with the true label by chance. Clean training examples, assigned noisy
labels, true labels on those same noisy inputs, clean validation and final-only
clean test predictions are scored separately. The test never selects a model,
seed, horizon or guard parameter.

Mixed into that training stream are independent random 64-token-prefix / 32-token-
suffix canaries, 16 per lifetime dose **0/1/4/16/64**. A separate cohort has four
3-token canaries per dose, sampled uniformly from a fixed 16-token alphabet.
Exact injection slots are frozen. They replace ordinary examples rather than
adding more updates. Counts refer to actual accepted presentations, not intended
copies in a token file. Rejected-window compute/presentations are reported as
attempted work and do not enter the accepted model trajectory.

| Measurement | Implementation and limit |
|---|---|
| Verbatim extraction | Full 32-token greedy exact recall; first observed exact checkpoint and dose |
| Partial recall | Free-running token match, longest correct prefix; teacher-forced accuracy kept separate |
| Canary exposure | **Exhaustive** summed conditional NLL ranking of all 16^3=4096 short suffixes; tie bounds, conservative primary exposure, maximum 12 bits; save every candidate score |
| Compression | Shortest successful prefix in 8/16/32/64 grid; fixed target/absolute position; failures censored, not assigned zero ratio; restricted prefix compression, **not an optimized adversarial attack** |
| Likelihood/perplexity | Per-example suffix NLL and exp(NLL), seen/zero-dose contrasts, separated by cohort/target length |
| Noise fitting/generalization | Assigned-label versus true-label fitting; clean held-out loss/accuracy; do not call a corrupted-train/clean-test loss difference a conventional gap |
| Membership proxy | Same-length dose-vs-zero NLL ROC-AUC and initialization-corrected AUC; not an externally validated privacy attack |
| Retention | Acquisition and post-withdrawal trajectories; no automatic monotonicity or half-life assumption |

Compression passes the known absolute position offset as side information. It
never appends target tokens to the prompt. Formal exposure concerns the short
finite-universe secrets only, not the unrestricted 32-token strings.

## Spectral interpretation and scientific claims

The single WW call uses `fix_fingers='clip_xmax'`, `ERG=True`, `randomize=True`.
Every returned column is retained, alongside raw/clipped alpha, matrix identity,
checkpoint hash, fit distance and available randomized-control diagnostics.
A predeclared **screen**, not a significance test, marks tail support of at least
20 eigenvalues and D <= 0.2. Random-like spectra still require inspection of the
ESD/randomized controls before interpreting a power-law exponent physically.
The numerical guard must not be mistaken for proof of correlated heavy tails.

The reports retain all seeds, distinguish numerical crossings from screened fits,
and flag incomplete runs. A finite fitted alpha alone is not memorization, and
memorization alone is not proof of harmed generalization. The desired below-two
AdamW outcome may not occur. Report that negative result rather than change the
protocol after inspecting the held-out test.

Error bars are **pointwise Student-t 95% intervals across training seeds**, after
averaging probes within each seed. A single seed has no estimated interval.
Canaries, layers and checkpoints are not independent model replications. Paired
contrasts use matching seed/data/initialization/runtime/source and common update
indices. Final contrasts include complete pairs only and report n. Multiple
endpoints/plots are exploratory, not simultaneous significance tests. Do not
claim causation or a universal alpha=2 law from this two-arm design.

## Numerical failures and existing results

Before gradient clipping, nonfinite entries are rejected without modification.
Finite-entry norm overflow is recomputed using scaled CPU-float64 norms; this
fallback is logged. Actual invalid gradients/parameters save diagnostic evidence
and stop that run. No NaN replacement, batch skipping, or unannounced LR change
is used to make AdamW complete. MPS has not been validated by CPU-only tests.

The separate historical `diagnose_gradients.py` fix in this PR recovers a missing
optimizer name by matching the saved complete profile, without changing the
saved manifest fingerprint. It does not diagnose the cause of the old failure.

## Outputs and review

Each arm/seed directory contains source/runtime manifest, target inventory,
exact presentation schedule, safe rolling checkpoint, periodic permanent model
states, behavior JSON files, accepted and rejected spectral CSVs, control events,
raw short-canary candidate-score arrays, and a completion or failure record.
Reports include per-probe tables, zero-dose contrasts, first observed recall,
compression censoring, exact exposure, membership proxies, eight separate matrix
trajectories, and paired seed differences. The review export contains small
analysis artifacts only. No measured scientific results are shipped in advance.

## Methodological references

- Carlini et al., *The Secret Sharer*, arXiv:1802.08232 (formal canary exposure).
- Carlini et al., *Quantifying Memorization Across Neural Language Models*, arXiv:2202.07646 (context and repetition).
- Schwarzschild et al., *Rethinking LLM Memorization through the Lens of Adversarial Compression*, arXiv:2404.15146 (distinguish our restricted prefix test).
- Martin and Mahoney, *Implicit Self-Regularization in Deep Neural Networks*, JMLR 22 (2021), paper 20-410 (spectral interpretation).
