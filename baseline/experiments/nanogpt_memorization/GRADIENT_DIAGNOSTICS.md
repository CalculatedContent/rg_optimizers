# Diagnose a nonfinite gradient without restarting the experiment

The original baseline reports a nonfinite **total gradient norm**, not which
parameter has invalid gradients. It calls `clip_grad_norm_` before checking the
returned norm, so the gradients have already been modified when it raises.
Finite gradient entries can also overflow a float32 norm reduction. Neither
case alone establishes an AdamW-specific failure or an alpha/overfitting effect.

This update adds an isolated diagnostic replay. **It does not change `run.py`,
any optimizer, the LR schedule, gradient-clipping threshold, WeightWatcher
settings, or checkpoint fingerprints. It is not a claimed fix for the cause of
the reported MPS failure.** Old data remain usable and no new scientific result
is inferred from an exception.

## One command, from this directory and your original environment

```bash
python diagnose_gradients.py --condition rule_random --optimizer adamw --seed 1337
```

The command selects the newest incomplete matching run in timestamped
`/tmp/nanogpt_memorization_*` study directories. It prints the selected path and
checkpoint step. A concurrently active trainer is refused using its existing
file lock. An exact path can instead be supplied:

```bash
python diagnose_gradients.py --run-dir /tmp/your_study/full/repository/rule_random/adamw/seed_1337
```

It reads the original checkpoint, verifies its tensor hash against the saved
audit row, checks manifest/source/data identity, restores the optimizer and
recorded Torch RNG state, and replays **up to 400 optimizer updates**. This is
not 39,063 updates from initialization. With a saved step-8500 checkpoint, the
window includes the previously reported failure at zero-based step index 8833.
`--updates` can change the diagnostic window without changing the original LR
schedule or original run.

The native clipping calculation is unchanged on successful updates. On a
nonfinite norm, `error_if_nonfinite=True` prevents in-place clipping from
modifying the evidence. The replay then saves the original accumulated
gradients, current model, optimizer state, exact inputs and labels, current LR,
and per-parameter finite checks. It also calculates scaled CPU-float64 gradient
norms. When replaying an accelerator run, it recomputes only the failing update
on CPU from the same pre-update weights and records; it never substitutes CPU
gradients into the scientific run. Use `--no-cpu-check` to omit this diagnostic.

## Files and interpretation

New files are written under the selected run:

```text
diagnostics/gradient_replay_<timestamp>_<process-id>/
  report.json             # observed failure and per-parameter checks
  replay_trace.jsonl      # successful replayed updates, loss, LR, native norm
  failure_state.pt        # present only when a failure is observed
```

No original checkpoint, metric, spectrum, completion flag, or manifest is
rewritten. Diagnostic output is not a resumable training checkpoint and is not
included in the memorization analysis as a scientific run.

- `native_norm_nonfinite_with_finite_gradients`: the native reduction failed or
  overflowed despite finite entries; inspect the independently computed norm.
- `nonfinite_gradient_entries`: the report lists the affected parameters and
  NaN/Inf counts before clipping. This does not by itself identify the operation
  that first produced them.
- `forward_or_backend_error`: the replay failed elsewhere; preserve the exact
  exception and do not relabel an unrelated error as the original failure.
- `no_failure_in_replay_window`: no failure was observed within this window;
  not proof that the original problem is fixed.

A finite CPU backward pass and invalid MPS backward pass at the same tensors is
useful evidence for investigating numerical/backend differences, not sufficient
proof of a particular MPS bug. CPU and MPS need not be bitwise identical. The
original Python, Torch, NumPy, platform and device are checked by default. An
explicit `--allow-runtime-mismatch` records rather than conceals a changed
environment. Package versions are saved in the diagnostic report.

Replaying omits WeightWatcher/evaluation checkpoints and is labeled accordingly.
The original zero-dropout trainer's diagnostics are intended not to change
weights or RNG. A replay mismatch can still be informative and must not be
hidden. This utility refuses a changed `run.py` or changed pinned model,
optimizer, or recipe source; it supports the historical baseline, not the
separately versioned spectral-guard trainers.

## Validation

18 new CPU unit/integration tests cover finite-norm equivalence, preservation of
invalid gradients, finite-entry/overflow distinction, robust norm computation,
source identity, run selection, active-run rejection, a toy checkpoint replay,
failure artifacts and parent-file preservation, and same-state CPU backward.
The replay integration model in these tests is a controlled small network, not
nanoGPT performance evidence. No user checkpoint, original MPS failure, or
numerical WeightWatcher fit was available in the development environment.

API reference: PyTorch `torch.nn.utils.clip_grad_norm_` documents that it changes
gradients in-place, returns their total norm, and defaults
`error_if_nonfinite=False`. The upstream `torch/nn/utils/clip_grad.py` implementation
checks the norm before scaling when `error_if_nonfinite=True`.
