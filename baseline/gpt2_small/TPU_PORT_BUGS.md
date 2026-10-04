# TPU port bug log

Finding and isolating correctness/performance defects in the PyTorch-to-TPU port is
an explicit objective of this project. Preserve failures as evidence. A passing
CPU smoke test does not establish TPU correctness. Do not silently lower learning
rates, change data, or skip nonfinite checks to make a run appear successful.

## 2026-10-04: GPT-2 AdamW nonfinite result, followed by stalled failure reporting

Status: numerical cause open; blocking diagnostic implementation replaced.
Upstream attribution: **unconfirmed**. No upstream issue has been submitted.

- Run: `ww-gpt2-validation-48h-20261004-s1337`, original commit `4631a3d`.
- Machine: one v5litepod-8, eight chips, SPMD; project `tpu-builders-504820`,
  zone `us-west4-a`. Reported runtime: Python 3.10.12, PyTorch 2.6.0+cpu,
  PyTorch/XLA 2.6.0. The library's `+cpu` build string does not identify where
  the model computations executed; the run explicitly selected XLA/TPU.
- Model: GPT-2 Small, 124,439,808 parameters; 12 layers, 12 heads, width 768,
  context 1024, tied embeddings. FineWeb reused from the persistent disk.
- Before updates 1 and 2, reported losses were finite (approximately 11.01 and
  10.24) and aggregate gradient norms were 16.177856 and 7.746079.
- After approximately 2.5 hours, the four-update check had not completed.
  Latest checkpoint pointer: step 0. Process 6802 had roughly 212 GiB RSS.
- Saved Python and native stacks identify `require_finite_update`, line 141,
  at `p.grad.detach().float().cpu()`. This branch executes only after detecting
  a nonfinite loss or aggregate gradient norm. It does not reveal which scalar
  failed, the first affected matrix, or the numerical root cause.
- Native frames include `THPVariable_cpu` and tensor conversion. They confirm
  waiting in the host transfer path; they do not prove a hardware failure,
  compiler bug, deadlock, or that all training ran on CPU.
- Service subsequently confirmed `MainPID=0`, `ActiveState=inactive`,
  `SubState=dead`. TPU allocation and data remain available.
- Evidence on disk:
  `/mnt/disks/rg-data/gpt2small/ww-gpt2-validation-48h-20261004-s1337/diagnostics/stall-20261004-041944-714308`.

### Reporting defect and correction

The old failure handler copied full gradients to CPU, serially, before writing
the failure report. That code could stall and hide the already detected failure.
Commit `f887593` removed these copies, added an execution barrier before host
scalar reads, saved the scalar failure immediately, and bounded validation phases.
These changes have local test coverage; they have not established numerical
correctness on TPU. GPT-2 now has its own `baseline/gpt2_small` package.

### Next diagnostic and attribution criteria

Run four fresh AdamW updates on the same allocation, with the original model,
corpus, seed and optimizer hyperparameters. Save each completed update. Record:

- Source commit, Python/torch/torch_xla/libtpu versions, relevant XLA settings.
- Exact input-window offsets, corpus identities, initialization/rolling full states.
- Per-tensor finite flags and extrema before clipping, after clipping, and after
  the optimizer update; parameter names plus Adam moment names.
- XLA compilation/execution counters, fallback counters, stage timestamps,
  Python tracebacks and a structured failure report.

Checks reduce tensors on device and transfer a small summary table, never full
gradients. Additional synchronization is recorded as instrumentation: it can change
fusion/compilation behavior. A pass under instrumentation does not by itself clear
the original execution path. The diagnostic stops after four updates or 20 minutes;
up to 10 additional minutes are reserved for verified cloud backup. Disk evidence
remains if upload fails. No long experiment starts automatically.

To attribute an upstream bug, isolate the first failing operation and compare a
matched CPU/TPU replay with the same inputs, weights and optimizer state. Preserve
both results and a minimal reproducer before claiming a PyTorch/XLA defect.

## 2026-10-04: instrumented diagnostic aborts at its first gradient check

Status: open; fatal native message still required for diagnosis.

Further stack evidence localizes this abort to `port_debug.py:68`: the new
diagnostic's `torch.stack((isfinite(value).all().float(), value.amin(), value.amax()))`.
Native frames include `torch_xla::Stack::Stack`, `XlaNode::GetOpShape`, and
`XLANativeFunctions::stack`. The abort occurs while assembling diagnostic summaries,
before the first optimizer update. It therefore does not reproduce or explain the
earlier nonfinite-result failure. CPU tests passed this operation, but TPU behavior
has not passed validation. The preceding native assertion/status text is still
needed; the Python abort trace alone is insufficient to identify its cause.

- Run: `port-check-20261004-045633`, commit `19e2bb0`.
- Supervisor report: child exit code `-6` (SIGABRT), with last recorded stage
  `before_clipping_started`, update 1, Unix time `1791089838.8617651`.
- No per-tensor results from this check were reported. This abort does not by
  itself establish a nonfinite gradient, a particular failing operator, or an
  upstream runtime/hardware bug. It is a separate observed failure from the
  earlier numerical check and stalled host copy.
- Cloud backup was explicitly verified for this run. Logs, initialization,
  exact input-window offsets and environment metadata remain on disk and in
  its cloud prefix. The next evidence to inspect is the fatal native message
  immediately preceding the abort in `run.log`.
- A later launch was blocked by an untracked repository-root `FETCH_HEAD`
  file in Cloud Shell. This local checkout issue is separate from the TPU
  abort; moving that file outside the checkout preserves it and clears this
  particular cleanliness check. The real Git metadata is under `.git`.

### User-authorized MuonClip bypass, 2026-10-04

`run_muonclip.py` starts a separate continuous MuonClip experiment on the remaining
48-hour allocation. Its config sets `validation_tensor_checks=false` and
`validation_gradient_checks=false`, bypassing the crashing per-tensor stack and its
verbose validation path. `finite_update_guard=true` still checks scalar losses and
aggregate norm before the optimizer update; the existing numerical guard is not
removed. Model/data/optimizer and long-run LR settings are unchanged. Additional
synchronization, reporting and denser checkpoint/spectral measurements are explicit.

The workaround has CPU integration coverage; TPU success is not claimed. It is not
a repair or root-cause diagnosis for either SIGABRT or the earlier nonfinite result.
The earlier source checkouts, diagnostics, initialization checkpoints and verified
cloud archives are retained. New failures produce separate evidence in the new run.
No automatic restart, cleanup, disk formatting or TPU reallocation is performed.
