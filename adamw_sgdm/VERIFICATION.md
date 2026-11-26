# Verification record

Date: November 25, 2026. Environment: Linux x86-64, Python 3.12.14,
PyTorch 2.6.0+cpu. This record accompanies the reusable optimizer package;
the historical experiments have their own results and provenance.

## Automated numerical checks

From the repository root:

```bash
python -m unittest discover -s adamw_sgdm/tests -v
```

Final local result: **13 tests collected; 12 passed; 1 skipped**. The skip is
explicitly the CUDA hardware test because CUDA was unavailable. The run took
1.477 seconds. Checks include:

- Independent native AdamW directions and separate native SGD momentum norms,
  with FP32/FP64, per-layer ratios, schedules, zero LR, AMSGrad and ascent.
- Numerical parity with the original `short_test_adamw_graft/graft.py` wrapper.
- Separate weight decay, zero versus missing gradients, and momentum memory.
- Bitwise identical checkpoint continuation and rejection of unrelated optimizer
  state formats.
- Embedding/tied-weight handling and the exact 16-matrix Shakespeare selector.
- CPU BF16 autocast while preserving FP32 parameters and optimizer moments.
- Near-one and zero betas, norm precision, and epsilon representability.
- Sparse-gradient and changing-beta errors before partial updates.
- No host scalar extraction during `optimizer.step()`.

The CUDA test is implemented and automatically runs when real CUDA is
available. It covers CPU/CUDA numerical agreement, FP16 autocast with
GradScaler, unscaling before clipping, and overflow skips that preserve both
parameters and optimizer state.

## Runnable learning and checkpoint examples

| Local check | Observed result |
| --- | --- |
| CPU FP32, 100 updates, seed 42 | Synthetic training CE 1.384682 → 0.015211; accuracy 27.15% → 100% |
| CPU BF16 autocast, 30 updates | Synthetic training CE 1.384682 → 0.204655; accuracy 26.95% → 97.07% |
| CPU 40 updates plus resume to 100 | Entire checkpoint, including every model/optimizer tensor, exactly equals uninterrupted 100-update run |
| Checkpoint portability | Every saved tensor is on CPU |
| Example syntax and multicore `--help` | Passed without requiring a TPU runtime import |

The synthetic problem is intentionally small and deterministic. Its training
accuracy establishes a functioning example and does not establish benchmark
quality, generalization gains, or a particular spectral alpha.

## Hardware status

| Target | Status |
| --- | --- |
| CPU FP32 and BF16 autocast | Executed successfully |
| PyTorch/XLA 2.6, PJRT CPU backend | Executed successfully, including checkpoint continuation |
| CUDA GPU | Example and hardware test implemented; no GPU execution available here |
| Native TorchTPU | Example and setup instructions implemented; no TPU execution available here |
| Legacy PyTorch/XLA TPU | Single-device and single-host multicore examples implemented; no TPU execution available here |

The examples and runtime calls were checked against official PyTorch,
TorchTPU and the PyTorch/XLA 2.6 source. An available-hardware skip is not a
passed hardware test. On the target machine, use the README commands and verify
finite loss, successful updates and checkpoint continuation. Record the actual
backend versions and device count when adding hardware results.

## Additional XLA execution check on CPU

Environment: Python 3.11.18, `torch==2.6.0+cpu`, `torch_xla==2.6.0`,
`PJRT_DEVICE=CPU`. The XLA 2.6 distribution has no Python 3.12 wheel; the
README's legacy XLA installation therefore uses Python 3.11.

The actual XLA device path (`xla:0`) executed the lazy optimizer operations and
`xm.optimizer_step` on a CPU device. Ten updates reduced synthetic training CE
from 1.384682 to 1.031498. Saving and reloading the portable checkpoint restored
that loss, then continuing to update 20 reduced it to 0.503298 (accuracy 90.23%).
This exercised the device-resident counters, moment calculations, graft norms,
lazy execution boundary and state relocation after loading. It was not a TPU
hardware or multicore execution test.

Reproduce in an XLA 2.6 environment from the repository root:

```bash
PJRT_DEVICE=CPU python adamw_sgdm/examples/train.py --device xla --steps 10 \
  --checkpoint adamw_sgdm/runs/xla_cpu10.pt
PJRT_DEVICE=CPU python adamw_sgdm/examples/train.py --device xla --steps 20 \
  --resume adamw_sgdm/runs/xla_cpu10.pt
```

The scoped GitHub Actions workflow repeats the CPU unit/demo tests and this
XLA-on-CPU checkpoint check. It does not allocate GPUs or TPUs.
