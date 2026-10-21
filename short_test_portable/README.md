# Portable five-AdamW / five-momentum-SGD comparison

This separate runner retains the [comparison protocol](../short_test_optimizer_comparison/README.md): validation-only calibration, five paired confirmation seeds per optimizer, 20,000 updates, full train/validation/test metrics, all 18 unique matrix alpha fits, and measured timing. It imports the frozen model and data/evaluation definitions from `short_test/`. Keep all three directories in your checkout.

**CPU:** regular PyTorch on macOS or Linux. Apple Silicon uses its CPU here, not MPS. Intel Macs need a PyTorch/Python combination with available macOS x86_64 wheels; Python 3.11 and the requirements below allow pip to choose an older compatible PyTorch. We do not promise identical floating-point results across PyTorch versions or CPU architectures.

**TPU:** Google's [TorchTPU](https://github.com/google-pytorch/torch_tpu), on a supported Google TPU VM. This is not a CPU emulator or the similarly named Sophgo package. The CPU wheel index in Google's install command supplies host dependencies; it does not make a Mac a TPU. The implementation uses native `tpu` tensors and the public `torch.accelerator` APIs, not `torch_xla`.

Status: Linux CPU contract tests passed. Mac hardware and real TPU hardware have **not** been tested here. TPU support is experimental until the preflight below passes on the target VM. Unsupported kernels or missing hardware fail explicitly; there is no fallback to CPU training. This eager implementation prioritizes comparable semantics, not TPU throughput. Small batches, frequent scalar checks, and host transfers may be slow on TPU.

## MacBook / Linux CPU setup

From the repository root, with Python 3.11 installed:

```bash
python3.11 -m venv .venv-short-cpu
source .venv-short-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r short_test_portable/requirements-cpu.txt
python short_test_portable/test_portable.py
python short_test/download_data.py --output data/tinyshakespeare/input.txt
python short_test_portable/comparison.py prepare --device cpu --output runs/gpt_cpu --data data/tinyshakespeare/input.txt
python -m pip freeze > runs/gpt_cpu/requirements-resolved.txt
python short_test_portable/comparison.py run --device cpu --output runs/gpt_cpu --data data/tinyshakespeare/input.txt
```

If no compatible torch wheel is available, stop and use a supported Python/OS environment; do not force an incompatible installation. No Linux `/sys/fs/cgroup` files are required on macOS. The memory admission check is only a host heuristic, not a guarantee that training fits in RAM.

## Google TPU setup

Use a separate Python 3.11+ environment on the TPU VM. The install command follows Google's README as checked on 2026-10-21. Record the resolved versions; APIs and wheel compatibility may change.

```bash
python3.11 -m venv .venv-short-tpu
source .venv-short-tpu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r short_test_portable/requirements-tpu.txt
SHORT_GPT_DEVICE=tpu python short_test_portable/test_portable.py
python short_test/download_data.py --output data/tinyshakespeare/input.txt
python short_test_portable/comparison.py prepare --device tpu --output runs/gpt_tpu --data data/tinyshakespeare/input.txt
python -m pip freeze > runs/gpt_tpu/requirements-resolved.txt
python short_test_portable/comparison.py run --device tpu --output runs/gpt_tpu --data data/tinyshakespeare/input.txt
```

The preflight exercises both optimizers, forward/backward, clipping, checkpoint save/reload, tied weights, full-tail evaluation, and resume comparison. Passing a short preflight is necessary but does not replace auditing complete runs. TPU resume comparisons use tolerances; CPU resume must be exact. WeightWatcher always runs on CPU; its separate dependency must remain version 0.7.7.

## Reproduction and resume

Use a **new output directory** for this portable runner. Do not point it at the active original experiment or migrate an existing CPU run onto TPU. Its own source, backend, platform, Python and package versions are frozen into `design.json`; incompatible resumes fail. Source changes also require a new study directory. Existing source files and live experiments remain unchanged.

Re-run the same `run` command to resume. Initialization and minibatch sampling occur on CPU for matching seeds across optimizer groups; zero dropout is enforced. The model is transferred before optimizer construction, with token/output tying explicitly retained. CPU optimizer/RNG checkpoint payloads and best weights are saved every 250 updates. This zero-dropout architecture has no stochastic accelerator operation, so there is no TPU RNG stream to resume. Adding dropout or random accelerator operations would require a new RNG contract.

TPU synchronization brackets each timed training update, including data transfer and any compilation costs incurred during it. There is no autocast or bfloat16 conversion; tensors are float32, but TPU kernel arithmetic can differ from CPU. Results from different backends must be labeled and analyzed separately, not pooled as identical replications.

```bash
python short_test_portable/comparison.py summarize --device cpu --output runs/gpt_cpu
python short_test_portable/comparison.py package --device cpu --output runs/gpt_cpu --data data/tinyshakespeare/input.txt
```

Substitute `--device tpu` and its output directory for the TPU study. Keep `requirements-resolved.txt`, design, selected recipes, full checkpoints, and generated package files. Outputs retain endpoint versus validation-selected metrics, paired seed statistics, and timing. The previously exposed test corpus remains disclosed; it never controls calibration. A 20,000-step cap does not establish convergence, and this bounded search does not prove globally optimal hyperparameters.

Public API reference: https://github.com/google-pytorch/torch_tpu/blob/main/docs/public_apis.md
