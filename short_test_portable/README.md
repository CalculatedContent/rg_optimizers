# Portable five-AdamW / five-momentum-SGD comparison

This separate runner retains the [comparison protocol](../short_test_optimizer_comparison/README.md): validation-only calibration, five paired confirmation seeds per optimizer, 20,000 updates, full train/validation/test metrics, all 18 unique matrix alpha fits, and measured timing. It imports the frozen model and data/evaluation definitions from `short_test/`. Keep all three directories in your checkout.

Choose the backend explicitly for both `prepare` and `run`:

| Hardware | Switch |
| --- | --- |
| Any supported Mac/Linux CPU | `--device cpu` |
| Apple Silicon Mac GPU (Metal) | `--device mps` |
| NVIDIA GPU on Linux / WSL2 | `--device cuda` |
| Google TPU VM, using Google TorchTPU | `--device tpu` |

No automatic backend selection is used. Every worker inherits the selected backend. The ordinary `comparison.py` command uses one accelerator. For up to eight TPU devices, use `tpu_parallel.py` below. Use `CUDA_VISIBLE_DEVICES` to select the NVIDIA GPU before starting. Native Windows is not supported because the experiment uses POSIX file locks; use WSL2.

**CPU:** regular PyTorch on macOS or Linux. Use `cpu` for the Mac CPU or `mps` for its supported GPU. Intel Macs need a PyTorch/Python combination with available macOS x86_64 wheels; Python 3.11 and the requirements below allow pip to choose an older compatible PyTorch. We do not promise identical floating-point results across PyTorch versions or CPU architectures.

**TPU:** Google's [TorchTPU](https://github.com/google-pytorch/torch_tpu), on a supported Google TPU VM. This is not a CPU emulator or the similarly named Sophgo package. The CPU wheel index in Google's install command supplies host dependencies; it does not make a Mac a TPU. The implementation uses native `tpu` tensors and the public `torch.accelerator` APIs, not `torch_xla`.

Status: Linux CPU contract tests passed. Mac MPS, CUDA and real TPU hardware have **not** been tested here. TPU support is experimental until the preflight below passes on the target VM. Unsupported kernels or missing hardware fail explicitly; there is no fallback to CPU training. This eager implementation prioritizes comparable semantics, not TPU throughput. Small batches, frequent scalar checks, and host transfers may be slow on TPU.

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

## Apple Silicon GPU and NVIDIA GPU

For Mac MPS, use the CPU setup environment above and replace the device/output flags:

```bash
python short_test_portable/test_portable.py --device mps
python short_test_portable/comparison.py prepare --device mps --output runs/gpt_mps --data data/tinyshakespeare/input.txt
python short_test_portable/comparison.py run --device mps --output runs/gpt_mps --data data/tinyshakespeare/input.txt
```

Unset `PYTORCH_ENABLE_MPS_FALLBACK`; the runner rejects silent CPU fallback so timing remains meaningful.

For NVIDIA, create a separate virtual environment and install the CUDA-enabled PyTorch wheel recommended for your driver by https://pytorch.org/get-started/locally/ . Then install `requirements-cpu.txt` (already installed compatible torch is retained). Do not install the CPU-only torch wheel or TorchTPU into this CUDA environment.

```bash
python short_test_portable/test_portable.py --device cuda
python short_test/download_data.py --output data/tinyshakespeare/input.txt
python short_test_portable/comparison.py prepare --device cuda --output runs/gpt_cuda --data data/tinyshakespeare/input.txt
python -m pip freeze > runs/gpt_cuda/requirements-resolved.txt
python short_test_portable/comparison.py run --device cuda --output runs/gpt_cuda --data data/tinyshakespeare/input.txt
```

Float32 tensors are used on every backend, with CUDA TF32 disabled. Optimizers and gradient clipping use the scalar-list path (`foreach=False`) for portability. Kernel differences still prevent a promise of identical results across hardware.

## Google TPU setup

Use a separate Python 3.11+ environment on the TPU VM. The install command follows Google's README as checked on 2026-10-21. Record the resolved versions; APIs and wheel compatibility may change.

```bash
python3.11 -m venv .venv-short-tpu
source .venv-short-tpu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r short_test_portable/requirements-tpu.txt
python short_test_portable/test_portable.py --device tpu
python short_test/download_data.py --output data/tinyshakespeare/input.txt
python short_test_portable/comparison.py prepare --device tpu --output runs/gpt_tpu --data data/tinyshakespeare/input.txt
python -m pip freeze > runs/gpt_tpu/requirements-resolved.txt
python short_test_portable/comparison.py run --device tpu --output runs/gpt_tpu --data data/tinyshakespeare/input.txt
```

The preflight exercises both optimizers, forward/backward, clipping, checkpoint save/reload, tied weights, full-tail evaluation, and resume comparison. Passing a short preflight is necessary but does not replace auditing complete runs. Accelerator resume comparisons use tolerances; CPU resume must be exact. WeightWatcher always runs on CPU; its separate dependency must remain version 0.7.7.

## Reproduction and resume

Use a **new output directory** for this portable runner. Do not point it at the active original experiment or migrate an existing CPU run onto TPU. Its own source, backend, platform, Python and package versions are frozen into `design.json`; incompatible resumes fail. Source changes also require a new study directory. Existing source files and live experiments remain unchanged.

Re-run the same `run` command to resume. Initialization and minibatch sampling occur on CPU for matching seeds across optimizer groups; zero dropout is enforced. The model is transferred before optimizer construction, with token/output tying explicitly retained. CPU optimizer/RNG checkpoint payloads and best weights are saved every 250 updates. This zero-dropout architecture has no stochastic accelerator operation, so there is no TPU RNG stream to resume. Adding dropout or random accelerator operations would require a new RNG contract.

CUDA, MPS or TPU synchronization brackets each timed training update, including data transfer and any compilation costs incurred during it. There is no autocast or bfloat16 conversion; tensors are float32, but TPU kernel arithmetic can differ from CPU. Results from different backends must be labeled and analyzed separately, not pooled as identical replications.

```bash
python short_test_portable/comparison.py summarize --device cpu --output runs/gpt_cpu
python short_test_portable/comparison.py package --device cpu --output runs/gpt_cpu --data data/tinyshakespeare/input.txt
```

Substitute `--device tpu` and its output directory for the TPU study. Keep `requirements-resolved.txt`, design, selected recipes, full checkpoints, and generated package files. Outputs retain endpoint versus validation-selected metrics, paired seed statistics, and timing. The previously exposed test corpus remains disclosed; it never controls calibration. A 20,000-step cap does not establish convergence, and this bounded search does not prove globally optimal hyperparameters.

Public API reference: https://github.com/google-pytorch/torch_tpu/blob/main/docs/public_apis.md

## TPU Builders first test

On your allocated TPU VM, use the TPU environment above and run the preflight before spending credits on calibration. It prints the actual backend, versions, pass/fail status, and elapsed test time; any failure exits nonzero. It exercises both AdamW and momentum SGD including a saved optimizer-state resume. Save the terminal log with your experiment records. No TPU has been provisioned or charged by this repository update.

After preflight succeeds, use the `prepare` and `run --device tpu` commands above. This executes the bounded calibration first, then five paired seeds for each optimizer. The ordinary command uses one TPU device; use the parallel launcher below for an eight-device slice. Measure this small model's performance before choosing a larger allocation. Do not compare preflight seconds to full 20,000-update training time.

## Up to eight TPU devices / cores

The parallel launcher runs **independent trials on up to eight TPU devices on one host**. It preserves batch size 12 per trial and all scientific settings; it does not split a single model across eight devices. Google exposes different core/chip/device counts across TPU generations: `--cores` means active PyTorch TPU devices, not a guaranteed chip count.

From the repository root on the allocated TPU VM:

```bash
# First exercise all eight active devices, including optimizer resume.
python short_test_portable/tpu_parallel.py --cores 8 --preflight --output runs/gpt_tpu8 --data data/tinyshakespeare/input.txt
# Only proceed when preflight exits successfully.
python short_test_portable/tpu_parallel.py --cores 8 --output runs/gpt_tpu8 --data data/tinyshakespeare/input.txt
```

Use `--cores 1` through `--cores 8` to cap simultaneous trials, up to the allocated device count. The launcher uses Google's `torch.tpu.distributed.set_tpu_launch_env()` topology helper and starts one process per device with `torchrun`, initializing `tpu_dist` before device access. On an eight-device slice, `--cores 4` still initializes all eight processes as required by the slice topology, but only four perform trials. This does not reduce the cost of the allocated slice. Multi-host slices are outside this launcher's supported scope.

Calibration screening completes before finalist selection, all finalist training completes before recipes are frozen, and confirmation starts only afterward. Only rank zero writes shared selection and final summary files; each trial retains its own model/optimizer/RNG checkpoint and lock. Completed jobs are skipped on resume. A numerical failure remains explicit; an infrastructure error makes the launcher exit nonzero and does not count as completion. Re-run the same command to resume saved work. A `PAUSE` file in the output directory requests a checkpointed stop; remove it before resuming.

There can be eight concurrent screening or confirmation trials, but only four finalist calibration trials exist. Idle devices are expected at stage boundaries. Per-trial timing remains synchronized; final `launch_wall_seconds` measures this invocation, not prior interrupted sessions. Progress is recorded in `rankN_status.json` and per-trial `progress.json`; preflight writes `preflight_rankN.json` for each active rank.

CPU tests cover assignment of each job exactly once for every concurrency from one through eight, plus optimizer resume and backend checks. Actual multi-device TPU execution remains unverified until the Builders VM preflight succeeds. Source: https://github.com/google-pytorch/torch_tpu/blob/main/docs/distributed_support.md

## Audit and regression tests

Run from the repository root:

```bash
python short_test_portable/test_portable.py --device cpu
python short_test_portable/test_audit.py
python short_test_portable/test_distributed.py
```

The last command runs two CPU/Gloo processes through shortened calibration, selection and confirmation using synthetic data, then reruns to verify completed-job skipping. It uses real model updates/checkpoints/evaluation; alpha fitting is explicitly stubbed. It does not validate TPU kernels. If your development sandbox prohibits Gloo sockets, use `test_distributed.py --simulate` to exercise the same two-process workflow with file-based test barriers; this tests coordination logic, not actual distributed communication.

Verified 2026-10-21: 12 unit checks passed (5 portable + 7 audit), plus the two-process workflow and resume check with simulated barriers. Real Gloo communication was blocked by the execution environment (`Operation not permitted`). CUDA/MPS hardware was absent and TorchTPU was not installed. No accelerator or eight-device hardware pass is claimed.

The audit fixed final checkpoint persistence when a target falls between checkpoint intervals, explicit lock cleanup, and JSON tuple/list normalization so unchanged frozen recipes survive resume. Scientific settings are unchanged. All fixes are in the portable runner; the original frozen experiment source remains intact. Because source hashes are guarded, begin a new portable output directory when updating from an earlier revision.
