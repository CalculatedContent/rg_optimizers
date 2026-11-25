# AdamW with the SGD-momentum magnitude graft

`AdamWSGDM` is a reusable PyTorch optimizer containing the modification used in
our MNIST/Fashion-MNIST and Shakespeare experiments: **keep AdamW's adaptive
direction and replace the selected tensor's step length with the same-history
SGD-momentum step length**. It supports CPU, CUDA GPU, native TorchTPU, and the
existing PyTorch/XLA TPU runtime through ordinary PyTorch tensor operations.

The package is independent of the experiment runners and needs only PyTorch.
The original experiments remain in their existing folders. See
[PROVENANCE.md](PROVENANCE.md) for their exact settings and
[VERIFICATION.md](VERIFICATION.md) for what has actually been run.

## Quick start on CPU

Commands below start at the repository root. The package requires Python 3.10
or later. Use Python 3.12 for the CPU/CUDA/native TPU examples and Python 3.11
for the legacy XLA 2.6 wheels, as shown in that section.

```bash
git clone --branch adamw-sgdm-portable-20261125 https://github.com/CalculatedContent/rg_optimizers.git
cd rg_optimizers
python3.12 -m venv .venv-adamw-sgdm
source .venv-adamw-sgdm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps --no-build-isolation -e ./adamw_sgdm
python adamw_sgdm/examples/train.py --device cpu --steps 100
```

If you already have a working PyTorch environment, install just this package
with the `--no-deps` command above. It preserves your chosen accelerator build.
PyTorch 2.6 is the verified baseline, not a requirement to downgrade a newer
working environment. The declared dependency is `torch>=2.6`.

The example trains a small MLP on deterministic synthetic classification data.
It downloads no datasets and reports its training loss and accuracy. These
numbers check that the implementation learns; they are not MNIST/Shakespeare
results or a comparison of generalization performance.

## Use in your own training loop

```python
import torch
from rg_adamw_sgdm import AdamWSGDM, parameter_groups

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)  # Keep parameter storage in FP32.

optimizer = AdamWSGDM(
    parameter_groups(model, weight_decay=0.01, reference_lr_ratio=100.0),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
)

optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(inputs.to(device)), targets.to(device))
loss.backward()
# Optional clipping belongs here, before the optimizer sees the gradients.
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

At `lr=0.001` and `reference_lr_ratio=100`, the reference SGDM rate is `0.1`.
Changing the Adam LR with a scheduler automatically changes the reference LR
by the same factor. The ratio sets the graft's magnitude; tune it for the model
and task rather than assuming one ratio is universally optimal.

`parameter_groups(model)` grafts trainable tensors with two or more dimensions,
excluding embedding weights and parameters tied to embeddings. It applies
weight decay to tensors with two or more dimensions and zero decay to vectors
and scalars. Ordinary vectors, biases, and embedding weights retain the AdamW
direction **and magnitude**. Inspect the returned `param_names` and `graft`
fields to see the selection.

For custom grouping, use the usual PyTorch dictionaries:

```python
optimizer = AdamWSGDM([
    {"params": [model.fc1.weight], "graft": True, "reference_lr_ratio": 40.0},
    {"params": [model.fc2.weight], "graft": True, "reference_lr_ratio": 200.0},
    {"params": [model.fc3.weight], "graft": True, "reference_lr_ratio": 800.0},
    {"params": [model.fc1.bias, model.fc2.bias, model.fc3.bias],
     "graft": False, "weight_decay": 0.0},
], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
```

These layer ratios reproduce the original MNIST **magnitude choices**. Its
sample-normalized decay needs the separate recipe in [PROVENANCE.md](PROVENANCE.md).

Passing `model.parameters()` directly is also supported: by default, tensors
with `ndim>=2` are grafted and all parameters receive the chosen decay. A bare
parameter iterator cannot identify embeddings; use the helper when exclusions
are desired. `graft=True` or `graft=False` forces the choice for a whole group.
Convolution tensors use one norm over the entire tensor, a generalization of
the original per-matrix rule.

## GPU

Use an existing CUDA-enabled PyTorch installation, or create a separate
environment and choose the build matching your GPU and driver from the
[official installation selector](https://pytorch.org/get-started/locally/).
A reproducible PyTorch 2.6/CUDA 12.4 setup is:

```bash
python3.12 -m venv .venv-adamw-sgdm-cuda
source .venv-adamw-sgdm-cuda/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-deps --no-build-isolation -e ./adamw_sgdm
python adamw_sgdm/examples/train.py --device cuda --steps 100
python adamw_sgdm/examples/train.py --device cuda --amp bf16 --steps 100
```

The BF16 example requires a GPU that supports BF16. CUDA FP16 is available with
`--amp fp16`; the example uses `torch.amp.GradScaler("cuda")`. Keep the model in
FP32 and apply autocast to the forward pass. If clipping FP16 gradients, call
`scaler.unscale_(optimizer)` first, then clip, then `scaler.step(optimizer)` and
`scaler.update()`. A skipped scaler step skips both moments and weight decay.

## TPU: existing PyTorch/XLA environment

This path is intended for the existing `torch==2.6.0` / `torch_xla==2.6.0`
environment, including the single-host v5litepod-8 setup. Keep that working
environment and install only the package:

```bash
python -m pip install --no-deps --no-build-isolation -e ./adamw_sgdm
PJRT_DEVICE=TPU python adamw_sgdm/examples/train.py --device xla --steps 100
PJRT_DEVICE=TPU python adamw_sgdm/examples/train.py --device xla --amp bf16 --steps 100
```

For a separate fresh **legacy XLA** environment on a TPU VM, Google's published
2.6 recipe uses matching packages. The published XLA 2.6 wheels support
Python 3.9/3.10/3.11, so use **Python 3.11** with this package rather than 3.12:

```bash
python3.11 -m venv .venv-adamw-sgdm-xla
source .venv-adamw-sgdm-xla/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'torch~=2.6.0' 'torch_xla[tpu]~=2.6.0' \
  --find-links https://storage.googleapis.com/libtpu-releases/index.html \
  --find-links https://storage.googleapis.com/libtpu-wheels/index.html
python -m pip install --no-deps --no-build-isolation -e ./adamw_sgdm
```

In an existing XLA training loop, move the model to the XLA device **before**
constructing the optimizer. Replace the normal step with:

```python
import torch_xla
import torch_xla.core.xla_model as xm

device = torch_xla.device()
# model, inputs, and targets must be on device; create the optimizer afterward.
# ... forward, loss.backward(), optional clipping ...
xm.optimizer_step(optimizer, barrier=True)
```

`xm.optimizer_step` performs the update; do not also call `optimizer.step()`.
The included single-host multi-device example uses every device launched by
PyTorch/XLA, with a distributed sampler, `MpDeviceLoader`, and gradient
averaging:

```bash
PJRT_DEVICE=TPU python adamw_sgdm/examples/train_xla_multicore.py \
  --steps 100 --batch-size 64 --amp bf16
```

`--batch-size` is per device in that example. With eight devices, 64 means a
global batch of 512. It is a training example, not an unchanged historical
experiment: global batch size affects the training trajectory. The multi-device
example does not implement checkpoint/resume; use the single-device example
for the portable checkpoint demonstration.

## TPU: native TorchTPU

Current Google documentation recommends TorchTPU for new TPU workloads and
marks PyTorch/XLA deprecated. Both paths are included here so the established
experiments can keep their matching runtime. In a **separate** TPU VM environment:

```bash
python3.12 -m venv .venv-adamw-sgdm-tpu
source .venv-adamw-sgdm-tpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch_tpu==0.2.0
python -m pip install --no-deps --no-build-isolation -e ./adamw_sgdm
python adamw_sgdm/examples/train.py --device tpu --steps 100
python adamw_sgdm/examples/train.py --device tpu --amp bf16 --steps 100
```

Native TorchTPU registers `torch.device("tpu")` and uses the ordinary
`optimizer.step()`. The supplied native example is single device. For native
multi-device jobs, use the runtime's DDP launcher and `tpu_dist` backend; the
XLA multicore script is specifically for the legacy XLA runtime. See the
[official TorchTPU distribution guide](https://torchtpu.readthedocs.io/en/0.2.0/distributed_support/).

## Checkpoints and scheduling

The optimizer implements ordinary `state_dict()` / `load_state_dict()` and
restores step counters and moments onto the current parameters' device.
The demo saves portable CPU tensors and a full optimizer configuration:

```bash
python adamw_sgdm/examples/train.py --device cpu --steps 50 \
  --checkpoint adamw_sgdm/runs/halfway.pt
python adamw_sgdm/examples/train.py --device cpu --steps 100 \
  --resume adamw_sgdm/runs/halfway.pt \
  --checkpoint adamw_sgdm/runs/finished.pt
```

`--steps` is the total target, not additional steps. For this deterministic demo,
same-device and same-precision continuation can be compared directly with an
uninterrupted run. Loading on another backend is supported; identical rounding
and trajectories across different kernels are not promised.
When moving a CUDA FP16 checkpoint to CPU or TPU, pass `--amp none` or
`--amp bf16` explicitly to replace the checkpoint's saved FP16 setting.

For your own model, restore its state, construct the optimizer with the same
parameter group membership and ordering, then load the optimizer state. Save
and restore your scheduler, scaler, data position and RNG states as appropriate.
The original `MagnitudeGraft` wrapper and native AdamW checkpoint formats are
different; this class rejects those optimizer states rather than guessing a
conversion. Model weights alone can always be loaded into a fresh optimizer.

Ordinary LR-only schedulers such as `LambdaLR` and `CosineAnnealingLR` work.
Betas must remain fixed: the identity `m/(1-beta1) = SGDM momentum` assumes a
fixed first-moment coefficient, and the bias corrections assume fixed betas.
For `OneCycleLR`, set `cycle_momentum=False`. A zero LR produces no parameter
change while advancing moments for parameters that have gradients, as in AdamW.

## Exact update

For the gradient supplied by the caller, let

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

The native AdamW displacement and raw momentum-SGD reference are

$$
A_t=-\eta_{A,t}\frac{m_t/(1-\beta_1^t)}
{\sqrt{v_t/(1-\beta_2^t)}+\epsilon},\qquad
B_t=\frac{m_t}{1-\beta_1},\qquad
\eta_{S,t}=r\eta_{A,t}.
$$

For each selected tensor independently, apply

$$
D_t=A_t\frac{\eta_{S,t}\lVert B_t\rVert_F}
{\max(\lVert A_t\rVert_F,\epsilon_{\mathrm{norm}})},\qquad
W_{t+1}=(1-\eta_{A,t}\lambda)W_t+D_t.
$$

Defaults are `epsilon=1e-8` and `norm_epsilon=1e-30`. When the native direction
is zero, the displacement is zero. Above the numerical floor, its norm equals
the reference norm. The reference uses this same model's gradients, zero
initial momentum, no dampening, and no Nesterov correction. It is not a second
model or a second SGD trajectory.

With `graft=False`, `D_t=A_t`. AMSGrad uses the running maximum second moment.
With `maximize=True`, the effective gradient changes sign before updating the
moments. Decay is separate and applied once. A parameter with `grad is None`
is skipped entirely; a zero gradient still allows existing momentum and decay
to act.

The standalone implementation avoids the old wrapper's parameter snapshots,
second writeback, and `.item()` inside the update. Tensor step counters and
stable `expm1` bias corrections keep arithmetic on the device. FP32 and FP64
parameter storage are supported; use FP32 plus autocast for BF16/FP16 compute.
`norm_dtype=torch.float64` is available for reproducing the MNIST norm
accumulation on hardware that supports FP64. Use the default FP32 on TPUs.

## Tests and scope

```bash
python -m unittest discover -s adamw_sgdm/tests -v
```

Tests compare against independent native AdamW and SGD references, cover
the original wrapper, decay, parameter selection, LR scheduling, checkpoint
continuation, numerical edge cases and mixed precision. CUDA tests run when
CUDA is available and explicitly skip otherwise. The examples are the target
hardware smoke checks; see [VERIFICATION.md](VERIFICATION.md) for the recorded
environment and results.

This version supports ordinary eager training and replicated data parallel
training. Sparse gradients, direct FP16/BF16 parameter storage, differentiable
optimizer steps, tensor LRs, fused/foreach mode, CUDA graph capture, compiled
optimizer steps, and sharded/FSDP tensor norms are outside its supported scope.
For sharded tensors, a local norm is not the required whole-tensor norm; global
norm reductions would need to be added. CPU tests establish update correctness,
not acceleration speed or improved model quality. The optimizer does not fit
or force a spectral alpha value.

## Runtime references

- [PyTorch 2.6 CPU and CUDA installation commands](https://pytorch.org/get-started/previous-versions/#v260)
- [PyTorch mixed precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [Google's matching XLA 2.6 setup and deprecation notice](https://docs.cloud.google.com/tpu/docs/run-calculation-pytorch)
- [XLA 2.6 wheel/Python compatibility](https://pypi.org/project/torch-xla/2.6.0/#files)
- [TorchTPU 0.2 VM setup](https://torchtpu.readthedocs.io/en/0.2.0/get_started_on_a_vm/)
- [XLA 2.6 optimizer-step source](https://github.com/pytorch/xla/blob/v2.6.0/torch_xla/core/xla_model.py)
