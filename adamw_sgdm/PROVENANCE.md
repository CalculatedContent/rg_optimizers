# Method and experiment provenance

This package extracts the existing **magnitude-only AdamW intervention** into a
standard `torch.optim.Optimizer`. It keeps AdamW's adaptive direction and uses
the norm of the same-history, undamped, non-Nesterov SGD-momentum reference.
There is no spectral projection or alpha-dependent control in the update.

## Source inspected

The repository base inspected for this addition was
`a5ce682eccba852991b9063211cc9070d170386a`.

- [Original standalone Shakespeare wrapper](https://github.com/CalculatedContent/rg_optimizers/blob/a5ce682eccba852991b9063211cc9070d170386a/short_test_adamw_graft/graft.py)
  has SHA-256 `e6c9fcf0e750e0efda8b716410623eff5e3f590751d5a345482e97a4b10c45a3`.
- [Original MNIST experiment](../experiments/adamw_modified/2026-11-13_mnist_mlp_experiment01/)
  contains the layer-specific graft and sample-normalized decay in
  `frozen/study.py`.
- [Latest five-seed Shakespeare plan](https://github.com/CalculatedContent/rg_optimizers/blob/shakespeare-graft-five-seeds-20261124/short_test_adamw_graft/experiments/2026-11-24_five_seeds/plan.json)
  and its
  [recorded tuned design](https://github.com/CalculatedContent/rg_optimizers/blob/shakespeare-graft-five-seeds-20261124/short_test_adamw_graft/experiments/2026-11-24_five_seeds/frozen_source/recorded_tuned_design.json)
  distinguish the tuned reference LR from the earlier defaults.

## Settings are experiment-specific

| Setting | Original MNIST/Fashion-MNIST graft | Shakespeare original default | Shakespeare latest five-seed run |
| --- | --- | --- | --- |
| Adam peak LR | 0.001 per layer | 0.001 | 0.001 |
| Betas | (0.9, 0.999) | (0.9, 0.99) | (0.9, 0.99) |
| Adam epsilon | 1e-8 | 1e-8 | 1e-8 |
| SGDM reference peak LR | FC1: 0.04; FC2: 0.2; FC3: 0.8 | 0.1 | 0.4 |
| Reference LR ratio | 40; 200; 800 | 100 | 400 |
| Grafted parameters | FC1/FC2/FC3 weights | 16 transformer block matrices | Same 16 matrices |
| Norm accumulation | FP64 | Parameter dtype | FP32 |
| Weight decay | Sample-normalized manual decay | Zero | Zero |

The generic class defaults are Adam LR 0.001, betas (0.9, 0.999), ratio 100 and
standard AdamW weight decay 0.01. Those defaults specify an API starting point;
they do not assert that all prior experiments shared one configuration.

## Latest Shakespeare optimizer selection

For the same nanoGPT module naming used in `short_test`:

```python
from rg_adamw_sgdm import AdamWSGDM, parameter_groups

groups = parameter_groups(
    model,
    selection="shakespeare",
    weight_decay=0.0,
    reference_lr_ratio=400.0,
)
optimizer = AdamWSGDM(groups, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
```

The exact selector is:

```text
transformer\.h\.\d+\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))\.weight
```

It selects packed QKV, attention output, MLP expansion and MLP projection in
each block. A four-block model has 16 such tensors. Token/position embeddings,
the tied output head, normalization parameters and biases retain native AdamW
updates. Pass the underlying model before a DDP wrapper adds name prefixes.

The rest of an experiment still matters: initialization, corpus split,
minibatch order, gradient clipping, scheduling, update horizon, precision and
evaluation all belong to its original runner. This package does not modify
those runners or claim bitwise identical long trajectories across backends.
It tests numerical equivalence of individual optimizer updates to the old
wrapper within floating-point tolerances.

## Original MNIST decay

The original MNIST intervention used zero optimizer weight decay and multiplied
every parameter, including biases, by

$$
d_t=(1-10^{-5}f_t)^{B_t/256},
$$

where `f_t` is the schedule factor and `B_t` the actual minibatch size. This
decay was independent of the learning-rate multiplier. A constant ordinary
AdamW weight-decay coefficient is equivalent only for particular settings
(for example, batch 256, multiplier 1 and Adam LR 0.001 at the schedule peak).

For exact historical decay handling in a full-precision loop, retain the
runner's existing decay, set **all** optimizer group weight decays to zero,
set the three reference ratios to 40/200/800, and use `norm_dtype=torch.float64`.
For example, after backward and before the optimizer step:

```python
with torch.no_grad():
    decay = (1.0 - 1e-5 * schedule_factor) ** (actual_batch_size / 256.0)
    for parameter in model.parameters():
        parameter.mul_(decay)
optimizer.step()
```

This historical snippet is for an unscaled full-precision step. For FP16
GradScaler training, external decay must also be skipped when a scaler skips
the optimizer update; the package's built-in standard decoupled decay already
has this property because it lives inside `step()`.

## Implementation differences that preserve the method

The legacy wrapper takes snapshots, executes native AdamW, then reconstructs
and overwrites selected updates. The new class computes the chosen update
directly and does not take model-weight snapshots or apply AdamW twice.
Step counters are device tensors. Bias correction uses the stable identity
`1 - beta**t = -expm1(t * log(beta))`, with an explicit beta-zero case; this
also avoids FP32 cancellation for betas extremely close to 1. Epsilons must
be normal finite values in their computation dtype.

These are numerical and portability changes. The per-tensor norm, SGDM
reference history, adaptive direction and separate decay retain the original
meaning. A norm at or below the tiny numerical floor follows the documented
clamped rule rather than an exact norm equality.
