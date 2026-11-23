# Tiny Shakespeare with an SGD-magnitude graft for AdamW

Does giving AdamW the per-matrix step lengths of momentum SGD improve held-out loss? [Open the experiment notebook](experiment.ipynb) to compare native AdamW, native momentum SGD, and the magnitude-grafted AdamW on five paired seeds. The primary outcome is test cross-entropy at 20,000 updates, in nats per character. A negative graft-minus-AdamW difference favors the graft. Spectral alignment with α = 2 is a separate outcome.

## Install and open

Run the commands from the `rg_optimizers` repository root. The notebook and runner import the existing `short_test` and `short_test_portable` folders. Use a checkout that includes both.

**Linux CPU or Mac.** With Python 3.11 installed, create a virtual environment and launch Jupyter from it. Leave the notebook’s `DEVICE` set to `"cpu"`, or select `"mps"` for a supported Apple Silicon GPU. For MPS, unset `PYTORCH_ENABLE_MPS_FALLBACK` before starting Jupyter. An Intel Mac needs a Python/PyTorch pair with compatible wheels.

```bash
python3.11 -m venv .venv-shakespeare
source .venv-shakespeare/bin/activate
python -m pip install --upgrade pip
python -m pip install -r short_test_portable/requirements-cpu.txt jupyterlab ipykernel
python -m jupyterlab short_test_adamw_graft/experiment.ipynb
```

**Google TPU.** Use a supported TPU VM with a native TorchTPU environment. These commands follow the repository’s pinned TPU setup. Launch Jupyter there and set `DEVICE = "tpu"`. This runner uses one TPU device per process and keeps batch size 12. The preflight exercises the selected device and checkpoint replay. [Repository TPU setup](../short_test_portable/README.md#google-tpu-setup) · [Google’s PyTorch on Cloud TPU documentation](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu)

```bash
python3.11 -m venv .venv-shakespeare-tpu
source .venv-shakespeare-tpu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r short_test_portable/requirements-tpu.txt jupyterlab ipykernel
python -m jupyterlab short_test_adamw_graft/experiment.ipynb
```

The settings cell lets you edit the data and output paths. It downloads the original Tiny Shakespeare text if needed and checks its SHA-256. Use a separate output directory per backend. Compute kernels and dependency versions can affect cross-device comparisons. The notebook and CLI share code and defaults.

## Run from a terminal

```bash
EXP=short_test_adamw_graft
DEVICE=cpu
DATA="$EXP/data/input.txt"
OUT="$EXP/runs/$DEVICE"
python short_test/download_data.py --output "$DATA"
python "$EXP/run.py" preflight --device "$DEVICE" --data "$DATA" --output "$OUT"
python "$EXP/run.py" run --device "$DEVICE" --data "$DATA" --output "$OUT"
python "$EXP/report.py" --output "$OUT"
```

To measure speed on your hardware first, append `--arm graft --seed 1340 --until-step 500` to the `run` command. Rerun without those three options to resume and complete all 15 fits. The runner checkpoints every 250 updates, preserves completed fits, and checks source, environment, and design before resuming. Endpoints appear in `runs/<device>/<arm>/seed_<seed>/result.json`.

## Design and optimizer

The model is the original 804,096-parameter character-level nanoGPT with four transformer blocks, four attention heads, width 128, context 64, vocabulary 65, zero dropout, and `bias=False`. Each minibatch has 12 windows of 64 characters. All arms use the same contiguous 90% training, 5% validation, and 5% test split, FP32 weights, and global gradient norm clipping at 1. Each seed from 1340 through 1344 fixes the initial weights and minibatch stream across arms. [Frozen Shakespeare comparison](../short_test_optimizer_comparison/frozen_design_20261021.json) · [Data and model definitions](../short_test/common.py) · [Portable optimizer and schedule](../short_test_portable/comparison.py)

| Arm | Parameter update | Peak learning rate |
| --- | --- | --- |
| `adamw` | Native AdamW with β₁ = 0.9, β₂ = 0.99, ε = 10⁻⁸ | 0.001 |
| `sgdm` | Native SGD with momentum 0.9, zero dampening, Nesterov disabled | 0.1 |
| `graft` | AdamW with the paper’s per-matrix SGDM magnitude | 0.001 for AdamW and 0.1 for the SGDM reference |

All arms use zero weight decay, 100 warmup updates, cosine decay to 10% of the peak at update 17,000, and 3,000 further updates at that rate. These are the historical comparison settings. The three-arm experiment tests whether the graft helps at this setting. The graft’s optimal learning rate remains an open question. The task uses Tiny Shakespeare characters. FineWeb speed-run losses use a different corpus and prediction unit.

For each grafted matrix let $g_t$ be its gradient **after** global clipping and $b_t=0.9b_{t-1}+g_t$, with $b_0=0$. Let $d_t$ be AdamW’s signed, bias-corrected adaptive update. The applied update is $d_t\,\eta_{\mathrm{SGD},t}\lVert b_t\rVert_F/\lVert d_t\rVert_F$, with a zero update when the direction vanishes. The reference follows the grafted model’s own gradients. The 16 grafted matrices comprise packed QKV, attention output, MLP expansion, and MLP projection in each of the four blocks. Token and position embeddings, the tied output head, and normalization parameters retain their native AdamW updates. [MNIST graft experiment](../experiments/adamw_modified/2026-11-13_mnist_mlp_experiment01/README.md) · [Earlier two-block GPT pilot](../experiments/adamw_modified/2026-11-15_tiny_shakespeare_gpt_experiment03/README.md) · [Graft implementation](graft.py)

## Read the results

Validation runs every 500 updates. The full test split is evaluated at the fixed endpoint and at the checkpoint with the lowest validation cross-entropy. The latter is a secondary comparison. This corpus and these seeds were used in the historical control comparison. Each arm has five seed-level observations.

| Output | What it contains |
| --- | --- |
| `per_seed.csv` | Split losses, next-character accuracy, confidence-weighted α, and fit counts for both checkpoint choices. |
| `arm_means.csv` | Equal-weight means across available seeds and their counts. |
| `paired_effects.csv` | Graft-minus-control differences per matched seed and their means with two-sided 95% Student *t* intervals. Check the number of available pairs. |
| `per_matrix.csv` | Raw WeightWatcher fit fields, fit status, and confidence weights for initialization and each recorded validation checkpoint. |

WeightWatcher 0.7.7 fits 18 physical matrices per checkpoint, comprising 16 transformer matrices and two embeddings. The tied output is counted once. For each model, the reported α is the weighted mean of those fits with $w_i=\sqrt{n_{\mathrm{tail},i}}/(D_i+0.02)$, where $D_i$ is the fit’s KS distance. The report requires all 18 valid fits for that model’s weighted mean. The arm summary averages those model-level values equally across seeds. This fit-quality weight is a heuristic. The seed-level intervals describe variation across initializations and minibatch streams on this particular corpus. [Reporting implementation](report.py)

## CPU verification

On the original corpus, all three optimizers resumed exactly across a checkpoint on Linux CPU and produced 18 successful WeightWatcher fits. At 500 updates with seed 1340, both control validation losses matched the archived runs to within 0.000001 nat per character.

| Arm | Validation loss at step 500 | Confidence-weighted α |
| --- | ---: | ---: |
| Native AdamW | 2.302431 | 2.094751 |
| SGD with momentum | 2.419553 | 2.989148 |
| AdamW with the graft | 2.296159 | 2.678472 |

This single early checkpoint leaves the five-seed endpoint comparison open. Mac and TPU execution still require a hardware check. [CPU measurements, archived control values, and 54 raw matrix fits](verification/seed_1340_step_500.json)
