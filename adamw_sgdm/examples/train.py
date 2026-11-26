#!/usr/bin/env python3
"""Small, data-free classification demo on CPU, CUDA, TorchTPU, or XLA.

Install the package first with ``pip install -e .`` from adamw_sgdm/.
``--steps`` is the total target, including updates loaded with ``--resume``.
Fixed data, cyclic minibatches, and no stochastic layers make same-backend,
same-precision CPU checkpoint continuation reproducible.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import TensorDataset

from rg_adamw_sgdm import AdamWSGDM, parameter_groups


INPUT_SIZE = 32
HIDDEN_SIZE = 64
NUM_CLASSES = 4
DEFAULTS = {
    "seed": 42,
    "batch_size": 64,
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.01,
    "reference_lr_ratio": 100.0,
    "amp": "none",
}


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def make_dataset(seed: int, samples: int = 512) -> TensorDataset:
    """Use a private CPU generator so every device sees the same problem."""
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    inputs = torch.randn(samples, INPUT_SIZE, generator=generator)
    teacher = torch.randn(INPUT_SIZE, NUM_CLASSES, generator=generator)
    targets = (inputs @ teacher).argmax(dim=1)
    return TensorDataset(inputs, targets)


def make_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    # Parameters stay FP32. Mixed precision is applied through autocast only.
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
    )


def autocast_context(device: torch.device, amp: str):
    if amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def resolve_device(name: str):
    """Import an optional accelerator runtime only when it was requested."""
    if name == "xla":
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "XLA requires a working, version-matched torch/torch_xla "
                "installation. See the README's existing-XLA instructions."
            ) from exc
        return torch_xla.device(), xm
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available.")
    if name == "tpu" and not hasattr(torch, "tpu"):
        raise RuntimeError(
            "The native TPU backend is unavailable. Install torch_tpu in "
            "a suitable TPU environment, or use --device xla for PyTorch/XLA."
        )
    return torch.device(name), None


def cpu_tree(value: Any) -> Any:
    """Materialize checkpoint tensors on CPU, outside the optimizer step."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(cpu_tree(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def training_config(args, checkpoint) -> dict[str, Any]:
    previous = checkpoint["config"] if checkpoint is not None else {}
    result = {}
    for key, default in DEFAULTS.items():
        requested = getattr(args, key)
        saved = previous.get(key, default)
        if checkpoint is not None and requested is not None and requested != saved:
            # AMP may change for a different device, but exact continuation then
            # is not expected. Model, data, and optimizer settings stay fixed.
            if key != "amp":
                raise ValueError(
                    f"Cannot change {key} on resume: checkpoint has {saved!r}. "
                    "Start a fresh run to change the training configuration."
                )
            print(f"Changing AMP from {saved} to {requested}; precision differs.")
        result[key] = saved if requested is None else requested
    if result["seed"] < 0:
        raise ValueError("seed must be nonnegative")
    return result


@torch.no_grad()
def evaluate(model, dataset, device, amp):
    model.eval()
    inputs, targets = (tensor.to(device) for tensor in dataset.tensors)
    with autocast_context(device, amp):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    # These two host reads are evaluation/logging, never inside optimizer.step.
    result = float(loss.cpu()), float(accuracy.cpu())
    model.train()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "tpu", "xla"), default="cpu")
    parser.add_argument("--steps", type=positive_int, default=100,
                        help="total updates, including any resumed updates (default: 100)")
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--log-every", type=positive_int, default=20)
    parser.add_argument("--checkpoint", type=Path, help="save a portable checkpoint at the end")
    parser.add_argument("--resume", type=Path, help="resume this demo's checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="default: 42, or checkpoint value")
    parser.add_argument("--batch-size", type=positive_int, default=None,
                        help="default: 64, or checkpoint value")
    parser.add_argument("--lr", type=float, default=None, help="default: 0.001, or checkpoint value")
    parser.add_argument("--beta1", type=float, default=None, help="default: 0.9, or checkpoint value")
    parser.add_argument("--beta2", type=float, default=None, help="default: 0.999, or checkpoint value")
    parser.add_argument("--eps", type=float, default=None, help="default: 1e-8, or checkpoint value")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="default: 0.01, or checkpoint value")
    parser.add_argument("--reference-lr-ratio", type=float, default=None,
                        help="SGDM reference LR / AdamW LR (default: 100, or checkpoint value)")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default=None,
                        help="default: none, or checkpoint value; fp16 requires CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
            raise ValueError("Expected a format_version=1 checkpoint from this demo.")
    config = training_config(args, checkpoint)
    if config["amp"] == "fp16" and args.device != "cuda":
        raise ValueError("--amp fp16 requires --device cuda; use bf16 or none elsewhere.")
    start_step = 0 if checkpoint is None else int(checkpoint["step"])
    if start_step < 0 or args.steps < start_step:
        raise ValueError("--steps must be at least the checkpoint's completed step count.")

    device, xm = resolve_device(args.device)
    dataset = make_dataset(config["seed"])
    model = make_model(config["seed"]).to(device)
    groups = parameter_groups(
        model, weight_decay=config["weight_decay"], selection="matrices",
        reference_lr_ratio=config["reference_lr_ratio"],
    )
    optimizer = AdamWSGDM(
        groups, lr=config["lr"], betas=(config["beta1"], config["beta2"]),
        eps=config["eps"], weight_decay=config["weight_decay"],
        reference_lr_ratio=config["reference_lr_ratio"],
    )
    scaler = torch.amp.GradScaler("cuda") if config["amp"] == "fp16" else None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    print(f"device={device} amp={config['amp']} torch={torch.__version__} "
          f"start_step={start_step} target_step={args.steps}", flush=True)
    print(f"lr={config['lr']:g} reference_lr_ratio={config['reference_lr_ratio']:g} "
          f"batch_size={config['batch_size']} seed={config['seed']}", flush=True)
    initial_loss, initial_accuracy = evaluate(model, dataset, device, config["amp"])
    print(f"initial loss={initial_loss:.6f} accuracy={initial_accuracy:.2%}", flush=True)
    inputs, targets = dataset.tensors
    batch_offsets = torch.arange(config["batch_size"])
    for step in range(start_step, args.steps):
        # The absolute update index fixes batch order even after a restart.
        indices = (batch_offsets + step * config["batch_size"]) % len(dataset)
        batch_x = inputs[indices].to(device)
        batch_y = targets[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, config["amp"]):
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if xm is None:
                optimizer.step()
            else:
                # A plain loop has no MpDeviceLoader to flush lazy execution.
                xm.optimizer_step(optimizer, barrier=True)
        completed = step + 1
        if completed % args.log_every == 0 or completed == args.steps:
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError(f"Nonfinite loss at step {completed}.")
            print(f"step={completed:4d} minibatch_loss={value:.6f}", flush=True)

    final_loss, final_accuracy = evaluate(model, dataset, device, config["amp"])
    if not math.isfinite(final_loss):
        raise RuntimeError("Final loss is nonfinite.")
    print(f"final loss={final_loss:.6f} accuracy={final_accuracy:.2%}", flush=True)
    if args.checkpoint is not None:
        if xm is not None:
            xm.mark_step()
        save_checkpoint(args.checkpoint, {
            "format_version": 1,
            "step": args.steps,
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
        })
        print(f"saved checkpoint={args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
