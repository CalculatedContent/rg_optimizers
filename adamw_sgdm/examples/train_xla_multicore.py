#!/usr/bin/env python3
"""Bounded single-host, multi-device TPU demo for existing PyTorch/XLA 2.6.

Run on the TPU host with:
    PJRT_DEVICE=TPU python examples/train_xla_multicore.py --steps 100 --amp bf16

This example uses synthetic data and one optimizer update per minibatch.
Use train.py --device tpu for the current native TorchTPU backend.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from rg_adamw_sgdm import AdamWSGDM, parameter_groups
from train import autocast_context, make_dataset, make_model, positive_int


def _combine_metrics(rows):
    loss_sum = sum(float(row[0]) for row in rows)
    correct = sum(float(row[1]) for row in rows)
    count = sum(int(row[2]) for row in rows)
    return loss_sum / count, correct / count, count


def _worker(index: int, args) -> None:
    # Device queries and tensors must be created inside the launched worker.
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr

    del index
    torch.set_num_threads(args.threads)
    device = torch_xla.device()
    rank, world_size = xr.global_ordinal(), xr.world_size()
    dataset = make_dataset(args.seed, samples=args.samples)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=args.seed, drop_last=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        drop_last=True, num_workers=0,
    )
    if len(loader) == 0:
        raise ValueError("--samples must be at least world_size * --batch-size.")
    model = make_model(args.seed).to(device)
    # Necessary even when older TPU runtimes launch threads sharing CPU RNG.
    xm.broadcast_master_param(model)
    groups = parameter_groups(
        model, weight_decay=args.weight_decay, selection="matrices",
        reference_lr_ratio=args.reference_lr_ratio,
    )
    optimizer = AdamWSGDM(
        groups, lr=args.lr, betas=(args.beta1, args.beta2),
        eps=args.eps, weight_decay=args.weight_decay,
        reference_lr_ratio=args.reference_lr_ratio,
    )
    if rank == 0:
        print(f"runtime=PyTorch/XLA devices={world_size} amp={args.amp} "
              f"steps={args.steps} per_device_batch={args.batch_size} "
              f"global_batch={world_size * args.batch_size}", flush=True)

    completed = 0
    loss_sum = torch.zeros((), device=device)
    correct_sum = torch.zeros((), device=device)
    sample_count = 0
    model.train()
    # Reusing this wrapper lets its next __iter__ close the previous epoch's
    # background loader before creating the next one.
    device_loader = pl.MpDeviceLoader(loader, device)
    for epoch in range(math.ceil(args.steps / len(loader))):
        sampler.set_epoch(epoch)
        for batch_x, batch_y in device_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                logits = model(batch_x)
                loss = nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            # This averages gradients and calls optimizer.step exactly once.
            # MpDeviceLoader provides normal step barriers on iteration.
            xm.optimizer_step(optimizer)
            loss_sum = loss_sum + loss.detach() * batch_y.shape[0]
            correct_sum = correct_sum + (logits.detach().argmax(1) == batch_y).sum()
            sample_count += batch_y.shape[0]
            completed += 1
            if completed % args.log_every == 0 or completed == args.steps:
                # Execute pending work before converting metrics for logging.
                xm.mark_step()
                mean_loss, accuracy, count = xm.mesh_reduce(
                    f"metrics-step-{completed}",
                    (loss_sum, correct_sum, sample_count),
                    _combine_metrics,
                )
                if not math.isfinite(mean_loss):
                    raise RuntimeError(f"Nonfinite global loss at step {completed}.")
                if rank == 0:
                    print(f"step={completed:4d} loss={mean_loss:.6f} "
                          f"accuracy={accuracy:.2%} samples_since_log={count}", flush=True)
                loss_sum = torch.zeros((), device=device)
                correct_sum = torch.zeros((), device=device)
                sample_count = 0
            if completed == args.steps:
                break
        if completed == args.steps:
            break
    xm.mark_step()
    xm.wait_device_ops()
    xm.rendezvous("training-finished")
    if rank == 0:
        print(f"completed {completed} synchronized optimizer updates", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=100)
    parser.add_argument("--samples", type=positive_int, default=4096)
    parser.add_argument("--batch-size", type=positive_int, default=64,
                        help="per-device batch size")
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--log-every", type=positive_int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--reference-lr-ratio", type=float, default=100.0)
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    try:
        import torch_xla
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install matching torch and torch_xla packages on the TPU host.") from exc
    torch_xla.launch(_worker, args=(args,))


if __name__ == "__main__":
    main()
