"""Bounded actual GPT-2 124M CPU training timing, not a convergence experiment.

Measures accumulated forward/backward microbatches, full-vocabulary validation,
and optimizer/gradient-clipping cost separately. Synthetic GPT-2 token IDs avoid
downloading a corpus; no extrapolation from a smaller model or peak FLOPS.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time


def emit(kind, value):
    print(json.dumps({"kind": kind, **value}, allow_nan=False), flush=True)


def summary(samples):
    return {"samples_seconds": samples, "median_seconds": statistics.median(samples),
            "mean_seconds": statistics.mean(samples), "min_seconds": min(samples),
            "max_seconds": max(samples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replica", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=540)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    import numpy as np
    import torch
    root = args.repo.resolve()
    sys.path.insert(0, str(root / "baseline/gpt2_small/muon_speedrun"))
    sys.path.insert(0, str(root / "adamw_sgdm/src"))
    import stock_model
    from rg_adamw_sgdm import AdamWSGDM

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    threads = min(4, len(affinity) or os.cpu_count() or 1)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    model = stock_model.make_model(seed=42, device=torch.device("cpu"))
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 124439808, parameters
    assert model.config.block_size == 1024
    assert model.lm_head.weight is model.transformer.wte.weight
    cpu_info = Path("/proc/cpuinfo").read_text()
    cpu_model = next((line.split(":", 1)[1].strip() for line in cpu_info.splitlines()
                      if line.startswith("model name")), "unknown")
    report = {"schema": 1, "replica": args.replica, "measurement": "real full-size CPU timing; synthetic input IDs",
              "model": "stock GPT-2 Small", "parameters": parameters,
              "context": 1024, "vocabulary": 50257, "precision": "float32",
              "attention": "PyTorch scaled_dot_product_attention, causal",
              "validation_includes_token_argmax": True,
              "torch_compile": False, "model_source_sha256": stock_model.SOURCE_SHA256,
              "git_sha": os.environ.get("GITHUB_SHA"),
              "github_run_id": os.environ.get("GITHUB_RUN_ID"),
              "runner_os": os.environ.get("RUNNER_OS"),
              "runner_arch": os.environ.get("RUNNER_ARCH"),
              "hardware": {"cpu_model": cpu_model, "logical_cpus": os.cpu_count(),
                           "affinity": affinity, "torch_threads": threads,
                           "memory_kb": int(Path("/proc/meminfo").read_text().splitlines()[0].split()[1]),
                           "platform": platform.platform()},
              "software": {"torch": torch.__version__, "python": platform.python_version(),
                           "numpy": np.__version__, "mkldnn": torch.backends.mkldnn.enabled},
              "torch_build": torch.__config__.show(),
              "protocol": {"updates": 19560, "batch_tokens": 524288,
                           "training_tokens": 19560 * 524288,
                           "validation_tokens_per_event": 10485760,
                           "validation_events_excluding_initial": 79},
              "microbatches": [], "optimizers": {},
              "excluded": ["dataset I/O", "WeightWatcher/SVD", "checkpoint storage and transfers",
                           "job startup, package installation and queues", "job restarts", "convergence to target loss"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["max_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    emit("environment", {k: v for k, v in report.items() if k not in {"torch_build", "microbatches", "optimizers"}})
    save()
    started = time.perf_counter()
    deadline = started + args.seconds
    generator = torch.Generator().manual_seed(1927)
    # Preallocate gradients: actual training accumulates many microbatches before
    # one optimizer update, so a fresh-None-gradient microbenchmark is misleading.
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    # Keep the real AdamW moment arrays resident during forward/backward timings,
    # so the memory footprint includes the normal training optimizer state.
    resident_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0,
                                           weight_decay=0.0, foreach=False, fused=False)
    resident_optimizer.step()

    for batch in (1, 2, 4):
        if time.perf_counter() > deadline - 120:
            break
        tokens = torch.randint(50257, (batch, 1025), generator=generator)
        x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        accum = 524288 // (batch * 1024)
        model.train()
        model.zero_grad(set_to_none=False)
        # One excluded full-shape warmup, then five accumulated timed passes.
        _, loss = model(x, y, return_logits=False)
        (loss / accum).backward()
        assert math.isfinite(float(loss.detach()))
        training = []
        for sample in range(args.samples):
            if time.perf_counter() >= deadline - 75:
                break
            tick = time.perf_counter()
            _, loss = model(x, y, return_logits=False)
            (loss / accum).backward()
            elapsed = time.perf_counter() - tick
            assert math.isfinite(float(loss.detach()))
            training.append(elapsed)
        if not training:
            break
        validation = []
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, y, return_logits=True)
            (logits.argmax(dim=-1) != y).sum()
            del logits
            for sample in range(3):
                tick = time.perf_counter()
                logits, val_loss = model(x, y, return_logits=True)
                (logits.argmax(dim=-1) != y).sum()
                del logits
                elapsed = time.perf_counter() - tick
                assert math.isfinite(float(val_loss))
                validation.append(elapsed)
        row = {"batch_sequences": batch, "tokens": batch * 1024,
               "accumulations_per_update": accum, "forward_backward": summary(training),
               "validation_forward": summary(validation),
               "training_tokens_per_second": batch * 1024 / statistics.median(training)}
        report["microbatches"].append(row)
        save()
        emit("microbatch", row)

    # Refill dense nonzero gradients for each optimizer. Record its arithmetic at
    # the true once-per-update frequency; no extra backward pass is charged here.
    del resident_optimizer
    gc.collect()
    groups = lambda: [dict(params=[p for p in model.parameters() if p.ndim >= 2], weight_decay=0.1),
                      dict(params=[p for p in model.parameters() if p.ndim < 2], weight_decay=0.0)]
    for name in ("adamw", "sgdm", "adamw_sgdm"):
        for parameter in model.parameters():
            parameter.grad.fill_(0.0001)
        if name == "adamw":
            opt = torch.optim.AdamW(groups(), lr=0.0006, betas=(0.9, 0.95), eps=1e-8,
                                    foreach=False, fused=False)
        elif name == "sgdm":
            opt = torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.9, foreach=False)
        else:
            selected, auxiliary = [], []
            for key, p in model.named_parameters():
                is_block_matrix = key.startswith("transformer.h.") and p.ndim == 2
                (selected if is_block_matrix else auxiliary).append(p)
            assert len(selected) == 48
            opt = AdamWSGDM([dict(params=selected, graft=True),
                             dict(params=auxiliary, graft=False)],
                            lr=0.0006, betas=(0.9, 0.95), reference_lr_ratio=400.0,
                            weight_decay=0.0)
        opt.step()  # excluded moment/state allocation
        samples = []
        for _ in range(5):
            tick = time.perf_counter()
            with torch.no_grad():
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                norm = sum(g.detach().float().square().sum() for g in grads).sqrt()
                scale = torch.clamp(1.0 / (norm + 1e-6), max=1.0)
                for grad in grads:
                    grad.mul_(scale)
                opt.step()
            samples.append(time.perf_counter() - tick)
        # Zeroing once at the boundary is also charged once per full update.
        tick = time.perf_counter()
        model.zero_grad(set_to_none=False)
        zero_seconds = time.perf_counter() - tick
        row = {"clip_and_step": summary(samples), "zero_grad_seconds": zero_seconds}
        report["optimizers"][name] = row
        emit("optimizer", {"name": name, **row})
        del opt
        gc.collect()

    report["elapsed_benchmark_seconds"] = time.perf_counter() - started
    report["status"] = "complete" if len(report["microbatches"]) == 3 else "partial_shapes_timed"
    save()
    emit("BENCHMARK_RESULT_JSON", report)


if __name__ == "__main__":
    main()
