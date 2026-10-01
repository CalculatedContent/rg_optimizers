from __future__ import annotations

"""Opt-in numerical, resume and throughput check on actual XLA devices.

Run with --backend tpu on the VM; --backend cpu is a local XLA compiler test,
not evidence of TPU hardware performance. This never downloads a corpus.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("tpu", "cpu"), default="tpu")
    parser.add_argument("--chips", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--benchmark-steps", type=int, default=30)
    args = parser.parse_args()
    if args.chips < 1 or args.benchmark_steps < 1:
        parser.error("chips and benchmark-steps must be positive")
    os.environ["PJRT_DEVICE"] = args.backend.upper()
    if args.backend == "cpu":
        os.environ["CPU_NUM_DEVICES"] = str(args.chips)
    torch.set_num_threads(1)

    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla
    from . import tpu_spmd as spmd
    from .muonclip import install_muonclip_extension

    install_muonclip_extension()
    from .checkpoints import save_training_checkpoint, load_training_checkpoint_for_resume
    from .config import load_config, optimizer_profile
    from .evaluation import evaluate_probe
    from .model import GPT, GPTConfig
    from .optimizers import make_optimizer_handles, zero_grad, optimizer_step
    from .runtime import mark_step, synchronize, tree_to_cpu, parameter_snapshot

    spmd._initialize_mesh(xr, args.chips)
    device = xm.xla_device()
    cfg_path = Path.cwd() / "configs" / "muonclip_reference.yaml"
    cfg = load_config(cfg_path)
    cfg["model"].update(vocab_size=64, block_size=8, n_embd=16, n_head=2, n_layer=1)
    cfg["training"].update(batch_size=2 * args.chips, grad_accum_steps=2)
    cfg["runtime"].update(tpu_spmd=True, tpu_expected_chips=args.chips)
    profile = optimizer_profile(cfg, "muon_clip")
    # Force clipping to activate so a wrong cross-chip max cannot pass quietly.
    profile.update(qk_clip_threshold=0.0001, qk_diagnostics_interval=100)
    torch.manual_seed(314)
    reference = GPT(GPTConfig(**cfg["model"]))
    model = deepcopy(reference).to(device)
    spmd.replicate_model(model)
    assert model.token_embedding.weight is model.lm_head.weight, "XLA transfer broke tied weights"
    cpu_handles = make_optimizer_handles(reference, profile)
    handles = make_optimizer_handles(model, profile)
    generator = torch.Generator().manual_seed(2718)
    batches = [
        (torch.randint(64, (2 * args.chips, 8), generator=generator),
         torch.randint(64, (2 * args.chips, 8), generator=generator))
        for _ in range(2)
    ]

    def update(net, opts, target):
        zero_grad(opts)
        for x, y in batches:
            _, loss = net(spmd.batch_to_device(x, target), spmd.batch_to_device(y, target))
            (loss / len(batches)).backward()
        spmd.replicate_gradients(net)
        gradients = {n: p.grad.detach().cpu().clone() for n, p in net.named_parameters()}
        maxima = [b.attn._muonclip_max_logits.detach().cpu().clone() for b in net.blocks]
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, foreach=False)
        optimizer_step(opts)
        mark_step(target)
        synchronize(target)
        return gradients, maxima

    cpu_grad, cpu_max = update(reference, cpu_handles, torch.device("cpu"))
    xla_grad, xla_max = update(model, handles, device)
    for name in cpu_grad:
        torch.testing.assert_close(xla_grad[name], cpu_grad[name], atol=3e-5, rtol=3e-3)
    for actual, expected in zip(xla_max, cpu_max):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-3)
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name].cpu(), expected, atol=3e-5, rtol=3e-3)
    for block_max in xla_max:
        assert bool((block_max > profile["qk_clip_threshold"]).any()), "QK clip was not exercised"

    probe_cpu = evaluate_probe(reference, batches, torch.device("cpu"))
    probe_xla = evaluate_probe(model, batches, device)
    for metric in ("loss", "accuracy", "top5_accuracy"):
        torch.testing.assert_close(torch.tensor(probe_xla[metric]), torch.tensor(probe_cpu[metric]), atol=3e-5, rtol=3e-3)

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint.pt"
        save_training_checkpoint(
            path, model=model, handles=handles, step=1,
            best_validation_loss=probe_xla["loss"], best_validation_step=1,
            elapsed_seconds=1.0, fingerprint="spmd-numerical-check", cfg=cfg,
            optimizer_name="muon_clip", seed=314, train_generator=generator,
            resume_diagnostics={"previous_eval_snapshot": parameter_snapshot(model),
                                "last_grad_pre": 1.0, "last_grad_post": 1.0, "last_clipped": False},
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert all(t.device.type == "cpu" for t in payload["model"].values())
        resumed = GPT(GPTConfig(**cfg["model"])).to(device)
        spmd.replicate_model(resumed)
        resumed_handles = make_optimizer_handles(resumed, profile)
        restored_generator = torch.Generator()
        state = load_training_checkpoint_for_resume(
            path, model=resumed, handles=resumed_handles,
            expected_fingerprint="spmd-numerical-check", train_generator=restored_generator,
        )
        spmd.replicate_model(resumed)
        assert state[0] == 1
        assert torch.equal(restored_generator.get_state(), generator.get_state())
        update(model, handles, device)
        update(resumed, resumed_handles, device)
        for name, expected in tree_to_cpu(model.state_dict()).items():
            torch.testing.assert_close(resumed.state_dict()[name].cpu(), expected, atol=1e-6, rtol=1e-5)
        # Check the next update as well: a missing momentum/Adam state can hide
        # behind an apparently successful model-only checkpoint load.
        update(model, handles, device)
        update(resumed, resumed_handles, device)
        for name, expected in tree_to_cpu(model.state_dict()).items():
            torch.testing.assert_close(resumed.state_dict()[name].cpu(), expected, atol=1e-6, rtol=1e-5)

    report = {"passed": True, "backend": xr.device_type(), **spmd.metadata(),
              "torch": torch.__version__, "torch_xla": torch_xla.__version__,
              "checks": ["global gradients", "global per-head QK maxima", "clipped update",
                         "train/eval metrics", "CPU checkpoint", "optimizer and sampler resume"],
              "max_gradient_error": max(float((cpu_grad[n] - xla_grad[n]).abs().max()) for n in cpu_grad)}

    if args.benchmark_config:
        benchmark_cfg = load_config(args.benchmark_config)
        batch = int(benchmark_cfg["training"]["batch_size"])
        accum = int(benchmark_cfg["training"]["grad_accum_steps"])
        context = int(benchmark_cfg["model"]["block_size"])
        if batch % args.chips:
            raise ValueError("Benchmark global batch must divide evenly across chips")
        net = GPT(GPTConfig(**benchmark_cfg["model"])).to(device)
        spmd.replicate_model(net)
        opts = make_optimizer_handles(net, optimizer_profile(benchmark_cfg, "muon_clip"))
        x = spmd.batch_to_device(torch.randint(net.cfg.vocab_size, (batch, context)), device)
        y = spmd.batch_to_device(torch.randint(net.cfg.vocab_size, (batch, context)), device)
        # Includes the same sampler copy/sharding boundary in each timed step.
        # Uses synthetic tokens; excludes WW, evaluation and checkpoint I/O.
        x_cpu, y_cpu = x.cpu(), y.cpu()
        def benchmark_step():
            zero_grad(opts)
            for _ in range(accum):
                _, loss = net(spmd.batch_to_device(x_cpu, device), spmd.batch_to_device(y_cpu, device))
                (loss / accum).backward()
            spmd.replicate_gradients(net)
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(benchmark_cfg["training"]["grad_clip"]), foreach=False)
            optimizer_step(opts)
            mark_step(device)
        for _ in range(5):
            benchmark_step()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.benchmark_steps):
            benchmark_step()
        synchronize(device)
        seconds = time.perf_counter() - start
        report["benchmark"] = {"steps": args.benchmark_steps, "seconds": seconds,
                               "global_tokens_per_update": batch * accum * context,
                               "tokens_per_second": args.benchmark_steps * batch * accum * context / seconds,
                               "includes_monitoring": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
