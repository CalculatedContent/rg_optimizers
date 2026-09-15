#!/usr/bin/env python3
"""Adaptive spectral-guard training for the nanoGPT memorization study.

The controller trains in windows, evaluates WeightWatcher on the six transformer
matrices, and only accepts a window when both raw and clip_xmax alpha are above
the target. Failed windows are rolled back. The controller then lowers the
offending matrix learning rate and increases a matrix-specific anchor penalty.
For SGD momentum it can also increase dropout on the offending transformer
block before retrying the same window.

This is an explicit intervention on the spectral diagnostic, not an unbiased
optimizer comparison. Use development seeds to tune controller settings and
lock them before evaluating on a comparison seed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARENT_PATH = HERE / "run.py"
CONFIG_PATH = HERE / "configs" / "suite.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_parent():
    spec = importlib.util.spec_from_file_location("_spectral_guard_parent", PARENT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def matrix_block(matrix_name: str) -> int:
    if not matrix_name.startswith("L") or "_" not in matrix_name:
        raise ValueError(f"unrecognized matrix name: {matrix_name}")
    return int(matrix_name[1:3])


def classify_alpha(table: pd.DataFrame, *, target: float, floor: float) -> dict:
    required = {"matrix_name", "alpha_clip_xmax", "alpha_raw", "step"}
    if not required.issubset(table.columns):
        raise ValueError(f"missing WeightWatcher columns: {sorted(required - set(table.columns))}")
    rows = []
    for _, r in table.iterrows():
        clip = float(r["alpha_clip_xmax"])
        raw = float(r["alpha_raw"])
        if not math.isfinite(clip) or not math.isfinite(raw):
            raise ValueError("nonfinite alpha")
        minimum = min(clip, raw)
        rows.append({
            "matrix_name": str(r["matrix_name"]),
            "alpha_clip_xmax": clip,
            "alpha_raw": raw,
            "min_alpha": minimum,
            "severity": "hard" if minimum < floor else ("mild" if minimum < target else "pass"),
        })
    offenders = [r for r in rows if r["min_alpha"] < target]
    worst = min(rows, key=lambda r: r["min_alpha"])
    return {
        "passed": not offenders,
        "target_alpha": float(target),
        "floor_alpha": float(floor),
        "min_alpha": float(worst["min_alpha"]),
        "worst_matrix": worst["matrix_name"],
        "offenders": offenders,
        "rows": rows,
    }


def adapt_controller(
    controller: dict,
    alpha_status: dict,
    *,
    optimizer_name: str,
    mild_backoff: float,
    hard_backoff: float,
    min_lr_scale: float,
    anchor_start: float,
    anchor_growth: float,
    max_anchor: float,
    dropout_step: float,
    max_dropout: float,
) -> dict:
    before = deepcopy(controller)
    blocks = set()
    for row in alpha_status["offenders"]:
        name = row["matrix_name"]
        factor = hard_backoff if row["severity"] == "hard" else mild_backoff
        controller["lr_scale"][name] = max(
            min_lr_scale, float(controller["lr_scale"][name]) * factor
        )
        old = float(controller["anchor_lambda"][name])
        controller["anchor_lambda"][name] = min(
            max_anchor, anchor_start if old == 0.0 else old * anchor_growth
        )
        blocks.add(matrix_block(name))

    if optimizer_name == "sgd_momentum":
        for block in blocks:
            key = str(block)
            increment = dropout_step
            if any(
                row["severity"] == "hard" and matrix_block(row["matrix_name"]) == block
                for row in alpha_status["offenders"]
            ):
                increment *= 2.0
            controller["dropout"][key] = min(
                max_dropout, float(controller["dropout"].get(key, 0.0)) + increment
            )

    if before == controller:
        raise RuntimeError("controller could not adapt further")
    return before


def set_block_dropout(model, dropout_by_block: dict[str, float]) -> None:
    for i, block in enumerate(model.blocks):
        p = float(dropout_by_block.get(str(i), 0.0))
        if not 0.0 <= p < 1.0:
            raise ValueError(f"invalid dropout for block {i}: {p}")
        block.attn.dropout = p
        block.attn.resid_dropout.p = p
        block.mlp.dropout.p = p


def cpu_model_state(model) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def capture_rng(torch, device: str) -> dict:
    state = {"torch": torch.get_rng_state()}
    if device == "cuda":
        state["device"] = torch.cuda.get_rng_state_all()
    elif device == "mps":
        state["device"] = torch.mps.get_rng_state()
    else:
        state["device"] = None
    return state


def restore_rng(torch, device: str, state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if device == "cuda" and state.get("device") is not None:
        torch.cuda.set_rng_state_all(state["device"])
    elif device == "mps" and state.get("device") is not None:
        torch.mps.set_rng_state(state["device"])


def optimizer_states(optimizers: dict[str, Any]) -> dict[str, dict]:
    return {name: deepcopy(opt.state_dict()) for name, opt in optimizers.items()}


def load_optimizer_states(optimizers: dict[str, Any], states: dict[str, dict]) -> None:
    if set(optimizers) != set(states):
        raise RuntimeError("optimizer topology changed")
    for name, opt in optimizers.items():
        opt.load_state_dict(states[name])


def build_optimizers(model, source: dict, optimizer_name: str, initial_lr_scale: float):
    import torch
    from rg_nanogpt_one_head.model import transformer_matrix_items
    from rg_nanogpt_one_head.optimizers import Muon

    matrix_items = transformer_matrix_items(model)
    matrix_params = {name: weight for name, _, _, weight in matrix_items}
    matrix_ids = {id(p) for p in matrix_params.values()}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    auxiliary = [(n, p) for n, p in named if id(p) not in matrix_ids]
    optimizers: dict[str, Any] = {}

    if optimizer_name == "muon":
        profile = deepcopy(source["optimizer_profiles"]["muon"])
        for name, p in matrix_params.items():
            optimizers[f"matrix:{name}"] = Muon(
                [p],
                lr=float(profile["matrix_learning_rate"]) * initial_lr_scale,
                momentum=float(profile["momentum"]),
                nesterov=bool(profile["nesterov"]),
                weight_decay=float(profile["matrix_weight_decay"]),
                newton_schulz_steps=int(profile["newton_schulz_steps"]),
                eps=float(profile.get("muon_epsilon", 1e-7)),
            )
        decay = [p for _, p in auxiliary if p.ndim >= 2]
        no_decay = [p for _, p in auxiliary if p.ndim < 2]
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": float(profile["aux_weight_decay"])})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        optimizers["aux"] = torch.optim.AdamW(
            groups,
            lr=float(profile["aux_learning_rate"]),
            betas=(float(profile["beta1"]), float(profile["beta2"])),
            eps=float(profile["epsilon"]),
        )
        return optimizers, matrix_params, profile

    if optimizer_name != "sgd_momentum":
        raise ValueError(optimizer_name)

    profile = deepcopy(source["optimizer_profiles"]["sgd_momentum"])
    for name, p in matrix_params.items():
        optimizers[f"matrix:{name}"] = torch.optim.SGD(
            [{"params": [p], "weight_decay": float(profile["weight_decay"])}],
            lr=float(profile["learning_rate"]) * initial_lr_scale,
            momentum=float(profile["momentum"]),
            dampening=float(profile.get("dampening", 0.0)),
            nesterov=bool(profile.get("nesterov", True)),
        )
    decay = [p for _, p in auxiliary if p.ndim >= 2]
    no_decay = [p for _, p in auxiliary if p.ndim < 2]
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(profile["weight_decay"])})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    optimizers["aux"] = torch.optim.SGD(
        groups,
        lr=float(profile["learning_rate"]),
        momentum=float(profile["momentum"]),
        dampening=float(profile.get("dampening", 0.0)),
        nesterov=bool(profile.get("nesterov", True)),
    )
    return optimizers, matrix_params, profile


def set_lrs(
    optimizers: dict[str, Any],
    controller: dict,
    profile: dict,
    optimizer_name: str,
    *,
    step: int,
    schedule_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    from rg_nanogpt_one_head.optimizers import cosine_learning_rate

    values = {}
    if optimizer_name == "muon":
        matrix_base = cosine_learning_rate(
            step,
            total_steps=schedule_steps,
            warmup_steps=warmup_steps,
            peak_lr=float(profile["matrix_learning_rate"]),
            min_lr=float(profile["matrix_min_learning_rate"]),
        )
        aux_lr = cosine_learning_rate(
            step,
            total_steps=schedule_steps,
            warmup_steps=warmup_steps,
            peak_lr=float(profile["aux_learning_rate"]),
            min_lr=float(profile["aux_min_learning_rate"]),
        )
        for name, scale in controller["lr_scale"].items():
            lr = matrix_base * float(scale)
            for group in optimizers[f"matrix:{name}"].param_groups:
                group["lr"] = lr
            values[name] = lr
        for group in optimizers["aux"].param_groups:
            group["lr"] = aux_lr
        values["aux"] = aux_lr
        return values

    base = cosine_learning_rate(
        step,
        total_steps=schedule_steps,
        warmup_steps=warmup_steps,
        peak_lr=float(profile["learning_rate"]),
        min_lr=float(profile["min_learning_rate"]),
    )
    for name, scale in controller["lr_scale"].items():
        lr = base * float(scale)
        for group in optimizers[f"matrix:{name}"].param_groups:
            group["lr"] = lr
        values[name] = lr
    for group in optimizers["aux"].param_groups:
        group["lr"] = base
    values["aux"] = base
    return values


def zero_grad(optimizers: dict[str, Any]) -> None:
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)


def step_optimizers(optimizers: dict[str, Any]) -> None:
    for opt in optimizers.values():
        opt.step()


def apply_anchor_gradients(matrix_params, anchors, controller) -> None:
    for name, p in matrix_params.items():
        lam = float(controller["anchor_lambda"][name])
        if lam <= 0.0 or p.grad is None:
            continue
        p.grad.add_(p.detach() - anchors[name], alpha=lam)


def spectral_status(parent, model, cfg, seed, step, dest, target, floor):
    parent.weightwatch(model, dest, cfg["weightwatcher"], seed, step)
    table = pd.read_csv(dest)
    return classify_alpha(table, target=target, floor=floor), table


def model_checkpoint_payload(model, step, fingerprint):
    return {"model": model.state_dict(), "step": step, "fingerprint": fingerprint}


def run(args) -> Path:
    parent = import_parent()
    cfg = json.loads(CONFIG_PATH.read_text())
    root = parent.set_environment(Path(args.root).expanduser().resolve())
    source = parent.load_source(cfg)

    sys.path.insert(0, str(REPO / "baseline/nanogpt_one_head/src"))
    import torch
    from rg_nanogpt_one_head.model import GPT, GPTConfig

    if importlib.metadata.version("weightwatcher") != cfg["weightwatcher"]["version"]:
        raise RuntimeError("install the suite-pinned WeightWatcher")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 0 < args.initial_lr_scale <= 1:
        raise ValueError("initial LR scale must be in (0,1]")
    if not 0 < args.hard_backoff <= args.mild_backoff < 1:
        raise ValueError("require 0 < hard_backoff <= mild_backoff < 1")
    if not 0 < args.min_lr_scale <= args.initial_lr_scale:
        raise ValueError("invalid min LR scale")
    if not 0 < args.alpha_floor < args.target_alpha:
        raise ValueError("alpha_floor must be positive and below target_alpha")
    if args.guard_interval < 1 or args.max_retries < 1:
        raise ValueError("guard interval/retries must be positive")

    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)

    model = GPT(GPTConfig(**source["model"])).to(args.device)
    initial_hash = parent.state_digest(model)
    optimizers, matrix_params, profile = build_optimizers(
        model, source, args.optimizer, args.initial_lr_scale
    )
    matrix_names = list(matrix_params)
    train = source["training"]
    effective_batch = int(train["batch_size"]) * int(train["grad_accum_steps"])
    study = parent.Study(cfg, "verbatim", args.stage, args.seed, effective_batch)
    tokens_per_step = effective_batch * int(source["model"]["block_size"])
    schedule_steps = math.ceil(
        int(source["dataset"]["train_tokens"]) * float(profile["lr_schedule_epochs"]) / tokens_per_step
    )
    warmup = min(
        schedule_steps - 1,
        math.ceil(schedule_steps * float(profile["warmup_fraction"])),
    )

    controller = {
        "lr_scale": {name: float(args.initial_lr_scale) for name in matrix_names},
        "anchor_lambda": {name: 0.0 for name in matrix_names},
        "dropout": {str(i): float(source["model"]["dropout"]) for i in range(len(model.blocks))},
    }
    set_block_dropout(model, controller["dropout"])

    run_name = f"spectral_guard_{args.optimizer}"
    run_dir = root / args.stage / run_name / "verbatim" / args.optimizer / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    spectral_dir = run_dir / "spectral"
    rejected_dir = run_dir / "spectral_rejected"
    spectral_dir.mkdir(exist_ok=True)
    rejected_dir.mkdir(exist_ok=True)

    controller_cfg = {
        "type": "spectral_guard_v1",
        "target_alpha": float(args.target_alpha),
        "alpha_floor": float(args.alpha_floor),
        "guard_interval": int(args.guard_interval),
        "mild_backoff": float(args.mild_backoff),
        "hard_backoff": float(args.hard_backoff),
        "min_lr_scale": float(args.min_lr_scale),
        "anchor_start": float(args.anchor_start),
        "anchor_growth": float(args.anchor_growth),
        "max_anchor": float(args.max_anchor),
        "dropout_step": float(args.dropout_step),
        "max_dropout": float(args.max_dropout),
        "max_retries": int(args.max_retries),
        "initial_lr_scale": float(args.initial_lr_scale),
    }
    manifest = {
        "suite": cfg,
        "source_model": source["model"],
        "profile": profile,
        "optimizer_name": args.optimizer,
        "condition": "verbatim",
        "stage": args.stage,
        "seed": args.seed,
        "controller": controller_cfg,
        "initial_model_sha256": initial_hash,
        "data_sha256": study.identity(),
        "runner_sha256": sha256(Path(__file__)),
        "base_runner_sha256": sha256(PARENT_PATH),
        "input_tokens_per_step": tokens_per_step,
        "schedule_steps": schedule_steps,
        "warmup_steps": warmup,
        "device": {
            "device": args.device,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "weightwatcher": importlib.metadata.version("weightwatcher"),
        },
        "objective": "answer_suffix_cross_entropy_plus_adaptive_anchor_gradient",
        "protected_test_used_for_selection": False,
    }
    fingerprint = parent.digest(manifest)
    manifest["fingerprint"] = fingerprint

    manifest_path = run_dir / "manifest.json"
    latest = run_dir / "checkpoint_latest.pt"
    metrics_path = run_dir / "metrics.jsonl"
    events_path = run_dir / "control_events.jsonl"

    accepted_step = 0
    exposures = Counter()
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            raise RuntimeError("run manifest changed; use a new output root")
        if not args.resume:
            raise RuntimeError("run exists; pass --resume or choose a new root")
        if (run_dir / "complete.json").exists():
            print(f"Already complete: {run_dir}")
            return run_dir
        if not latest.is_file():
            raise RuntimeError("resume requested but no checkpoint_latest.pt exists")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError("resume fingerprint mismatch")
        model.load_state_dict(saved["model"])
        load_optimizer_states(optimizers, saved["optimizers"])
        controller = saved["controller"]
        set_block_dropout(model, controller["dropout"])
        restore_rng(torch, args.device, saved["rng"])
        accepted_step = int(saved["step"])
        exposures = Counter(saved["exposures"])
    else:
        atomic_json(manifest_path, manifest)
        atomic_json(run_dir / "probe_inventory.json", [asdict(r) for r in study.audit])
        atomic_json(
            run_dir / "injection_schedule.json",
            {str(k): r.eid for k, r in study.injections.items()},
        )

    permanent = {round(study.steps * i / 16) for i in range(17)}
    permanent.add(study.boundary)
    permanent.add(study.steps)
    permanent = sorted(permanent)

    def save_safe(step: int, last_loss: float | None, alpha_status: dict, table: pd.DataFrame):
        before = parent.state_digest(model)
        audit = parent.evaluate(
            model, study.probes(final=(step == study.steps)), int(train["batch_size"]), args.device
        )
        row = {
            "step": step,
            "phase": "A" if step <= study.boundary else "B",
            "last_train_loss": last_loss,
            "input_tokens": step * tokens_per_step,
            "model_sha256": before,
            "audit": audit,
            "controller": deepcopy(controller),
            "alpha_status": {k: v for k, v in alpha_status.items() if k != "rows"},
        }
        append_jsonl(metrics_path, row)
        table.to_csv(spectral_dir / f"step_{step:08d}.csv", index=False)

        state = {
            "step": step,
            "fingerprint": fingerprint,
            "model": model.state_dict(),
            "optimizers": optimizer_states(optimizers),
            "controller": deepcopy(controller),
            "exposures": dict(exposures),
            "rng": capture_rng(torch, args.device),
        }
        temp = latest.with_suffix(".tmp")
        torch.save(state, temp)
        os.replace(temp, latest)
        if step in permanent:
            dest = run_dir / f"model_step_{step:08d}.pt"
            tmp = dest.with_suffix(".tmp")
            torch.save(model_checkpoint_payload(model, step, fingerprint), tmp)
            os.replace(tmp, dest)

        counts = {r.eid: exposures.get(r.eid, 0) for r in study.audit}
        counts.update(exposures)
        atomic_json(run_dir / "exposures.json", counts)

    if accepted_step == 0 and not latest.exists():
        init_tmp = rejected_dir / "initial_step_00000000.csv"
        alpha0, _ = spectral_status(
            parent, model, cfg, args.seed, 0, init_tmp,
            args.target_alpha, args.alpha_floor
        )
        init_tmp.replace(spectral_dir / "step_00000000.csv")
        audit0 = parent.evaluate(
            model, study.probes(final=False), int(train["batch_size"]), args.device
        )
        row0 = {
            "step": 0,
            "phase": "A",
            "last_train_loss": None,
            "input_tokens": 0,
            "model_sha256": parent.state_digest(model),
            "audit": audit0,
            "controller": deepcopy(controller),
            "alpha_status": {k: v for k, v in alpha0.items() if k != "rows"},
        }
        append_jsonl(metrics_path, row0)
        state = {
            "step": 0,
            "fingerprint": fingerprint,
            "model": model.state_dict(),
            "optimizers": optimizer_states(optimizers),
            "controller": deepcopy(controller),
            "exposures": {},
            "rng": capture_rng(torch, args.device),
        }
        tmp = latest.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, latest)
        dest = run_dir / "model_step_00000000.pt"
        tmp2 = dest.with_suffix(".tmp")
        torch.save(model_checkpoint_payload(model, 0, fingerprint), tmp2)
        os.replace(tmp2, dest)
        counts = {r.eid: 0 for r in study.audit}
        counts["background"] = 0
        atomic_json(run_dir / "exposures.json", counts)
        print(
            f"initial alpha min={alpha0['min_alpha']:.4f}; step 0 is diagnostic only and not gated",
            flush=True,
        )

    while accepted_step < study.steps:
        next_regular = min(study.steps, accepted_step + args.guard_interval)
        future_permanent = [s for s in permanent if accepted_step < s <= next_regular]
        window_end = min(future_permanent) if future_permanent else next_regular

        safe_model = cpu_model_state(model)
        safe_opts = optimizer_states(optimizers)
        safe_rng = capture_rng(torch, args.device)
        anchors = {name: p.detach().clone() for name, p in matrix_params.items()}

        accepted = False
        for attempt in range(1, args.max_retries + 1):
            if attempt > 1:
                model.load_state_dict(safe_model)
                load_optimizer_states(optimizers, safe_opts)
                restore_rng(torch, args.device, safe_rng)
                set_block_dropout(model, controller["dropout"])

            local_exposures = Counter()
            last_loss = None
            model.train()
            for step in range(accepted_step, window_end):
                zero_grad(optimizers)
                set_lrs(
                    optimizers,
                    controller,
                    profile,
                    args.optimizer,
                    step=step,
                    schedule_steps=schedule_steps,
                    warmup_steps=warmup,
                )
                records = study.sample(step)
                for r in records:
                    local_exposures[r.eid] += 1
                total_loss = 0.0
                for i in range(0, effective_batch, int(train["batch_size"])):
                    batch = records[i : i + int(train["batch_size"])]
                    x, y = parent.pack(batch, int(source["model"]["block_size"]))
                    _, loss = model(
                        torch.as_tensor(x, device=args.device),
                        torch.as_tensor(y, device=args.device),
                    )
                    (loss / int(train["grad_accum_steps"])).backward()
                    total_loss += float(loss.detach().cpu())
                apply_anchor_gradients(matrix_params, anchors, controller)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["grad_clip"]))
                step_optimizers(optimizers)
                last_loss = total_loss / int(train["grad_accum_steps"])

            candidate = rejected_dir / f"step_{window_end:08d}_attempt_{attempt:02d}.csv"
            alpha_status, table = spectral_status(
                parent, model, cfg, args.seed, window_end, candidate,
                args.target_alpha, args.alpha_floor
            )
            event = {
                "accepted_step_before": accepted_step,
                "trial_end_step": window_end,
                "attempt": attempt,
                "optimizer": args.optimizer,
                "last_train_loss": last_loss,
                "alpha": {k: v for k, v in alpha_status.items() if k != "rows"},
                "controller_before_adaptation": deepcopy(controller),
            }

            if alpha_status["passed"]:
                exposures.update(local_exposures)
                candidate.unlink(missing_ok=True)
                save_safe(window_end, last_loss, alpha_status, table)
                event["action"] = "accept"
                event["controller_after_adaptation"] = deepcopy(controller)
                append_jsonl(events_path, event)
                print(
                    f"ACCEPT {args.optimizer} seed={args.seed} step={window_end}/{study.steps} "
                    f"loss={last_loss:.6f} min_alpha={alpha_status['min_alpha']:.4f}",
                    flush=True,
                )
                accepted_step = window_end
                accepted = True
                break

            before = adapt_controller(
                controller,
                alpha_status,
                optimizer_name=args.optimizer,
                mild_backoff=args.mild_backoff,
                hard_backoff=args.hard_backoff,
                min_lr_scale=args.min_lr_scale,
                anchor_start=args.anchor_start,
                anchor_growth=args.anchor_growth,
                max_anchor=args.max_anchor,
                dropout_step=args.dropout_step,
                max_dropout=args.max_dropout,
            )
            event["action"] = "rollback_and_adapt"
            event["controller_before_adaptation"] = before
            event["controller_after_adaptation"] = deepcopy(controller)
            append_jsonl(events_path, event)
            print(
                f"ROLLBACK {args.optimizer} seed={args.seed} trial_end={window_end} "
                f"attempt={attempt} min_alpha={alpha_status['min_alpha']:.4f} "
                f"worst={alpha_status['worst_matrix']}",
                flush=True,
            )
            model.load_state_dict(safe_model)
            load_optimizer_states(optimizers, safe_opts)
            restore_rng(torch, args.device, safe_rng)
            set_block_dropout(model, controller["dropout"])

        if not accepted:
            atomic_json(
                run_dir / "failed.json",
                {
                    "status": "failed",
                    "accepted_step": accepted_step,
                    "trial_end_step": window_end,
                    "controller": controller,
                    "reason": "max retries exhausted without satisfying alpha gate",
                },
            )
            raise RuntimeError(
                f"spectral guard failed at window ending {window_end}; max retries={args.max_retries}"
            )

    atomic_json(
        run_dir / "complete.json",
        {
            "status": "complete",
            "steps": study.steps,
            "fingerprint": fingerprint,
            "target_alpha": args.target_alpha,
            "accepted_trajectory_gated": True,
            "controller": controller,
        },
    )
    print(f"\nComplete: {run_dir}", flush=True)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=("muon", "sgd_momentum"), required=True)
    parser.add_argument("--stage", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-alpha", type=float, default=2.0001)
    parser.add_argument("--alpha-floor", type=float, default=1.95)
    parser.add_argument("--guard-interval", type=int, default=125)
    parser.add_argument("--initial-lr-scale", type=float, default=1.0)
    parser.add_argument("--mild-backoff", type=float, default=0.75)
    parser.add_argument("--hard-backoff", type=float, default=0.5)
    parser.add_argument("--min-lr-scale", type=float, default=0.01)
    parser.add_argument("--anchor-start", type=float, default=0.5)
    parser.add_argument("--anchor-growth", type=float, default=2.0)
    parser.add_argument("--max-anchor", type=float, default=32.0)
    parser.add_argument("--dropout-step", type=float, default=0.025)
    parser.add_argument("--max-dropout", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=8)
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped. Resume restarts from the last accepted spectral checkpoint.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
