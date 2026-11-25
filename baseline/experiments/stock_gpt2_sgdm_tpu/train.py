"""Stock GPT-2 / FineWeb with plain SGD + momentum and passive layer tracking.

TPU execution requires a matching local preflight. CPU mode exists for explicit
synthetic smoke checks, not as a practical route to the full training target.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time
import uuid

import numpy as np
import torch

from config import TrainingConfig, BENCHMARK_VALIDATION_TOKENS
from checkpoint import load_checkpoint, recover_checkpoint_aliases, rng_state, restore_rng, save_checkpoint, sha256_file, to_cpu, write_json
from dataset import build_data, MANIFEST_SHA256
from optim import make_optimizer

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parents[1] / "gpt2_small/muon_speedrun"
sys.path.insert(0, str(REFERENCE))
import stock_model as architecture
from runtime import Runtime
# Restore local module priority; old experiment config/tracking must not leak in.
sys.path.remove(str(REFERENCE))


class NumericalDivergence(RuntimeError):
    pass


def record(root, row, *, stdout=True):
    row = {**row, "recorded_unix": time.time()}
    with (Path(root) / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
    if stdout:
        print(json.dumps(row, allow_nan=False), flush=True)


def target_met(config, validation):
    return bool(config.mode == "production" and config.validation_kind == "benchmark"
                and config.validation_offset == 0 and config.validation_tokens == BENCHMARK_VALIDATION_TOKENS and validation
                and validation.get("validation_kind") == "benchmark"
                and validation.get("validation_offset") == 0
                and validation.get("evaluation_tokens") == BENCHMARK_VALIDATION_TOKENS
                and validation.get("full_benchmark_evaluation") is True
                and isinstance(validation.get("val_nll"), (float, int))
                and math.isfinite(validation["val_nll"])
                and validation["val_nll"] <= config.target_val_nll)


def make_model(config, rt):
    architecture.configure_attention(rt.attention("flash" if rt.tpu else "math"))
    model = architecture.make_model(architecture.GPTConfig(
        block_size=config.context_length, vocab_size=config.vocab_size,
        n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd),
        seed=config.seed, device=rt.device)
    for tensor in (*model.parameters(), *model.buffers()):
        rt.replicate(tensor)
    if model.lm_head.weight is not model.transformer.wte.weight:
        raise RuntimeError("Token embedding and output head must remain tied")
    return model


@torch.no_grad()
def clip_gradients(model, maximum):
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not grads:
        raise RuntimeError("Missing gradients")
    norm = sum(gradient.detach().float().square().sum() for gradient in grads).sqrt()
    scale = torch.clamp(maximum / (norm + 1e-6), max=1.0)
    for gradient in grads:
        gradient.mul_(scale)
    return norm


def train_step(model, optimizer, stream, config, rt, step):
    """One global mean-token update; clip once before coupled decay/momentum."""
    before = stream.state_dict()
    optimizer.zero_grad(set_to_none=False)
    total = torch.zeros((), device=rt.device, dtype=torch.float32)
    for _ in range(config.accumulation_steps):
        x, y = stream.next_batch(config.microbatch_sequences, config.context_length)
        with rt.autocast():
            _, loss = model(rt.put(x), rt.put(y), return_logits=False)
        (loss / config.accumulation_steps).backward()
        total += loss.detach().float() / config.accumulation_steps
        rt.step()
    for parameter in model.parameters():
        if parameter.grad is not None:
            rt.replicate(parameter.grad)
    norm = clip_gradients(model, config.grad_clip)
    rt.step(wait=True)
    loss_value, norm_value = float(total.cpu()), float(norm.cpu())
    if not math.isfinite(loss_value) or not math.isfinite(norm_value):
        stream.load_state_dict(before)
        raise NumericalDivergence("Nonfinite loss/gradient before optimizer update")
    learning_rate = config.peak_lr * config.schedule_factor(step)
    for group in optimizer.param_groups:
        # TensorLrSGD multiplies by this tensor instead of converting it through
        # torch.optim.SGD's scalar alpha argument on every XLA update.
        group["lr"] = rt.scalar(learning_rate)
    optimizer.step()
    rt.step(wait=True)
    return {"train_nll": loss_value, "gradient_norm_before_clip": norm_value,
            "gradient_clip_factor": min(1.0, config.grad_clip / (norm_value + 1e-6)),
            "lr": learning_rate}


@torch.no_grad()
def evaluate(model, tokens, config, rt, step):
    begin = time.monotonic()
    model.eval()
    count = config.microbatch_sequences * config.context_length
    end = config.validation_offset + config.validation_tokens
    if end + 1 > len(tokens):
        raise ValueError("Validation slice exceeds shard")
    total = torch.zeros((), device=rt.device, dtype=torch.float32)
    evaluated = 0
    for offset in range(config.validation_offset, end, count):
        buf = torch.from_numpy(np.array(tokens[offset:offset + count + 1], dtype=np.int64))
        x = rt.put(buf[:-1].reshape(config.microbatch_sequences, config.context_length))
        y = rt.put(buf[1:].reshape(config.microbatch_sequences, config.context_length))
        with rt.autocast():
            _, loss = model(x, y, return_logits=False)
        total += loss.detach().float() * count
        evaluated += count
        rt.step()
    rt.step(wait=True)
    nll = float(total.cpu()) / evaluated
    model.train()
    if evaluated != config.validation_tokens or not math.isfinite(nll):
        raise NumericalDivergence("Incomplete/nonfinite validation")
    row = {"kind": "validation", "validation_kind": config.validation_kind,
           "validation_offset": config.validation_offset, "evaluation_tokens": evaluated,
           "step": step, "tokens_seen": step * config.tokens_per_step,
           "val_nll": nll, "val_perplexity": math.exp(nll) if nll < 709 else None,
           "perplexity_overflow": nll >= 709, "seconds": time.monotonic() - begin,
           "full_benchmark_evaluation": config.mode == "production" and
               config.validation_kind == "benchmark" and config.validation_offset == 0 and
               evaluated == BENCHMARK_VALIDATION_TOKENS}
    row["target_met"] = target_met(config, row)
    return row


def _code_hashes():
    names = ("train.py", "config.py", "dataset.py", "checkpoint.py", "metrics.py", "optim.py")
    result = {name: sha256_file(HERE / name) for name in names if (HERE / name).exists()}
    for name in ("runtime.py", "stock_model.py", "stock_config.py"):
        result["reference/" + name] = sha256_file(REFERENCE / name)
    return result


def _read_json(path, fallback):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def run(args):
    config = TrainingConfig.from_json(args.config)
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.no_spectral and config.mode != "smoke":
        raise ValueError("Spectral tracking cannot be disabled for production")
    if args.device == "tpu":
        if config.mode != "production" or config.microbatch_sequences % 8:
            raise ValueError("TPU requires production model and a global microbatch divisible by eight")
        proof = _read_json(root / "PREFLIGHT.json", {})
        if not (proof.get("status") == "passed" and proof.get("config_sha256") == config.config_sha256
                and proof.get("device") == "tpu" and proof.get("data_dir") == str(args.data_dir.resolve())):
            raise RuntimeError("Missing/mismatched TPU preflight; use launch.py preflight/start")
    latest_path = root / "checkpoints/latest.pt"
    if args.resume:
        if not latest_path.exists() or not (root / "config.json").exists():
            raise FileNotFoundError("Resume requires this run's saved config and checkpoint")
        if TrainingConfig.from_json(root / "config.json").config_sha256 != config.config_sha256:
            raise ValueError("Saved run configuration differs")
    elif any((root / name).exists() for name in ("config.json", "manifest.json", "metrics.jsonl", "checkpoints")):
        raise FileExistsError("Run already contains training state; use --resume or a new run directory")
    lock_path = root / "TRAINER.lock"
    # flock is released even after SIGKILL; a stale filename is harmless.
    import fcntl
    lock = lock_path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    config.write_json(root / "config.json")
    (root / "TRAINING_DONE").unlink(missing_ok=True)
    torch.set_num_threads(args.cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rt = Runtime(args.device, root / "xla-cache" if args.device == "tpu" else None)
    if rt.tpu:
        rt.xm.set_rng_state(config.seed, rt.device)
    model = make_model(config, rt)
    optimizer = make_optimizer(model, config)
    stream, validation_tokens, data_receipts = build_data(config, args.data_dir)
    code_hashes = _code_hashes()
    run_id = str(uuid.uuid4())
    manifest = {"schema_version": 1, "run_id": run_id, "config_sha256": config.config_sha256,
                "architecture": config.architecture, "model_source_sha256": architecture.SOURCE_SHA256,
                "parameters": sum(p.numel() for p in model.parameters()), "optimizer": "TensorLrSGD",
                "optimizer_equivalence": "Conventional PyTorch SGD momentum/decay equations and state, with tensor multiplication for dynamic learning rates",
                "optimizer_convention": "unnormalized momentum; dampening=0; Nesterov=false; coupled matrix-only decay after gradient clipping",
                "layer_metrics_control_training": False, "code_sha256": code_hashes,
                "data_manifest_sha256": MANIFEST_SHA256, "data_receipts": data_receipts,
                "torch": torch.__version__, "python": platform.python_version(), "device": args.device,
                "numerics": "FP32 weights/momentum; BF16 TPU autocast" if rt.tpu else "FP32 CPU smoke/reference",
                "attention": "TPU causal flash attention, explicit head_dim**-0.5 scale" if rt.tpu else "PyTorch causal SDPA",
                "tpu_convergence_verified": False, "created_unix": time.time(),
                "comparison_limits": ["Sequential continuous data order differs from shuffled CUDA reference",
                                      "SGDM search candidates are unverified; no optimum is guaranteed",
                                      "CPU resume parity does not establish bitwise TPU resume parity"]}
    step, best_nll, latest_validation = 0, None, None
    best_validation, first_target_validation = None, None
    prior_active_seconds = 0.0
    elapsed = {"training_seconds": 0.0, "validation_seconds": 0.0, "checkpoint_seconds": 0.0,
               "spectral_queue_wait_seconds": 0.0}
    expected_steps = set(_read_json(root / "tracking/EXPECTED_STEPS.json", {"steps": []})["steps"])
    prior = None
    if args.resume:
        saved_manifest = _read_json(root / "manifest.json", {})
        prior = load_checkpoint(latest_path, config, expected_run_id=saved_manifest.get("run_id"))
        if prior["manifest"]["code_sha256"] != code_hashes:
            raise ValueError("Training/metric code changed since checkpoint; restore its original checkout")
        if prior["manifest"]["device"] != args.device:
            raise ValueError("Resume requires the original execution device kind")
        recover_checkpoint_aliases(root, prior)
        model.load_state_dict(prior["model"], strict=True)
        optimizer.load_state_dict(prior["optimizer"])
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    rt.replicate(value)
        stream.load_state_dict(prior["data_cursor"])
        if stream.consumed != prior["tokens_seen"]:
            raise ValueError("Data cursor is not at the saved optimizer boundary")
        restore_rng(prior["rng"], rt)
        step, run_id, manifest = prior["step"], prior["run_id"], prior["manifest"]
        latest_validation, best_nll = prior["latest_validation"], prior["best_val_nll"]
        best_validation = prior.get("best_validation")
        first_target_validation = prior.get("first_target_validation")
        prior_active_seconds = prior.get("active_run_seconds", 0.0)
        old_status = _read_json(root / "TRAINING_STATUS.json", {})
        if old_status.get("step") == step and old_status.get("run_id") == run_id:
            prior_active_seconds = max(prior_active_seconds, old_status.get("active_run_seconds", 0.0))
        elapsed.update(prior["elapsed"])
    write_json(root / "manifest.json", manifest)
    write_json(root / "data_receipts.json", data_receipts)
    stop_requested = {"signal": None}
    def request_stop(signum, frame):
        stop_requested["signal"] = signum
    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    began = time.monotonic()
    initial_step = step
    terminal_status, error = "training", None
    last_persisted = (prior["step"], prior.get("spectral_required", False)) if prior else None

    def status(name, *, terminal=False):
        tracking = _read_json(root / "TRACKING_STATUS.json", {"complete": False, "failed": 0, "pending": None})
        row = {"status": name, "terminal": terminal, "run_id": run_id,
               "config_sha256": config.config_sha256, "candidate_id": config.candidate_id,
               "stage_id": config.stage_id, "seed": config.seed, "mode": config.mode,
               "step": step, "tokens_seen": step * config.tokens_per_step, "budget_tokens": config.budget_tokens,
               "full_training_recipe_completed": step == config.max_steps and name not in {"failed", "diverged"},
               "train_finished": terminal, "target_met": target_met(config, latest_validation),
               "target_ever_met": first_target_validation is not None,
               "first_target_validation": first_target_validation, "best_validation": best_validation,
               "validation": latest_validation, "best_val_nll": best_nll, "elapsed": elapsed,
               "active_run_seconds": prior_active_seconds + time.monotonic() - began,
               "timing_definition": "Active training-loop wall time, including validation/checkpoint/queue wait; excludes preflight, data preparation, model setup and downtime between sessions. Replay after a hard crash is not included before the saved checkpoint.",
               "spectral_status": tracking, "spectral_required": not args.no_spectral,
               "expected_snapshot_steps": sorted(expected_steps), "error": error, "updated_unix": time.time()}
        write_json(root / "TRAINING_STATUS.json", row)
        write_json(root / "status.json", row)
        return row

    def snapshot(payload):
        if args.no_spectral:
            return
        from metrics import queue_snapshot
        expected_steps.add(payload["step"])
        write_json(root / "tracking/EXPECTED_STEPS.json", {"steps": sorted(expected_steps)})
        waiting = time.monotonic()
        announced = False
        while True:
            queued = queue_snapshot(root, payload, n_layers=config.n_layer,
                                    include_embeddings=config.include_embeddings,
                                    max_pending=config.max_pending_snapshots)
            if queued["status"] != "queue_full":
                break
            if not announced:
                print("Waiting for CPU spectral worker: snapshot queue is full", flush=True)
                announced = True
            if stop_requested["signal"] is not None:
                raise InterruptedError("Paused while waiting to queue required spectral snapshot; resume will queue it")
            if time.monotonic() - waiting > args.spectral_timeout_seconds:
                raise RuntimeError("Spectral queue made no room before timeout; inspect tracking log and resume")
            time.sleep(1)
        elapsed["spectral_queue_wait_seconds"] += time.monotonic() - waiting

    def persist(*, spectral=False, is_best=False):
        nonlocal last_persisted
        required_spectral = spectral and not args.no_spectral
        if last_persisted is not None and last_persisted[0] == step and (not required_spectral or last_persisted[1]):
            return
        checkpoint_began = time.monotonic()
        rt.step(wait=True)
        payload = {"schema_version": 1, "run_id": run_id, "step": step,
                   "tokens_seen": step * config.tokens_per_step, "config": config.as_dict(),
                   "config_sha256": config.config_sha256, "model": to_cpu(model.state_dict()),
                   "optimizer": to_cpu(optimizer.state_dict()), "data_cursor": stream.state_dict(),
                   "rng": rng_state(rt), "manifest": manifest, "latest_validation": latest_validation,
                   "validation": latest_validation if latest_validation and latest_validation["step"] == step else None,
                   "best_val_nll": best_nll, "elapsed": dict(elapsed),
                   "best_validation": best_validation, "first_target_validation": first_target_validation,
                   "active_run_seconds": prior_active_seconds + time.monotonic() - began,
                   "next_lr_factor": config.schedule_factor(step), "spectral_required": spectral and not args.no_spectral}
        save_checkpoint(root, payload, best=is_best,
                        first_target=first_target_validation is not None and first_target_validation["step"] == step)
        last_persisted = (step, spectral and not args.no_spectral)
        elapsed["checkpoint_seconds"] += time.monotonic() - checkpoint_began
        if spectral:
            snapshot(payload)

    def validate_and_save(*, spectral=False):
        nonlocal latest_validation, best_nll, best_validation, first_target_validation
        if latest_validation is None or latest_validation["step"] != step:
            latest_validation = evaluate(model, validation_tokens, config, rt, step)
            elapsed["validation_seconds"] += latest_validation["seconds"]
            record(root, latest_validation)
            write_json(root / "latest_validation.json", latest_validation)
        is_best = best_nll is None or latest_validation["val_nll"] < best_nll
        if is_best:
            best_nll = latest_validation["val_nll"]
            best_validation = dict(latest_validation)
        if first_target_validation is None and target_met(config, latest_validation):
            first_target_validation = {**latest_validation,
                "active_run_seconds": prior_active_seconds + time.monotonic() - began,
                "cumulative_training_seconds": elapsed["training_seconds"],
                "cumulative_validation_seconds": elapsed["validation_seconds"]}
        persist(spectral=spectral, is_best=is_best)

    try:
        status("initializing")
        if prior is None:
            validate_and_save(spectral=True)
        elif prior.get("spectral_required"):
            snapshot(prior)  # Recover a crash between checkpoint replacement and snapshot queueing.
        while step < config.max_steps:
            if stop_requested["signal"] is not None or (args.max_steps_this_session is not None and
                    step - initial_step >= args.max_steps_this_session) or (args.max_seconds is not None and
                    time.monotonic() - began >= args.max_seconds):
                terminal_status = "paused"
                break
            if config.stop_at_target and target_met(config, latest_validation):
                terminal_status = "target_reached"
                break
            update_began = time.monotonic()
            row = train_step(model, optimizer, stream, config, rt, step)
            duration = time.monotonic() - update_began
            elapsed["training_seconds"] += duration
            step += 1
            if step <= 5 or step % args.log_every == 0 or step == config.max_steps:
                record(root, {"kind": "train", "step": step, "tokens_seen": step * config.tokens_per_step,
                              "seconds": duration, "tokens_per_second": config.tokens_per_step / duration, **row})
                status("training")
            tokens = step * config.tokens_per_step
            spectral_due = tokens % config.spectral_interval_tokens == 0
            evaluation_due = tokens % config.eval_interval_tokens == 0
            if spectral_due or evaluation_due or step == config.max_steps:
                validate_and_save(spectral=spectral_due or step == config.max_steps)
            elif tokens % config.checkpoint_interval_tokens == 0:
                persist()
        if step == config.max_steps:
            terminal_status = "budget_complete_target_met" if target_met(config, latest_validation) else "budget_complete_target_not_met"
        validate_and_save(spectral=True)
    except NumericalDivergence as exc:
        terminal_status, error = "diverged", str(exc)
        # Preserve the last finite checkpoint; never replace it with divergent weights.
    except InterruptedError as exc:
        terminal_status, error = "paused", str(exc)
    except Exception as exc:
        terminal_status, error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        write_json(root / "tracking/EXPECTED_STEPS.json", {"steps": sorted(expected_steps)})
        final_status = status(terminal_status, terminal=True)
        write_json(root / "TRAINING_DONE", {"status": terminal_status, "step": step, "error": error})
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    print(json.dumps(final_status, allow_nan=False), flush=True)
    if terminal_status in {"diverged", "failed"}:
        raise RuntimeError(f"{terminal_status}: {error}")
    return final_status


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "tpu"), default="tpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-spectral", action="store_true", help="Explicit synthetic smoke checks only")
    parser.add_argument("--max-steps-this-session", type=int, help="Pause at an optimizer boundary without changing the schedule")
    parser.add_argument("--max-seconds", type=float, help="Soft session limit; final validation/checkpoint/drain adds time")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--spectral-timeout-seconds", type=float, default=3600)
    args = parser.parse_args(argv)
    for field in ("cpu_threads", "log_every", "spectral_timeout_seconds"):
        if getattr(args, field) <= 0:
            parser.error(field + " must be positive")
    if args.max_steps_this_session is not None and args.max_steps_this_session < 1:
        parser.error("max-steps-this-session must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("max-seconds must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
