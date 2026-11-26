"""Deterministic, resumable one-thread CPU training for the frozen MLP study."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch
from torch import nn

try:
    from .data import load_data
    from .optimizer import make_optimizer
except ImportError:
    from data import load_data
    from optimizer import make_optimizer


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        for layer in (self.fc1, self.fc2, self.fc3, self.fc4):
            x = torch.relu(layer(x))
        return self.fc5(x).squeeze(-1)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def learning_rate(step, peak_lr, plan):
    """Update 1..warmup ramps linearly; cosine reaches floor at final update."""
    total = int(plan.get("total_updates", 20000))
    warmup = int(plan.get("warmup_updates", 100))
    floor = float(plan.get("minimum_lr_ratio", 0.1))
    if not 1 <= step <= total or not 0 <= warmup < total or not 0 <= floor <= 1:
        raise ValueError("Invalid learning-rate schedule or step")
    if warmup and step <= warmup:
        return float(peak_lr) * step / warmup
    progress = (step - warmup) / (total - warmup)
    return float(peak_lr) * (floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2)


def _execution_metadata():
    packages = {}
    for package in ("torch", "numpy", "scipy", "scikit-learn", "weightwatcher", "powerlaw"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    source = {}
    for name in ("data.py", "train.py", "optimizer.py", "spectral.py"):
        path = Path(__file__).parent / name
        if path.exists():
            source[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=Path(__file__).parent, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    names = ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "GITHUB_SHA",
             "GITHUB_REF", "GITHUB_REPOSITORY", "GITHUB_WORKFLOW", "RUNNER_OS",
             "RUNNER_ARCH", "RUNNER_NAME", "ImageOS", "ImageVersion")
    return {"python": platform.python_version(), "packages": packages,
            "platform": platform.platform(), "cpu": cpu, "precision": "float32",
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "thread_environment": {k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "github": {key: os.environ.get(key) for key in names},
            "git_commit": commit, "source_sha256": source,
            "source_bundle_sha256": canonical_sha256(source)}


@torch.no_grad()
def evaluate(model, arrays, split):
    x = torch.from_numpy(arrays[f"x_{split}"])
    predictions = torch.cat([model(chunk) for chunk in x.split(1024)]).double().numpy()
    if not np.isfinite(predictions).all():
        raise FloatingPointError(f"Non-finite {split} predictions")
    target_std = arrays[f"y_{split}"].astype(np.float64)
    prediction_original = predictions * float(arrays["y_scale"]) + float(arrays["y_mean"])
    residual = prediction_original - arrays[f"y_{split}_original"]
    return {"mse": float(np.mean(residual**2)),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "standardized_mse": float(np.mean((predictions - target_std)**2))}


def _rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


@torch.no_grad()
def optimizer_diagnostics(model, optimizer, config, before, preclip_grad_norm):
    """Observe a completed update without changing weights, moments or RNG.

    The comparison buffer follows this run's own clipped gradient history.
    It is not the buffer from the separately trained SGDM control.
    """
    native = getattr(optimizer, "optimizer", optimizer)
    by_parameter = {p: group for group in native.param_groups for p in group["params"]}
    matrices = {}
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        group = by_parameter[parameter]
        state = native.state[parameter]
        lr = float(group["lr"])
        if config["optimizer"] == "sgdm":
            buffer = state["momentum_buffer"].double()
            reference_lr = lr
        else:
            buffer = state["exp_avg"].double() / (1.0 - group["betas"][0])
            reference_lr = lr * (float(config["reference_lr_ratio"])
                                 if config["optimizer"] == "modified_adamw" else 1.0)
        delta = (parameter.detach() - before[name]).double()
        update_norm = float(torch.linalg.vector_norm(delta))
        buffer_norm = float(torch.linalg.vector_norm(buffer))
        reference_norm = reference_lr * buffer_norm
        finite = all(math.isfinite(v) for v in (update_norm, buffer_norm, reference_norm))
        cosine = None
        if finite and update_norm > 0 and buffer_norm > 0:
            cosine = float(torch.dot(delta.flatten(), -buffer.flatten()) / (update_norm * buffer_norm))
            cosine = min(1.0, max(-1.0, cosine))
        matrices[name] = {"realized_update_norm": update_norm if finite else None,
                          "own_momentum_buffer_norm": buffer_norm if finite else None,
                          "momentum_reference_lr": reference_lr,
                          "momentum_reference_update_norm": reference_norm if finite else None,
                          "cosine_with_own_sgdm_direction": cosine, "finite": finite}
    return {"available": True, "preclip_global_gradient_norm": float(preclip_grad_norm),
            "reference_history": "this run's own clipped gradients",
            "reference_lr_definition": "current LR; multiplied by reference_lr_ratio only for modified AdamW",
            "matrices": matrices}


def train(plan, config, data_dir, output_dir, seed, phase="confirmation",
          stop_step=None, resume=True):
    """Acquire exclusive ownership of one output directory before training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another worker owns output directory {output_dir}; refusing duplicate training") from error
        try:
            return _train(plan, config, data_dir, output_dir, seed, phase, stop_step, resume)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _train(plan, config, data_dir, output_dir, seed, phase="confirmation",
           stop_step=None, resume=True):
    """Train to an absolute update count, resuming a matching run exactly.

    A calibration stop uses the initial segment of the final 20,000-step
    schedule. It never evaluates the test set or calls the spectral fitter.
    """
    if phase not in ("calibration", "confirmation"):
        raise ValueError("phase must be calibration or confirmation")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    total = int(plan.get("total_updates", 20000))
    target = total if stop_step is None else int(stop_step)
    if not 0 <= target <= total:
        raise ValueError("stop_step must be between zero and total_updates")
    evaluate_every = int(plan.get("eval_every", 500))
    batch_size = int(plan.get("batch_size", 128))
    if evaluate_every < 1 or batch_size < 1:
        raise ValueError("Invalid cadence or batch size")
    checkpoint_updates = {int(v) for v in plan.get("checkpoint_updates", [2000, 20000])}
    arrays, manifest = load_data(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256((Path(data_dir) / "manifest.json").read_bytes()).hexdigest()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    model = MLP().float()
    optimizer = make_optimizer(model, config)
    sampler_seed = int(seed) + 1000000
    generator = torch.Generator(device="cpu").manual_seed(sampler_seed)
    initial_hash = _model_hash(model)
    execution = _execution_metadata()
    runtime_signature = {key: execution[key] for key in (
        "python", "packages", "precision", "torch_num_threads", "torch_num_interop_threads",
        "deterministic_algorithms", "thread_environment")}
    identity = {"optimizer": config["optimizer"], "seed": int(seed), "phase": phase,
                "config": config, "plan_sha256": canonical_sha256(plan),
                "data_manifest_sha256": manifest_hash,
                "numerical_runtime_sha256": canonical_sha256(runtime_signature),
                "source_bundle_sha256": execution["source_bundle_sha256"]}
    result_path, resume_path = output_dir / "result.json", output_dir / "resume.pt"
    result = {**identity, "schema_version": 1, "status": "running", "completed_updates": 0,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "initialization_sha256": initial_hash, "sampler_seed": sampler_seed,
              "minibatch_sha256": "0" * 64,
              "minibatch_hash_method": "SHA256(previous 32-byte digest || int64 sampled indices), starting 32 zero bytes",
              "minibatch_rng_sha256": _tensor_hash(generator.get_state()),
              "metrics_unit": {"rmse": "100000 USD", "mae": "100000 USD",
                               "mse": "(100000 USD)^2", "standardized_mse": "standardized target squared"},
              "history": [], "execution": execution, "resume_executions": []}
    completed = 0
    elapsed_prior = 0.0
    minibatch_digest = "0" * 64
    if result_path.exists() or resume_path.exists():
        if not resume:
            raise FileExistsError("Existing result/checkpoint; refusing to overwrite with --no-resume")
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if any(previous.get(k) != v for k, v in identity.items()):
                raise ValueError("Existing result identity differs from requested run")
            if previous.get("completed_updates", 0) >= target and previous.get("status") in ("complete", "paused"):
                if not resume_path.exists():
                    raise ValueError("Result exists without its resumable checkpoint")
                retained = torch.load(resume_path, map_location="cpu", weights_only=False)
                if (retained.get("identity") != identity
                        or retained.get("completed_updates") != previous["completed_updates"]):
                    raise ValueError("Retained checkpoint does not match completed result")
                return previous
        if not resume_path.exists():
            raise ValueError("Existing result has no resumable checkpoint")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume checkpoint identity differs from requested run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["minibatch_rng"])
        _restore_rng(checkpoint["rng"])
        result = checkpoint["result"]
        completed = checkpoint["completed_updates"]
        if completed > target:
            raise ValueError("Requested stop predates retained checkpoint")
        elapsed_prior = checkpoint["elapsed_seconds"]
        minibatch_digest = checkpoint["minibatch_sha256"]
        result["resume_executions"].append(execution)
        result["status"] = "running"

    training_x = torch.from_numpy(arrays["x_train"])
    training_y = torch.from_numpy(arrays["y_train"])
    started = time.perf_counter()
    latest_diagnostics = {"available": False, "reason": "initialization; no optimizer update yet"}

    def persist(measure=False):
        nonlocal result
        elapsed = elapsed_prior + time.perf_counter() - started
        if measure:
            saved_rng = _rng_state()
            model.eval()
            try:
                observation = {"step": completed,
                               "train": evaluate(model, arrays, "train"),
                               "validation": evaluate(model, arrays, "validation"),
                               "test": evaluate(model, arrays, "test") if phase == "confirmation" else None,
                               "spectral": None, "elapsed_seconds": elapsed,
                               "optimizer_diagnostics": latest_diagnostics}
                if phase == "confirmation":
                    try:
                        from .spectral import analyze_model
                    except ImportError:
                        from spectral import analyze_model
                    observation["spectral"] = analyze_model(model)
                result["history"] = [row for row in result["history"] if row["step"] != completed]
                result["history"].append(observation)
                result["history"].sort(key=lambda row: row["step"])
                alpha = (observation["spectral"] or {}).get("confidence_tail_alpha")
                alpha_text = f"{alpha:.6f}" if alpha is not None else "n/a"
                print(f"{phase} {config['optimizer']} seed={seed} step={completed} "
                      f"train_RMSE={observation['train']['rmse']:.6f} "
                      f"val_RMSE={observation['validation']['rmse']:.6f} alpha={alpha_text}", flush=True)
            finally:
                _restore_rng(saved_rng)
                model.train()
        elapsed = elapsed_prior + time.perf_counter() - started
        result.update(completed_updates=completed, elapsed_seconds=elapsed,
                      minibatch_sha256=minibatch_digest,
                      minibatch_rng_sha256=_tensor_hash(generator.get_state()),
                      final_weights_sha256=_model_hash(model))
        checkpoint = {"identity": identity, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "rng": _rng_state(),
                      "minibatch_rng": generator.get_state(), "minibatch_sha256": minibatch_digest,
                      "completed_updates": completed, "result": result, "elapsed_seconds": elapsed,
                      "plan": plan, "config": config}
        # Checkpoint first: result.json must never advertise an unsaved state.
        _atomic_checkpoint(resume_path, checkpoint)
        if completed in checkpoint_updates:
            _atomic_checkpoint(output_dir / f"checkpoint_step_{completed}.pt", checkpoint)
        _atomic_json(result_path, result)

    if not result["history"]:
        persist(measure=True)
    try:
        model.train()
        for step in range(completed + 1, target + 1):
            lr = learning_rate(step, float(config["peak_lr"]), plan)
            for group in optimizer.param_groups:
                group["lr"] = lr
            indices = torch.randint(len(training_y), (batch_size,), generator=generator)
            minibatch_digest = hashlib.sha256(bytes.fromhex(minibatch_digest)
                                               + indices.numpy().tobytes()).hexdigest()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(training_x[indices])
            loss = torch.nn.functional.mse_loss(prediction, training_y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at update {step}")
            loss.backward()
            preclip_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(plan.get("grad_clip", 1.0)), error_if_nonfinite=True)
            measure_update = step % evaluate_every == 0 or step in checkpoint_updates or step == target
            before = {name: p.detach().clone() for name, p in model.named_parameters()
                      if p.ndim == 2} if measure_update else None
            optimizer.step()
            completed = step
            if measure_update:
                latest_diagnostics = optimizer_diagnostics(model, optimizer, config, before, preclip_norm)
                persist(measure=True)
        result["status"] = "complete" if completed == total or phase == "calibration" else "paused"
        persist(measure=False)
        return result
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["failure_after_updates"] = completed
        _atomic_json(result_path, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--config", required=True, help="JSON file containing one optimizer configuration")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--phase", default="confirmation", choices=("calibration", "confirmation"))
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    outcome = train(json.loads(Path(args.plan).read_text()), json.loads(Path(args.config).read_text()),
                    args.data_dir, args.output, args.seed, args.phase, args.stop_step, not args.no_resume)
    print(json.dumps({key: outcome[key] for key in ("optimizer", "seed", "status", "completed_updates")},
                     sort_keys=True))
