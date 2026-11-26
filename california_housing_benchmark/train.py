"""CPU reproduction of the published MLP protocol, with resumable extensions.

The primary model is selected only before the first validation patience stop.
Confirmation runs then continue along the same training trajectory to 20,000
updates when necessary. The later trajectory never changes the primary model.
"""
import argparse
from copy import deepcopy
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

import delu
import numpy as np
import tabm
import torch
import sklearn.metrics
from torch import nn

from .data import load_data
from .optimizer import make_optimizer


class MLP(nn.Module):
    def __init__(self, model_spec=None):
        super().__init__()
        spec = dict(model_spec or {})
        spec = spec.get("backbone", spec)
        self.backbone = tabm.MLPBackbone(
            d_in=int(spec.get("d_in", 8)), n_blocks=int(spec.get("n_blocks", 2)),
            d_block=int(spec.get("d_block", 784)),
            dropout=float(spec.get("dropout", 0.3690073279296028)))
        self.output = nn.Linear(int(spec.get("d_block", 784)), 1)

    def forward(self, x):
        return self.output(self.backbone(x)).squeeze(-1)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_hash(model_or_state):
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def execution_metadata():
    packages = {}
    for package in ("torch", "numpy", "scipy", "scikit-learn", "tabm", "delu",
                    "weightwatcher", "powerlaw"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    source = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ("data.py", "train.py", "optimizer.py", "spectral.py")
              if Path(__file__).with_name(name).exists()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=Path(__file__).parent, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    github_names = ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "GITHUB_SHA",
                    "GITHUB_REF", "GITHUB_REPOSITORY", "GITHUB_WORKFLOW", "RUNNER_OS",
                    "RUNNER_ARCH", "RUNNER_NAME", "ImageOS", "ImageVersion")
    return {"python": platform.python_version(), "packages": packages,
            "platform": platform.platform(), "cpu": cpu, "precision": "float32",
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "thread_environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "github": {key: os.environ.get(key) for key in github_names},
            "git_commit": commit, "source_sha256": source,
            "source_bundle_sha256": canonical_sha256(source)}


@torch.no_grad()
def evaluate(model, arrays, split, batch_size=32768):
    """Match upstream float32 inverse scaling and sklearn metric reductions."""
    model.eval()
    x = torch.from_numpy(arrays[f"x_{split}"])
    prediction_std = torch.cat([model(chunk) for chunk in x.split(batch_size)]).numpy()
    if not np.isfinite(prediction_std).all():
        raise FloatingPointError(f"Non-finite {split} predictions")
    prediction = prediction_std.copy()
    prediction *= float(arrays["y_scale"])
    prediction += float(arrays["y_mean"])
    target = arrays[f"y_{split}_original"]
    # The original benchmark passes float32 predictions and targets directly
    # to these sklearn functions. In particular, checkpoint selection must
    # retain that rounding behavior because validation improvements are strict.
    mse = float(sklearn.metrics.mean_squared_error(target, prediction))
    residual = prediction.astype(np.float64) - target.astype(np.float64)
    return {"mse": mse, "rmse": float(mse ** 0.5),
            "mae": float(sklearn.metrics.mean_absolute_error(target, prediction)),
            "r2": float(sklearn.metrics.r2_score(target, prediction)),
            "mape_percent": float(100 * np.mean(abs(residual) / np.maximum(abs(target), 1e-12))),
            "standardized_mse": float(np.mean((prediction_std.astype(np.float64)
                                                 - arrays[f"y_{split}"].astype(np.float64))**2))}


def train(plan, config, data_dir, output_dir, seed, phase="confirmation", stop_step=None, resume=True):
    """Run or resume one immutable recipe, guarded against duplicate workers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {output_dir}; duplicate refused") from exc
        try:
            return _train(plan, config, data_dir, output_dir, seed, phase, stop_step, resume)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _train(plan, config, data_dir, output_dir, seed, phase, stop_step, resume):
    if phase not in ("calibration", "confirmation", "reference"):
        raise ValueError("Unknown phase")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    arrays, manifest = load_data(data_dir)
    del manifest
    total = int(plan.get("total_updates", 20000))
    safety = int(plan.get("safety_max_updates", 100000))
    calibration_limit = int(plan.get("calibration", {}).get("max_updates", 20000))
    limit = min(safety, int(stop_step) if stop_step is not None else (
        calibration_limit if phase == "calibration" else safety))
    every = int(plan.get("eval_every", 500))
    patience = int(plan.get("patience", 16))
    batch_size = int(plan.get("batch_size", 256))
    if min(total, safety, limit, every, patience, batch_size) <= 0 or safety < total:
        raise ValueError("Invalid budgets, cadence, or patience")
    checkpoints = set(map(int, plan.get("checkpoint_updates", [2000, 20000])))
    include_test = phase in ("confirmation", "reference")
    include_spectral = phase == "confirmation" and bool(plan.get("measure_spectra", True))
    eval_batch = int(plan.get("eval_batch_size", 32768))

    delu.random.seed(int(seed))
    model = MLP(plan.get("model")).float()
    optimizer = make_optimizer(model, config)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial_hash = model_hash(model)
    execution = execution_metadata()
    runtime = {key: execution[key] for key in ("python", "packages", "precision",
               "torch_num_threads", "torch_num_interop_threads", "deterministic_algorithms",
               "thread_environment")}
    identity = {"optimizer": config["optimizer"], "seed": int(seed), "phase": phase,
                "config": config, "plan_sha256": canonical_sha256(plan),
                "data_manifest_sha256": hashlib.sha256((Path(data_dir) / "manifest.json").read_bytes()).hexdigest(),
                "numerical_runtime_sha256": canonical_sha256(runtime),
                "source_bundle_sha256": execution["source_bundle_sha256"]}
    n_train = len(arrays["y_train"])
    epoch_size = math.ceil(n_train / batch_size)
    result = {**identity, "schema_version": 1, "status": "running", "completed_updates": 0,
              "completed_epochs": 0, "parameter_count": sum(p.numel() for p in model.parameters()),
              "epoch_size": epoch_size, "initialization_sha256": initial_hash,
              "sampler_seed": int(seed), "sampling": "one torch.randperm per epoch, including final partial minibatch",
              "minibatch_sha256": "0" * 64,
              "minibatch_hash_method": "SHA256(previous 32-byte digest || int64 indices), initially zero digest",
              "metrics_unit": {"rmse": "100000 USD", "mae": "100000 USD", "mse": "(100000 USD)^2"},
              "history": [], "epoch_history": [], "primary_best_validation": None,
              "paper_stopping": {"reached": False, "patience": patience, "censored": False},
              "execution": execution, "resume_executions": [],
              "protocol": "constant LR; first validation-patience model is primary; later fixed-budget continuation is secondary"}
    completed = epoch = offset = bad_epochs = 0
    permutation = None
    batch_loss_sum = 0.0
    epoch_examples = 0
    elapsed_prior = 0.0
    minibatch_digest = "0" * 64
    best_weights = None
    best_state = {"validation_rmse": None, "step": None, "epoch": None,
                  "minibatch_sha256": None}
    result_path, resume_path = output_dir / "result.json", output_dir / "resume.pt"

    if result_path.exists() or resume_path.exists():
        if not resume:
            raise FileExistsError("Existing run; --no-resume cannot overwrite it")
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if any(previous.get(k) != v for k, v in identity.items()):
                raise ValueError("Existing run identity differs")
            if previous.get("status") in ("complete", "censored"):
                if not resume_path.exists():
                    raise ValueError("Completed result lacks resumable state")
                retained = torch.load(resume_path, map_location="cpu", weights_only=False)
                if (retained.get("identity") != identity or retained.get("completed_updates")
                        != previous["completed_updates"]):
                    raise ValueError("Completed result/checkpoint mismatch")
                return previous
        if not resume_path.exists():
            raise ValueError("Existing run lacks resumable state")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Resume identity differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["batch_generator"])
        restore_rng(state["rng"])
        completed, epoch = state["completed_updates"], state["completed_epochs"]
        if completed > limit:
            raise ValueError("Requested pause predates retained progress")
        permutation, offset = state["epoch_permutation"], state["next_offset"]
        batch_loss_sum, epoch_examples = state["epoch_loss_sum"], state["epoch_examples"]
        best_weights, best_state = state["best_weights"], state["best_state"]
        bad_epochs = state["bad_epochs"]
        elapsed_prior, minibatch_digest = state["elapsed_seconds"], state["minibatch_sha256"]
        result = state["result"]
        result["resume_executions"].append(execution)
        result["status"] = "running"

    x_train = torch.from_numpy(arrays["x_train"])
    y_train = torch.from_numpy(arrays["y_train"])
    started = time.perf_counter()

    def observation(measured_model, step, observed_epoch, digest, spectral):
        saved = rng_state()
        was_training = measured_model.training
        try:
            measured_model.eval()
            row = {"step": step, "epoch": observed_epoch,
                   "train": evaluate(measured_model, arrays, "train", eval_batch),
                   "validation": evaluate(measured_model, arrays, "validation", eval_batch),
                   "test": evaluate(measured_model, arrays, "test", eval_batch) if include_test else None,
                   "spectral": None, "minibatch_sha256": digest,
                   "elapsed_seconds": elapsed_prior + time.perf_counter() - started}
            if spectral:
                from .spectral import measure
                row["spectral"] = measure(measured_model)
            return row
        finally:
            restore_rng(saved)
            measured_model.train(was_training)

    def measure_current():
        row = observation(model, completed, epoch, minibatch_digest, include_spectral)
        row["minibatch_rng_sha256"] = tensor_hash(generator.get_state())
        row["training_rng_sha256"] = tensor_hash(torch.get_rng_state())
        result["history"] = [x for x in result["history"] if x["step"] != completed] + [row]
        result["history"].sort(key=lambda x: x["step"])

    def freeze_primary(reason, censored=False):
        nonlocal best_weights, best_state
        if result["primary_best_validation"] is not None:
            return
        if best_weights is None:
            validation = evaluate(model, arrays, "validation", eval_batch)
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_state = {"validation_rmse": validation["rmse"], "step": completed,
                          "epoch": epoch, "minibatch_sha256": minibatch_digest}
        selected_model = deepcopy(model)
        selected_model.load_state_dict(best_weights)
        primary = observation(selected_model, best_state["step"], best_state["epoch"],
                              best_state["minibatch_sha256"], include_spectral)
        primary["selection"] = "minimum full validation RMSE before the first patience stop"
        primary["selection_censored"] = censored
        result["primary_best_validation"] = primary
        result["best_weights_sha256"] = model_hash(best_weights)
        result["paper_stopping"] = {"reached": not censored, "step": completed,
                                     "epoch": epoch, "best_step": best_state["step"],
                                     "best_epoch": best_state["epoch"], "patience": patience,
                                     "censored": censored, "reason": reason}
        atomic_checkpoint(output_dir / "best_checkpoint.pt",
                          {"checkpoint_kind": "weights_only", "identity": identity,
                           "model": best_weights, "selected_step": best_state["step"],
                           "selected_epoch": best_state["epoch"], "metrics": primary,
                           "weights_sha256": result["best_weights_sha256"]})

    def persist():
        elapsed = elapsed_prior + time.perf_counter() - started
        result.update(completed_updates=completed, completed_epochs=epoch,
                      elapsed_seconds=elapsed, minibatch_sha256=minibatch_digest,
                      minibatch_rng_sha256=tensor_hash(generator.get_state()),
                      final_weights_sha256=model_hash(model))
        checkpoint = {"checkpoint_kind": "full_resume", "identity": identity,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "rng": rng_state(), "batch_generator": generator.get_state(),
                      "epoch_permutation": permutation, "next_offset": offset,
                      "epoch_loss_sum": batch_loss_sum, "epoch_examples": epoch_examples,
                      "completed_updates": completed, "completed_epochs": epoch,
                      "best_weights": best_weights, "best_state": best_state,
                      "bad_epochs": bad_epochs, "minibatch_sha256": minibatch_digest,
                      "result": result, "elapsed_seconds": elapsed, "plan": plan, "config": config}
        atomic_checkpoint(resume_path, checkpoint)
        if completed in checkpoints:
            atomic_checkpoint(output_dir / f"checkpoint_step_{completed}.pt", checkpoint)
        atomic_json(result_path, result)

    if not result["history"]:
        measure_current()
        persist()
    try:
        while completed < limit:
            if permutation is None:
                permutation = torch.randperm(n_train, generator=generator)
                offset, batch_loss_sum, epoch_examples = 0, 0.0, 0
            model.train()
            indices = permutation[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x_train[indices]), y_train[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss before update {completed + 1}")
            loss.backward()
            clip = plan.get("grad_clip", plan.get("gradient_clipping_norm", 1.0))
            if clip is not None:
                try:
                    nn.utils.clip_grad_norm_(model.parameters(), float(clip), error_if_nonfinite=True)
                except RuntimeError as exc:
                    if "non-finite" in str(exc).lower():
                        raise FloatingPointError(f"Non-finite gradient before update {completed + 1}") from exc
                    raise
            optimizer.step()
            completed += 1
            minibatch_digest = hashlib.sha256(bytes.fromhex(minibatch_digest)
                                               + indices.numpy().tobytes()).hexdigest()
            offset += len(indices)
            batch_loss_sum += float(loss.detach()) * len(indices)
            epoch_examples += len(indices)
            epoch_ended = offset == n_train
            if epoch_ended:
                epoch += 1
                validation = evaluate(model, arrays, "validation", eval_batch)
                improved = False
                if not result["paper_stopping"]["reached"]:
                    improved = (best_state["validation_rmse"] is None
                                or validation["rmse"] < best_state["validation_rmse"])
                    if improved:
                        best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        best_state = {"validation_rmse": validation["rmse"], "step": completed,
                                      "epoch": epoch, "minibatch_sha256": minibatch_digest}
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                    if bad_epochs >= patience:
                        freeze_primary("first_validation_patience_stop")
                result["epoch_history"].append({"epoch": epoch, "step": completed,
                    "training_loss": batch_loss_sum / epoch_examples, "validation": validation,
                    "improved_before_paper_stop": improved, "bad_epochs_before_stop": bad_epochs,
                    "primary_selection_frozen": result["paper_stopping"]["reached"]})
                print(f"{phase} {config['optimizer']} seed={seed} epoch={epoch} step={completed} "
                      f"val_RMSE={validation['rmse']:.6f} best={best_state['validation_rmse']:.6f} "
                      f"paper_stop={result['paper_stopping']['reached']}", flush=True)
                permutation, offset = None, 0
                batch_loss_sum, epoch_examples = 0.0, 0
            fixed_measurement = completed % every == 0 or completed in checkpoints
            if fixed_measurement:
                measure_current()
            if epoch_ended or fixed_measurement or completed == limit:
                persist()
            if result["paper_stopping"]["reached"] and (
                    phase in ("calibration", "reference") or completed >= total):
                break

        if phase == "calibration":
            if not result["paper_stopping"]["reached"]:
                freeze_primary("calibration_update_cap", censored=True)
            result["calibration_capped"] = result["paper_stopping"]["censored"]
            result["status"] = "complete"
        elif result["paper_stopping"]["reached"] and (completed >= total or phase == "reference"):
            result["status"] = "complete"
        elif completed >= safety:
            freeze_primary("safety_update_cap", censored=True)
            result["status"] = "censored"
        else:
            result["status"] = "paused"
        if not any(row["step"] == completed for row in result["history"]):
            measure_current()
        persist()
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["failure_after_updates"] = completed
        atomic_json(result_path, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("calibration", "confirmation", "reference"), default="confirmation")
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    outcome = train(json.loads(Path(args.plan).read_text()), json.loads(Path(args.config).read_text()),
                    args.data_dir, args.output, args.seed, args.phase, args.stop_step, not args.no_resume)
    print(json.dumps({key: outcome[key] for key in ("optimizer", "seed", "status", "completed_updates")}))
