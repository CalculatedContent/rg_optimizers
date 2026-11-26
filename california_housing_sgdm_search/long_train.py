"""Fixed-budget SGDM training and an explicit continuation of frozen checkpoints.

Imports the unchanged published-MLP implementation from commit
67a0bf203ab0f6b34565e37b4f510f123f669876. No patience rule stops this protocol.
Continuation begins from the parent's full state and selects validation models
only inside its declared continuation window; it never reuses a frozen old best.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

from california_housing_benchmark import train as base

import numpy as np
import torch
from torch import nn

BASE_SOURCE_COMMIT = "67a0bf203ab0f6b34565e37b4f510f123f669876"
BASE_HASHES = {
    "data.py": "f9bd19bf6cb07551c23099e46959fe0b12a5f488910fb73d12bcfec2a0d0ac46",
    "train.py": "fa83694f5389a3b44926152e6839cb81bafb47803decc088646d0f1ac78d272d",
    "optimizer.py": "dc93c4264479886eb3d719cb22cbbb229956ee037e5151ae2c36482f2b89ad2d",
    "spectral.py": "737e99be6fd600f388582d449648b13c945243a9f3032f685e9e12736a41242c",
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structured_state_sha256(value):
    """Hash nested optimizer/RNG state without pickle storage identifiers."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor")
            visit(str(item.dtype))
            visit(tuple(item.shape))
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"ndarray")
            visit(item.dtype.str)
            visit(tuple(item.shape))
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            visit(len(item))
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"list" if isinstance(item, list) else b"tuple")
            visit(len(item))
            for child in item:
                visit(child)
        elif isinstance(item, np.generic):
            visit(item.item())
        elif item is None or isinstance(item, (str, int, float, bool)):
            payload = json.dumps(item, allow_nan=False, separators=(",", ":")).encode()
            digest.update(type(item).__name__.encode() + b":" + str(len(payload)).encode() + b":" + payload)
        else:
            raise TypeError(f"Unsupported checkpoint value: {type(item).__name__}")
    visit(value)
    return digest.hexdigest()


def checkpoint_fingerprints(checkpoint):
    """Pure inspection of a saved state; does not construct or train a model."""
    permutation = checkpoint["epoch_permutation"]
    return {
        "model_sha256": base.model_hash(checkpoint["model"]),
        "optimizer_sha256": structured_state_sha256(checkpoint["optimizer"]),
        "torch_rng_sha256": base.tensor_hash(checkpoint["rng"]["torch"]),
        "numpy_rng_sha256": structured_state_sha256(checkpoint["rng"]["numpy"]),
        "python_rng_sha256": structured_state_sha256(checkpoint["rng"]["python"]),
        "batch_generator_sha256": base.tensor_hash(checkpoint["batch_generator"]),
        "epoch_permutation_sha256": None if permutation is None else base.tensor_hash(permutation),
        "next_offset": checkpoint["next_offset"],
        "completed_updates": checkpoint["completed_updates"],
        "completed_epochs": checkpoint["completed_epochs"],
        "minibatch_sha256": checkpoint["minibatch_sha256"],
        "epoch_loss_sum": checkpoint["epoch_loss_sum"],
        "epoch_examples": checkpoint["epoch_examples"],
    }


def model_specification(plan, config):
    spec = dict(plan.get("model", {}))
    spec = spec.get("backbone", spec)
    return {"d_in": int(spec.get("d_in", 8)),
            "n_blocks": int(spec.get("n_blocks", 2)),
            "d_block": int(spec.get("d_block", 784)),
            "dropout": float(config.get("dropout", spec.get("dropout", 0.3690073279296028)))}


def numerical_runtime(execution):
    return {key: execution[key] for key in (
        "python", "packages", "precision", "torch_num_threads",
        "torch_num_interop_threads", "deterministic_algorithms", "thread_environment")}


def execution_metadata():
    execution = base.execution_metadata()
    if execution["source_sha256"] != BASE_HASHES:
        raise RuntimeError("Frozen published-MLP helper source changed")
    execution["base_source_commit"] = BASE_SOURCE_COMMIT
    execution["base_source_bundle_sha256"] = execution["source_bundle_sha256"]
    execution["source_sha256"] = {
        **execution["source_sha256"], "long_train.py": file_sha256(__file__)}
    execution["source_bundle_sha256"] = base.canonical_sha256(execution["source_sha256"])
    manifest = Path(__file__).with_name("SOURCE_SHA256.json")
    execution["source_manifest_sha256"] = file_sha256(manifest) if manifest.exists() else None
    return execution


def measure_spectra(model):
    from california_housing_benchmark.spectral import measure
    return measure(model)


def _load_parent(plan, config, seed, checkpoint_path, model_spec, data_hash, runtime_hash):
    contract = plan.get("continuation", {})
    record = contract.get("checkpoints", {}).get(str(seed))
    required = ("repository", "run_id", "launch_commit", "archive_commit", "checkpoint_step")
    if record is None or any(not contract.get(key) for key in required):
        raise ValueError("Continuation requires a complete pinned parent contract")
    if contract["launch_commit"] != BASE_SOURCE_COMMIT:
        raise ValueError("Continuation launch source is not the frozen parent source")
    payload = Path(checkpoint_path).read_bytes()
    checkpoint_hash = hashlib.sha256(payload).hexdigest()
    git_blob_hash = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if checkpoint_hash != record["sha256"] or git_blob_hash != record["git_blob_sha"]:
        raise ValueError("Parent checkpoint file hash differs from its pinned archive")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = state["identity"]
    if state.get("checkpoint_kind") != "full_resume":
        raise ValueError("Continuation requires a full optimizer/RNG checkpoint")
    if state["completed_updates"] != int(contract["checkpoint_step"]):
        raise ValueError("Parent checkpoint has the wrong update budget")
    optimizer_config = {key: value for key, value in config.items() if key != "dropout"}
    if identity["seed"] != int(seed) or identity["optimizer"] != "sgdm":
        raise ValueError("Parent seed or optimizer differs")
    if identity["config"] != optimizer_config or state["config"] != optimizer_config:
        raise ValueError("Continuation cannot change the parent optimizer settings")
    if model_specification(state["plan"], state["config"]) != model_spec:
        raise ValueError("Continuation cannot change the parent model")
    if int(state["plan"].get("batch_size", 256)) != int(plan.get("batch_size", 256)):
        raise ValueError("Continuation cannot change parent minibatches")
    old_clip = state["plan"].get("grad_clip", state["plan"].get("gradient_clipping_norm", 1.0))
    new_clip = plan.get("grad_clip", plan.get("gradient_clipping_norm", 1.0))
    if old_clip != new_clip:
        raise ValueError("Continuation cannot change parent gradient clipping")
    if identity["data_manifest_sha256"] != data_hash:
        raise ValueError("Parent preprocessing/data identity differs")
    if identity["numerical_runtime_sha256"] != runtime_hash:
        raise ValueError("Parent numerical runtime differs")
    if identity["source_bundle_sha256"] != base.canonical_sha256(BASE_HASHES):
        raise ValueError("Parent checkpoint was not produced by the frozen helpers")
    github = state["result"]["execution"]["github"]
    for key, expected in (("GITHUB_RUN_ID", contract["run_id"]),
                          ("GITHUB_SHA", contract["launch_commit"]),
                          ("GITHUB_REPOSITORY", contract["repository"])):
        if str(github.get(key)) != str(expected):
            raise ValueError(f"Parent execution provenance differs: {key}")
    provenance = {key: contract[key] for key in required}
    provenance.update(checkpoint_sha256=checkpoint_hash, checkpoint_git_blob_sha=git_blob_hash,
                      repository_path=record["repository_path"],
                      identity=deepcopy(identity),
                      weights_sha256=base.model_hash(state["model"]),
                      original_primary_best_validation=deepcopy(state["result"].get("primary_best_validation")),
                      original_paper_stopping=deepcopy(state["result"].get("paper_stopping")),
                      original_elapsed_seconds=state.get("elapsed_seconds"))
    return state, provenance


def train_long(plan, config, data_dir, output_dir, seed, phase,
               parent_checkpoint=None, stop_step=None, resume=True):
    """Run/resume a fixed budget; an optional parent is immutable provenance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {output_dir}; duplicate refused") from exc
        try:
            return _train_long(plan, config, data_dir, output_dir, int(seed), phase,
                               parent_checkpoint, stop_step, resume)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _train_long(plan, config, data_dir, output_dir, seed, phase,
                parent_checkpoint, stop_step, resume):
    if phase not in ("development", "confirmation", "continuation"):
        raise ValueError("Unknown fixed-budget phase")
    if config.get("optimizer") != "sgdm":
        raise ValueError("This extension is prespecified for SGDM")
    if (phase == "continuation") != (parent_checkpoint is not None):
        raise ValueError("Only continuation requires a parent checkpoint")
    total = int(plan.get("development_updates", 50000) if phase == "development"
                else plan.get("total_updates", 100000))
    limit = total if stop_step is None else int(stop_step)
    every, batch_size = int(plan.get("eval_every", 5000)), int(plan.get("batch_size", 256))
    eval_batch = int(plan.get("eval_batch_size", 32768))
    if min(total, every, batch_size, eval_batch) <= 0 or not 0 <= limit <= total:
        raise ValueError("Invalid fixed budget, cadence, or minibatch size")
    checkpoints = {int(step) for step in plan.get("checkpoint_updates", [2000, 20000, 50000, 100000])
                   if 0 < int(step) <= total}
    checkpoints.add(total)
    include_test = phase != "development"
    spectra_enabled = bool(plan.get("measure_spectra", True))
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    arrays, _ = base.load_data(data_dir)
    data_hash = file_sha256(Path(data_dir) / "manifest.json")
    execution = execution_metadata()
    runtime_hash = base.canonical_sha256(numerical_runtime(execution))
    model_spec = model_specification(plan, config)
    parent_state = parent = None
    if parent_checkpoint is not None:
        parent_state, parent = _load_parent(plan, config, seed, parent_checkpoint,
                                           model_spec, data_hash, runtime_hash)
        parent_path = Path(parent_checkpoint).resolve()
        if parent_path in {output_dir.resolve() / "resume.pt",
                           output_dir.resolve() / f"checkpoint_step_{parent_state['completed_updates']}.pt"}:
            raise ValueError("The new run cannot overwrite its parent checkpoint")
    start_step = parent_state["completed_updates"] if parent_state is not None else 0
    if total <= start_step or limit < start_step:
        raise ValueError("Continuation budget must extend its retained parent")
    identity = {
        "optimizer": "sgdm", "seed": seed, "phase": phase, "config": config,
        "model": model_spec, "plan_sha256": base.canonical_sha256(plan),
        "data_manifest_sha256": data_hash, "numerical_runtime_sha256": runtime_hash,
        "source_bundle_sha256": execution["source_bundle_sha256"],
        "source_manifest_sha256": execution["source_manifest_sha256"],
        "base_source_commit": BASE_SOURCE_COMMIT,
        "parent_checkpoint_sha256": None if parent is None else parent["checkpoint_sha256"],
    }
    result_path, resume_path = output_dir / "result.json", output_dir / "resume.pt"
    if result_path.exists() or resume_path.exists():
        if not resume:
            raise FileExistsError("Existing run cannot be overwritten")
        if not resume_path.exists():
            raise ValueError("Existing run lacks full resume state")
        previous = json.loads(result_path.read_text()) if result_path.exists() else None
        if previous is not None and any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing fixed-budget identity differs")
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Fixed-budget resume identity differs")
        if previous is not None and previous.get("status") == "complete":
            if (saved["completed_updates"] != total or previous["completed_updates"] != total
                    or base.model_hash(saved["model"]) != previous["final_weights_sha256"]
                    or not (output_dir / "best_checkpoint.pt").exists()
                    or not (output_dir / f"checkpoint_step_{total}.pt").exists()):
                raise ValueError("Completed result and retained checkpoints disagree")
            return previous
    else:
        saved = None
    base.delu.random.seed(seed)
    model = base.MLP(model_spec).float()
    optimizer = base.make_optimizer(model, config)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_hash = base.model_hash(model)
    n_train, epoch = len(arrays["y_train"]), 0
    epoch_size = math.ceil(n_train / batch_size)
    completed = offset = epoch_examples = 0
    batch_loss_sum = elapsed_prior = 0.0
    permutation = None
    minibatch_digest = "0" * 64
    best_weights = None
    best_state = {"validation_rmse": None, "step": None, "epoch": None,
                  "minibatch_sha256": None}
    result = {
        **identity, "schema_version": 1, "status": "running", "completed_updates": start_step,
        "completed_epochs": 0, "new_updates": 0, "start_step": start_step, "total_updates": total,
        "parameter_count": sum(p.numel() for p in model.parameters()), "epoch_size": epoch_size,
        "initialization_sha256": initial_hash, "sampler_seed": seed,
        "sampling": "one torch.randperm per epoch, including final partial minibatch",
        "minibatch_sha256": minibatch_digest, "execution": execution, "resume_executions": [],
        "history": [], "epoch_history": [], "best_validation": None, "primary_endpoint": None,
        "primary_estimand": "fixed_updates" if phase == "continuation" else "best_validation",
        "parent": parent, "continuity": None, "patience_stopping": False,
        "selection_window": {"start": start_step, "end": total, "observed_through": start_step,
                             "complete": False},
        "selection_rule": "Strict minimum full validation RMSE at the window start, each complete epoch, and final declared budget; ties retain earlier checkpoint",
        "selection_scope": "Only the continuation window; no claim about the unretained best over updates 0-20000" if parent else "Complete declared training window",
        "metrics_unit": {"rmse": "100000 USD", "mae": "100000 USD", "mse": "(100000 USD)^2"},
        "protocol": "constant LR; no patience stopping; fixed declared budget; test never selects settings or checkpoints",
    }

    def runtime_checkpoint():
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": base.rng_state(), "batch_generator": generator.get_state(),
                "epoch_permutation": permutation, "next_offset": offset,
                "epoch_loss_sum": batch_loss_sum, "epoch_examples": epoch_examples,
                "completed_updates": completed, "completed_epochs": epoch,
                "minibatch_sha256": minibatch_digest}

    inherited = saved if saved is not None else parent_state
    if inherited is not None:
        model.load_state_dict(inherited["model"])
        optimizer.load_state_dict(inherited["optimizer"])
        generator.set_state(inherited["batch_generator"])
        base.restore_rng(inherited["rng"])
        completed, epoch = inherited["completed_updates"], inherited["completed_epochs"]
        permutation, offset = inherited["epoch_permutation"], inherited["next_offset"]
        batch_loss_sum, epoch_examples = inherited["epoch_loss_sum"], inherited["epoch_examples"]
        minibatch_digest = inherited["minibatch_sha256"]
        if completed > limit:
            raise ValueError("Requested pause predates retained progress")
        if saved is not None:
            best_weights, best_state = inherited["best_weights"], inherited["best_state"]
            elapsed_prior = inherited["elapsed_seconds"]
            result = inherited["result"]
            result["resume_executions"].append(execution)
            result["status"] = "running"
        else:
            before = checkpoint_fingerprints(parent_state)
            after = checkpoint_fingerprints(runtime_checkpoint())
            if before != after:
                raise RuntimeError("Parent optimizer/RNG/sampler state was not loaded exactly")
            result["continuity"] = {"parent": before, "start": after, "verified": True}
            result["initialization_sha256"] = parent_state["result"]["initialization_sha256"]

    x_train, y_train = torch.from_numpy(arrays["x_train"]), torch.from_numpy(arrays["y_train"])
    started = time.perf_counter()

    def observation(measured_model, step, observed_epoch, digest, spectral):
        saved_rng, was_training = base.rng_state(), measured_model.training
        try:
            measured_model.eval()
            weights_hash = base.model_hash(measured_model)
            row = {"step": step, "epoch": observed_epoch,
                   "train": base.evaluate(measured_model, arrays, "train", eval_batch),
                   "validation": base.evaluate(measured_model, arrays, "validation", eval_batch),
                   "test": base.evaluate(measured_model, arrays, "test", eval_batch) if include_test else None,
                   "spectral": measure_spectra(measured_model) if spectral else None,
                   "weights_sha256": weights_hash,
                   "spectral_weights_sha256": weights_hash if spectral else None,
                   "minibatch_sha256": digest,
                   "elapsed_seconds": elapsed_prior + time.perf_counter() - started}
            if base.model_hash(measured_model) != weights_hash:
                raise RuntimeError("Metric measurement changed the checkpoint weights")
            return row
        finally:
            base.restore_rng(saved_rng)
            measured_model.train(was_training)

    def select(validation):
        nonlocal best_weights, best_state
        value = float(validation["rmse"])
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite validation RMSE at {completed}")
        improved = best_state["validation_rmse"] is None or value < best_state["validation_rmse"]
        if improved:
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_state = {"validation_rmse": value, "step": completed, "epoch": epoch,
                          "minibatch_sha256": minibatch_digest}
        return improved

    def measure_current():
        spectral = spectra_enabled and (phase != "development" or completed in checkpoints)
        row = observation(model, completed, epoch, minibatch_digest, spectral)
        row["minibatch_rng_sha256"] = base.tensor_hash(generator.get_state())
        row["training_rng_sha256"] = base.tensor_hash(torch.get_rng_state())
        row["weights_sha256"] = base.model_hash(model)
        result["history"] = [value for value in result["history"] if value["step"] != completed] + [row]
        result["history"].sort(key=lambda value: value["step"])
        return row

    def persist():
        elapsed = elapsed_prior + time.perf_counter() - started
        result.update(completed_updates=completed, completed_epochs=epoch,
                      new_updates=completed - start_step, elapsed_seconds=elapsed,
                      minibatch_sha256=minibatch_digest,
                      minibatch_rng_sha256=base.tensor_hash(generator.get_state()),
                      final_weights_sha256=base.model_hash(model))
        result["selection_window"].update(observed_through=completed, complete=completed == total)
        state = {**runtime_checkpoint(), "checkpoint_kind": "full_resume", "identity": identity,
                 "best_weights": best_weights, "best_state": best_state, "elapsed_seconds": elapsed,
                 "result": result, "plan": plan, "config": config}
        base.atomic_checkpoint(resume_path, state)
        if completed in checkpoints:
            base.atomic_checkpoint(output_dir / f"checkpoint_step_{completed}.pt", state)
        base.atomic_json(result_path, result)

    if not result["history"]:
        initial = measure_current()
        select(initial["validation"])
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
            minibatch_digest = hashlib.sha256(bytes.fromhex(minibatch_digest) + indices.numpy().tobytes()).hexdigest()
            offset += len(indices)
            batch_loss_sum += float(loss.detach()) * len(indices)
            epoch_examples += len(indices)
            epoch_ended = offset == n_train
            if epoch_ended:
                epoch += 1
                validation = base.evaluate(model, arrays, "validation", eval_batch)
                improved = select(validation)
                result["epoch_history"].append({
                    "epoch": epoch, "step": completed, "training_loss": batch_loss_sum / epoch_examples,
                    "validation": validation, "improved": improved, "best_step": best_state["step"]})
                print(f"{phase} sgdm seed={seed} epoch={epoch} step={completed} "
                      f"val_RMSE={validation['rmse']:.6f} best={best_state['validation_rmse']:.6f}", flush=True)
                permutation, offset = None, 0
                batch_loss_sum, epoch_examples = 0.0, 0
            fixed = completed % every == 0 or completed in checkpoints
            if fixed:
                row = measure_current()
                if completed == total:
                    select(row["validation"])
                    result["primary_endpoint"] = row
            if epoch_ended or fixed or completed == limit:
                persist()

        result["status"] = "complete" if completed == total else "paused"
        if not any(row["step"] == completed for row in result["history"]):
            measure_current()
        selected_model = deepcopy(model)
        selected_model.load_state_dict(best_weights)
        best = observation(selected_model, best_state["step"], best_state["epoch"],
                           best_state["minibatch_sha256"], spectra_enabled)
        result["selection_window"].update(observed_through=completed, complete=completed == total)
        best["selection_window"] = deepcopy(result["selection_window"])
        best["selection_rule"] = result["selection_rule"]
        best["selection_scope"] = result["selection_scope"]
        best["weights_sha256"] = base.model_hash(best_weights)
        result["best_validation"], result["best_weights_sha256"] = best, best["weights_sha256"]
        base.atomic_checkpoint(output_dir / "best_checkpoint.pt", {
            "checkpoint_kind": "weights_only", "identity": identity, "model": best_weights,
            "selected_step": best_state["step"], "selected_epoch": best_state["epoch"],
            "metrics": best, "weights_sha256": best["weights_sha256"],
            "selection_window": deepcopy(result["selection_window"])})
        persist()
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["failure_after_updates"] = completed
        base.atomic_json(result_path, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("development", "confirmation", "continuation"), required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = train_long(json.loads(Path(args.plan).read_text()), json.loads(Path(args.config).read_text()),
                        args.data_dir, args.output, args.seed, args.phase,
                        args.parent_checkpoint, args.stop_step, not args.no_resume)
    print(json.dumps({key: result[key] for key in ("seed", "phase", "status", "completed_updates")}))


if __name__ == "__main__":
    main()
