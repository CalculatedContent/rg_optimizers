from __future__ import annotations

"""Executable accelerator/optimizer/checkpoint/WeightWatcher smoke test.

This module deliberately uses a tiny model and synthetic tokens.  It is a
backend gate, not an experiment run, and writes only beneath explicitly
provided temporary paths.
"""

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence


_OPTIMIZERS = ("adamw", "muon_clip")
_CACHE_ENVIRONMENTS = {
    "HOME": "home",
    "HF_HOME": "huggingface",
    "HF_DATASETS_CACHE": "huggingface/datasets",
    "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
    "HF_HUB_CACHE": "huggingface/hub",
    "HF_ASSETS_CACHE": "huggingface/assets",
    "HF_MODULES_CACHE": "huggingface/modules",
    "TRANSFORMERS_CACHE": "huggingface/transformers",
    "TIKTOKEN_CACHE_DIR": "tiktoken",
    "MPLCONFIGDIR": "matplotlib",
    "XDG_CACHE_HOME": "xdg/cache",
    "XDG_CONFIG_HOME": "xdg/config",
    "XDG_DATA_HOME": "xdg/data",
    "XDG_STATE_HOME": "xdg/state",
    "TORCH_HOME": "torch",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "CUDA_CACHE_PATH": "cuda",
    "TRITON_CACHE_DIR": "triton",
    "CUPY_CACHE_DIR": "cupy",
    "XLA_PERSISTENT_CACHE_PATH": "xla",
    "NUMBA_CACHE_DIR": "numba",
    "JOBLIB_TEMP_FOLDER": "joblib",
    "PYTHONPYCACHEPREFIX": "pycache",
}


def _is_below_temporary_root(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    candidates = {
        Path("/tmp").resolve(strict=False),
        Path("/private/tmp").resolve(strict=False),
        Path(tempfile.gettempdir()).resolve(strict=False),
    }
    for candidate in candidates:
        try:
            resolved.relative_to(candidate)
            return resolved != candidate
        except ValueError:
            continue
    return False


def _require_temporary_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if not _is_below_temporary_root(path):
        raise ValueError(
            f"{label} must be a descendant of /tmp (or macOS /private/tmp): "
            f"{path}"
        )
    return path


def _configure_private_caches(session_dir: Path) -> dict[str, str]:
    cache_root = session_dir / "cache"
    configured: dict[str, str] = {}
    for name, relative in _CACHE_ENVIRONMENTS.items():
        path = cache_root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
        configured[name] = str(path)
    temporary = session_dir / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = str(temporary)
        configured[name] = str(temporary)
    os.environ["MPLBACKEND"] = "Agg"
    configured["MPLBACKEND"] = "Agg"
    return configured


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{label} is not finite: {result}")
    return result


def _state_is_finite(state: dict[str, Any]) -> bool:
    import torch

    for value in state.values():
        if torch.is_tensor(value) and (
            value.is_floating_point() or value.is_complex()
        ):
            if not bool(torch.isfinite(value).all()):
                return False
    return True


def _gradient_norm(model: Any) -> float:
    import torch

    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not squares:
        raise RuntimeError("the smoke backward pass produced no gradients")
    total = torch.stack(squares).sum().sqrt()
    return _finite_float(total.detach().cpu(), label="gradient norm")


def _snapshot_equal(expected: Sequence[Any], observed: Sequence[Any]) -> bool:
    import torch

    return len(expected) == len(observed) and all(
        torch.equal(left, right)
        for left, right in zip(expected, observed, strict=True)
    )


def _optimizer_smoke(
    *,
    optimizer_name: str,
    cfg: dict[str, Any],
    device: Any,
    session_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], Any]:
    import torch

    from rg_nanogpt_one_head.checkpoints import (
        load_training_checkpoint_for_resume,
        save_training_checkpoint,
    )
    from rg_nanogpt_one_head.model import GPT, GPTConfig
    from rg_nanogpt_one_head.optimizers import (
        make_optimizer_handles,
        optimizer_step,
        zero_grad,
    )
    from rg_nanogpt_one_head.run_utils import model_state_sha256
    from rg_nanogpt_one_head.runtime import (
        mark_step,
        seed_everything,
        synchronize,
    )

    tiny_config = GPTConfig(
        vocab_size=128,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=32,
        dropout=0.0,
        bias=False,
        tie_weights=True,
    )
    seed_everything(seed, device)
    model = GPT(tiny_config).to(device)
    profile = deepcopy(cfg["optimizer_profiles"][optimizer_name])
    handles = make_optimizer_handles(model, profile)

    token_generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    inputs = torch.randint(
        0,
        tiny_config.vocab_size,
        (2, tiny_config.block_size),
        generator=token_generator,
        dtype=torch.long,
    ).to(device)
    targets = torch.randint(
        0,
        tiny_config.vocab_size,
        (2, tiny_config.block_size),
        generator=token_generator,
        dtype=torch.long,
    ).to(device)

    model.train()
    zero_grad(handles)
    _, loss = model(inputs, targets)
    if loss is None:
        raise RuntimeError("the smoke forward pass did not return a loss")
    loss.backward()
    gradient_norm = _gradient_norm(model)
    optimizer_step(handles)
    mark_step(device)
    synchronize(device)
    loss_value = _finite_float(loss.detach().cpu(), label=f"{optimizer_name} loss")

    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    if not _state_is_finite(state):
        raise FloatingPointError(
            f"{optimizer_name} produced non-finite model parameters"
        )
    state_hash = model_state_sha256(state)
    previous_eval_snapshot = [
        parameter.detach().cpu().clone() for parameter in model.parameters()
    ]
    diagnostics = {
        "previous_eval_snapshot": previous_eval_snapshot,
        "last_grad_pre": gradient_norm,
        "last_grad_post": gradient_norm,
        "last_clipped": False,
    }
    train_generator = torch.Generator(device="cpu").manual_seed(seed + 29)
    # Advance the stream so this verifies restoration of non-initial state.
    torch.randint(0, 2**16, (19,), generator=train_generator)
    expected_generator_state = train_generator.get_state().clone()
    checkpoint_path = session_dir / "checkpoints" / optimizer_name / "step_1.pt"
    fingerprint = f"doctor-smoke-v1:{optimizer_name}:{seed}"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        handles=handles,
        step=1,
        best_validation_loss=loss_value,
        best_validation_step=1,
        elapsed_seconds=0.0,
        fingerprint=fingerprint,
        cfg=cfg,
        optimizer_name=optimizer_name,
        seed=seed,
        train_generator=train_generator,
        resume_diagnostics=diagnostics,
    )
    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(raw_checkpoint.get("schema_version", -1)) != 5:
        raise RuntimeError(
            f"{optimizer_name} smoke checkpoint is not schema version 5"
        )
    if len(raw_checkpoint.get("optimizers", ())) != len(handles):
        raise RuntimeError(
            f"{optimizer_name} smoke checkpoint optimizer inventory changed"
        )

    seed_everything(seed + 1, device)
    resumed_model = GPT(tiny_config).to(device)
    resumed_handles = make_optimizer_handles(resumed_model, profile)
    resumed_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    loaded = load_training_checkpoint_for_resume(
        checkpoint_path,
        model=resumed_model,
        handles=resumed_handles,
        expected_fingerprint=fingerprint,
        train_generator=resumed_generator,
    )
    step, best_loss, best_step, elapsed, loaded_diagnostics = loaded
    if (step, best_step, elapsed) != (1, 1, 0.0):
        raise RuntimeError(
            f"{optimizer_name} checkpoint metadata changed during round-trip"
        )
    if best_loss != loss_value:
        raise RuntimeError(
            f"{optimizer_name} validation loss changed during round-trip"
        )
    if loaded_diagnostics is None:
        raise RuntimeError(
            f"{optimizer_name} resume diagnostics were not restored"
        )
    if not _snapshot_equal(
        previous_eval_snapshot,
        loaded_diagnostics["previous_eval_snapshot"],
    ):
        raise RuntimeError(
            f"{optimizer_name} resume snapshot changed during round-trip"
        )
    if (
        loaded_diagnostics["last_grad_pre"] != gradient_norm
        or loaded_diagnostics["last_grad_post"] != gradient_norm
        or loaded_diagnostics["last_clipped"] is not False
    ):
        raise RuntimeError(
            f"{optimizer_name} scalar resume diagnostics changed"
        )
    if not torch.equal(resumed_generator.get_state(), expected_generator_state):
        raise RuntimeError(
            f"{optimizer_name} train generator state was not restored"
        )
    resumed_state = {
        name: value.detach().cpu().contiguous()
        for name, value in resumed_model.state_dict().items()
    }
    resumed_hash = model_state_sha256(resumed_state)
    if resumed_hash != state_hash:
        raise RuntimeError(
            f"{optimizer_name} model state changed during checkpoint round-trip"
        )

    # Optimizer state device/dtype mistakes (especially on XLA and MPS) can
    # remain latent until the first update after load_state_dict. Exercise that
    # boundary before the backend is approved for a multi-day run.
    resumed_model.train()
    zero_grad(resumed_handles)
    _, resumed_loss = resumed_model(inputs, targets)
    if resumed_loss is None:
        raise RuntimeError(
            f"{optimizer_name} resumed forward pass returned no loss"
        )
    resumed_loss.backward()
    resumed_gradient_norm = _gradient_norm(resumed_model)
    optimizer_step(resumed_handles)
    mark_step(device)
    synchronize(device)
    resumed_loss_value = _finite_float(
        resumed_loss.detach().cpu(),
        label=f"{optimizer_name} resumed loss",
    )
    resumed_updated_state = {
        name: value.detach().cpu().contiguous()
        for name, value in resumed_model.state_dict().items()
    }
    if not _state_is_finite(resumed_updated_state):
        raise FloatingPointError(
            f"{optimizer_name} produced non-finite state after resume"
        )

    return (
        {
            "optimizer": optimizer_name,
            "loss": loss_value,
            "gradient_norm": gradient_norm,
            "optimizer_handles": len(handles),
            "checkpoint_schema_version": 5,
            "checkpoint_path": str(checkpoint_path),
            "model_state_sha256": state_hash,
            "checkpoint_roundtrip": True,
            "resume_diagnostics_roundtrip": True,
            "train_generator_roundtrip": True,
            "resumed_optimizer_step": True,
            "resumed_loss": resumed_loss_value,
            "resumed_gradient_norm": resumed_gradient_norm,
        },
        resumed_model,
    )


def _weightwatcher_smoke(
    *,
    model: Any,
    cfg: dict[str, Any],
    session_dir: Path,
    seed: int,
) -> dict[str, Any]:
    from rg_nanogpt_one_head.model import transformer_matrix_items
    from rg_nanogpt_one_head.spectral import run_weightwatcher

    ww_config = deepcopy(cfg["weightwatcher"])
    if ww_config.get("fix_fingers") != "clip_xmax":
        raise RuntimeError(
            "the backend smoke requires weightwatcher.fix_fingers=clip_xmax"
        )
    expected_names = {
        name for name, _, _, _ in transformer_matrix_items(model)
    }
    if len(expected_names) != 6:
        raise RuntimeError("the tiny model does not expose exactly six matrices")

    run_dir = session_dir / "weightwatcher"
    summary = run_weightwatcher(
        model,
        run_dir,
        step=1,
        tokens_seen=16,
        train_tokens=int(cfg["dataset"]["train_tokens"]),
        config=ww_config,
        seed=seed,
        fingerprint=f"doctor-smoke-v1:weightwatcher:{seed}",
    )
    raw_path = run_dir / "spectral" / "raw" / "weightwatcher_step_0000001.csv"
    with raw_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed_names = {str(row["matrix_name"]) for row in rows}
    if len(rows) != 6 or observed_names != expected_names:
        raise RuntimeError(
            "WeightWatcher did not return exactly the six transformer matrices"
        )
    for index, row in enumerate(rows):
        if int(float(row["weightwatcher_analysis_calls"])) != 1:
            raise RuntimeError(
                "WeightWatcher smoke result does not prove a one-pass analysis"
            )
        _finite_float(row["raw_alpha"], label=f"raw_alpha row {index}")
        _finite_float(row["alpha_raw"], label=f"alpha_raw row {index}")
        _finite_float(row["alpha"], label=f"alpha row {index}")
        _finite_float(
            row["alpha_clip_xmax"],
            label=f"alpha_clip_xmax row {index}",
        )
        if row["finger_policy"] != "fix_fingers=clip_xmax":
            raise RuntimeError("WeightWatcher finger policy metadata changed")

    return {
        "analysis_calls": 1,
        "matrix_count": 6,
        "matrix_names": sorted(observed_names),
        "finger_policy": "fix_fingers=clip_xmax",
        "raw_alpha_count": int(summary["alpha_raw_n"]),
        "clipped_alpha_count": int(summary["alpha_clip_xmax_n"]),
        "raw_csv": str(raw_path),
    }


def run_smoke(
    *,
    config_path: str | Path,
    work_dir: str | Path,
    summary_path: str | Path,
    device_request: str,
    seed: int,
) -> dict[str, Any]:
    work_root = _require_temporary_path(work_dir, label="work directory")
    output_path = _require_temporary_path(summary_path, label="summary path")
    work_root.mkdir(parents=True, exist_ok=True)
    session_dir = work_root / (
        f"doctor-smoke-{time.time_ns()}-pid-{os.getpid()}"
    )
    session_dir.mkdir(parents=False, exist_ok=False)
    cache_environment = _configure_private_caches(session_dir)

    # MuonClip extends both profile validation and optimizer construction, so
    # it must be installed before the frozen YAML is loaded and validated.
    from rg_nanogpt_one_head.muonclip import install_muonclip_extension

    install_muonclip_extension()

    from rg_nanogpt_one_head.config import load_config
    from rg_nanogpt_one_head.runtime import (
        accelerator_name,
        choose_device,
        configure_runtime,
        runtime_metadata,
    )

    config = Path(config_path).expanduser().resolve(strict=True)
    cfg = load_config(config)
    device = choose_device(device_request)
    configure_runtime(device, cfg)

    optimizer_results: list[dict[str, Any]] = []
    spectral_model = None
    for optimizer_name in _OPTIMIZERS:
        result, spectral_model = _optimizer_smoke(
            optimizer_name=optimizer_name,
            cfg=cfg,
            device=device,
            session_dir=session_dir,
            seed=int(seed),
        )
        optimizer_results.append(result)
    if spectral_model is None:  # pragma: no cover - fixed non-empty inventory
        raise AssertionError("no optimizer smoke model was produced")
    weightwatcher = _weightwatcher_smoke(
        model=spectral_model,
        cfg=cfg,
        session_dir=session_dir,
        seed=int(seed),
    )

    payload = {
        "schema_version": 1,
        "completed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config),
        "device_request": str(device_request),
        "resolved_device": str(device),
        "accelerator": accelerator_name(device),
        "seed": int(seed),
        "session_dir": str(session_dir),
        "cache_environment": cache_environment,
        "runtime": runtime_metadata(device),
        "tiny_model": {
            "vocab_size": 128,
            "block_size": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 32,
            "batch_size": 2,
        },
        "optimizers": optimizer_results,
        "weightwatcher": weightwatcher,
    }
    _atomic_json(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the selected backend with all campaign optimizers, "
            "schema-v5 checkpoint resume, and one-pass WeightWatcher."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps", "tpu", "xla"),
    )
    parser.add_argument("--seed", type=int, default=24_681_357)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload = run_smoke(
        config_path=args.config,
        work_dir=args.work_dir,
        summary_path=args.summary,
        device_request=args.device,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
