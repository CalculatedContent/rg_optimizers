from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

_TPU_ENV_HINTS = (
    "TPU_NAME",
    "TPU_WORKER_ID",
    "TPU_ACCELERATOR_TYPE",
    "CLOUD_TPU_TASK_ID",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_HOST_BOUNDS",
)


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^\s*(\d+)\.(\d+)", str(version))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _tpu_hardware_hint() -> bool:
    pjrt = str(os.environ.get("PJRT_DEVICE", "")).upper()
    if pjrt.startswith("TPU"):
        return True
    # Variables such as TPU_NAME are often exported on a developer laptop to
    # manage a remote TPU. They are evidence of local hardware only on Linux.
    if platform.system() != "Linux":
        return False
    if any(os.environ.get(name) for name in _TPU_ENV_HINTS):
        return True
    candidates = (
        Path("/dev/accel0"),
        Path("/sys/class/accel/accel0"),
    )
    return any(path.exists() for path in candidates)


def _load_xla(*, required: bool) -> tuple[Any, Any, Any] | None:
    if importlib.util.find_spec("torch_xla") is None:
        if required:
            raise RuntimeError(
                "A TPU environment was requested or detected, but torch_xla is "
                "not installed. Install the TPU extra with "
                "`python -m pip install -e '.[tpu]'` in the TPU VM."
            )
        return None
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
    except Exception as exc:
        if required:
            raise RuntimeError(
                "torch_xla is installed but could not initialize the TPU runtime"
            ) from exc
        return None
    return torch_xla, xr, xm


def _xla_device_type(xr: Any) -> str:
    value = xr.device_type()
    return "" if value is None else str(value).upper()


def xla_tpu_available() -> bool:
    hinted = _tpu_hardware_hint()
    modules = _load_xla(required=hinted)
    if modules is None:
        return False
    _, xr, _ = modules
    try:
        return _xla_device_type(xr) == "TPU"
    except Exception as exc:
        if hinted:
            raise RuntimeError(
                "TPU hardware was detected, but PyTorch/XLA could not enumerate it"
            ) from exc
        return False


def is_tpu_environment(
    requested: str | torch.device = "auto",
) -> bool:
    if isinstance(requested, torch.device):
        return requested.type == "xla"
    requested = str(requested).lower()
    if requested in {"tpu", "xla"}:
        return True
    if requested != "auto":
        return False
    if _tpu_hardware_hint():
        return True
    if platform.system() != "Linux":
        return False
    return xla_tpu_available()


def _choose_tpu_device() -> torch.device:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    modules = _load_xla(required=True)
    assert modules is not None
    torch_xla, xr, xm = modules
    if _xla_device_type(xr) != "TPU":
        raise RuntimeError(
            "torch_xla initialized, but its PJRT device is not TPU. "
            f"Observed PJRT device: {_xla_device_type(xr) or 'unknown'}"
        )
    device = (
        torch_xla.device()
        if hasattr(torch_xla, "device")
        else xm.xla_device()
    )
    return torch.device(device)


def choose_device(
    requested: str | torch.device = "auto",
) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    requested = str(requested).lower()
    aliases = {"gpu": "cuda", "apple": "mps", "xla": "tpu"}
    requested = aliases.get(requested, requested)

    if requested == "auto":
        if is_tpu_environment("auto"):
            return _choose_tpu_device()
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "tpu":
        return _choose_tpu_device()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available in this PyTorch runtime"
            )
        return torch.device("cuda")
    if requested == "mps":
        if not _mps_available():
            raise RuntimeError(
                "MPS was requested but is not available in this PyTorch build/runtime"
            )
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(
        "unsupported device request "
        f"{requested!r}; choose auto, tpu, cuda, mps, or cpu"
    )


def is_xla_device(device: torch.device | str) -> bool:
    return torch.device(device).type == "xla"


def accelerator_name(device: torch.device | str) -> str:
    resolved = torch.device(device)
    return "tpu" if resolved.type == "xla" else resolved.type


def _xla_process_count(xr: Any) -> int:
    for name in ("process_count", "world_size"):
        function = getattr(xr, name, None)
        if callable(function):
            return int(function())
    return 1


def _xla_process_index(xr: Any) -> int:
    for name in ("process_index", "global_ordinal"):
        function = getattr(xr, name, None)
        if callable(function):
            return int(function())
    return 0


def configure_runtime(device: torch.device, cfg: dict) -> None:
    torch.set_float32_matmul_precision(
        str(cfg["runtime"].get("matmul_precision", "high"))
    )
    if device.type == "mps" and bool(cfg["runtime"].get("mps_fallback", True)):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if device.type == "xla":
        reduced_precision = [
            name
            for name in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16")
            if str(os.environ.get(name, "")).lower()
            in {"1", "true", "yes", "on"}
        ]
        if reduced_precision:
            raise RuntimeError(
                "The reference protocol is float32, but TPU reduced precision is "
                "enabled through "
                + ", ".join(reduced_precision)
                + ". Unset those variables; this reference protocol does not "
                "silently change numerical precision across accelerators."
            )
        modules = _load_xla(required=True)
        assert modules is not None
        torch_xla, xr, _ = modules
        torch_version = _major_minor(torch.__version__)
        xla_version = _major_minor(
            getattr(torch_xla, "__version__", "")
        )
        if (
            torch_version is not None
            and xla_version is not None
            and torch_version != xla_version
        ):
            raise RuntimeError(
                "PyTorch and PyTorch/XLA major.minor versions must match. "
                f"Observed torch={torch.__version__}, "
                f"torch_xla={getattr(torch_xla, '__version__', 'unknown')}."
            )
        count = _xla_process_count(xr)
        if count != 1:
            raise RuntimeError(
                "This baseline currently supports one PyTorch/XLA process. "
                f"Detected process_count={count}. Run the ordinary single-process "
                "launcher or implement an explicitly distributed protocol."
            )
    if bool(cfg["runtime"].get("deterministic_algorithms", False)):
        torch.use_deterministic_algorithms(True, warn_only=True)


def mark_step(device: torch.device) -> None:
    if device.type != "xla":
        return
    modules = _load_xla(required=True)
    assert modules is not None
    torch_xla, _, xm = modules
    sync = getattr(torch_xla, "sync", None)
    if callable(sync):
        try:
            sync(wait=False)
        except TypeError:
            sync()
    else:
        xm.mark_step()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "xla":
        modules = _load_xla(required=True)
        assert modules is not None
        torch_xla, _, xm = modules
        sync = getattr(torch_xla, "sync", None)
        if callable(sync):
            try:
                sync(wait=True)
            except TypeError:
                sync()
                wait = getattr(xm, "wait_device_ops", None)
                if callable(wait):
                    wait()
        else:
            xm.mark_step()
            wait = getattr(xm, "wait_device_ops", None)
            if callable(wait):
                wait()


def seed_everything(seed: int, device: torch.device | None = None) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if (
        _mps_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
    ):
        torch.mps.manual_seed(int(seed))
    if device is not None and torch.device(device).type == "xla":
        modules = _load_xla(required=True)
        assert modules is not None
        torch_xla, _, xm = modules
        manual_seed = getattr(torch_xla, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(int(seed), device=device)
        else:
            xm.set_rng_state(int(seed), device=device)


def model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def capture_accelerator_rng_state(
    device: torch.device | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if torch.cuda.is_available():
        state["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and _mps_available()
    ):
        state["mps_random_state"] = torch.mps.get_rng_state()
    if device is not None and torch.device(device).type == "xla":
        modules = _load_xla(required=True)
        assert modules is not None
        _, _, xm = modules
        state["xla_random_state"] = int(xm.get_rng_state(device=device))
    return state


def restore_accelerator_rng_state(
    payload: dict[str, Any],
    device: torch.device | None = None,
) -> None:
    if torch.cuda.is_available() and "cuda_random_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_random_state_all"])
    if (
        "mps_random_state" in payload
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
        and _mps_available()
    ):
        torch.mps.set_rng_state(payload["mps_random_state"])
    if (
        device is not None
        and torch.device(device).type == "xla"
        and "xla_random_state" in payload
    ):
        modules = _load_xla(required=True)
        assert modules is not None
        _, _, xm = modules
        xm.set_rng_state(int(payload["xla_random_state"]), device=device)


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "accelerator": accelerator_name(device),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    if device.type == "xla":
        modules = _load_xla(required=True)
        assert modules is not None
        torch_xla, xr, _ = modules
        metadata.update(
            {
                "torch_xla_version": getattr(
                    torch_xla, "__version__", "unknown"
                ),
                "pjrt_device": _xla_device_type(xr),
                "xla_process_count": _xla_process_count(xr),
                "xla_process_index": _xla_process_index(xr),
                "xla_addressable_device_count": int(
                    getattr(xr, "addressable_device_count", lambda: 1)()
                ),
            }
        )
    return metadata


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    return (
        torch.linalg.vector_norm(torch.stack(norms), ord=2)
        if norms
        else torch.tensor(0.0)
    )


def model_weight_norm(model: nn.Module) -> float:
    total: torch.Tensor | None = None
    for parameter in model.parameters():
        term = (parameter.detach().float() ** 2).sum()
        total = term if total is None else total + term
    if total is None:
        return 0.0
    return math.sqrt(float(total.detach().cpu()))


def parameter_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [
        parameter.detach().float().cpu().clone()
        for parameter in model.parameters()
    ]


def update_norm(
    previous: list[torch.Tensor] | None,
    current: list[torch.Tensor],
) -> float:
    if previous is None:
        return 0.0
    if len(previous) != len(current):
        raise RuntimeError("parameter inventory changed during training")
    squared = 0.0
    for old, new in zip(previous, current, strict=True):
        squared += float(((new - old) ** 2).sum())
    return math.sqrt(squared)


def tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(item) for item in value)
    return value


def mps_memory_megabytes(device: torch.device) -> tuple[float, float]:
    if device.type != "mps" or not hasattr(torch, "mps"):
        return float("nan"), float("nan")
    current = (
        float(torch.mps.current_allocated_memory()) / (1024**2)
        if hasattr(torch.mps, "current_allocated_memory")
        else float("nan")
    )
    driver = (
        float(torch.mps.driver_allocated_memory()) / (1024**2)
        if hasattr(torch.mps, "driver_allocated_memory")
        else float("nan")
    )
    return current, driver


def empty_mps_cache(device: torch.device) -> None:
    if (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one-head nanoGPT accelerator and storage detection"
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    from .config import roots

    payload = {
        "runtime": runtime_metadata(device),
        "roots": {
            key: str(value)
            for key, value in roots(device).items()
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
