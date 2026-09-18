from __future__ import annotations

import math
import os
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def choose_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch build/runtime")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_runtime(device: torch.device, cfg: dict) -> None:
    torch.set_float32_matmul_precision(str(cfg["runtime"].get("matmul_precision", "high")))
    if device.type == "mps" and bool(cfg["runtime"].get("mps_fallback", True)):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if bool(cfg["runtime"].get("deterministic_algorithms", False)):
        torch.use_deterministic_algorithms(True, warn_only=True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(int(seed))


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    return torch.linalg.vector_norm(torch.stack(norms), ord=2) if norms else torch.tensor(0.0)


def model_weight_norm(model: nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        squared += float((parameter.detach().float() ** 2).sum().cpu())
    return math.sqrt(squared)


def parameter_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().float().cpu().clone() for parameter in model.parameters()]


def update_norm(previous: list[torch.Tensor] | None, current: list[torch.Tensor]) -> float:
    if previous is None:
        return 0.0
    if len(previous) != len(current):
        raise RuntimeError("parameter inventory changed during training")
    squared = 0.0
    for old, new in zip(previous, current, strict=True):
        squared += float(((new - old) ** 2).sum())
    return math.sqrt(squared)


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
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
