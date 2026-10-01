from __future__ import annotations

"""Single-host XLA SPMD data parallelism; batch sizes are always GLOBAL.

There is one Python process, RNG stream, optimizer and checkpoint writer. XLA
partitions the batch and inserts collectives for global reductions. Never use
xm.optimizer_step or manually divide the loss by the chip count in this mode.
"""

import os
from typing import Any

import numpy as np
import torch

_MESH: Any = None
_XS: Any = None
_CHIPS = 0


def initialize(cfg: dict, requested: str | torch.device) -> None:
    """Must run before choose_device creates any XLA tensors."""
    enabled = cfg.get("runtime", {}).get("tpu_spmd", False)
    if not isinstance(enabled, bool):
        raise ValueError("runtime.tpu_spmd must be a boolean")
    if not enabled:
        if _MESH is not None or os.environ.get("XLA_USE_SPMD") == "1":
            raise RuntimeError("SPMD is already enabled; use a tpu_spmd config in a fresh process")
        return
    if str(requested) not in {"auto", "tpu", "xla", "xla:0"}:
        raise ValueError("runtime.tpu_spmd requires --device tpu (or auto)")
    expected = int(cfg["runtime"].get("tpu_expected_chips", 4))
    batch = int(cfg["training"]["batch_size"])
    if expected < 1 or batch % expected:
        raise ValueError("global training.batch_size must be divisible by tpu_expected_chips")
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    from .runtime import _load_xla

    modules = _load_xla(required=True)
    assert modules is not None
    _, xr, _ = modules
    if xr.device_type() != "TPU":
        raise RuntimeError("TPU SPMD requested but PJRT_DEVICE is not TPU")
    _initialize_mesh(xr, expected)


def _initialize_mesh(xr, expected: int) -> None:
    """Kept separate so the same sharding code can be checked on XLA CPU."""
    global _MESH, _XS, _CHIPS
    if _MESH is not None:
        if _CHIPS != expected:
            raise RuntimeError("Cannot change the SPMD mesh in a running process")
        return
    xr.use_spmd()
    # This implementation deliberately uses one host, as on the v5e-4.
    # Multi-host SPMD requires a separate input/checkpoint ownership protocol.
    count = int(xr.global_runtime_device_count())
    local = int(xr.addressable_runtime_device_count())
    if int(xr.process_count()) != 1 or local != count:
        raise RuntimeError("TPU SPMD trainer currently requires one host/process")
    if count != expected:
        raise RuntimeError(
            f"Expected {expected} TPU chips, found {count}; check TPU_VISIBLE_CHIPS "
            "and remove per-chip sweep environment settings"
        )
    import torch_xla.distributed.spmd as xs

    _XS = xs
    _MESH = xs.Mesh(np.arange(count), (count,), ("data",))
    _CHIPS = count


def metadata() -> dict:
    return {"xla_spmd": _MESH is not None, "xla_spmd_chips": _CHIPS or 1}


def replicate(tensor: torch.Tensor) -> torch.Tensor:
    if _MESH is not None and tensor.device.type == "xla":
        _XS.mark_sharding(tensor, _MESH, (None,) * tensor.ndim)
    return tensor


def replicate_model(model) -> None:
    for tensor in (*model.parameters(), *model.buffers()):
        replicate(tensor)


def replicate_gradients(model) -> None:
    # Replicated gradients MUST precede gradient clipping, momentum and NS.
    # These are constraints on GLOBAL tensors, not a second gradient average.
    for parameter in model.parameters():
        if parameter.grad is not None:
            replicate(parameter.grad)


def batch_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = tensor.to(device)
    if _MESH is not None and value.device.type == "xla":
        if value.ndim < 1 or value.shape[0] % _CHIPS:
            raise ValueError("SPMD batch must divide evenly across the data mesh")
        _XS.mark_sharding(value, _MESH, ("data",) + (None,) * (value.ndim - 1))
    return value
