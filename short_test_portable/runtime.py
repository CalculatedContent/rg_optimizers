"""Explicit CPU / Google TorchTPU backend; CPU initialization and sampling."""
import os
import platform
import importlib.metadata
from pathlib import Path
import torch

DEVICE = os.environ.get('SHORT_GPT_DEVICE', 'cpu')

def initialize():
    if DEVICE not in ('cpu', 'tpu'):
        raise ValueError('Device must be cpu or tpu')
    if DEVICE == 'tpu':
        try:
            import torch_tpu  # noqa: F401; registers native TPU backend
        except ImportError as exc:
            raise RuntimeError('Install Google torch_tpu on a supported TPU VM; CPU needs only PyTorch.') from exc
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None or accelerator.type != 'tpu' or not torch.accelerator.is_available():
            raise RuntimeError('TPU requested but no Google TPU is available; refusing CPU fallback')
        torch.accelerator.set_device_index(0)
    return torch.device(DEVICE)

def synchronize():
    if DEVICE == 'tpu':
        torch.accelerator.synchronize()

def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value

def environment():
    result = dict(device=DEVICE, system=platform.system(), machine=platform.machine(),
                  python=platform.python_version(), torch=torch.__version__,
                  precision='float32 tensors; backend-dependent kernels',
                  stochastic_operations='CPU initialization and minibatches; dropout must remain zero')
    if DEVICE == 'tpu':
        result['torch_tpu'] = importlib.metadata.version('torch_tpu')
    return result

def admission(cgroup=Path('/sys/fs/cgroup')):
    # macOS and Linux hosts without cgroup-v2 have no memory.max/stat files.
    # This is a host admission heuristic, not a device-memory guarantee.
    try:
        limit = (cgroup / 'memory.max').read_text().strip()
        if limit == 'max':
            return True
        stat = dict(line.split() for line in (cgroup / 'memory.stat').read_text().splitlines())
        return int(limit) - int(stat['anon']) - int(stat.get('kernel', 0)) >= 950 * 2**20
    except FileNotFoundError:
        return True
