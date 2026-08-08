"""Post-step wrapper for arbitrary PyTorch optimizers."""
from __future__ import annotations
import torch
from .config import FullMatrixLogConfig
from .geometry import full_matrix_log_geometry, remove_inward_matrix_log_flow


class FullMatrixLogRG(torch.optim.Optimizer):
    """Apply a one-sided full matrix-log RG correction to completed optimizer steps."""
    def __init__(self, base_optimizer, named_parameters, config: FullMatrixLogConfig | None = None):
        self.base_optimizer = base_optimizer
        self.config = config or FullMatrixLogConfig()
        self.config.validate()
        self._named_parameters = [(n, p) for n, p in named_parameters]
        self._supports: dict[str, tuple[int, float | None]] = {}
        self._step_count = 0
        self.last_stats: dict[str, dict] = {}
        self.state = base_optimizer.state
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults

    def set_support(self, parameter_name: str, retained_rank: int, normalization_dimension: float | None = None):
        self._supports[str(parameter_name)] = (int(retained_rank), normalization_dimension)

    def set_supports(self, supports: dict[str, int | tuple[int, float | None]], *, replace: bool = True):
        if replace:
            self._supports = {}
        for name, value in supports.items():
            if isinstance(value, tuple):
                self.set_support(name, int(value[0]), value[1])
            else:
                self.set_support(name, int(value), None)

    def zero_grad(self, *args, **kwargs):
        return self.base_optimizer.zero_grad(*args, **kwargs)

    def state_dict(self):
        payload = self.base_optimizer.state_dict()
        payload['_full_matrix_log_rg'] = {
            'step_count': self._step_count,
            'supports': self._supports,
            'config': self.config.__dict__.copy(),
        }
        return payload

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop('_full_matrix_log_rg', None)
        result = self.base_optimizer.load_state_dict(state_dict)
        if extra:
            self._step_count = int(extra.get('step_count', 0))
            self._supports = dict(extra.get('supports', {}))
        return result

    @torch.no_grad()
    def step(self, closure=None):
        before = {
            id(p): p.detach().clone()
            for _, p in self._named_parameters
            if p.requires_grad and p.ndim == 2
        }
        loss = self.base_optimizer.step(closure)
        self._step_count += 1
        self.last_stats = {}
        if self._step_count % int(self.config.apply_every_steps) != 0:
            return loss

        for name, p in self._named_parameters:
            if id(p) not in before or p.ndim != 2:
                continue
            w0 = before[id(p)]
            delta = p.detach() - w0
            if name in self._supports:
                rank, d = self._supports[name]
            else:
                rank = min(p.shape)
                d = self.config.normalization_dimension
            rank = max(int(self.config.min_retained_rank), min(int(rank), min(p.shape)))
            geometry = full_matrix_log_geometry(
                w0,
                rank,
                normalization_dimension=d if d is not None else self.config.normalization_dimension,
                ridge_relative=self.config.ridge_relative,
                eps=self.config.eps,
            )
            result = remove_inward_matrix_log_flow(
                delta,
                geometry,
                projection_strength=self.config.projection_strength,
                max_correction_ratio=self.config.max_correction_ratio,
                eps=self.config.eps,
            )
            p.copy_(w0 + result.corrected_delta)
            self.last_stats[name] = {
                'phi': float(geometry.potential.detach().float().cpu()),
                'retained_rank': int(geometry.retained_rank),
                'base_drift': result.base_drift,
                'corrected_drift': result.corrected_drift,
                'correction_ratio': result.correction_ratio,
                'applied': result.applied,
                'capped': result.capped,
            }
        return loss
