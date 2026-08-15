from __future__ import annotations

"""MuonClip extension for the one-head nanoGPT baseline.

This module intentionally leaves the historical SGD, AdamW, and Muon entry
point unchanged. The dedicated ``rg-onehead-muonclip`` command installs the
extension in-process and then delegates to the existing training launcher.
"""

import csv
from copy import deepcopy
import importlib
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

_INSTALLED = False
_CURRENT_RUN_DIR: Path | None = None


def _zeropower(
    update: torch.Tensor,
    *,
    steps: int,
    eps: float,
) -> torch.Tensor:
    """Use the repository's existing Newton--Schulz implementation."""

    from .optimizers import zeropower_via_newton_schulz_5

    return zeropower_via_newton_schulz_5(
        update,
        steps=int(steps),
        eps=float(eps),
    )


class MuonClip(torch.optim.Optimizer):
    """Muon with decoupled weight decay, RMS matching, and QK-Clip.

    The hidden-matrix update is

    ``M_t = momentum * M_(t-1) + G_t``
    ``O_t = NS(M_t) * update_rms_scale * sqrt(max(n, m))``
    ``W_t = W_(t-1) - lr * (O_t + weight_decay * W_(t-1))``

    QK-Clip is applied after the matrix update. For regular multi-head
    attention, query and key rows belonging to head ``h`` are multiplied by
    ``gamma_h ** balance`` and ``gamma_h ** (1-balance)`` respectively, where
    ``gamma_h = min(1, threshold / max_logit_h)``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        model,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        newton_schulz_steps: int = 5,
        eps: float = 1e-7,
        update_rms_scale: float = 0.2,
        qk_clip_threshold: float = 100.0,
        qk_clip_balance: float = 0.5,
        diagnostics_interval: int = 250,
        diagnostics_path: Path | None = None,
    ) -> None:
        params = list(params)
        if not params:
            raise ValueError("MuonClip requires at least one parameter")
        if any(parameter.ndim != 2 for parameter in params):
            raise ValueError("MuonClip accepts only 2-D parameters")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if float(weight_decay) < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if int(newton_schulz_steps) < 1 or float(eps) <= 0.0:
            raise ValueError("Newton--Schulz steps and epsilon must be positive")
        if float(update_rms_scale) <= 0.0:
            raise ValueError("update_rms_scale must be positive")
        if float(qk_clip_threshold) <= 0.0:
            raise ValueError("qk_clip_threshold must be positive")
        if not 0.0 <= float(qk_clip_balance) <= 1.0:
            raise ValueError("qk_clip_balance must be in [0, 1]")
        if int(diagnostics_interval) < 1:
            raise ValueError("diagnostics_interval must be positive")

        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "weight_decay": float(weight_decay),
            "newton_schulz_steps": int(newton_schulz_steps),
            "eps": float(eps),
            "update_rms_scale": float(update_rms_scale),
            "qk_clip_threshold": float(qk_clip_threshold),
            "qk_clip_balance": float(qk_clip_balance),
        }
        super().__init__(params, defaults)
        self.model = model
        self.diagnostics_interval = int(diagnostics_interval)
        self.diagnostics_path = diagnostics_path
        self.step_index = 0
        self.last_diagnostics: dict[str, float] = {}
        self._diagnostic_interval_state: dict[str, torch.Tensor] | None = None

    @property
    def qk_clip_threshold(self) -> float:
        return float(self.param_groups[0]["qk_clip_threshold"])

    @property
    def qk_clip_balance(self) -> float:
        return float(self.param_groups[0]["qk_clip_balance"])

    def reset_qk_tracking(self) -> None:
        for block in self.model.blocks:
            setattr(block.attn, "_muonclip_max_logits", None)

    def _consume_qk_logits(self) -> tuple[torch.Tensor, ...]:
        values: list[torch.Tensor] = []
        for block_index, block in enumerate(self.model.blocks):
            value = getattr(block.attn, "_muonclip_max_logits", None)
            setattr(block.attn, "_muonclip_max_logits", None)
            if value is None:
                raise RuntimeError(
                    "MuonClip did not observe pre-softmax QK logits for "
                    f"block {block_index}; the dedicated MuonClip launcher "
                    "must be used for this optimizer"
                )
            value = value.detach().float().reshape(-1)
            if value.numel() != block.attn.n_head:
                raise RuntimeError("QK-logit head inventory changed")
            values.append(value)
        return tuple(values)

    def _empty_diagnostic_interval(
        self,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return {
            "steps": zero.clone(),
            "head_observations": zero.clone(),
            "active_heads": zero.clone(),
            "sum_max_logit": zero.clone(),
            "max_logit": torch.full(
                (),
                -float("inf"),
                device=device,
                dtype=torch.float32,
            ),
            "sum_gamma": zero.clone(),
            "min_gamma": torch.ones(
                (),
                device=device,
                dtype=torch.float32,
            ),
        }

    def _ensure_diagnostic_interval(
        self,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if self._diagnostic_interval_state is None:
            self._diagnostic_interval_state = self._empty_diagnostic_interval(
                device
            )
        return self._diagnostic_interval_state

    @torch.no_grad()
    def _apply_qk_clip(self) -> None:
        observations = self._consume_qk_logits()
        threshold = self.qk_clip_threshold
        balance = self.qk_clip_balance
        all_logits: list[torch.Tensor] = []
        all_gamma: list[torch.Tensor] = []

        for block, max_logits in zip(
            self.model.blocks,
            observations,
            strict=True,
        ):
            ones = torch.ones_like(max_logits)
            gamma = torch.where(
                max_logits > threshold,
                threshold
                / max_logits.clamp_min(
                    torch.finfo(max_logits.dtype).tiny
                ),
                ones,
            )
            q_scale = gamma.pow(balance)
            k_scale = gamma.pow(1.0 - balance)
            head_width = block.attn.n_embd // block.attn.n_head

            q_weight = block.attn.q_proj.weight.view(
                block.attn.n_head,
                head_width,
                -1,
            )
            k_weight = block.attn.k_proj.weight.view(
                block.attn.n_head,
                head_width,
                -1,
            )
            q_weight.mul_(q_scale.to(q_weight)[:, None, None])
            k_weight.mul_(k_scale.to(k_weight)[:, None, None])
            if block.attn.q_proj.bias is not None:
                block.attn.q_proj.bias.view(
                    block.attn.n_head,
                    head_width,
                ).mul_(q_scale.to(block.attn.q_proj.bias)[:, None])
            if block.attn.k_proj.bias is not None:
                block.attn.k_proj.bias.view(
                    block.attn.n_head,
                    head_width,
                ).mul_(k_scale.to(block.attn.k_proj.bias)[:, None])

            all_logits.append(max_logits)
            all_gamma.append(gamma)

        logits = torch.cat(all_logits)
        gamma = torch.cat(all_gamma)
        interval = self._ensure_diagnostic_interval(logits.device)
        interval["steps"].add_(1.0)
        interval["head_observations"].add_(float(logits.numel()))
        interval["active_heads"].add_((gamma < 1.0).float().sum())
        interval["sum_max_logit"].add_(logits.sum())
        interval["max_logit"].copy_(
            torch.maximum(interval["max_logit"], logits.max())
        )
        interval["sum_gamma"].add_(gamma.sum())
        interval["min_gamma"].copy_(
            torch.minimum(interval["min_gamma"], gamma.min())
        )

    def _flush_diagnostics(self) -> None:
        interval = self._diagnostic_interval_state
        if interval is None:
            return
        values = {
            key: float(value.detach().cpu())
            for key, value in interval.items()
        }
        observations = max(values["head_observations"], 1.0)
        diagnostics = {
            "step": float(self.step_index),
            "threshold": float(self.qk_clip_threshold),
            "steps_in_interval": values["steps"],
            "head_observations": values["head_observations"],
            "active_heads": values["active_heads"],
            "active_fraction": values["active_heads"] / observations,
            "mean_max_logit": values["sum_max_logit"] / observations,
            "max_logit": values["max_logit"],
            "mean_gamma": values["sum_gamma"] / observations,
            "min_gamma": values["min_gamma"],
        }
        self.last_diagnostics = diagnostics
        self._diagnostic_interval_state = None
        self._write_diagnostics(diagnostics)

    def _write_diagnostics(self, values: dict[str, float]) -> None:
        if self.diagnostics_path is None:
            return
        path = Path(self.diagnostics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "step",
            "threshold",
            "steps_in_interval",
            "head_observations",
            "active_heads",
            "active_fraction",
            "mean_max_logit",
            "max_logit",
            "mean_gamma",
            "min_gamma",
        ]
        write_header = not path.is_file() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(values)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            rms_scale = float(group["update_rms_scale"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    raise RuntimeError("MuonClip does not support sparse gradients")
                state = self.state[parameter]
                buffer = state.get("momentum_buffer")
                if buffer is None:
                    buffer = torch.zeros_like(gradient)
                    state["momentum_buffer"] = buffer
                # Public Moonlight form: unnormalized momentum accumulation.
                buffer.mul_(momentum).add_(gradient)
                update_source = (
                    gradient.add(buffer, alpha=momentum)
                    if bool(group["nesterov"])
                    else buffer
                )
                update = _zeropower(
                    update_source,
                    steps=int(group["newton_schulz_steps"]),
                    eps=float(group["eps"]),
                )
                update.mul_(
                    rms_scale
                    * math.sqrt(
                        max(parameter.shape[0], parameter.shape[1])
                    )
                )
                decay = float(group["weight_decay"])
                if decay:
                    parameter.mul_(max(0.0, 1.0 - lr * decay))
                parameter.add_(update, alpha=-lr)

        self.step_index += 1
        self._apply_qk_clip()
        if self.step_index % self.diagnostics_interval == 0:
            self._flush_diagnostics()
        return loss

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["muonclip_global_state"] = {
            "step_index": int(self.step_index),
            "last_diagnostics": dict(self.last_diagnostics),
            "diagnostic_interval_state": self._diagnostic_interval_state,
        }
        return payload

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        payload = deepcopy(state_dict)
        global_state = payload.pop("muonclip_global_state", {})
        super().load_state_dict(payload)
        self.step_index = int(global_state.get("step_index", 0))
        self.last_diagnostics = dict(
            global_state.get("last_diagnostics", {})
        )
        interval = global_state.get("diagnostic_interval_state")
        if interval is None:
            self._diagnostic_interval_state = None
        else:
            device = self.param_groups[0]["params"][0].device
            self._diagnostic_interval_state = {
                key: value.to(device)
                if torch.is_tensor(value)
                else torch.tensor(value, device=device, dtype=torch.float32)
                for key, value in interval.items()
            }


def _record_qk_logits(attention, scores: torch.Tensor) -> None:
    if not attention.training:
        return
    value = (
        scores.detach()
        .amax(dim=(-2, -1))
        .amax(dim=0)
        .float()
    )
    previous = getattr(attention, "_muonclip_max_logits", None)
    setattr(
        attention,
        "_muonclip_max_logits",
        value if previous is None else torch.maximum(previous, value),
    )


def _muonclip_attention_forward(attention, x: torch.Tensor) -> torch.Tensor:
    """Track exact causal pre-softmax maxima while preserving native SDPA."""

    batch, sequence, channels = x.shape
    head_width = channels // attention.n_head
    q = attention.q_proj(x).view(
        batch,
        sequence,
        attention.n_head,
        head_width,
    ).transpose(1, 2)
    k = attention.k_proj(x).view(
        batch,
        sequence,
        attention.n_head,
        head_width,
    ).transpose(1, 2)
    v = attention.v_proj(x).view(
        batch,
        sequence,
        attention.n_head,
        head_width,
    ).transpose(1, 2)
    dropout_p = attention.dropout if attention.training else 0.0

    if q.device.type == "xla":
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        mask = attention.causal_mask[:, :, :sequence, :sequence]
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )
        _record_qk_logits(attention, scores)
        probabilities = F.softmax(scores, dim=-1)
        if dropout_p:
            probabilities = F.dropout(
                probabilities,
                p=dropout_p,
                training=True,
            )
        y = probabilities @ v
    else:
        if attention.training:
            with torch.no_grad():
                scores = (
                    q.detach() @ k.detach().transpose(-2, -1)
                ) / math.sqrt(q.shape[-1])
                mask = attention.causal_mask[:, :, :sequence, :sequence]
                scores = scores.masked_fill(
                    ~mask,
                    torch.finfo(scores.dtype).min,
                )
                _record_qk_logits(attention, scores)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

    y = y.transpose(1, 2).contiguous().view(
        batch,
        sequence,
        channels,
    )
    return attention.resid_dropout(attention.out_proj(y))


def _validate_muonclip_profile(profile: dict[str, Any]) -> None:
    warmup = float(profile.get("warmup_fraction", -1.0))
    if not 0.0 <= warmup < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if str(profile.get("schedule")) != "warmup_cosine":
        raise ValueError("MuonClip currently requires warmup_cosine")
    peak = float(profile["learning_rate"])
    floor = float(profile["min_learning_rate"])
    if peak <= 0.0 or floor < 0.0 or floor > peak:
        raise ValueError("MuonClip learning-rate peak/floor are inconsistent")
    if not 0.0 <= float(profile["momentum"]) < 1.0:
        raise ValueError("MuonClip momentum must be in [0, 1)")
    if int(profile["newton_schulz_steps"]) < 1:
        raise ValueError("MuonClip newton_schulz_steps must be positive")
    if float(profile.get("muon_epsilon", 0.0)) <= 0.0:
        raise ValueError("MuonClip muon_epsilon must be positive")
    if float(profile["weight_decay"]) < 0.0:
        raise ValueError("MuonClip weight_decay must be nonnegative")
    if float(profile["update_rms_scale"]) <= 0.0:
        raise ValueError("MuonClip update_rms_scale must be positive")
    if float(profile["qk_clip_threshold"]) <= 0.0:
        raise ValueError("MuonClip qk_clip_threshold must be positive")
    balance = float(profile.get("qk_clip_balance", 0.5))
    if not 0.0 <= balance <= 1.0:
        raise ValueError("MuonClip qk_clip_balance must be in [0, 1]")


def _partition(model):
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    hidden = [
        parameter
        for name, parameter in named
        if name.startswith("blocks.") and parameter.ndim == 2
    ]
    hidden_ids = {id(parameter) for parameter in hidden}
    auxiliary = [
        (name, parameter)
        for name, parameter in named
        if id(parameter) not in hidden_ids
    ]
    if not hidden or not auxiliary:
        raise ValueError(
            "MuonClip partition must contain hidden matrices and auxiliary parameters"
        )
    return hidden, auxiliary


def _decay_groups(named, weight_decay: float) -> list[dict[str, Any]]:
    return [
        {
            "params": [parameter for _, parameter in named if parameter.ndim >= 2],
            "weight_decay": float(weight_decay),
        },
        {
            "params": [parameter for _, parameter in named if parameter.ndim < 2],
            "weight_decay": 0.0,
        },
    ]


def install_muonclip_extension() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    # Training must not depend on plotting/aggregation helpers. In particular,
    # isolated workers may execute from a namespace-style source package where
    # analysis.py is unavailable, as long as the training modules are present.
    try:
        analysis_module = importlib.import_module(".analysis", __package__)
    except ModuleNotFoundError as exc:
        expected = f"{__package__}.analysis"
        if exc.name != expected:
            raise
        analysis_module = None

    from . import config as config_module
    from . import engine as engine_module
    from . import model as model_module
    from . import optimizers as optimizers_module
    from . import training as training_module
    from . import train_loop as train_loop_module

    all_optimizers = tuple(
        dict.fromkeys((*config_module.SUPPORTED_OPTIMIZERS, "muon_clip"))
    )
    config_module.SUPPORTED_OPTIMIZERS = all_optimizers
    engine_module.SUPPORTED_OPTIMIZERS = all_optimizers
    training_module.SUPPORTED_OPTIMIZERS = all_optimizers
    if analysis_module is not None:
        analysis_module.OPTIMIZER_LABELS["muon_clip"] = (
            "MuonClip + auxiliary AdamW"
        )
        analysis_module.OPTIMIZER_COLORS["muon_clip"] = "#CC79A7"

    original_validate = config_module.validate_optimizer_profile
    original_make_handles = optimizers_module.make_optimizer_handles
    original_zero_grad = optimizers_module.zero_grad
    original_training_run_one = training_module.run_one

    def validate_optimizer_profile(profile: dict[str, Any]) -> None:
        if str(profile.get("family", "")) == "muon_clip":
            _validate_muonclip_profile(profile)
            return
        original_validate(profile)

    def make_optimizer_handles(model, profile: dict):
        if str(profile.get("family", "")) != "muon_clip":
            return original_make_handles(model, profile)

        hidden, auxiliary_named = _partition(model)
        learning_rate = float(profile["learning_rate"])
        minimum = float(profile["min_learning_rate"])
        diagnostics_path = (
            None
            if _CURRENT_RUN_DIR is None
            else _CURRENT_RUN_DIR / "muonclip_qk.csv"
        )
        primary = MuonClip(
            hidden,
            model=model,
            lr=learning_rate,
            momentum=float(profile["momentum"]),
            nesterov=bool(profile.get("nesterov", True)),
            weight_decay=float(profile["weight_decay"]),
            newton_schulz_steps=int(profile["newton_schulz_steps"]),
            eps=float(profile.get("muon_epsilon", 1e-7)),
            update_rms_scale=float(profile["update_rms_scale"]),
            qk_clip_threshold=float(profile["qk_clip_threshold"]),
            qk_clip_balance=float(profile.get("qk_clip_balance", 0.5)),
            diagnostics_interval=int(
                profile.get("qk_diagnostics_interval", 250)
            ),
            diagnostics_path=diagnostics_path,
        )
        auxiliary = torch.optim.AdamW(
            _decay_groups(auxiliary_named, float(profile["weight_decay"])),
            lr=learning_rate,
            betas=(float(profile["beta1"]), float(profile["beta2"])),
            eps=float(profile["epsilon"]),
        )
        return [
            optimizers_module.OptimizerHandle(
                role="primary",
                optimizer=primary,
                peak_lr=learning_rate,
                min_lr=minimum,
            ),
            optimizers_module.OptimizerHandle(
                role="auxiliary",
                optimizer=auxiliary,
                peak_lr=learning_rate,
                min_lr=minimum,
            ),
        ]

    def zero_grad(handles) -> None:
        for handle in handles:
            if isinstance(handle.optimizer, MuonClip):
                handle.optimizer.reset_qk_tracking()
        original_zero_grad(handles)

    def run_one_with_context(*args, **kwargs):
        global _CURRENT_RUN_DIR
        optimizer_name = str(kwargs.get("optimizer_name", ""))
        results_root = Path(kwargs["results_root"])
        seed = int(kwargs["seed"])
        _CURRENT_RUN_DIR = (
            results_root / optimizer_name / f"seed_{seed}"
        )
        try:
            return original_training_run_one(*args, **kwargs)
        finally:
            _CURRENT_RUN_DIR = None

    config_module.validate_optimizer_profile = validate_optimizer_profile
    optimizers_module.make_optimizer_handles = make_optimizer_handles
    engine_module.make_optimizer_handles = make_optimizer_handles
    optimizers_module.zero_grad = zero_grad
    train_loop_module.zero_grad = zero_grad
    training_module.run_one = run_one_with_context
    model_module.CausalSelfAttention.forward = _muonclip_attention_forward

    _INSTALLED = True


def main() -> None:
    install_muonclip_extension()
    from .training import main as training_main

    training_main()


if __name__ == "__main__":
    main()
