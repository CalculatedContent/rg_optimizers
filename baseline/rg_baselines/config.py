"""Configuration for the clean MNIST optimizer baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

OptimizerName = Literal["sgd_momentum", "adamw", "sgd_momentum_muon"]


@dataclass(frozen=True)
class BaselineConfig:
    """Complete configuration for one MLP3/MNIST baseline run."""

    optimizer: OptimizerName
    seed: int = 1337
    epochs: int = 20
    batch_size: int = 128
    num_workers: int = 0
    grad_clip_norm: float = 1.0
    train_eval_max_batches: Optional[int] = None

    # Ordinary SGD + classical momentum.
    sgd_learning_rate: float = 5e-2
    sgd_momentum: float = 0.9
    sgd_dampening: float = 0.0
    sgd_nesterov: bool = False
    sgd_weight_decay: float = 1e-4

    # AdamW.
    adamw_learning_rate: float = 1e-3
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 1e-2
    adamw_amsgrad: bool = False

    # Muon on hidden matrix updates; auxiliary parameters use SGD + momentum.
    muon_parameter_names: tuple[str, ...] = ("fc1.weight", "fc2.weight")
    muon_learning_rate: float = 2e-2
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_weight_decay: float = 0.0
    muon_newton_schulz_steps: int = 5
    muon_eps: float = 1e-7
    muon_aux_learning_rate: float = 5e-2
    muon_aux_momentum: float = 0.9
    muon_aux_dampening: float = 0.0
    muon_aux_nesterov: bool = False
    muon_aux_weight_decay: float = 1e-4

    # WeightWatcher checkpoint analysis. ERG metrics come from ERG=True.
    # Correlation traps come from the randomized MP analysis in randomize=True.
    ww_min_evals: int = 8
    ww_max_evals: Optional[int] = None
    ww_svd_method: str = "accurate"
    ww_randomize: bool = True

    save_epoch_checkpoints: bool = False
    strict_metrics: bool = True

    def validate(self) -> None:
        if self.optimizer not in {"sgd_momentum", "adamw", "sgd_momentum_muon"}:
            raise ValueError(f"Unknown optimizer: {self.optimizer!r}")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        for name, value in {
            "sgd_learning_rate": self.sgd_learning_rate,
            "adamw_learning_rate": self.adamw_learning_rate,
            "muon_learning_rate": self.muon_learning_rate,
            "muon_aux_learning_rate": self.muon_aux_learning_rate,
        }.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "sgd_weight_decay": self.sgd_weight_decay,
            "adamw_weight_decay": self.adamw_weight_decay,
            "muon_weight_decay": self.muon_weight_decay,
            "muon_aux_weight_decay": self.muon_aux_weight_decay,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in {
            "sgd_momentum": self.sgd_momentum,
            "muon_momentum": self.muon_momentum,
            "muon_aux_momentum": self.muon_aux_momentum,
        }.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if self.sgd_dampening < 0.0 or self.muon_aux_dampening < 0.0:
            raise ValueError("dampening must be non-negative")
        if self.sgd_nesterov and (self.sgd_momentum <= 0.0 or self.sgd_dampening != 0.0):
            raise ValueError("Nesterov SGD requires positive momentum and zero dampening")
        if self.muon_aux_nesterov and (
            self.muon_aux_momentum <= 0.0 or self.muon_aux_dampening != 0.0
        ):
            raise ValueError("Nesterov auxiliary SGD requires positive momentum and zero dampening")
        if self.muon_newton_schulz_steps < 1:
            raise ValueError("muon_newton_schulz_steps must be positive")
        if self.muon_eps <= 0.0 or self.adamw_eps <= 0.0:
            raise ValueError("optimizer eps values must be positive")
        if self.ww_min_evals < 2:
            raise ValueError("ww_min_evals must be at least two")
        if not self.ww_randomize:
            raise ValueError(
                "ww_randomize must be True because num_traps from "
                "WeightWatcher analyze(randomize=True) is a required baseline metric"
            )

    @property
    def optimizer_label(self) -> str:
        return {
            "sgd_momentum": "SGD + momentum",
            "adamw": "AdamW",
            "sgd_momentum_muon": "SGD + momentum + Muon",
        }[self.optimizer]

    @property
    def run_slug(self) -> str:
        return self.optimizer
