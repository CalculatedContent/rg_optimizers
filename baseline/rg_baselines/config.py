"""Configuration for the clean MNIST optimizer baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

OptimizerName = Literal["sgd_momentum", "adamw", "sgd_momentum_muon"]


@dataclass(frozen=True)
class BaselineConfig:
    """Complete configuration for one MLP3/MNIST reference run.

    The official 60,000-example MNIST training set is deterministically split
    into 55,000 optimization examples and 5,000 validation examples. The
    official test set is monitoring-only. Every optimizer uses step-level
    linear warm-up followed by cosine decay to a non-zero floor.

    ``sgd_momentum_muon`` remains the historical result-directory key; its
    implementation is Muon on the two hidden matrices plus auxiliary AdamW.
    """

    optimizer: OptimizerName
    seed: int = 1337
    epochs: int = 30
    batch_size: int = 128
    validation_size: int = 5_000
    split_seed: int = 20_260_807
    num_workers: int = 0
    grad_clip_norm: float = 1.0
    train_eval_max_batches: Optional[int] = None
    schedule: str = "warmup_cosine"
    checkpoint_every_epochs: int = 1
    test_monitoring_only: bool = True

    # SGD + Nesterov momentum.
    sgd_learning_rate: float = 5e-2
    sgd_min_learning_rate: float = 5e-4
    sgd_warmup_epochs: int = 2
    sgd_momentum: float = 0.9
    sgd_dampening: float = 0.0
    sgd_nesterov: bool = True
    sgd_weight_decay: float = 1e-4

    # AdamW.
    adamw_learning_rate: float = 1e-3
    adamw_min_learning_rate: float = 1e-5
    adamw_warmup_epochs: int = 1
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 1e-2
    adamw_amsgrad: bool = False

    # Muon on hidden matrices; classifier/bias parameters use auxiliary AdamW.
    muon_parameter_names: tuple[str, ...] = ("fc1.weight", "fc2.weight")
    muon_learning_rate: float = 2e-2
    muon_min_learning_rate: float = 2e-3
    muon_warmup_epochs: int = 2
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_weight_decay: float = 1e-2
    muon_newton_schulz_steps: int = 5
    muon_eps: float = 1e-7
    muon_aux_learning_rate: float = 3e-4
    muon_aux_min_learning_rate: float = 3e-5
    muon_aux_beta1: float = 0.9
    muon_aux_beta2: float = 0.95
    muon_aux_eps: float = 1e-8
    muon_aux_weight_decay: float = 1e-2

    # WeightWatcher checkpoint analysis. ERG metrics come from ERG=True and
    # correlation traps come directly from randomize=True.
    ww_min_evals: int = 8
    ww_max_evals: Optional[int] = None
    ww_svd_method: str = "accurate"
    ww_randomize: bool = True

    save_epoch_checkpoints: bool = True
    strict_metrics: bool = True

    def validate(self) -> None:
        if self.optimizer not in {
            "sgd_momentum",
            "adamw",
            "sgd_momentum_muon",
        }:
            raise ValueError(f"Unknown optimizer: {self.optimizer!r}")
        if self.epochs < 2:
            raise ValueError(
                "epochs must be at least two for warm-up/cosine schedules"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 < self.validation_size < 60_000:
            raise ValueError("validation_size must lie between zero and 60,000")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if self.checkpoint_every_epochs < 1:
            raise ValueError("checkpoint_every_epochs must be positive")
        if not self.test_monitoring_only:
            raise ValueError("the official test set must remain monitoring-only")
        if self.schedule != "warmup_cosine":
            raise ValueError(
                "only the preregistered warmup_cosine schedule is supported"
            )

        lr_pairs = {
            "sgd": (self.sgd_learning_rate, self.sgd_min_learning_rate),
            "adamw": (self.adamw_learning_rate, self.adamw_min_learning_rate),
            "muon": (self.muon_learning_rate, self.muon_min_learning_rate),
            "muon_aux": (
                self.muon_aux_learning_rate,
                self.muon_aux_min_learning_rate,
            ),
        }
        for name, (peak, floor) in lr_pairs.items():
            if peak <= 0.0 or floor < 0.0 or floor > peak:
                raise ValueError(
                    f"{name} peak/floor learning rates are inconsistent"
                )

        for name, value in {
            "sgd_warmup_epochs": self.sgd_warmup_epochs,
            "adamw_warmup_epochs": self.adamw_warmup_epochs,
            "muon_warmup_epochs": self.muon_warmup_epochs,
        }.items():
            if value < 0 or value >= self.epochs:
                raise ValueError(
                    f"{name} must satisfy 0 <= warmup < epochs"
                )

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
        }.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if self.sgd_dampening < 0.0:
            raise ValueError("dampening must be non-negative")
        if self.sgd_nesterov and (
            self.sgd_momentum <= 0.0 or self.sgd_dampening != 0.0
        ):
            raise ValueError(
                "Nesterov SGD requires positive momentum and zero dampening"
            )

        for name, value in {
            "adamw_beta1": self.adamw_beta1,
            "adamw_beta2": self.adamw_beta2,
            "muon_aux_beta1": self.muon_aux_beta1,
            "muon_aux_beta2": self.muon_aux_beta2,
        }.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if self.muon_newton_schulz_steps < 1:
            raise ValueError("muon_newton_schulz_steps must be positive")
        if min(self.muon_eps, self.adamw_eps, self.muon_aux_eps) <= 0.0:
            raise ValueError("optimizer eps values must be positive")
        if self.ww_min_evals < 2:
            raise ValueError("ww_min_evals must be at least two")
        if not self.ww_randomize:
            raise ValueError(
                "ww_randomize must be True because WeightWatcher "
                "analyze(randomize=True) supplies the required num_traps metric"
            )

    @property
    def warmup_epochs(self) -> int:
        return {
            "sgd_momentum": self.sgd_warmup_epochs,
            "adamw": self.adamw_warmup_epochs,
            "sgd_momentum_muon": self.muon_warmup_epochs,
        }[self.optimizer]

    @property
    def optimizer_label(self) -> str:
        return {
            "sgd_momentum": "SGD + Nesterov momentum",
            "adamw": "AdamW",
            "sgd_momentum_muon": "Muon + auxiliary AdamW",
        }[self.optimizer]

    @property
    def run_slug(self) -> str:
        return self.optimizer
