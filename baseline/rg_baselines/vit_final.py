"""Final qualified CIFAR-10 small-ViT reference runtime.

The original implementation intentionally avoided a timm dependency. This
module keeps that portability while correcting the material differences between
the custom model/scheduler and the DeiT reference recipe:

* LayerNorm uses eps=1e-6.
* The patch projection keeps Conv2d's fan-in initialization instead of being
  overwritten by the transformer Linear initialization.
* Warm-up begins from an explicit small learning rate.
* Cosine decay reaches the non-zero floor before a fixed cooldown at that floor.
* The validation-best checkpoint is guaranteed to represent epoch zero when the
  untrained model is genuinely best.

The public notebook imports this module. The lower-level ``vit_cifar10`` module
is retained for backward compatibility and unit-level components.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from . import vit_cifar10 as core
from . import vit_runtime as hardened


@dataclass(frozen=True)
class ViTBaselineConfig(core.ViTBaselineConfig):
    """Source-faithful final recipe for the selected six-block CIFAR-10 ViT."""

    recipe_version: int = 4
    norm_eps: float = 1e-6
    cooldown_epochs: int = 10

    sgd_warmup_start_lr: float = 1e-3
    adamw_warmup_start_lr: float = 1e-6
    muon_warmup_start_lr: float = 2e-4
    muon_aux_warmup_start_lr: float = 3e-6

    def validate(self) -> None:
        super().validate()
        if self.recipe_version < 1:
            raise ValueError("recipe_version must be positive")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be positive")
        if not 0 <= self.cooldown_epochs < self.epochs:
            raise ValueError("cooldown_epochs must lie in [0, epochs)")
        for name, warmup in {
            "sgd": self.sgd_warmup_epochs,
            "adamw": self.adamw_warmup_epochs,
            "muon": self.muon_warmup_epochs,
        }.items():
            if warmup + self.cooldown_epochs >= self.epochs:
                raise ValueError(
                    f"{name} warm-up plus cooldown must be shorter than training"
                )
        for name, start, peak in (
            ("sgd", self.sgd_warmup_start_lr, self.sgd_lr),
            ("adamw", self.adamw_warmup_start_lr, self.adamw_lr),
            ("muon", self.muon_warmup_start_lr, self.muon_lr),
            ("muon_aux", self.muon_aux_warmup_start_lr, self.muon_aux_lr),
        ):
            if start < 0.0 or start > peak:
                raise ValueError(
                    f"{name} warm-up start must lie between zero and peak LR"
                )


class SmallViT(core.SmallViT):
    """The selected architecture with corrected norm and patch initialization."""

    def __init__(self, config: ViTBaselineConfig) -> None:
        config.validate()
        super().__init__(config)
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                module.eps = float(config.norm_eps)

        # ``core.SmallViT.apply`` initializes every Conv2d with transformer
        # trunc-normal weights. DeiT/timm leaves the patch Conv2d on its fan-in
        # initialization; restore that initialization explicitly.
        self.patch_embed.proj.reset_parameters()


def cosine_learning_rate(
    epoch_index: int,
    *,
    epochs: int,
    warmup_epochs: int,
    cooldown_epochs: int,
    warmup_start_lr: float,
    peak_lr: float,
    min_lr: float,
) -> float:
    """DeiT-style warm-up, cosine decay, and non-zero cooldown floor."""

    if epochs < 2:
        raise ValueError("epochs must be at least two")
    if not 0 <= warmup_epochs < epochs:
        raise ValueError("warmup_epochs must lie in [0, epochs)")
    if not 0 <= cooldown_epochs < epochs:
        raise ValueError("cooldown_epochs must lie in [0, epochs)")
    if warmup_epochs + cooldown_epochs >= epochs:
        raise ValueError("warm-up plus cooldown must be shorter than training")
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    if not 0.0 <= warmup_start_lr <= peak_lr:
        raise ValueError("warm-up start and peak LR are inconsistent")
    if not 0.0 <= min_lr <= peak_lr:
        raise ValueError("minimum and peak LR are inconsistent")

    index = min(int(epoch_index), int(epochs) - 1)
    if warmup_epochs and index < warmup_epochs:
        progress = index / float(warmup_epochs)
        return float(warmup_start_lr) + progress * (
            float(peak_lr) - float(warmup_start_lr)
        )

    decay_end = int(epochs) - int(cooldown_epochs)
    if index >= decay_end:
        return float(min_lr)

    decay_epochs = decay_end - int(warmup_epochs)
    progress = (index - int(warmup_epochs)) / max(1, decay_epochs - 1)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + cosine * (float(peak_lr) - float(min_lr))


def set_learning_rates(
    optimizer,
    name: str,
    config: ViTBaselineConfig,
    epoch_index: int,
) -> dict[str, float]:
    """Apply the optimizer-specific final reference schedule."""

    config.validate()
    if name == "sgd_momentum":
        primary = cosine_learning_rate(
            epoch_index,
            epochs=config.epochs,
            warmup_epochs=config.sgd_warmup_epochs,
            cooldown_epochs=config.cooldown_epochs,
            warmup_start_lr=config.sgd_warmup_start_lr,
            peak_lr=config.sgd_lr,
            min_lr=config.sgd_min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = primary
        return {"primary": primary, "auxiliary": float("nan")}

    if name == "adamw":
        primary = cosine_learning_rate(
            epoch_index,
            epochs=config.epochs,
            warmup_epochs=config.adamw_warmup_epochs,
            cooldown_epochs=config.cooldown_epochs,
            warmup_start_lr=config.adamw_warmup_start_lr,
            peak_lr=config.adamw_lr,
            min_lr=config.adamw_min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = primary
        return {"primary": primary, "auxiliary": float("nan")}

    if name != "muon":
        raise ValueError(f"unknown optimizer {name!r}")
    matrix_lr = cosine_learning_rate(
        epoch_index,
        epochs=config.epochs,
        warmup_epochs=config.muon_warmup_epochs,
        cooldown_epochs=config.cooldown_epochs,
        warmup_start_lr=config.muon_warmup_start_lr,
        peak_lr=config.muon_lr,
        min_lr=config.muon_min_lr,
    )
    auxiliary_lr = cosine_learning_rate(
        epoch_index,
        epochs=config.epochs,
        warmup_epochs=config.muon_warmup_epochs,
        cooldown_epochs=config.cooldown_epochs,
        warmup_start_lr=config.muon_aux_warmup_start_lr,
        peak_lr=config.muon_aux_lr,
        min_lr=config.muon_aux_min_lr,
    )
    optimizer.set_learning_rates(matrix_lr, auxiliary_lr)
    return {"primary": matrix_lr, "auxiliary": auxiliary_lr}


def _validation_selected_epoch(history: pd.DataFrame) -> int:
    candidates = history.dropna(subset=["validation_loss"]).sort_values(
        ["validation_loss", "epoch"], ascending=[True, True]
    )
    if candidates.empty:
        raise RuntimeError("ViT history contains no finite validation loss")
    return int(candidates.iloc[0]["epoch"])


def _ensure_validation_best(
    run_dir: Path,
    history: pd.DataFrame,
    *,
    optimizer_name: str,
    seed: int,
    config: ViTBaselineConfig,
) -> Path:
    """Repair the epoch-zero edge case before delegating to the hardened check."""

    selected_epoch = _validation_selected_epoch(history)
    best_path = Path(run_dir) / "checkpoint_best.pt"
    if best_path.is_file() and selected_epoch == 0:
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        if int(payload.get("epoch", -1)) != 0:
            best_path.unlink()
    return _ORIGINAL_ENSURE_BEST(
        run_dir,
        history,
        optimizer_name=optimizer_name,
        seed=seed,
        config=config,
    )


_ORIGINAL_ENSURE_BEST = hardened._ensure_best_checkpoint


def run_vit_baseline(
    optimizer_name: str,
    seed: int,
    *,
    data_dir: Path,
    output_dir: Path,
    config: ViTBaselineConfig = ViTBaselineConfig(),
    device: torch.device | None = None,
    progress: bool = True,
    resume: bool = True,
):
    """Run the final reference while preserving the hardened restart contract."""

    config.validate()
    original_model = core.SmallViT
    original_schedule = core.set_learning_rates
    original_ensure = hardened._ensure_best_checkpoint
    core.SmallViT = SmallViT
    core.set_learning_rates = set_learning_rates
    hardened._ensure_best_checkpoint = _ensure_validation_best
    try:
        return hardened.run_vit_baseline(
            optimizer_name,
            int(seed),
            data_dir=Path(data_dir),
            output_dir=Path(output_dir),
            config=config,
            device=device,
            progress=progress,
            resume=resume,
        )
    finally:
        core.SmallViT = original_model
        core.set_learning_rates = original_schedule
        hardened._ensure_best_checkpoint = original_ensure


DEFAULT_VIT_SEEDS = core.DEFAULT_VIT_SEEDS
choose_device = core.choose_device
build_optimizer = core.build_optimizer
