"""Optimized small-ViT CIFAR-10 optimizer baselines.

The reference recipe follows the strong DeiT-style training ingredients that
matter for training Vision Transformers from scratch: a fixed validation split,
RandAugment, mixup, CutMix, label smoothing, random erasing, stochastic depth,
optimizer-specific warm-up/cosine schedules, and non-zero learning-rate floors.
The implementation remains small enough for a single Apple-MPS device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .muon import zeropower_via_newton_schulz_5

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_VIT_SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class ViTBaselineConfig:
    epochs: int = 300
    batch_size: int = 128
    validation_size: int = 5_000
    split_seed: int = 20_260_807
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path: float = 0.10
    num_classes: int = 10

    randaugment_ops: int = 2
    randaugment_magnitude: int = 9
    color_jitter: float = 0.30
    random_erasing_probability: float = 0.25
    mixup_alpha: float = 0.80
    cutmix_alpha: float = 1.00
    mix_switch_probability: float = 0.50
    label_smoothing: float = 0.10
    grad_clip: float = 1.0

    ww_every: int = 1
    ww_min_evals: int = 20
    checkpoint_every: int = 25
    test_monitoring_only: bool = True

    sgd_lr: float = 0.10
    sgd_min_lr: float = 1e-3
    sgd_warmup_epochs: int = 5
    sgd_momentum: float = 0.90
    sgd_weight_decay: float = 5e-4

    # DeiT's 5e-4 reference LR, linearly scaled from batch 512 to batch 128.
    adamw_lr: float = 1.25e-4
    adamw_min_lr: float = 1e-5
    adamw_warmup_epochs: int = 5
    adamw_beta1: float = 0.90
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.05

    muon_lr: float = 0.02
    muon_min_lr: float = 0.002
    muon_warmup_epochs: int = 5
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    muon_ns_steps: int = 5
    muon_aux_lr: float = 3e-4
    muon_aux_min_lr: float = 3e-5
    muon_aux_beta1: float = 0.90
    muon_aux_beta2: float = 0.95
    muon_aux_eps: float = 1e-8
    muon_aux_weight_decay: float = 0.01

    def validate(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.epochs < 2 or self.batch_size < 1:
            raise ValueError("epochs must be >=2 and batch_size must be positive")
        if not 0 < self.validation_size < 50_000:
            raise ValueError("validation_size must lie between zero and 50,000")
        for name, warmup in {
            "sgd": self.sgd_warmup_epochs,
            "adamw": self.adamw_warmup_epochs,
            "muon": self.muon_warmup_epochs,
        }.items():
            if not 0 <= warmup < self.epochs:
                raise ValueError(f"{name} warm-up must satisfy 0 <= warmup < epochs")
        for name, peak, floor in (
            ("sgd", self.sgd_lr, self.sgd_min_lr),
            ("adamw", self.adamw_lr, self.adamw_min_lr),
            ("muon", self.muon_lr, self.muon_min_lr),
            ("muon_aux", self.muon_aux_lr, self.muon_aux_min_lr),
        ):
            if peak <= 0 or floor < 0 or floor > peak:
                raise ValueError(f"{name} peak/floor learning rates are inconsistent")
        if self.ww_every < 1 or self.ww_min_evals < 2:
            raise ValueError("WeightWatcher cadence/min_evals are invalid")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        for value in (
            self.dropout,
            self.attn_dropout,
            self.drop_path,
            self.random_erasing_probability,
            self.mix_switch_probability,
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError("probability values must lie in [0,1)")


class DropPath(nn.Module):
    """Per-example stochastic depth."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, attn_dropout, dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class SmallViT(nn.Module):
    def __init__(self, config: ViTBaselineConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.patch_embed = PatchEmbed(config.image_size, config.patch_size, config.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        drop_rates = torch.linspace(0.0, config.drop_path, config.depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                    config.attn_dropout,
                    drop_rates[index],
                )
                for index in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def _training_transform(config: ViTBaselineConfig):
    transform_list: list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(
            num_ops=config.randaugment_ops,
            magnitude=config.randaugment_magnitude,
        ),
    ]
    if config.color_jitter > 0:
        transform_list.append(
            transforms.ColorJitter(
                config.color_jitter,
                config.color_jitter,
                config.color_jitter,
            )
        )
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            transforms.RandomErasing(
                p=config.random_erasing_probability,
                value="random",
            ),
        ]
    )
    return transforms.Compose(transform_list)


def _evaluation_transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def make_datasets(data_dir: Path, config: ViTBaselineConfig):
    train_augmented = datasets.CIFAR10(
        data_dir,
        train=True,
        download=True,
        transform=_training_transform(config),
    )
    train_plain = datasets.CIFAR10(
        data_dir,
        train=True,
        download=False,
        transform=_evaluation_transform(),
    )
    test = datasets.CIFAR10(
        data_dir,
        train=False,
        download=True,
        transform=_evaluation_transform(),
    )
    generator = torch.Generator().manual_seed(config.split_seed)
    permutation = torch.randperm(50_000, generator=generator).tolist()
    validation_indices = permutation[: config.validation_size]
    training_indices = permutation[config.validation_size :]
    return (
        Subset(train_augmented, training_indices),
        Subset(train_plain, training_indices),
        Subset(train_plain, validation_indices),
        test,
    )


def make_loaders(
    data_dir: Path,
    config: ViTBaselineConfig,
    seed: int,
    device: torch.device,
):
    train, train_eval, validation, test = make_datasets(data_dir, config)
    workers = 0 if device.type == "mps" else 2
    train_generator = torch.Generator().manual_seed(seed)
    common = {"num_workers": workers, "pin_memory": device.type == "cuda"}
    return (
        DataLoader(
            train,
            config.batch_size,
            shuffle=True,
            generator=train_generator,
            drop_last=True,
            **common,
        ),
        DataLoader(train_eval, 256, shuffle=False, **common),
        DataLoader(validation, 256, shuffle=False, **common),
        DataLoader(test, 256, shuffle=False, **common),
        train_generator,
    )


def mixup_cutmix_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    config: ViTBaselineConfig,
):
    use_cutmix = config.cutmix_alpha > 0 and (
        config.mixup_alpha <= 0 or random.random() < config.mix_switch_probability
    )
    permutation = torch.randperm(x.size(0), device=x.device)
    if use_cutmix:
        lam = float(np.random.beta(config.cutmix_alpha, config.cutmix_alpha))
        height, width = x.shape[-2:]
        cut_ratio = math.sqrt(1.0 - lam)
        cut_width = max(1, int(width * cut_ratio))
        cut_height = max(1, int(height * cut_ratio))
        center_x = random.randrange(width)
        center_y = random.randrange(height)
        x1 = max(0, center_x - cut_width // 2)
        x2 = min(width, center_x + cut_width // 2)
        y1 = max(0, center_y - cut_height // 2)
        y2 = min(height, center_y + cut_height // 2)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        lam = 1.0 - area / float(width * height)
        return mixed, y, y[permutation], lam
    if config.mixup_alpha > 0:
        lam = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
        return lam * x + (1.0 - lam) * x[permutation], y, y[permutation], lam
    return x, y, y, 1.0


def cosine_learning_rate(
    epoch_index: int,
    *,
    epochs: int,
    warmup_epochs: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    if warmup_epochs and epoch_index < warmup_epochs:
        return peak_lr * (epoch_index + 1) / warmup_epochs
    progress = (epoch_index - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (peak_lr - min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss_sum += F.cross_entropy(logits, targets).item() * targets.numel()
        correct += (logits.argmax(1) == targets).sum().item()
        count += targets.numel()
    return loss_sum / count, correct / count


def _decay_groups(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    weight_decay: float,
):
    decay = [
        parameter
        for name, parameter in named_parameters
        if parameter.requires_grad
        and parameter.ndim >= 2
        and name not in {"cls_token", "pos_embed"}
    ]
    decay_ids = {id(parameter) for parameter in decay}
    no_decay = [
        parameter
        for _, parameter in named_parameters
        if parameter.requires_grad and id(parameter) not in decay_ids
    ]
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def muon_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.ndim == 2
        and name.startswith("blocks.")
        and name.endswith("weight")
    )


class MuonWithAuxAdamW:
    """Muon on hidden transformer matrices and AdamW elsewhere."""

    def __init__(self, model: nn.Module, config: ViTBaselineConfig) -> None:
        named = list(model.named_parameters())
        selected = set(muon_parameter_names(model))
        self.muon_named = [(name, parameter) for name, parameter in named if name in selected]
        self.muon_params = [parameter for _, parameter in self.muon_named]
        selected_ids = {id(parameter) for parameter in self.muon_params}
        auxiliary_named = [
            (name, parameter)
            for name, parameter in named
            if parameter.requires_grad and id(parameter) not in selected_ids
        ]
        self.muon = torch.optim.SGD(
            self.muon_params,
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            nesterov=True,
            weight_decay=0.0,
        )
        self.aux = torch.optim.AdamW(
            _decay_groups(auxiliary_named, config.muon_aux_weight_decay),
            lr=config.muon_aux_lr,
            betas=(config.muon_aux_beta1, config.muon_aux_beta2),
            eps=config.muon_aux_eps,
        )
        self.config = config

    @property
    def param_groups(self):
        return [*self.muon.param_groups, *self.aux.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.aux.zero_grad(set_to_none=set_to_none)

    def set_learning_rates(self, matrix_lr: float, auxiliary_lr: float) -> None:
        self.muon.param_groups[0]["lr"] = matrix_lr
        for group in self.aux.param_groups:
            group["lr"] = auxiliary_lr

    @torch.no_grad()
    def step(self) -> None:
        group = self.muon.param_groups[0]
        lr = float(group["lr"])
        momentum = float(group["momentum"])
        for parameter in self.muon_params:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            state = self.muon.state[parameter]
            buffer = state.get("momentum_buffer")
            if buffer is None:
                buffer = torch.zeros_like(gradient)
                state["momentum_buffer"] = buffer
            buffer.lerp_(gradient, 1.0 - momentum)
            update = gradient.lerp(buffer, momentum)
            update = zeropower_via_newton_schulz_5(
                update, steps=self.config.muon_ns_steps
            )
            if self.config.muon_weight_decay:
                parameter.mul_(1.0 - lr * self.config.muon_weight_decay)
            shape_scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
            parameter.add_(update, alpha=-lr * shape_scale)
        self.aux.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "aux": self.aux.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.aux.load_state_dict(state["aux"])


def build_optimizer(model: nn.Module, name: str, config: ViTBaselineConfig):
    named = list(model.named_parameters())
    if name == "sgd_momentum":
        return torch.optim.SGD(
            _decay_groups(named, config.sgd_weight_decay),
            lr=config.sgd_lr,
            momentum=config.sgd_momentum,
            nesterov=True,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            _decay_groups(named, config.adamw_weight_decay),
            lr=config.adamw_lr,
            betas=(config.adamw_beta1, config.adamw_beta2),
            eps=config.adamw_eps,
        )
    if name == "muon":
        return MuonWithAuxAdamW(model, config)
    raise ValueError(f"unknown optimizer {name!r}")


def set_learning_rates(
    optimizer,
    name: str,
    config: ViTBaselineConfig,
    epoch_index: int,
) -> dict[str, float]:
    if name == "sgd_momentum":
        primary = cosine_learning_rate(
            epoch_index,
            epochs=config.epochs,
            warmup_epochs=config.sgd_warmup_epochs,
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
            peak_lr=config.adamw_lr,
            min_lr=config.adamw_min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = primary
        return {"primary": primary, "auxiliary": float("nan")}
    matrix_lr = cosine_learning_rate(
        epoch_index,
        epochs=config.epochs,
        warmup_epochs=config.muon_warmup_epochs,
        peak_lr=config.muon_lr,
        min_lr=config.muon_min_lr,
    )
    auxiliary_lr = cosine_learning_rate(
        epoch_index,
        epochs=config.epochs,
        warmup_epochs=config.muon_warmup_epochs,
        peak_lr=config.muon_aux_lr,
        min_lr=config.muon_aux_min_lr,
    )
    optimizer.set_learning_rates(matrix_lr, auxiliary_lr)
    return {"primary": matrix_lr, "auxiliary": auxiliary_lr}


class _MatrixHolder(nn.Module):
    """CPU-only views of each ViT block matrix for WeightWatcher."""

    def __init__(self, model: SmallViT) -> None:
        super().__init__()
        self.metadata: list[dict[str, object]] = []
        for block_index, block in enumerate(model.blocks):
            qkv = block.attn.qkv.weight.detach().float().cpu()
            width = qkv.shape[1]
            matrices = (
                ("W_Q", qkv[:width]),
                ("W_K", qkv[width : 2 * width]),
                ("W_V", qkv[2 * width :]),
                ("W_O", block.attn.proj.weight.detach().float().cpu()),
                ("W_MLP_IN", block.mlp.fc1.weight.detach().float().cpu()),
                ("W_MLP_OUT", block.mlp.fc2.weight.detach().float().cpu()),
            )
            for matrix_type, weight in matrices:
                name = f"L{block_index:02d}_{matrix_type}"
                layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
                layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
                self.add_module(name, layer)
                self.metadata.append(
                    {
                        "matrix_name": name,
                        "matrix_type": matrix_type,
                        "block": block_index,
                    }
                )


def _attach_metadata(
    details: pd.DataFrame,
    metadata: list[dict[str, object]],
) -> pd.DataFrame:
    frame = details.copy().reset_index(drop=True)
    if len(frame) != len(metadata):
        raise RuntimeError("WeightWatcher did not return one row per selected ViT matrix")
    if "layer_id" in frame:
        order = pd.to_numeric(frame["layer_id"], errors="raise").sort_values().index.tolist()
    else:
        order = list(frame.index)
    names = [None] * len(frame)
    types = [None] * len(frame)
    blocks = [None] * len(frame)
    for item, row_index in zip(metadata, order, strict=True):
        names[row_index] = item["matrix_name"]
        types[row_index] = item["matrix_type"]
        blocks[row_index] = item["block"]
    frame.insert(0, "matrix_name", names)
    frame.insert(1, "matrix_type", types)
    frame.insert(2, "block", blocks)
    return frame


def _ww_snapshot(model: SmallViT, epoch: int, config: ViTBaselineConfig):
    import weightwatcher as ww

    holder = _MatrixHolder(model)
    details = ww.WeightWatcher(model=holder).analyze(
        ERG=True,
        randomize=True,
        plot=False,
        min_evals=config.ww_min_evals,
    )
    frame = _attach_metadata(pd.DataFrame(details), holder.metadata)
    required = {"alpha", "ERG_gap", "num_traps"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"WeightWatcher is missing required columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise RuntimeError("required WeightWatcher values contain NaN")
    frame.insert(0, "epoch", int(epoch))
    return frame


def _fingerprint(
    optimizer_name: str,
    seed: int,
    config: ViTBaselineConfig,
) -> str:
    payload = json.dumps(
        {
            "optimizer": optimizer_name,
            "seed": int(seed),
            "config": asdict(config),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: SmallViT,
    optimizer,
    train_generator: torch.Generator,
    config: ViTBaselineConfig,
    optimizer_name: str,
    seed: int,
    best_validation_loss: float,
    fingerprint: str,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_generator_state": train_generator.get_state(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "config": asdict(config),
        "optimizer": optimizer_name,
        "seed": int(seed),
        "best_validation_loss": float(best_validation_loss),
        "fingerprint": fingerprint,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    model: SmallViT,
    optimizer,
    train_generator: torch.Generator,
    expected_fingerprint: str,
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("fingerprint") != expected_fingerprint:
        raise RuntimeError("checkpoint protocol fingerprint does not match")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    train_generator.set_state(payload["train_generator_state"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    return int(payload["epoch"]), float(payload["best_validation_loss"])


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
    config.validate()
    device = device or choose_device()
    set_seed(seed)
    (
        train_loader,
        train_eval_loader,
        validation_loader,
        test_loader,
        train_generator,
    ) = make_loaders(data_dir, config, seed, device)
    model = SmallViT(config).to(device)
    optimizer = build_optimizer(model, optimizer_name, config)
    run_dir = Path(output_dir) / optimizer_name / f"seed_{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(optimizer_name, seed, config)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "optimizer": optimizer_name,
                "seed": seed,
                "fingerprint": fingerprint,
                **asdict(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    history_path = run_dir / "history.csv"
    spectral_path = run_dir / "weightwatcher_by_epoch_layer.csv"
    latest_path = run_dir / "checkpoint_latest.pt"
    best_path = run_dir / "checkpoint_best.pt"
    final_path = run_dir / "final_state.pt"
    start_epoch = 0
    best_validation_loss = float("inf")
    if resume and latest_path.is_file():
        start_epoch, best_validation_loss = _load_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            train_generator=train_generator,
            expected_fingerprint=fingerprint,
        )
        model.to(device)
        history = (
            pd.read_csv(history_path)
            .loc[lambda frame: frame["epoch"].astype(int) <= start_epoch]
            .to_dict("records")
            if history_path.is_file()
            else []
        )
        ww_frames = (
            [
                pd.read_csv(spectral_path).loc[
                    lambda frame: frame["epoch"].astype(int) <= start_epoch
                ]
            ]
            if spectral_path.is_file()
            else []
        )
    else:
        history = []
        ww_frames = []

    def measure(epoch: int, lrs: dict[str, float], train_time: float):
        train_loss, train_accuracy = evaluate(model, train_eval_loader, device)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        row = {
            "epoch": int(epoch),
            "primary_lr": float(lrs["primary"]),
            "auxiliary_lr": float(lrs["auxiliary"]),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "train_time_sec": float(train_time),
            "test_monitoring_only": 1,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        return row

    if start_epoch == 0 and not history:
        initial_lrs = set_learning_rates(optimizer, optimizer_name, config, 0)
        measure(0, initial_lrs, 0.0)
        ww_frames.append(_ww_snapshot(model, 0, config))
        pd.concat(ww_frames, ignore_index=True).to_csv(spectral_path, index=False)

    for epoch_index in range(start_epoch, config.epochs):
        lrs = set_learning_rates(optimizer, optimizer_name, config, epoch_index)
        model.train()
        started = time.perf_counter()
        optimization_loss_sum = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, target_a, target_b, lam = mixup_cutmix_batch(inputs, targets, config)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = (
                lam
                * F.cross_entropy(
                    logits,
                    target_a,
                    label_smoothing=config.label_smoothing,
                )
                + (1.0 - lam)
                * F.cross_entropy(
                    logits,
                    target_b,
                    label_smoothing=config.label_smoothing,
                )
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimization_loss_sum += loss.item() * targets.numel()
            seen += targets.numel()

        completed = epoch_index + 1
        row = measure(completed, lrs, time.perf_counter() - started)
        row["optimization_loss"] = optimization_loss_sum / max(1, seen)
        pd.DataFrame(history).to_csv(history_path, index=False)

        if row["validation_loss"] < best_validation_loss:
            best_validation_loss = row["validation_loss"]
            _save_checkpoint(
                best_path,
                epoch=completed,
                model=model,
                optimizer=optimizer,
                train_generator=train_generator,
                config=config,
                optimizer_name=optimizer_name,
                seed=seed,
                best_validation_loss=best_validation_loss,
                fingerprint=fingerprint,
            )

        if completed % config.ww_every == 0 or completed == config.epochs:
            ww_frames.append(_ww_snapshot(model, completed, config))
            pd.concat(ww_frames, ignore_index=True).to_csv(spectral_path, index=False)

        _save_checkpoint(
            latest_path,
            epoch=completed,
            model=model,
            optimizer=optimizer,
            train_generator=train_generator,
            config=config,
            optimizer_name=optimizer_name,
            seed=seed,
            best_validation_loss=best_validation_loss,
            fingerprint=fingerprint,
        )
        if completed % config.checkpoint_every == 0 or completed == config.epochs:
            _save_checkpoint(
                checkpoint_dir / f"epoch_{completed:03d}.pt",
                epoch=completed,
                model=model,
                optimizer=optimizer,
                train_generator=train_generator,
                config=config,
                optimizer_name=optimizer_name,
                seed=seed,
                best_validation_loss=best_validation_loss,
                fingerprint=fingerprint,
            )
        if progress:
            print(
                f"{optimizer_name:12s} seed={seed:2d} "
                f"epoch={completed:3d}/{config.epochs} "
                f"val_acc={100*row['validation_accuracy']:6.2f}% "
                f"test_acc={100*row['test_accuracy']:6.2f}% "
                f"val_loss={row['validation_loss']:.4f}"
            )

    _save_checkpoint(
        final_path,
        epoch=config.epochs,
        model=model,
        optimizer=optimizer,
        train_generator=train_generator,
        config=config,
        optimizer_name=optimizer_name,
        seed=seed,
        best_validation_loss=best_validation_loss,
        fingerprint=fingerprint,
    )
    (run_dir / "run_complete.json").write_text(
        json.dumps(
            {
                "completed": True,
                "optimizer": optimizer_name,
                "seed": int(seed),
                "epochs": config.epochs,
                "best_validation_loss": best_validation_loss,
                "fingerprint": fingerprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pd.DataFrame(history), pd.concat(ww_frames, ignore_index=True)


def summarize_final(performance: pd.DataFrame, final_epoch: int) -> pd.DataFrame:
    tcrit_n3 = 4.302652729911275
    rows = []
    final = performance[performance["epoch"] == final_epoch]
    metrics = (
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "test_loss",
        "test_accuracy",
    )
    for optimizer, group in final.groupby("optimizer"):
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            half = tcrit_n3 * std / math.sqrt(len(values))
            rows.append(
                {
                    "optimizer": optimizer,
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "std": std,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                }
            )
    return pd.DataFrame(rows)
