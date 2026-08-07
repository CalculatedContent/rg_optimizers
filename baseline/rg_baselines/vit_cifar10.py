"""Small Vision Transformer CIFAR-10 optimizer baselines.

This module provides a reproducible from-scratch CIFAR-10 benchmark for
SGD+Nesterov momentum, AdamW, and Muon with auxiliary AdamW.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import random
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .muon import zeropower_via_newton_schulz_5

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_VIT_SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class ViTBaselineConfig:
    epochs: int = 120
    batch_size: int = 128
    warmup_epochs: int = 5
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    num_classes: int = 10
    mixup_alpha: float = 0.2
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    ww_every: int = 10
    checkpoint_every: int = 10

    # SGD + Nesterov momentum
    sgd_lr: float = 0.10
    sgd_momentum: float = 0.9
    sgd_weight_decay: float = 5e-4

    # AdamW
    adamw_lr: float = 5e-4
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.05

    # Muon hidden matrices + auxiliary AdamW
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    muon_ns_steps: int = 5
    muon_aux_lr: float = 3e-4
    muon_aux_beta1: float = 0.9
    muon_aux_beta2: float = 0.95
    muon_aux_eps: float = 1e-8
    muon_aux_weight_decay: float = 0.01

    def validate(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.warmup_epochs < 0 or self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must satisfy 0 <= warmup_epochs < epochs")


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
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        a = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        a = self.attn_drop(a)
        x = (a @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float,
                 dropout: float, attn_dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SmallViT(nn.Module):
    def __init__(self, config: ViTBaselineConfig) -> None:
        super().__init__()
        config.validate()
        self.patch_embed = PatchEmbed(config.image_size, config.patch_size, config.embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            Block(config.embed_dim, config.num_heads, config.mlp_ratio,
                  config.dropout, config.attn_dropout)
            for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
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


def make_datasets(data_dir: Path):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    train_eval = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_transform)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_transform)
    return train, train_eval, test


def make_loaders(data_dir: Path, config: ViTBaselineConfig, seed: int, device: torch.device):
    train, train_eval, test = make_datasets(data_dir)
    workers = 0 if device.type == "mps" else 2
    gen = torch.Generator().manual_seed(seed)
    common = dict(num_workers=workers, pin_memory=device.type == "cuda")
    return (
        DataLoader(train, config.batch_size, shuffle=True, generator=gen, **common),
        DataLoader(train_eval, 256, shuffle=False, **common),
        DataLoader(test, 256, shuffle=False, **common),
    )


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


def cosine_multiplier(epoch: int, config: ViTBaselineConfig) -> float:
    if epoch < config.warmup_epochs:
        return float(epoch + 1) / float(max(1, config.warmup_epochs))
    progress = (epoch - config.warmup_epochs) / float(
        max(1, config.epochs - config.warmup_epochs - 1)
    )
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y).item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        count += y.numel()
    return loss_sum / count, correct / count


def muon_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name for name, p in model.named_parameters()
        if p.requires_grad and p.ndim == 2 and name.startswith("blocks.") and name.endswith("weight")
    )


class MuonWithAuxAdamW:
    """Muon on hidden transformer matrices and AdamW on all remaining parameters."""
    def __init__(self, model: nn.Module, config: ViTBaselineConfig) -> None:
        named = list(model.named_parameters())
        selected = set(muon_parameter_names(model))
        self.muon_params = [p for n, p in named if n in selected]
        aux_params = [p for n, p in named if n not in selected and p.requires_grad]
        self.muon = torch.optim.SGD(
            self.muon_params, lr=config.muon_lr, momentum=config.muon_momentum,
            nesterov=True, weight_decay=0.0,
        )
        self.aux = torch.optim.AdamW(
            aux_params, lr=config.muon_aux_lr,
            betas=(config.muon_aux_beta1, config.muon_aux_beta2),
            eps=config.muon_aux_eps, weight_decay=config.muon_aux_weight_decay,
        )
        self.config = config

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.aux.zero_grad(set_to_none=set_to_none)

    def set_multiplier(self, m: float) -> None:
        self.muon.param_groups[0]["lr"] = self.config.muon_lr * m
        for group in self.aux.param_groups:
            group["lr"] = self.config.muon_aux_lr * m

    @torch.no_grad()
    def step(self) -> None:
        group = self.muon.param_groups[0]
        lr = float(group["lr"])
        momentum = float(group["momentum"])
        for p in self.muon_params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            state = self.muon.state[p]
            buf = state.get("momentum_buffer")
            if buf is None:
                buf = torch.zeros_like(grad)
                state["momentum_buffer"] = buf
            buf.lerp_(grad, 1.0 - momentum)
            update = grad.lerp(buf, momentum)
            update = zeropower_via_newton_schulz_5(update, steps=self.config.muon_ns_steps)
            if self.config.muon_weight_decay:
                p.mul_(1.0 - lr * self.config.muon_weight_decay)
            shape_scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
            p.add_(update, alpha=-lr * shape_scale)
        self.aux.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "aux": self.aux.state_dict()}


def build_optimizer(model: nn.Module, name: str, config: ViTBaselineConfig):
    if name == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(), lr=config.sgd_lr, momentum=config.sgd_momentum,
            weight_decay=config.sgd_weight_decay, nesterov=True,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.adamw_lr,
            betas=(config.adamw_beta1, config.adamw_beta2), eps=config.adamw_eps,
            weight_decay=config.adamw_weight_decay,
        )
    if name == "muon":
        return MuonWithAuxAdamW(model, config)
    raise ValueError(f"unknown optimizer {name!r}")


def set_lr_multiplier(optimizer, name: str, config: ViTBaselineConfig, m: float) -> None:
    if name == "muon":
        optimizer.set_multiplier(m)
        return
    peak = config.sgd_lr if name == "sgd_momentum" else config.adamw_lr
    for group in optimizer.param_groups:
        group["lr"] = peak * m


def _ww_snapshot(model: nn.Module, epoch: int):
    import weightwatcher as ww
    details = ww.WeightWatcher(model=model).analyze(ERG=True, randomize=True)
    keep = [c for c in (
        "layer_id", "name", "longname", "alpha", "num_traps", "detX_num",
        "num_pl_spikes", "ERG_gap", "erg_gap", "lambda_max", "spectral_norm",
    ) if c in details.columns]
    out = details[keep].copy()
    out.insert(0, "epoch", epoch)
    if "ERG_gap" not in out and "erg_gap" in out:
        out["ERG_gap"] = out["erg_gap"]
    return out


def run_vit_baseline(
    optimizer_name: str,
    seed: int,
    *,
    data_dir: Path,
    output_dir: Path,
    config: ViTBaselineConfig = ViTBaselineConfig(),
    device: torch.device | None = None,
    progress: bool = True,
):
    config.validate()
    device = device or choose_device()
    set_seed(seed)
    train_loader, train_eval_loader, test_loader = make_loaders(data_dir, config, seed, device)
    model = SmallViT(config).to(device)
    optimizer = build_optimizer(model, optimizer_name, config)
    run_dir = Path(output_dir) / optimizer_name / f"seed_{seed}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as handle:
        json.dump({"optimizer": optimizer_name, "seed": seed, **asdict(config)}, handle, indent=2)

    history: list[dict] = []
    ww_frames: list[pd.DataFrame] = []
    train_loss, train_acc = evaluate(model, train_eval_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    history.append(dict(epoch=0, train_loss=train_loss, train_accuracy=train_acc,
                        test_loss=test_loss, test_accuracy=test_acc, lr_multiplier=0.0,
                        train_time_sec=0.0))
    ww_frames.append(_ww_snapshot(model, 0))

    for epoch in range(config.epochs):
        m = cosine_multiplier(epoch, config)
        set_lr_multiplier(optimizer, optimizer_name, config, m)
        model.train()
        start = time.time()
        opt_loss_sum = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, ya, yb, lam = mixup_batch(x, y, config.mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = (
                lam * F.cross_entropy(logits, ya, label_smoothing=config.label_smoothing)
                + (1.0 - lam) * F.cross_entropy(logits, yb, label_smoothing=config.label_smoothing)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            opt_loss_sum += loss.item() * y.numel()
            seen += y.numel()

        completed = epoch + 1
        train_loss, train_acc = evaluate(model, train_eval_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        history.append(dict(
            epoch=completed, optimization_loss=opt_loss_sum / seen,
            train_loss=train_loss, train_accuracy=train_acc,
            test_loss=test_loss, test_accuracy=test_acc,
            lr_multiplier=m, train_time_sec=time.time() - start,
        ))
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if completed % config.ww_every == 0 or completed == config.epochs:
            ww_frames.append(_ww_snapshot(model, completed))
            pd.concat(ww_frames, ignore_index=True).to_csv(
                run_dir / "weightwatcher_by_epoch_layer.csv", index=False
            )
        if completed % config.checkpoint_every == 0 or completed == config.epochs:
            torch.save({
                "epoch": completed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
            }, ckpt_dir / f"epoch_{completed:03d}.pt")
        if progress:
            print(
                f"{optimizer_name:12s} seed={seed:2d} epoch={completed:3d}/{config.epochs} "
                f"test_acc={100*test_acc:6.2f}% test_loss={test_loss:.4f} "
                f"train_acc={100*train_acc:6.2f}%"
            )

    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config)}, run_dir / "final_state.pt")
    return pd.DataFrame(history), pd.concat(ww_frames, ignore_index=True)


def summarize_final(performance: pd.DataFrame, final_epoch: int) -> pd.DataFrame:
    tcrit_n3 = 4.302652729911275
    rows = []
    final = performance[performance["epoch"] == final_epoch]
    for optimizer, group in final.groupby("optimizer"):
        for metric in ("train_loss", "train_accuracy", "test_loss", "test_accuracy"):
            values = group[metric].astype(float).to_numpy()
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            half = tcrit_n3 * std / math.sqrt(len(values))
            rows.append(dict(optimizer=optimizer, metric=metric, n=len(values), mean=mean,
                             std=std, ci95_low=mean-half, ci95_high=mean+half))
    return pd.DataFrame(rows)
