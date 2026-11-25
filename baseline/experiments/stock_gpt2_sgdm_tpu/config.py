"""Immutable, CPU-importable configuration for the plain-SGDM TPU experiment.

Schedule indices identify the *next* optimizer update, starting at zero. Every
executed update has positive LR; ``schedule_factor(max_steps)`` is the terminal
zero, never an invitation to keep training with zero LR.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


FULL_BUDGET_TOKENS = 10_255_073_280
BENCHMARK_VALIDATION_TOKENS = 10_485_760
TUNING_VALIDATION_TOKENS = 1_048_576
TUNING_VALIDATION_OFFSET = BENCHMARK_VALIDATION_TOKENS
MODEL_SOURCE_SHA256 = "757d0cea0d48cbc4c7d7d70371f955d49cdf3a7cfb4c87701a720c2fe0905c34"
ARCHITECTURE = "gpt2-small-upstream-packed-v2"
DATASET = "HuggingFaceFW/fineweb:sample-10BT"
TARGET_VAL_NLL = 3.28


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def scientific_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _integer(name: str, value: Any, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(name: str, value: Any, *, minimum: float = 0.0, strict: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (strict and value <= minimum) or (not strict and value < minimum):
        op = ">" if strict else ">="
        raise ValueError(f"{name} must be {op} {minimum}")


@dataclass(frozen=True)
class TrainingConfig:
    schema_version: int = 1
    run_name: str = "unselected-candidate"
    candidate_id: str = "unselected"
    stage_id: str = "candidate"
    mode: str = "production"
    architecture: str = ARCHITECTURE
    model_source_sha256: str = MODEL_SOURCE_SHA256
    dataset: str = DATASET
    tokenizer: str = "gpt2"
    seed: int = 42
    data_seed: int = 42
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50_257
    context_length: int = 1_024
    global_batch_sequences: int = 16
    microbatch_sequences: int = 16
    budget_tokens: int = FULL_BUDGET_TOKENS
    peak_lr: float = 0.03
    momentum: float = 0.98
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    weight_decay_scope: str = "matrix_only"
    grad_clip: float = 1.0
    schedule: str = "cosine"
    warmup_fraction: float = 0.1
    stable_fraction: float = 0.8
    eval_interval_tokens: int = 131_072_000
    spectral_interval_tokens: int = 131_072_000
    checkpoint_interval_tokens: int = 131_072_000
    validation_tokens: int = TUNING_VALIDATION_TOKENS
    validation_offset: int = TUNING_VALIDATION_OFFSET
    validation_kind: str = "tuning"
    target_val_nll: float = TARGET_VAL_NLL
    stop_at_target: bool = False
    max_pending_snapshots: int = 4
    include_embeddings: bool = True
    keep_snapshots: bool = False
    proposal_status: str = "proposed_unverified"

    def __post_init__(self) -> None:
        self.validate()

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch_sequences * self.context_length

    @property
    def max_steps(self) -> int:
        return self.budget_tokens // self.tokens_per_step

    @property
    def accumulation_steps(self) -> int:
        return self.global_batch_sequences // self.microbatch_sequences

    @property
    def warmup_steps(self) -> int:
        if self.warmup_fraction == 0:
            return 0
        return min(self.max_steps - 1, max(1, round(self.warmup_fraction * self.max_steps)))

    @property
    def config_sha256(self) -> str:
        return scientific_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported configuration schema_version")
        for name in ("run_name", "candidate_id", "stage_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in value):
                raise ValueError(f"{name} must be a nonempty filename-safe identifier")
        if self.mode not in {"smoke", "production"}:
            raise ValueError("mode must be smoke or production")
        if self.architecture != ARCHITECTURE or self.model_source_sha256 != MODEL_SOURCE_SHA256:
            raise ValueError("Architecture and upstream model source hash are pinned")
        if self.dataset != DATASET or self.tokenizer != "gpt2":
            raise ValueError("Production protocol uses pinned FineWeb and GPT-2 tokenization")
        for name in ("seed", "data_seed", "validation_offset"):
            _integer(name, getattr(self, name), 0)
        for name in ("n_layer", "n_head", "n_embd", "vocab_size", "context_length", "global_batch_sequences", "microbatch_sequences", "budget_tokens", "validation_tokens", "eval_interval_tokens", "spectral_interval_tokens", "checkpoint_interval_tokens", "max_pending_snapshots"):
            _integer(name, getattr(self, name))
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.global_batch_sequences % self.microbatch_sequences:
            raise ValueError("microbatch_sequences must divide global_batch_sequences")
        if self.budget_tokens % self.tokens_per_step or self.max_steps < 2:
            raise ValueError("budget_tokens must contain at least two complete optimizer updates")
        for name in ("eval_interval_tokens", "spectral_interval_tokens", "checkpoint_interval_tokens"):
            if getattr(self, name) % self.tokens_per_step:
                raise ValueError(f"{name} must be a multiple of tokens_per_step")
        if self.spectral_interval_tokens % self.checkpoint_interval_tokens:
            raise ValueError("Spectral intervals must align with checkpoint intervals")
        if self.validation_tokens % (self.microbatch_sequences * self.context_length):
            raise ValueError("validation_tokens must contain complete evaluation microbatches")
        if self.validation_offset % self.context_length:
            raise ValueError("validation_offset must align with context_length")
        _finite("peak_lr", self.peak_lr, strict=True)
        _finite("momentum", self.momentum)
        if self.momentum >= 1:
            raise ValueError("momentum must be in [0, 1)")
        if self.dampening != 0 or self.nesterov is not False:
            raise ValueError("This experiment pins plain SGDM: dampening=0, nesterov=False")
        _finite("weight_decay", self.weight_decay)
        _finite("grad_clip", self.grad_clip, strict=True)
        if self.weight_decay_scope != "matrix_only":
            raise ValueError("weight_decay_scope must be matrix_only (ndim >= 2)")
        if self.schedule not in {"cosine", "wsd"}:
            raise ValueError("schedule must be cosine or wsd")
        _finite("warmup_fraction", self.warmup_fraction)
        _finite("stable_fraction", self.stable_fraction)
        if self.warmup_fraction >= 1 or self.stable_fraction >= 1:
            raise ValueError("Warmup and stable fractions must be less than 1")
        if self.schedule == "wsd" and self.warmup_fraction + self.stable_fraction >= 1:
            raise ValueError("WSD requires an explicit positive final decay fraction")
        if self.target_val_nll != TARGET_VAL_NLL:
            raise ValueError("The official validation target is fixed at 3.28 nats/token")
        if self.validation_kind not in {"tuning", "benchmark"}:
            raise ValueError("validation_kind must be tuning or benchmark")
        for name in ("stop_at_target", "include_embeddings", "keep_snapshots"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.proposal_status not in {"proposed_unverified", "historical_hyperparameter_control", "smoke_only", "selected_from_search"}:
            raise ValueError("Unknown proposal_status")
        if self.mode == "production":
            if (self.n_layer, self.n_head, self.n_embd, self.vocab_size, self.context_length) != (12, 12, 768, 50_257, 1_024):
                raise ValueError("Production requires stock GPT-2 124M dimensions and context1024")
            if self.validation_kind == "benchmark":
                if (self.validation_offset, self.validation_tokens) != (0, BENCHMARK_VALIDATION_TOKENS):
                    raise ValueError("Benchmark evaluation must use the exact official prefix and token count")
            elif (self.validation_offset, self.validation_tokens) != (TUNING_VALIDATION_OFFSET, TUNING_VALIDATION_TOKENS):
                raise ValueError("Tuning must use the disjoint reserved validation window")
        elif self.validation_kind != "tuning" or self.stop_at_target:
            raise ValueError("Smoke runs cannot qualify for or stop at the benchmark target")
        if self.stop_at_target and self.validation_kind != "benchmark":
            raise ValueError("Only a full official benchmark evaluation can stop at target")

    def schedule_factor(self, step: int) -> float:
        """Multiplier for a zero-based update; terminal value is exactly zero."""
        _integer("step", step, 0)
        if step > self.max_steps:
            raise ValueError("Step exceeds the immutable training horizon; create a fresh config")
        if step == self.max_steps:
            return 0.0
        warmup = self.warmup_steps
        if step < warmup:
            return (step + 1) / warmup
        decay_start = warmup
        if self.schedule == "wsd":
            decay_start = min(self.max_steps - 1, warmup + round(self.stable_fraction * self.max_steps))
            if step < decay_start:
                return 1.0
        progress = (step - decay_start) / (self.max_steps - decay_start)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, require_hash: bool = False) -> "TrainingConfig":
        payload = dict(value)
        supplied_hash = payload.pop("config_sha256", None)
        allowed = {item.name for item in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        config = cls(**payload)
        if require_hash and supplied_hash is None:
            raise ValueError("Configuration JSON must include its immutable config_sha256")
        if supplied_hash is not None and supplied_hash != config.config_sha256:
            raise ValueError("Configuration hash mismatch: scientific settings were modified")
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        return cls.from_dict(json.loads(Path(path).read_text()), require_hash=True)

    def write_json(self, path: str | Path) -> Path:
        """Create once; an identical existing config is allowed, replacement is not."""
        path = Path(path)
        payload = {**self.as_dict(), "config_sha256": self.config_sha256}
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = TrainingConfig.from_json(path)
            if existing.config_sha256 != self.config_sha256:
                raise FileExistsError(f"Refusing to replace immutable configuration: {path}")
            return path
        with path.open("x") as handle:
            handle.write(text)
        return path

    def at_budget(self, budget_tokens: int, *, stage_id: str, validation_kind: str = "tuning", **changes: Any) -> "TrainingConfig":
        """Create a new, fresh-initialization horizon; never a resume operation."""
        if validation_kind == "benchmark":
            evaluation = dict(validation_tokens=BENCHMARK_VALIDATION_TOKENS, validation_offset=0)
        else:
            evaluation = dict(validation_tokens=TUNING_VALIDATION_TOKENS, validation_offset=TUNING_VALIDATION_OFFSET)
        return replace(self, budget_tokens=budget_tokens, stage_id=stage_id,
                       run_name=f"{self.candidate_id}-{stage_id}-seed{self.seed}",
                       validation_kind=validation_kind, **evaluation, **changes)


def smoke_config() -> TrainingConfig:
    return TrainingConfig(run_name="cpu-smoke", candidate_id="smoke", stage_id="smoke", mode="smoke",
                          n_layer=2, n_head=2, n_embd=32, vocab_size=256, context_length=32,
                          global_batch_sequences=2, microbatch_sequences=2, budget_tokens=128,
                          eval_interval_tokens=128, spectral_interval_tokens=128,
                          checkpoint_interval_tokens=128, validation_tokens=128,
                          validation_offset=0, proposal_status="smoke_only")
