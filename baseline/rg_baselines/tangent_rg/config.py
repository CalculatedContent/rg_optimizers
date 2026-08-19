"""Configuration objects for the long-horizon MNIST tangent-RG experiment.

The established :mod:`rg_baselines` MNIST recipe is intentionally left
unchanged.  This module defines a separate protocol whose training horizon may
be thousands of epochs while its learning-rate schedule ends much earlier and
then remains at the declared floor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Optional


OptimizerName = Literal["adamw", "muon", "muonclip_rms"]
SUPPORTED_OPTIMIZERS: tuple[str, ...] = (
    "adamw",
    "muon",
    "muonclip_rms",
)


@dataclass(frozen=True)
class AdamWProfile:
    """AdamW control matching the audited MLP3 baseline."""

    learning_rate: float = 1.0e-3
    min_learning_rate: float = 1.0e-5
    warmup_epochs: float = 1.0
    beta1: float = 0.90
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    weight_decay: float = 1.0e-2


@dataclass(frozen=True)
class MuonProfile:
    """Ordinary Muon plus auxiliary AdamW control."""

    matrix_learning_rate: float = 2.0e-2
    matrix_min_learning_rate: float = 2.0e-3
    warmup_epochs: float = 2.0
    momentum: float = 0.95
    nesterov: bool = True
    weight_decay: float = 1.0e-2
    newton_schulz_steps: int = 5
    epsilon: float = 1.0e-7
    auxiliary_learning_rate: float = 3.0e-4
    auxiliary_min_learning_rate: float = 3.0e-5
    auxiliary_beta1: float = 0.90
    auxiliary_beta2: float = 0.95
    auxiliary_epsilon: float = 1.0e-8
    auxiliary_weight_decay: float = 1.0e-2
    parameter_names: tuple[str, ...] = ("fc1.weight", "fc2.weight")


@dataclass(frozen=True)
class MuonClipRMSProfile:
    """MLP-specific MuonClip matrix rule.

    There are no query/key logits in an MLP, so QK clipping is not applicable.
    The retained MuonClip operation is Newton--Schulz polar orthogonalization
    followed by exact RMS matching.  ``rms_scale=0.20`` is the declared
    Kimi-style matrix-update RMS used by this experiment; it is a protocol
    parameter, not an inferred or silently tuned value.
    """

    matrix_learning_rate: float = 2.0e-4
    matrix_min_learning_rate: float = 2.0e-5
    warmup_epochs: float = 1.0
    momentum: float = 0.95
    nesterov: bool = False
    weight_decay: float = 1.0e-1
    newton_schulz_steps: int = 5
    epsilon: float = 1.0e-7
    rms_scale: float = 0.20
    auxiliary_learning_rate: float = 2.0e-4
    auxiliary_min_learning_rate: float = 2.0e-5
    auxiliary_beta1: float = 0.90
    auxiliary_beta2: float = 0.95
    auxiliary_epsilon: float = 1.0e-8
    auxiliary_weight_decay: float = 1.0e-1
    parameter_names: tuple[str, ...] = ("fc1.weight", "fc2.weight")
    qk_clipping_applicable: bool = False


@dataclass(frozen=True)
class TangentRGConfig:
    """Resolved configuration for one optimizer and one independent seed."""

    suite_name: str = "mnist_mlp3_tangent_rg_v1"
    schema_version: int = 1
    description: str = "Long-horizon MLP3 tangent-RG baseline"

    optimizer: OptimizerName = "muon"
    seed: int = 1337
    epochs: int = 1_000
    lr_schedule_epochs: float = 30.0
    batch_size: int = 128
    validation_size: int = 5_000
    split_seed: int = 20_260_807
    num_workers: int = 0
    grad_clip_norm: float = 1.0
    train_eval_max_batches: Optional[int] = None
    validation_every_epochs: int = 5
    latest_every_epochs: int = 1
    test_monitoring_only: bool = True

    log_analysis_points: int = 96
    explicit_analysis_epochs: tuple[int, ...] = (0, 1, 2, 5, 10, 30)
    tail_analysis_start_epoch: Optional[int] = None
    tail_analysis_every_epochs: Optional[int] = None
    dense_burst_anchor_epochs: tuple[int, ...] = (0, 1, 10, 100, 999)
    dense_burst_length_steps: int = 8
    capture_parameter_names: tuple[str, ...] = (
        "fc1.weight",
        "fc2.weight",
    )
    weightwatcher_enabled: bool = True
    weightwatcher_required: bool = True
    weightwatcher_min_evals: int = 8
    weightwatcher_max_evals: Optional[int] = None
    weightwatcher_max_fingers: int = 10
    weightwatcher_svd_method: str = "accurate"
    weightwatcher_randomize: bool = True
    weightwatcher_primary_variant: str = "clip_xmax"
    weightwatcher_analysis_seed_offset: int = 904_271

    device: str = "auto"
    data_dir: str = "./data"
    run_root: str = "./runs"
    tail_checkpoint_cache_root: str = "/tmp/rg-mnist-mlp3-tangent-checkpoints"

    adamw: AdamWProfile = AdamWProfile()
    muon: MuonProfile = MuonProfile()
    muonclip_rms: MuonClipRMSProfile = MuonClipRMSProfile()

    def validate(self) -> None:
        if self.optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {SUPPORTED_OPTIMIZERS}, "
                f"got {self.optimizer!r}"
            )
        if self.schema_version < 1 or not self.suite_name.strip():
            raise ValueError("schema_version and suite_name must be valid")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.suite_name) is None:
            raise ValueError(
                "suite_name must be one safe path-component slug containing only "
                "letters, digits, '.', '_', and '-'"
            )
        if self.seed < 0 or self.split_seed < 0:
            raise ValueError("seed values must be non-negative")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if not 0.0 < self.lr_schedule_epochs <= float(self.epochs):
            raise ValueError(
                "lr_schedule_epochs must be positive and no greater than epochs"
            )
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and workers non-negative")
        if not 0 < self.validation_size < 60_000:
            raise ValueError("validation_size must lie in (0, 60000)")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if self.train_eval_max_batches is not None and self.train_eval_max_batches < 1:
            raise ValueError("train_eval_max_batches must be positive or None")
        if self.validation_every_epochs < 1 or self.latest_every_epochs < 1:
            raise ValueError("validation/latest checkpoint cadences must be positive")
        if not self.test_monitoring_only:
            raise ValueError("the official MNIST test split must be monitoring-only")
        if self.log_analysis_points < 2:
            raise ValueError("log_analysis_points must be at least two")
        if any(epoch < 0 for epoch in self.explicit_analysis_epochs):
            raise ValueError("explicit analysis epochs must be non-negative")
        if any(epoch < 0 for epoch in self.dense_burst_anchor_epochs):
            raise ValueError("dense burst anchors must be non-negative")
        if self.dense_burst_length_steps < 1:
            raise ValueError("dense_burst_length_steps must be positive")
        if not self.capture_parameter_names or len(set(self.capture_parameter_names)) != len(
            self.capture_parameter_names
        ):
            raise ValueError("capture_parameter_names must be non-empty and unique")
        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.capture_parameter_names
        ):
            raise ValueError("capture_parameter_names must contain non-empty strings")
        if (self.tail_analysis_start_epoch is None) != (
            self.tail_analysis_every_epochs is None
        ):
            raise ValueError("tail analysis start and cadence must be supplied together")
        if self.tail_analysis_start_epoch is not None:
            if (
                self.tail_analysis_start_epoch < 0
                or self.tail_analysis_start_epoch > self.epochs
                or self.tail_analysis_every_epochs < 1
            ):
                raise ValueError("tail analysis settings are invalid")
        if self.weightwatcher_required and not self.weightwatcher_enabled:
            raise ValueError("required WeightWatcher analysis cannot be disabled")
        if self.weightwatcher_min_evals < 2:
            raise ValueError("weightwatcher_min_evals must be at least two")
        if (
            self.weightwatcher_max_evals is not None
            and self.weightwatcher_max_evals < self.weightwatcher_min_evals
        ):
            raise ValueError("weightwatcher_max_evals must be >= min_evals or None")
        if self.weightwatcher_max_fingers < 0:
            raise ValueError("weightwatcher_max_fingers must be non-negative")
        if not self.weightwatcher_svd_method.strip():
            raise ValueError("weightwatcher_svd_method must not be empty")
        if not self.weightwatcher_randomize:
            raise ValueError("randomize=True is required for ERG correlation traps")
        if self.weightwatcher_primary_variant != "clip_xmax":
            raise ValueError("the preregistered primary WeightWatcher fit is clip_xmax")
        if self.weightwatcher_analysis_seed_offset < 0:
            raise ValueError("WeightWatcher analysis seed offset must be non-negative")
        cache_root = Path(self.tail_checkpoint_cache_root).expanduser()
        tmp_root = Path("/tmp").resolve()
        if not self.tail_checkpoint_cache_root.strip() or not cache_root.is_absolute():
            raise ValueError("tail_checkpoint_cache_root must be an absolute /tmp path")
        try:
            cache_root.resolve().relative_to(tmp_root)
        except ValueError as error:
            raise ValueError(
                "tail_checkpoint_cache_root must resolve beneath /tmp"
            ) from error
        if cache_root.resolve() == tmp_root:
            raise ValueError("tail_checkpoint_cache_root must not be /tmp itself")

        _validate_adamw(self.adamw, self.lr_schedule_epochs)
        _validate_muon(self.muon, self.lr_schedule_epochs)
        _validate_muonclip(self.muonclip_rms, self.lr_schedule_epochs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(
        self,
        *,
        optimizer: Optional[str] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        data_dir: Optional[str] = None,
        run_root: Optional[str] = None,
        tail_checkpoint_cache_root: Optional[str] = None,
    ) -> "TangentRGConfig":
        updated = replace(
            self,
            optimizer=self.optimizer if optimizer is None else optimizer,
            seed=self.seed if seed is None else int(seed),
            device=self.device if device is None else str(device),
            data_dir=self.data_dir if data_dir is None else str(data_dir),
            run_root=self.run_root if run_root is None else str(run_root),
            tail_checkpoint_cache_root=(
                self.tail_checkpoint_cache_root
                if tail_checkpoint_cache_root is None
                else str(tail_checkpoint_cache_root)
            ),
        )
        updated.validate()
        return updated

    @property
    def optimizer_label(self) -> str:
        return {
            "adamw": "AdamW",
            "muon": "Muon + auxiliary AdamW",
            "muonclip_rms": "MuonClip-RMS + auxiliary AdamW (QK N/A)",
        }[self.optimizer]


def _validate_lr_pair(peak: float, floor: float, name: str) -> None:
    if peak <= 0.0 or floor < 0.0 or floor > peak:
        raise ValueError(f"{name} peak/floor learning rates are inconsistent")


def _validate_parameter_names(names: tuple[str, ...], label: str) -> None:
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{label} parameter_names must be non-empty and unique")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} parameter_names contain an empty/non-string value")


def _validate_betas(beta1: float, beta2: float, label: str) -> None:
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError(f"{label} beta values must lie in [0, 1)")


def _validate_adamw(profile: AdamWProfile, schedule_epochs: float) -> None:
    _validate_lr_pair(profile.learning_rate, profile.min_learning_rate, "AdamW")
    if not 0.0 <= profile.warmup_epochs < schedule_epochs:
        raise ValueError("AdamW warmup must be inside the LR schedule horizon")
    _validate_betas(profile.beta1, profile.beta2, "AdamW")
    if profile.epsilon <= 0.0 or profile.weight_decay < 0.0:
        raise ValueError("AdamW epsilon/weight decay are invalid")


def _validate_muon(profile: MuonProfile, schedule_epochs: float) -> None:
    _validate_lr_pair(
        profile.matrix_learning_rate,
        profile.matrix_min_learning_rate,
        "Muon matrix",
    )
    _validate_lr_pair(
        profile.auxiliary_learning_rate,
        profile.auxiliary_min_learning_rate,
        "Muon auxiliary",
    )
    if not 0.0 <= profile.warmup_epochs < schedule_epochs:
        raise ValueError("Muon warmup must be inside the LR schedule horizon")
    if not 0.0 <= profile.momentum < 1.0:
        raise ValueError("Muon momentum must lie in [0, 1)")
    if profile.nesterov and profile.momentum <= 0.0:
        raise ValueError("Nesterov Muon requires positive momentum")
    if profile.newton_schulz_steps < 1 or profile.epsilon <= 0.0:
        raise ValueError("Muon Newton--Schulz settings are invalid")
    if profile.weight_decay < 0.0 or profile.auxiliary_weight_decay < 0.0:
        raise ValueError("Muon weight decay must be non-negative")
    if profile.auxiliary_epsilon <= 0.0:
        raise ValueError("Muon auxiliary epsilon must be positive")
    _validate_betas(profile.auxiliary_beta1, profile.auxiliary_beta2, "Muon auxiliary")
    _validate_parameter_names(profile.parameter_names, "Muon")


def _validate_muonclip(profile: MuonClipRMSProfile, schedule_epochs: float) -> None:
    _validate_lr_pair(
        profile.matrix_learning_rate,
        profile.matrix_min_learning_rate,
        "MuonClip-RMS matrix",
    )
    _validate_lr_pair(
        profile.auxiliary_learning_rate,
        profile.auxiliary_min_learning_rate,
        "MuonClip-RMS auxiliary",
    )
    if not 0.0 <= profile.warmup_epochs < schedule_epochs:
        raise ValueError("MuonClip-RMS warmup must be inside the LR schedule horizon")
    if not 0.0 <= profile.momentum < 1.0:
        raise ValueError("MuonClip-RMS momentum must lie in [0, 1)")
    if profile.nesterov and profile.momentum <= 0.0:
        raise ValueError("Nesterov MuonClip-RMS requires positive momentum")
    if profile.newton_schulz_steps < 1 or profile.epsilon <= 0.0:
        raise ValueError("MuonClip-RMS Newton--Schulz settings are invalid")
    if profile.rms_scale <= 0.0:
        raise ValueError("MuonClip-RMS rms_scale must be positive")
    if profile.weight_decay < 0.0 or profile.auxiliary_weight_decay < 0.0:
        raise ValueError("MuonClip-RMS weight decay must be non-negative")
    if profile.auxiliary_epsilon <= 0.0:
        raise ValueError("MuonClip-RMS auxiliary epsilon must be positive")
    _validate_betas(
        profile.auxiliary_beta1,
        profile.auxiliary_beta2,
        "MuonClip-RMS auxiliary",
    )
    _validate_parameter_names(profile.parameter_names, "MuonClip-RMS")
    if profile.qk_clipping_applicable:
        raise ValueError("QK clipping is not applicable to the MLP3 experiment")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _reject_unknown_keys(
    values: Mapping[str, Any],
    *,
    allowed: set[str],
    section: str,
) -> None:
    """Reject configuration typos instead of silently using a default."""

    unknown = sorted((key for key in values if key not in allowed), key=str)
    if unknown:
        rendered = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"unknown key(s) in {section}: {rendered}")


def _tuple_ints(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(int(item) for item in value)


def _tuple_strings(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    return tuple(str(item) for item in value)


def config_from_mapping(payload: Mapping[str, Any]) -> TangentRGConfig:
    """Build a strict configuration from the experiment YAML/JSON schema."""

    _reject_unknown_keys(
        payload,
        allowed={"protocol", "training", "analysis", "runtime", "optimizers"},
        section="configuration root",
    )
    defaults = TangentRGConfig()
    protocol = _mapping(payload.get("protocol"), "protocol")
    training = _mapping(payload.get("training"), "training")
    analysis = _mapping(payload.get("analysis"), "analysis")
    runtime = _mapping(payload.get("runtime"), "runtime")
    profiles = _mapping(payload.get("optimizers"), "optimizers")

    _reject_unknown_keys(
        protocol,
        allowed={"suite_name", "schema_version", "description"},
        section="protocol",
    )
    _reject_unknown_keys(
        training,
        allowed={
            "optimizer",
            "seed",
            "epochs",
            "lr_schedule_epochs",
            "batch_size",
            "validation_size",
            "split_seed",
            "num_workers",
            "grad_clip_norm",
            "train_eval_max_batches",
            "validation_every_epochs",
            "latest_every_epochs",
            "test_monitoring_only",
        },
        section="training",
    )
    _reject_unknown_keys(
        analysis,
        allowed={
            "log_points",
            "explicit_epochs",
            "tail_start_epoch",
            "tail_every_epochs",
            "dense_burst_anchor_epochs",
            "dense_burst_length_steps",
            "capture_parameter_names",
            "weightwatcher_enabled",
            "weightwatcher_required",
            "weightwatcher_min_evals",
            "weightwatcher_max_evals",
            "weightwatcher_max_fingers",
            "weightwatcher_svd_method",
            "weightwatcher_randomize",
            "weightwatcher_primary_variant",
            "weightwatcher_analysis_seed_offset",
        },
        section="analysis",
    )
    _reject_unknown_keys(
        runtime,
        allowed={
            "device",
            "data_dir",
            "run_root",
            "tail_checkpoint_cache_root",
        },
        section="runtime",
    )
    _reject_unknown_keys(
        profiles,
        allowed={"adamw", "muon", "muonclip_rms"},
        section="optimizers",
    )

    adamw_values = dict(_mapping(profiles.get("adamw"), "optimizers.adamw"))
    _reject_unknown_keys(
        adamw_values,
        allowed={field.name for field in fields(AdamWProfile)},
        section="optimizers.adamw",
    )
    adamw = AdamWProfile(**adamw_values)
    muon_values = dict(_mapping(profiles.get("muon"), "optimizers.muon"))
    _reject_unknown_keys(
        muon_values,
        allowed={field.name for field in fields(MuonProfile)},
        section="optimizers.muon",
    )
    if "parameter_names" in muon_values:
        muon_values["parameter_names"] = tuple(muon_values["parameter_names"])
    muon = MuonProfile(**muon_values)
    clip_values = dict(
        _mapping(profiles.get("muonclip_rms"), "optimizers.muonclip_rms")
    )
    _reject_unknown_keys(
        clip_values,
        allowed={field.name for field in fields(MuonClipRMSProfile)},
        section="optimizers.muonclip_rms",
    )
    if "parameter_names" in clip_values:
        clip_values["parameter_names"] = tuple(clip_values["parameter_names"])
    muonclip = MuonClipRMSProfile(**clip_values)

    config = TangentRGConfig(
        suite_name=str(protocol.get("suite_name", defaults.suite_name)),
        schema_version=int(protocol.get("schema_version", defaults.schema_version)),
        description=str(protocol.get("description", defaults.description)),
        optimizer=str(training.get("optimizer", defaults.optimizer)),
        seed=int(training.get("seed", defaults.seed)),
        epochs=int(training.get("epochs", defaults.epochs)),
        lr_schedule_epochs=float(
            training.get("lr_schedule_epochs", defaults.lr_schedule_epochs)
        ),
        batch_size=int(training.get("batch_size", defaults.batch_size)),
        validation_size=int(
            training.get("validation_size", defaults.validation_size)
        ),
        split_seed=int(training.get("split_seed", defaults.split_seed)),
        num_workers=int(training.get("num_workers", defaults.num_workers)),
        grad_clip_norm=float(
            training.get("grad_clip_norm", defaults.grad_clip_norm)
        ),
        train_eval_max_batches=(
            None
            if training.get("train_eval_max_batches") is None
            else int(training["train_eval_max_batches"])
        ),
        validation_every_epochs=int(
            training.get(
                "validation_every_epochs", defaults.validation_every_epochs
            )
        ),
        latest_every_epochs=int(
            training.get("latest_every_epochs", defaults.latest_every_epochs)
        ),
        test_monitoring_only=bool(
            training.get("test_monitoring_only", defaults.test_monitoring_only)
        ),
        log_analysis_points=int(
            analysis.get("log_points", defaults.log_analysis_points)
        ),
        explicit_analysis_epochs=_tuple_ints(
            analysis.get("explicit_epochs"), defaults.explicit_analysis_epochs
        ),
        tail_analysis_start_epoch=(
            None
            if analysis.get("tail_start_epoch") is None
            else int(analysis["tail_start_epoch"])
        ),
        tail_analysis_every_epochs=(
            None
            if analysis.get("tail_every_epochs") is None
            else int(analysis["tail_every_epochs"])
        ),
        dense_burst_anchor_epochs=_tuple_ints(
            analysis.get("dense_burst_anchor_epochs"),
            defaults.dense_burst_anchor_epochs,
        ),
        dense_burst_length_steps=int(
            analysis.get(
                "dense_burst_length_steps", defaults.dense_burst_length_steps
            )
        ),
        capture_parameter_names=_tuple_strings(
            analysis.get("capture_parameter_names"),
            defaults.capture_parameter_names,
        ),
        weightwatcher_enabled=bool(
            analysis.get("weightwatcher_enabled", defaults.weightwatcher_enabled)
        ),
        weightwatcher_required=bool(
            analysis.get("weightwatcher_required", defaults.weightwatcher_required)
        ),
        weightwatcher_min_evals=int(
            analysis.get("weightwatcher_min_evals", defaults.weightwatcher_min_evals)
        ),
        weightwatcher_max_evals=(
            None
            if analysis.get("weightwatcher_max_evals") is None
            else int(analysis["weightwatcher_max_evals"])
        ),
        weightwatcher_max_fingers=int(
            analysis.get(
                "weightwatcher_max_fingers", defaults.weightwatcher_max_fingers
            )
        ),
        weightwatcher_svd_method=str(
            analysis.get(
                "weightwatcher_svd_method", defaults.weightwatcher_svd_method
            )
        ),
        weightwatcher_randomize=bool(
            analysis.get(
                "weightwatcher_randomize", defaults.weightwatcher_randomize
            )
        ),
        weightwatcher_primary_variant=str(
            analysis.get(
                "weightwatcher_primary_variant",
                defaults.weightwatcher_primary_variant,
            )
        ),
        weightwatcher_analysis_seed_offset=int(
            analysis.get(
                "weightwatcher_analysis_seed_offset",
                defaults.weightwatcher_analysis_seed_offset,
            )
        ),
        device=str(runtime.get("device", defaults.device)),
        data_dir=str(runtime.get("data_dir", defaults.data_dir)),
        run_root=str(runtime.get("run_root", defaults.run_root)),
        tail_checkpoint_cache_root=str(
            runtime.get(
                "tail_checkpoint_cache_root",
                defaults.tail_checkpoint_cache_root,
            )
        ),
        adamw=adamw,
        muon=muon,
        muonclip_rms=muonclip,
    )
    config.validate()
    return config


def load_config(path: str | Path) -> TangentRGConfig:
    """Load JSON or YAML, importing PyYAML only when a YAML file is used."""

    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load tangent-RG YAML configurations"
            ) from exc
        with source.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("configuration root must be a mapping")
    return config_from_mapping(payload)


__all__ = [
    "AdamWProfile",
    "MuonClipRMSProfile",
    "MuonProfile",
    "OptimizerName",
    "SUPPORTED_OPTIMIZERS",
    "TangentRGConfig",
    "config_from_mapping",
    "load_config",
]
