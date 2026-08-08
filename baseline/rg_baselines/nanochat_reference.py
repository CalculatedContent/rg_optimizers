"""Pinned nanochat reference baselines for RG optimizer experiments.

Two explicit profiles are provided:

``d12``
    The canonical upstream reference. It preserves nanochat's tuned d12
    architecture, initialization, optimizer groups, scaling rules and schedules.
    This is a CUDA/server baseline and is intentionally not weakened to fit a
    laptop.

``mac``
    A smaller d4 profile using the same upstream implementation and optimizer
    logic, with full-context attention and a bounded batch/data preparation
    footprint suitable for Apple MPS development. It is a separate baseline
    version, not a surrogate result for d12.

The checkout is pinned by commit. The only source patch makes nanochat's global
initialization seed configurable so independent complete-run replicates are
possible. No architecture or optimization code is reimplemented here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

NANOCHAT_REPOSITORY = "https://github.com/karpathy/nanochat.git"
NANOCHAT_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
DEFAULT_NANOCHAT_SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class NanoChatD12Config:
    """Canonical pinned nanochat d12 reference profile."""

    profile_name: str = "d12"
    depth: int = 12
    max_seq_len: int = 2048
    window_pattern: str = "SSSL"
    target_param_data_ratio: float = 12.0
    num_iterations: int = -1
    device_batch_size: int = 32
    total_batch_size: int = -1

    embedding_lr: float = 0.30
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.020
    scalar_lr: float = 0.50
    weight_decay: float = 0.28
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05

    eval_every: int = 250
    eval_tokens: int = 80 * 524_288
    save_every: int = 250
    core_metric_every: int = 999_999
    core_metric_max_per_task: int = 500

    dataset_shards: int = 1000
    tokenizer_max_chars: int = 2_000_000_000
    vocab_size: int = 32_768
    compute_dtype: str | None = None

    @property
    def model_dim(self) -> int:
        # Upstream rounds depth*64 to a multiple of the 128-wide head.
        base = self.depth * 64
        return ((base + 127) // 128) * 128

    def validate(self) -> None:
        _validate_config(self)
        if self.profile_name != "d12" or self.depth != 12:
            raise ValueError("NanoChatD12Config is fixed to the canonical d12 profile")


@dataclass(frozen=True)
class NanoChatMacConfig:
    """Apple-MPS development profile using nanochat's native training code.

    The profile is deliberately meaningful rather than a 20-step smoke test:
    the target data ratio remains 12, while depth, sequence length, total token
    batch, data shards, and tokenizer sample are reduced to make repeated local
    optimizer experiments feasible.
    """

    profile_name: str = "mac_d4"
    depth: int = 4
    max_seq_len: int = 512
    window_pattern: str = "L"
    target_param_data_ratio: float = 12.0
    num_iterations: int = -1
    device_batch_size: int = 1
    total_batch_size: int = 32_768

    embedding_lr: float = 0.30
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.020
    scalar_lr: float = 0.50
    weight_decay: float = 0.28
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05

    eval_every: int = 100
    eval_tokens: int = 131_072
    save_every: int = 100
    core_metric_every: int = -1
    core_metric_max_per_task: int = 100

    dataset_shards: int = 100
    tokenizer_max_chars: int = 200_000_000
    vocab_size: int = 32_768
    compute_dtype: str | None = "float32"

    @property
    def model_dim(self) -> int:
        base = self.depth * 64
        return ((base + 127) // 128) * 128

    def validate(self) -> None:
        _validate_config(self)
        if self.profile_name != "mac_d4" or self.depth != 4:
            raise ValueError("NanoChatMacConfig is fixed to the d4 MPS profile")
        if self.window_pattern != "L":
            raise ValueError("the MPS profile uses full-context attention")


NanoChatConfig = NanoChatD12Config | NanoChatMacConfig


def _validate_config(config: NanoChatConfig) -> None:
    if config.depth < 1 or config.max_seq_len < 2:
        raise ValueError("depth and max_seq_len must be positive")
    if config.target_param_data_ratio <= 0 and config.num_iterations < 0:
        raise ValueError("a positive target ratio or explicit iteration count is required")
    if config.device_batch_size < 1:
        raise ValueError("device_batch_size must be positive")
    if config.total_batch_size == 0 or config.total_batch_size < -1:
        raise ValueError("total_batch_size must be -1 or positive")
    if not 0 < config.warmdown_ratio <= 1:
        raise ValueError("warmdown_ratio must be in (0, 1]")
    if not 0 <= config.final_lr_frac <= 1:
        raise ValueError("final_lr_frac must be in [0, 1]")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.save_every == 0 or config.eval_every == 0:
        raise ValueError("save_every/eval_every must be -1 or positive")
    if config.dataset_shards < 1 or config.tokenizer_max_chars < 1:
        raise ValueError("data/tokenizer preparation sizes must be positive")
    if config.compute_dtype not in {None, "float32", "bfloat16", "float16"}:
        raise ValueError("unsupported compute_dtype")


def detect_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_profile(
    profile: str = "auto",
    *,
    device_type: str | None = None,
) -> NanoChatConfig:
    device_type = device_type or detect_device_type()
    normalized = str(profile).strip().lower()
    if normalized == "auto":
        normalized = "d12" if device_type == "cuda" else "mac"
    if normalized in {"d12", "canonical", "server"}:
        return NanoChatD12Config()
    if normalized in {"mac", "mps", "d4", "mac_d4"}:
        return NanoChatMacConfig()
    raise ValueError(f"unknown nanochat profile: {profile!r}")


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def ensure_checkout(
    checkout_dir: Path,
    *,
    commit: str = NANOCHAT_COMMIT,
) -> Path:
    """Clone nanochat if necessary and hard-pin it to the audited commit."""

    checkout_dir = Path(checkout_dir).expanduser().resolve()
    if not checkout_dir.exists():
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(
            ["git", "clone", NANOCHAT_REPOSITORY, str(checkout_dir)],
            cwd=checkout_dir.parent,
        )
    if not (checkout_dir / ".git").is_dir():
        raise RuntimeError(f"{checkout_dir} exists but is not a git checkout")
    _run(["git", "fetch", "origin"], cwd=checkout_dir)
    _run(["git", "checkout", "--detach", commit], cwd=checkout_dir)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout_dir, text=True
    ).strip()
    if head != commit:
        raise RuntimeError(f"nanochat pin failed: expected {commit}, got {head}")
    _install_seed_patch(checkout_dir)
    return checkout_dir


def _install_seed_patch(checkout_dir: Path) -> None:
    """Make only nanochat's global initialization seed configurable."""

    path = checkout_dir / "nanochat" / "common.py"
    text = path.read_text(encoding="utf-8")
    if "NANOCHAT_SEED" in text:
        return
    old = """    torch.manual_seed(42)\n    if device_type == \"cuda\":\n        torch.cuda.manual_seed(42)\n"""
    new = """    seed = int(os.environ.get(\"NANOCHAT_SEED\", \"42\"))\n    torch.manual_seed(seed)\n    if device_type == \"cuda\":\n        torch.cuda.manual_seed(seed)\n    elif device_type == \"mps\" and hasattr(torch.mps, \"manual_seed\"):\n        torch.mps.manual_seed(seed)\n"""
    if old not in text:
        raise RuntimeError(
            "pinned nanochat common.py no longer matches the audited seed patch"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def ensure_environment(
    checkout_dir: Path,
    *,
    device_type: str = "auto",
) -> None:
    """Create the upstream uv environment for CUDA, CPU, or Apple MPS."""

    if shutil.which("uv") is None:
        raise RuntimeError("uv is required to create nanochat's pinned environment")
    resolved = detect_device_type() if device_type == "auto" else device_type
    if resolved not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device_type: {resolved}")
    command = ["uv", "sync", "--group", "dev"]
    if resolved == "cuda":
        command.extend(["--extra", "gpu"])
    elif resolved == "cpu" and platform.system() != "Darwin":
        command.extend(["--extra", "cpu"])
    # macOS deliberately uses PyPI's native torch wheel, not the Linux CPU index.
    _run(command, cwd=Path(checkout_dir))


def _uv_python(checkout_dir: Path) -> str:
    candidate = Path(checkout_dir) / ".venv" / "bin" / "python"
    if not candidate.is_file():
        raise RuntimeError("nanochat .venv is missing; run ensure_environment first")
    return str(candidate)


def prepare_data(
    checkout_dir: Path,
    cache_dir: Path,
    config: NanoChatConfig,
) -> None:
    """Run nanochat's native shard and tokenizer preparation for one profile."""

    config.validate()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = str(cache_dir)
    python = _uv_python(checkout_dir)
    _run(
        [python, "-m", "nanochat.dataset", "-n", str(config.dataset_shards)],
        cwd=Path(checkout_dir),
        env=env,
    )
    _run(
        [
            python,
            "-m",
            "scripts.tok_train",
            f"--max-chars={config.tokenizer_max_chars}",
            f"--vocab-size={config.vocab_size}",
        ],
        cwd=Path(checkout_dir),
        env=env,
    )


def model_tag(config: NanoChatConfig, seed: int) -> str:
    return f"rg_{config.profile_name}_seed{int(seed)}"


def checkpoint_dir(
    cache_dir: Path,
    *,
    config: NanoChatConfig,
    seed: int,
) -> Path:
    return (
        Path(cache_dir).expanduser().resolve()
        / "base_checkpoints"
        / model_tag(config, seed)
    )


def find_resume_step(
    cache_dir: Path,
    *,
    config: NanoChatConfig,
    seed: int,
    nproc_per_node: int,
) -> int | None:
    """Find the newest checkpoint with model, metadata, and every optimizer shard."""

    directory = checkpoint_dir(cache_dir, config=config, seed=seed)
    if not directory.is_dir():
        return None
    candidates = sorted(
        (
            int(match.group(1))
            for path in directory.glob("model_*.pt")
            if (match := re.fullmatch(r"model_(\d+)\.pt", path.name))
        ),
        reverse=True,
    )
    for step in candidates:
        required = [directory / f"meta_{step:06d}.json"]
        required.extend(
            directory / f"optim_{step:06d}_rank{rank}.pt"
            for rank in range(int(nproc_per_node))
        )
        if all(path.is_file() for path in required):
            return step
    return None


def _config_fingerprint(
    config: NanoChatConfig,
    *,
    seed: int,
    device_type: str,
    nproc_per_node: int,
) -> str:
    payload = {
        "nanochat_commit": NANOCHAT_COMMIT,
        "config": asdict(config),
        "seed": int(seed),
        "device_type": str(device_type),
        "nproc_per_node": int(nproc_per_node),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def training_command(
    checkout_dir: Path,
    config: NanoChatConfig,
    *,
    seed: int,
    device_type: str = "auto",
    nproc_per_node: int = 1,
    resume_from_step: int | None = None,
) -> list[str]:
    """Return the exact pinned upstream training command for one replicate."""

    config.validate()
    resolved = detect_device_type() if device_type == "auto" else device_type
    if resolved not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device_type: {resolved}")
    if resolved != "cuda" and nproc_per_node != 1:
        raise ValueError("CPU/MPS nanochat runs must use one process")
    if nproc_per_node < 1:
        raise ValueError("nproc_per_node must be positive")
    if config.total_batch_size > 0:
        micro_tokens = (
            config.device_batch_size * config.max_seq_len * nproc_per_node
        )
        if config.total_batch_size % micro_tokens:
            raise ValueError(
                "total_batch_size must be divisible by device_batch_size * "
                "max_seq_len * world size"
            )

    arguments = [
        "-m",
        "scripts.base_train",
        f"--device-type={resolved}",
        f"--depth={config.depth}",
        f"--max-seq-len={config.max_seq_len}",
        f"--window-pattern={config.window_pattern}",
        f"--num-iterations={config.num_iterations}",
        f"--target-param-data-ratio={config.target_param_data_ratio}",
        f"--device-batch-size={config.device_batch_size}",
        f"--total-batch-size={config.total_batch_size}",
        f"--embedding-lr={config.embedding_lr}",
        f"--unembedding-lr={config.unembedding_lr}",
        f"--matrix-lr={config.matrix_lr}",
        f"--scalar-lr={config.scalar_lr}",
        f"--weight-decay={config.weight_decay}",
        f"--warmup-steps={config.warmup_steps}",
        f"--warmdown-ratio={config.warmdown_ratio}",
        f"--final-lr-frac={config.final_lr_frac}",
        f"--eval-every={config.eval_every}",
        f"--eval-tokens={config.eval_tokens}",
        f"--save-every={config.save_every}",
        f"--core-metric-every={config.core_metric_every}",
        f"--core-metric-max-per-task={config.core_metric_max_per_task}",
        "--sample-every=-1",
        "--run=dummy",
        f"--model-tag={model_tag(config, seed)}",
    ]
    if resume_from_step is not None:
        arguments.append(f"--resume-from-step={int(resume_from_step)}")

    python = _uv_python(checkout_dir)
    if resolved == "cuda" and nproc_per_node > 1:
        torchrun = Path(checkout_dir) / ".venv" / "bin" / "torchrun"
        return [
            str(torchrun),
            "--standalone",
            f"--nproc_per_node={int(nproc_per_node)}",
            *arguments,
        ]
    return [python, *arguments]


def run_seed(
    checkout_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    config: NanoChatConfig,
    *,
    seed: int,
    device_type: str = "auto",
    nproc_per_node: int = 1,
    resume: bool = True,
) -> Path:
    """Run or resume one replicate and return its persistent log path."""

    resolved = detect_device_type() if device_type == "auto" else device_type
    if resolved != "cuda":
        nproc_per_node = 1
    fingerprint = _config_fingerprint(
        config,
        seed=seed,
        device_type=resolved,
        nproc_per_node=nproc_per_node,
    )
    seed_dir = Path(output_dir).expanduser().resolve() / f"seed_{int(seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_path = seed_dir / "training.log"
    completion_path = seed_dir / "run_complete.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("fingerprint") != fingerprint:
            raise RuntimeError("completed nanochat run belongs to a different profile")
        return log_path

    resume_step = (
        find_resume_step(
            cache_dir,
            config=config,
            seed=seed,
            nproc_per_node=nproc_per_node,
        )
        if resume
        else None
    )
    command = training_command(
        checkout_dir,
        config,
        seed=seed,
        device_type=resolved,
        nproc_per_node=nproc_per_node,
        resume_from_step=resume_step,
    )
    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = str(Path(cache_dir).expanduser().resolve())
    env["NANOCHAT_SEED"] = str(int(seed))
    env.setdefault("OMP_NUM_THREADS", "1")
    if config.compute_dtype is not None:
        env["NANOCHAT_DTYPE"] = config.compute_dtype

    mode = "a" if resume_step is not None and log_path.exists() else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        if mode == "a":
            log.write(f"\n# RESUME FROM STEP {resume_step}\n")
        print("+", " ".join(command), flush=True)
        process = subprocess.Popen(
            command,
            cwd=Path(checkout_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    final_step = find_resume_step(
        cache_dir,
        config=config,
        seed=seed,
        nproc_per_node=nproc_per_node,
    )
    if final_step is None:
        raise RuntimeError("nanochat exited successfully without a complete checkpoint")
    config_payload = {
        "fingerprint": fingerprint,
        "nanochat_commit": NANOCHAT_COMMIT,
        "seed": int(seed),
        "device_type": resolved,
        "nproc_per_node": int(nproc_per_node),
        "config": asdict(config),
    }
    (seed_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary = seed_dir / "run_complete.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                **config_payload,
                "completed": True,
                "final_step": int(final_step),
                "checkpoint_dir": str(
                    checkpoint_dir(cache_dir, config=config, seed=seed)
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(completion_path)
    return log_path


_TRAIN_RE = re.compile(
    r"step\s+(\d+)/(\d+).*?loss:\s+([0-9.eE+-]+).*?"
    r"lrm:\s+([0-9.eE+-]+).*?tok/sec:\s+([0-9,]+).*?"
    r"total time:\s+([0-9.eE+-]+)m"
)
_VAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([0-9.eE+-]+)")
_CORE_RE = re.compile(r"Step\s+(\d+)\s+\|\s+CORE metric:\s+([0-9.eE+-]+)")


def parse_training_log(
    log_path: Path,
    *,
    seed: int,
    profile_name: str | None = None,
) -> pd.DataFrame:
    """Parse a possibly resumed nanochat log into unique step-level rows."""

    rows: dict[int, dict[str, float | int | str]] = {}
    for line in Path(log_path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        match = _TRAIN_RE.search(line)
        if match:
            step, total, loss, lr_multiplier, tokens_per_sec, minutes = match.groups()
            row = rows.setdefault(
                int(step),
                {"seed": int(seed), "step": int(step)},
            )
            row.update(
                num_iterations=int(total),
                train_loss=float(loss),
                lr_multiplier=float(lr_multiplier),
                tokens_per_sec=int(tokens_per_sec.replace(",", "")),
                total_training_minutes=float(minutes),
            )
        match = _VAL_RE.search(line)
        if match:
            step, value = match.groups()
            rows.setdefault(
                int(step), {"seed": int(seed), "step": int(step)}
            )["validation_bpb"] = float(value)
        match = _CORE_RE.search(line)
        if match:
            step, value = match.groups()
            rows.setdefault(
                int(step), {"seed": int(seed), "step": int(step)}
            )["core_metric"] = float(value)
    frame = pd.DataFrame(rows.values()).sort_values("step").reset_index(drop=True)
    if profile_name is not None and not frame.empty:
        frame.insert(0, "profile", profile_name)
    return frame


def collect_metrics(
    log_paths: Iterable[tuple[int, Path]],
    output_path: Path | None = None,
    *,
    profile_name: str | None = None,
) -> pd.DataFrame:
    frames = [
        parse_training_log(path, seed=seed, profile_name=profile_name)
        for seed, path in log_paths
    ]
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not metrics.empty:
        metrics = metrics.drop_duplicates(["seed", "step"], keep="last")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(output_path, index=False)
    return metrics


def _mean_ci95(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(array.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "ci95_low": np.nan, "ci95_high": np.nan}
    std = float(array.std(ddof=1))
    critical = 4.302652729911275 if n == 3 else 1.959963984540054
    half = critical * std / math.sqrt(n)
    return {"n": n, "mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def summarize_training_metrics(
    metrics: pd.DataFrame,
    *,
    value: str,
    expected_seeds: Sequence[int] = DEFAULT_NANOCHAT_SEEDS,
) -> pd.DataFrame:
    rows = []
    expected = tuple(sorted(int(seed) for seed in expected_seeds))
    subset = metrics.dropna(subset=[value]).copy()
    for step, group in subset.groupby("step", sort=True):
        observed = tuple(sorted(group["seed"].astype(int).unique()))
        if observed != expected or len(group) != len(expected):
            continue
        rows.append({"step": int(step), **_mean_ci95(group[value].tolist())})
    return pd.DataFrame(rows)


class _PrincipalMatrixHolder(nn.Module):
    """CPU-only views of the six principal matrices in each nanochat block."""

    def __init__(self, model) -> None:
        super().__init__()
        self.metadata: list[dict[str, object]] = []
        for block_index, block in enumerate(model.transformer.h):
            matrices = (
                ("W_Q", block.attn.c_q.weight),
                ("W_K", block.attn.c_k.weight),
                ("W_V", block.attn.c_v.weight),
                ("W_O", block.attn.c_proj.weight),
                ("W_MLP_IN", block.mlp.c_fc.weight),
                ("W_MLP_OUT", block.mlp.c_proj.weight),
            )
            for matrix_type, parameter in matrices:
                weight = parameter.detach().float().cpu()
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


def _attach_matrix_metadata(
    details: pd.DataFrame,
    metadata: list[dict[str, object]],
) -> pd.DataFrame:
    frame = details.copy().reset_index(drop=True)
    if len(frame) != len(metadata):
        raise RuntimeError(
            "WeightWatcher did not return one row per principal nanochat matrix"
        )
    order = (
        pd.to_numeric(frame["layer_id"], errors="raise").sort_values().index.tolist()
        if "layer_id" in frame
        else list(frame.index)
    )
    names: list[str | None] = [None] * len(frame)
    matrix_types: list[str | None] = [None] * len(frame)
    blocks: list[int | None] = [None] * len(frame)
    for item, row_index in zip(metadata, order, strict=True):
        names[row_index] = str(item["matrix_name"])
        matrix_types[row_index] = str(item["matrix_type"])
        blocks[row_index] = int(item["block"])
    frame.insert(0, "matrix_name", names)
    frame.insert(1, "matrix_type", matrix_types)
    frame.insert(2, "block", blocks)
    return frame


def analyze_weightwatcher_checkpoints(
    checkout_dir: Path,
    cache_dir: Path,
    *,
    config: NanoChatConfig,
    seed: int,
    output_csv: Path,
    min_evals: int = 20,
) -> pd.DataFrame:
    """Analyze only the principal hidden matrices of every saved checkpoint."""

    os.environ["NANOCHAT_BASE_DIR"] = str(Path(cache_dir).expanduser().resolve())
    if str(checkout_dir) not in sys.path:
        sys.path.insert(0, str(checkout_dir))
    import weightwatcher as ww
    from nanochat.checkpoint_manager import build_model

    directory = checkpoint_dir(cache_dir, config=config, seed=seed)
    steps = sorted(
        int(match.group(1))
        for path in directory.glob("model_*.pt")
        if (match := re.fullmatch(r"model_(\d+)\.pt", path.name))
    )
    if not steps:
        raise FileNotFoundError(f"no nanochat checkpoints found in {directory}")

    rows = []
    for step in steps:
        model, _, _ = build_model(
            str(directory), step, torch.device("cpu"), phase="eval"
        )
        holder = _PrincipalMatrixHolder(model)
        details = ww.WeightWatcher(model=holder).analyze(
            ERG=True,
            randomize=True,
            plot=False,
            min_evals=int(min_evals),
        )
        frame = _attach_matrix_metadata(pd.DataFrame(details), holder.metadata)
        required = {"alpha", "ERG_gap", "num_traps"}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(
                f"WeightWatcher is missing required columns: {sorted(missing)}"
            )
        if frame[list(required)].isna().any().any():
            raise RuntimeError("required WeightWatcher values contain NaN")
        frame.insert(0, "step", int(step))
        frame.insert(0, "seed", int(seed))
        frame.insert(0, "profile", config.profile_name)
        rows.append(frame)
        del holder
        del model

    result = pd.concat(rows, ignore_index=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def summarize_weightwatcher(
    frame: pd.DataFrame,
    metric: str,
    *,
    expected_seeds: Sequence[int] = DEFAULT_NANOCHAT_SEEDS,
) -> pd.DataFrame:
    expected = tuple(sorted(int(seed) for seed in expected_seeds))
    rows = []
    keys = ["matrix_name", "matrix_type", "block", "step"]
    for key, group in frame.groupby(keys, sort=True):
        observed = tuple(sorted(group["seed"].astype(int).unique()))
        if observed != expected or len(group) != len(expected):
            continue
        row = dict(zip(keys, key, strict=True))
        row.update(_mean_ci95(group[metric].tolist()))
        rows.append(row)
    return pd.DataFrame(rows)
