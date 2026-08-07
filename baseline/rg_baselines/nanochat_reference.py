"""Pinned nanochat d12 reference baseline for RG optimizer experiments.

The goal of this module is deliberately conservative: run Andrej Karpathy's
nanochat training code at the d12 reference scale with its native tuned recipe,
rather than reimplementing nanochat inside rg_optimizers.

The upstream checkout is pinned by commit.  The only source modification made
at runtime is replacing nanochat's hard-coded seed=42 with the NANOCHAT_SEED
environment variable so that statistically independent baseline replicates are
possible.  All architecture, initialization, optimizer, learning-rate,
momentum, weight-decay, data, and scaling-law logic remains upstream code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Iterable

import pandas as pd

NANOCHAT_REPOSITORY = "https://github.com/karpathy/nanochat.git"
# Current upstream master inspected when this baseline was authored (2026-08-07).
NANOCHAT_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
DEFAULT_NANOCHAT_SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class NanoChatD12Config:
    """Strong, research-sized nanochat reference recipe.

    d12 is nanochat's reference/tuning scale.  Width, number of heads, optimal
    batch size, token horizon, LR transfer, and weight-decay transfer are then
    derived by upstream nanochat exactly as in scripts/base_train.py.
    """

    depth: int = 12
    max_seq_len: int = 2048
    target_param_data_ratio: float = 12.0
    device_batch_size: int = 32
    total_batch_size: int = -1  # upstream auto-compute; d12 reference ~= 2**19 tokens

    # Upstream tuned base values.  nanochat applies its own batch/dmodel scaling.
    embedding_lr: float = 0.30
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.020
    scalar_lr: float = 0.50
    weight_decay: float = 0.28

    # Upstream schedule: linear warmup -> plateau -> long linear warmdown.
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05

    eval_every: int = 250
    eval_tokens: int = 80 * 524_288
    save_every: int = 250
    core_metric_every: int = 999_999  # final step still evaluates CORE
    core_metric_max_per_task: int = 500

    # Dataset/tokenizer preparation used by nanochat's miniseries script.
    dataset_shards: int = 1000
    tokenizer_max_chars: int = 2_000_000_000
    vocab_size: int = 32_768

    @property
    def model_dim(self) -> int:
        # d12*64=768, already divisible by the 128 head dimension.
        return self.depth * 64

    def validate(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >=2")
        if self.target_param_data_ratio <= 0:
            raise ValueError("target_param_data_ratio must be positive")
        if self.device_batch_size < 1:
            raise ValueError("device_batch_size must be positive")
        if not 0 < self.warmdown_ratio <= 1:
            raise ValueError("warmdown_ratio must be in (0,1]")
        if not 0 <= self.final_lr_frac <= 1:
            raise ValueError("final_lr_frac must be in [0,1]")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_checkout(checkout_dir: Path, *, commit: str = NANOCHAT_COMMIT) -> Path:
    """Clone nanochat if necessary and hard-pin the checkout to ``commit``."""
    checkout_dir = Path(checkout_dir).expanduser().resolve()
    if not checkout_dir.exists():
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", NANOCHAT_REPOSITORY, str(checkout_dir)], cwd=checkout_dir.parent)
    if not (checkout_dir / ".git").is_dir():
        raise RuntimeError(f"{checkout_dir} exists but is not a git checkout")
    _run(["git", "fetch", "origin"], cwd=checkout_dir)
    _run(["git", "checkout", "--detach", commit], cwd=checkout_dir)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout_dir, text=True).strip()
    if head != commit:
        raise RuntimeError(f"nanochat pin failed: expected {commit}, got {head}")
    _install_seed_patch(checkout_dir)
    return checkout_dir


def _install_seed_patch(checkout_dir: Path) -> None:
    """Allow replicate seeds while leaving nanochat's default seed equal to 42."""
    path = checkout_dir / "nanochat" / "common.py"
    text = path.read_text(encoding="utf-8")
    if 'NANOCHAT_SEED' in text:
        return
    old = """    torch.manual_seed(42)\n    if device_type == \"cuda\":\n        torch.cuda.manual_seed(42)\n"""
    new = """    seed = int(os.environ.get(\"NANOCHAT_SEED\", \"42\"))\n    torch.manual_seed(seed)\n    if device_type == \"cuda\":\n        torch.cuda.manual_seed(seed)\n"""
    if old not in text:
        raise RuntimeError(
            "Pinned nanochat common.py no longer matches the audited seed patch; "
            "do not silently modify an unknown upstream revision."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def ensure_environment(checkout_dir: Path, *, gpu: bool = True) -> None:
    """Create nanochat's uv environment using its own dependency lock/config."""
    if shutil.which("uv") is None:
        raise RuntimeError("uv is required. Install uv before running the nanochat baseline.")
    extra = "gpu" if gpu else "cpu"
    _run(["uv", "sync", "--extra", extra, "--group", "dev"], cwd=checkout_dir)


def _uv_python(checkout_dir: Path) -> str:
    candidate = checkout_dir / ".venv" / "bin" / "python"
    if not candidate.is_file():
        raise RuntimeError("nanochat .venv is missing; run ensure_environment first")
    return str(candidate)


def prepare_data(checkout_dir: Path, cache_dir: Path, config: NanoChatD12Config) -> None:
    """Prepare the same dataset/tokenizer family used by nanochat miniseries runs."""
    config.validate()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = str(cache_dir)
    py = _uv_python(checkout_dir)
    _run([py, "-m", "nanochat.dataset", "-n", str(config.dataset_shards)], cwd=checkout_dir, env=env)
    _run(
        [
            py, "-m", "scripts.tok_train",
            f"--max-chars={config.tokenizer_max_chars}",
            f"--vocab-size={config.vocab_size}",
        ],
        cwd=checkout_dir,
        env=env,
    )


def training_command(
    checkout_dir: Path,
    config: NanoChatD12Config,
    *,
    seed: int,
    nproc_per_node: int = 8,
) -> list[str]:
    """Return the exact command for one pinned d12 reference replicate."""
    config.validate()
    tag = f"rg_d12_seed{seed}"
    args = [
        "-m", "scripts.base_train",
        f"--depth={config.depth}",
        f"--max-seq-len={config.max_seq_len}",
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
        f"--model-tag={tag}",
    ]
    py = _uv_python(checkout_dir)
    if nproc_per_node > 1:
        torchrun = checkout_dir / ".venv" / "bin" / "torchrun"
        return [str(torchrun), "--standalone", f"--nproc_per_node={nproc_per_node}", *args]
    return [py, *args]


def run_seed(
    checkout_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    config: NanoChatD12Config,
    *,
    seed: int,
    nproc_per_node: int = 8,
) -> Path:
    """Run one replicate and tee stdout/stderr to a persistent log."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"nanochat_d12_seed{seed}.log"
    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = str(Path(cache_dir).expanduser().resolve())
    env["NANOCHAT_SEED"] = str(seed)
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = training_command(checkout_dir, config, seed=seed, nproc_per_node=nproc_per_node)
    print("+", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=checkout_dir,
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
        raise subprocess.CalledProcessError(return_code, cmd)
    (output_dir / f"config_seed{seed}.json").write_text(
        json.dumps({"seed": seed, "nanochat_commit": NANOCHAT_COMMIT, **asdict(config)}, indent=2),
        encoding="utf-8",
    )
    return log_path


_TRAIN_RE = re.compile(
    r"step\s+(\d+)/(\d+).*?loss:\s+([0-9.eE+-]+).*?lrm:\s+([0-9.eE+-]+).*?tok/sec:\s+([0-9,]+).*?total time:\s+([0-9.eE+-]+)m"
)
_VAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([0-9.eE+-]+)")
_CORE_RE = re.compile(r"Step\s+(\d+)\s+\|\s+CORE metric:\s+([0-9.eE+-]+)")


def parse_training_log(log_path: Path, *, seed: int) -> pd.DataFrame:
    """Parse nanochat's native training log into tidy step-level metrics."""
    rows: dict[int, dict[str, float | int]] = {}
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = _TRAIN_RE.search(line)
        if match:
            step, total, loss, lrm, tps, minutes = match.groups()
            row = rows.setdefault(int(step), {"seed": seed, "step": int(step)})
            row.update(
                num_iterations=int(total),
                train_loss=float(loss),
                lr_multiplier=float(lrm),
                tokens_per_sec=int(tps.replace(",", "")),
                total_training_minutes=float(minutes),
            )
        match = _VAL_RE.search(line)
        if match:
            step, value = match.groups()
            rows.setdefault(int(step), {"seed": seed, "step": int(step)})["val_bpb"] = float(value)
        match = _CORE_RE.search(line)
        if match:
            step, value = match.groups()
            rows.setdefault(int(step), {"seed": seed, "step": int(step)})["core_metric"] = float(value)
    return pd.DataFrame(rows.values()).sort_values("step").reset_index(drop=True)


def collect_metrics(log_paths: Iterable[tuple[int, Path]], output_path: Path | None = None) -> pd.DataFrame:
    frames = [parse_training_log(path, seed=seed) for seed, path in log_paths]
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(output_path, index=False)
    return metrics


def checkpoint_dir(cache_dir: Path, *, seed: int) -> Path:
    return Path(cache_dir).expanduser().resolve() / "base_checkpoints" / f"rg_d12_seed{seed}"


def analyze_weightwatcher_checkpoints(
    checkout_dir: Path,
    cache_dir: Path,
    *,
    seed: int,
    output_csv: Path,
) -> pd.DataFrame:
    """Run WeightWatcher offline on every saved nanochat checkpoint.

    This keeps WeightWatcher out of the timed training loop.  We persist every
    column returned by ``analyze(ERG=True, randomize=True)`` and add seed/step,
    including alpha, randomized-MP trap information, and ERG metrics whenever
    provided by the installed WeightWatcher version.
    """
    env_base = str(Path(cache_dir).expanduser().resolve())
    os.environ["NANOCHAT_BASE_DIR"] = env_base
    if str(checkout_dir) not in sys.path:
        sys.path.insert(0, str(checkout_dir))
    import torch
    import weightwatcher as ww
    from nanochat.checkpoint_manager import build_model, find_last_step

    cdir = checkpoint_dir(cache_dir, seed=seed)
    steps = sorted(
        int(path.stem.split("_")[-1])
        for path in cdir.glob("model_*.pt")
    )
    if not steps:
        raise FileNotFoundError(f"No nanochat checkpoints found in {cdir}")
    rows = []
    for step in steps:
        model, _, _ = build_model(str(cdir), step, torch.device("cpu"), phase="eval")
        details = ww.WeightWatcher(model=model).analyze(ERG=True, randomize=True)
        details = details.copy()
        details.insert(0, "step", step)
        details.insert(0, "seed", seed)
        rows.append(details)
        del model
    result = pd.concat(rows, ignore_index=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
