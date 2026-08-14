from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
import torch

from .model import GPT, GPTConfig
from .spectral import WeightMatrixHolder, _attach_matrix_metadata


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int = 4242
    optimizer: str = "muon_clip"
    run_dir: str | None = None
    results_root: str | None = None
    runroot: str | None = None
    initial_checkpoint: str | None = None
    final_checkpoint: str | None = None
    output_dir: str | None = None
    angular_nulls: int = 100
    entry_nulls: int = 24
    min_tail: int = 20
    null_seed: int = 91337
    show_plots: bool = True

    @classmethod
    def from_env(cls) -> "AnalysisConfig":
        return cls(
            seed=int(_env("TARGET_SEED", "RG_SEED", default="4242")),
            optimizer=str(
                _env(
                    "TARGET_OPTIMIZER",
                    "OPTIMIZER_NAME",
                    default="muon_clip",
                )
            ).lower(),
            run_dir=_env("RUN_DIR"),
            results_root=_env("RESULTS_ROOT"),
            runroot=_env("RUNROOT"),
            initial_checkpoint=_env("INITIAL_CHECKPOINT_PATH"),
            final_checkpoint=_env(
                "FINAL_CHECKPOINT_PATH",
                "CHECKPOINT_PATH",
            ),
            output_dir=_env("ANGULAR_OUTPUT_DIR"),
            angular_nulls=int(_env("ANGULAR_N_NULL", default="100")),
            entry_nulls=int(
                _env("ANGULAR_N_ENTRY_NULL", default="24")
            ),
            min_tail=int(_env("ANGULAR_MIN_TAIL", default="20")),
            null_seed=int(_env("ANGULAR_NULL_SEED", default="91337")),
            show_plots=str(
                _env("ANGULAR_SHOW_PLOTS", default="1")
            ).lower()
            not in {"0", "false", "no", "off"},
        )

    def validate(self) -> None:
        if self.angular_nulls < 10:
            raise ValueError("ANGULAR_N_NULL must be at least 10")
        if self.entry_nulls < 3:
            raise ValueError("ANGULAR_N_ENTRY_NULL must be at least 3")
        if self.min_tail < 3:
            raise ValueError("ANGULAR_MIN_TAIL must be at least 3")


@dataclass(frozen=True)
class ResolvedRun:
    run_dir: Path
    run_dir_source: str
    initial_path: Path
    final_path: Path
    final_step: int
    output_dir: Path


def _checkpoint_paths(run_dir: Path) -> list[Path]:
    paths = list(run_dir.glob("*.pt"))
    paths += list((run_dir / "epoch_checkpoints").glob("*.pt"))
    return sorted(set(paths))


def _has_checkpoint(run_dir: Path) -> bool:
    return run_dir.is_dir() and bool(_checkpoint_paths(run_dir))


def _payload_step(path: Path) -> int | None:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        return int(payload.get("step")) if isinstance(payload, dict) else None
    except Exception:
        return None


def _environment_run_dir(config: AnalysisConfig) -> tuple[Path | None, str | None]:
    if config.run_dir:
        return Path(config.run_dir).expanduser().resolve(), "RUN_DIR"
    if config.results_root:
        return (
            Path(config.results_root).expanduser().resolve()
            / config.optimizer
            / f"seed_{config.seed}",
            "RESULTS_ROOT",
        )
    if config.runroot:
        return (
            Path(config.runroot).expanduser().resolve()
            / "results"
            / config.optimizer
            / f"seed_{config.seed}",
            "RUNROOT",
        )
    return None, None


def _discover_run(config: AnalysisConfig) -> tuple[Path, str]:
    exact, source = _environment_run_dir(config)
    if exact is not None:
        if not _has_checkpoint(exact):
            raise FileNotFoundError(
                f"{source} resolved to a run without checkpoints: {exact}"
            )
        return exact, str(source)

    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        candidates.append(
            base
            / "results"
            / config.optimizer
            / f"seed_{config.seed}"
        )
    for tmp in (Path("/tmp"), Path("/private/tmp")):
        if not tmp.is_dir():
            continue
        for pattern in (
            f"*/results/{config.optimizer}/seed_{config.seed}",
            f"*/*/results/{config.optimizer}/seed_{config.seed}",
        ):
            candidates.extend(tmp.glob(pattern))
    candidates = [
        path.resolve()
        for path in dict.fromkeys(candidates)
        if _has_checkpoint(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            "No matching run. Export RUN_DIR, RESULTS_ROOT, or RUNROOT."
        )
    newest = max(
        candidates,
        key=lambda path: max(
            checkpoint.stat().st_mtime
            for checkpoint in _checkpoint_paths(path)
        ),
    )
    return newest, "automatic discovery"


def _initial_checkpoint(config: AnalysisConfig, run_dir: Path) -> Path:
    if config.initial_checkpoint:
        path = Path(config.initial_checkpoint).expanduser().resolve()
        if not path.is_file() or _payload_step(path) != 0:
            raise ValueError(
                "INITIAL_CHECKPOINT_PATH must be a saved step-zero "
                f"checkpoint: {path}"
            )
        return path
    preferred = run_dir / "checkpoint_initial.pt"
    if preferred.is_file() and _payload_step(preferred) == 0:
        return preferred
    candidates = [
        path
        for path in _checkpoint_paths(run_dir)
        if _payload_step(path) == 0
    ]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(
        "No saved step-zero checkpoint. Strict analysis will not "
        "reconstruct W0 from the seed."
    )


def _final_checkpoint(config: AnalysisConfig, run_dir: Path) -> Path:
    path = (
        Path(config.final_checkpoint).expanduser().resolve()
        if config.final_checkpoint
        else run_dir / "checkpoint_final.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Final checkpoint is unavailable: {path}"
        )
    return path


def resolve_run(config: AnalysisConfig) -> ResolvedRun:
    config.validate()
    run_dir, source = _discover_run(config)
    initial = _initial_checkpoint(config, run_dir)
    final = _final_checkpoint(config, run_dir)
    final_step = _payload_step(final)
    if final_step is None or final_step <= 0:
        raise ValueError(
            f"Final checkpoint must have a positive step: {final}"
        )
    output = (
        Path(config.output_dir).expanduser().resolve()
        if config.output_dir
        else run_dir
        / "diagnostics"
        / f"angular_saved_step_0000000_vs_final_{final_step:07d}"
    )
    output.mkdir(parents=True, exist_ok=True)
    return ResolvedRun(
        run_dir=run_dir,
        run_dir_source=source,
        initial_path=initial,
        final_path=final,
        final_step=final_step,
        output_dir=output,
    )


def _load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise TypeError(f"Invalid RG checkpoint: {path}")
    return payload


def _normalized_state(state: dict) -> dict:
    state = dict(state)
    for prefix in ("_orig_mod.", "module."):
        if state and all(key.startswith(prefix) for key in state):
            state = {
                key[len(prefix) :]: value
                for key, value in state.items()
            }
    return state


def _model_config(initial: dict, final: dict, run_dir: Path) -> dict:
    for payload in (final, initial):
        config = payload.get("config", {})
        if isinstance(config, dict) and isinstance(config.get("model"), dict):
            return dict(config["model"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return dict(manifest["model"])


def _build_model(payload: dict, model_config: dict) -> GPT:
    model = GPT(GPTConfig(**model_config))
    model.load_state_dict(_normalized_state(payload["model"]), strict=True)
    model.eval()
    return model


def _model_digest(model: GPT) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(
            tensor.detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def _extract_weights(model: GPT) -> dict[str, np.ndarray]:
    import weightwatcher as ww

    holder = WeightMatrixHolder(model)
    watcher = ww.WeightWatcher(model=holder)
    described = watcher.describe(min_evals=1)
    table = _attach_matrix_metadata(
        pd.DataFrame(described),
        holder.matrix_metadata,
    )
    expected = {
        name: layer.weight.detach().cpu().numpy()
        for name, layer in holder.named_children()
    }
    method = getattr(watcher, "get_weights", None) or getattr(
        watcher,
        "get_Weights",
        None,
    )
    if method is None:
        raise AttributeError(
            "WeightWatcher exposes neither get_weights nor get_Weights"
        )
    result: dict[str, np.ndarray] = {}
    for _, row in table.iterrows():
        name = str(row["matrix_name"])
        values = method(layer=int(row["layer_id"]))
        values = [values] if isinstance(values, np.ndarray) else list(values)
        matrix = next(
            np.asarray(value, dtype=np.float64)
            for value in values
            if np.asarray(value).ndim == 2
        )
        target = np.asarray(expected[name], dtype=np.float64)
        if matrix.shape == target.shape and np.allclose(matrix, target):
            result[name] = matrix
        elif matrix.T.shape == target.shape and np.allclose(
            matrix.T,
            target,
        ):
            result[name] = matrix.T
        else:
            raise RuntimeError(
                f"WeightWatcher matrix mismatch for {name}"
            )
    return result


def load_weight_pairs(
    config: AnalysisConfig,
    resolved: ResolvedRun,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    initial = _load_payload(resolved.initial_path)
    final = _load_payload(resolved.final_path)
    if int(initial.get("step", -1)) != 0:
        raise ValueError("Initial checkpoint is not step zero")
    if int(initial.get("seed", config.seed)) != int(
        final.get("seed", config.seed)
    ):
        raise ValueError("Checkpoint seeds differ")
    optimizer = str(
        final.get("optimizer_name", config.optimizer)
    ).lower()
    if optimizer != config.optimizer:
        raise ValueError(
            f"Checkpoint optimizer={optimizer!r}, target={config.optimizer!r}"
        )
    model_config = _model_config(initial, final, resolved.run_dir)
    initial_model = _build_model(initial, model_config)
    final_model = _build_model(final, model_config)
    initial_weights = _extract_weights(initial_model)
    final_weights = _extract_weights(final_model)
    if set(initial_weights) != set(final_weights):
        raise RuntimeError("Initial and final matrix inventories differ")
    metadata = {
        "model_config": model_config,
        "initial_digest": _model_digest(initial_model),
        "final_digest": _model_digest(final_model),
        "optimizer": optimizer,
        "seed": int(final.get("seed", config.seed)),
    }
    return initial_weights, final_weights, metadata


@dataclass(frozen=True)
class TailFit:
    success: bool
    alpha: float = np.nan
    xmin: float = np.nan
    ks: float = np.nan
    n_tail: int = 0


def polar(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(
        np.asarray(matrix, dtype=np.float64),
        full_matrices=False,
    )
    return left @ right


def angular_from_polar(
    initial: np.ndarray,
    final: np.ndarray,
) -> dict[str, np.ndarray]:
    rows, columns = initial.shape
    relative = (
        final.T @ initial
        if rows >= columns
        else final @ initial.T
    )
    identity = np.eye(relative.shape[0])
    tilt_op = identity - relative.T @ relative
    tilt = np.linalg.eigvalsh((tilt_op + tilt_op.T) / 2)
    left, _, right = np.linalg.svd(relative, full_matrices=False)
    rotation = left @ right
    twist_op = 2 * identity - rotation - rotation.T
    twist = np.linalg.eigvalsh((twist_op + twist_op.T) / 2)
    return {
        "tilt": np.sort(np.clip(tilt, 0, 1)),
        "twist": np.sort(np.clip(twist, 0, 4)),
    }


def angular_spectra(
    initial: np.ndarray,
    final: np.ndarray,
) -> dict[str, np.ndarray]:
    return angular_from_polar(polar(initial), polar(final))


def haar_stiefel(
    rows: int,
    columns: int,
    rng: np.random.Generator,
) -> np.ndarray:
    basis, triangular = np.linalg.qr(
        rng.normal(size=(rows, columns)),
        mode="reduced",
    )
    return basis * np.where(np.diag(triangular) < 0, -1.0, 1.0)


def random_polar(
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    rows, columns = shape
    if rows >= columns:
        return haar_stiefel(rows, columns, rng)
    return haar_stiefel(columns, rows, rng).T


def projective(values: np.ndarray, upper: float) -> np.ndarray:
    normalized = np.clip(np.asarray(values) / upper, 0, 1)
    result = np.zeros_like(normalized)
    mask = normalized > 1e-12
    selected = np.clip(normalized[mask], 1e-12, 1 - 1e-12)
    result[mask] = selected / (1 - selected)
    return result


def fit_tail(values: np.ndarray, min_tail: int) -> TailFit:
    array = np.asarray(values)
    array = np.sort(array[np.isfinite(array) & (array > 0)])
    if array.size < min_tail:
        return TailFit(False)
    best: tuple[tuple, TailFit] | None = None
    starts = np.unique(
        np.linspace(
            0,
            array.size - min_tail,
            min(80, array.size - min_tail + 1),
            dtype=int,
        )
    )
    for start in starts:
        tail = array[start:]
        xmin = float(array[start])
        denominator = np.log(tail / xmin).sum()
        if denominator <= 1e-12:
            continue
        alpha = 1 + tail.size / denominator
        empirical = (
            np.arange(1, tail.size + 1) - 0.5
        ) / tail.size
        model = 1 - (tail / xmin) ** (1 - alpha)
        ks = float(np.max(np.abs(empirical - model)))
        fit = TailFit(
            True,
            float(alpha),
            xmin,
            ks,
            int(tail.size),
        )
        candidate = (ks, -tail.size, alpha, xmin)
        if best is None or candidate < best[0]:
            best = candidate, fit
    return best[1] if best else TailFit(False)


def gram_esd(matrix: np.ndarray) -> np.ndarray:
    values = np.linalg.svd(matrix, compute_uv=False) ** 2
    return np.sort(values / max(values.mean(), 1e-12))[::-1]


def shuffle_entries(
    matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    values = matrix.reshape(-1).copy()
    rng.shuffle(values)
    return values.reshape(matrix.shape)


def null_interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray([value for value in values if np.isfinite(value)])
    if not array.size:
        return np.nan, np.nan, np.nan
    return tuple(np.quantile(array, [0.025, 0.5, 0.975]))


def monte_carlo_ks(
    actual: np.ndarray,
    nulls: list[np.ndarray],
) -> tuple[float, float]:
    pooled = np.concatenate(nulls)
    observed = stats.ks_2samp(actual, pooled).statistic
    reference = []
    for index, sample in enumerate(nulls):
        others = np.concatenate(nulls[:index] + nulls[index + 1 :])
        reference.append(stats.ks_2samp(sample, others).statistic)
    probability = (
        1 + np.sum(np.asarray(reference) >= observed)
    ) / (len(reference) + 1)
    return float(observed), float(probability)


def ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values)
    array = np.sort(array[np.isfinite(array) & (array > 0)])
    if not array.size:
        return array, array
    return array, (array.size - np.arange(array.size)) / array.size


def assert_gauge_invariance() -> None:
    rng = np.random.default_rng(1)
    initial = rng.normal(size=(12, 8))
    final = rng.normal(size=(12, 8))
    left = haar_stiefel(12, 12, rng)
    right = haar_stiefel(8, 8, rng)
    original = angular_spectra(initial, final)
    transformed = angular_spectra(
        left @ initial @ right.T,
        left @ final @ right.T,
    )
    for key in ("tilt", "twist"):
        np.testing.assert_allclose(
            original[key],
            transformed[key],
            atol=1e-8,
        )
