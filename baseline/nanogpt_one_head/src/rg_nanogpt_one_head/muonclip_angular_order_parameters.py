from __future__ import annotations

"""Candidate angular quotient order parameters for MuonClip spectral RG.

Each transform removes the instantaneous singular values of a layer and
reconstructs an ordinary real matrix from the polar/singular-vector geometry.
The resulting matrix can be passed to WeightWatcher, so its Gram spectrum can
be tested with the same finite-window HTSR and first-moment RG power counting
used for ordinary weight matrices.

These transforms are hypotheses, not identities. Every candidate must be
compared with the saved random initialization and a method-matched Haar or
scrambled control.
"""

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from scipy.linalg import logm


METHOD_ORDER = (
    "raw",
    "polar_intensity",
    "stiefel_log_gauge",
    "flag_shell_curvature",
    "haar_connected_susceptibility",
    "diffusion_green",
    "temporal_drift",
)


@dataclass
class TransformResult:
    method: str
    matrix: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def polar_factor(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_t


def random_polar(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Haar orthogonal or uniform Stiefel polar factor of ``shape``."""
    rows, columns = map(int, shape)
    if rows >= columns:
        gaussian = rng.normal(size=(rows, columns))
        q, r = np.linalg.qr(gaussian, mode="reduced")
        signs = np.sign(np.diag(r))
        signs[signs == 0.0] = 1.0
        return q * signs
    return random_polar((columns, rows), rng).T


def spectral_unit_mean(
    matrix: np.ndarray,
    *,
    relative_floor: float = 1e-14,
    absolute_floor: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Fix the RG scalar gauge: mean positive Gram eigenvalue equals one."""
    matrix = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if (
        singular_values.size == 0
        or float(singular_values[0]) <= float(absolute_floor)
    ):
        return np.zeros_like(matrix), 0.0
    positive = singular_values > float(singular_values[0]) * relative_floor
    rank = int(np.count_nonzero(positive))
    energy = float(np.sum(singular_values[positive] ** 2))
    if rank <= 0 or energy <= 0.0:
        return np.zeros_like(matrix), 0.0
    scale = math.sqrt(rank / energy)
    return matrix * scale, scale


def _as_tall(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] >= matrix.shape[1]:
        return matrix, False
    return matrix.T, True


def _from_tall(matrix: np.ndarray, transposed: bool) -> np.ndarray:
    return matrix.T if transposed else matrix


def _small_side_vectors(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return right_t.T if matrix.shape[0] >= matrix.shape[1] else left


def _signed_sqrt_symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    transformed = np.sign(values) * np.sqrt(np.abs(values))
    return (vectors * transformed) @ vectors.T


def polar_intensity(matrix: np.ndarray) -> TransformResult:
    """Connected coordinate intensity of the polar factor."""
    q = polar_factor(matrix)
    rank = min(q.shape)
    uniform = rank / float(q.size)
    connected = q * q - uniform
    normalized, scale = spectral_unit_mean(connected)
    return TransformResult(
        "polar_intensity",
        normalized,
        {
            "uniform_intensity": float(uniform),
            "spectral_scale": float(scale),
        },
    )


def _orthogonal_log(orthogonal: np.ndarray) -> tuple[np.ndarray, float]:
    raw = logm(np.asarray(orthogonal, dtype=np.float64))
    imaginary_ratio = float(
        np.linalg.norm(np.imag(raw)) / max(np.linalg.norm(raw), 1e-30)
    )
    real = np.real(raw)
    return 0.5 * (real - real.T), imaginary_ratio


def stiefel_log_gauge(
    reference: np.ndarray,
    target: np.ndarray,
) -> TransformResult:
    """Ambient approximation to ``Log_Q0(Qt)`` after Procrustes gauge fixing.

    Wide matrices are transposed to a tall Stiefel representation. Internal
    twist is represented by the logarithm of the Procrustes rotation; subspace
    tilt is represented by the horizontal tangent projection of the aligned
    residual. For square matrices this reduces to ``Q0 log(Q0.T @ Qt)``.
    """
    q0, transposed = _as_tall(polar_factor(reference))
    qt, target_transposed = _as_tall(polar_factor(target))
    if transposed != target_transposed or q0.shape != qt.shape:
        raise ValueError("reference and target polar factors are incompatible")

    overlap = q0.T @ qt
    left, cosines, right_t = np.linalg.svd(overlap, full_matrices=False)
    procrustes = left @ right_t
    aligned = qt @ procrustes.T

    twist_log, imaginary_ratio = _orthogonal_log(procrustes)
    twist = q0 @ twist_log
    delta = aligned - q0
    symmetric_part = 0.5 * (q0.T @ delta + delta.T @ q0)
    horizontal = delta - q0 @ symmetric_part
    field = _from_tall(twist + horizontal, transposed)
    normalized, scale = spectral_unit_mean(field)

    return TransformResult(
        "stiefel_log_gauge",
        normalized,
        {
            "principal_angle_cos_min": float(np.min(cosines)),
            "principal_angle_cos_median": float(np.median(cosines)),
            "twist_norm": float(np.linalg.norm(twist)),
            "horizontal_norm": float(np.linalg.norm(horizontal)),
            "log_imaginary_ratio": imaginary_ratio,
            "spectral_scale": float(scale),
        },
    )


def dyadic_shell_slices(rank: int) -> list[slice]:
    """Broad shells of sizes approximately 1/16, 1/16, 1/8, 1/4, 1/2."""
    rank = int(rank)
    if rank < 5:
        return [slice(index, index + 1) for index in range(rank)]
    sizes = [
        max(1, rank // 16),
        max(1, rank // 16),
        max(1, rank // 8),
        max(1, rank // 4),
    ]
    while sum(sizes) >= rank:
        largest = int(np.argmax(sizes))
        if sizes[largest] <= 1:
            break
        sizes[largest] -= 1
    sizes.append(rank - sum(sizes))
    result: list[slice] = []
    start = 0
    for size in sizes:
        if size > 0:
            result.append(slice(start, start + size))
            start += size
    return result


def flag_shell_field(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Projector-valued multiscale flag with within-shell gauge removed."""
    vectors = _small_side_vectors(matrix)
    shells = dyadic_shell_slices(vectors.shape[1])
    coefficients = np.arange(len(shells), dtype=np.float64)
    coefficients -= float(np.mean(coefficients))
    norm = float(np.linalg.norm(coefficients))
    if norm > 0.0:
        coefficients /= norm

    field = np.zeros((vectors.shape[0], vectors.shape[0]), dtype=np.float64)
    shell_sizes = []
    for coefficient, shell in zip(coefficients, shells):
        block = vectors[:, shell]
        field += float(coefficient) * (block @ block.T)
        shell_sizes.append(int(shell.stop - shell.start))
    field = 0.5 * (field + field.T)
    return field, {
        "flag_shell_count": len(shells),
        "flag_shell_sizes": shell_sizes,
    }


def flag_shell_curvature(
    reference: np.ndarray,
    target: np.ndarray,
) -> TransformResult:
    """Commutator of the initial and current multiscale flag fields."""
    reference_field, metadata = flag_shell_field(reference)
    target_field, _ = flag_shell_field(target)
    curvature = reference_field @ target_field - target_field @ reference_field
    normalized, scale = spectral_unit_mean(curvature)
    metadata = dict(metadata)
    metadata.update(
        {
            "curvature_norm": float(np.linalg.norm(curvature)),
            "spectral_scale": float(scale),
        }
    )
    return TransformResult("flag_shell_curvature", normalized, metadata)


def _intensity_susceptibility_from_polar(q: np.ndarray) -> np.ndarray:
    tall, _ = _as_tall(q)
    ambient, _ = tall.shape
    connected = tall * tall - 1.0 / float(ambient)
    return connected.T @ connected


def haar_susceptibility_mean(
    shape: tuple[int, int],
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Monte Carlo Haar mean of the quartic intensity susceptibility."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    rng = np.random.default_rng(seed)
    accumulator = None
    for _ in range(int(samples)):
        susceptibility = _intensity_susceptibility_from_polar(
            random_polar(shape, rng)
        )
        accumulator = (
            susceptibility.copy()
            if accumulator is None
            else accumulator + susceptibility
        )
    assert accumulator is not None
    return accumulator / float(samples)


def haar_connected_susceptibility(
    matrix: np.ndarray,
    *,
    haar_mean: np.ndarray,
) -> TransformResult:
    """Signed square root of the Haar-connected quartic susceptibility."""
    susceptibility = _intensity_susceptibility_from_polar(polar_factor(matrix))
    if susceptibility.shape != haar_mean.shape:
        raise ValueError(
            f"Haar mean shape {haar_mean.shape} != {susceptibility.shape}"
        )
    connected = susceptibility - np.asarray(haar_mean, dtype=np.float64)
    field = _signed_sqrt_symmetric(connected)
    normalized, scale = spectral_unit_mean(field)
    values = np.linalg.eigvalsh(0.5 * (connected + connected.T))
    return TransformResult(
        "haar_connected_susceptibility",
        normalized,
        {
            "connected_positive_modes": int(np.count_nonzero(values > 0.0)),
            "connected_negative_modes": int(np.count_nonzero(values < 0.0)),
            "connected_operator_norm": float(np.max(np.abs(values))),
            "spectral_scale": float(scale),
        },
    )


def diffusion_green(
    matrix: np.ndarray,
    *,
    mass: float = 0.05,
) -> TransformResult:
    """Connected Green susceptibility of angular-intensity transport."""
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    tall, _ = _as_tall(polar_factor(matrix))
    intensity = tall * tall
    affinity = intensity.T @ intensity
    degrees = np.sum(affinity, axis=1)
    inverse_sqrt = 1.0 / np.sqrt(np.clip(degrees, 1e-15, None))
    transfer = (inverse_sqrt[:, None] * affinity) * inverse_sqrt[None, :]
    transfer = 0.5 * (transfer + transfer.T)

    values, vectors = np.linalg.eigh(transfer)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, 1.0)
    vectors = vectors[:, order]

    perfect_mixing = 1.0 / math.sqrt(1.0 + mass)
    green_values = 1.0 / np.sqrt(1.0 - values + mass) - perfect_mixing
    if green_values.size:
        green_values[0] = 0.0
    field = (vectors * green_values) @ vectors.T
    field = 0.5 * (field + field.T)
    normalized, scale = spectral_unit_mean(field)
    return TransformResult(
        "diffusion_green",
        normalized,
        {
            "diffusion_mass": float(mass),
            "transfer_second_eigenvalue": float(values[1])
            if values.size > 1
            else float("nan"),
            "transfer_gap": float(1.0 - values[1])
            if values.size > 1
            else float("nan"),
            "spectral_scale": float(scale),
        },
    )


def largest_dyadic_at_most(value: int, maximum: int) -> int:
    value = min(int(value), int(maximum))
    if value <= 0:
        return 0
    return 1 << int(math.floor(math.log2(value)))


def temporal_drift(
    matrices: list[np.ndarray],
    index: int,
    *,
    maximum_block: int = 8,
    randomized: bool = False,
    seed: int = 0,
) -> TransformResult:
    """Block-spin sum of consecutive Stiefel gauge increments.

    The ``1/sqrt(b)`` normalization keeps incoherent Brownian motion at order
    one. The randomized control independently permutes and sign-flips each
    increment, preserving its singular values while destroying alignment.
    """
    if index <= 0:
        return TransformResult(
            "temporal_drift",
            np.zeros_like(np.asarray(matrices[0], dtype=np.float64)),
            {
                "temporal_block": 0,
                "temporal_coherence": 0.0,
                "randomized_control": bool(randomized),
            },
        )

    block = largest_dyadic_at_most(index, maximum_block)
    start = index - block + 1
    increments: list[np.ndarray] = []
    for current in range(start, index + 1):
        result = stiefel_log_gauge(matrices[current - 1], matrices[current])
        increment = np.asarray(result.matrix, dtype=np.float64)
        scale = float(result.metadata.get("spectral_scale", 0.0))
        if scale > 0.0:
            increment = increment / scale
        increments.append(increment)

    if randomized:
        rng = np.random.default_rng(seed + 1009 * index)
        scrambled = []
        for increment in increments:
            row_order = rng.permutation(increment.shape[0])
            column_order = rng.permutation(increment.shape[1])
            row_signs = rng.choice((-1.0, 1.0), size=increment.shape[0])
            column_signs = rng.choice((-1.0, 1.0), size=increment.shape[1])
            scrambled.append(
                row_signs[:, None]
                * increment[row_order][:, column_order]
                * column_signs[None, :]
            )
        increments = scrambled

    summed = np.sum(increments, axis=0)
    incoherent_norm = math.sqrt(
        max(
            float(sum(np.linalg.norm(item) ** 2 for item in increments)),
            1e-30,
        )
    )
    coherence = float(np.linalg.norm(summed) / incoherent_norm)
    field = summed / math.sqrt(block)
    normalized, scale = spectral_unit_mean(field)
    return TransformResult(
        "temporal_drift",
        normalized,
        {
            "temporal_block": int(block),
            "temporal_coherence": coherence,
            "randomized_control": bool(randomized),
            "spectral_scale": float(scale),
        },
    )


def build_transform(
    method: str,
    matrices: list[np.ndarray],
    index: int,
    *,
    haar_mean: np.ndarray | None = None,
    diffusion_mass: float = 0.05,
    temporal_max_block: int = 8,
    seed: int = 0,
) -> TransformResult:
    """Build one actual transformed matrix at one checkpoint."""
    matrix = np.asarray(matrices[index], dtype=np.float64)
    initial = np.asarray(matrices[0], dtype=np.float64)
    if method == "raw":
        return TransformResult("raw", matrix.copy(), {})
    if method == "polar_intensity":
        return polar_intensity(matrix)
    if method == "stiefel_log_gauge":
        return stiefel_log_gauge(initial, matrix)
    if method == "flag_shell_curvature":
        return flag_shell_curvature(initial, matrix)
    if method == "haar_connected_susceptibility":
        if haar_mean is None:
            raise ValueError("haar_mean is required")
        return haar_connected_susceptibility(matrix, haar_mean=haar_mean)
    if method == "diffusion_green":
        return diffusion_green(matrix, mass=diffusion_mass)
    if method == "temporal_drift":
        return temporal_drift(
            matrices,
            index,
            maximum_block=temporal_max_block,
            seed=seed,
        )
    raise KeyError(f"Unknown method: {method}")


def method_matched_null(
    method: str,
    matrices: list[np.ndarray],
    index: int,
    *,
    rng: np.random.Generator,
    haar_mean: np.ndarray | None = None,
    diffusion_mass: float = 0.05,
    temporal_max_block: int = 8,
    seed: int = 0,
) -> TransformResult:
    """Build a null under exactly the same nonlinear observation map."""
    shape = matrices[index].shape
    initial = np.asarray(matrices[0], dtype=np.float64)
    if method == "raw":
        sigma = float(np.std(matrices[index]))
        return TransformResult(
            "raw",
            rng.normal(0.0, sigma, size=shape),
            {"null": "matched_gaussian"},
        )
    if method == "temporal_drift":
        return temporal_drift(
            matrices,
            index,
            maximum_block=temporal_max_block,
            randomized=True,
            seed=seed,
        )

    random_matrix = random_polar(shape, rng)
    radial_scale = max(
        float(np.mean(np.linalg.svd(initial, compute_uv=False))),
        1e-12,
    )
    random_matrix = radial_scale * random_matrix
    if method == "polar_intensity":
        return polar_intensity(random_matrix)
    if method == "stiefel_log_gauge":
        return stiefel_log_gauge(initial, random_matrix)
    if method == "flag_shell_curvature":
        return flag_shell_curvature(initial, random_matrix)
    if method == "haar_connected_susceptibility":
        if haar_mean is None:
            raise ValueError("haar_mean is required")
        return haar_connected_susceptibility(random_matrix, haar_mean=haar_mean)
    if method == "diffusion_green":
        return diffusion_green(random_matrix, mass=diffusion_mass)
    raise KeyError(f"Unknown method: {method}")
