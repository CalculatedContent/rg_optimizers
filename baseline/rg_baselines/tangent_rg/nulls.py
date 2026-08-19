"""Matched scale, rotation, Gaussian, and Haar null generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NullSampleRecord:
    sample: FloatArray
    null_kind: str
    preserved_frobenius_norm: bool
    operator_kind: str
    map_definition: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InvarianceCaseRecord:
    name: str
    passed: bool
    absolute_error: float
    relative_error: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class InvarianceCheckRecord:
    cases: tuple[InvarianceCaseRecord, ...]
    all_passed: bool
    operator_kind: str
    map_definition: str


def _rng(value: np.random.Generator | int | None) -> np.random.Generator:
    return value if isinstance(value, np.random.Generator) else np.random.default_rng(value)


def _matrix(value: ArrayLike) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("matrix must be finite, two-dimensional, and non-empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains non-finite values")
    return matrix


def haar_orthogonal(
    dimension: int,
    rng: np.random.Generator | int | None = None,
    *,
    proper: bool = False,
) -> FloatArray:
    """Generate a Haar orthogonal matrix using sign-corrected QR."""

    if int(dimension) < 1:
        raise ValueError("dimension must be positive")
    generator = _rng(rng)
    q, triangular = np.linalg.qr(
        generator.normal(size=(int(dimension), int(dimension)))
    )
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    if proper and np.linalg.det(q) < 0.0:
        q[:, -1] *= -1.0
    return q


def haar_stiefel(
    rows: int,
    columns: int,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Generate a Haar column-Stiefel matrix with ``Q.T Q=I``."""

    if int(rows) < int(columns) or int(columns) < 1:
        raise ValueError("haar_stiefel requires rows >= columns >= 1")
    generator = _rng(rng)
    q, triangular = np.linalg.qr(
        generator.normal(size=(int(rows), int(columns))),
        mode="reduced",
    )
    return q * np.where(np.diag(triangular) < 0.0, -1.0, 1.0)[None, :]


def scale_null(
    matrix: ArrayLike,
    rng: np.random.Generator | int | None = None,
    *,
    scale: float | None = None,
    log_scale_standard_deviation: float = 1.0,
) -> NullSampleRecord:
    """Apply a positive global scale, useful for testing scale invariance."""

    work = _matrix(matrix)
    if scale is None:
        if log_scale_standard_deviation < 0.0:
            raise ValueError("log_scale_standard_deviation must be non-negative")
        scale = float(
            np.exp(_rng(rng).normal(scale=float(log_scale_standard_deviation)))
        )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return NullSampleRecord(
        sample=float(scale) * work,
        null_kind="global_scale",
        preserved_frobenius_norm=bool(np.isclose(scale, 1.0)),
        operator_kind="positive_global_scale_null",
        map_definition="W_null=c W with c>0",
        metadata={"scale": float(scale)},
    )


def rotation_null(
    matrix: ArrayLike,
    rng: np.random.Generator | int | None = None,
    *,
    side: str = "both",
    proper: bool = False,
) -> NullSampleRecord:
    """Apply independent Haar rotations while preserving singular values."""

    work = _matrix(matrix)
    generator = _rng(rng)
    selected = str(side).lower()
    if selected not in {"left", "right", "both"}:
        raise ValueError("side must be left, right, or both")
    left = (
        haar_orthogonal(work.shape[0], generator, proper=proper)
        if selected in {"left", "both"}
        else np.eye(work.shape[0])
    )
    right = (
        haar_orthogonal(work.shape[1], generator, proper=proper)
        if selected in {"right", "both"}
        else np.eye(work.shape[1])
    )
    sample = left @ work @ right.T
    return NullSampleRecord(
        sample=sample,
        null_kind=f"haar_{selected}_rotation",
        preserved_frobenius_norm=True,
        operator_kind="singular_value_preserving_haar_rotation_null",
        map_definition="W_null=L W R^T with selected Haar orthogonal L/R",
        metadata={"side": selected, "left": left, "right": right},
    )


def gaussian_null(
    matrix: ArrayLike,
    rng: np.random.Generator | int | None = None,
    *,
    match: str = "mean_std",
) -> NullSampleRecord:
    """Generate an iid Gaussian entry null with explicit matching convention."""

    work = _matrix(matrix)
    generator = _rng(rng)
    selected = str(match).lower()
    if selected == "mean_std":
        mean = float(np.mean(work))
        std = float(np.std(work))
        sample = generator.normal(mean, std, size=work.shape)
    elif selected in {"frobenius", "frobenius_zero_mean"}:
        mean = 0.0
        raw = generator.normal(size=work.shape)
        norm = float(np.linalg.norm(raw, ord="fro"))
        target = float(np.linalg.norm(work, ord="fro"))
        sample = raw * (target / max(norm, np.finfo(np.float64).tiny))
        std = float(np.std(sample))
    else:
        raise ValueError("match must be mean_std or frobenius")
    return NullSampleRecord(
        sample=np.asarray(sample, dtype=np.float64),
        null_kind="iid_gaussian",
        preserved_frobenius_norm=selected.startswith("frobenius"),
        operator_kind="iid_gaussian_entry_null",
        map_definition=(
            "iid Gaussian entries matched by mean/std or exact Frobenius norm"
        ),
        metadata={"match": selected, "mean": mean, "std": std},
    )


def haar_polar_null(
    shape: tuple[int, int],
    rng: np.random.Generator | int | None = None,
) -> NullSampleRecord:
    """Generate a Haar partial isometry with the requested rectangular shape."""

    rows, columns = (int(shape[0]), int(shape[1]))
    if rows < 1 or columns < 1:
        raise ValueError("shape entries must be positive")
    generator = _rng(rng)
    sample = (
        haar_stiefel(rows, columns, generator)
        if rows >= columns
        else haar_stiefel(columns, rows, generator).T
    )
    return NullSampleRecord(
        sample=sample,
        null_kind="haar_rectangular_polar",
        preserved_frobenius_norm=False,
        operator_kind="haar_stiefel_partial_isometry_null",
        map_definition=(
            "Haar column-Stiefel Q for tall shapes and its transpose for wide shapes"
        ),
        metadata={"shape": (rows, columns)},
    )


def _observable_array(
    observable: Callable[[FloatArray], Any],
    matrix: FloatArray,
    value_fn: Callable[[Any], ArrayLike] | None,
) -> FloatArray:
    value = observable(matrix)
    if value_fn is not None:
        value = value_fn(value)
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("observable produced non-finite values")
    return array


def _case(
    name: str,
    reference: FloatArray,
    transformed: FloatArray,
    *,
    atol: float,
    rtol: float,
    definition: str,
) -> InvarianceCaseRecord:
    if transformed.shape != reference.shape:
        return InvarianceCaseRecord(
            name=name,
            passed=False,
            absolute_error=np.inf,
            relative_error=np.inf,
            operator_kind="failed_observable_invariance_check",
            map_definition=definition,
        )
    absolute = float(np.linalg.norm(transformed - reference))
    relative = absolute / max(
        float(np.linalg.norm(reference)), np.finfo(np.float64).tiny
    )
    return InvarianceCaseRecord(
        name=name,
        passed=bool(np.allclose(transformed, reference, atol=atol, rtol=rtol)),
        absolute_error=absolute,
        relative_error=relative,
        operator_kind="observable_invariance_check",
        map_definition=definition,
    )


def check_invariances(
    observable: Callable[[FloatArray], Any],
    matrix: ArrayLike,
    *,
    rng: np.random.Generator | int | None = 0,
    value_fn: Callable[[Any], ArrayLike] | None = None,
    scale: float = 1.7,
    atol: float = 1e-9,
    rtol: float = 1e-7,
) -> InvarianceCheckRecord:
    """Check global-scale and bi-orthogonal invariance of an observable."""

    work = _matrix(matrix)
    generator = _rng(rng)
    reference = _observable_array(observable, work, value_fn)
    scaled = _observable_array(observable, float(scale) * work, value_fn)
    left = haar_orthogonal(work.shape[0], generator)
    right = haar_orthogonal(work.shape[1], generator)
    rotated = _observable_array(observable, left @ work @ right.T, value_fn)
    cases = (
        _case(
            "positive_global_scale",
            reference,
            scaled,
            atol=atol,
            rtol=rtol,
            definition="O(cW)=O(W) for c>0",
        ),
        _case(
            "left_right_orthogonal",
            reference,
            rotated,
            atol=atol,
            rtol=rtol,
            definition="O(L W R^T)=O(W) for orthogonal L,R",
        ),
    )
    return InvarianceCheckRecord(
        cases=cases,
        all_passed=all(case.passed for case in cases),
        operator_kind="scale_and_orthogonal_invariance_audit",
        map_definition=(
            "numerical allclose audit under positive scaling and Haar left/right rotations"
        ),
    )


__all__ = [
    "InvarianceCaseRecord",
    "InvarianceCheckRecord",
    "NullSampleRecord",
    "check_invariances",
    "gaussian_null",
    "haar_orthogonal",
    "haar_polar_null",
    "haar_stiefel",
    "rotation_null",
    "scale_null",
]
