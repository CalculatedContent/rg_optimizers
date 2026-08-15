from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import rg_nanogpt_one_head.muonclip_angular_order_parameters as module


def random_matrix(shape=(32, 32), seed=1):
    return np.random.default_rng(seed).normal(size=shape)


def test_method_inventory():
    assert module.METHOD_ORDER == (
        "raw",
        "polar_intensity",
        "stiefel_log_gauge",
        "flag_shell_curvature",
        "haar_connected_susceptibility",
        "diffusion_green",
        "temporal_drift",
    )


def test_polar_factor_square_and_rectangular():
    for shape in ((32, 32), (48, 24), (24, 48)):
        q = module.polar_factor(random_matrix(shape, sum(shape)))
        if shape[0] >= shape[1]:
            assert np.allclose(q.T @ q, np.eye(shape[1]), atol=1e-9)
        else:
            assert np.allclose(q @ q.T, np.eye(shape[0]), atol=1e-9)


def test_random_polar_square_and_rectangular():
    rng = np.random.default_rng(3)
    for shape in ((32, 32), (48, 24), (24, 48)):
        q = module.random_polar(shape, rng)
        assert q.shape == shape
        if shape[0] >= shape[1]:
            assert np.allclose(q.T @ q, np.eye(shape[1]), atol=1e-9)
        else:
            assert np.allclose(q @ q.T, np.eye(shape[0]), atol=1e-9)


def test_unit_mean_gauge():
    matrix, scale = module.spectral_unit_mean(random_matrix((40, 20), 4))
    singular = np.linalg.svd(matrix, compute_uv=False)
    assert scale > 0.0
    assert np.isclose(np.mean(singular**2), 1.0, rtol=1e-10)


def test_polar_intensity_is_nontrivial():
    result = module.polar_intensity(random_matrix((32, 32), 5))
    assert result.matrix.shape == (32, 32)
    assert np.isfinite(result.matrix).all()
    assert np.linalg.norm(result.matrix) > 0.0


def test_stiefel_log_zero_for_identity_and_nonzero_for_rotation():
    matrix = random_matrix((32, 32), 6)
    identical = module.stiefel_log_gauge(matrix, matrix)
    assert np.linalg.norm(identical.matrix) < 1e-7

    q = module.polar_factor(matrix)
    theta = 0.2
    rotation = np.eye(32)
    rotation[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
    moved = module.stiefel_log_gauge(matrix, q @ rotation)
    assert np.linalg.norm(moved.matrix) > 0.1


def test_flag_curvature_zero_for_identical():
    matrix = random_matrix((48, 24), 7)
    result = module.flag_shell_curvature(matrix, matrix)
    assert result.matrix.shape == (24, 24)
    assert np.linalg.norm(result.matrix) < 1e-8
    shells = module.dyadic_shell_slices(24)
    assert sum(item.stop - item.start for item in shells) == 24


def test_connected_susceptibility_finite():
    shape = (48, 24)
    mean = module.haar_susceptibility_mean(shape, samples=12, seed=8)
    result = module.haar_connected_susceptibility(
        random_matrix(shape, 9),
        haar_mean=mean,
    )
    assert result.matrix.shape == (24, 24)
    assert np.isfinite(result.matrix).all()


def test_diffusion_green_finite():
    result = module.diffusion_green(random_matrix((48, 24), 10), mass=0.1)
    assert result.matrix.shape == (24, 24)
    assert np.isfinite(result.matrix).all()
    assert np.linalg.norm(result.matrix) > 0.0


def test_temporal_drift_and_scrambled_null():
    rng = np.random.default_rng(11)
    matrices = []
    current = rng.normal(size=(32, 32))
    for _ in range(5):
        matrices.append(current.copy())
        current = current + 0.01 * rng.normal(size=current.shape)
    actual = module.temporal_drift(matrices, 4, maximum_block=4)
    null = module.temporal_drift(
        matrices,
        4,
        maximum_block=4,
        randomized=True,
        seed=12,
    )
    assert actual.matrix.shape == matrices[0].shape
    assert null.matrix.shape == matrices[0].shape
    assert np.isfinite(actual.matrix).all()
    assert np.isfinite(null.matrix).all()


def test_dispatch_all_methods():
    rng = np.random.default_rng(13)
    matrices = [rng.normal(size=(32, 32)) for _ in range(3)]
    haar_mean = module.haar_susceptibility_mean(
        (32, 32), samples=10, seed=14
    )
    for method in module.METHOD_ORDER:
        result = module.build_transform(
            method,
            matrices,
            2,
            haar_mean=haar_mean,
            seed=15,
        )
        assert result.matrix.ndim == 2
        assert np.isfinite(result.matrix).all()


def test_notebook_smoke_contract():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "angular"
        / "10_muonclip_five_angular_quotient_order_parameters.ipynb"
    )
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = []
    for index, cell in enumerate(payload["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"angular_order_parameter_cell_{index}", "exec")
    full = "\n".join(sources)
    for required in (
        "stiefel_log_gauge",
        "flag_shell_curvature",
        "haar_connected_susceptibility",
        "diffusion_green",
        "temporal_drift",
        "watcher.analyze",
        "plot=True",
        "randomize=True",
        'fix_fingers="clip_xmax"',
        "Native WeightWatcher ESD contact sheets",
        "alpha=2",
    ):
        assert required in full
