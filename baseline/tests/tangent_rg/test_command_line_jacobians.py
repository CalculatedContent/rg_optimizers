from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[2]
    / "experiments"
    / "mnist_mlp3_tangent_rg"
    / "scripts"
    / "run_short100_jacobians_cli.py"
)


def load_cli_module():
    spec = importlib.util.spec_from_file_location("short100_jacobian_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_computes_all_five_base_jacobian_spectra():
    module = load_cli_module()
    rng = np.random.default_rng(123)
    weight = rng.normal(size=(6, 9))
    spectra = module.compute_base_spectra(weight, ns_steps=5, ns_eps=1e-7)
    assert set(spectra) == set(module.BASE_METHODS)
    for amplitudes, record in spectra.values():
        assert amplitudes.ndim == 1
        assert amplitudes.size > 0
        assert np.all(np.isfinite(amplitudes))
        assert np.all(amplitudes > 0)
        assert record.derivative_rank == amplitudes.size


def test_cli_duration_and_argument_defaults_are_observable_tmp_paths():
    module = load_cli_module()
    assert module.format_duration(3661.2) == "1h 01m 01s"
    args = module.build_parser().parse_args([])
    assert str(args.output_root).startswith("/private/tmp/")
    assert args.epoch_stride == 10
    assert args.top_k == "0"
