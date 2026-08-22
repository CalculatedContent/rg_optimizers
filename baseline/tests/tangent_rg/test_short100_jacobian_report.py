from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[2]
    / "experiments"
    / "mnist_mlp3_tangent_rg"
    / "scripts"
    / "build_short100_jacobian_report.py"
)


def load_report_module():
    spec = importlib.util.spec_from_file_location("short100_jacobian_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_empirical_ccdf_is_sorted_and_normalized():
    module = load_report_module()
    x, y = module.empirical_ccdf([4.0, 1.0, 2.0])
    assert np.array_equal(x, [1.0, 2.0, 4.0])
    assert np.allclose(y, [1.0, 2.0 / 3.0, 1.0 / 3.0])


def test_report_defaults_to_reduced_tmp_output():
    module = load_report_module()
    args = module.build_parser().parse_args([])
    assert str(args.analysis_root) == (
        "/private/tmp/rg-mnist-mlp3-short100-jacobians-reduced"
    )
    assert args.seed == 101
