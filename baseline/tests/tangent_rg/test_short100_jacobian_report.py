from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


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


def test_method_coverage_requires_ecs_on_fc1_and_fc2_but_not_fc3():
    module = load_report_module()
    rows = []
    for optimizer in module.OPTIMIZERS:
        for layer, methods in module.EXPECTED_METHODS_BY_LAYER.items():
            for method in methods:
                for epoch in range(10, 101, 10):
                    rows.append({
                        "optimizer": optimizer, "layer": layer,
                        "method": method, "epoch": epoch,
                    })
    coverage = module.build_method_coverage(pd.DataFrame(rows))
    assert coverage["coverage_status"].eq("complete").all()
    assert len(coverage) == sum(
        len(methods) for methods in module.EXPECTED_METHODS_BY_LAYER.values()
    ) * len(module.OPTIMIZERS)
    fc2 = coverage[coverage["layer"].eq("fc2.weight")]
    fc3 = coverage[coverage["layer"].eq("fc3.weight")]
    assert set(fc2["method"]) == set(module.EXPECTED_METHODS_BY_LAYER["fc2.weight"])
    assert set(fc3["method"]) == {"centered_log_singular_radial_pullback"}


def test_end_to_end_contract_counts_every_alpha_family():
    module = load_report_module()
    primary = pd.DataFrame([
        {"optimizer": optimizer, "layer": layer, "epoch": epoch, "method": method}
        for optimizer in module.OPTIMIZERS
        for layer, methods in module.EXPECTED_METHODS_BY_LAYER.items()
        for epoch in module.ANALYSIS_EPOCHS
        for method in methods
    ])
    search = pd.DataFrame([
        {
            "optimizer": optimizer, "layer": layer, "epoch": epoch,
            "tikhonov_z_boundary_ratio": ratio,
            "selected": ratio == module.TIKHONOV_Z_RATIOS[0],
        }
        for optimizer in module.OPTIMIZERS
        for layer in ("fc1.weight", "fc2.weight")
        for epoch in module.ANALYSIS_EPOCHS
        for ratio in module.TIKHONOV_Z_RATIOS
    ])
    quotient = pd.DataFrame([
        {
            "optimizer": optimizer, "layer": layer, "epoch": epoch,
            "method": method, "profile_id": profile, "fit_variant": variant,
        }
        for optimizer in module.OPTIMIZERS
        for layer in ("fc1.weight", "fc2.weight")
        for epoch in module.ANALYSIS_EPOCHS
        for method, profile in module.QUOTIENT_EXPECTED_PROFILES
        for variant in ("raw", "clip_xmax")
    ])
    flow = pd.DataFrame([
        {
            "optimizer": optimizer, "layer": layer, "epoch_end": epoch,
            "method": method, "spectrum_kind": "energy_derived_from_amplitude",
            "clip_top_k": 0,
        }
        for optimizer in module.OPTIMIZERS
        for layer in ("fc1.weight", "fc2.weight")
        for epoch in module.ANALYSIS_EPOCHS[1:]
        for method in module.FLOW_LABELS
    ])
    transport = pd.DataFrame([
        {"optimizer": optimizer, "layer": layer, "epoch_end": epoch}
        for optimizer in module.OPTIMIZERS
        for layer in ("fc1.weight", "fc2.weight")
        for epoch in module.ANALYSIS_EPOCHS[1:]
    ])
    coverage = module.build_analysis_contract_coverage(
        primary, search, quotient, flow, transport
    )
    assert coverage["coverage_status"].eq("complete").all()
    assert coverage["missing_unit_count"].eq(0).all()
