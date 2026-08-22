from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = (
    Path(__file__).parents[2]
    / "experiments"
    / "mnist_mlp3_tangent_rg"
    / "scripts"
    / "run_short100_jacobians_cli.py"
)
EXPERIMENT_SCRIPTS = SCRIPT.parent


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
    assert args.ecs_layers == "fc1.weight,fc2.weight"
    assert args.ecs_rcond == 1e-9
    assert args.extended_ecs_jacobians is False


def test_ecs_group_compression_removes_only_uniform_coordinate_copies():
    module = load_cli_module()
    record = SimpleNamespace(
        deterministic_shell_multiplicity=4,
        retained_singular_values=np.array([2.0, 4.0, 8.0]),
        singular_amplitudes=np.repeat(np.array([1.0, 0.5, 0.25]), 4),
    )
    compressed, metadata = module.ecs_fit_amplitudes(record, compress_groups=True)
    assert np.allclose(compressed, [1.0, 0.5, 0.25])
    assert metadata["ecs_groups_compressed"] is True
    assert metadata["ecs_uniform_group_multiplicity"] == 4
    assert metadata["ecs_expanded_mode_count"] == 12


def test_mp_median_and_optimal_signal_rank_are_auditable():
    module = load_cli_module()
    median = module.marchenko_pastur_median(1.0)
    assert 0.60 < median < 0.75
    singular = np.array([8.0, 5.0, 1.2, 1.1, 1.0, 0.9])
    rank, metadata = module.optimal_mp_signal_rank(
        singular, (6, 6), numerical_rank=6
    )
    assert 2 <= rank < 6
    assert metadata["mp_raw_signal_rank"] >= 0
    assert metadata["mp_selected_signal_rank"] == rank
    assert metadata["mp_selection_is_frozen_during_differentiation"] is True


def test_tikhonov_grid_selection_uses_ks_not_alpha_target(monkeypatch):
    module = load_cli_module()

    def fake_fit(_amplitudes, _record, metadata, _top_k, _minimum_tail):
        ratio = float(metadata["tikhonov_z_boundary_ratio"])
        return [{
            "spectrum_kind": "energy_derived_from_amplitude",
            "clip_top_k": 0,
            "alpha": 2.0 if ratio == 0.1 else 7.0,
            "ks_D": 0.20 if ratio == 0.1 else 0.05,
            "xmin": 1.0,
            "n_tail": 20,
            "tail_decades": 1.0,
            "fit_ok": True,
        }]

    monkeypatch.setattr(module, "fit_spectrum", fake_fit)
    candidates = {
        "near_two": (
            np.array([1.0, 2.0]), SimpleNamespace(),
            {"tikhonov_z_boundary_ratio": 0.1, "tikhonov_candidate_index": 0},
        ),
        "better_ks": (
            np.array([1.0, 2.0]), SimpleNamespace(),
            {"tikhonov_z_boundary_ratio": 1.0, "tikhonov_candidate_index": 1},
        ),
    }
    selected_id, _, rows = module.select_tikhonov_candidate(
        candidates, base_metadata={}, minimum_tail=2
    )
    assert selected_id == "better_ks"
    assert next(row for row in rows if row["selected"])["alpha"] == 7.0
    assert "alpha proximity to 2 is never used" in rows[0]["selection_rule"]


def test_complete_cli_separates_state_flow_and_local_jacobian_claims():
    source = (EXPERIMENT_SCRIPTS / "run_short100_quotient_flow_cli.py").read_text()
    wrapper = (EXPERIMENT_SCRIPTS / "run_short100_complete_rg_analysis.sh").read_text()
    report = (EXPERIMENT_SCRIPTS / "build_short100_jacobian_report.py").read_text()
    assert "analyze_weightwatcher_dual" in source
    assert '"gram_ridge"' in source
    assert '"feshbach_downfolding"' in source
    assert '"calibrated_mp_shrinker"' in source
    assert '"is_training_jacobian": False' in source
    assert "run_short100_quotient_flow_cli.py" in wrapper
    assert "Case 1 — heavy tails on a weight quotient representative" in report
    assert "Case 2a — flow between checkpoints" in report
    assert "Case 2b — a Jacobian at one checkpoint" in report
    assert "Tikhonov grid" in report
    assert "Optimal MP space" in report
