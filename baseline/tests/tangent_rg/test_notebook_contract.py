"""Dependency-free contract checks for the generated tangent-RG notebooks."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


BASELINE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = BASELINE_ROOT / "experiments" / "mnist_mlp3_tangent_rg"
NOTEBOOK_ROOT = EXPERIMENT_ROOT / "notebooks"
BUILDER_PATH = EXPERIMENT_ROOT / "scripts" / "build_notebooks.py"
MUONCLIP_RUNNER_PATH = (
    EXPERIMENT_ROOT / "scripts" / "run_muonclip_jacobians.sh"
)
SHORT100_RUNNER_PATH = (
    EXPERIMENT_ROOT / "scripts" / "run_short100_quotients_jacobians.sh"
)

EXPECTED_NOTEBOOKS = {
    "00_Protocol_and_Smoke.ipynb",
    "01_Long_Horizon_AdamW.ipynb",
    "02_Long_Horizon_Muon.ipynb",
    "03_Long_Horizon_MuonClip_RMS.ipynb",
    "04_Fixed_Point_Comparison.ipynb",
    "10_Two_Checkpoint_Finite_Flow.ipynb",
    "11_Muon_Update_Stiefel_Tangent.ipynb",
    "12_Radial_Angular_Quotients.ipynb",
    "13_Single_Checkpoint_Map_Jacobians.ipynb",
    "14_Calibrated_Local_Training_Map.ipynb",
    "15_Method_Nulls_Stability_Comparison.ipynb",
    "16_Additional_Weight_Only_ECS_Jacobians.ipynb",
    "17_Data_Dependent_ECS_Jacobians.ipynb",
    "18_Single_Run_MuonClip_Jacobian_Audit.ipynb",
    "19_One_Seed_Muon_MuonClip_Weight_Quotients.ipynb",
    "20_Three_Seed_Muon_MuonClip_Weight_Quotients.ipynb",
    "21_Single_Run_Metrics_and_WeightWatcher_Audit.ipynb",
    "22_MuonClip_AdamW_10Seed_Bollinger_Comparison.ipynb",
    "23_Short100_10Seed_Weight_Quotients.ipynb",
}

ANALYSIS_NOTEBOOKS = EXPECTED_NOTEBOOKS - {
    "00_Protocol_and_Smoke.ipynb",
    "01_Long_Horizon_AdamW.ipynb",
    "02_Long_Horizon_Muon.ipynb",
    "03_Long_Horizon_MuonClip_RMS.ipynb",
}

TAIL_CACHE_ANALYSIS_NOTEBOOKS = {
    "10_Two_Checkpoint_Finite_Flow.ipynb",
    "12_Radial_Angular_Quotients.ipynb",
    "13_Single_Checkpoint_Map_Jacobians.ipynb",
    "15_Method_Nulls_Stability_Comparison.ipynb",
    "16_Additional_Weight_Only_ECS_Jacobians.ipynb",
    "19_One_Seed_Muon_MuonClip_Weight_Quotients.ipynb",
    "20_Three_Seed_Muon_MuonClip_Weight_Quotients.ipynb",
    "23_Short100_10Seed_Weight_Quotients.ipynb",
}

CAPTURE_ANALYSIS_NOTEBOOKS = {
    "11_Muon_Update_Stiefel_Tangent.ipynb",
    "14_Calibrated_Local_Training_Map.ipynb",
    "17_Data_Dependent_ECS_Jacobians.ipynb",
}

METHOD_OUTPUT_NOTEBOOKS = TAIL_CACHE_ANALYSIS_NOTEBOOKS | CAPTURE_ANALYSIS_NOTEBOOKS

EXPECTED_METHOD_SOURCE_BY_NOTEBOOK = {
    "10_Two_Checkpoint_Finite_Flow.ipynb": "verified_tail_checkpoint_cache_model_only",
    "11_Muon_Update_Stiefel_Tangent.ipynb": "verified_dense_update_capture",
    "12_Radial_Angular_Quotients.ipynb": "verified_tail_checkpoint_cache_model_only",
    "13_Single_Checkpoint_Map_Jacobians.ipynb": "verified_tail_checkpoint_cache_model_only",
    "14_Calibrated_Local_Training_Map.ipynb": "verified_calibrated_dense_capture",
    "16_Additional_Weight_Only_ECS_Jacobians.ipynb": "verified_tail_checkpoint_cache_plus_exact_sparse_weightwatcher_trace_metrics",
    "17_Data_Dependent_ECS_Jacobians.ipynb": "verified_calibrated_dense_capture_plus_exact_sparse_weightwatcher_trace_metrics",
    "19_One_Seed_Muon_MuonClip_Weight_Quotients.ipynb": "verified_final_100_tail_cache",
    "20_Three_Seed_Muon_MuonClip_Weight_Quotients.ipynb": "verified_final_100_tail_cache",
    "23_Short100_10Seed_Weight_Quotients.ipynb": "verified_final_100_tail_cache",
}

TRAINING_NOTEBOOKS = {
    "00_Protocol_and_Smoke.ipynb",
    "01_Long_Horizon_AdamW.ipynb",
    "02_Long_Horizon_Muon.ipynb",
    "03_Long_Horizon_MuonClip_RMS.ipynb",
}


def _source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "tangent_rg_build_notebooks_contract", BUILDER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load notebook builder: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_single_checkpoint_module():
    path = BASELINE_ROOT / "rg_baselines" / "tangent_rg" / "single_checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "tangent_rg_single_checkpoint_contract", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load single-checkpoint module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NotebookContractTests(unittest.TestCase):
    def test_short100_runner_is_analysis_only_and_complete(self) -> None:
        self.assertTrue(SHORT100_RUNNER_PATH.is_file())
        self.assertNotEqual(SHORT100_RUNNER_PATH.stat().st_mode & 0o111, 0)
        source = SHORT100_RUNNER_PATH.read_text(encoding="utf-8")
        for notebook in (
            "13_Single_Checkpoint_Map_Jacobians.ipynb",
            "16_Additional_Weight_Only_ECS_Jacobians.ipynb",
            "23_Short100_10Seed_Weight_Quotients.ipynb",
        ):
            self.assertEqual(source.count(f'run_notebook "{notebook}"'), 1)
        for forbidden in (
            "11_Muon_Update_Stiefel_Tangent.ipynb",
            "14_Calibrated_Local_Training_Map.ipynb",
            "17_Data_Dependent_ECS_Jacobians.ipynb",
            " rg_baselines.tangent_rg.cli train",
            "$HOME",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("/private/tmp/rg-mnist-mlp3-short100-runs", source)
        self.assertIn("/private/tmp/rg-mnist-mlp3-short100-checkpoints", source)
        self.assertIn('checkpoint_count" != "100"', source)
        self.assertIn("SEEDS=(101)", source)
        self.assertIn('"SEEDS": [101]', source)
        self.assertIn('"ANALYSIS_EPOCH_STRIDE": 10', source)
        self.assertIn('"LIVE_PROGRESS_EVERY_MATRICES": 5', source)
        self.assertIn("--autosave-cell-every 30", source)
        self.assertIn("--log-output", source)

    def test_checkpoint_epoch_stride_selects_exact_positive_multiples(self) -> None:
        import numpy as np

        builder = _load_builder()
        seed_dir = Path("/tmp/contract_stride_seed")
        refs = tuple(
            SimpleNamespace(
                epoch=epoch,
                global_step=epoch * 10,
                path=Path(f"/tmp/analysis_epoch_{epoch:06d}.pt"),
            )
            for epoch in range(0, 101)
        )
        namespace = {
            "Path": Path,
            "np": np,
            "lru_cache": lru_cache,
            "CHECKPOINT_PAYLOAD_CACHE_SIZE": 1,
            "_VERIFIED_TAIL_CACHE_REFS": {seed_dir.resolve(): refs},
        }
        exec(builder.CHECKPOINT_HELPERS, namespace)
        namespace["checkpoint_matrix"] = (
            lambda path, layer: (str(path), str(layer))
        )
        selected = list(namespace["selected_trajectory_matrices"](
            seed_dir,
            layers=("fc1.weight",),
            maximum_checkpoints=100,
            epoch_stride=10,
        ))
        self.assertEqual(
            [int(row[0].epoch) for row in selected],
            list(range(10, 101, 10)),
        )
        self.assertTrue(all("epoch_stride=10" in row[3] for row in selected))
        with self.assertRaises(ValueError):
            list(namespace["selected_trajectory_matrices"](
                seed_dir,
                layers=("fc1.weight",),
                maximum_checkpoints=100,
                epoch_stride=0,
            ))

    def test_muonclip_runner_executes_only_declared_jacobian_notebooks(self) -> None:
        self.assertTrue(MUONCLIP_RUNNER_PATH.is_file())
        self.assertNotEqual(MUONCLIP_RUNNER_PATH.stat().st_mode & 0o111, 0)
        source = MUONCLIP_RUNNER_PATH.read_text(encoding="utf-8")
        expected = {
            "11_Muon_Update_Stiefel_Tangent.ipynb",
            "13_Single_Checkpoint_Map_Jacobians.ipynb",
            "14_Calibrated_Local_Training_Map.ipynb",
            "16_Additional_Weight_Only_ECS_Jacobians.ipynb",
            "17_Data_Dependent_ECS_Jacobians.ipynb",
        }
        for notebook in expected:
            self.assertEqual(source.count(notebook), 1)
        self.assertNotIn("10_Two_Checkpoint_Finite_Flow.ipynb", source)
        self.assertNotIn("12_Radial_Angular_Quotients.ipynb", source)
        self.assertIn('CACHE_ROOT="${RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT:-/tmp/', source)
        self.assertIn('RUN_MODE="--resume"', source)
        self.assertIn('checkpoint_count" != "100"', source)

    def test_single_run_muonclip_audit_is_jacobian_only_and_no_ci(self) -> None:
        path = NOTEBOOK_ROOT / "18_Single_Run_MuonClip_Jacobian_Audit.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(_source_text(cell) for cell in notebook["cells"])
        expected_children = {
            "11_Muon_Update_Stiefel_Tangent.ipynb",
            "13_Single_Checkpoint_Map_Jacobians.ipynb",
            "14_Calibrated_Local_Training_Map.ipynb",
            "16_Additional_Weight_Only_ECS_Jacobians.ipynb",
            "17_Data_Dependent_ECS_Jacobians.ipynb",
        }
        for child in expected_children:
            self.assertEqual(source.count(f'"{child}"'), 1)
        self.assertNotIn('"10_Two_Checkpoint_Finite_Flow.ipynb"', source)
        self.assertNotIn('"12_Radial_Angular_Quotients.ipynb"', source)
        self.assertIn("pm.execute_notebook", source)
        self.assertIn("fit_clipping_sensitivity", source)
        self.assertIn("fix_fingers=clip_xmax", source)
        self.assertIn('UNCERTAINTY_POLICY = "no_seed_error_bars"', source)
        self.assertIn('OPTIMIZER_SLUG = "muonclip_rms"', source)
        self.assertIn("expected_grid =", source)
        self.assertIn("protocol_fingerprint", source)
        self.assertIn("single_run_audit_manifest.json", source)
        self.assertIn("nonzero_eigenvalues_of_J_star_J", source)
        self.assertIn(']["source_seed_dir"]', source)
        self.assertIn("_VERIFIED_TAIL_CACHE_REFS[Path(seed_dir).resolve()]", source)
        self.assertNotIn(
            "validate_run_identity(\n    seed_dir, optimizer_slug=OPTIMIZER_SLUG, seed=SEED",
            source,
        )
        report_plot_source = "\n".join(
            _source_text(cell)
            for cell in notebook["cells"]
            if "jacobian_latest_alpha_heatmap.png" in _source_text(cell)
        )
        self.assertTrue(report_plot_source)
        self.assertNotIn("fill_between", report_plot_source)

    def test_single_run_metrics_weightwatcher_audit_is_lightweight(self) -> None:
        path = NOTEBOOK_ROOT / "21_Single_Run_Metrics_and_WeightWatcher_Audit.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(_source_text(cell) for cell in notebook["cells"])
        for required in (
            "performance_by_analysis_epoch.csv",
            "weightwatcher_fits.csv",
            "test_accuracy_and_loss.png",
            "train_validation_test_context.png",
            "weightwatcher_alpha_raw_vs_clip_xmax.png",
            "weightwatcher_fit_availability_by_checkpoint.csv",
            "sanity_checks.csv",
            "method_provenance.json",
            '"raw"',
            '"clip_xmax"',
            "fix_fingers=clip_xmax",
            "validate_run_identity(",
            "verified_completed_run_saved_metrics_only",
            "final performance row matches completion horizon",
            "one_completed_seed_no_error_bars",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "pm.execute_notebook",
            "torch.load",
            "np.load(",
            "load_verified_tail_checkpoint_refs",
            "run_cli_training(",
            "run_training(",
            "analyze_weightwatcher_dual(",
            "fit_clipping_sensitivity(",
            "fill_between(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("SEEDS = [int(SEED)]", source)
        self.assertLess(
            source.index("SEEDS = [int(SEED)]"),
            source.index("from pathlib import Path"),
        )

    def test_muonclip_adamw_10seed_bollinger_comparison_contract(self) -> None:
        path = NOTEBOOK_ROOT / "22_MuonClip_AdamW_10Seed_Bollinger_Comparison.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(_source_text(cell) for cell in notebook["cells"])
        for required in (
            'OPTIMIZER_SLUGS = ["muonclip_rms", "adamw"]',
            "SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]",
            'PRIMARY_FIT_VARIANT = "clip_xmax"',
            'BAND_STD_MULTIPLIER = 2.0',
            'ACCURACY_Y_MIN = 0.90',
            'ACCURACY_Y_MAX = 1.005',
            "performance_by_analysis_epoch.csv",
            "weightwatcher_fits.csv",
            "train_test_accuracy_bollinger_2sd.png",
            "train_test_loss_bollinger_2sd.png",
            "default_weightwatcher_alpha_by_layer_bollinger_2sd.png",
            "performance_bollinger_summary.csv",
            "default_weightwatcher_alpha_bollinger_summary.csv",
            "default_weightwatcher_fit_availability.csv",
            "per_seed_peak_to_final_degradation.csv",
            "fill_between",
            'performance_long["epoch"].gt(0)',
            'performance_summary["epoch"].gt(0)',
            "xlim=(1, None)",
            "float(ACCURACY_Y_MIN)",
            "float(ACCURACY_Y_MAX)",
            "mean plus or minus two sample",
            "fix_fingers=clip_xmax",
            "test_monitoring_only",
            "validate_run_identity(",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "torch.load",
            "np.load(",
            "run_training(",
            "run_cli_training(",
            "analyze_weightwatcher_dual(",
            "pm.execute_notebook",
        ):
            self.assertNotIn(forbidden, source)


    def test_ecs_exact_rank_join_rejects_inexact_or_malformed_states(self) -> None:
        import numpy as np
        import pandas as pd

        builder = _load_builder()
        namespace = {
            "np": np,
            "pd": pd,
            "Path": Path,
            "single_checkpoint": _load_single_checkpoint_module(),
        }
        exec(builder.ECS_COVER_METRIC_HELPERS, namespace)
        select = namespace["exact_ecs_cover_rank_record"]
        identity = {
            "optimizer": "muon",
            "seed": 1337,
            "protocol_fingerprint": "fingerprint",
            "epoch": 1000,
            "global_step": 430000,
            "layer": "fc1.weight",
            "fit_variant": "clip_xmax",
        }
        fits = pd.DataFrame([
            {
                **identity,
                "fit_ok": True,
                "detX_num": 32,
                "pl_support_rank": 27,
            }
        ])
        traces = pd.DataFrame([
            {
                **identity,
                "qualification_role": "preregistered_independent_fit_support",
                "sensitivity_only": False,
                "certification_eligible": True,
                "support_rank_source": "weightwatcher_backend_xmax_exact_fit_tail",
                "support_rank": 27,
                "support_window_start_descending_zero_based": 2,
                "support_window_end_descending_exclusive": 29,
                "pl_support_rank_before_finger_clip": 27,
                "n_fingers_removed": 2,
            },
            {
                **identity,
                "qualification_role": "same_curve_audit_cannot_certify",
                "sensitivity_only": False,
                "certification_eligible": False,
                "support_rank_source": "weightwatcher_detX",
                "support_rank": 32,
                "support_window_start_descending_zero_based": 0,
                "support_window_end_descending_exclusive": 32,
            },
            {
                **identity,
                "qualification_role": "same_curve_audit_cannot_certify",
                "sensitivity_only": False,
                "certification_eligible": False,
                "support_rank_source": "weightwatcher_midpoint",
                "support_rank": 29,
                "support_window_start_descending_zero_based": 0,
                "support_window_end_descending_exclusive": 29,
            },
        ])

        def evaluate(selected_fits=fits, selected_traces=traces, **overrides):
            arguments = {
                "optimizer_slug": "muon",
                "seed": 1337,
                "epoch": 1000,
                "global_step": 430000,
                "layer": "fc1.weight",
                "maximum_rank": 40,
                "fit_path": "/runs/metrics/weightwatcher_fits.csv",
                "trace_path": "/runs/metrics/trace_log.csv",
            }
            arguments.update(overrides)
            return select(selected_fits, selected_traces, **arguments)

        record = evaluate()
        self.assertTrue(record["ecs_rank_metrics_available"])
        self.assertTrue(record["ecs_full_shell_available"])
        self.assertTrue(record["ecs_detx_shell_available"])
        self.assertEqual(record["retained_rank"], 29)
        self.assertEqual(record["full_shell_outer_rank"], 40)
        self.assertEqual(record["detx_shell_outer_rank"], 32)
        self.assertEqual(record["detx_shell_rank"], 3)
        self.assertEqual(record["k_boundary_mid"], 30)
        self.assertEqual(record["weightwatcher_midpoint_rank"], 29)
        self.assertEqual(record["weightwatcher_pl_support_rank_recorded"], 27)
        self.assertTrue(record["ecs_rank_exact_weightwatcher_state_found"])
        self.assertTrue(record["ecs_rank_exact_trace_state_found"])

        missing = evaluate(epoch=999)
        self.assertFalse(missing["ecs_rank_metrics_available"])
        self.assertFalse(missing["ecs_full_shell_available"])
        self.assertFalse(missing["ecs_detx_shell_available"])
        self.assertFalse(missing["ecs_rank_exact_epoch_match_found"])
        self.assertFalse(missing["ecs_rank_exact_weightwatcher_state_found"])

        with self.assertRaises(RuntimeError):
            evaluate(pd.concat([fits, fits], ignore_index=True), traces)
        detx_mismatch = traces.copy()
        detx_mismatch.loc[
            detx_mismatch["support_rank_source"].eq("weightwatcher_detX"),
            "support_rank",
        ] = 33
        with self.assertRaises(RuntimeError):
            evaluate(fits, detx_mismatch)
        fractional = fits.astype({"detX_num": float})
        fractional.loc[0, "detX_num"] = 32.25
        with self.assertRaises(RuntimeError):
            evaluate(fractional, traces)
        pl_mismatch = fits.copy()
        pl_mismatch.loc[0, "pl_support_rank"] = 28
        with self.assertRaises(RuntimeError):
            evaluate(pl_mismatch, traces)

    def test_inventory_parameters_cleanliness_and_analysis_metadata(self) -> None:
        actual_names = {path.name for path in NOTEBOOK_ROOT.glob("*.ipynb")}
        self.assertEqual(actual_names, EXPECTED_NOTEBOOKS)

        for name in sorted(EXPECTED_NOTEBOOKS):
            with self.subTest(notebook=name):
                notebook = json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
                self.assertEqual(notebook.get("nbformat"), 4)
                self.assertIsInstance(notebook.get("cells"), list)

                code_cells = [
                    cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
                ]
                self.assertTrue(code_cells, "notebook must contain at least one code cell")
                parameter_cells = [
                    cell
                    for cell in code_cells
                    if "parameters" in cell.get("metadata", {}).get("tags", [])
                ]
                self.assertEqual(len(parameter_cells), 1)
                self.assertIs(parameter_cells[0], code_cells[0])
                parameter_source = _source_text(parameter_cells[0])
                self.assertIn("CHECKPOINT_CACHE_ROOT =", parameter_source)

                for cell in code_cells:
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs"), [])

                if name in ANALYSIS_NOTEBOOKS:
                    all_source = "\n".join(_source_text(cell) for cell in notebook["cells"])
                    self.assertIn("operator_kind", all_source)
                    self.assertIn("map_definition", all_source)

                if name == "13_Single_Checkpoint_Map_Jacobians.ipynb":
                    all_source = "\n".join(
                        _source_text(cell) for cell in notebook["cells"]
                    )
                    self.assertIn("qualify_replicated_group_fits", all_source)
                    self.assertIn("ecs_clip_core_groups", all_source)
                    self.assertIn("ecs_tail_core_group_count", all_source)
                    self.assertIn(
                        "shell-dimension/multiplicity sensitivity", all_source
                    )
                    for live_progress_contract in (
                        "LIVE_PROGRESS_EVERY_MATRICES = 50",
                        'live_progress_dir / "status.json"',
                        "epoch_stride=ANALYSIS_EPOCH_STRIDE",
                        "persist_live_seed(",
                        "display_live_optimizer_layer(",
                        "analyze_optimizer_layer_block(",
                        'live_progress_dir / f"{slug}_alpha.png"',
                        "Completed cell output:",
                        '"state": "running_trajectory"',
                        'flush=True',
                    ):
                        self.assertIn(live_progress_contract, all_source)
                    incremental_calls = {
                        _source_text(cell).strip()
                        for cell in code_cells
                        if _source_text(cell).strip().startswith(
                            "analyze_optimizer_layer_block("
                        )
                    }
                    self.assertEqual(
                        incremental_calls,
                        {
                            f'analyze_optimizer_layer_block("{optimizer}", "{layer}")'
                            for optimizer in ("muonclip_rms", "adamw", "muon")
                            for layer in (
                                "fc2.weight", "fc1.weight", "fc3.weight"
                            )
                        },
                    )

                if name == "15_Method_Nulls_Stability_Comparison.ipynb":
                    all_source = "\n".join(
                        _source_text(cell) for cell in notebook["cells"]
                    )
                    self.assertIn("REQUIRED_ECS_PRIMARY_METHOD", all_source)
                    self.assertIn("REQUIRED_ECS_FIT_CONTRACT_TOKEN", all_source)
                    self.assertIn("expected_ecs_primary_grid", all_source)
                    self.assertIn("observed_ecs_primary_grid", all_source)
                    self.assertIn("missing_ecs_contract", all_source)
                    self.assertIn("ecs_group_tail_qualified", all_source)
                    self.assertIn("REQUIRED_PRIMARY_FIT_METHODS", all_source)
                    self.assertIn("expected_all_run_grid", all_source)
                    self.assertIn("observed_muon_step_grid", all_source)

                if name == "16_Additional_Weight_Only_ECS_Jacobians.ipynb":
                    all_source = "\n".join(
                        _source_text(cell) for cell in notebook["cells"]
                    )
                    for function_name in (
                        "gap_aware_projector_spectrum",
                        "soft_ecs_projector_spectrum",
                        "outer_trace_free_log_gram_spectrum",
                        "outer_resolvent_spectrum",
                        "feshbach_trace_free_log_spectrum",
                    ):
                        self.assertIn(function_name, all_source)
                    self.assertIn("central_difference_jacobian", all_source)
                    self.assertIn("energy_convention", all_source)
                    self.assertIn(
                        "additional_weight_only_ecs_jacobians_v1", all_source
                    )

                if name == "17_Data_Dependent_ECS_Jacobians.ipynb":
                    all_source = "\n".join(
                        _source_text(cell) for cell in notebook["cells"]
                    )
                    for function_name in (
                        "input_output_jacobian_spectrum",
                        "grassmann_parameter_output_jacobian",
                        "per_example_quotient_loss_jacobian",
                        "quotient_generalized_gauss_newton",
                        "step_quotient_jacobian_sketch",
                    ):
                        self.assertIn(function_name, all_source)
                    self.assertIn("replay_calibrated_step", all_source)
                    self.assertIn("exact_ecs_cover_rank_record", all_source)
                    self.assertIn("restricted_domain", all_source)
                    self.assertIn(
                        'CAPTURE_ONLY_SOURCE_KIND = "verified_calibrated_dense_capture"',
                        all_source,
                    )
                    self.assertIn("data_dependent_ecs_jacobians_v1", all_source)

    def test_tail_cache_is_the_only_model_checkpoint_analysis_source(self) -> None:
        for name in sorted(TAIL_CACHE_ANALYSIS_NOTEBOOKS):
            with self.subTest(notebook=name):
                notebook = json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
                all_source = "\n".join(
                    _source_text(cell) for cell in notebook["cells"]
                )
                self.assertIn(
                    "/tmp/rg-mnist-mlp3-tangent-checkpoints", all_source
                )
                self.assertIn("load_verified_tail_checkpoint_refs", all_source)
                self.assertIn("expected_fingerprint=", all_source)
                self.assertIn("expected_epochs=expected_epochs", all_source)
                self.assertIn("tail_checkpoint_cache_dir", all_source)
                self.assertIn("epoch_step_grid", all_source)
                self.assertIn(
                    "seed_dir = require_tail_checkpoint_cache(optimizer, seed)",
                    all_source,
                )
                self.assertIn("verified_final_100_tail_cache", all_source)
                self.assertNotIn(
                    "seed_dir = require_complete_seed(optimizer, seed)", all_source
                )
                self.assertNotIn("run_cli_training(", all_source)
                self.assertNotIn("run_training(", all_source)

        for name in sorted(CAPTURE_ANALYSIS_NOTEBOOKS):
            with self.subTest(notebook=name):
                notebook = json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
                all_source = "\n".join(
                    _source_text(cell) for cell in notebook["cells"]
                )
                normalized_source = " ".join(all_source.split())
                self.assertIn(
                    "seed_dir = require_complete_seed(optimizer, seed)", all_source
                )
                self.assertIn("capture_payloads(", all_source)
                self.assertIn("analysis_plan total_steps", all_source)
                self.assertIn("best_validation_epoch", all_source)
                self.assertIn(
                    "never launches or resumes training", normalized_source
                )
                self.assertNotIn("run_cli_training(", all_source)
                self.assertNotIn("run_training(", all_source)

    def test_training_notebooks_write_and_verify_the_tail_cache(self) -> None:
        for name in sorted(TRAINING_NOTEBOOKS):
            with self.subTest(notebook=name):
                notebook = json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
                all_source = "\n".join(
                    _source_text(cell) for cell in notebook["cells"]
                )
                self.assertIn("--tail-checkpoint-root", all_source)
                self.assertIn("require_tail_checkpoint_cache(", all_source)
                self.assertIn(
                    "RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT", all_source
                )

        for name in (
            "10_Two_Checkpoint_Finite_Flow.ipynb",
            "12_Radial_Angular_Quotients.ipynb",
        ):
            notebook = json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
            all_source = "\n".join(_source_text(cell) for cell in notebook["cells"])
            self.assertIn("MAXIMUM_PAIRS = 100", all_source)
            self.assertIn("PAIR_STRIDES = [1]", all_source)
            self.assertIn("SPACING_SENSITIVITY_STRIDES = [2, 4, 8]", all_source)
            self.assertIn("SPACING_SENSITIVITY_PAIRS_PER_STRIDE = 8", all_source)
            self.assertIn("bounded_spacing_sensitivity", all_source)
        notebook = json.loads(
            (NOTEBOOK_ROOT / "13_Single_Checkpoint_Map_Jacobians.ipynb").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(
            "MAXIMUM_CHECKPOINTS = 100",
            "\n".join(_source_text(cell) for cell in notebook["cells"]),
        )

    def test_analysis_method_fingerprint_manifests_are_strict(self) -> None:
        for name in sorted(METHOD_OUTPUT_NOTEBOOKS):
            with self.subTest(notebook=name):
                notebook = json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
                all_source = "\n".join(
                    _source_text(cell) for cell in notebook["cells"]
                )
                self.assertIn("method_provenance.json", all_source)
                self.assertIn("optimizer_seed_protocol_fingerprints", all_source)
                self.assertIn("protocol_fingerprint", all_source)
                self.assertIn("source_artifact_kind", all_source)

        for name, expected_source in EXPECTED_METHOD_SOURCE_BY_NOTEBOOK.items():
            with self.subTest(source_mapping=name):
                notebook = json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
                all_source = "\n".join(
                    _source_text(cell) for cell in notebook["cells"]
                )
                self.assertIn(expected_source, all_source)
        notebook10 = json.loads(
            (NOTEBOOK_ROOT / "10_Two_Checkpoint_Finite_Flow.ipynb").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn(
            '"source_artifact_kind": "verified_dense_update_capture"',
            "\n".join(_source_text(cell) for cell in notebook10["cells"]),
        )
        notebook04 = json.loads(
            (NOTEBOOK_ROOT / "04_Fixed_Point_Comparison.ipynb").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn(
            "run_fingerprint",
            "\n".join(_source_text(cell) for cell in notebook04["cells"]),
        )

        notebook = json.loads(
            (NOTEBOOK_ROOT / "15_Method_Nulls_Stability_Comparison.ipynb").read_text(
                encoding="utf-8"
            )
        )
        all_source = "\n".join(_source_text(cell) for cell in notebook["cells"])
        self.assertIn("current_run_fingerprints", all_source)
        self.assertIn("expected_method_grid", all_source)
        self.assertIn("observed_fit_grid != expected_method_grid", all_source)
        self.assertIn("fit_row_count", all_source)
        self.assertIn("additional_weight_only_ecs_jacobians", all_source)
        self.assertIn("data_dependent_ecs_jacobians", all_source)
        self.assertIn("REQUIRED_METHOD_CONTRACT_TOKENS", all_source)

    def test_single_checkpoint_notebook_fits_only_six_declared_jacobians(self) -> None:
        notebook = json.loads(
            (NOTEBOOK_ROOT / "13_Single_Checkpoint_Map_Jacobians.ipynb").read_text(
                encoding="utf-8"
            )
        )
        all_source = "\n".join(_source_text(cell) for cell in notebook["cells"])
        for method in (
            "polar_pullback",
            "normalized_gram_pullback",
            "centered_log_gram_pullback",
            "centered_log_singular_radial_pullback",
            "finite_muon_ns5_pullback",
        ):
            self.assertIn(f'("{method}",', all_source)
        self.assertIn(
            '"ecs_grassmann_cartan_cover_full_row_shell_pullback"',
            all_source,
        )
        self.assertIn(
            '"ecs_grassmann_cartan_cover_detx_shell_pullback"',
            all_source,
        )
        self.assertIn("muon_newton_schulz_analytic_spectrum", all_source)
        self.assertIn("centered_log_gram_analytic_spectrum", all_source)
        self.assertIn("centered_log_singular_analytic_spectrum", all_source)
        self.assertIn("ecs_grassmann_cover_analytic_spectrum", all_source)
        self.assertIn("ecs_grassmann_cover_map", all_source)
        self.assertIn("ecs_grassmann_retracted_core", all_source)
        self.assertIn("requested anchored retracted-core cover", all_source)
        self.assertIn("J[E]=2V_c^T E^T U_k Sigma_k^-1", all_source)
        self.assertIn(
            ") = load_verified_ecs_metric_tables(",
            all_source,
        )
        self.assertIn("exact_ecs_cover_rank_record", all_source)
        self.assertIn("ecs_rank_nearest_or_forward_fill_used", all_source)
        self.assertIn("verified_tail_checkpoint_cache_plus_exact_sparse", all_source)
        self.assertEqual(all_source.count("np.linalg.svd(W, compute_uv=False)"), 1)
        self.assertIn(
            "precomputed_singular_values=checkpoint_singular_values",
            all_source,
        )
        self.assertNotIn("gram_translation_esd", all_source)
        self.assertIn(
            "fits only spectra of these derivatives--never the ESD",
            all_source,
        )
        self.assertIn("if int(selected.epoch) == final_cache_epoch", all_source)
        self.assertIn("Fits cover every selected state", all_source)

        notebook15 = json.loads(
            (NOTEBOOK_ROOT / "15_Method_Nulls_Stability_Comparison.ipynb").read_text(
                encoding="utf-8"
            )
        )
        notebook15_source = "\n".join(
            _source_text(cell) for cell in notebook15["cells"]
        )
        self.assertIn(
            "verified_tail_checkpoint_cache_plus_exact_sparse_weightwatcher_trace_metrics",
            notebook15_source,
        )

    def test_weight_quotient_notebooks_materialize_all_five_methods(self) -> None:
        names = (
            "19_One_Seed_Muon_MuonClip_Weight_Quotients.ipynb",
            "20_Three_Seed_Muon_MuonClip_Weight_Quotients.ipynb",
            "23_Short100_10Seed_Weight_Quotients.ipynb",
        )
        methods = (
            "gram_ridge",
            "blockwise_singular",
            "feshbach_downfolding",
            "rectangular_d_transform",
            "calibrated_mp_shrinker",
        )
        for name in names:
            with self.subTest(notebook=name):
                notebook = json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
                all_source = "\n".join(
                    _source_text(cell) for cell in notebook["cells"]
                )
                code_sources = [
                    _source_text(cell)
                    for cell in notebook["cells"]
                    if cell.get("cell_type") == "code"
                ]
                method_cells = {}
                for method in methods:
                    token = f'run_quotient_method(\n    "{method}"'
                    matching = [
                        index for index, source in enumerate(code_sources)
                        if token in source
                    ]
                    self.assertEqual(matching, [matching[0]] if matching else [])
                    self.assertEqual(len(matching), 1, method)
                    method_cells[method] = matching[0]
                self.assertEqual(len(set(method_cells.values())), len(methods))
                self.assertIn("MAXIMUM_CHECKPOINTS = 100", all_source)
                self.assertIn("load_verified_tail_checkpoint_refs", all_source)
                self.assertIn("analyze_weightwatcher_dual", all_source)
                self.assertIn("replace_model_matrix(model, layer, result.weight)", all_source)
                self.assertIn("rectangular_diagonal_canonical_section", all_source)
                self.assertIn("O(out) x O(in)", all_source)
                self.assertIn("fix_fingers=clip_xmax", all_source)
                self.assertIn('for variant in ("raw", "clip_xmax")', all_source)
                self.assertIn("same_transformed_model_for_raw_and_fix_fingers_clip_xmax", all_source)
                self.assertIn("transformed_spectrum_sha256", all_source)
                self.assertIn("Raw and clip_xmax fits used different transformed spectra", all_source)
                self.assertIn("expected_grid", all_source)
                self.assertIn("observed_grid", all_source)
                self.assertIn("model_layer_esd", all_source)
                self.assertIn("WeightWatcher analyzed a different ESD", all_source)
                self.assertIn("validate_weightwatcher_measurement", all_source)
                self.assertIn("WeightQuotientUnavailable", all_source)
                self.assertIn("RESUME_PARTIAL_RESULTS = True", all_source)
                self.assertIn("load_resumable_tables", all_source)
                self.assertIn("normalize_papermill_sequence", all_source)
                self.assertIn("normalize_papermill_bool", all_source)
                self.assertIn(
                    'OPTIMIZER_SLUGS, name="OPTIMIZER_SLUGS"', all_source
                )
                self.assertIn("analysis_code_sha256", all_source)
                self.assertIn("inspect.getsource(weightwatcher_fit)", all_source)
                self.assertIn("WEIGHT_QUOTIENT_ANALYSIS_SETTINGS_SHA256", all_source)
                self.assertIn('"analysis_settings_sha256"', all_source)
                self.assertIn('"completed": False', all_source)
                self.assertIn('manifest["completed"] = True', all_source)
                self.assertNotIn('glob("*_final_spectra_index.csv")', all_source)
                self.assertIn("GRAM_RIDGE_SCAN", all_source)
                self.assertIn("BLOCKWISE_SCAN", all_source)
                self.assertIn("FESHBACH_SCAN", all_source)
                self.assertIn("RECTANGULAR_D_SCAN", all_source)
                self.assertIn("CALIBRATED_SHRINKER_SCAN", all_source)
                self.assertNotIn("run_cli_training(", all_source)
                self.assertNotIn("run_training(", all_source)

        one_seed = json.loads((NOTEBOOK_ROOT / names[0]).read_text(encoding="utf-8"))
        one_source = "\n".join(_source_text(cell) for cell in one_seed["cells"])
        one_plot_source = next(
            _source_text(cell)
            for cell in one_seed["cells"]
            if "one_seed = quotient_fit_rows.copy()" in _source_text(cell)
        )
        self.assertIn("RUN_PARAMETER_SCANS = True", one_source)
        self.assertIn("UNCERTAINTY_POLICY = 'no_seed_error_bars'", one_source)
        self.assertNotIn("fill_between(", one_plot_source)

        three_seed = json.loads((NOTEBOOK_ROOT / names[1]).read_text(encoding="utf-8"))
        three_source = "\n".join(
            _source_text(cell) for cell in three_seed["cells"]
        )
        self.assertIn("RUN_PARAMETER_SCANS = False", three_source)
        self.assertIn("ACTIVE_SEEDS != (1337, 2027, 31415)", three_source)
        self.assertIn("student_t_95_ci_across_complete_seeded_runs", three_source)
        self.assertIn("validate_cross_run_provenance(active_run_manifests())", three_source)
        self.assertIn("summarize_numeric_metrics(", three_source)
        self.assertIn('qualified_fits = numeric_fits[numeric_fits["fit_success"]]', three_source)
        self.assertIn("fit_availability_by_checkpoint.csv", three_source)
        self.assertIn("fill_between(", three_source)

        short_ten_seed = json.loads(
            (NOTEBOOK_ROOT / names[2]).read_text(encoding="utf-8")
        )
        short_source = "\n".join(
            _source_text(cell) for cell in short_ten_seed["cells"]
        )
        self.assertIn(
            "SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]",
            short_source,
        )
        self.assertIn(
            'OPTIMIZER_SLUGS = ["muonclip_rms", "adamw"]', short_source
        )
        self.assertIn("RUN_PARAMETER_SCANS = False", short_source)
        self.assertIn("EXPECTED_SHORT100_SEEDS", short_source)
        self.assertIn("issubset(EXPECTED_SHORT100_SEEDS)", short_source)
        self.assertIn("SHORT100_FULL_CONFIRMATORY_GRID", short_source)
        self.assertIn("student_t_95_ci_across_complete_seeded_runs", short_source)
        self.assertIn("fill_between(", short_source)

    def test_generated_notebooks_are_up_to_date(self) -> None:
        builder = _load_builder()
        generated = dict(builder.build_all_notebooks())
        self.assertEqual(set(generated), EXPECTED_NOTEBOOKS)

        for name, expected_notebook in sorted(generated.items()):
            with self.subTest(notebook=name):
                actual_notebook = json.loads(
                    (NOTEBOOK_ROOT / name).read_text(encoding="utf-8")
                )
                self.assertEqual(actual_notebook, expected_notebook)


if __name__ == "__main__":
    unittest.main()
