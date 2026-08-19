"""Dependency-free contract checks for the generated tangent-RG notebooks."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = BASELINE_ROOT / "experiments" / "mnist_mlp3_tangent_rg"
NOTEBOOK_ROOT = EXPERIMENT_ROOT / "notebooks"
BUILDER_PATH = EXPERIMENT_ROOT / "scripts" / "build_notebooks.py"

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
}

CAPTURE_ANALYSIS_NOTEBOOKS = {
    "11_Muon_Update_Stiefel_Tangent.ipynb",
    "14_Calibrated_Local_Training_Map.ipynb",
}

METHOD_OUTPUT_NOTEBOOKS = TAIL_CACHE_ANALYSIS_NOTEBOOKS | CAPTURE_ANALYSIS_NOTEBOOKS

EXPECTED_METHOD_SOURCE_BY_NOTEBOOK = {
    "10_Two_Checkpoint_Finite_Flow.ipynb": "verified_tail_checkpoint_cache_model_only",
    "11_Muon_Update_Stiefel_Tangent.ipynb": "verified_dense_update_capture",
    "12_Radial_Angular_Quotients.ipynb": "verified_tail_checkpoint_cache_model_only",
    "13_Single_Checkpoint_Map_Jacobians.ipynb": "verified_tail_checkpoint_cache_model_only",
    "14_Calibrated_Local_Training_Map.ipynb": "verified_calibrated_dense_capture",
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


class NotebookContractTests(unittest.TestCase):
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
                self.assertIn(
                    f'"source_artifact_kind": "{expected_source}"', all_source
                )
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

    def test_single_checkpoint_notebook_fits_only_five_declared_jacobians(self) -> None:
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
        self.assertIn("muon_newton_schulz_analytic_spectrum", all_source)
        self.assertIn("centered_log_gram_analytic_spectrum", all_source)
        self.assertIn("centered_log_singular_analytic_spectrum", all_source)
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
