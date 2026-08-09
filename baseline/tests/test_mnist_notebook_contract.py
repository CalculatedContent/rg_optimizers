"""Regression guards for the versioned MNIST notebook artifact contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from rg_baselines.config import (
    MNIST_REFERENCE_RECIPE_VERSION,
    MNIST_REFERENCE_SUITE_SLUG,
)


BASELINE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = BASELINE_ROOT / "notebooks"
TRAINING_NOTEBOOKS = (
    "MNIST_MLP3_SGD_Momentum_Baseline.ipynb",
    "MNIST_MLP3_AdamW_Baseline.ipynb",
    "MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb",
)
COMPARISON_NOTEBOOK = "MNIST_MLP3_Baseline_Comparison.ipynb"


def _notebook_source(name: str) -> str:
    notebook = json.loads(
        (NOTEBOOK_ROOT / name).read_text(encoding="utf-8")
    )
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


class MnistNotebookArtifactContractTests(unittest.TestCase):
    def test_reference_suite_slug_is_recipe_versioned(self) -> None:
        self.assertEqual(MNIST_REFERENCE_RECIPE_VERSION, 3)
        self.assertEqual(
            MNIST_REFERENCE_SUITE_SLUG,
            f"mnist_mlp3_recipe_v{MNIST_REFERENCE_RECIPE_VERSION}",
        )

    def test_training_notebooks_isolate_legacy_artifacts(self) -> None:
        for name in TRAINING_NOTEBOOKS:
            source = _notebook_source(name)
            with self.subTest(notebook=name):
                self.assertIn(
                    "RUN_ROOT = BASE_RUN_ROOT / MNIST_REFERENCE_SUITE_SLUG",
                    source,
                )
                self.assertIn("RUN_DIR = RUN_ROOT / CONFIG.run_slug", source)
                self.assertIn("CONFIG.validate()", source)
                self.assertIn("overwrite=False", source)
                self.assertIn("Ignoring legacy unversioned artifacts", source)


    def test_comparison_reads_only_the_versioned_suite(self) -> None:
        source = _notebook_source(COMPARISON_NOTEBOOK)
        self.assertIn(
            "RUN_ROOT = BASE_RUN_ROOT / MNIST_REFERENCE_SUITE_SLUG",
            source,
        )
        self.assertIn("config['recipe_version']", source)
        self.assertIn("result.epochs == 30", source)
        self.assertIn("Ignoring legacy unversioned result directories", source)


if __name__ == "__main__":
    unittest.main()
