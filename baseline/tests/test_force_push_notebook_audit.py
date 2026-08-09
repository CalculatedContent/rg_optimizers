"""Repository-wide notebook integrity audit for the 2026-08-09 force push.

This test is intentionally placed on a temporary audit branch.  It verifies the
live repository tree and the four committed MNIST output notebooks without
rerunning the expensive three-seed training campaigns.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from typing import Any

from IPython.core.inputtransformer2 import TransformerManager


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_NOTEBOOK_COUNT = 30
OUTPUT_PAIRS = {
    "baseline/notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb":
        "baseline/notebooks/MNIST_MLP3_SGD_Momentum_Baseline.out.ipynb",
    "baseline/notebooks/MNIST_MLP3_AdamW_Baseline.ipynb":
        "baseline/notebooks/MNIST_MLP3_AdamW_Baseline.out.ipynb",
    "baseline/notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb":
        "baseline/notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.out.ipynb",
    "baseline/notebooks/MNIST_MLP3_Baseline_Comparison.ipynb":
        "baseline/notebooks/MNIST_MLP3_Baseline_Comparison.out.ipynb",
}


def _read_notebook(relative_path: str | Path) -> dict[str, Any]:
    path = REPOSITORY_ROOT / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _source_signature(notebook: dict[str, Any]) -> list[tuple[str, str | None, str]]:
    """Return the source-bearing cell identity that execution must preserve."""

    return [
        (str(cell.get("cell_type")), cell.get("id"), _cell_source(cell))
        for cell in notebook.get("cells", [])
    ]


def _output_text(output: dict[str, Any]) -> str:
    chunks: list[str] = []
    text = output.get("text")
    if isinstance(text, list):
        chunks.extend(str(item) for item in text)
    elif text is not None:
        chunks.append(str(text))
    data = output.get("data", {})
    if isinstance(data, dict):
        for mime, payload in data.items():
            if mime.startswith("image/"):
                continue
            if isinstance(payload, list):
                chunks.extend(str(item) for item in payload)
            elif payload is not None:
                chunks.append(str(payload))
    return "".join(chunks)


class ForcePushNotebookAuditTests(unittest.TestCase):
    def test_all_repository_notebooks_are_valid_and_parseable(self) -> None:
        notebooks = sorted(REPOSITORY_ROOT.rglob("*.ipynb"))
        self.assertEqual(
            len(notebooks),
            EXPECTED_NOTEBOOK_COUNT,
            "The live notebook inventory changed during the force-push audit.",
        )

        transformer = TransformerManager()
        failures: list[str] = []
        for path in notebooks:
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                failures.append(f"{path.relative_to(REPOSITORY_ROOT)}: {exc}")
                continue

            self.assertEqual(notebook.get("nbformat"), 4, path)
            self.assertIsInstance(notebook.get("cells"), list, path)
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = _cell_source(cell)
                if not source.strip():
                    continue
                try:
                    transformed = transformer.transform_cell(source)
                    ast.parse(transformed, filename=f"{path}:cell-{index}")
                except Exception as exc:  # report all syntax/transform failures together
                    failures.append(
                        f"{path.relative_to(REPOSITORY_ROOT)}: cell {index}: {exc}"
                    )

        self.assertFalse(
            failures,
            "Notebook JSON or code-cell failures:\n" + "\n".join(failures),
        )

    def test_committed_outputs_match_current_sources_and_completed_cleanly(self) -> None:
        for source_path, output_path in OUTPUT_PAIRS.items():
            with self.subTest(output=output_path):
                source_notebook = _read_notebook(source_path)
                output_notebook = _read_notebook(output_path)

                self.assertEqual(
                    _source_signature(output_notebook),
                    _source_signature(source_notebook),
                    "Executed notebook sources differ from the current source notebook.",
                )

                executed_code_cells = 0
                all_output_text: list[str] = []
                for index, cell in enumerate(output_notebook.get("cells", [])):
                    papermill = cell.get("metadata", {}).get("papermill", {})
                    self.assertIsNot(
                        papermill.get("exception"),
                        True,
                        f"Papermill exception in cell {index}",
                    )
                    self.assertNotEqual(
                        papermill.get("status"),
                        "failed",
                        f"Papermill failure in cell {index}",
                    )

                    if cell.get("cell_type") != "code":
                        continue
                    if _cell_source(cell).strip():
                        executed_code_cells += 1
                        self.assertIsNotNone(
                            cell.get("execution_count"),
                            f"Non-empty code cell {index} was not executed.",
                        )

                    for output in cell.get("outputs", []):
                        self.assertNotEqual(
                            output.get("output_type"),
                            "error",
                            f"Error output in cell {index}: {output}",
                        )
                        all_output_text.append(_output_text(output))

                self.assertGreater(executed_code_cells, 0)
                rendered_text = "\n".join(all_output_text)
                self.assertNotIn("Traceback (most recent call last)", rendered_text)
                self.assertIn("mnist_mlp3_recipe_v3", rendered_text)

    def test_output_campaign_contract_is_current(self) -> None:
        training_outputs = [
            path for path in OUTPUT_PAIRS.values() if "Comparison" not in path
        ]
        for output_path in training_outputs:
            with self.subTest(output=output_path):
                notebook = _read_notebook(output_path)
                source = "\n".join(
                    _cell_source(cell) for cell in notebook.get("cells", [])
                )
                rendered = json.dumps(notebook)
                self.assertIn("epochs=30", source)
                self.assertIn("DEFAULT_BASELINE_SEEDS", source)
                self.assertIn("assert len(SEEDS) == 3", source)
                self.assertIn("overwrite=False", source)
                self.assertIn('"recipe_version"', rendered)


if __name__ == "__main__":
    unittest.main()
