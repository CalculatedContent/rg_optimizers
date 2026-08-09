"""Repository-wide notebook integrity audit for the 2026-08-09 force push."""

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
    return json.loads((REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8"))


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _is_papermill_injected(cell: dict[str, Any]) -> bool:
    tags = cell.get("metadata", {}).get("tags", [])
    return isinstance(tags, list) and "injected-parameters" in tags


def _source_signature(notebook: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(cell.get("cell_type")), _cell_source(cell))
        for cell in notebook.get("cells", [])
        if not _is_papermill_injected(cell)
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


def _emit_error(path: str | Path, message: str) -> None:
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error file={path},line=1::{escaped}")


class ForcePushNotebookAuditTests(unittest.TestCase):
    def test_all_repository_notebooks_are_valid_and_parseable(self) -> None:
        notebooks = sorted(REPOSITORY_ROOT.rglob("*.ipynb"))
        if len(notebooks) != EXPECTED_NOTEBOOK_COUNT:
            message = f"found {len(notebooks)} notebooks; expected {EXPECTED_NOTEBOOK_COUNT}"
            _emit_error("baseline/notebooks", message)
            self.fail(message)

        transformer = TransformerManager()
        failures: list[tuple[str, str]] = []
        for path in notebooks:
            relative = str(path.relative_to(REPOSITORY_ROOT))
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                failures.append((relative, str(exc)))
                continue

            if notebook.get("nbformat") != 4:
                failures.append((relative, f"nbformat={notebook.get('nbformat')!r}, expected 4"))
                continue
            if not isinstance(notebook.get("cells"), list):
                failures.append((relative, "cells is not a list"))
                continue

            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = _cell_source(cell)
                if not source.strip():
                    continue
                try:
                    ast.parse(
                        transformer.transform_cell(source),
                        filename=f"{path}:cell-{index}",
                    )
                except Exception as exc:
                    failures.append((relative, f"cell {index}: {exc}"))

        for path, message in failures:
            _emit_error(path, message)
        if failures:
            self.fail(f"{len(failures)} notebook JSON/code-cell failure(s)")

    def test_committed_outputs_match_current_sources_and_completed_cleanly(self) -> None:
        failures: list[tuple[str, str]] = []
        for source_path, output_path in OUTPUT_PAIRS.items():
            source_notebook = _read_notebook(source_path)
            output_notebook = _read_notebook(output_path)
            source_sig = _source_signature(source_notebook)
            output_sig = _source_signature(output_notebook)
            if output_sig != source_sig:
                first = next(
                    (i for i, pair in enumerate(zip(output_sig, source_sig)) if pair[0] != pair[1]),
                    min(len(output_sig), len(source_sig)),
                )
                failures.append(
                    (
                        output_path,
                        "source/output recipe mismatch after normalizing generated metadata: "
                        f"first differing cell={first}, output_cells={len(output_sig)}, "
                        f"source_cells={len(source_sig)}",
                    )
                )

            executed_code_cells = 0
            all_output_text: list[str] = []
            for index, cell in enumerate(output_notebook.get("cells", [])):
                papermill = cell.get("metadata", {}).get("papermill", {})
                if papermill.get("exception") is True or papermill.get("status") == "failed":
                    failures.append((output_path, f"Papermill failure in cell {index}"))

                if cell.get("cell_type") == "code" and _cell_source(cell).strip():
                    executed_code_cells += 1
                    if cell.get("execution_count") is None:
                        failures.append((output_path, f"non-empty code cell {index} was not executed"))

                for output in cell.get("outputs", []):
                    if output.get("output_type") == "error":
                        failures.append((output_path, f"error output in cell {index}: {output}"))
                    all_output_text.append(_output_text(output))

            rendered_text = "\n".join(all_output_text)
            if executed_code_cells == 0:
                failures.append((output_path, "no non-empty code cells were executed"))
            if "Traceback (most recent call last)" in rendered_text:
                failures.append((output_path, "rendered output contains a traceback"))
            if "mnist_mlp3_recipe_v3" not in rendered_text:
                failures.append((output_path, "recipe-v3 marker is absent from rendered output"))

        for path, message in failures:
            _emit_error(path, message)
        if failures:
            self.fail(f"{len(failures)} committed-output integrity failure(s)")

    def test_output_campaign_contract_is_current(self) -> None:
        failures: list[tuple[str, str]] = []
        for output_path in OUTPUT_PAIRS.values():
            if "Comparison" in output_path:
                continue
            notebook = _read_notebook(output_path)
            source = "\n".join(_cell_source(cell) for cell in notebook.get("cells", []))
            rendered = json.dumps(notebook)
            expected = {
                "epochs=30": source,
                "DEFAULT_BASELINE_SEEDS": source,
                "assert len(SEEDS) == 3": source,
                "overwrite=False": source,
                '"recipe_version"': rendered,
            }
            for marker, haystack in expected.items():
                if marker not in haystack:
                    failures.append((output_path, f"missing campaign-contract marker: {marker}"))

        for path, message in failures:
            _emit_error(path, message)
        if failures:
            self.fail(f"{len(failures)} campaign-contract failure(s)")


if __name__ == "__main__":
    unittest.main()
