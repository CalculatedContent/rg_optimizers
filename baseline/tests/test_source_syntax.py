"""Dependency-free syntax checks for baseline Python and notebook sources."""

from __future__ import annotations

import ast
import json
import py_compile
import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = BASELINE_ROOT / "rg_baselines"
NOTEBOOK_ROOT = BASELINE_ROOT / "notebooks"
EXPERIMENT_ROOT = BASELINE_ROOT / "experiments"


class BaselineSourceSyntaxTests(unittest.TestCase):
    def test_all_python_sources_compile(self) -> None:
        failures: list[str] = []
        python_sources = sorted(PACKAGE_ROOT.rglob("*.py"))
        python_sources.extend(sorted(EXPERIMENT_ROOT.rglob("*.py")))
        for path in python_sources:
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{path}: {exc.msg}")
        self.assertFalse(
            failures,
            "Baseline Python syntax failures:\n" + "\n".join(failures),
        )

    def test_all_notebook_code_cells_parse(self) -> None:
        failures: list[str] = []
        notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
        notebooks.extend(sorted(EXPERIMENT_ROOT.rglob("*.ipynb")))
        self.assertTrue(notebooks, "No baseline notebooks were found")
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                try:
                    ast.parse(source, filename=f"{path}:cell-{index}")
                except SyntaxError as exc:
                    failures.append(
                        f"{path}: cell {index}: line {exc.lineno}: {exc.msg}"
                    )
        self.assertFalse(
            failures,
            "Baseline notebook syntax failures:\n" + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
