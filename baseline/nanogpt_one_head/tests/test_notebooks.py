"""Static checks for one-head nanoGPT notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = EXPERIMENT_ROOT / "notebooks"
MANUAL_ESD_NOTEBOOK = (
    NOTEBOOK_ROOT / "06_first_layer_esd_binning_powerlaw.ipynb"
)


def _cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return str(source)


def test_all_notebook_code_cells_parse() -> None:
    notebooks = sorted(NOTEBOOK_ROOT.glob("*.ipynb"))
    assert notebooks, "No one-head nanoGPT notebooks were found"

    failures: list[str] = []
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            try:
                ast.parse(
                    _cell_source(cell),
                    filename=f"{path}:cell-{index}",
                )
            except SyntaxError as exception:
                failures.append(
                    f"{path}: cell {index}: line {exception.lineno}: "
                    f"{exception.msg}"
                )

    assert not failures, "Notebook syntax failures:\n" + "\n".join(failures)


def test_manual_esd_notebook_contract() -> None:
    notebook = json.loads(
        MANUAL_ESD_NOTEBOOK.read_text(encoding="utf-8")
    )
    source = "\n".join(
        _cell_source(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    required_fragments = (
        "watcher.get_ESD(layer=layer_id)",
        "LAYER_ID =",
        "REMOVE_TOP_EIGENVALUES =",
        "powerlaw.Fit(",
        "xmin=float(RETAINED_ESD.min())",
        'fit.distribution_compare(\n            "power_law",\n            "truncated_power_law"',
        "for remove_top in range(maximum_remove + 1)",
    )

    missing = [fragment for fragment in required_fragments if fragment not in source]
    assert not missing, f"Manual ESD notebook is missing: {missing}"
