from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "angular"
    / "muonclip_angular_radial_rg.ipynb"
)


def _code_sources() -> list[str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_memory_safe_angular_notebook_code_cells_compile() -> None:
    for index, source in enumerate(_code_sources()):
        compile(source, f"angular_notebook_cell_{index}", "exec")


def test_single_matrix_wrapper_preserves_loader_metadata() -> None:
    source = "\n".join(_code_sources())
    assert "weights, payloads, model_cfg = original_loader" in source
    assert "return filtered, payloads, model_cfg" in source


def test_notebook_displays_the_actual_results_variable() -> None:
    source = "\n".join(_code_sources())
    assert "display(RESULTS)" in source
    assert "model_cg" not in source
    assert "RESULS" not in source
