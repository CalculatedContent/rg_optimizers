from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "angular" / "09_muonclip_remove_radial_run_weightwatcher.ipynb"


def test_notebook_code_cells_compile() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"angular_residual_cell_{index}", "exec")


def test_notebook_uses_exact_polar_residual_and_native_weightwatcher() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "q0 = polar(initial)" in source
    assert "qt = polar(final)" in source
    assert "residual = scale * (qt - q0)" in source
    assert "replace_transformer_matrices" in source
    assert "plot=True" in source
    assert "savefig=str(savedir)" in source
    assert "randomize=True" in source
    assert "display(Image(filename=str(path)))" in source
