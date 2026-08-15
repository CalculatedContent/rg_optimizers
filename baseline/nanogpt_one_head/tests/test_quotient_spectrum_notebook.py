from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "angular"
    / "08_muonclip_initial_metric_quotient_rg.ipynb"
)


def _code_sources() -> list[str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_quotient_notebook_code_cells_compile() -> None:
    for index, source in enumerate(_code_sources()):
        compile(source, f"quotient_notebook_cell_{index}", "exec")


def test_notebook_uses_saved_initial_best_final_checkpoints() -> None:
    source = "\n".join(_code_sources())
    assert '"initial": resolved.initial_path' in source
    assert '"best": BEST_PATH' in source
    assert '"final": resolved.final_path' in source


def test_notebook_runs_native_weightwatcher_and_shows_plots() -> None:
    source = "\n".join(_code_sources())
    assert "plot=True" in source
    assert "savefig=str(savedir)" in source
    assert "randomize=True" in source
    assert "ERG=False" in source
    assert "display(Image(filename=str(path)))" in source
    assert "SHOW_PLOTS = True" in source


def test_notebook_contains_initial_metric_whitening_and_control() -> None:
    source = "\n".join(_code_sources())
    assert "W0.T @ W0" in source
    assert "W0 @ W0.T" in source
    assert "whiten_matrix" in source
    assert "W_gaussian" in source
    assert "EPS_GRID" in source
