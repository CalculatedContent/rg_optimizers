from __future__ import annotations

import json
import sys
import types

import pandas as pd
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


def test_multiseed_notebook_preserves_tables_and_loader_metadata(monkeypatch) -> None:
    # The notebook now delegates loading to run_multiseed_analysis. Exercise
    # its real result/display cell; the removed single-matrix wrapper is not
    # part of this entry point any more.
    cell = next(s for s in _code_sources() if "SEED_RESULTS, CROSS_SEED, MANIFEST =" in s)
    seeds = pd.DataFrame({"matrix_name": ["L00_W_Q", "L00_W_K"], "seed": [1337, 1337]})
    cross = pd.DataFrame({"matrix_name": ["L00_W_Q", "L00_W_K"], "alpha_mean": [2.0, 2.1]})
    manifest = {"seeds": [1337], "error_bar_contract": "seed SD", "output_dir": "results"}
    calls, displayed = [], []
    config = object()
    def run(config_arg, **kwargs):
        calls.append((config_arg, kwargs))
        return seeds, cross, manifest
    display_module = types.ModuleType("IPython.display")
    display_module.display = displayed.append
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    namespace = {"CONFIG": config, "ANGULAR_SEEDS": "1337", "RG_MATRIX_NAME": "L00_W_Q",
                 "run_multiseed_analysis": run}
    exec(compile(cell, "angular_result_cell", "exec"), namespace)
    assert calls == [(config, {"seed_spec": "1337"})]
    assert namespace["SEED_RESULTS"] is seeds
    assert namespace["CROSS_SEED"] is cross
    assert namespace["MANIFEST"] is manifest
    assert len(displayed) == 2
    pd.testing.assert_frame_equal(displayed[0], seeds.iloc[:1])
    pd.testing.assert_frame_equal(displayed[1], cross.iloc[:1])


def test_notebook_displays_the_actual_results_variable() -> None:
    source = "\n".join(_code_sources())
    assert "display(SEED_RESULTS[" in source
    assert "display(CROSS_SEED[" in source
    assert "model_cg" not in source
    assert "RESULS" not in source
