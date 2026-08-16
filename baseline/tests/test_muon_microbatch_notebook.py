from __future__ import annotations

import json
from pathlib import Path


def test_notebook_contains_requested_analysis() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "MNIST_MLP3_Muon_Microbatch_RG_ESD.ipynb"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "powerlaw.Fit" in source
    assert "matrix_esd_eigenvalues" in source
    assert "relative_flow_esd_eigenvalues" in source
    assert "log_flow_deviation" in source
    assert "alpha" in source
    assert "global_step" in source
