from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


class MuonRectangularNotebookTests(unittest.TestCase):
    def test_notebook_contains_fc1_fc2_rectangular_analysis(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "MNIST_MLP3_Muon_Rectangular_RG_ESD.ipynb"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload.get("cells", [])
        )
        self.assertIn("analyze_rectangular_muon_run", source)
        self.assertIn("fc1.weight", source)
        self.assertIn("fc2.weight", source)
        self.assertIn("core_log_deviation", source)
        self.assertIn("angular_theta_squared", source)
        self.assertIn("POWERLAW_ALPHA_RANGE", source)
        for index, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            ast.parse(
                "".join(cell.get("source", [])),
                filename=f"{path}:cell-{index}",
            )


if __name__ == "__main__":
    unittest.main()
