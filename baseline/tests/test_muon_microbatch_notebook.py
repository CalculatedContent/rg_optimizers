from __future__ import annotations

import json
from pathlib import Path
import unittest


class MuonMicrobatchNotebookTests(unittest.TestCase):
    def test_notebook_contains_requested_analysis(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "MNIST_MLP3_Muon_Microbatch_RG_ESD.ipynb"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload.get("cells", [])
        )
        self.assertIn("powerlaw.Fit", source)
        self.assertIn("matrix_esd_eigenvalues", source)
        self.assertIn("relative_flow_esd_eigenvalues", source)
        self.assertIn("log_flow_deviation", source)
        self.assertIn("alpha", source)
        self.assertIn("global_step", source)


if __name__ == "__main__":
    unittest.main()
