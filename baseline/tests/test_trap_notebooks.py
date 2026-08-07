import ast
import unittest
from pathlib import Path

import nbformat


class TrapNotebookTests(unittest.TestCase):
    def test_all_three_training_notebooks_require_and_plot_num_traps(self):
        notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
        names = (
            "MNIST_MLP3_SGD_Momentum_Baseline.ipynb",
            "MNIST_MLP3_AdamW_Baseline.ipynb",
            "MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb",
        )
        for name in names:
            path = notebook_dir / name
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            source = "\n".join(cell.source for cell in notebook.cells)
            self.assertIn("ww_randomize=True", source, name)
            self.assertIn("num_traps", source, name)
            self.assertIn("watcher.analyze(ERG=True, randomize=True)", source, name)
            self.assertIn("7_layerwise_weightwatcher_num_traps_95ci.png", source, name)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    ast.parse(cell.source, filename=f"{name}:cell{index}")


if __name__ == "__main__":
    unittest.main()
