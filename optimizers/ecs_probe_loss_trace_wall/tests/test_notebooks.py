from __future__ import annotations

import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "notebooks" / "MNIST_MLP3_AdamW_vs_ECS_Probe_Loss_TraceWall.ipynb",
    ROOT
    / "notebooks"
    / "MNIST_MLP3_SGD_Momentum_vs_ECS_Probe_Loss_TraceWall.ipynb",
)


class NotebookTests(unittest.TestCase):
    def test_notebooks_are_valid_and_code_cells_compile(self) -> None:
        for path in NOTEBOOKS:
            with self.subTest(path=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type == "code":
                        compile(cell.source, f"{path.name}:cell{index}", "exec")
                        self.assertIsNone(cell.execution_count)
                        self.assertEqual(cell.outputs, [])
                    if cell.cell_type == "markdown":
                        self.assertNotIn("\x0c", cell.source)
                        self.assertNotIn("\r", cell.source)
                        self.assertNotIn("\t", cell.source)

    def test_primary_protocol_is_explicit(self) -> None:
        for path in NOTEBOOKS:
            with self.subTest(path=path.name):
                notebook = nbformat.read(path, as_version=4)
                source = "\n".join(cell.source for cell in notebook.cells)
                for required in (
                    "seeds=(1337, 2027, 31415)",
                    "epochs=20",
                    "batch_size=128",
                    "corrections_per_epoch=1",
                    "probe_batch_size=256",
                    "probe_batches_per_correction=2",
                    "measure_weightwatcher=True",
                    "require_weightwatcher=True",
                    "official_test_set_used_for_optimization",
                    "baseline",
                    "trace_wall",
                    "plot_all",
                ):
                    self.assertIn(required, source)
                self.assertIn("rotating subset of the MNIST training set", source)
                self.assertIn("one-epoch linear warmup", source)
                self.assertIn("cosine decay", source)

    def test_each_notebook_selects_the_requested_base_optimizer(self) -> None:
        adamw = nbformat.read(NOTEBOOKS[0], as_version=4)
        sgd = nbformat.read(NOTEBOOKS[1], as_version=4)
        adamw_source = "\n".join(cell.source for cell in adamw.cells)
        sgd_source = "\n".join(cell.source for cell in sgd.cells)
        self.assertIn("BaseOptimizerConfig.adamw_baseline()", adamw_source)
        self.assertIn("BaseOptimizerConfig.sgd_momentum_baseline()", sgd_source)


if __name__ == "__main__":
    unittest.main()
