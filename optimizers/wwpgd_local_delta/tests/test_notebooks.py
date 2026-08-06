import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_notebooks_are_valid_and_compile(self):
        root = Path(__file__).resolve().parents[1] / "notebooks"
        notebooks = sorted(root.glob("MNIST_MLP3_*_LocalDeltaECS_5Runs.ipynb"))
        self.assertEqual(len(notebooks), 2)
        for path in notebooks:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)
            sources = []
            for index, cell in enumerate(payload["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell.get("source", []))
                compile(source, f"{path.name}:cell{index}", "exec")
                sources.append(source)
            joined = "\n".join(sources)
            self.assertIn("epochs=10", joined)
            self.assertIn("1337, 2027, 4099, 7919, 104729", joined)
            self.assertIn('ecs_reference="epoch_end"', joined)
            self.assertIn("ww_required=True", joined)
            self.assertIn("damping_error", joined)
            self.assertIn("initial_state_checksum", joined)


if __name__ == "__main__":
    unittest.main()
