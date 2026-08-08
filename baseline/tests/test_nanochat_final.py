import tempfile
import unittest
from pathlib import Path

from rg_baselines.nanochat_final import _reject_unversioned_existing_run


class NanoChatFinalRuntimeTests(unittest.TestCase):
    def test_empty_or_versioned_seed_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _reject_unversioned_existing_run(root, 17)
            seed_dir = root / "seed_17"
            seed_dir.mkdir()
            _reject_unversioned_existing_run(root, 17)
            (seed_dir / "runtime_policy.json").write_text("{}")
            (seed_dir / "training.log").write_text("existing")
            _reject_unversioned_existing_run(root, 17)

    def test_nonempty_legacy_seed_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_dir = root / "seed_17"
            seed_dir.mkdir()
            (seed_dir / "run_complete.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "predates"):
                _reject_unversioned_existing_run(root, 17)


if __name__ == "__main__":
    unittest.main()
