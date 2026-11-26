"""Independent review of durable-archive and protected-calibration contracts."""

import json
from pathlib import Path
import tempfile
import unittest

from california_housing.publish import check_monotonic, valid_paths, PREFIX


class PublicationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.local = self.root / "local"
        self.remote = self.root / "remote"
        self.record = {
            "optimizer": "modified_adamw", "seed": 1340,
            "phase": "confirmation", "config": {"peak_lr": .001},
            "plan_sha256": "fixed_plan", "data_manifest_sha256": "fixed_data",
            "source_bundle_sha256": "fixed_source",
            "frozen_selection_sha256": "fixed_selection",
            "completed_updates": 20000, "status": "complete",
        }

    def write_pair(self, local, remote):
        for root, record in ((self.local, local), (self.remote, remote)):
            folder = root / "modified_adamw"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "result.json").write_text(json.dumps(record))

    def test_archive_accepts_identical_or_forward_progress(self):
        previous = {**self.record, "completed_updates": 2000, "status": "paused"}
        self.write_pair(self.record, previous)
        check_monotonic(self.local, self.remote)
        self.write_pair(self.record, self.record)
        check_monotonic(self.local, self.remote)

    def test_archive_rejects_older_progress(self):
        previous = {**self.record, "completed_updates": 2000, "status": "paused"}
        self.write_pair(previous, self.record)
        with self.assertRaises(RuntimeError):
            check_monotonic(self.local, self.remote)

    def test_archive_preserves_a_completed_run(self):
        failed = {**self.record, "status": "failed"}
        self.write_pair(failed, self.record)
        with self.assertRaises(RuntimeError):
            check_monotonic(self.local, self.remote)

    def test_archive_rejects_foreign_source_or_selection(self):
        for field in ("source_bundle_sha256", "frozen_selection_sha256"):
            with self.subTest(field=field):
                other = {**self.record, field: "another_experiment"}
                self.write_pair(other, self.record)
                with self.assertRaises(RuntimeError):
                    check_monotonic(self.local, self.remote)

    def test_publication_paths_remain_within_campaign(self):
        self.assertEqual(valid_paths([PREFIX + "/runs/seed_1340"]),
                         [PREFIX + "/runs/seed_1340"])
        for path in ("README.md", "/" + PREFIX, PREFIX + "/../other_campaign"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                valid_paths([path])


if __name__ == "__main__":
    unittest.main()
