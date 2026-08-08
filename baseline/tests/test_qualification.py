import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rg_baselines.qualification import (
    freeze_winner,
    mnist_candidates,
    one_head_profile_candidates,
    rank_validation_candidates,
    vit_candidates,
)


class QualificationTests(unittest.TestCase):
    def test_candidate_neighborhoods_are_unique_and_include_committed_center(self):
        families = [
            mnist_candidates("sgd_momentum"),
            mnist_candidates("adamw"),
            mnist_candidates("sgd_momentum_muon"),
            vit_candidates("sgd_momentum"),
            vit_candidates("adamw"),
            vit_candidates("muon"),
            one_head_profile_candidates("sgd_momentum"),
            one_head_profile_candidates("adamw"),
            one_head_profile_candidates("muon"),
        ]
        for candidates in families:
            self.assertGreaterEqual(len(candidates), 5)
            self.assertEqual(
                len({candidate.candidate_id for candidate in candidates}),
                len(candidates),
            )
            self.assertEqual(
                sum(candidate.is_committed_center for candidate in candidates),
                1,
            )

    def test_ranking_uses_validation_even_when_test_order_is_opposite(self):
        rows = []
        for seed in (17, 29, 43):
            rows.extend(
                [
                    {
                        "candidate_id": "validation-winner",
                        "seed": seed,
                        "epoch": 1,
                        "validation_loss": 0.40,
                        "validation_accuracy": 0.80,
                        "test_loss": 9.0,
                        "test_accuracy": 0.0,
                    },
                    {
                        "candidate_id": "test-winner",
                        "seed": seed,
                        "epoch": 1,
                        "validation_loss": 0.60,
                        "validation_accuracy": 0.70,
                        "test_loss": 0.01,
                        "test_accuracy": 1.0,
                    },
                ]
            )
        leaderboard, selected = rank_validation_candidates(
            pd.DataFrame(rows), expected_seeds=(17, 29, 43)
        )
        self.assertEqual(
            leaderboard.iloc[0]["candidate_id"], "validation-winner"
        )
        self.assertEqual(len(selected), 6)

    def test_incomplete_candidate_seed_set_is_rejected(self):
        frame = pd.DataFrame(
            [
                {
                    "candidate_id": "a",
                    "seed": 17,
                    "epoch": 1,
                    "validation_loss": 1.0,
                },
                {
                    "candidate_id": "a",
                    "seed": 29,
                    "epoch": 1,
                    "validation_loss": 0.9,
                },
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "expected"):
            rank_validation_candidates(
                frame, expected_seeds=(17, 29, 43)
            )

    def test_freeze_file_records_protected_test_policy(self):
        candidate = mnist_candidates("adamw")[0]
        leaderboard = pd.DataFrame(
            [
                {
                    "validation_rank": 1,
                    "candidate_id": candidate.candidate_id,
                    "n": 3,
                    "mean_best_validation_loss": 0.1,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = freeze_winner(
                Path(temporary) / "lock.json",
                candidate=candidate,
                leaderboard=leaderboard,
                evidence_paths=["validation.csv"],
                source_commit="abc123",
                data_identity={"dataset": "MNIST"},
            )
            payload = json.loads(path.read_text())
            self.assertFalse(payload["protected_test_used_for_selection"])
            self.assertEqual(payload["candidate_id"], candidate.candidate_id)


if __name__ == "__main__":
    unittest.main()
