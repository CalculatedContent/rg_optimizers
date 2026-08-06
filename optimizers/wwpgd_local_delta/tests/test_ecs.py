import unittest

import torch

from wwpgd_local_delta.ecs import damp_delta_outside_ecs, select_self_consistent_ecs


class ECSTests(unittest.TestCase):
    def test_self_consistent_scan_returns_valid_rank(self):
        s = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5])
        scan = select_self_consistent_ecs(s, min_retained=2)
        self.assertGreaterEqual(scan.rank, 2)
        self.assertLessEqual(scan.rank, 5)
        self.assertGreaterEqual(scan.normalization_dimension, scan.rank)

    def test_fraction_zero_is_identity(self):
        torch.manual_seed(0)
        w = torch.randn(6, 5)
        delta = torch.randn(6, 5)
        result = damp_delta_outside_ecs(delta, w, correction_fraction=0.0, min_retained=2)
        self.assertTrue(torch.allclose(result.corrected_delta, delta, atol=1e-5))
        self.assertAlmostEqual(result.removed_fraction_of_base, 0.0, places=6)

    def test_fraction_one_removes_right_orthogonal_component(self):
        torch.manual_seed(1)
        w = torch.randn(8, 6)
        delta = torch.randn(8, 6)
        result = damp_delta_outside_ecs(delta, w, correction_fraction=1.0, min_retained=2, max_retained=2)
        _, _, vh = torch.linalg.svd(w.float(), full_matrices=False)
        basis = vh[: result.ecs_rank].T.to(delta.dtype)
        orth = result.corrected_delta - (result.corrected_delta @ basis) @ basis.T
        self.assertLess(float(torch.linalg.vector_norm(orth)), 1e-4)

    def test_fraction_half_removes_half_orthogonal_norm(self):
        torch.manual_seed(2)
        w = torch.randn(10, 7)
        delta = torch.randn(10, 7)
        hard = damp_delta_outside_ecs(delta, w, correction_fraction=1.0, min_retained=3, max_retained=3)
        soft = damp_delta_outside_ecs(delta, w, correction_fraction=0.5, min_retained=3, max_retained=3)
        removed_soft = torch.linalg.vector_norm(delta - soft.corrected_delta)
        removed_hard = torch.linalg.vector_norm(delta - hard.corrected_delta)
        self.assertTrue(torch.allclose(removed_soft, 0.5 * removed_hard, rtol=1e-4, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
