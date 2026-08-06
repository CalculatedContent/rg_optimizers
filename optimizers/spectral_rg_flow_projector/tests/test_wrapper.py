import unittest

import torch

from rg_spectral_flow.ecs import AdaptiveSupportState
from rg_spectral_flow.wrapper import SpectralRGFlowConfig, SpectralRGFlowProjector


class DiagonalModel(torch.nn.Module):
    def __init__(self, diagonal) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.diag(torch.tensor(diagonal, dtype=torch.float64)))


class WrapperTests(unittest.TestCase):
    def _wrapper(self, diagonal):
        model = DiagonalModel(diagonal)
        base = torch.optim.SGD(model.parameters(), lr=1.0)
        wrapper = SpectralRGFlowProjector(
            base,
            model.named_parameters(),
            config=SpectralRGFlowConfig(
                projection_strength=1.0,
                max_abs_log_eigenvalue_correction=None,
                max_correction_ratio=None,
                min_retained=2,
                apply_every_steps=1,
            ),
        )
        state = AdaptiveSupportState(
            ecs_rank=4,
            normalization_dimension=4.0,
            bulk_effective_count=0.0,
            trace_log_per_eval=0.0,
            status="test",
            pl_rank=4,
            working_rank=4,
        )
        wrapper.set_support_states({"weight": state}, replace=True)
        return model, wrapper

    def test_wrapper_projects_completed_collapse_step(self) -> None:
        before_diag = [4.0, 3.0, 2.0, 1.0]
        base_diag = torch.tensor([4.6, 2.9, 1.8, 0.8], dtype=torch.float64)
        model, wrapper = self._wrapper(before_diag)
        before = model.weight.detach().clone()
        target = torch.diag(base_diag)
        model.weight.grad = before - target
        wrapper.step()
        stats = wrapper.pop_step_stats()
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["status"], "ok")
        self.assertGreater(stats[0]["base_flow_component"], 0.0)
        self.assertAlmostEqual(stats[0]["corrected_flow_component"], 0.0, places=8)
        self.assertFalse(torch.equal(model.weight.detach(), target))

    def test_wrapper_preserves_away_step(self) -> None:
        before_diag = [5.0, 2.5, 1.2, 0.6]
        base_diag = torch.tensor([4.2, 2.8, 1.5, 0.9], dtype=torch.float64)
        model, wrapper = self._wrapper(before_diag)
        before = model.weight.detach().clone()
        target = torch.diag(base_diag)
        model.weight.grad = before - target
        wrapper.step()
        stats = wrapper.pop_step_stats()
        self.assertEqual(stats[0]["status"], "skipped")
        self.assertTrue(torch.allclose(model.weight.detach(), target, atol=1e-12, rtol=0.0))

    def test_state_dict_preserves_support_state(self) -> None:
        _, first = self._wrapper([4.0, 3.0, 2.0, 1.0])
        saved = first.state_dict()
        _, second = self._wrapper([4.0, 3.0, 2.0, 1.0])
        second.load_state_dict(saved)
        restored = second.get_support_states()["weight"]
        self.assertEqual(restored.working_rank, 4)


if __name__ == "__main__":
    unittest.main()
