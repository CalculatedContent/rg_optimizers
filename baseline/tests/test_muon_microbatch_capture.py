from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from rg_baselines.muon_microbatch_capture import (
    MuonMicrobatchCheckpointRecorder,
    load_microbatch_checkpoint,
    log_flow_deviation,
    matrix_esd_eigenvalues,
    relative_flow_esd_eigenvalues,
    relative_flow_operator,
)


class TinyMLP3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(7, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 3)


class MuonMicrobatchCaptureTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = TinyMLP3()
            recorder = MuonMicrobatchCheckpointRecorder(
                run_dir=root,
                model=model,
                capture_every=1,
                dtype="float32",
            )
            first = recorder.capture(
                global_step=0,
                epoch=0,
                batch_index=0,
                examples_seen=0,
                learning_rates={"primary": 0.1, "auxiliary": 0.01},
            )
            self.assertIsNotNone(first)
            assert first is not None
            self.assertTrue(first.is_file())
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.01)
            second = recorder.capture(
                global_step=1,
                epoch=1,
                batch_index=1,
                examples_seen=8,
                training_loss=1.25,
                learning_rates={"primary": 0.1, "auxiliary": 0.01},
            )
            self.assertIsNotNone(second)
            assert second is not None
            payload = load_microbatch_checkpoint(second)
            self.assertEqual(payload["global_step"], 1)
            self.assertEqual(
                set(payload["matrices"]),
                {"fc1.weight", "fc2.weight", "fc3.weight"},
            )
            index = root / "microbatch_checkpoints" / "checkpoint_index.csv"
            self.assertTrue(index.is_file())
            self.assertEqual(
                len(index.read_text(encoding="utf-8").splitlines()), 3
            )

    def test_relative_operator_is_identity_for_unchanged_full_rank_matrices(
        self,
    ) -> None:
        for shape in ((5, 7), (5, 5), (7, 5)):
            matrix = torch.randn(*shape, dtype=torch.float64)
            operator, side = relative_flow_operator(
                matrix, matrix, pinv_rtol=1e-10
            )
            expected = torch.eye(min(shape), dtype=torch.float64)
            self.assertEqual(operator.shape, expected.shape)
            self.assertIn(side, {"input", "output"})
            self.assertTrue(
                torch.allclose(operator, expected, atol=1e-8, rtol=1e-8)
            )
            eigenvalues, _ = relative_flow_esd_eigenvalues(
                matrix, matrix, pinv_rtol=1e-10
            )
            self.assertTrue(
                np.allclose(eigenvalues, np.ones(min(shape)), atol=1e-8)
            )
            self.assertEqual(log_flow_deviation(eigenvalues).size, 0)

    def test_weight_esd_is_squared_singular_values(self) -> None:
        matrix = torch.diag(torch.tensor([3.0, 2.0, 0.5]))
        eigenvalues = matrix_esd_eigenvalues(matrix)
        self.assertTrue(
            np.allclose(eigenvalues, np.array([0.25, 4.0, 9.0]))
        )

    def test_training_runner_smoke_with_synthetic_mnist(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        from rg_baselines import mnist_muon_microbatch as experiment

        generator = torch.Generator().manual_seed(123)
        inputs = torch.randn(16, 1, 28, 28, generator=generator)
        targets = torch.randint(0, 10, (16,), generator=generator)
        dataset = TensorDataset(inputs, targets)

        def fake_loaders(config, *, data_dir, device):
            del data_dir, device
            loader = DataLoader(
                dataset, batch_size=config.batch_size, shuffle=False
            )
            return (
                loader,
                loader,
                loader,
                loader,
                torch.Generator().manual_seed(config.seed + 101),
                list(range(12)),
                list(range(12, 16)),
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                experiment, "_make_datasets_and_loaders", fake_loaders
            ):
                run_dir = experiment.run_muon_microbatch_capture(
                    data_dir=root / "data",
                    output_dir=root / "run",
                    epochs=3,
                    batch_size=8,
                    max_steps=2,
                    device="cpu",
                    capture_every=1,
                    overwrite=True,
                    progress=False,
                )
            index = (
                Path(run_dir)
                / "microbatch_checkpoints"
                / "checkpoint_index.csv"
            )
            self.assertTrue(index.is_file())
            self.assertEqual(
                len(index.read_text(encoding="utf-8").splitlines()), 4
            )
            final = torch.load(
                Path(run_dir) / "final_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(final["global_step"], 2)


if __name__ == "__main__":
    unittest.main()
