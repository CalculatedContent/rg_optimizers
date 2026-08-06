import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from wwpgd_local_delta import MNISTRunConfig
from wwpgd_local_delta.mnist_experiment import run_mnist_comparison


def _synthetic_loaders(config: MNISTRunConfig, seed: int):
    generator = torch.Generator().manual_seed(9000 + int(seed))
    x_train = torch.randn(24, 1, 28, 28, generator=generator)
    y_train = torch.randint(0, 10, (24,), generator=generator)
    x_test = torch.randn(12, 1, 28, 28, generator=generator)
    y_test = torch.randint(0, 10, (12,), generator=generator)
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    shuffle_generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=6,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=12,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=12,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, train_eval_loader, test_loader


class ExperimentSmokeTests(unittest.TestCase):
    def test_paired_runtime_for_adamw_and_sgd_momentum(self):
        for optimizer_kind in ("adamw", "sgd_momentum"):
            with self.subTest(optimizer_kind=optimizer_kind):
                config = MNISTRunConfig(
                    optimizer_kind=optimizer_kind,
                    epochs=1,
                    batch_size=6,
                    test_batch_size=12,
                    hidden_width=8,
                    correction_fraction=0.25,
                    grad_clip_norm=1.0,
                    seeds=(17,),
                    ww_enabled=False,
                    ww_required=False,
                )
                with mock.patch(
                    "wwpgd_local_delta.mnist_experiment.make_loaders",
                    side_effect=_synthetic_loaders,
                ):
                    result = run_mnist_comparison(
                        config,
                        device=torch.device("cpu"),
                        progress=False,
                    )

                performance = result.performance
                self.assertEqual(set(performance["arm"]), {"baseline", "local_delta_ecs"})
                self.assertEqual(len(performance), 4)
                self.assertTrue(
                    performance.groupby("arm")["epoch"].nunique().eq(2).all()
                )
                self.assertEqual(
                    performance.groupby("seed")["initial_state_checksum"].nunique().iloc[0],
                    1,
                )
                for column in ("train_loss", "test_loss", "train_acc", "test_acc"):
                    self.assertTrue(np.isfinite(performance[column]).all())

                corrections = result.corrections
                self.assertFalse(corrections.empty)
                ok = corrections[corrections["status"] == "ok"]
                self.assertEqual(set(ok["parameter"]), {"fc1.weight", "fc2.weight", "fc3.weight"})
                self.assertLess(float(ok["damping_error"].max()), 1e-4)
                self.assertLess(float(ok["pythagorean_error"].max()), 1e-4)
                self.assertLess(float(ok["correction_identity_error"].max()), 1e-6)
                self.assertTrue(
                    np.allclose(
                        ok["removed_fraction_of_base"].to_numpy(),
                        config.correction_fraction
                        * ok["orthogonal_fraction"].to_numpy(),
                        rtol=1e-4,
                        atol=1e-7,
                    )
                )

                spectral = result.spectral
                self.assertFalse(spectral.empty)
                self.assertTrue(spectral["diagnostic_source"].eq("fallback_svd").all())
                self.assertEqual(set(spectral["arm"]), {"baseline", "local_delta_ecs"})


if __name__ == "__main__":
    unittest.main()
