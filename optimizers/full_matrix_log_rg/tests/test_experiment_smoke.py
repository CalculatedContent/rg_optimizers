from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from full_matrix_log_rg import FullMatrixLogConfig, MatrixLogSupport
from full_matrix_log_rg.experiment import run_mnist_sgd
from full_matrix_log_rg.support import SupportCheckpoint


@dataclass(frozen=True)
class FakeBaselineConfig:
    optimizer: str = "sgd_momentum"
    seed: int = 7
    epochs: int = 2
    split_seed: int = 11
    validation_size: int = 4
    batch_size: int = 4
    num_workers: int = 0
    train_eval_max_batches: int | None = None
    ww_min_evals: int = 2
    ww_max_evals: int | None = None
    ww_svd_method: str = "accurate"
    ww_randomize: bool = True
    grad_clip_norm: float = 1.0
    save_epoch_checkpoints: bool = True
    sgd_learning_rate: float = 0.05

    def validate(self) -> None:
        if self.optimizer != "sgd_momentum":
            raise ValueError("wrong optimizer")


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 3)
        self.fc2 = torch.nn.Linear(3, 3)
        self.fc3 = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def fake_build_optimizer(model, config):
    return torch.optim.SGD(
        model.parameters(), lr=config.sgd_learning_rate, momentum=0.9
    )


def fake_set_learning_rates(optimizer, config, **_):
    for group in optimizer.param_groups:
        group["lr"] = config.sgd_learning_rate
    return {"primary": config.sgd_learning_rate}


def fake_set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def fake_evaluate(model, loader, *, device, max_batches=None):
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for index, (x, y) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        logits = model(x.to(device))
        targets = y.to(device)
        loss_sum += float(torch.nn.functional.cross_entropy(logits, targets)) * len(y)
        correct += int((logits.argmax(1) == targets).sum())
        seen += len(y)
    return {
        "loss": loss_sum / max(seen, 1),
        "accuracy": correct / max(seen, 1),
        "examples": seen,
    }


def fake_make_loaders(config, *, data_dir: Path, device: torch.device):
    del data_dir, device
    generator = torch.Generator().manual_seed(config.seed + 101)
    x = torch.randn(20, 4, generator=torch.Generator().manual_seed(99))
    y = (x.sum(dim=1) > 0).long()
    train_indices = list(range(12))
    validation_indices = list(range(12, 16))
    train = TensorDataset(x[:12], y[:12])
    validation = TensorDataset(x[12:16], y[12:16])
    test = TensorDataset(x[16:], y[16:])
    return (
        DataLoader(train, batch_size=4, shuffle=True, generator=generator),
        DataLoader(train, batch_size=4, shuffle=False),
        DataLoader(validation, batch_size=4, shuffle=False),
        DataLoader(test, batch_size=4, shuffle=False),
        generator,
        train_indices,
        validation_indices,
    )


def fake_checkpoint(model, *, run_label, epoch, global_step, **_):
    rows = []
    for layer_id, name in enumerate(("fc1.weight", "fc2.weight"), start=1):
        rows.append(
            {
                "run": run_label,
                "epoch": epoch,
                "global_step": global_step,
                "layer_id": layer_id,
                "layer": name.removesuffix(".weight"),
                "parameter_name": name,
                "status": "ok",
                "alpha": 2.1,
                "abs_alpha_minus_2": 0.1,
                "detX_num": 3,
                "num_pl_spikes": 3,
                "ERG_gap": 0,
                "m_midpoint": 3,
                "num_traps": 0,
            }
        )
    frame = pd.DataFrame(rows)
    return types.SimpleNamespace(
        details=frame.copy(), metrics=frame, esd_arrays={}
    )


def fake_analyze_supports(model, *, run_label, epoch, global_step, **_):
    checkpoint = fake_checkpoint(
        model, run_label=run_label, epoch=epoch, global_step=global_step
    )
    supports = {}
    parameters = dict(model.named_parameters())
    for name in ("fc1.weight", "fc2.weight"):
        weight = parameters[name].detach()
        work = weight if weight.shape[0] >= weight.shape[1] else weight.T
        transposed = weight.shape[0] < weight.shape[1]
        _, _, vh = torch.linalg.svd(work.float(), full_matrices=False)
        rank = min(3, min(work.shape))
        supports[name] = MatrixLogSupport(
            retained_rank=rank,
            normalization_dimension=float(work.shape[1]),
            right_basis=vh[:rank].T.contiguous(),
            transposed=transposed,
            checkpoint_epoch=epoch,
        )
    return SupportCheckpoint(
        checkpoint.details, checkpoint.metrics, supports, {}
    )


class ExperimentSmokeTests(unittest.TestCase):
    def test_restartable_baseline_and_rg_runs(self):
        imports = (
            FakeBaselineConfig,
            TinyMLP,
            fake_build_optimizer,
            fake_checkpoint,
            fake_set_learning_rates,
            lambda: torch.device("cpu"),
            fake_evaluate,
            fake_set_seed,
            lambda checkpoint: checkpoint,
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "full_matrix_log_rg.experiment._baseline_imports", return_value=imports
        ), patch(
            "full_matrix_log_rg.experiment._make_loaders", side_effect=fake_make_loaders
        ), patch(
            "full_matrix_log_rg.experiment.analyze_supports",
            side_effect=fake_analyze_supports,
        ):
            root = Path(temporary)
            config = FakeBaselineConfig()
            rg_config = FullMatrixLogConfig(
                mode="modewise",
                apply_every_steps=2,
                max_correction_ratio=0.10,
                parameter_names=("fc1.weight", "fc2.weight"),
            )
            baseline = run_mnist_sgd(
                config,
                rg_config=None,
                data_dir=root / "data",
                output_dir=root / "baseline",
                device=torch.device("cpu"),
                progress=False,
            )
            extended = run_mnist_sgd(
                config,
                rg_config=rg_config,
                data_dir=root / "data",
                output_dir=root / "extended",
                device=torch.device("cpu"),
                progress=False,
            )
            self.assertTrue(baseline.completed)
            self.assertTrue(extended.completed)
            self.assertFalse(extended.corrections.empty)
            for run_dir in (root / "baseline", root / "extended"):
                for filename in (
                    "checkpoint_latest.pt",
                    "checkpoint_best.pt",
                    "final_state.pt",
                    "run_complete.json",
                    "performance_by_epoch.csv",
                    "spectral_metrics_by_epoch_and_layer.csv",
                ):
                    self.assertTrue((run_dir / filename).is_file(), filename)

            loaded = run_mnist_sgd(
                config,
                rg_config=rg_config,
                data_dir=root / "data",
                output_dir=root / "extended",
                device=torch.device("cpu"),
                resume=True,
                progress=False,
            )
            self.assertTrue(loaded.completed)
            self.assertEqual(len(loaded.performance), len(extended.performance))


if __name__ == "__main__":
    unittest.main()
