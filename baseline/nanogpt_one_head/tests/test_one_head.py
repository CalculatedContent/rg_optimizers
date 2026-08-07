from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.analysis import mean_ci95
from rg_nanogpt_one_head.checkpoints import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from rg_nanogpt_one_head.config import (
    epoch_step_map,
    load_config,
    max_steps,
    optimizer_profile,
)
from rg_nanogpt_one_head.data import write_token_splits
from rg_nanogpt_one_head.model import GPT, GPTConfig, transformer_matrix_items
from rg_nanogpt_one_head.optimizers import make_optimizer_handles, optimizer_step
from rg_nanogpt_one_head.spectral import summarize_spectral_frame
from rg_nanogpt_one_head.training import run_one


class FakeEncoder:
    n_vocab = 64
    eot_token = 0

    def encode_ordinary(self, text: str) -> list[int]:
        return [1 + (ord(character) % 62) for character in text]


def reference_config() -> dict:
    return load_config(EXPERIMENT_ROOT / "configs" / "reference.yaml")


def tiny_config(optimizer: str) -> dict:
    cfg = deepcopy(reference_config())
    cfg["dataset"].update(
        {
            "name": "unit/fineweb",
            "config": "unit",
            "revision": "unit-revision",
            "train_tokens": 2_048,
            "val_tokens": 512,
            "test_tokens": 512,
        }
    )
    cfg["model"].update(
        {
            "vocab_size": 64,
            "block_size": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 16,
        }
    )
    cfg["training"].update(
        {
            "seeds": [13],
            "batch_size": 2,
            "grad_accum_steps": 1,
            "target_epochs": 0.02,
            "epoch_interval": 1.0,
            "eval_interval_steps": 1,
            "eval_batches": 1,
            "checkpoint_interval_steps": 1,
        }
    )
    cfg["evaluation"].update(
        {
            "bleu_examples": 2,
            "bleu_prompt_tokens": 3,
            "bleu_continuation_tokens": 2,
            "bleu_batch_size": 2,
        }
    )
    return cfg


def write_tiny_data(path: Path, cfg: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    splits = {
        "train": int(cfg["dataset"]["train_tokens"]),
        "val": int(cfg["dataset"]["val_tokens"]),
        "test": int(cfg["dataset"]["test_tokens"]),
    }
    for split, size in splits.items():
        rng.integers(0, cfg["model"]["vocab_size"], size=size, dtype=np.uint16).tofile(
            path / f"{split}.bin"
        )
    (path / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tokenizer": "gpt2",
                "vocab_size": cfg["model"]["vocab_size"],
                "dtype": "uint16",
                "splits": splits,
                "document_disjoint_splits": True,
                "dataset_name": cfg["dataset"]["name"],
                "dataset_config": cfg["dataset"]["config"],
                "dataset_revision": cfg["dataset"]["revision"],
            }
        ),
        encoding="utf-8",
    )


def test_reference_protocol_is_one_block_one_head_and_has_required_ww_flags():
    cfg = reference_config()
    assert cfg["model"]["n_layer"] == 1
    assert cfg["model"]["n_head"] == 1
    assert cfg["dataset"]["name"] == "HuggingFaceFW/fineweb-edu"
    assert cfg["weightwatcher"]["ERG"] is True
    assert cfg["weightwatcher"]["randomize"] is True
    assert max_steps(cfg) == 6104
    assert list(epoch_step_map(cfg).values()) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_model_inventory_and_all_optimizer_updates_are_finite():
    cfg = reference_config()
    model = GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=1, n_embd=16))
    matrices = transformer_matrix_items(model)
    assert len(matrices) == 6
    assert {matrix_type for _, matrix_type, _, _ in matrices} == {
        "W_Q", "W_K", "W_V", "W_O", "W_MLP_IN", "W_MLP_OUT"
    }
    for optimizer_name in ("sgd_momentum", "adamw", "muon"):
        candidate = GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=1, n_embd=16))
        handles = make_optimizer_handles(candidate, optimizer_profile(cfg, optimizer_name))
        x = torch.randint(0, 64, (2, 8))
        _, loss = candidate(x, x)
        assert loss is not None
        loss.backward()
        optimizer_step(handles)
        assert all(torch.isfinite(parameter).all() for parameter in candidate.parameters())
        assert len(handles) == (2 if optimizer_name == "muon" else 1)


def test_document_disjoint_writer_produces_exact_splits(tmp_path):
    metadata = write_token_splits(
        iter(["abcdef", "ghijkl", "mnopqr", "stuvwx"]),
        FakeEncoder(),
        tmp_path,
        train_tokens=4,
        val_tokens=3,
        test_tokens=2,
        dataset_metadata={
            "dataset_name": "unit/fineweb",
            "dataset_config": "unit",
            "dataset_revision": "unit",
        },
        progress_every_documents=0,
    )
    assert metadata["document_disjoint_splits"] is True
    assert metadata["splits"] == {"train": 4, "val": 3, "test": 2}
    assert np.fromfile(tmp_path / "train.bin", dtype=np.uint16).size == 4
    assert np.fromfile(tmp_path / "val.bin", dtype=np.uint16).size == 3
    assert np.fromfile(tmp_path / "test.bin", dtype=np.uint16).size == 2


def test_checkpoint_roundtrip_restores_optimizer_and_generator(tmp_path):
    cfg = reference_config()
    model = GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=1, n_embd=16))
    handles = make_optimizer_handles(model, optimizer_profile(cfg, "adamw"))
    generator = torch.Generator(device="cpu").manual_seed(99)
    x = torch.randint(0, 64, (2, 8), generator=generator)
    _, loss = model(x, x)
    assert loss is not None
    loss.backward()
    optimizer_step(handles)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    path = save_training_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        handles=handles,
        step=7,
        best_validation_loss=2.5,
        best_validation_step=6,
        elapsed_seconds=12.0,
        fingerprint="abc",
        cfg=cfg,
        optimizer_name="adamw",
        seed=13,
        train_generator=generator,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    restored = load_training_checkpoint(
        path,
        model=model,
        handles=handles,
        expected_fingerprint="abc",
        train_generator=generator,
    )
    assert restored == (7, 2.5, 6, 12.0)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_spectral_summary_keeps_direct_trap_and_erg_fields():
    frame = pd.DataFrame(
        {
            "alpha": [2.1, 2.3, np.nan],
            "ERG_gap": [0.0, 2.0, np.nan],
            "num_traps": [1.0, 3.0, 2.0],
        }
    )
    summary = summarize_spectral_frame(frame, step=10, tokens_seen=80, epoch=0.5)
    assert summary["alpha_median"] == pytest.approx(2.2)
    assert summary["ERG_gap_mean"] == pytest.approx(1.0)
    assert summary["num_traps_mean"] == pytest.approx(2.0)


def test_student_t_interval_is_run_level():
    result = mean_ci95([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["ci95_half_width"] == pytest.approx(4.3026527297 / np.sqrt(3))


@pytest.mark.parametrize("optimizer_name", ["sgd_momentum", "adamw", "muon"])
def test_tiny_cpu_training_writes_restart_and_epoch_artifacts(
    tmp_path, monkeypatch, optimizer_name
):
    cfg = tiny_config(optimizer_name)
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    write_tiny_data(data_root, cfg)

    monkeypatch.setattr(
        "rg_nanogpt_one_head.train_loop.evaluate_bleu",
        lambda *args, **kwargs: {"bleu": 0.0},
    )
    monkeypatch.setattr(
        "rg_nanogpt_one_head.run_utils.evaluate_bleu",
        lambda *args, **kwargs: {"bleu": 0.0},
    )
    monkeypatch.setattr(
        "rg_nanogpt_one_head.train_loop.run_weightwatcher",
        lambda *args, **kwargs: {
            "alpha_median": 2.0,
            "ERG_gap_median": 0.0,
            "num_traps_mean": 0.0,
        },
    )

    run_dir = run_one(
        cfg=cfg,
        data_root=data_root,
        results_root=results_root,
        optimizer_name=optimizer_name,
        seed=13,
        device="cpu",
        resume=True,
        progress=False,
    )
    assert (run_dir / "checkpoint_latest.pt").is_file()
    assert (run_dir / "checkpoint_best.pt").is_file()
    assert (run_dir / "checkpoint_final.pt").is_file()
    assert (run_dir / "run_complete.json").is_file()
    epoch_metrics = pd.read_csv(run_dir / "epoch_metrics.csv")
    assert len(epoch_metrics) >= 1
    assert epoch_metrics["test_monitoring_only"].eq(1).all()
    assert all(Path(path).is_file() for path in epoch_metrics["checkpoint_path"])


def test_notebooks_are_valid_and_expose_requested_metrics():
    notebook_paths = sorted((EXPERIMENT_ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebook_paths] == [
        "01_sgd_momentum_baseline.ipynb",
        "02_adamw_baseline.ipynb",
        "03_muon_baseline.ipynb",
        "04_compare_baselines.ipynb",
    ]
    for path in notebook_paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        source = "\n".join(cell.source for cell in notebook.cells)
        for required in (
            "test_accuracy", "train_accuracy", "test_loss", "train_loss",
            "test_perplexity", "test_bleu", "alpha", "ERG_gap",
            "num_traps", "95% Student-t",
        ):
            assert required in source
