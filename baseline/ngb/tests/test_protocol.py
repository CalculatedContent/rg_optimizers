from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from rg_nanogpt_one_head.optimizers import make_optimizer_handles, optimizer_step

from rg_ngb.config import (
    DEFAULT_ROOT,
    canonical_seeds,
    epoch_step_map,
    expected_matrix_count,
    load_config,
    max_steps,
    optimizer_profile,
    roots,
)
from rg_ngb.model import GPT, GPTConfig, transformer_matrix_items

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def config(name: str) -> dict:
    return load_config(EXPERIMENT_ROOT / "configs" / name)


def test_defaults_are_tmp_and_protocol_scoped(monkeypatch):
    for name in ("RG_NGB_ROOT", "RG_NGB_DATA_ROOT", "RG_NGB_RESULTS_ROOT", "RG_NGB_PLOTS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    cfg = config("v4_one_head.yaml")
    assert DEFAULT_ROOT == Path("/tmp/rg-ngb")
    assert roots(cfg) == {
        "root": Path("/tmp/rg-ngb"),
        "data": Path("/tmp/rg-ngb/data"),
        "results": Path("/tmp/rg-ngb/results/ngb_v4_one_head_2epoch"),
        "plots": Path("/tmp/rg-ngb/plots/ngb_v4_one_head_2epoch"),
    }


def test_one_head_v4_contract():
    cfg = config("v4_one_head.yaml")
    assert cfg["protocol"]["version"] == 4
    assert cfg["model"]["n_layer"] == 1
    assert cfg["model"]["n_head"] == 1
    assert cfg["training"]["target_epochs"] == 2.0
    assert cfg["training"]["epoch_interval"] == 0.25
    assert cfg["training"]["eval_interval_steps"] == 500
    assert max_steps(cfg) == 19_532
    assert list(epoch_step_map(cfg).values()) == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    assert expected_matrix_count(cfg) == 6
    assert canonical_seeds(cfg) == (1337, 2027, 4099)
    sgd = optimizer_profile(cfg, "sgd_momentum")
    adamw = optimizer_profile(cfg, "adamw")
    muon = optimizer_profile(cfg, "muon")
    assert sgd["learning_rate"] == pytest.approx(0.05)
    assert sgd["min_learning_rate"] == pytest.approx(0.0005)
    assert sgd["warmup_fraction"] == pytest.approx(0.05)
    assert adamw["learning_rate"] == pytest.approx(3e-4)
    assert adamw["min_learning_rate"] == pytest.approx(1e-5)
    assert adamw["warmup_fraction"] == pytest.approx(0.025)
    assert muon["matrix_learning_rate"] == pytest.approx(0.01)
    assert muon["matrix_min_learning_rate"] == pytest.approx(2e-4)
    assert muon["aux_weight_decay"] == pytest.approx(0.10)


def test_small_4x4_is_distinct_and_has_expected_inventory():
    cfg = config("v4_small_4x4.yaml")
    assert cfg["protocol"]["architecture_id"] == "small_4x4"
    assert cfg["model"]["n_layer"] == 4
    assert cfg["model"]["n_head"] == 4
    assert expected_matrix_count(cfg) == 24
    model = GPT(GPTConfig(**cfg["model"]))
    matrices = transformer_matrix_items(model)
    assert len(matrices) == 24
    assert {block for _, _, block, _ in matrices} == {0, 1, 2, 3}
    assert model.parameter_count() == 7_253_248
    assert optimizer_profile(cfg, "sgd_momentum")["learning_rate"] == pytest.approx(0.03)


@pytest.mark.parametrize("configuration", ["v4_one_head.yaml", "v4_small_4x4.yaml"])
@pytest.mark.parametrize("optimizer_name", ["sgd_momentum", "adamw", "muon"])
def test_all_optimizer_paths_take_finite_step(configuration, optimizer_name):
    cfg = config(configuration)
    model_cfg = dict(cfg["model"])
    model_cfg.update({"vocab_size": 64, "block_size": 8, "n_embd": 16})
    model = GPT(GPTConfig(**model_cfg))
    handles = make_optimizer_handles(model, optimizer_profile(cfg, optimizer_name))
    x = torch.randint(0, 64, (2, 8))
    _, loss = model(x, x)
    assert loss is not None
    loss.backward()
    optimizer_step(handles)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert len(handles) == (2 if optimizer_name == "muon" else 1)


def test_notebooks_are_python3_and_parse():
    expected = {
        "01_v4_one_head_train.ipynb",
        "02_v4_one_head_compare.ipynb",
        "11_v4_small_4x4_train.ipynb",
        "12_v4_small_4x4_compare.ipynb",
    }
    paths = sorted((EXPERIMENT_ROOT / "notebooks").glob("*.ipynb"))
    assert {path.name for path in paths} == expected
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            ast.parse(source, filename=f"{path}:cell-{index}")


def test_ngb_contains_no_home_directory_defaults():
    forbidden = ("$HOME", "${HOME}", "Path.home(", ".expanduser(", "/home/", "~/")
    paths = [EXPERIMENT_ROOT / "README.md"]
    paths.extend((EXPERIMENT_ROOT / "src").rglob("*.py"))
    paths.extend((EXPERIMENT_ROOT / "configs").glob("*.yaml"))
    paths.extend((EXPERIMENT_ROOT / "notebooks").glob("*.ipynb"))
    violations = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(EXPERIMENT_ROOT)}: {token}")
    assert not violations, "forbidden path defaults found:\n" + "\n".join(violations)
