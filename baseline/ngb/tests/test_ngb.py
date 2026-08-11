from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import sys

import nbformat
import pandas as pd
import pytest

NGB_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = NGB_ROOT.parent
RUNTIME_SRC = BASELINE_ROOT / "nanogpt_one_head" / "src"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

from rg_nanogpt_one_head import (  # noqa: E402
    GPT,
    GPTConfig,
    expected_transformer_matrix_count,
    final_test_summary,
    load_config,
    paired_test_differences,
    roots,
    run_slug,
)
from rg_nanogpt_one_head.model import transformer_matrix_items  # noqa: E402


def _config(name: str) -> dict:
    return load_config(NGB_ROOT / "configs" / name)


def test_v4_protocols_are_separate_tuned_two_epoch_experiments() -> None:
    one = _config("v4_one_head.yaml")
    four = _config("v4_small_4x4.yaml")

    assert one["protocol"]["version"] == 4
    assert four["protocol"]["version"] == 4
    assert run_slug(one) == "v4_one_head"
    assert run_slug(four) == "v4_small_4x4"
    assert one["training"]["target_epochs"] == 2.0
    assert four["training"]["target_epochs"] == 2.0
    assert one["training"]["epoch_interval"] == 0.25
    assert one["training"]["eval_interval_steps"] == 500

    assert one["optimizer_profiles"]["sgd_momentum"]["min_learning_rate"] == 5e-4
    assert one["optimizer_profiles"]["adamw"]["learning_rate"] == 3e-4
    assert one["optimizer_profiles"]["adamw"]["warmup_fraction"] == 0.025
    assert one["optimizer_profiles"]["muon"]["matrix_learning_rate"] == 0.01
    assert one["optimizer_profiles"]["muon"]["aux_learning_rate"] == 2e-4
    resolved = roots(one)
    assert resolved["root"] == Path("/tmp/rg-ngb")
    assert resolved["data"] == Path("/tmp/rg-nanogpt-one-head/data")
    assert resolved["results"] == Path("/tmp/rg-ngb/results/v4_one_head")


def test_one_head_and_four_by_four_parameter_inventories() -> None:
    one_cfg = _config("v4_one_head.yaml")
    four_cfg = _config("v4_small_4x4.yaml")
    one = GPT(GPTConfig(**one_cfg["model"]))
    four = GPT(GPTConfig(**four_cfg["model"]))

    assert one.parameter_count() == 6_662_656
    assert four.parameter_count() == 7_253_248
    assert len(transformer_matrix_items(one)) == 6
    assert len(transformer_matrix_items(four)) == 24
    assert expected_transformer_matrix_count(one_cfg) == 6
    assert expected_transformer_matrix_count(four_cfg) == 24
    assert sum(item[3].numel() for item in transformer_matrix_items(four)) == 786_432


def test_perplexity_interval_is_exponentiated_from_loss_space() -> None:
    frame = pd.DataFrame(
        {
            "optimizer": ["adamw"] * 3,
            "checkpoint": ["final"] * 3,
            "test_loss": [5.0, 6.0, 7.0],
            "test_perplexity": [math.exp(5.0), math.exp(6.0), math.exp(7.0)],
            "test_accuracy": [0.1, 0.2, 0.3],
            "test_bleu": [0.0, 0.1, 0.2],
        }
    )
    summary = final_test_summary(frame)
    loss = summary[summary["metric"].eq("test_loss")].iloc[0]
    perplexity = summary[summary["metric"].eq("test_perplexity")].iloc[0]
    assert perplexity["interval_method"] == "exp_test_loss_student_t"
    assert perplexity["mean"] == pytest.approx(math.exp(loss["mean"]))
    assert perplexity["ci95_lower"] == pytest.approx(math.exp(loss["ci95_lower"]))
    assert perplexity["ci95_lower"] > 0.0


def test_paired_optimizer_contrasts_use_matched_seeds() -> None:
    rows = []
    for optimizer, offset in (("sgd_momentum", 0.0), ("adamw", -0.2), ("muon", -0.1)):
        for seed, base in ((11, 6.0), (13, 6.2), (17, 6.4)):
            rows.append(
                {
                    "optimizer": optimizer,
                    "checkpoint": "final",
                    "seed": seed,
                    "test_loss": base + offset,
                    "test_perplexity": math.exp(base + offset),
                    "test_accuracy": 0.1 - offset,
                    "test_bleu": 0.0,
                }
            )
    contrasts = paired_test_differences(pd.DataFrame(rows))
    row = contrasts[
        contrasts["metric"].eq("test_loss")
        & contrasts["left_optimizer"].eq("sgd_momentum")
        & contrasts["right_optimizer"].eq("adamw")
    ].iloc[0]
    assert row["n"] == 3
    assert row["mean"] == pytest.approx(0.2)


def test_ngb_notebooks_are_valid_python3_papermill_entrypoints() -> None:
    paths = sorted(
        path
        for path in (NGB_ROOT / "notebooks").glob("*.ipynb")
        if not path.name.endswith(".out.ipynb")
    )
    assert [path.name for path in paths] == [
        "01_run_v4_one_head.ipynb",
        "02_compare_v4_one_head.ipynb",
        "03_run_v4_small_4x4.ipynb",
        "04_compare_v4_small_4x4.ipynb",
        "05_compare_v4_architectures.ipynb",
        "06_compare_v3_v4_one_head.ipynb",
    ]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata.kernelspec.name == "python3"
        parameter_cells = [
            cell
            for cell in notebook.cells
            if "parameters" in cell.get("metadata", {}).get("tags", [])
        ]
        assert len(parameter_cells) == 1
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=str(path))


def test_ngb_contains_no_home_or_wrapper_defaults() -> None:
    forbidden = (
        "$HOME",
        "${HOME}",
        "Path.home(",
        ".expanduser(",
        "/home/",
        "~/",
        "scripts/setup_mac.sh",
    )
    this_file = Path(__file__).resolve()
    for path in NGB_ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.suffix in {".pyc"}
            or path.resolve() == this_file
        ):
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"{marker!r} found in {path}"
