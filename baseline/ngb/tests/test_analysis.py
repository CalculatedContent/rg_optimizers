from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

import rg_ngb.analysis as analysis


def test_matched_seed_discovery_uses_intersection(tmp_path, monkeypatch):
    complete = {
        "sgd_momentum": {1, 2, 3, 4},
        "adamw": {2, 3, 4, 5},
        "muon": {3, 4, 5, 6},
    }
    for optimizer, seeds in complete.items():
        for seed in seeds:
            (tmp_path / optimizer / f"seed_{seed}").mkdir(parents=True)

    monkeypatch.setattr(
        analysis,
        "run_is_complete",
        lambda root, optimizer, seed: int(seed) in complete[str(optimizer)],
    )
    assert analysis.discover_matched_seeds(tmp_path) == (3, 4)
    assert analysis.discover_matched_seeds(tmp_path, requested=(4,)) == (4,)
    with pytest.raises(FileNotFoundError, match="requested seeds"):
        analysis.discover_matched_seeds(tmp_path, requested=(2, 3))


def test_perplexity_interval_is_exponentiated_from_loss_space():
    frame = pd.DataFrame(
        {
            "optimizer": ["adamw"] * 3,
            "optimizer_label": ["AdamW"] * 3,
            "seed": [1, 2, 3],
            "checkpoint": ["final"] * 3,
            "step": [10] * 3,
            "test_loss": [5.0, 5.2, 5.4],
            "test_accuracy": [0.1, 0.2, 0.3],
            "test_bleu": [0.0, 0.1, 0.2],
        }
    )
    summary = analysis.final_test_summary(frame)
    loss = summary[summary["metric"].eq("test_loss")].iloc[0]
    perplexity = summary[summary["metric"].eq("test_perplexity")].iloc[0]
    assert perplexity["interval_method"] == "exp_of_loss_space_student_t_interval"
    assert perplexity["mean"] == pytest.approx(math.exp(loss["mean"]))
    assert perplexity["ci95_lower"] == pytest.approx(math.exp(loss["ci95_lower"]))
    assert perplexity["ci95_upper"] == pytest.approx(math.exp(loss["ci95_upper"]))
    assert perplexity["ci95_lower"] > 0


def test_paired_differences_preserve_seed_matching():
    rows = []
    for optimizer, offset in (("sgd_momentum", 0.0), ("adamw", -0.2), ("muon", -0.3)):
        for seed, base in ((1, 6.0), (2, 7.0), (3, 8.0)):
            rows.append(
                {
                    "optimizer": optimizer,
                    "optimizer_label": optimizer,
                    "seed": seed,
                    "checkpoint": "final",
                    "step": 10,
                    "test_loss": base + offset,
                    "test_accuracy": 0.1 - offset,
                    "test_bleu": 0.2 - offset,
                }
            )
            rows.append({**rows[-1], "checkpoint": "validation_selected", "step": 8})
    result = analysis.paired_optimizer_differences(pd.DataFrame(rows))
    contrast = result[
        result["checkpoint"].eq("final")
        & result["metric"].eq("test_loss")
        & result["left_optimizer"].eq("adamw")
        & result["right_optimizer"].eq("sgd_momentum")
    ].iloc[0]
    assert contrast["n"] == 3
    assert contrast["mean"] == pytest.approx(-0.2)
    assert contrast["ci95_half_width"] == pytest.approx(0.0, abs=1e-12)


def test_mean_ci95_uses_student_t_for_three_runs():
    result = analysis.mean_ci95([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["ci95_half_width"] == pytest.approx(4.3026527297 / math.sqrt(3))
