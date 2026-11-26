"""Focused scientific-kernel checks; no training and no network access."""

from copy import deepcopy
import json
import math

import pytest

from california_housing_benchmark.analysis import (
    ARMS, SEEDS, _history_row, independent_difference, published_comparison,
    seed_statistics, summarize_rows, write_outputs,
)
from california_housing_benchmark.spectral import aggregate_matrices, confidence_weight


def test_seed_student_t_interval_uses_seed_sample_sd():
    result = seed_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["n"] == 5
    assert result["mean"] == 3.0
    assert result["sample_sd"] == pytest.approx(math.sqrt(2.5))
    assert result["degrees_of_freedom"] == 4
    assert result["t_critical"] == pytest.approx(2.7764451051977987)
    assert result["ci95_half_width"] == pytest.approx(1.9632431614775607)
    assert result["ci95_low"] == pytest.approx(1.0367568385224393)
    assert result["ci95_high"] == pytest.approx(4.9632431614775605)
    assert result["complete_five_seeds"]


def test_missing_seed_never_becomes_a_zero_or_complete_interval():
    result = seed_statistics([1.0, None, float("nan"), True])
    assert result["n"] == 1
    assert result["mean"] == 1.0
    assert result["sample_sd"] is None
    assert result["ci95_half_width"] is None
    assert not result["complete_five_seeds"]
    assert seed_statistics([])["mean"] is None


def test_pairing_cancels_shared_seed_variation_and_keeps_endpoint_definitions():
    rows = []
    for seed in SEEDS:
        for endpoint, step in (
            ("fixed_updates", 2000),
            ("fixed_updates", 20000),
            ("primary_best_validation", 1000 + seed),
        ):
            for arm in ARMS:
                offset = {"sgdm": 0.1, "adamw": 0.0, "modified_adamw": -0.02}[arm]
                rows.append({
                    "seed": seed, "optimizer": arm, "endpoint": endpoint,
                    "step": step, "test_rmse": 0.5 + 0.01 * (seed - SEEDS[0]) + offset,
                })
    summaries, paired = summarize_rows(rows)
    effects = [row for row in paired if row["metric"] == "test_rmse"]
    assert len(effects) == 3
    for effect in effects:
        assert effect["n"] == 5
        assert effect["mean"] == pytest.approx(-0.02)
        assert effect["ci95_half_width"] == pytest.approx(0.0, abs=1e-12)
        assert set(effect["seed_differences"]) == {str(seed) for seed in SEEDS}
    primary = [row for row in summaries if row["endpoint"] == "primary_best_validation"
               and row["optimizer"] == "adamw" and row["metric"] == "test_rmse"][0]
    assert primary["step"] is None
    assert primary["n"] == 5
    assert primary["sample_sd"] > 0
    with pytest.raises(ValueError, match="duplicate seed"):
        summarize_rows(rows + [rows[0]])


def test_independent_reference_uses_welch_not_paired_uncertainty():
    current = seed_statistics([1.0, 2.0, 3.0])
    reference = seed_statistics([2.0, 3.0, 4.0])
    result = independent_difference(current, reference)
    assert result["mean_difference"] == -1.0
    assert result["degrees_of_freedom"] == pytest.approx(4.0)
    expected = 2.7764451051977987 * math.sqrt(2.0 / 3.0)
    assert result["ci95_half_width"] == pytest.approx(expected)
    assert result["ci95_low"] == pytest.approx(-1.0 - expected)
    assert result["ci95_high"] == pytest.approx(-1.0 + expected)
    assert independent_difference(seed_statistics([1.0]), reference)["ci95_half_width"] is None


def matrix_record():
    return {
        "matrix": "backbone.blocks.1.0.weight",
        "shape": [784, 784],
        "fit": {"status": "success", "alpha": 2.4, "D": 0.1,
                "num_pl_spikes": 100, "num_evals": 784},
        "eigenvalues": [float(index) for index in range(784)],
    }


def test_one_eligible_matrix_has_no_layer_dispersion_or_survivor_average():
    matrix = matrix_record()
    declared = [matrix["matrix"]]
    before = deepcopy(matrix)
    assert confidence_weight(matrix["fit"], 784) == pytest.approx(10.0 / 0.12)
    result = aggregate_matrices([matrix], declared)
    assert result["complete"]
    assert result["confidence_tail_alpha"] == 2.4
    assert result["layer_alpha_sd"] is None
    assert not result["layer_dispersion_defined"]
    assert result["model_average_equals_single_matrix_fit"]
    assert matrix == before
    assert not aggregate_matrices([], declared)["complete"]
    assert not aggregate_matrices([matrix, matrix], declared)["complete"]
    assert not aggregate_matrices([matrix], ["wrong.matrix"])["complete"]
    matrix["fit"]["status"] = "failed"
    invalid = aggregate_matrices([matrix], declared)
    assert not invalid["complete"]
    assert invalid["confidence_tail_alpha"] is None


def test_raw_fit_recalculation_detects_a_tampered_stored_alpha():
    matrix = matrix_record()
    metrics = {"rmse": 0.5, "mse": 0.25, "mae": 0.3, "r2": 0.8, "mape_percent": 18.0}
    measurement = {
        "step": 2000, "epoch": 38, "train": metrics, "validation": metrics,
        "test": metrics, "spectral": {
            "eligible_matrices": [matrix["matrix"]], "n_eligible": 1,
            "matrices": [matrix], "confidence_tail_alpha": 9.0, "layer_alpha_sd": None,
        },
    }
    errors, warnings = [], []
    row = _history_row({"seed": SEEDS[0], "optimizer": "adamw"},
                       measurement, "fixed_updates", errors, warnings)
    assert row["confidence_tail_alpha"] == 2.4
    assert any("stored average alpha differs from the raw fit" in error for error in errors)
    assert not warnings


def test_published_reference_is_recomputed_from_the_authors_seed_records():
    rows, _ = summarize_rows([])
    errors = []
    result = published_comparison(rows, errors)
    assert not errors, errors
    reference = {item["optimizer"]: item["recalculated_test_rmse"]
                 for item in result["references"]}
    assert reference["adamw"]["n"] == 10
    assert reference["adamw"]["mean"] == pytest.approx(0.49358551758796604)
    assert reference["adamw"]["sample_sd"] == pytest.approx(0.011003265043117574)
    assert reference["muon-ema"]["mean"] == pytest.approx(0.4708079134886385)
    assert result["native_replication_minus_published_adamw"]["mean_difference"] is None


def test_missing_cohort_produces_explicitly_incomplete_diagnostics(tmp_path):
    summary = write_outputs(tmp_path, allow_incomplete=True)
    assert not summary["complete_five_seeds"]
    assert summary["n_completed_runs"] == 0
    assert summary["audit_errors"]
    assert not any("published" in error for error in summary["audit_errors"])
    outputs = (
        "summary.json", "execution_cohort.json", "published_comparison.json",
        "raw_matrix_fits.json", "seed_metrics.csv", "summary.csv",
        "paired_effects.csv", "RESULTS.md", "figures/alpha_vs_updates.png",
        "figures/rmse_vs_updates.pdf",
    )
    assert all((tmp_path / "analysis" / path).is_file() for path in outputs)
    saved = json.loads((tmp_path / "analysis" / "summary.json").read_text())
    assert not saved["complete_five_seeds"]
    report = (tmp_path / "analysis" / "RESULTS.md").read_text()
    assert "INCOMPLETE OR FAILED AUDIT" in report
    assert "not 50% error" in report
    with pytest.raises(ValueError, match="Cohort incomplete or failed audit"):
        write_outputs(tmp_path)
