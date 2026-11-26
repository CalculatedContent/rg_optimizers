"""Analysis invariants on synthetic fixtures; no experimental training."""
from copy import deepcopy
import json
import math

import pytest
from scipy.stats import t as student_t

from california_housing_sgdm_200k import analysis as module


def spectrum(alpha=2.4):
    weight = math.sqrt(100) / (.03 + .02)
    return {"n_eligible": 1, "n_valid": 1, "complete": True,
            "eligible_matrices": ["backbone.blocks.1.0.weight"],
            "confidence_tail_alpha": alpha, "layer_alpha_sd": None,
            "layer_dispersion_defined": False,
            "matrices": [{"matrix": "backbone.blocks.1.0.weight", "shape": [784, 784],
                          "alpha": alpha, "weight": weight,
                          "fit": {"alpha": alpha, "alpha_weighted": 99.9, "D": .03,
                                  "num_pl_spikes": 100, "num_evals": 784, "status": "success"}}]}


def measurement(step=100000, alpha=2.4):
    metrics = {"rmse": .5, "mae": .4, "mse": .25, "r2": .8}
    return {"step": step, "weights_sha256": "a" * 64, "spectral_weights_sha256": "a" * 64,
            "spectral": spectrum(alpha), "train": dict(metrics), "validation": dict(metrics),
            "test": dict(metrics)}


def test_seed_interval_uses_sample_sd_and_t_four_degrees_of_freedom():
    stats = module.seed_statistics([1, 2, 3, 4, 5])
    assert stats["mean"] == 3
    assert stats["sample_sd"] == pytest.approx(math.sqrt(2.5))
    assert stats["ci95_half_width"] == pytest.approx(student_t.ppf(.975, 4) / math.sqrt(2))
    assert stats["degrees_of_freedom"] == 4
    assert stats["complete_expected_seeds"] is True
    with pytest.raises(ValueError, match="finite"):
        module.seed_statistics([1, float("nan")])


def test_paired_interval_keeps_seed_covariance():
    earlier = dict(zip(module.SEEDS, [1, 10, 100, 1000, 10000]))
    later = {seed: value + 2 for seed, value in earlier.items()}
    effect = module.paired_effect(later, earlier)
    assert effect["mean"] == 2
    assert effect["sample_sd"] == 0
    assert effect["ci95_half_width"] == 0
    assert effect["n"] == 5


def test_alpha_uses_ordinary_fit_and_no_layer_dispersion():
    alpha, weight = module.spectral_alpha(spectrum())
    assert alpha == pytest.approx(2.4)
    assert weight == pytest.approx(200)
    bad = spectrum()
    bad["confidence_tail_alpha"] = 99.9
    with pytest.raises(ValueError, match="independent recomputation"):
        module.spectral_alpha(bad)
    bad = spectrum()
    bad["layer_alpha_sd"] = .1
    with pytest.raises(ValueError, match="undefined"):
        module.spectral_alpha(bad)


def test_fits_must_be_eligible_and_complete():
    for alter in (lambda s: s.update(n_eligible=2),
                  lambda s: s["matrices"][0].update(shape=[784, 8]),
                  lambda s: s["matrices"][0]["fit"].update(num_pl_spikes=0),
                  lambda s: s.update(complete=False)):
        value = spectrum()
        alter(value)
        with pytest.raises(ValueError):
            module.spectral_alpha(value)


def test_joint_metric_point_requires_same_weights():
    row = module.measurement_row(measurement(), 2340, "new")
    assert row["same_checkpoint_verified"] is True
    assert row["confidence_tail_alpha"] == pytest.approx(2.4)
    bad = measurement()
    bad["spectral_weights_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="matching checkpoint"):
        module.measurement_row(bad, 2340, "new")
    bad = measurement()
    bad["test"]["mse"] = .30
    with pytest.raises(ValueError, match="RMSE squared"):
        module.measurement_row(bad, 2340, "new")


def test_history_rejects_duplicate_and_missing_scheduled_observations():
    errors = []
    result = {"history": [measurement(100000), measurement(100000), measurement(110000)]}
    module._history(result, 2340, "new", 100000, 110000, errors)
    assert any("Duplicate" in value for value in errors)
    assert any("105000" in value for value in errors)


def test_empty_or_malformed_inputs_preserve_an_incomplete_report(tmp_path):
    parent = tmp_path / "parent" / "seed_2340"
    parent.mkdir(parents=True)
    (parent / "result.json").write_text("malformed")
    summary = module.analyze(tmp_path, make_plots=False)
    assert summary["complete_five_seeds"] is False
    assert summary["n_completed"] == 0
    assert summary["n_parent_completed"] == 0
    assert summary["audit_errors"]
    assert (tmp_path / "analysis" / "RESULTS.md").exists()
    assert json.loads((tmp_path / "analysis" / "summary.json").read_text())["complete_five_seeds"] is False


def test_actual_checkpoint_must_bind_weights_and_full_resume_state(tmp_path):
    import torch
    from california_housing_benchmark.train import model_hash, rng_state
    state = {"layer.weight": torch.tensor([[1., 2.], [3., 4.]])}
    record = {"config": module.CONFIG, "plan_sha256": "plan", "data_manifest_sha256": "data",
              "source_bundle_sha256": "source"}
    checkpoint = {"checkpoint_kind": "full_resume", "completed_updates": 200000,
                  "identity": {"seed": 2340, **record}, "model": state,
                  "optimizer": {}, "rng": rng_state(), "batch_generator": torch.Generator().get_state(),
                  "epoch_permutation": None, "next_offset": 0, "minibatch_sha256": "digest",
                  "completed_epochs": 0, "epoch_loss_sum": 0., "epoch_examples": 0}
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    errors = []
    audited = module._checkpoint(path, 2340, 200000, {"weights_sha256": model_hash(state)}, record, errors)
    assert not errors
    assert len(audited["sha256"]) == 64
    errors = []
    module._checkpoint(path, 2341, 200000, {"weights_sha256": "wrong"}, record, errors)
    assert any("seed" in error for error in errors)
    assert any("weights" in error for error in errors)
    del checkpoint["rng"]
    torch.save(checkpoint, path)
    errors = []
    module._checkpoint(path, 2340, 200000, {"weights_sha256": model_hash(state)}, record, errors)
    assert any("full-resume field rng" in error for error in errors)


def test_complete_summary_uses_five_seeds_and_paired_budgets(tmp_path, monkeypatch):
    # Statistical integration fixture: bypass source auditing, not numerical
    # aggregation. These constructed values are never experiment evidence.
    contract = {"cohort_origin": {"github_run_id": "fixture"}, "parent_manifest": {}}
    monkeypatch.setattr(module, "_load_contract", lambda root, errors: contract)

    def audit(root, seed, contract):
        index = seed - 2340
        rows = {}
        for step in range(20000, 200001, 5000):
            row = module.measurement_row(measurement(step, alpha=2.5 + .1 * index - step / 1e6), seed, "fixture")
            row["test_rmse"] = .6 + .01 * index - step / 1e6
            row["test_mse"] = row["test_rmse"] ** 2
            rows[step] = row
        return rows, {"seed": seed, "valid": True, "audit_errors": []}

    monkeypatch.setattr(module, "_audit_seed", audit)
    for seed in module.SEEDS:
        directory = tmp_path / "parent" / f"seed_{seed}"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({"status": "complete", "seed": seed, "completed_updates": 100000}))
    summary = module.analyze(tmp_path, make_plots=False)
    assert summary["complete_five_seeds"] is True
    assert summary["n_completed"] == summary["n_parent_completed"] == 5
    assert summary["endpoints"]["200000"]["test_rmse"]["mean"] == pytest.approx(.42)
    change = summary["paired_changes"]["200,000 minus 100,000"]["test_rmse"]
    assert change["mean"] == pytest.approx(-.1)
    assert change["ci95_half_width"] == pytest.approx(0, abs=1e-15)
    assert summary["plots_generated"] is False
    assert len((tmp_path / "analysis" / "endpoint_seed_values.csv").read_text().splitlines()) == 16
