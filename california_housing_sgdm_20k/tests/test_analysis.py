"""Selection truncation, complete-grid reuse, failed fits, and seed uncertainty."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from california_housing_sgdm_20k import analysis


def metrics(rmse):
    return {"rmse": rmse, "mse": rmse ** 2, "mae": rmse * .75,
            "r2": .5, "mape_percent": 20.}


def measurement(step, value, failed=False):
    fit = {"status": "failed" if failed else "success", "alpha": -1 if failed else 2.2,
           "D": -1 if failed else .05, "num_pl_spikes": -1 if failed else 100,
           "num_evals": 784, "xmin": -1 if failed else .2}
    name = "backbone.blocks.1.0.weight"
    spectral = {"n_eligible": 1, "n_valid": 0 if failed else 1, "complete": not failed,
                "confidence_tail_alpha": None if failed else 2.2,
                "eligible_matrices": [name], "layer_alpha_sd": None,
                "matrices": [{"matrix": name, "shape": [784, 784], "fit": fit,
                              "diagnostic": "explicit failed fit" if failed else None}]}
    return {"step": step, "epoch": step // 52, "train": metrics(value),
            "validation": metrics(value), "test": None, "spectral": spectral,
            "weights_sha256": "a" * 64, "spectral_weights_sha256": "a" * 64}


def complete_validation_record():
    return {"epoch_size": 52,
            "history": [measurement(step, 1. if step == 0 else .6) for step in analysis.HISTORY_STEPS],
            "epoch_history": [{"step": step, "validation": {"rmse": .5}} for step in range(52, 20001, 52)]}


def test_selection_uses_only_initialization_completed_epochs_and_final_cap():
    record = complete_validation_record()
    record["history"][2]["validation"]["rmse"] = .00001  # 5k is not a completed epoch.
    record["history"].append(measurement(50000, .000001))
    record["epoch_history"].append({"step": 20020, "validation": {"rmse": .000001}})
    record["best_validation"] = measurement(50000, .000001)
    best = analysis.minimum_validation_within_cap(record)
    assert best == {"step": 52, "validation_rmse": .5, "n_selection_observations": 386}
    record["history"][0]["validation"]["rmse"] = .3
    assert analysis.minimum_validation_within_cap(record)["step"] == 0
    record["history"][-2]["validation"]["rmse"] = .2
    assert analysis.minimum_validation_within_cap(record)["step"] == 20000


def test_missing_epoch_and_fresh_budget_above_cap_are_rejected():
    record = complete_validation_record()
    del record["epoch_history"][12]
    with pytest.raises(ValueError, match="coverage"):
        analysis.minimum_validation_within_cap(record)
    plan = json.loads((analysis.PACKAGE / "plan.json").read_text())
    plan["total_updates"] = 50000
    with pytest.raises(ValueError, match="stop at 20000"):
        analysis.validate_cap(plan)


def test_failed_spectrum_is_preserved_but_does_not_remove_performance():
    row = analysis.checked_measurement(measurement(20000, 1.17, failed=True), "sgdm_30", 3338, test=False)
    assert row["confidence_tail_alpha"] is None and not row["spectral_fit_valid"]
    assert row["validation_rmse"] == 1.17
    assert row["fit_status"] == "failed"
    inconsistent = measurement(20000, 1.17, failed=True)
    inconsistent["spectral"]["confidence_tail_alpha"] = 68.3
    with pytest.raises(ValueError, match="explicitly undefined"):
        analysis.checked_measurement(inconsistent, "sgdm_30", 3338, test=False)


def test_alpha_raw_fit_and_same_weight_binding_are_checked():
    row = measurement(20000, .5)
    row["spectral"]["confidence_tail_alpha"] = 1.5
    with pytest.raises(ValueError, match="raw fit"):
        analysis.checked_measurement(row, "x", 1, test=False)
    row = measurement(20000, .5)
    row["spectral_weights_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="weight hashes"):
        analysis.checked_measurement(row, "x", 1, test=False)


def test_seed_statistics_use_sample_sd_and_student_t_not_layer_dispersion():
    stats = analysis.seed_statistics([1., 2., 3., 4., 5.])
    assert stats["n"] == 5 and stats["mean"] == 3.
    assert stats["sample_sd"] == pytest.approx(math.sqrt(2.5))
    assert stats["degrees_of_freedom"] == 4
    assert stats["ci95_half_width"] == pytest.approx(1.9632431615)
    missing = analysis.seed_statistics([1., 2., None, 4., 5.])
    assert missing["n"] == 4 and not missing["complete_expected_seeds"]
    effect = analysis.welch_effect([1., 2., 3., 4., 5.], [2., 3., 4., 5., 6.])
    assert effect["mean_difference"] == -1
    assert effect["degrees_of_freedom"] == pytest.approx(8)
    assert "independent-seed" in effect["ci_method"]


@pytest.fixture
def complete_grid(tmp_path):
    plan = json.loads((analysis.PACKAGE / "plan.json").read_text())
    source = plan["parent_source"]
    old_plan = json.loads((analysis.PACKAGE.parent / "california_housing_sgdm_search/plan.json").read_text())
    sources = {name: analysis.file_sha256(analysis.PACKAGE.parent / "california_housing_benchmark" / name)
               for name in ("data.py", "train.py", "optimizer.py", "spectral.py")}
    sources["long_train.py"] = analysis.file_sha256(analysis.PACKAGE.parent / "california_housing_sgdm_search/long_train.py")
    identity = {"data_manifest_sha256": "data-fixture", "source_manifest_sha256": "new-manifest",
                "orchestration_source_manifest_sha256": "new-manifest", "training_source_manifest_sha256": source["source_manifest_sha256"],
                "plan_file_sha256": "plan-fixture", "github_run_id": "new-run", "launch_commit": "new-source"}
    def archive(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(value, sort_keys=True) + "\n").encode()
        path.write_bytes(payload)
        return {"local_path": relative, "repository_path": "old/" + relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "git_blob_sha": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
                "bytes": len(payload), "repository": source["repository"], "archive_commit": source["archive_commit"]}
    inventory = {"parent_source": source, "development": []}
    for index, candidate in enumerate(plan["candidates"]):
        for seed in analysis.DEV_SEEDS:
            record = complete_validation_record()
            record.update(optimizer="sgdm", seed=seed, phase="development", config=candidate["config"],
                          completed_updates=50000, status="complete", plan_sha256=analysis.canonical_sha256(old_plan),
                          source_bundle_sha256=analysis.canonical_sha256(sources),
                          source_manifest_sha256=source["source_manifest_sha256"], data_manifest_sha256=identity["data_manifest_sha256"],
                          execution={"github": {"GITHUB_RUN_ID": source["run_id"], "GITHUB_SHA": source["launch_commit"],
                                                 "GITHUB_REPOSITORY": source["repository"]}})
            score = .5 + index * .001
            for epoch in record["epoch_history"]:
                epoch["validation"]["rmse"] = score
            record["history"][-1] = measurement(20000, score + .02, failed=index >= 30)
            if index == 31:
                # A later improvement and a non-epoch intermediate observation cannot win.
                record["history"][2] = measurement(5000, .00001)
                record["history"].append(measurement(50000, .000001))
                record["epoch_history"].append({"step": 20020, "validation": {"rmse": .000001}})
                record["best_validation"] = measurement(50000, .000001)
            execution = {"seed": seed, "candidate_id": candidate["id"], "phase": "development",
                         "github_run_id": source["run_id"], "launch_commit": source["launch_commit"]}
            prefix = f"inputs/development/{candidate['id']}/seed_{seed}"
            inventory["development"].append({"candidate_id": candidate["id"], "seed": seed,
                "action": "reuse_20k_history", "new_updates": 0, "source_completed_updates": 50000, "source_status": "complete",
                "result": archive(prefix + "/result.json", record),
                "github_execution": archive(prefix + "/github_execution.json", execution)})
    analysis.write_json(tmp_path / "source_inventory.json", inventory)
    identity["source_inventory_sha256"] = analysis.file_sha256(tmp_path / "source_inventory.json")
    return tmp_path, plan, identity


def test_all64_prefixes_select_without_later_data_or_spectral_exclusion(complete_grid):
    root, plan, identity = complete_grid
    rows, scores, winner = analysis.compute_development(root, plan, identity)
    assert len(rows) == 64 and len(scores) == 32
    assert winner["candidate_id"] == "sgdm_00" and winner["mean_min_validation_rmse"] == .5
    assert sum(row["spectral_fit_valid"] for row in rows) == 60
    assert all(row["source_completed_updates"] == 50000 and row["analysis_max_step"] == 20000 for row in rows)
    assert all(row["new_development_updates"] == 0 for row in rows)
    assert next(row for row in scores if row["candidate_id"] == "sgdm_31")["mean_min_validation_rmse"] == .531
    frozen = analysis.expected_frozen(plan, identity, rows, scores, winner)
    analysis.write_json(root / "frozen_selection.json", frozen)
    assert analysis.validate_frozen_selection(root, plan, identity) == frozen
    frozen["selected_candidate_id"] = "sgdm_31"
    analysis.write_json(root / "frozen_selection.json", frozen)
    with pytest.raises(ValueError, match="Frozen selection differs"):
        analysis.validate_frozen_selection(root, plan, identity)


def test_grid_input_hash_and_no_test_gate(complete_grid):
    root, plan, identity = complete_grid
    path = root / "inputs/development/sgdm_00/seed_3338/result.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Input hash differs"):
        analysis.compute_development(root, plan, identity)
