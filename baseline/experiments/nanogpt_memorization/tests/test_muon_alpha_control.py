"""Dependency-light tests for the Muon high-alpha control workflow."""
from pathlib import Path
import json
import sys

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import muon_alpha_control as m
import compare_membership as c


def spectral_run(path, rows):
    path.mkdir(parents=True)
    (path / "spectral").mkdir()
    (path / "complete.json").write_text(json.dumps({"status": "complete"}))
    for step, values in rows.items():
        table = pd.DataFrame({
            "step": [step] * 6,
            "matrix_name": ["Q", "K", "V", "O", "MI", "MO"],
            "alpha_clip_xmax": values,
            "alpha_raw": [x + 0.05 for x in values],
            "tail_support_at_least_20": [True] * 6,
        })
        table.to_csv(path / "spectral" / f"step_{step:08d}.csv", index=False)


def test_alpha_gate_requires_every_raw_and_clipped_value(tmp_path):
    good = tmp_path / "good"
    spectral_run(good, {0: [3, 3, 3, 3, 3, 3], 100: [2.1] * 6})
    gate = m.alpha_gate(good, 2.0)
    assert gate["passed"] and gate["min_alpha_both_fits"] == pytest.approx(2.1)
    bad = tmp_path / "bad"
    spectral_run(bad, {0: [3] * 6, 100: [2.1, 2.1, 1.99, 2.1, 2.1, 2.1]})
    gate = m.alpha_gate(bad, 2.0)
    assert not gate["passed"] and gate["violations"] == 1 and gate["min_matrix"] == "V"


def test_scaled_profile_changes_only_matrix_lr_pair():
    source = {"optimizer_profiles": {"muon": {
        "family": "muon", "matrix_learning_rate": 0.02, "matrix_min_learning_rate": 0.002,
        "aux_learning_rate": 0.0003, "aux_min_learning_rate": 0.00003,
        "momentum": 0.95, "matrix_weight_decay": 0.01,
    }}}
    def resolver(src, optimizer, recipe):
        return dict(src["optimizer_profiles"][optimizer])
    profile = m.scaled_profile(resolver, 0.25)(source, "muon", "anything")
    assert profile["matrix_learning_rate"] == pytest.approx(0.005)
    assert profile["matrix_min_learning_rate"] == pytest.approx(0.0005)
    assert profile["aux_learning_rate"] == 0.0003
    assert profile["momentum"] == 0.95
    assert profile["control_matrix_lr_scale"] == 0.25


def membership_bundle(root, family, effect, auc, *, seed=1337):
    membership = root / family / "membership"
    audit = root / family / "audit"
    run = root / family / "run"
    membership.mkdir(parents=True)
    audit.mkdir(parents=True)
    run.mkdir(parents=True)
    pd.DataFrame({
        "step": [100, 200],
        "baseline_corrected_membership_effect": [effect / 2, effect],
        "auc_baseline_corrected": [0.5, auc],
        "raw_fresh_minus_seen_nll": [0.0, effect],
    }).to_csv(membership / "membership_stats.csv", index=False)
    (membership / "membership_metadata.json").write_text(json.dumps({"source_audit": str(audit)}))
    (audit / "protocol.json").write_text(json.dumps({"source_run": str(run)}))
    manifest = {
        "condition": "verbatim", "stage": "full", "seed": seed,
        "data_sha256": "data", "initial_model_sha256": "init",
        "source_model": {"n_layer": 1}, "device": {"device": "mps"},
        "profile": {"family": family},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return membership, run


def test_compare_requires_passing_gate_and_matched_runs(tmp_path):
    adamw, _ = membership_bundle(tmp_path, "adamw", 0.04, 0.61)
    muon, muon_run = membership_bundle(tmp_path, "muon", 0.01, 0.52)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"passed": True, "threshold": 2.0,
                                "min_alpha_both_fits": 2.2, "run_dir": str(muon_run)}))
    out = c.compare(adamw, muon, gate, tmp_path / "comparison")
    table = pd.read_csv(out / "optimizer_membership_comparison.csv")
    assert table.iloc[-1]["muon_minus_adamw_membership_effect"] == pytest.approx(-0.03)
    assert "qualified high-alpha Muon" in (out / "comparison_report.md").read_text()


def test_compare_rejects_failed_alpha_gate(tmp_path):
    adamw, _ = membership_bundle(tmp_path, "adamw", 0.04, 0.61)
    muon, muon_run = membership_bundle(tmp_path, "muon", 0.01, 0.52)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"passed": False, "run_dir": str(muon_run)}))
    with pytest.raises(ValueError, match="high-alpha"):
        c.compare(adamw, muon, gate, tmp_path / "comparison")
