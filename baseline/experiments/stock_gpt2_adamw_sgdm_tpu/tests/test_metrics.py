"""Cheap contract tests plus a real, tiny CPU WeightWatcher capability test."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest


from .. import metrics


def payload(n_layers=1, include_embeddings=False):
    names = metrics.expected_matrix_names(n_layers, include_embeddings)
    return {"run_id": "unit-test", "step": 5, "tokens_seen": 81920,
            "validation": {"step": 5, "tokens_seen": 81920, "val_nll": 3.75, "evaluation_tokens": 1024},
            "matrices": {name: np.eye(4, dtype=np.float32) for name in names},
            "manifest": {"spectral_tracking": {"n_layers": n_layers, "include_embeddings": include_embeddings,
                                               "expected_matrices": names}}}


def native_rows(matrices, **options):
    clipped = options["fix_fingers"] == "clip_xmax"
    rows = []
    for index, name in enumerate(matrices):
        # Independent numbers detect accidental raw/clipped quality reuse.
        alpha = index + (3.0 if clipped else 2.0)
        tail = (index + 2) ** 2 * (4 if clipped else 25)
        D = (index + 1) * (0.02 if clipped else 0.05)
        rows.append({"longname": name, "name": "Linear", "alpha": alpha, "raw_alpha": 999.0,
                     "D": D, "xmin": 0.5 if clipped else 0.25, "xmax": 9.0 if clipped else 16.0,
                     "observer_n_tail": tail, "observer_n_tail_source": "test_exact_interval_count",
                     "num_pl_spikes": tail + (4 if clipped else 0), "num_fingers": 3 if clipped else 0,
                     "status": "success", "detX_num": tail + 10, "num_ERG_spikes": tail + 10,
                     "ERG_gap": 6 if clipped else 10, "num_traps": index,
                     "rand_num_spikes": index, "rand_distance": 0.123,
                     "native_future_field": [1, 2, 3]})
    return pd.DataFrame(rows)


def cpu_state(n_layers=1, width=4):
    torch = pytest.importorskip("torch")
    result = {}
    for block in range(n_layers):
        base = f"transformer.h.{block}."
        result[base + "attn.c_attn.weight"] = torch.arange(3 * width * width, dtype=torch.float32).reshape(3 * width, width)
        result[base + "attn.c_proj.weight"] = torch.ones(width, width)
        result[base + "mlp.c_fc.weight"] = torch.ones(4 * width, width) * 2
        result[base + "mlp.c_proj.weight"] = torch.ones(width, 4 * width) * 3
    result["transformer.wte.weight"] = torch.ones(32, width)
    result["transformer.wpe.weight"] = torch.ones(8, width)
    result["lm_head.weight"] = result["transformer.wte.weight"]
    return result


def checkpoint(step=0, *, keep_snapshots=False):
    return {"step": step, "tokens_seen": step * 16, "run_id": "unit-test",
            "validation": {"step": step, "val_nll": 4.0}, "model": cpu_state(),
            "config": {"n_layer": 1, "keep_snapshots": keep_snapshots}, "manifest": {"test": True}}


def test_extracts_all_72_packed_matrices_and_separate_tied_embedding_without_rng_draws():
    torch = pytest.importorskip("torch")
    state = cpu_state(n_layers=12)
    before = torch.random.get_rng_state().clone()
    found = metrics.extract_matrices(state, n_layers=12, include_embeddings=True)
    assert list(found) == metrics.expected_matrix_names(12, True)
    assert len(found) == 74
    assert torch.equal(before, torch.random.get_rng_state())
    for block in range(12):
        packed = state[f"transformer.h.{block}.attn.c_attn.weight"]
        for index, role in enumerate(("Q", "K", "V")):
            assert torch.equal(found[f"L{block:02d}_W_{role}"], packed[index * 4:(index + 1) * 4])
            assert found[f"L{block:02d}_W_{role}"].untyped_storage().nbytes() == 4 * 4 * 4
    assert "lm_head" not in found
    old = found["L00_W_Q"].clone()
    state["transformer.h.0.attn.c_attn.weight"].fill_(999)
    assert torch.equal(old, found["L00_W_Q"])


def test_extract_rejects_incomplete_or_transposed_stock_model():
    state = cpu_state()
    missing = dict(state)
    del missing["transformer.h.0.mlp.c_proj.weight"]
    with pytest.raises(KeyError, match="Missing"):
        metrics.extract_matrices(missing, 1)
    state["transformer.h.0.attn.c_attn.weight"] = state["transformer.h.0.attn.c_attn.weight"].T
    with pytest.raises(ValueError, match="row-packed"):
        metrics.extract_matrices(state, 1)


def test_two_fits_use_their_own_ks_tail_and_interval_and_all_weighted_means():
    result = metrics.measure(payload(), analyzer=native_rows)
    first = result["layers"][0]
    assert first["alpha_raw"] == 2.0  # never clipped frame's raw_alpha=999
    assert first["alpha_clip_xmax"] == 3.0
    assert (first["D_raw"], first["n_tail_raw"], first["xmin_raw"], first["xmax_raw"]) == (0.05, 100, 0.25, 16)
    assert (first["D_clip_xmax"], first["n_tail_clip_xmax"], first["xmin_clip_xmax"], first["xmax_clip_xmax"]) == (0.02, 16, 0.5, 9)
    assert first["num_pl_spikes_clip_xmax"] == 20  # native ERG count intentionally differs
    assert first["native_clip_xmax"]["native_future_field"] == [1, 2, 3]
    assert first["native_raw"]["rand_distance"] == 0.123
    summary = result["summary"]
    for fit in ("raw", "clip_xmax"):
        rows = result["layers"]
        for label, weights in (
            ("", [1.0] * 6),
            ("confidence_weighted_", [1 / (r[f"D_{fit}"] + 0.02) for r in rows]),
            ("tail_weighted_", [math.sqrt(r[f"n_tail_{fit}"]) for r in rows]),
            ("confidence_tail_weighted_", [math.sqrt(r[f"n_tail_{fit}"]) / (r[f"D_{fit}"] + 0.02) for r in rows]),
        ):
            expected = sum(r[f"alpha_{fit}"] * w for r, w in zip(rows, weights)) / sum(weights)
            assert summary[f"alpha_{fit}_{label}mean"] == pytest.approx(expected)
            assert summary[f"alpha_{fit}_{label}effective_n"] == pytest.approx(sum(weights) ** 2 / sum(w * w for w in weights))
            assert summary[f"alpha_{fit}_{label}valid_count"] == 6
    assert result["coverage"]["status"] == "complete"
    assert result["summary"]["val_nll"] == 3.75
    assert len(result["groups"]) == 7  # six roles and one block


def test_72_matrix_coverage_keeps_embeddings_out_of_canonical_average():
    def analyzer(matrices, **options):
        frame = native_rows(matrices, **options)
        # Keep D in range for all 74 fake rows.
        frame["D"] = 0.1
        frame.loc[frame.longname.str.startswith("EMBED"), "alpha"] = 10000
        return frame
    result = metrics.measure(payload(12, True), analyzer=analyzer)
    assert result["summary"]["matrix_count"] == 72
    assert result["summary"]["alpha_raw_valid_count"] == 72
    assert len(result["layers"]) == 74
    assert result["coverage"]["expected_transformer_matrices"] == 72
    assert result["coverage"]["expected_embedding_matrices"] == 2
    assert result["summary"]["alpha_raw_mean"] == pytest.approx(sum(range(2, 74)) / 72)
    group = next(g for g in result["groups"] if g["group"] == "embeddings")
    assert group["alpha_raw_mean"] == 10000


def test_invalid_missing_and_unmatched_rows_are_explicit_and_clipped_tail_never_falls_back():
    def analyzer(matrices, **options):
        frame = native_rows(matrices, **options)
        if options["fix_fingers"]:
            frame.loc[0, "observer_n_tail"] = np.nan
            frame.loc[1, "D"] = np.nan
            frame = frame[frame.longname != "L00_W_V"]
        else:
            frame.loc[0, "alpha"] = np.nan
        extra = frame.iloc[0].copy()
        extra["longname"] = "L00_W_Q_unexpected_suffix"
        return pd.concat([frame, extra.to_frame().T], ignore_index=True)
    result = metrics.measure(payload(), analyzer=analyzer)
    rows = {row["matrix_name"]: row for row in result["layers"]}
    assert len(rows) == 6
    assert rows["L00_W_Q"]["raw_fit_status"] == "invalid"
    assert rows["L00_W_Q"]["clipped_fit_status"] == "invalid"
    assert rows["L00_W_Q"]["n_tail_clip_xmax"] is None
    assert rows["L00_W_Q"]["num_pl_spikes_clip_xmax"] == 20
    assert rows["L00_W_K"]["D_clip_xmax"] is None
    assert rows["L00_W_V"]["clipped_fit_status"] == "not_returned"
    assert result["coverage"]["status"] == "warning"
    assert result["coverage"]["raw_valid_count"] == 5
    assert result["coverage"]["clipped_valid_count"] == 3
    assert len(result["coverage"]["unassigned_rows"]["raw"]) == 1
    json.dumps(result, allow_nan=False)


def test_missing_snapshot_matrix_still_emits_required_row():
    source = payload()
    del source["matrices"]["L00_W_Q"]
    result = metrics.measure(source, analyzer=native_rows)
    assert len(result["layers"]) == 6
    assert result["layers"][0]["matrix_name"] == "L00_W_Q"
    assert result["layers"][0]["raw_fit_status"] == "not_returned"
    assert result["coverage"]["missing_snapshot_matrices"] == ["L00_W_Q"]


def test_native_erg_absence_stays_unavailable():
    def analyzer(matrices, **options):
        frame = native_rows(matrices, **options)
        return frame.drop(columns=["ERG_gap"])
    result = metrics.measure(payload(), analyzer=analyzer)
    assert result["layers"][0]["ERG_gap_raw"] is None
    assert result["layers"][0]["ERG_status_raw"] == "unavailable"
    assert result["summary"]["ERG_gap_raw_valid_count"] == 0
    assert result["coverage"]["status"] == "warning"


def test_seeded_analysis_restores_all_caller_cpu_rng_states_even_on_error():
    torch = pytest.importorskip("torch")
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    seen = []
    def analyzer(matrices, **options):
        seen.append((random.random(), float(np.random.random()), float(torch.rand(()))))
        if not options["fix_fingers"]:
            raise RuntimeError("injected raw pass error")
        return native_rows(matrices, **options)
    result = metrics.measure(payload(), analyzer=analyzer)
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[0] == numpy_before[0]
    assert np.random.get_state()[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert seen[0] == seen[1]
    assert all(r["raw_fit_status"] == "error" for r in result["layers"])
    assert result["coverage"]["clipped_valid_count"] == 6


def test_snapshot_rejects_mismatched_validation_identity_and_tampering(tmp_path):
    source = checkpoint()
    source["validation"]["step"] = 1
    with pytest.raises(ValueError, match="same update"):
        metrics.queue_snapshot(tmp_path, source)
    data = payload()
    data["content_sha256"] = metrics.snapshot_fingerprint(data)
    data["matrices"]["L00_W_Q"][0, 0] = 4
    with pytest.raises(ValueError, match="digest mismatch"):
        metrics.measure(data, analyzer=native_rows)


def test_bounded_idempotent_queue_prunes_only_after_measurements_and_exports(tmp_path):
    first = metrics.queue_snapshot(tmp_path, checkpoint(0), max_pending=1)
    assert first["status"] == "queued"
    assert metrics.queue_snapshot(tmp_path, checkpoint(0), max_pending=1)["status"] == "already_queued"
    assert metrics.queue_snapshot(tmp_path, checkpoint(1), max_pending=1)["status"] == "queue_full"
    assert len(list((tmp_path / "tracking/snapshots").glob("*.pt"))) == 1
    changed = checkpoint(0)
    changed["model"]["transformer.h.0.attn.c_attn.weight"][0, 0] = 90
    with pytest.raises(ValueError, match="collision"):
        metrics.queue_snapshot(tmp_path, changed)
    status = metrics.process_pending(tmp_path, analyzer=native_rows)
    assert status["completed_steps"] == [0]
    assert status["pending"] == status["failed"] == 0
    assert not Path(first["path"]).exists()
    assert (tmp_path / "tracking/measurements/000000000.json").exists()
    assert (tmp_path / "tracking/summary.csv").exists()
    exports = json.loads((tmp_path / "tracking/EXPORTS.json").read_text())
    for name, digest in exports["files"].items():
        assert metrics._file_sha256(tmp_path / "tracking" / name) == digest
    receipt = json.loads((tmp_path / "tracking/receipts/000000000.json").read_text())
    assert receipt["pruned"] is True
    assert receipt["snapshot_sha256"]
    assert metrics.queue_snapshot(tmp_path, checkpoint(0))["status"] == "already_measured"
    assert metrics.queue_snapshot(tmp_path, checkpoint(1), max_pending=1)["status"] == "queued"
    with pytest.raises(ValueError, match="collision"):
        metrics.queue_snapshot(tmp_path, changed)


def test_full_expected_step_manifest_controls_complete_and_keep_config_is_respected(tmp_path):
    metrics.queue_snapshot(tmp_path, checkpoint(0, keep_snapshots=True))
    metrics.write_json(tmp_path / "tracking/EXPECTED_STEPS.json", {"steps": [0, 1]})
    (tmp_path / "TRAINING_DONE").touch()
    status = metrics.process_pending(tmp_path, analyzer=native_rows)
    assert not status["complete"]
    assert status["missing_expected_steps"] == [1]
    assert (tmp_path / "tracking/snapshots/000000000.pt").exists()
    metrics.queue_snapshot(tmp_path, checkpoint(1, keep_snapshots=True))
    status = metrics.process_pending(tmp_path, analyzer=native_rows)
    assert status["complete"]
    assert status["completed_steps"] == [0, 1]
    assert status["all_fits_valid"]
    assert len(list((tmp_path / "tracking/snapshots").glob("*.pt"))) == 2


def test_failed_worker_retains_snapshot_and_retry_can_commit(tmp_path):
    def broken(matrices, **options):
        raise RuntimeError("deliberate analyzer failure")
    receipt = metrics.queue_snapshot(tmp_path, checkpoint())
    status = metrics.process_pending(tmp_path, analyzer=broken)
    assert status["failed"] == 1
    assert status["completed"] == 0
    assert Path(receipt["path"]).exists()
    status = metrics.process_pending(tmp_path, analyzer=native_rows)
    assert status["failed"] == 1  # explicit retry, never an endless tight loop
    status = metrics.process_pending(tmp_path, analyzer=native_rows, retry_failed=True)
    assert status["failed"] == 0
    assert status["completed"] == 1
    assert not Path(receipt["path"]).exists()


def test_export_failure_keeps_snapshot_and_restart_recovers_without_refitting(tmp_path, monkeypatch):
    receipt = metrics.queue_snapshot(tmp_path, checkpoint())
    original = metrics._csv_write
    def fail_export(*args, **kwargs):
        raise OSError("injected disk export failure")
    monkeypatch.setattr(metrics, "_csv_write", fail_export)
    with pytest.raises(OSError, match="export failure"):
        metrics.process_pending(tmp_path, analyzer=native_rows)
    assert Path(receipt["path"]).exists()
    assert (tmp_path / "tracking/measurements/000000000.json").exists()
    monkeypatch.setattr(metrics, "_csv_write", original)
    def no_refit(*args, **kwargs):
        raise AssertionError("Committed measurement must not be refitted")
    status = metrics.process_pending(tmp_path, analyzer=no_refit)
    assert status["completed"] == 1
    assert not Path(receipt["path"]).exists()


def test_real_weightwatcher_cpu_probe_preserves_native_erg_and_exact_tail():
    pytest.importorskip("torch")
    pytest.importorskip("weightwatcher")
    result = metrics.capability_check()
    assert result["ok"], result
    for fit in ("raw", "clip_xmax"):
        probe = result["probes"][fit]
        assert probe["missing_fields"] == []
        assert probe["n_tail"] > 0
        assert probe["native_ERG_formula_matches"]

