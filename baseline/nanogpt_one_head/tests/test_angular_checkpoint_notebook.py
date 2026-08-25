from __future__ import annotations

from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.angular_three_checkpoint import (
    PAIR_ORDER,
    resolve_three_checkpoints,
)
from rg_nanogpt_one_head.angular_multiseed import (
    aggregate_seed_results,
    discover_complete_seeds,
    parse_seed_spec,
)
from rg_nanogpt_one_head.angular_weightwatcher_core import (
    AnalysisConfig,
    resolve_run,
)

ENGINE_PATH = EXPERIMENT_ROOT / "src" / "rg_nanogpt_one_head" / "engine.py"
CORE_PATH = EXPERIMENT_ROOT / "src" / "rg_nanogpt_one_head" / "angular_weightwatcher_core.py"
THREE_PATH = EXPERIMENT_ROOT / "src" / "rg_nanogpt_one_head" / "angular_three_checkpoint.py"
MULTI_PATH = EXPERIMENT_ROOT / "src" / "rg_nanogpt_one_head" / "angular_multiseed.py"
PYPROJECT_PATH = EXPERIMENT_ROOT / "pyproject.toml"
CANONICAL_NOTEBOOK = EXPERIMENT_ROOT / "notebooks" / "angular" / "07_muonclip_initial_final_angular_weightwatcher.ipynb"
EXPLORATORY_NOTEBOOK = EXPERIMENT_ROOT / "notebooks" / "angular" / "muonclip_angular_radial_rg.ipynb"


def test_training_saves_immutable_initial_checkpoint_before_training():
    source = ENGINE_PATH.read_text(encoding="utf-8")
    assert 'initial_checkpoint = run_dir / "checkpoint_initial.pt"' in source
    assert "if start_step == 0 and not initial_checkpoint.is_file():" in source
    save_position = source.index("save_training_checkpoint(\n            initial_checkpoint")
    training_position = source.index("execute_training_loop(")
    assert save_position < training_position


def _write_minimal_checkpoint(path: Path, step: int, seed: int = 777):
    torch.save({"step": step, "model": {}, "seed": seed}, path)


def test_runroot_resolves_any_seed_and_standard_results_layout(tmp_path):
    runroot = tmp_path / "repeatable-run-root"
    run_dir = runroot / "results" / "muon_clip" / "seed_777"
    run_dir.mkdir(parents=True)
    _write_minimal_checkpoint(run_dir / "checkpoint_initial.pt", 0)
    _write_minimal_checkpoint(run_dir / "checkpoint_final.pt", 29)
    config = AnalysisConfig(seed=777, optimizer="muon_clip", runroot=str(runroot), angular_nulls=10, entry_nulls=3, show_plots=False)
    resolved = resolve_run(config)
    assert resolved.run_dir == run_dir.resolve()
    assert resolved.initial_path.name == "checkpoint_initial.pt"
    assert resolved.final_path.name == "checkpoint_final.pt"
    assert resolved.final_step == 29


def test_three_checkpoint_resolver_requires_actual_initial_best_final(tmp_path, monkeypatch):
    runroot = tmp_path / "repeatable-run-root"
    run_dir = runroot / "results" / "muon_clip" / "seed_777"
    run_dir.mkdir(parents=True)
    _write_minimal_checkpoint(run_dir / "checkpoint_initial.pt", 0)
    _write_minimal_checkpoint(run_dir / "checkpoint_best.pt", 17)
    _write_minimal_checkpoint(run_dir / "checkpoint_final.pt", 29)
    config = AnalysisConfig(seed=777, optimizer="muon_clip", runroot=str(runroot), angular_nulls=10, entry_nulls=3, show_plots=False)
    monkeypatch.delenv("BEST_CHECKPOINT_PATH", raising=False)
    resolved, paths, steps = resolve_three_checkpoints(config)
    assert resolved.run_dir == run_dir.resolve()
    assert paths["initial"].name == "checkpoint_initial.pt"
    assert paths["best"].name == "checkpoint_best.pt"
    assert paths["final"].name == "checkpoint_final.pt"
    assert steps == {"initial": 0, "best": 17, "final": 29}
    assert PAIR_ORDER == (("initial", "best"), ("initial", "final"), ("best", "final"))


def test_three_checkpoint_resolver_rejects_best_after_final(tmp_path, monkeypatch):
    runroot = tmp_path / "repeatable-run-root"
    run_dir = runroot / "results" / "muon_clip" / "seed_777"
    run_dir.mkdir(parents=True)
    _write_minimal_checkpoint(run_dir / "checkpoint_initial.pt", 0)
    _write_minimal_checkpoint(run_dir / "checkpoint_best.pt", 30)
    _write_minimal_checkpoint(run_dir / "checkpoint_final.pt", 29)
    config = AnalysisConfig(seed=777, optimizer="muon_clip", runroot=str(runroot), angular_nulls=10, entry_nulls=3, show_plots=False)
    monkeypatch.delenv("BEST_CHECKPOINT_PATH", raising=False)
    with pytest.raises(ValueError, match="after final"):
        resolve_three_checkpoints(config)


def test_seed_spec_and_complete_seed_discovery(tmp_path):
    results_root = tmp_path / "results"
    optimizer_root = results_root / "muon_clip"
    for seed in (1337, 2027, 31415):
        run_dir = optimizer_root / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        _write_minimal_checkpoint(run_dir / "checkpoint_initial.pt", 0, seed)
        _write_minimal_checkpoint(run_dir / "checkpoint_best.pt", 5, seed)
        _write_minimal_checkpoint(run_dir / "checkpoint_final.pt", 10, seed)
    incomplete = optimizer_root / "seed_999"
    incomplete.mkdir(parents=True)
    _write_minimal_checkpoint(incomplete / "checkpoint_final.pt", 10, 999)

    config = AnalysisConfig(optimizer="muon_clip", results_root=str(results_root), angular_nulls=10, entry_nulls=3, show_plots=False)
    root, seeds = discover_complete_seeds(config)
    assert root == results_root.resolve()
    assert seeds == (1337, 2027, 31415)
    assert parse_seed_spec("1337, 2027  ;31415") == (1337, 2027, 31415)
    with pytest.raises(ValueError, match="duplicate"):
        parse_seed_spec("1337,1337")


def test_cross_seed_aggregation_uses_seed_level_student_t_statistics():
    rows = []
    for seed, alpha in ((1, 1.8), (2, 2.0), (3, 2.2), (4, 2.4)):
        rows.append({
            "seed": seed,
            "matrix_name": "L00_W_Q",
            "angular_type": "twist",
            "pair": "initial->final",
            "actual_alpha": alpha,
            "actual_xmin": 1.0 + seed,
            "actual_D": 0.05,
            "actual_tail_n": 30,
            "actual_tail_decades": 1.5,
            "actual_endpoint_atoms": 0,
            "full_continuous_ks_D": 0.1,
            "full_continuous_ks_mc_p": 0.02,
            "tail_conditional_ks_D": 0.2,
            "tail_conditional_ks_mc_p": 0.01,
            "null_alpha_median": 1.5,
            "null_xmin_median": 1.1,
            "null_D_median": 0.06,
            "null_tail_n_median": 25,
            "null_tail_decades_median": 1.0,
            "actual_fit_success": True,
            "alpha_outside_random_null": True,
            "tail_longer_than_random_97p5": True,
            "candidate_nonrandom_long_tail": True,
        })
    summary = aggregate_seed_results(pd.DataFrame(rows))
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["actual_alpha_n"] == 4
    assert row["actual_alpha_mean"] == pytest.approx(2.1)
    expected_sd = np.std([1.8, 2.0, 2.2, 2.4], ddof=1)
    assert row["actual_alpha_sd"] == pytest.approx(expected_sd)
    assert row["actual_alpha_ci95"] > 0.0
    assert row["candidate_nonrandom_long_tail_fraction"] == 1.0


def test_both_angular_notebooks_are_multiseed_papermill_entrypoints():
    for notebook_path in (CANONICAL_NOTEBOOK, EXPLORATORY_NOTEBOOK):
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        parameter_indexes = [
            i for i, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", [])
        ]
        assert len(parameter_indexes) == 1
        parameter_index = parameter_indexes[0]
        parameter_source = notebook.cells[parameter_index].source
        later_source = "\n".join(cell.source for cell in notebook.cells[parameter_index + 1:])
        full_source = "\n".join(cell.source for cell in notebook.cells)
        assert "CONFIG =" not in parameter_source
        assert "AnalysisConfig(" in later_source
        assert "run_multiseed_analysis" in later_source
        for name in (
            "TARGET_OPTIMIZER", "RUNROOT", "RESULTS_ROOT", "RUN_DIR", "ANGULAR_SEEDS",
            "ANGULAR_N_NULL", "ANGULAR_ENDPOINT_TOL", "ANGULAR_SHOW_PLOTS",
        ):
            assert name in parameter_source
        for text in (
            "initial", "best", "final", "powerlaw.Fit", "no `xmin`", "no `xmax`",
            "random", "95% Student-t", "individual seed", "zoomed-y", "Papermill",
        ):
            assert text.lower() in full_source.lower()
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{notebook_path.name}:cell-{index}", "exec")

    for path in (CORE_PATH, THREE_PATH, MULTI_PATH):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_papermill_is_a_runtime_dependency():
    source = PYPROJECT_PATH.read_text(encoding="utf-8")
    assert '"papermill>=2.6,<3"' in source
