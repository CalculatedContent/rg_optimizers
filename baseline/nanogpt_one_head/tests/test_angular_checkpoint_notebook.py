from __future__ import annotations

from pathlib import Path
import sys

import nbformat
import pytest
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.angular_three_checkpoint import (
    PAIR_ORDER,
    resolve_three_checkpoints,
)
from rg_nanogpt_one_head.angular_weightwatcher_core import (
    AnalysisConfig,
    resolve_run,
)

ENGINE_PATH = EXPERIMENT_ROOT / "src" / "rg_nanogpt_one_head" / "engine.py"
CORE_PATH = (
    EXPERIMENT_ROOT
    / "src"
    / "rg_nanogpt_one_head"
    / "angular_weightwatcher_core.py"
)
THREE_PATH = (
    EXPERIMENT_ROOT
    / "src"
    / "rg_nanogpt_one_head"
    / "angular_three_checkpoint.py"
)
PYPROJECT_PATH = EXPERIMENT_ROOT / "pyproject.toml"
CANONICAL_NOTEBOOK = (
    EXPERIMENT_ROOT
    / "notebooks"
    / "angular"
    / "07_muonclip_initial_final_angular_weightwatcher.ipynb"
)
EXPLORATORY_NOTEBOOK = (
    EXPERIMENT_ROOT
    / "notebooks"
    / "muonclip_angular_radial_rg.ipynb"
)


def test_training_saves_immutable_initial_checkpoint_before_training():
    source = ENGINE_PATH.read_text(encoding="utf-8")
    assert 'initial_checkpoint = run_dir / "checkpoint_initial.pt"' in source
    assert "if start_step == 0 and not initial_checkpoint.is_file():" in source
    save_position = source.index(
        "save_training_checkpoint(\n            initial_checkpoint"
    )
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

    config = AnalysisConfig(
        seed=777,
        optimizer="muon_clip",
        runroot=str(runroot),
        angular_nulls=10,
        entry_nulls=3,
        show_plots=False,
    )
    resolved = resolve_run(config)
    assert resolved.run_dir == run_dir.resolve()
    assert resolved.initial_path.name == "checkpoint_initial.pt"
    assert resolved.final_path.name == "checkpoint_final.pt"
    assert resolved.final_step == 29


def test_three_checkpoint_resolver_requires_actual_initial_best_final(
    tmp_path,
    monkeypatch,
):
    runroot = tmp_path / "repeatable-run-root"
    run_dir = runroot / "results" / "muon_clip" / "seed_777"
    run_dir.mkdir(parents=True)
    _write_minimal_checkpoint(run_dir / "checkpoint_initial.pt", 0)
    _write_minimal_checkpoint(run_dir / "checkpoint_best.pt", 17)
    _write_minimal_checkpoint(run_dir / "checkpoint_final.pt", 29)

    config = AnalysisConfig(
        seed=777,
        optimizer="muon_clip",
        runroot=str(runroot),
        angular_nulls=10,
        entry_nulls=3,
        show_plots=False,
    )
    monkeypatch.delenv("BEST_CHECKPOINT_PATH", raising=False)
    resolved, paths, steps = resolve_three_checkpoints(config)

    assert resolved.run_dir == run_dir.resolve()
    assert paths["initial"].name == "checkpoint_initial.pt"
    assert paths["best"].name == "checkpoint_best.pt"
    assert paths["final"].name == "checkpoint_final.pt"
    assert steps == {"initial": 0, "best": 17, "final": 29}
    assert PAIR_ORDER == (
        ("initial", "best"),
        ("initial", "final"),
        ("best", "final"),
    )


def test_three_checkpoint_resolver_rejects_best_after_final(
    tmp_path,
    monkeypatch,
):
    runroot = tmp_path / "repeatable-run-root"
    run_dir = runroot / "results" / "muon_clip" / "seed_777"
    run_dir.mkdir(parents=True)
    _write_minimal_checkpoint(run_dir / "checkpoint_initial.pt", 0)
    _write_minimal_checkpoint(run_dir / "checkpoint_best.pt", 30)
    _write_minimal_checkpoint(run_dir / "checkpoint_final.pt", 29)

    config = AnalysisConfig(
        seed=777,
        optimizer="muon_clip",
        runroot=str(runroot),
        angular_nulls=10,
        entry_nulls=3,
        show_plots=False,
    )
    monkeypatch.delenv("BEST_CHECKPOINT_PATH", raising=False)
    with pytest.raises(ValueError, match="after final"):
        resolve_three_checkpoints(config)


def test_both_angular_notebooks_are_valid_papermill_parameterized_entrypoints():
    for notebook_path in (CANONICAL_NOTEBOOK, EXPLORATORY_NOTEBOOK):
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)

        parameter_indexes = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
            and "parameters" in cell.metadata.get("tags", [])
        ]
        assert len(parameter_indexes) == 1
        parameter_index = parameter_indexes[0]
        assert parameter_index + 1 < len(notebook.cells)

        parameter_source = notebook.cells[parameter_index].source
        later_source = "\n".join(
            cell.source for cell in notebook.cells[parameter_index + 1 :]
        )
        full_source = "\n".join(cell.source for cell in notebook.cells)

        # Papermill injects overrides immediately after the tagged parameter
        # cell.  CONFIG must therefore be constructed later, not inside the
        # tagged cell from environment values that Papermill cannot override.
        assert "CONFIG =" not in parameter_source
        assert "AnalysisConfig(" in later_source
        assert "run_three_checkpoint_analysis(CONFIG)" in later_source

        for name in (
            "TARGET_SEED",
            "TARGET_OPTIMIZER",
            "RUNROOT",
            "RESULTS_ROOT",
            "RUN_DIR",
            "INITIAL_CHECKPOINT_PATH",
            "BEST_CHECKPOINT_PATH",
            "FINAL_CHECKPOINT_PATH",
            "ANGULAR_N_NULL",
            "ANGULAR_ENDPOINT_TOL",
            "ANGULAR_SHOW_PLOTS",
        ):
            assert name in parameter_source

        for text in (
            "checkpoint_initial.pt",
            "checkpoint_best.pt",
            "checkpoint_final.pt",
            "powerlaw.Fit",
            "no `xmin`",
            "no `xmax`",
            "endpoint",
            "random",
            "papermill",
        ):
            assert text.lower() in full_source.lower()

        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(
                    cell.source,
                    f"{notebook_path.name}:cell-{index}",
                    "exec",
                )

    compile(CORE_PATH.read_text(encoding="utf-8"), str(CORE_PATH), "exec")
    compile(THREE_PATH.read_text(encoding="utf-8"), str(THREE_PATH), "exec")


def test_papermill_is_a_runtime_dependency():
    source = PYPROJECT_PATH.read_text(encoding="utf-8")
    assert '"papermill>=2.6,<3"' in source
