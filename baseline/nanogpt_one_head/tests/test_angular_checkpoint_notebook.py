from __future__ import annotations

from pathlib import Path
import sys

import nbformat
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

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
PIPELINE_PATH = (
    EXPERIMENT_ROOT
    / "src"
    / "rg_nanogpt_one_head"
    / "angular_weightwatcher_pipeline.py"
)
NOTEBOOK_PATH = (
    EXPERIMENT_ROOT
    / "notebooks"
    / "angular"
    / "07_muonclip_initial_final_angular_weightwatcher.ipynb"
)


def test_training_saves_immutable_initial_checkpoint_before_training():
    source = ENGINE_PATH.read_text(encoding="utf-8")
    assert 'initial_checkpoint = run_dir / "checkpoint_initial.pt"' in source
    assert (
        "if start_step == 0 and not initial_checkpoint.is_file():"
        in source
    )
    save_position = source.index(
        "save_training_checkpoint(\n            initial_checkpoint"
    )
    training_position = source.index("execute_training_loop(")
    assert save_position < training_position


def test_runroot_resolves_any_seed_and_standard_results_layout(tmp_path):
    runroot = tmp_path / "repeatable-run-root"
    run_dir = runroot / "results" / "muon_clip" / "seed_777"
    run_dir.mkdir(parents=True)
    torch.save(
        {"step": 0, "model": {}, "seed": 777},
        run_dir / "checkpoint_initial.pt",
    )
    torch.save(
        {"step": 29, "model": {}, "seed": 777},
        run_dir / "checkpoint_final.pt",
    )

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
    assert resolved.run_dir_source == "RUNROOT"
    assert resolved.initial_path.name == "checkpoint_initial.pt"
    assert resolved.final_path.name == "checkpoint_final.pt"
    assert resolved.final_step == 29


def test_angular_notebook_and_modules_are_valid_and_environment_driven():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)

    module_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (CORE_PATH, PIPELINE_PATH)
    )
    notebook_source = "\n".join(
        cell.source for cell in notebook.cells
    )
    source = module_source + "\n" + notebook_source

    required = (
        "RUNROOT",
        "RESULTS_ROOT",
        "RUN_DIR",
        "TARGET_OPTIMIZER",
        "TARGET_SEED",
        "INITIAL_CHECKPOINT_PATH",
        "FINAL_CHECKPOINT_PATH",
        "checkpoint_initial.pt",
        "checkpoint_final.pt",
        "randomized_initial_to_fixed_final",
        "get_weights",
        "analysis_manifest.json",
    )
    for text in required:
        assert text in source

    compile(CORE_PATH.read_text(encoding="utf-8"), str(CORE_PATH), "exec")
    compile(
        PIPELINE_PATH.read_text(encoding="utf-8"),
        str(PIPELINE_PATH),
        "exec",
    )
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(
                cell.source,
                f"{NOTEBOOK_PATH.name}:cell-{index}",
                "exec",
            )
