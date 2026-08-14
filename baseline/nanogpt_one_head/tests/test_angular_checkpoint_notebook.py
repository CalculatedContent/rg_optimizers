from __future__ import annotations

from pathlib import Path

import nbformat

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
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
