from __future__ import annotations

from pathlib import Path
import sys

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.config import (
    DEFAULT_ROOT,
    TPU_EPHEMERAL_ENV,
    TPU_PERSISTENT_ENV,
    TPU_ROOT_ENV,
    roots,
)


ROOT_ENV_VARS = (
    "RG_NANOGPT_ONE_HEAD_ROOT",
    "RG_NANOGPT_ONE_HEAD_DATA_ROOT",
    "RG_NANOGPT_ONE_HEAD_RESULTS_ROOT",
    "RG_NANOGPT_ONE_HEAD_PLOTS_ROOT",
    TPU_ROOT_ENV,
    TPU_PERSISTENT_ENV,
    TPU_EPHEMERAL_ENV,
)
FORBIDDEN_HOME_TOKENS = (
    "$HOME",
    "${HOME}",
    "Path.home(",
    "/home/",
    "~/",
)


def _clear_roots(monkeypatch) -> None:
    for name in ROOT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_default_non_tpu_root_remains_tmp(monkeypatch):
    _clear_roots(monkeypatch)

    assert DEFAULT_ROOT == Path("/tmp/rg-nanogpt-one-head")
    resolved = roots(device="cpu")
    assert resolved == {
        "root": Path("/tmp/rg-nanogpt-one-head"),
        "data": Path("/tmp/rg-nanogpt-one-head/data"),
        "results": Path("/tmp/rg-nanogpt-one-head/results"),
        "plots": Path("/tmp/rg-nanogpt-one-head/plots"),
    }


def test_tpu_root_uses_detected_persistent_mount(monkeypatch):
    _clear_roots(monkeypatch)

    resolved = roots(
        device="tpu",
        mount_points=[
            Path("/"),
            Path("/mnt/disks/rg-data"),
        ],
    )
    root = Path("/mnt/disks/rg-data/rg-nanogpt-one-head")
    assert resolved == {
        "root": root,
        "data": root / "data",
        "results": root / "results",
        "plots": root / "plots",
    }


def test_tpu_persistent_root_environment_override(monkeypatch):
    _clear_roots(monkeypatch)
    monkeypatch.setenv(
        TPU_PERSISTENT_ENV,
        "/mnt/disks/experiment-disk",
    )

    resolved = roots(
        device="tpu",
        mount_points=[
            Path("/"),
            Path("/mnt/disks/experiment-disk"),
        ],
    )
    assert resolved["root"] == Path(
        "/mnt/disks/experiment-disk/rg-nanogpt-one-head"
    )


def test_tpu_without_persistent_storage_fails_closed(monkeypatch):
    _clear_roots(monkeypatch)

    with pytest.raises(RuntimeError, match="no persistent data volume"):
        roots(device="tpu", mount_points=[Path("/")])


def test_tpu_ephemeral_storage_requires_explicit_opt_in(monkeypatch):
    _clear_roots(monkeypatch)
    monkeypatch.setenv(TPU_EPHEMERAL_ENV, "1")

    resolved = roots(device="tpu", mount_points=[Path("/")])
    assert resolved["root"] == DEFAULT_ROOT


def test_generic_root_override_must_be_persistent_on_tpu(monkeypatch):
    _clear_roots(monkeypatch)
    monkeypatch.setenv(
        "RG_NANOGPT_ONE_HEAD_ROOT",
        "/tmp/rg-nanogpt-one-head",
    )

    with pytest.raises(
        RuntimeError,
        match="not located on a detected TPU persistent mount",
    ):
        roots(device="tpu", mount_points=[Path("/")])


def test_explicit_data_and_results_roots_bypass_default_mount(monkeypatch):
    _clear_roots(monkeypatch)
    monkeypatch.setenv(
        "RG_NANOGPT_ONE_HEAD_DATA_ROOT",
        "/custom/data",
    )
    monkeypatch.setenv(
        "RG_NANOGPT_ONE_HEAD_RESULTS_ROOT",
        "/custom/results",
    )

    resolved = roots(device="tpu", mount_points=[Path("/")])
    assert resolved["data"] == Path("/custom/data")
    assert resolved["results"] == Path("/custom/results")


def test_nanogpt_sources_docs_and_notebooks_do_not_reference_home():
    paths = [EXPERIMENT_ROOT / "README.md", EXPERIMENT_ROOT / "TPU.md"]
    paths.extend((EXPERIMENT_ROOT / "src").rglob("*.py"))
    paths.extend((EXPERIMENT_ROOT / "notebooks").glob("*.ipynb"))
    paths.extend((EXPERIMENT_ROOT / "configs").glob("*.yaml"))

    violations = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_HOME_TOKENS:
            if token in text:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}: {token}"
                )

    assert not violations, (
        "home-directory references found:\n"
        + "\n".join(violations)
    )


def test_repo_shell_scripts_never_reference_home():
    violations = []
    for path in REPO_ROOT.rglob("*.sh"):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_HOME_TOKENS:
            if token in text:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}: {token}"
                )

    assert not violations, (
        "home-directory references found in shell scripts:\n"
        + "\n".join(violations)
    )


def test_removed_nanogpt_wrapper_scripts_do_not_return():
    assert not (EXPERIMENT_ROOT / "scripts").exists()
