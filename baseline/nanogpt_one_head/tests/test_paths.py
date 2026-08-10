from __future__ import annotations

from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.config import DEFAULT_ROOT, roots


ROOT_ENV_VARS = (
    "RG_NANOGPT_ONE_HEAD_ROOT",
    "RG_NANOGPT_ONE_HEAD_DATA_ROOT",
    "RG_NANOGPT_ONE_HEAD_RESULTS_ROOT",
    "RG_NANOGPT_ONE_HEAD_PLOTS_ROOT",
)
FORBIDDEN_HOME_TOKENS = (
    "$HOME",
    "${HOME}",
    "Path.home(",
    ".expanduser(",
    "/home/",
    "~/",
)


def test_default_root_is_tmp(monkeypatch):
    for name in ROOT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    assert DEFAULT_ROOT == Path("/tmp/rg-nanogpt-one-head")
    resolved = roots()
    assert resolved == {
        "root": Path("/tmp/rg-nanogpt-one-head"),
        "data": Path("/tmp/rg-nanogpt-one-head/data"),
        "results": Path("/tmp/rg-nanogpt-one-head/results"),
        "plots": Path("/tmp/rg-nanogpt-one-head/plots"),
    }


def test_nanogpt_sources_docs_and_notebooks_do_not_reference_home():
    paths = [EXPERIMENT_ROOT / "README.md"]
    paths.extend((EXPERIMENT_ROOT / "src").rglob("*.py"))
    paths.extend((EXPERIMENT_ROOT / "notebooks").glob("*.ipynb"))
    paths.extend((EXPERIMENT_ROOT / "configs").glob("*.yaml"))

    violations = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_HOME_TOKENS:
            if token in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {token}")

    assert not violations, "home-directory references found:\n" + "\n".join(violations)


def test_repo_shell_scripts_never_reference_home():
    violations = []
    for path in REPO_ROOT.rglob("*.sh"):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_HOME_TOKENS:
            if token in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {token}")

    assert not violations, "home-directory references found in shell scripts:\n" + "\n".join(violations)


def test_removed_nanogpt_wrapper_scripts_do_not_return():
    assert not (EXPERIMENT_ROOT / "scripts").exists()
