from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANAGER = (
    REPOSITORY_ROOT
    / "experiments"
    / "mnist_mlp3_tangent_rg"
    / "scripts"
    / "manage_short100_complete_rg_analysis.sh"
)


def test_manager_never_changes_the_callers_errexit_or_attaches_a_log_tail():
    source = MANAGER.read_text(encoding="utf-8")
    executable_lines = [
        line.strip() for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert not any(line.startswith("set -e") for line in executable_lines)
    assert "tail -f" not in source
    assert "tail -F" not in source


def test_manager_launches_a_new_session_and_exposes_safe_lifecycle_commands():
    source = MANAGER.read_text(encoding="utf-8")

    assert "start_new_session=True" in source
    assert "stdin=subprocess.DEVNULL" in source
    assert "stdout=log_handle" in source
    assert "stderr=subprocess.STDOUT" in source
    for command in ("start", "status", "log", "stop", "open"):
        assert f"{command})" in source


def test_legacy_stop_never_signals_the_interactive_process_group():
    source = MANAGER.read_text(encoding="utf-8")

    assert "old nohup launcher" in source
    assert "os.kill(item, signal.SIGTERM)" in source
    assert "os.killpg(pid, signal.SIGTERM)" in source
    assert "if managed:" in source
