from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = EXPERIMENT_ROOT / "src" / "rg_nanogpt_one_head"


def test_muonclip_worker_help_does_not_require_analysis_module(
    tmp_path: Path,
) -> None:
    """Reproduce the isolated-worker package layout that failed on MPS."""

    package_root = tmp_path / "rg_nanogpt_one_head"
    package_root.mkdir()
    (package_root / "__init__.py").write_text(
        '"""Minimal isolated-worker package used by this regression test."""\n',
        encoding="utf-8",
    )
    for source in SOURCE_PACKAGE.glob("*.py"):
        if source.name in {"__init__.py", "analysis.py"}:
            continue
        shutil.copy2(source, package_root / source.name)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rg_nanogpt_one_head.muonclip",
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--optimizer" in completed.stdout
    assert "muon_clip" in completed.stdout


def test_muonclip_24h_config_has_exact_update_and_snapshot_budget() -> None:
    code = """
from pathlib import Path
from rg_nanogpt_one_head.muonclip import install_muonclip_extension
install_muonclip_extension()
from rg_nanogpt_one_head.config import (
    epoch_step_map,
    load_config,
    lr_schedule_steps,
    max_steps,
    optimizer_profile,
    warmup_steps,
)
root = Path.cwd()
cfg = load_config(root / 'configs' / 'muonclip_24h_125k.yaml')
profile = optimizer_profile(cfg, 'muon_clip')
points = epoch_step_map(cfg)
schedule_steps = lr_schedule_steps(cfg, profile)
print(
    max_steps(cfg),
    len(points),
    min(points),
    max(points),
    schedule_steps,
    warmup_steps(profile, schedule_steps),
    cfg['training']['checkpoint_interval_steps'],
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=EXPERIMENT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().endswith(
        "125000 100 0 125000 9766 500 2500"
    )
