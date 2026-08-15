from __future__ import annotations

from pathlib import Path
import subprocess
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_muonclip_24h_mb2_preserves_effective_update_budget() -> None:
    code = """
from pathlib import Path
from rg_nanogpt_one_head.muonclip import install_muonclip_extension
install_muonclip_extension()
from rg_nanogpt_one_head.config import (
    epoch_step_map,
    load_config,
    max_steps,
    tokens_per_step,
)
root = Path.cwd()
cfg = load_config(root / 'configs' / 'muonclip_24h_125k_mb2.yaml')
points = epoch_step_map(cfg)
print(
    cfg['training']['batch_size'],
    cfg['training']['grad_accum_steps'],
    tokens_per_step(cfg),
    max_steps(cfg),
    len(points),
    min(points),
    max(points),
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
        "2 16 8192 125000 100 0 125000 500"
    )
