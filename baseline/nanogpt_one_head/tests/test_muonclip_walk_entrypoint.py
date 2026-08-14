from __future__ import annotations

from pathlib import Path


def test_legacy_walk_entrypoint_delegates_to_capture() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "src"
        / "rg_nanogpt_one_head"
        / "muonclip_walk.py"
    ).read_text()
    assert "from .muonclip_capture import main as capture_main" in source
    assert "capture_main()" in source
