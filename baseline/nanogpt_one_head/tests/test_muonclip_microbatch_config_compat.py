from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_microbatch_config_sets_legacy_and_modern_capture_counts() -> None:
    cfg = yaml.safe_load(
        (ROOT / "configs" / "muonclip_microbatch10.yaml").read_text()
    )
    profile = cfg["optimizer_profiles"]["muon_clip"]
    assert profile["walk_capture_steps"] == 10
    assert profile["walk_capture_effective_batches"] == 10
    assert profile["walk_capture_microbatches"] is True
    assert profile["walk_max_checkpoints"] == 500
