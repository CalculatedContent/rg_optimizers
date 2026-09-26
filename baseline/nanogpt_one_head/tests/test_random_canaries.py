from pathlib import Path
import torch

from rg_nanogpt_one_head.random_canaries import RandomCanaryExperiment


def _cfg():
    return {
        "model": {"block_size": 16, "vocab_size": 101},
        "training": {"batch_size": 2, "grad_accum_steps": 2},
        "memorization": {
            "enabled": True,
            "data_seed": 7,
            "doses": [0, 1, 4],
            "canaries_per_dose": 2,
            "prefix_tokens": 4,
            "suffix_tokens": 4,
            "acquisition_fraction": 0.5,
        },
    }


def test_random_canaries_exact_doses_and_zero_controls(tmp_path: Path):
    exp = RandomCanaryExperiment(_cfg(), seed=1337, total_steps=20, run_dir=tmp_path)
    counts = {c["id"]: 0 for c in exp.canaries}
    for c in exp.schedule.values():
        counts[c["id"]] += 1
    for c in exp.canaries:
        assert counts[c["id"]] == c["dose"]
    assert all(counts[c["id"]] == 0 for c in exp.canaries if c["dose"] == 0)


def test_injection_replaces_only_scheduled_row(tmp_path: Path):
    exp = RandomCanaryExperiment(_cfg(), seed=1337, total_steps=20, run_dir=tmp_path)
    (step, micro, row), canary = next(iter(exp.schedule.items()))
    x = torch.zeros((2, 16), dtype=torch.long)
    y = torch.zeros((2, 16), dtype=torch.long)
    xx, yy = exp.inject(x, y, completed_step=step, micro_index=micro)
    assert torch.equal(xx[row], canary["tokens"][:-1])
    assert torch.equal(yy[row], canary["tokens"][1:])
    other = 1 - row
    assert torch.equal(xx[other], x[other])
    assert torch.equal(yy[other], y[other])


def test_harmful_load_adds_repeated_unscored_random_windows(tmp_path: Path):
    cfg = _cfg()
    cfg["memorization"]["harmful_load_fraction"] = 0.25
    cfg["memorization"]["harmful_load_dose"] = 4
    exp = RandomCanaryExperiment(cfg, seed=1337, total_steps=20, run_dir=tmp_path)
    harmful = [v for v in exp.schedule.values() if v["id"].startswith("harmful_canary")]
    # acquisition slots = round(20*0.5) * 2 grad accum * 2 rows = 40
    assert len(harmful) == 10
    ids = [v["id"] for v in harmful]
    assert len(set(ids)) == 3
    # Tracked dose-0 controls remain absent from the injection schedule.
    assert not any(v["id"].startswith("dose0_") for v in exp.schedule.values())
