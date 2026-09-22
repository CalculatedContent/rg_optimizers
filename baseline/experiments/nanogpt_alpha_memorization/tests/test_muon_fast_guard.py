import json
from pathlib import Path

from am_runtime import baseline, make_model, imports

HERE = Path(__file__).resolve().parents[1]


def test_fast_guard_protocol_builds_scaled_muon_groups():
    cfg = json.loads((HERE / "protocol_four_head_muon_fast_guard.json").read_text())
    source = baseline(cfg)
    model = make_model(source, cfg, 2027, "cpu")
    _, _, make_handles, _, _, _ = imports()

    profile = dict(source["optimizer_profiles"]["muon"])
    profile.update(cfg["optimizer_overrides"]["muon"])
    handles = make_handles(model, profile)
    primary = next(h for h in handles if h.role == "primary")
    groups = {g["matrix_name"]: g for g in primary.optimizer.param_groups}

    assert model.blocks[0].attn.n_head == 4
    assert groups["blocks.0.attn.q_proj.weight"]["lr_scale"] == 0.75
    assert groups["blocks.0.attn.k_proj.weight"]["lr_scale"] == 0.75
    assert groups["blocks.0.attn.v_proj.weight"]["lr_scale"] == 1.0
    assert groups["blocks.0.attn.q_proj.weight"]["weight_decay"] == 0.02
    assert groups["blocks.0.attn.k_proj.weight"]["weight_decay"] == 0.02
    assert groups["blocks.0.attn.v_proj.weight"]["weight_decay"] == 0.01
