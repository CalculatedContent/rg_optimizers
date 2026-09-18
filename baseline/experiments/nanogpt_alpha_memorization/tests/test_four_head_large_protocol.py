import json
from pathlib import Path

from am_data import Dataset
from am_runtime import baseline, make_model

HERE = Path(__file__).resolve().parents[1]


def test_four_head_large_protocol_shape_and_data():
    cfg = json.loads((HERE / "protocol_four_head_large.json").read_text())
    source = baseline(cfg)
    data = Dataset(cfg, 1337, source["training"]["batch_size"] * source["training"]["grad_accum_steps"])

    assert cfg["model_overrides"]["n_head"] == 4
    assert len(data.train) == 31500
    assert sum(r.cohort == "validation_clean" for r in data.audit) == 15750
    assert sum(r.cohort == "test_clean" for r in data.audit) == 15751
    assert data.withdrawal == 15000

    model = make_model(source, cfg, 1337, "cpu")
    assert model.blocks[0].attn.n_head == 4
    assert model.cfg.n_embd % 4 == 0
