import torch

from rg_nanogpt_muon_hyperball.config import load_config, optimizer_profile
from rg_nanogpt_muon_hyperball.optimizers import (
    Muon,
    MuonHyperBall,
    make_optimizer_handles,
)


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = torch.nn.Linear(8, 8, bias=False)
        self.k = torch.nn.Linear(8, 8, bias=False)
        self.v = torch.nn.Linear(8, 8, bias=False)
        self.o = torch.nn.Linear(8, 8, bias=False)
        self.mlp_in = torch.nn.Linear(8, 32, bias=False)
        self.mlp_out = torch.nn.Linear(32, 8, bias=False)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([TinyBlock()])
        self.embedding = torch.nn.Embedding(32, 8)
        self.norm = torch.nn.LayerNorm(8)


def test_matched_partition_uses_six_hidden_matrices() -> None:
    from pathlib import Path

    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "reference.yaml"
    )
    model = TinyModel()

    plain = make_optimizer_handles(model, optimizer_profile(cfg, "muon"))
    ball = make_optimizer_handles(
        model, optimizer_profile(cfg, "muon_hyperball")
    )

    assert isinstance(plain[0].optimizer, Muon)
    assert isinstance(ball[0].optimizer, MuonHyperBall)
    assert len(plain[0].optimizer.param_groups[0]["params"]) == 6
    assert len(ball[0].optimizer.param_groups[0]["params"]) == 6
    assert len(plain) == len(ball) == 2
