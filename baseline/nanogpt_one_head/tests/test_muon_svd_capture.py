from __future__ import annotations

import torch

from rg_nanogpt_one_head.muon_svd_capture import MuonUpdateSVDCapture
from rg_nanogpt_one_head.muon_svd_runner import _requested_optimizer


def _recorder_for_family(family: str) -> MuonUpdateSVDCapture:
    recorder = MuonUpdateSVDCapture.__new__(MuonUpdateSVDCapture)
    recorder.profile = {"family": family}
    return recorder


def test_requested_optimizer_supports_both_cli_forms() -> None:
    assert _requested_optimizer(["--optimizer", "muon"]) == "muon"
    assert _requested_optimizer(["--optimizer=muon_clip"]) == "muon_clip"
    assert _requested_optimizer(["--config", "x.yaml"]) is None


def test_muon_update_source_matches_optimizer_equations() -> None:
    recorder = _recorder_for_family("muon")
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    momentum_before = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    group = {"momentum": 0.9, "nesterov": True}

    momentum_after, update_source = recorder._reconstruct_update_source(
        gradient=gradient,
        momentum_before=momentum_before,
        group=group,
    )

    expected_momentum = momentum_before * 0.9 + gradient * 0.1
    expected_source = gradient * 0.1 + expected_momentum * 0.9
    torch.testing.assert_close(momentum_after, expected_momentum)
    torch.testing.assert_close(update_source, expected_source)


def test_muonclip_update_source_matches_optimizer_equations() -> None:
    recorder = _recorder_for_family("muon_clip")
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    momentum_before = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    group = {"momentum": 0.95, "nesterov": False}

    momentum_after, update_source = recorder._reconstruct_update_source(
        gradient=gradient,
        momentum_before=momentum_before,
        group=group,
    )

    expected_momentum = momentum_before * 0.95 + gradient
    torch.testing.assert_close(momentum_after, expected_momentum)
    torch.testing.assert_close(update_source, expected_momentum)


def test_thin_svd_frame_produces_square_comoving_weight() -> None:
    generator = torch.Generator().manual_seed(17)
    update_source = torch.randn(7, 3, generator=generator)
    weight = torch.randn(7, 3, generator=generator)

    u, s, vh = torch.linalg.svd(update_source, full_matrices=False)
    v = vh.transpose(-2, -1)
    quotient_weight = u.transpose(-2, -1) @ weight @ v

    assert u.shape == (7, 3)
    assert s.shape == (3,)
    assert vh.shape == (3, 3)
    assert quotient_weight.shape == (3, 3)
    torch.testing.assert_close(
        u @ torch.diag(s) @ vh,
        update_source,
        rtol=1e-5,
        atol=1e-6,
    )
