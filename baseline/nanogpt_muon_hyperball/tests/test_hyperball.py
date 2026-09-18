import torch

from rg_nanogpt_muon_hyperball.base_optimizers import Muon
from rg_nanogpt_muon_hyperball.optimizers import MuonHyperBall


def _clone_parameter(value: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(value.detach().clone())


def test_hyperball_caps_relative_frobenius_displacement() -> None:
    torch.manual_seed(7)
    initial = torch.randn(16, 8)
    parameter = _clone_parameter(initial)
    parameter.grad = torch.randn_like(parameter)

    rho = 1.0e-3
    optimizer = MuonHyperBall(
        [parameter],
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        relative_radius=rho,
    )
    optimizer.step()

    displacement = torch.linalg.vector_norm(parameter.detach() - initial)
    radius = rho * torch.linalg.vector_norm(initial)
    assert displacement <= radius * (1.0 + 1e-5)

    summary = optimizer.last_hyperball_summary()
    assert summary is not None
    assert float(summary["active_updates"]) == 1.0
    assert float(summary["applied_uwr_max"]) <= rho * (1.0 + 1e-5)


def test_radial_projection_preserves_muon_direction() -> None:
    torch.manual_seed(11)
    initial = torch.randn(16, 8)
    gradient = torch.randn_like(initial)

    plain_parameter = _clone_parameter(initial)
    ball_parameter = _clone_parameter(initial)
    plain_parameter.grad = gradient.clone()
    ball_parameter.grad = gradient.clone()

    plain = Muon(
        [plain_parameter],
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
    )
    ball = MuonHyperBall(
        [ball_parameter],
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        relative_radius=1.0e-4,
    )
    plain.step()
    ball.step()

    proposed = (plain_parameter.detach() - initial).flatten()
    applied = (ball_parameter.detach() - initial).flatten()
    cosine = torch.dot(proposed, applied) / (
        torch.linalg.vector_norm(proposed)
        * torch.linalg.vector_norm(applied)
    )
    assert torch.allclose(cosine, torch.tensor(1.0), atol=2e-5, rtol=2e-5)


def test_effectively_infinite_radius_matches_plain_muon() -> None:
    torch.manual_seed(19)
    initial = torch.randn(12, 12)
    gradient = torch.randn_like(initial)

    plain_parameter = _clone_parameter(initial)
    ball_parameter = _clone_parameter(initial)
    plain_parameter.grad = gradient.clone()
    ball_parameter.grad = gradient.clone()

    plain = Muon(
        [plain_parameter],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.02,
    )
    ball = MuonHyperBall(
        [ball_parameter],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.02,
        relative_radius=1.0e6,
    )
    plain.step()
    ball.step()

    assert torch.allclose(
        ball_parameter.detach(),
        plain_parameter.detach(),
        atol=2e-6,
        rtol=2e-6,
    )
    summary = ball.last_hyperball_summary()
    assert summary is not None
    assert float(summary["active_updates"]) == 0.0
