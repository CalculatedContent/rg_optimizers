from __future__ import annotations

import math

import torch

from rg_nanogpt_one_head.model import GPT, GPTConfig, transformer_matrix_items
from rg_nanogpt_one_head.optimizers import make_optimizer_handles


def _tiny_config(*, residual_mode: str, n_layer: int = 1) -> GPTConfig:
    return GPTConfig(
        vocab_size=97,
        block_size=16,
        n_layer=n_layer,
        n_head=1,
        n_embd=32,
        dropout=0.0,
        bias=False,
        tie_weights=False,
        residual_mode=residual_mode,
    )


def _muon_profile() -> dict:
    return {
        "family": "muon",
        "matrix_learning_rate": 0.01,
        "matrix_min_learning_rate": 0.0005,
        "aux_learning_rate": 0.0003,
        "aux_min_learning_rate": 0.00001,
        "momentum": 0.95,
        "nesterov": True,
        "newton_schulz_steps": 3,
        "muon_epsilon": 1.0e-7,
        "matrix_weight_decay": 0.01,
        "beta1": 0.90,
        "beta2": 0.95,
        "epsilon": 1.0e-8,
        "aux_weight_decay": 0.01,
    }


def test_full_attnres_forward_backward_is_finite() -> None:
    torch.manual_seed(7)
    model = GPT(_tiny_config(residual_mode="full_attnres"))
    idx = torch.randint(0, model.cfg.vocab_size, (3, 12))
    targets = torch.randint(0, model.cfg.vocab_size, (3, 12))

    logits, loss = model(idx, targets)

    assert logits.shape == (3, 12, model.cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    block = model.blocks[0]
    assert block.attn_res_router is not None
    assert block.mlp_res_router is not None
    assert block.attn_res_router.query.grad is not None
    assert block.mlp_res_router.query.grad is not None
    assert torch.isfinite(block.attn_res_router.query.grad).all()
    assert torch.isfinite(block.mlp_res_router.query.grad).all()


def test_one_block_routing_weights_have_expected_depth() -> None:
    torch.manual_seed(11)
    model = GPT(_tiny_config(residual_mode="full_attnres"))
    idx = torch.randint(0, model.cfg.vocab_size, (2, 10))
    model(idx)

    weights = model.attention_residual_weights()
    assert set(weights) == {"L00_ATTN", "L00_MLP"}
    assert len(weights["L00_ATTN"]) == 1
    assert len(weights["L00_MLP"]) == 2
    assert math.isclose(weights["L00_ATTN"][0], 1.0, abs_tol=1e-6)
    assert math.isclose(sum(weights["L00_MLP"]), 1.0, abs_tol=1e-6)


def test_transformer_matrices_remain_separate_from_attnres_queries() -> None:
    model = GPT(_tiny_config(residual_mode="full_attnres"))
    items = transformer_matrix_items(model)

    assert [item[1] for item in items] == [
        "W_Q",
        "W_K",
        "W_V",
        "W_O",
        "W_MLP_IN",
        "W_MLP_OUT",
    ]
    assert len(items) == 6
    matrix_ids = {id(item[3]) for item in items}
    assert id(model.blocks[0].attn_res_router.query) not in matrix_ids
    assert id(model.blocks[0].mlp_res_router.query) not in matrix_ids


def test_muon_keeps_attnres_queries_in_auxiliary_optimizer() -> None:
    model = GPT(_tiny_config(residual_mode="full_attnres"))
    handles = make_optimizer_handles(model, _muon_profile())

    assert [handle.role for handle in handles] == ["primary", "auxiliary"]
    primary_ids = {
        id(parameter)
        for group in handles[0].optimizer.param_groups
        for parameter in group["params"]
    }
    auxiliary_ids = {
        id(parameter)
        for group in handles[1].optimizer.param_groups
        for parameter in group["params"]
    }

    for _, _, _, matrix in transformer_matrix_items(model):
        assert id(matrix) in primary_ids
    for block in model.blocks:
        assert id(block.attn_res_router.query) not in primary_ids
        assert id(block.mlp_res_router.query) not in primary_ids
        assert id(block.attn_res_router.query) in auxiliary_ids
        assert id(block.mlp_res_router.query) in auxiliary_ids


def test_full_attnres_is_depth_ready_while_preserving_one_head() -> None:
    torch.manual_seed(13)
    model = GPT(_tiny_config(residual_mode="full_attnres", n_layer=3))
    idx = torch.randint(0, model.cfg.vocab_size, (2, 8))
    logits, _ = model(idx)

    assert logits.shape == (2, 8, model.cfg.vocab_size)
    assert len(transformer_matrix_items(model)) == 18
    weights = model.attention_residual_weights()
    assert len(weights["L00_ATTN"]) == 1
    assert len(weights["L00_MLP"]) == 2
    assert len(weights["L01_ATTN"]) == 3
    assert len(weights["L01_MLP"]) == 4
    assert len(weights["L02_ATTN"]) == 5
    assert len(weights["L02_MLP"]) == 6
