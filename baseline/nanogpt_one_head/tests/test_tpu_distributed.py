from copy import deepcopy

import pytest

from rg_nanogpt_one_head.tpu_distributed import (
    derive_distributed_batch_plan,
    distributed_protocol_config,
)


def _config():
    return {
        "model": {"block_size": 256},
        "training": {
            "batch_size": 4,
            "grad_accum_steps": 8,
        },
        "runtime": {"matmul_precision": "highest"},
    }


def test_fourway_plan_preserves_global_batch():
    cfg = _config()
    plan = derive_distributed_batch_plan(cfg, world_size=4)

    assert plan.local_grad_accum_steps == 2
    assert plan.global_grad_accum_steps == 8
    assert plan.global_sequences_per_step == 32
    assert plan.global_tokens_per_step == 8192
    assert plan.local_tokens_per_step == 2048
    assert plan.local_tokens_per_step * plan.world_size == 8192


def test_world_size_must_divide_global_accumulation():
    with pytest.raises(ValueError, match="divisible"):
        derive_distributed_batch_plan(_config(), world_size=3)


def test_distributed_metadata_is_additive_and_does_not_mutate_input():
    cfg = _config()
    original = deepcopy(cfg)
    plan = derive_distributed_batch_plan(cfg, world_size=4)

    distributed = distributed_protocol_config(cfg, plan=plan)

    assert cfg == original
    assert distributed["runtime"]["matmul_precision"] == "highest"
    metadata = distributed["runtime"]["distributed_tpu"]
    assert metadata["enabled"] is True
    assert metadata["world_size"] == 4
    assert metadata["local_grad_accum_steps"] == 2
    assert metadata["global_tokens_per_step"] == 8192
