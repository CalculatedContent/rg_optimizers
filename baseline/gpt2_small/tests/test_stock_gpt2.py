"""Architecture parity, parameter ownership and both training paths on CPU."""
import ast
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/'muon_speedrun'))
import stock_model as stock
from optim import make_optimizers, apply_update
from runtime import Runtime


def reference():
    # The pinned upstream GPT-2 reference already in the repository. Execute
    # definitions only: never its CUDA, distributed or data-loading top level.
    path = BASE/'speedrun30/vendor/llmc_train_gpt2.py'
    allowed = {'NewGELU', 'CausalSelfAttention', 'MLP', 'Block', 'GPTConfig', 'GPT'}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name in allowed]
    namespace = dict(torch=torch, nn=nn, F=F, math=math, dataclass=dataclass, FLASH=1)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def tiny(**kwargs):
    return stock.GPT(stock.GPTConfig(vocab_size=128, block_size=32, n_layer=2,
                                     n_head=3, n_embd=24), **kwargs)


def test_stock_dimensions_count_and_tying():
    with torch.device('meta'):
        model = stock.GPT()
    assert sum(p.numel() for p in model.parameters()) == 124439808
    assert model.lm_head.weight is model.transformer.wte.weight
    inventory = stock.matrix_inventory(model)
    assert len(inventory) == 75  # 72 block matrices, two embeddings, one tied alias.
    expected = {'c_q':(768,768), 'c_k':(768,768), 'c_v':(768,768), 'c_proj':(768,768)}
    for block in model.transformer.h:
        assert block.attn.n_head == 12 and block.attn.head_dim == 64
        for name, shape in expected.items():
            layer = getattr(block.attn, name)
            assert tuple(layer.weight.shape) == shape and layer.bias is not None
        assert block.mlp.c_fc.weight.shape == (3072,768)
        assert block.mlp.c_proj.weight.shape == (768,3072)
        assert isinstance(block.ln_1, nn.LayerNorm) and block.ln_1.bias is not None
    converted = tiny().bfloat16().float()
    assert converted.lm_head.weight is converted.transformer.wte.weight


def test_forward_and_all_gradients_match_packed_qkv_gpt2():
    torch.set_num_threads(1)
    ref = reference()
    cfg = stock.GPTConfig(vocab_size=128, block_size=32, n_layer=2, n_head=3, n_embd=24)
    expected = ref.GPT(ref.GPTConfig(**vars(cfg)))
    actual = stock.GPT(cfg)
    # Exercise nonzero biases as well as the standard random matrices.
    with torch.no_grad():
        for name, p in expected.named_parameters():
            if name.endswith('.bias'):
                p.normal_(std=.01)
    source = expected.state_dict()
    mapped = {}
    for name in actual.state_dict():
        if any('.'+role+'.' in name for role in ('c_q','c_k','c_v')):
            role = next(r for r in ('c_q','c_k','c_v') if '.'+r+'.' in name)
            packed = name.replace('.'+role+'.', '.c_attn.')
            mapped[name] = source[packed].chunk(3, dim=0)[('c_q','c_k','c_v').index(role)]
        else:
            mapped[name] = source[name]
    actual.load_state_dict(mapped)
    x = torch.randint(128, (2, 13)); y = torch.randint(128, (2, 13))
    logits, loss = expected(x, y)
    actual_logits = actual.logits(x)
    actual_loss = actual(x, y)
    torch.testing.assert_close(actual_logits, logits, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(actual_loss, loss, rtol=1e-6, atol=1e-7)
    loss.backward(); actual_loss.backward()
    grads = dict(expected.named_parameters())
    for name, p in actual.named_parameters():
        if any('.'+role+'.' in name for role in ('c_q','c_k','c_v')):
            role = next(r for r in ('c_q','c_k','c_v') if '.'+r+'.' in name)
            g = grads[name.replace('.'+role+'.', '.c_attn.')].grad.chunk(3, dim=0)[('c_q','c_k','c_v').index(role)]
        else:
            g = grads[name].grad
        torch.testing.assert_close(p.grad, g, rtol=1e-4, atol=2e-7)
    changed = x.clone(); changed[:,7:] = (changed[:,7:] + 1) % 128
    torch.testing.assert_close(actual.logits(changed)[:,:7], actual_logits[:,:7], rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['muon', 'adamw'])
def test_both_recipes_own_every_parameter_once_and_learn(kind, tmp_path):
    torch.set_num_threads(1); torch.manual_seed(1337)
    model = tiny(activation_dtype=torch.bfloat16)
    rt = Runtime('cpu'); muon, adam = make_optimizers(model, rt, kind)
    assert type(adam) is torch.optim.AdamW
    owned = [p for group in adam.param_groups for p in group['params']]
    if muon is not None:
        owned += [p for group in muon.groups for _,p in group['entries']]
    assert len(owned) == len({id(p) for p in owned}) == len(list(model.parameters()))
    assert {id(p) for p in owned} == {id(p) for p in model.parameters()}
    assert all(p.dtype == torch.float32 for p in owned)
    assert all(g['peak_lr'] == .0006 for g in adam.param_groups)
    x = torch.randint(128, (4, 16)); losses = []
    for step in range(8):
        model.zero_grad(set_to_none=False)
        loss = model(x, x); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in owned)
        apply_update(muon, adam, rt, step)
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    measured, errors = model(x, x, return_token_errors=True)
    torch.testing.assert_close(measured, model(x, x), rtol=0, atol=0)
    assert int(errors) == int((model.logits(x).argmax(-1) != x).sum())
    spec = importlib.util.spec_from_file_location('stock_runner', BASE/'muon_speedrun/run.py')
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    assert run.architecture is stock  # Both CLI optimizer choices use this model.
    validation = dict(step=125, val_nll=float(measured.detach()), val_token_error=int(errors)/x.numel(),
                      full_benchmark_evaluation=True, evaluation_tokens=10485760)
    run.save_checkpoint(tmp_path, model, muon, adam, types.SimpleNamespace(shard=1,position=0),
                        125, {'architecture':stock.ARCHITECTURE}, rt, validation, float('inf'))
    snapshot = torch.load(tmp_path/'tracking/snapshots/0000125.pt', weights_only=False)
    assert len(snapshot['matrices']) == 12
    assert snapshot['manifest']['architecture'] == stock.ARCHITECTURE
    state = torch.load(tmp_path/'checkpoint_latest.pt', weights_only=False)
    restored = tiny(activation_dtype=torch.bfloat16); restored.load_state_dict(state['model'])
    assert restored.lm_head.weight is restored.transformer.wte.weight
    torch.testing.assert_close(restored(x,x), model(x,x), rtol=0, atol=0)
