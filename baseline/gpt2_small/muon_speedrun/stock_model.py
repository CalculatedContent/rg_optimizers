"""Load the byte-identical pinned upstream GPT-2 model; no replacement layers.

Hardware attention selection and experiment seeds live outside model definitions.
The original source and MIT license are retained under speedrun30/vendor/.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import torch
import torch.nn.functional as F
from stock_config import ARCHITECTURE

SOURCE = Path(__file__).resolve().parents[1]/'speedrun30/vendor/llmc_train_gpt2.py'
SOURCE_SHA256 = '757d0cea0d48cbc4c7d7d70371f955d49cdf3a7cfb4c87701a720c2fe0905c34'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError('Pinned upstream GPT-2 source changed; refusing to train')
spec = importlib.util.spec_from_file_location('rg_pinned_llmc_gpt2', SOURCE)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
reference.FLASH = 1
GPT = reference.GPT
GPTConfig = reference.GPTConfig


def make_model(config=None, seed=42, device=None):
    model = GPT(config or GPTConfig())
    # The upstream constructor uses seed 42. Repeated-seed experiments call
    # its unchanged initializer with their explicit seed, not a new algorithm.
    if seed != 42:
        model.init_rng.manual_seed(seed)
        model.apply(model._init_weights)
    if device is not None:
        model.to(device)
        model.lm_head.weight = model.transformer.wte.weight
    model.architecture_id = ARCHITECTURE
    return model


def configure_attention(implementation=None):
    """Select a numerically checked TPU kernel without editing upstream forward."""
    reference.F = F
    if implementation is not None:
        def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            if attn_mask is not None or dropout_p != 0 or not is_causal or kwargs:
                raise ValueError('Unexpected upstream attention arguments')
            return implementation(q, k, v)
        reference.F = types.SimpleNamespace(**{**vars(F), 'scaled_dot_product_attention':sdpa})


def matrix_inventory(model):
    return [dict(name=name, shape=list(p.shape), elements=p.numel(),
                 shared_with='transformer.wte.weight' if name=='lm_head.weight' else None)
            for name,p in model.named_parameters(remove_duplicate=False) if p.ndim==2]


if __name__ == '__main__':
    import json
    with torch.device('meta'):
        model = make_model()
    print(json.dumps(dict(architecture=ARCHITECTURE, source_sha256=SOURCE_SHA256,
                         parameters=sum(p.numel() for p in model.parameters()),
                         matrices=matrix_inventory(model)), indent=2))
