"""Numerical diagnostics and isolated replay tests; not MPS failure reproduction."""
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import diagnose_gradients as diag
import run as driver


@pytest.fixture(autouse=True)
def restore_runtime():
    state = torch.random.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    precision = torch.get_float32_matmul_precision()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    env = {k: os.environ.get(k) for k in ('PYTORCH_ENABLE_MPS_FALLBACK', 'CUBLAS_WORKSPACE_CONFIG')}
    yield
    torch.random.set_rng_state(state)
    torch.use_deterministic_algorithms(deterministic)
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_native_nonfinite_check_preserves_original_gradients():
    model = torch.nn.Linear(3, 1, bias=False)
    model.weight.grad = torch.tensor([[1., float('inf'), float('nan')]])
    before = model.weight.grad.clone()
    with pytest.raises(RuntimeError):
        diag.checked_clip(model, 1.)
    torch.testing.assert_close(model.weight.grad, before, equal_nan=True)
    report = diag.gradient_statistics(model)
    assert report['nonfinite_parameters'] == ['weight']
    assert report['per_parameter']['weight']['nan_count'] == 1
    assert report['per_parameter']['weight']['inf_count'] == 1
    assert report['total_l2_cpu_float64'] is None


def test_finite_gradients_can_overflow_native_norm():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 1e20)
    before = model.weight.grad.clone()
    with pytest.raises(RuntimeError):
        diag.checked_clip(model, 1.)
    torch.testing.assert_close(before, model.weight.grad)
    report = diag.gradient_statistics(model)
    assert report['all_gradients_finite']
    assert math.isfinite(report['total_l2_cpu_float64'])
    assert report['total_l2_cpu_float64'] == pytest.approx(math.sqrt(2) * 1e20)


@pytest.mark.parametrize('scale', [0., .01, 1., 1000.])
def test_checked_clip_matches_original_for_finite_norms(scale):
    torch.manual_seed(3)
    a, b = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    for p, q in zip(a.parameters(), b.parameters()):
        p.grad = scale * torch.randn_like(p)
        q.grad = p.grad.clone()
    x = diag.checked_clip(a, 1.)
    y = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.)
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_statistics_do_not_change_model_or_rng():
    model = torch.nn.Linear(3, 2)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    state = torch.random.get_rng_state().clone()
    report = diag.gradient_statistics(model)
    json.dumps(report, allow_nan=False)
    assert torch.equal(state, torch.random.get_rng_state())
    for n, p in model.named_parameters():
        assert torch.equal(weights[n], p)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 0., 1.])
def test_json_encoding_of_nonfinite_numbers(value):
    json.dumps({'value': diag.encoded_number(value)}, allow_nan=False)


def test_selects_latest_incomplete_matching_run(tmp_path):
    old = tmp_path / 'old' / 'full/repository/rule_random/adamw/seed_1337'
    new = tmp_path / 'new' / 'full/repository/rule_random/adamw/seed_1337'
    for r in [old, new]:
        r.mkdir(parents=True)
        (r / 'checkpoint_latest.pt').write_bytes(b'stub')
        (r / 'manifest.json').write_text('{}')
    args = SimpleNamespace(run_dir=None, root=str(tmp_path / 'new'), condition='rule_random', optimizer='adamw', seed=1337)
    assert diag.select_run(args) == new
    (new / 'complete.json').write_text('{}')
    with pytest.raises(ValueError, match='No saved incomplete'):
        diag.select_run(args)
    args.run_dir = str(old)
    assert diag.select_run(args) == old


class TinyGPT(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = torch.nn.Embedding(8, 4)
        self.output = torch.nn.Linear(4, 8)

    def forward(self, x, y):
        logits = self.output(self.embedding(x))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 8), y.reshape(-1))
        return logits, loss


class TinyStudy:
    steps = 5
    def __init__(self, *a, **kw):
        pass
    def identity(self):
        return 'test-data'
    def sample(self, step):
        return [driver.Record('a', 'test', (1, 2), (3,)),
                driver.Record('b', 'test', (2, 3), (4,))]


def install_toy_modules(monkeypatch):
    module = ModuleType('rg_nanogpt_one_head.model')
    module.GPT, module.GPTConfig = TinyGPT, SimpleNamespace
    monkeypatch.setitem(sys.modules, 'rg_nanogpt_one_head.model', module)
    opt = ModuleType('rg_nanogpt_one_head.optimizers')
    opt.make_optimizer_handles = lambda model, profile: [SimpleNamespace(optimizer=torch.optim.AdamW(model.parameters(), lr=.01))]
    opt.zero_grad = lambda hs: [h.optimizer.zero_grad(set_to_none=True) for h in hs]
    opt.optimizer_step = lambda hs: [h.optimizer.step() for h in hs]
    def schedule(hs, **kwargs):
        return {'primary': .01}
    opt.set_learning_rates = schedule
    monkeypatch.setitem(sys.modules, 'rg_nanogpt_one_head.optimizers', opt)
    return opt


def toy_run(tmp_path, monkeypatch):
    import platform
    opt = install_toy_modules(monkeypatch)
    source = {'model': {'block_size': 8, 'dropout': 0.},
              'training': {'batch_size': 1, 'grad_accum_steps': 2, 'grad_clip': 1.}}
    monkeypatch.setattr(driver, 'load_source', lambda cfg: source)
    monkeypatch.setattr(driver, 'resolve_profile', lambda *a: {'family': 'adamw'})
    monkeypatch.setattr(driver, 'Study', TinyStudy)
    manifest = {'suite': {}, 'runner_sha256': diag.sha256(HERE / 'run.py'),
                'source_model': source['model'], 'profile': {'family': 'adamw'},
                'optimizer': 'adamw', 'condition': 'rule_random', 'stage': 'full',
                'recipe': 'repository', 'seed': 1337, 'data_sha256': 'test-data',
                'schedule_steps': 5, 'warmup_steps': 1,
                'device': {'device': 'cpu', 'torch': str(torch.__version__), 'numpy': np.__version__,
                           'python': platform.python_version(), 'machine': platform.machine(),
                           'platform': platform.platform()}}
    manifest['fingerprint'] = driver.digest(manifest)
    model = TinyGPT(SimpleNamespace(**source['model']))
    handles = opt.make_optimizer_handles(model, manifest['profile'])
    run = tmp_path / 'seed_1337'
    run.mkdir()
    (run / '.lock').write_bytes(b'')
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'metrics.jsonl').write_text(json.dumps({'step': 0, 'model_sha256': driver.state_digest(model)}) + '\n')
    torch.save({'step': 0, 'model': model.state_dict(), 'fingerprint': manifest['fingerprint'],
                'optimizers': [h.optimizer.state_dict() for h in handles], 'torch_rng': torch.get_rng_state(),
                'device_rng': None}, run / 'checkpoint_latest.pt')
    args = SimpleNamespace(run_dir=str(run), root=None, device=None, allow_runtime_mismatch=False,
                           updates=3, no_cpu_check=False)
    return run, args


def test_replay_leaves_parent_files_unchanged_and_does_not_mark_complete(tmp_path, monkeypatch):
    run, args = toy_run(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    dest = diag.diagnose(args)
    report = json.loads((dest / 'report.json').read_text())
    assert report['status'] == 'no_failure_in_replay_window'
    assert report['completed_updates'] == 3
    assert report['original_checkpoint_unchanged']
    assert len((dest / 'replay_trace.jsonl').read_text().splitlines()) == 3
    assert before == {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    assert not (run / 'complete.json').exists()


def test_replay_preserves_failure_evidence_before_clipping(tmp_path, monkeypatch):
    run, args = toy_run(tmp_path, monkeypatch)
    original = diag.checked_clip
    def inject(model, max_norm):
        p = next(model.parameters())
        p.grad[0, 0] = float('nan')
        return original(model, max_norm)
    monkeypatch.setattr(diag, 'checked_clip', inject)
    dest = diag.diagnose(args)
    report = json.loads((dest / 'report.json').read_text())
    assert report['status'] == 'failure_observed'
    assert report['observed_failure'] == 'nonfinite_gradient_entries'
    assert report['failure_step_index'] == 0
    saved = torch.load(dest / 'failure_state.pt', weights_only=True)
    assert saved['diagnostic_only']
    assert torch.isnan(saved['gradients']['embedding.weight'][0, 0])
    assert report['original_checkpoint_unchanged']


def test_replay_rejects_invalid_manifest(tmp_path, monkeypatch):
    run, args = toy_run(tmp_path, monkeypatch)
    path = run / 'manifest.json'
    data = json.loads(path.read_text()); data['seed'] = 8
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Manifest fingerprint'):
        diag.diagnose(args)
    assert not (run / 'diagnostics').exists()


def test_replay_refuses_active_trainer(tmp_path, monkeypatch):
    import fcntl
    run, args = toy_run(tmp_path, monkeypatch)
    with (run / '.lock').open('rb') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='still active'):
            diag.diagnose(args)


def test_cpu_recompute_is_gradient_only(tmp_path, monkeypatch):
    run, args = toy_run(tmp_path, monkeypatch)
    checkpoint = torch.load(run / 'checkpoint_latest.pt', weights_only=True)
    source = driver.load_source({})
    result = diag.same_state_cpu_check(TinyGPT, SimpleNamespace(**source['model']),
                                       checkpoint['model'], TinyStudy().sample(0), source, driver)
    assert result['status'] == 'completed'
    assert result['gradients']['all_gradients_finite']
    assert len(result['microbatch_losses']) == 2
