"""CPU tests; no downloads, GPU allocation, training or upstream CUDA imports."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = load('leaderboard_reference', HERE/'experiment.py')
exporter = load('leaderboard_weight_export', HERE/'weight_export.py')


def test_exact_upstream_bytes_and_complete_import_package():
    assert reference.verify() == 58
    manifest = reference.read_json(HERE/'upstream_git_files.json')
    for name, item in manifest.items():
        data = (HERE/'vendor'/name).read_bytes()
        assert len(data) == item['size']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == item['git_blob']
        if name.endswith('.py'):
            for node in ast.walk(ast.parse(data)):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('track_1_short'):
                    target = node.module.replace('.', '/')
                    assert target+'.py' in manifest or target+'/__init__.py' in manifest


def test_source_change_is_rejected(tmp_path):
    shutil.copytree(HERE/'vendor', tmp_path/'source')
    with (tmp_path/'source/train_gpt.py').open('a') as handle:
        handle.write('\n# changed\n')
    with pytest.raises(RuntimeError, match='changed'):
        reference.verify(tmp_path/'source')


def test_manifest_schedule_and_model_match_pinned_source(monkeypatch):
    from itertools import accumulate
    monkeypatch.delenv('NUM_SCHEDULED_ITERATIONS', raising=False)
    monkeypatch.delenv('TRAIN_SEED', raising=False)
    config = load('leaderboard_config', HERE/'vendor/track_1_short/config.py')
    args = config.Hyperparameters()
    manifest = reference.read_json(HERE/'upstream.json')
    assert args.num_scheduled_iterations + config.SCHEDULE_GROWTH_STEPS + args.num_extension_iterations == manifest['total_steps'] == 1194
    ends = [0, *[round(c * args.num_scheduled_iterations) for c in accumulate(s.duration for s in config.TRAINING_STAGES[:-1])], manifest['total_steps']]
    ends[config.SCHEDULE_GROWTH_STAGE+1:-1] = [x+config.SCHEDULE_GROWTH_STEPS for x in ends[config.SCHEDULE_GROWTH_STAGE+1:-1]]
    assert ends == manifest['stage_boundaries']
    assert sum((b-a)*s.batch_size for a,b,s in zip(ends,ends[1:],config.TRAINING_STAGES)) == manifest['total_train_tokens']
    tree = ast.parse((HERE/'vendor/train_gpt.py').read_text())
    model_call = next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='GPT')
    dims = {k.arg:ast.literal_eval(k.value) for k in model_call.keywords if k.arg in ('num_layers','num_heads','model_dim','head_dim')}
    assert dims == dict(num_layers=11,num_heads=6,model_dim=768,head_dim=128)
    assert args.save_checkpoint is False and args.val_loss_every == 0 and args.train_seed is None
    assert 'dist.destroy_process_group()' in (HERE/'vendor/train_gpt.py').read_text()


@pytest.mark.parametrize('save_weights',[False,True])
def test_run_stages_exact_source_and_isolates_results(tmp_path,monkeypatch,save_weights):
    monkeypatch.delenv('NUM_SCHEDULED_ITERATIONS', raising=False)
    monkeypatch.delenv('TRAIN_SEED', raising=False)
    data_root = tmp_path/'data-cache'
    for path in reference.required_data(data_root):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'test-data')
    root = tmp_path/'runs/one'
    command, env = reference.stage_run(root, data_root, {'mock':True}, save_weights)
    assert '--nproc_per_node=8' in command
    assert Path(command[-1]) == root/('rank_entry.py' if save_weights else 'source/train_gpt.py')
    assert env['DATA_PATH'] == str(data_root)
    assert ('RG_LEADERBOARD_EXPORT_ROOT' in env) == save_weights
    assert reference.verify(root/'source') == 58
    manifest = reference.read_json(root/'manifest.json')
    assert len(manifest['data_sha256']) == 10 and manifest['save_weights'] == save_weights
    with pytest.raises(FileExistsError):
        reference.stage_run(root, data_root, {}, save_weights)


def test_cpu_cannot_launch_reference(monkeypatch):
    monkeypatch.setattr(torch, '__version__', '2.10.0+cu128')
    monkeypatch.setattr(torch.version, 'cuda', '12.8')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda:False)
    with pytest.raises(RuntimeError, match='TPU/CPU unsupported'):
        reference.check_environment()


def test_recipe_overrides_and_missing_data_fail_before_creating_run(tmp_path,monkeypatch):
    root=tmp_path/'run'
    data=tmp_path/'cache'
    with pytest.raises(RuntimeError,match='Missing FineWeb'):
        reference.stage_run(root,data,{},False)
    assert not root.exists()
    for path in reference.required_data(data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'test')
    monkeypatch.setenv('NUM_SCHEDULED_ITERATIONS','10')
    with pytest.raises(RuntimeError,match='Unset'):
        reference.stage_run(root,data,{},False)
    assert not root.exists()


def fixture_model():
    model = torch.nn.Linear(3,2)
    model.register_buffer('canon_mask', torch.ones(4,dtype=torch.uint8), persistent=False)
    model.register_buffer('prefix_table', torch.arange(4), persistent=False)
    return model


def test_weight_export_keeps_dense_state_and_all_sparse_rows(tmp_path):
    model = fixture_model()
    table = torch.arange(30,dtype=torch.bfloat16).reshape(10,3)
    for rank in range(2):
        exporter.export_rank(tmp_path, model, types.SimpleNamespace(shard=table[rank*5:(rank+1)*5], first_row=rank*5),
                             rank=rank, world_size=2, step=1194, chunk_rows=2)
    result = exporter.verify_export(tmp_path, 2, 10)
    assert result['status']=='complete'
    dense = torch.load(tmp_path/'rank-00/model.pt', map_location='cpu', weights_only=True)
    torch.testing.assert_close(dense['model']['weight'],model.weight)
    torch.testing.assert_close(dense['evaluation_buffers']['canon_mask'],model.canon_mask)
    chunks = [torch.load(p,map_location='cpu',weights_only=True) for p in sorted(tmp_path.glob('rank-*/ngram-*.pt'))]
    assert [x['first_row'] for x in chunks] == [0,2,4,5,7,9]
    torch.testing.assert_close(torch.cat([x['weights'] for x in chunks]),table)
    assert all(x['weights'].device.type=='cpu' for x in chunks)
    path = tmp_path/'rank-01/ngram-000000000.pt'
    path.write_bytes(b'corrupted')
    with pytest.raises(RuntimeError,match='integrity'):
        exporter.verify_export(tmp_path,2,10)


def test_missing_rank_cannot_be_reported_complete(tmp_path):
    exporter.export_rank(tmp_path, fixture_model(), types.SimpleNamespace(shard=torch.zeros(2,3),first_row=0),rank=0,world_size=2,step=1194)
    with pytest.raises(FileNotFoundError):
        exporter.verify_export(tmp_path,2,4)
    assert not (tmp_path/'WEIGHTS_COMPLETE.json').exists()


def test_missing_chunk_receipt_cannot_hide_partial_export(tmp_path):
    exporter.export_rank(tmp_path,fixture_model(),types.SimpleNamespace(shard=torch.zeros(4,3),first_row=0),rank=0,world_size=1,step=1194,chunk_rows=2)
    path=tmp_path/'rank-00/EXPORT.json'
    data=json.loads(path.read_text());data['files'].pop();path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='do not cover'):
        exporter.verify_export(tmp_path,1,4)
    assert not (tmp_path/'WEIGHTS_COMPLETE.json').exists()


def test_optional_adapter_exports_only_at_completed_main_shutdown(tmp_path,monkeypatch):
    entry = load('leaderboard_rank_entry', HERE/'rank_entry.py')
    monkeypatch.setattr(entry,'__file__',str(tmp_path/'rank_entry.py'))
    monkeypatch.setenv('RG_LEADERBOARD_EXPORT_ROOT',str(tmp_path/'weights'))
    events = []
    monkeypatch.setitem(sys.modules,'weight_export',types.SimpleNamespace(export_rank=lambda *a,**k:events.append(('export',k))))
    import torch.distributed as dist
    original = lambda *a,**k:events.append(('destroy',None))
    monkeypatch.setattr(dist,'destroy_process_group',original)
    (tmp_path/'source').mkdir()
    (tmp_path/'source/train_gpt.py').write_text('''from types import SimpleNamespace
import torch.distributed as dist
def main():
    step=1194
    training_schedule=SimpleNamespace(total_steps=1194)
    uncompiled_model,ngram_table='model','table'
    env=SimpleNamespace(rank=0,world_size=8)
    dist.destroy_process_group()
main()
''')
    entry.main()
    assert events == [('export',dict(rank=0,world_size=8,step=1194)),('destroy',None)]
    assert dist.destroy_process_group is original
