"""Fixed epoch20-to60 continuation; no optimizer reset or endpoint selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import concurrent.futures as cf
import copy
import hashlib
import importlib.util
import json
import math
import multiprocessing
import shutil
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'Protocol.json').read_text())
REPO = next(p for p in ROOT.parents if (p / '.git').exists())
PARENT = REPO / PLAN['original_fashion_folder']
OLD = REPO / PLAN['parent_execution_folder']
KERNEL = PARENT / 'execution'
STATES = ROOT / 'parent_states'


def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


assert file_sha(KERNEL / 'study.py') == PLAN['kernel_sha256']
assert file_sha(OLD / 'run.py') == PLAN['parent_runner_sha256']
assert file_sha(OLD / 'Protocol.json') == PLAN['parent_protocol_sha256']
os.environ['FASHION_PARENT'] = str(PARENT)
spec_module = importlib.util.spec_from_file_location('frozen_clean_helpers', OLD / 'run.py')
c = importlib.util.module_from_spec(spec_module)
spec_module.loader.exec_module(c)
s = c.s
c.ROOT = ROOT
s.OUT = ROOT


def schedule(epoch):
    assert 20 <= epoch <= 60
    return .01 + .045 * (1 + math.cos(math.pi * (epoch - 20) / 40))


def equal_state(a, b):
    if s.torch.is_tensor(a):
        return s.torch.is_tensor(b) and s.torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def restore_model(sp, ck):
    net = s.MLP()
    opt = s.make_opt(net, sp)
    net.load_state_dict(ck['model'])
    # CPU state tensors may otherwise alias the checkpoint or a reference optimizer.
    opt.load_state_dict(copy.deepcopy(ck['optimizer']))
    s.torch.set_rng_state(ck['torch_rng'])
    assert s.weight_hash(net) == ck['model_hash']
    assert equal_state(opt.state_dict(), ck['optimizer'])
    assert s.torch.equal(s.torch.get_rng_state(), ck['torch_rng'])
    return net, opt


def compare_recorded(loss, stats, old_loss, old_stats):
    limits = PLAN['cross_runtime_replay']
    a, b = np.asarray(stats, np.float32), np.asarray(old_stats, np.float32)
    finite = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a.astype(float) - b.astype(float))
    bound = limits['statistics_absolute_tolerance'] + limits['statistics_relative_tolerance'] * np.abs(b.astype(float))
    same_pattern = np.array_equal(np.isnan(a), np.isnan(b)) and not np.isinf(a).any() and not np.isinf(b).any()
    loss_error = abs(loss - old_loss)
    passed = bool(np.isfinite(loss) and np.isfinite(old_loss) and loss_error <= limits['loss_absolute_tolerance'] and same_pattern and np.all(diff[finite] <= bound[finite]))
    return {'passed': passed, 'saved_loss': old_loss, 'replayed_loss': loss,
            'loss_absolute_error': loss_error, 'loss_exact': loss == old_loss,
            'statistics_exact': bool(np.array_equal(a, b, equal_nan=True)),
            'maximum_statistics_absolute_error': float(diff[finite].max()),
            'maximum_statistics_bound_fraction': float((diff[finite] / bound[finite]).max()),
            'saved_statistics': b.tolist(), 'replayed_statistics': a.tolist(),
            'same_nan_pattern': same_pattern}


def replay_two(sp, checkpoint, perturbation=None):
    net, opt = restore_model(sp, checkpoint)
    if perturbation == 'zeroed_moments':
        for state in opt.state.values():
            for key in ('momentum_buffer', 'exp_avg', 'exp_avg_sq'):
                if key in state: state[key].zero_()
    epoch = checkpoint['epoch'] + 1
    order = s.order(sp['seed'], epoch)
    values = []
    for j in range(2):
        batch = j + 2 if perturbation == 'wrong_batch' else j
        ix = order[batch * sp['batch_size']:(batch + 1) * sp['batch_size']]
        factor = c.schedule(epoch) * (1.1 if perturbation == 'wrong_lr' else 1.)
        loss, stats = s.train_step(net, opt, sp, s.DATA[0][ix], s.DATA[1][ix], factor, True)
        values.append((loss, np.array(stats, np.float32)))
    return net, opt, values


def verify_parent_replay(sp, checkpoint, recorded, dest):
    a, oa, values = replay_two(sp, checkpoint)
    b, ob, duplicate = replay_two(sp, checkpoint)
    exact = equal_state(a.state_dict(), b.state_dict()) and equal_state(oa.state_dict(), ob.state_dict())
    exact = exact and all(x[0] == y[0] and np.array_equal(x[1], y[1], equal_nan=True) for x, y in zip(values, duplicate))
    checks = []
    for j, (loss, stats) in enumerate(values):
        k = int(np.flatnonzero(recorded['batch_index'] == j)[0])
        checks.append({'batch': j, **compare_recorded(loss, stats, float(recorded['loss'][k]), recorded['values'][k])})
    result = {'spec': sp, 'checks': checks, 'current_runtime_replay_exact': bool(exact),
              'cross_runtime_replay_exact': all(x['loss_exact'] and x['statistics_exact'] for x in checks),
              'passed': bool(exact and all(x['passed'] for x in checks))}
    s.writej(dest / 'Parent_Replay.json', result)
    assert result['passed'], f"Parent replay mismatch: {sp}; evidence={dest / 'Parent_Replay.json'}"
    return result


def negative_controls(specs):
    records = []
    for sp in specs:
        d = STATES / 'runs' / s.run_id(sp)
        ck = s.torch.load(d / 'epoch_001.pt', weights_only=False)
        _, _, normal = replay_two(sp, ck)
        for perturbation in ('wrong_batch', 'zeroed_moments', 'wrong_lr'):
            _, _, changed = replay_two(sp, ck, perturbation)
            checks = [compare_recorded(a[0], a[1], b[0], b[1]) for a, b in zip(changed, normal)]
            records.append({'spec': sp, 'perturbation': perturbation,
                            'correctly_rejected': not all(x['passed'] for x in checks), 'checks': checks})
    result = {'passed': len(records) == 9 and all(x['correctly_rejected'] for x in records), 'records': records}
    s.writej(ROOT / 'Replay_Negative_Controls.json', result)
    assert result['passed'], 'Replay validator failed a deliberately incorrect state/input/rate control'


def guarded_parent_audit(sp):
    try:
        return parent_audit(sp)
    except Exception as exc:
        row = {'spec': sp, 'passed': False, 'error': repr(exc), 'traceback': traceback.format_exc()}
        s.writej(ROOT / 'restoration' / s.run_id(sp) / 'Failure.json', row)
        return row


def parent_audit(sp):
    d = STATES / 'runs' / s.run_id(sp)
    r = json.loads((d / 'done.json').read_text())
    assert r['spec'] == sp and r['endpoint_epoch'] == 20
    assert file_sha(d / 'final.pt') == r['checkpoint_hashes']['final.pt']
    ck = s.torch.load(d / 'final.pt', weights_only=False)
    assert ck['epoch'] == 20 and ck['spec'] == sp
    assert ck['kernel_sha256'] == PLAN['kernel_sha256'] and ck['runner_sha256'] == PLAN['parent_runner_sha256']
    net, opt = restore_model(sp, ck)
    assert ck['initial_hash'] == r['initial_hash']
    for group, name, rate in zip(opt.param_groups, s.LAYERS, s.RATES if sp['arm'] == 'sgdm' else [.001] * 3):
        assert group['layer'] == name and abs(group['lr'] - .1 * rate * sp['multiplier']) < 1e-14
        assert group['weight_decay'] == 0.0  # Common decay is applied explicitly by the frozen kernel.
        if sp['arm'] == 'sgdm':
            assert group['momentum'] == .9 and group['dampening'] == 0 and not group['nesterov']
        else:
            assert tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8
    nonzero = 0
    for name in s.LAYERS:
        st = opt.state[getattr(net, name).weight]
        moment = st['momentum_buffer'] if sp['arm'] == 'sgdm' else st['exp_avg']
        assert s.frobenius(moment) > 0
        if sp['arm'] != 'sgdm':
            assert int(st['step']) == 20 * math.ceil(55000 / sp['batch_size'])
        nonzero += 1
    _, _, xv, yv, _, _ = s.DATA
    val, logits = c.evaluate(net, xv, yv)
    saved = np.load(d / 'final_validation_logits.npz')
    assert np.array_equal(saved['labels'], yv.numpy())
    max_error = float(np.max(np.abs(logits.astype(float) - saved['logits'].astype(float))))
    tol = PLAN['restoration_tolerances']
    assert max_error <= tol['parent_logit_max_abs']
    assert abs(val['loss'] - r['validation']['loss']) <= tol['parent_validation_loss_abs']
    assert val['errors'] == r['validation']['errors']
    dest = ROOT / 'restoration' / s.run_id(sp)
    (dest / 'spectra').mkdir(parents=True, exist_ok=True)
    spectra = s.spectra(net, dest, 20)
    alpha_error = max(abs(spectra[n]['alpha'] - r['spectra'][n]['alpha'])
                      for n in s.LAYERS if spectra[n]['alpha'] is not None and r['spectra'][n]['alpha'] is not None)
    assert alpha_error <= tol['parent_alpha_max_abs']
    a, b = np.load(d / 'spectra/epoch_020.npz'), np.load(dest / 'spectra/epoch_020.npz')
    spectrum_error = max(float(np.max(np.abs(a[n] - b[n]) / np.maximum(np.abs(a[n]), 1e-30))) for n in s.LAYERS)
    assert spectrum_error <= tol['parent_raw_spectrum_max_relative']
    # Replay two already completed minibatches. Discard this state; never count it as new training.
    replay = s.torch.load(d / 'epoch_001.pt', weights_only=False)
    recorded = np.load(d / 'statistics/epoch_002.npz')
    replay_check = verify_parent_replay(sp, replay, recorded, dest)
    row = {'spec': sp, 'passed': True, 'parent_checkpoint_sha256': file_sha(d / 'final.pt'),
           'parent_model_hash': ck['model_hash'], 'initial_hash': ck['initial_hash'],
           'optimizer_and_rng_exact': True, 'saved_hyperparameters_verified': True, 'nonzero_matrix_moments': nonzero,
           'parent_replay_epoch1_to2_exact': replay_check['cross_runtime_replay_exact'],
           'parent_replay_within_declared_tolerance': True, 'current_runtime_replay_exact': True,
           'maximum_replay_loss_error': max(x['loss_absolute_error'] for x in replay_check['checks']),
           'maximum_replay_statistics_bound_fraction': max(x['maximum_statistics_bound_fraction'] for x in replay_check['checks']),
           'logits_exact': np.array_equal(logits, saved['logits']),
           'max_logit_error': max_error, 'max_alpha_error': alpha_error,
           'max_raw_spectrum_relative_error': spectrum_error}
    s.writej(dest / 'Restoration.json', row)
    return row


def resumed_contract(sp, checkpoint):
    """Independent SGDM buffer and reference Adam, starting from the saved history."""
    net, opt = restore_model(sp, checkpoint)
    ref = copy.deepcopy(net)
    ropt = s.make_opt(ref, {**sp, 'arm': 'adamw'})
    ropt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    momentum = {n: opt.state[getattr(net, n).weight]['exp_avg'].detach().clone() / .1 for n in s.LAYERS}
    x, y = s.DATA[:2]
    order = s.order(sp['seed'], 21)
    factor = schedule(21)
    records = []
    for step in range(5):
        old = {n: p.detach().clone() for n, p in net.named_parameters()}
        ref.load_state_dict(net.state_dict())
        opt.zero_grad(set_to_none=True)
        ix = order[step * sp['batch_size']:(step + 1) * sp['batch_size']]
        s.F.cross_entropy(net(x[ix]), y[ix]).backward()
        for p, q in zip(net.parameters(), ref.parameters()):
            q.grad = p.grad.detach().clone()
        for n in s.LAYERS:
            momentum[n].mul_(.9).add_(getattr(net, n).weight.grad)
        decay = (1 - 1e-5 * factor) ** (len(ix) / 256)
        s.apply_update(net, opt, sp, factor, len(ix), True)
        s.apply_update(ref, ropt, {**sp, 'arm': 'adamw'}, factor, len(ix), True)
        for n, rate in zip(s.LAYERS, s.RATES):
            w = getattr(net, n).weight
            rw = getattr(ref, n).weight
            st = opt.state[w]
            t = float(st['step'])
            native = -.001 * sp['multiplier'] * factor * (st['exp_avg'] / (1 - .9 ** t)) / ((st['exp_avg_sq'] / (1 - .999 ** t)).sqrt() + 1e-8)
            actual = (w - old[n + '.weight'] * decay).double()
            target = rate * sp['multiplier'] * factor * s.frobenius(momentum[n])
            cosine = float((actual * native.double()).sum() / (s.frobenius(actual) * s.frobenius(native)))
            error = abs(float(s.frobenius(actual) / target) - 1)
            moment_error = float(s.frobenius(st['exp_avg'] / .1 - momentum[n]) / s.frobenius(momentum[n]))
            bias_equal = s.torch.equal(getattr(net, n).bias, getattr(ref, n).bias)
            assert error < 2e-3 and cosine > 1 - 2e-4 and moment_error < 3e-5 and bias_equal
            assert s.torch.equal(st['exp_avg'], ropt.state[rw]['exp_avg'])
            assert s.torch.equal(st['exp_avg_sq'], ropt.state[rw]['exp_avg_sq'])
            records.append({'step': step, 'layer': n, 'actual_norm': float(s.frobenius(actual)),
                            'target_norm': float(target), 'relative_norm_error': error,
                            'cosine_with_native_adam': cosine, 'moment_error': moment_error,
                            'bias_exact': bias_equal, 'adam_moments_exact': True})
    return {'passed': True, 'nonzero_checks': len(records), 'restored_nonzero_history': True,
            'records': records, 'rounding_note': 'Actual float32 weight differences incur subtraction rounding at low LR; formula-level norm checks also retained.'}


def train(sp, deadline):
    d = ROOT / 'runs' / s.run_id(sp)
    assert not d.exists(), 'No duplicate continuation or overwrite'
    for name in ('spectra', 'statistics'):
        (d / name).mkdir(parents=True, exist_ok=True)
    parent = STATES / 'runs' / s.run_id(sp)
    ck = s.torch.load(parent / 'final.pt', weights_only=False)
    net, opt = restore_model(sp, ck)
    initial = ck['initial_hash']
    history = copy.deepcopy(ck['history'])
    shutil.copyfile(parent / 'final.pt', d / 'source_epoch_020.pt')
    shutil.copyfile(parent / 'done.json', d / 'parent_done.json')
    s.writej(d / 'history.json', history)
    tic = time.time()
    fit_error = 0.
    x, y, xv, yv, xt, yt = s.DATA
    try:
        for epoch in range(21, 61):
            assert time.time() < deadline, 'Authorized training deadline reached'
            net.train()
            order = s.order(sp['seed'], epoch)
            order_hash = hashlib.sha256(order.tobytes()).hexdigest()
            samples = []
            for j, start in enumerate(range(0, len(y), sp['batch_size'])):
                assert time.time() < deadline, 'Authorized training deadline reached'
                ix = order[start:start + sp['batch_size']]
                record = j < 2 or j % 10 == 0
                loss, stats = s.train_step(net, opt, sp, x[ix], y[ix], schedule(epoch), record)
                if record:
                    samples.append((j, loss, stats))
            np.savez_compressed(d / f'statistics/epoch_{epoch:03}.npz',
                                batch_index=[r[0] for r in samples], loss=[r[1] for r in samples],
                                values=np.array([r[2] for r in samples], np.float32),
                                columns=s.STAT_NAMES, layers=s.LAYERS)
            val, _ = c.evaluate(net, xv, yv)
            row = {'epoch': epoch, 'schedule': schedule(epoch), 'validation': val,
                   'batch_order_sha256': order_hash}
            if epoch % 5 == 0:
                row['train'] = s.evaluate(net, x, y)
                row['spectra'] = s.spectra(net, d, epoch)
                fit_error = max(fit_error, c.fit_audit(row['spectra'], d / f'spectra/epoch_{epoch:03}.npz'))
            history.append(row)
            c.checkpoint(d, 'last.pt', net, opt, sp, epoch, history, initial)
            if epoch in (21, 40):
                c.checkpoint(d, f'epoch_{epoch:03}.pt', net, opt, sp, epoch, history, initial)
            s.writej(d / 'history.json', history)
            if epoch % 10 == 0:
                print(s.run_id(sp), 'epoch', epoch, 'val_loss', val['loss'], 'fixed_endpoint', 60, flush=True)
        assert all(s.torch.isfinite(p).all() for p in net.parameters())
        shutil.copyfile(d / 'last.pt', d / 'final.pt')
        final = s.torch.load(d / 'final.pt', weights_only=False)
        assert final['epoch'] == 60
        net, opt = restore_model(sp, final)
        val, val_logits = c.evaluate(net, xv, yv)
        assert val == history[-1]['validation']
        restored = d / 'restored'
        (restored / 'spectra').mkdir(parents=True, exist_ok=True)
        spectra = s.spectra(net, restored, 60)
        a, b = np.load(d / 'spectra/epoch_060.npz'), np.load(restored / 'spectra/epoch_060.npz')
        assert all(np.array_equal(a[n], b[n]) for n in s.LAYERS)
        assert all(spectra[n]['alpha'] == history[-1]['spectra'][n]['alpha'] for n in s.LAYERS)
        # Exact replay of nonzero optimizer history at the start of the continuation.
        replay = s.torch.load(d / 'epoch_021.pt', weights_only=False)
        net, opt = restore_model(sp, replay)
        order = s.order(sp['seed'], 22)
        saved = np.load(d / 'statistics/epoch_022.npz')
        for j in range(2):
            ix = order[j * sp['batch_size']:(j + 1) * sp['batch_size']]
            loss, stats = s.train_step(net, opt, sp, x[ix], y[ix], schedule(22), True)
            k = int(np.flatnonzero(saved['batch_index'] == j)[0])
            assert loss == float(saved['loss'][k])
            assert np.array_equal(np.array(stats, np.float32), saved['values'][k], equal_nan=True)
        net, opt = restore_model(sp, final)
        test, test_logits = c.evaluate(net, xt, yt)
        assert np.isfinite(test_logits).all() and np.isfinite(val_logits).all()
        np.savez_compressed(d / 'final_test_logits.npz', logits=test_logits, labels=yt.numpy())
        np.savez_compressed(d / 'final_validation_logits.npz', logits=val_logits, labels=yv.numpy())
        status = 'USABLE' if val['accuracy'] >= .8 and sum(v > 0 for v in val['prediction_counts']) > 1 else 'FUNCTIONAL_FAILURE'
        r = {'spec': sp, 'status': status, 'endpoint_epoch': 60, 'start_epoch': 20,
             'stop_reason': 'fixed_epoch60', 'initial_hash': initial,
             'parent_checkpoint_sha256': file_sha(parent / 'final.pt'), 'parent_model_hash': ck['model_hash'],
             'final_model_hash': s.weight_hash(net), 'validation': val, 'test': test,
             'final_train': history[-1]['train'], 'spectra': spectra, 'seconds': time.time() - tic,
             'verification': {'passed': True, 'checkpoint_replay_exact': True, 'replay_origin': 21,
                              'restored_validation_and_spectra_exact': True, 'optimizer_state_restored_exact': True,
                              'fixed_endpoint': True, 'selection_used_test': False,
                              'selection_used_validation': False, 'selection_used_alpha': False,
                              'maximum_independent_alpha_error': fit_error},
             'checkpoint_hashes': {n: file_sha(d / n) for n in ['source_epoch_020.pt', 'epoch_021.pt', 'epoch_040.pt', 'final.pt', 'last.pt']}}
        s.writej(d / 'done.json', r)
        return r
    except Exception:
        s.savept(d / 'interrupted_state.pt', {'model': net.state_dict(), 'optimizer': opt.state_dict(),
                 'torch_rng': s.torch.get_rng_state(), 'spec': sp, 'history': history,
                 'last_fully_completed_epoch': history[-1]['epoch'], 'partial_state_not_an_endpoint': True})
        raise


def guarded(sp, deadline):
    try:
        return train(sp, deadline)
    except Exception as exc:
        failure = {'spec': sp, 'status': 'ERROR', 'error': repr(exc), 'traceback': traceback.format_exc()}
        s.writej(ROOT / 'runs' / s.run_id(sp) / 'failure.json', failure)
        return failure


def main():
    continuation = '--continue' in sys.argv
    if continuation:
        launch = json.loads((ROOT / 'Launch.json').read_text())
        ready = json.loads((ROOT / 'Preflight_Ready.json').read_text())
        assert ready['runner_sha256'] == file_sha(ROOT / 'run.py') and ready['protocol_sha256'] == file_sha(ROOT / 'Protocol.json')
        assert not (ROOT / 'Continuation_Freeze.json').exists(), 'No duplicate continuation'
        tic, deadline = launch['started_epoch_seconds'], launch['deadline_epoch_seconds']
    else:
        assert not (ROOT / 'Launch.json').exists(), 'Do not duplicate this extension'
        tic = time.time()
        deadline = tic + PLAN['maximum_training_seconds']
        s.writej(ROOT / 'Launch.json', {'started_epoch_seconds': tic, 'deadline_epoch_seconds': deadline,
                 'planned': 120, 'resume_epoch': 20, 'fixed_final_epoch': 60, 'workers': 3})
    import platform, subprocess
    runtime = {'python': platform.python_version(), 'platform': platform.platform(),
               'torch': s.torch.__version__, 'numpy': np.__version__, 'torch_configuration': s.torch.__config__.show(),
               'threads': s.torch.get_num_threads(), 'cpu': subprocess.check_output(['lscpu'], text=True)}
    s.writej(ROOT / 'Runtime.json', runtime)
    c.prepare()
    assert file_sha(ROOT / 'data/split.npz') == PLAN['split_sha256']
    c.optimizer_contract()
    central = [x for x in PLAN['specifications'] if x['seed'] == 31001 and x['batch_size'] == 256 and x['multiplier'] == 1]
    assert len(central) == 3
    negative_controls(central)
    with cf.ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('fork')) as pool:
        audits = json.loads((ROOT / 'Parent_Validation.json').read_text())['records'] if continuation else list(pool.map(guarded_parent_audit, PLAN['specifications']))
        passed = len(audits) == 120 and all(x['passed'] for x in audits)
        s.writej(ROOT / 'Parent_Validation.json', {'passed': passed, 'count': len(audits), 'records': audits})
        assert passed, 'Parent verification failed; detailed per-model diagnostics saved; no continuation started'
        graft = next(x for x in central if x['arm'] == 'magnitude_graft')
        ck = s.torch.load(STATES / 'runs' / s.run_id(graft) / 'final.pt', weights_only=False)
        s.writej(ROOT / 'Resumed_Optimizer_Contract.json', resumed_contract(graft, ck))
        if '--preflight-only' in sys.argv:
            s.writej(ROOT / 'Preflight_Ready.json', {'passed': True, 'runner_sha256': file_sha(ROOT / 'run.py'),
                     'protocol_sha256': file_sha(ROOT / 'Protocol.json'), 'all120_verified': True,
                     'parent_validation_sha256': file_sha(ROOT / 'Parent_Validation.json')})
            print('PREFLIGHT PASSED:120 parent states, replay diagnostics and9 failure controls verified.', flush=True)
            return
        if continuation:
            assert ready['parent_validation_sha256'] == file_sha(ROOT / 'Parent_Validation.json')
        s.writej(ROOT / 'Continuation_Freeze.json', {'epoch_seconds': time.time(),
                 'protocol_sha256': file_sha(ROOT / 'Protocol.json'), 'runner_sha256': file_sha(ROOT / 'run.py'),
                 'kernel_sha256': PLAN['kernel_sha256'], 'parent_run_id': PLAN['parent_run_id'],
                 'all_primary_endpoints': 60, 'frozen_before_epoch21': True,
                 'extension_decided_after_parent_results': True})
        results = list(pool.map(guarded, central, [deadline] * 3))
        gate = {'passed': all('test' in r and r['verification']['passed'] for r in results),
                'numerical_gate_only': True, 'alpha_accuracy_loss_not_gates': True,
                'central_arms_count_once': True, 'results': results}
        s.writej(ROOT / 'Continuation_Gate.json', gate)
        if not gate['passed']:
            s.writej(ROOT / 'Completion.json', {'planned': 120, 'completed': sum('test' in r for r in results), 'results': results, 'gate_failed': True})
            raise RuntimeError('Continuation numerical gate failed; expansion stopped')
        remaining = [x for x in PLAN['specifications'] if x not in central]
        futures = {pool.submit(guarded, x, deadline): x for x in remaining}
        while futures:
            ready, _ = cf.wait(futures, timeout=20, return_when=cf.FIRST_COMPLETED)
            for f in ready:
                results.append(f.result())
                futures.pop(f)
            s.writej(ROOT / 'Run_Register.json', {'planned': 120, 'returned': len(results),
                     'completed': sum('test' in r for r in results), 'training_active': bool(futures),
                     'errors': [r for r in results if r['status'] == 'ERROR']})
    completed = sum('test' in r for r in results)
    s.writej(ROOT / 'Completion.json', {'planned': 120, 'completed': completed,
             'results': results, 'seconds': time.time() - tic, 'gpt_on_hold': True})
    assert completed == 120, 'Partial continuation: preserve all available states and report'


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        s.writej(ROOT / 'Failure.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
