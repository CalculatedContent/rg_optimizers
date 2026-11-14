"""Verify the completed fixed pass, including the documented runtime restoration.

The frozen training and original verification files are never modified. Historical
checkpoints have the separately recorded cross-runtime first-step replay audit;
new checkpoints must pass exact eleven-step replay in the current environment.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import copy
import json
import time
from pathlib import Path
import study as s

OUT = Path(__file__).resolve().parent


def direction_check(spec, x, y):
    """Compare the graft with native Adam using identical local state/gradients."""
    d = OUT / 'runs' / s.run_id(spec)
    ck = s.torch.load(d / 'checkpoints/epoch_030.pt', weights_only=False)
    net = s.MLP(); net.load_state_dict(ck['model'])
    opt = s.make_opt(net, spec); opt.load_state_dict(ck['optimizer'])
    ref = copy.deepcopy(net)
    ropt = s.make_opt(ref, spec); ropt.load_state_dict(copy.deepcopy(ck['optimizer']))
    old = {name: p.detach().clone() for name, p in net.named_parameters()}
    ix = s.order(spec['seed'], 31)[:spec['batch_size']]
    loss = s.F.cross_entropy(net(x[ix]), y[ix]); loss.backward()
    for p, q in zip(net.parameters(), ref.parameters()):
        q.grad = p.grad.detach().clone()
    factor = s.schedule(31)
    decay = (1 - 1e-5 * factor) ** (len(ix) / 256)
    rows = s.apply_update(net, opt, spec, factor, len(ix), True)
    with s.torch.no_grad():
        for group in ropt.param_groups:
            group['lr'] = .001 * spec['multiplier'] * factor
        for q in ref.parameters():
            q.mul_(decay)
        ropt.step()
    layers = []
    for n, rate, group in zip(s.LAYERS, s.RATES, ropt.param_groups):
        w = getattr(net, n).weight; q = getattr(ref, n).weight
        state = opt.state[w]; expected_state = ropt.state[q]
        for field in ('step', 'exp_avg', 'exp_avg_sq'):
            assert s.torch.equal(state[field], expected_state[field]), (spec, n, field)
        assert s.torch.equal(getattr(net, n).bias, getattr(ref, n).bias), (spec, n, 'bias')
        b1, b2 = group['betas']; step = float(expected_state['step'])
        mom = expected_state['exp_avg'] / (1 - b1)
        adam = -group['lr'] * (expected_state['exp_avg'] / (1 - b1 ** step)) / (
            (expected_state['exp_avg_sq'] / (1 - b2 ** step)).sqrt() + group['eps'])
        target = rate * spec['multiplier'] * factor * s.frobenius(mom)
        update = adam * (target / s.frobenius(adam).clamp_min(1e-30))
        expected = old[n + '.weight'] * decay + update
        assert s.torch.equal(w, expected), (spec, n, 'graft reconstruction')
        # Subtraction of the two stored float32 weights adds quantization noise.
        actual = w.double() - (old[n + '.weight'] * decay).double()
        actual_norm = s.frobenius(actual); update_norm = s.frobenius(update)
        adam_norm = s.frobenius(adam)
        direction_defined = bool(update_norm > 0 and adam_norm > 0)
        if float(target) == 0:
            assert float(update_norm) == 0 and float(actual_norm) == 0
            intended_rel = 0.; rel = 0.
        else:
            intended_rel = float(abs(update_norm / target - 1))
            rel = float(abs(actual_norm / target - 1))
        intended_cosine_error = (float(abs((update.double() * adam.double()).sum() /
                                           (update_norm * adam_norm) - 1))
                                 if direction_defined else None)
        cosine_error = (float(abs((actual * adam.double()).sum() / (actual_norm * adam_norm) - 1))
                        if actual_norm > 0 and adam_norm > 0 else None)
        norm_floor_active = bool(adam_norm < 1e-30)
        expected_applied_magnitude = target * adam_norm / adam_norm.clamp_min(1e-30)
        magnitude_abs_error = float(abs(update_norm - expected_applied_magnitude))
        # The frozen updater regularizes the denominator at 1e-30. In collapsed
        # runs this can attenuate subnormal updates; verify that exact rule and
        # report the exception instead of calling it SGDM-magnitude equality.
        quantization_bound = float(s.np.finfo(s.np.float32).smallest_subnormal) * w.numel() ** .5
        assert magnitude_abs_error <= 3e-6 * float(expected_applied_magnitude) + quantization_bound
        if not norm_floor_active:
            assert intended_rel < 3e-6, (spec, n, float(target), float(adam_norm), intended_rel)
        assert intended_cosine_error is None or intended_cosine_error < 3e-6
        layers.append({'layer': n, 'moments_equal_native_adam': True,
                       'bias_equal_native_adam': True, 'matrix_reconstruction_exact': True,
                       'direction_defined_for_nonzero_update': direction_defined,
                       'sgdm_target_magnitude': float(target), 'native_adam_magnitude': float(adam_norm),
                       'frozen_denominator_floor_active': norm_floor_active,
                       'expected_applied_magnitude_including_floor': float(expected_applied_magnitude),
                       'applied_magnitude_rule_absolute_error': magnitude_abs_error,
                       'intended_update_magnitude_relative_error': intended_rel,
                       'intended_update_direction_cosine_error': intended_cosine_error,
                       'actual_delta_magnitude_relative_error': rel,
                       'actual_delta_direction_cosine_error': cosine_error})
    return {'spec': spec, 'from_epoch': 30, 'epoch': 31, 'batch': 0, 'layers': layers,
            'target_history': 'this checkpoint and this run gradient, not the separate SGDM trajectory'}


def main():
    tic = time.time()
    protocol = json.loads((OUT / 'Protocol.json').read_text())
    analysis = json.loads((OUT / 'Analysis_Audit.json').read_text())
    completion = json.loads((OUT / 'Completion.json').read_text())
    restoration = json.loads((OUT / 'Pass_Restoration_Audit.json').read_text())
    assert analysis['passed'] and completion['completed'] == 135 and not completion['failures']
    assert completion['prior_completed_unchanged'] and restoration['passed']
    assert protocol['source_sha256'] == s.sha(OUT / 'study.py')
    assert protocol['seeds'] == [31001, 31002, 31003, 31004, 31005]
    assert protocol['architecture'] == [784, 256, 256, 10]
    old_ids = {s.run_id(r['spec']) for r in restoration['checks']}
    assert len(old_ids) == 14
    x, y, *_ = s.load_data()
    manifest = []; restored = []; replayed = []; graft_checks = []
    initial_hashes = json.loads((OUT / 'Preflight_Audit.json').read_text())['initial_hashes']
    for spec in s.specifications():
        d = OUT / 'runs' / s.run_id(spec)
        done = json.loads((d / 'done.json').read_text())
        for ep in (0, 30, 60, 90, 120):
            cp = d / f'checkpoints/epoch_{ep:03}.pt'
            manifest.append({'path': str(cp.relative_to(OUT)), 'bytes': cp.stat().st_size,
                             'sha256': s.sha(cp), 'spec': spec, 'epoch': ep})
        ck = s.torch.load(d / 'checkpoints/epoch_120.pt', weights_only=False)
        net = s.MLP(); net.load_state_dict(ck['model'])
        assert s.weight_hash(net) == done['final_weight_hash']
        assert ck['config']['initial_hash'] == done['initial_hash'] and ck['epoch'] == 120
        initial = s.torch.load(d / 'checkpoints/epoch_000.pt', weights_only=False)
        initial_net = s.MLP(); initial_net.load_state_dict(initial['model'])
        assert s.weight_hash(initial_net) == initial_hashes[str(spec['seed'])] == done['initial_hash']
        selected = (
            (spec['seed'] in (31001, 31002) and spec['batch_size'] == 256 and spec['multiplier'] == 1)
            or (spec['seed'] == 31005 and spec['batch_size'] == 128 and spec['multiplier'] == 2))
        if selected:
            dest = OUT / 'verification_pass' / s.run_id(spec)
            (dest / 'spectra').mkdir(parents=True, exist_ok=True)
            sp = s.spectra(net, dest, 120)
            err = max(abs(sp[n]['alpha'] - done['final']['spectra'][n]['alpha']) for n in s.LAYERS)
            assert err < 1e-10
            old = s.np.load(d / 'spectra/epoch_120.npz')
            new = s.np.load(dest / 'spectra/epoch_120.npz')
            identical = all(s.np.array_equal(old[n], new[n]) for n in s.LAYERS)
            max_spectrum_error = max(float(s.np.abs(old[n] - new[n]).max()) for n in s.LAYERS)
            historical = s.run_id(spec) in old_ids
            if historical:
                # Pre-final comparison of all 42 restored historical spectra showed
                # at most 2.274e-13 absolute differences from the prior runtime.
                assert all(s.np.allclose(old[n], new[n], rtol=1e-12, atol=1e-12) for n in s.LAYERS)
            else:
                assert identical, (spec, 'raw spectrum mismatch')
            restored.append({'spec': spec, 'max_alpha_difference': err, 'raw_spectra_identical': identical,
                             'max_raw_spectrum_absolute_error': max_spectrum_error,
                             'historical_cross_runtime': historical,
                             'raw_spectrum_tolerance': {'rtol': 1e-12, 'atol': 1e-12} if historical else 'exact'})
            if s.run_id(spec) not in old_ids:
                ck = s.torch.load(d / 'checkpoints/epoch_030.pt', weights_only=False)
                net.load_state_dict(ck['model']); opt = s.make_opt(net, spec)
                opt.load_state_dict(ck['optimizer']); s.torch.set_rng_state(ck['torch_rng'])
                ix = s.order(spec['seed'], 31); z = s.np.load(d / 'statistics/epoch_031.npz')
                for j in range(11):
                    ind = ix[j * spec['batch_size']:(j + 1) * spec['batch_size']]
                    record = j in (0, 1, 10)
                    loss, ss = s.train_step(net, opt, spec, x[ind], y[ind], s.schedule(31), record)
                    if record:
                        k = int(s.np.flatnonzero(z['batch_index'] == j)[0])
                        assert loss == float(z['loss'][k]), (spec, j, 'loss replay')
                        assert s.np.array_equal(s.np.array(ss, dtype=s.np.float32), z['values'][k], equal_nan=True), (spec, j, 'statistics replay')
                replayed.append({'spec': spec, 'from_epoch': 30, 'through_batch': 10,
                                 'three_sampled_losses_and_update_statistics_match_exactly': True})
        if spec['arm'] == 'magnitude_graft':
            graft_checks.append(direction_check(spec, x, y))
    s.writej(OUT / 'Checkpoint_Manifest.json', manifest)
    result = {'passed': True, 'completed_runs': 135, 'all_final_model_hashes_match': True,
              'all_135_initial_checkpoint_hashes_match_preflight_and_paired_runs': True,
              'checkpoints_hashed': len(manifest), 'checkpoint_total_bytes': sum(r['bytes'] for r in manifest),
              'restored_spectrum_checks': restored, 'exact_current_runtime_replay_checks': replayed,
              'historical_runtime_replay_audit': restoration,
              'graft_direction_magnitude_moment_bias_checks': graft_checks,
              'denominator_floor_layer_checks': sum(r['frozen_denominator_floor_active'] for g in graft_checks for r in g['layers']),
              'analysis_audit': analysis, 'source_sha256': protocol['source_sha256'],
              'training_wall_seconds': completion['seconds'], 'verification_seconds': time.time() - tic,
              'verification_policy': 'Historical 14 runs: pretraining first-step tolerance documented in restoration audit; cross-runtime raw spectra use rtol=atol=1e-12 (observed max abs difference 2.274e-13 across 42 historical spectra). Current-runtime selected runs: exact raw spectra and exact eleven-step replay. All 135 initial and final hashes and all 675 checkpoint hashes checked. All 45 grafts checked against native Adam moments and biases with identical gradients; direction is undefined for zero updates. Existing denominator-floor cases are checked against the exact clamped rule and are not labeled SGDM-magnitude equality.',
              'next_experiment_started': False, 'stop_for_user_review': True}
    s.writej(OUT / 'Final_Verification.json', result)
    print(json.dumps({'passed': True, 'checkpoints': len(manifest), 'spectral_restorations': len(restored),
                      'exact_eleven_step_replays': len(replayed), 'graft_checks': len(graft_checks)}, indent=2))


if __name__ == '__main__':
    main()
