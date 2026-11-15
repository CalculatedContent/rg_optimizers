"""Exact checkpoint-local Adam direction, moment, bias and magnitude checks."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import copy
import json
import time
from pathlib import Path
import study as s

OUT = Path(__file__).resolve().parent


def direction_check(spec, x, y, from_epoch=30):
    """Compare the graft with native Adam using identical local state/gradients."""
    d = OUT / 'runs' / s.run_id(spec)
    ck = s.torch.load(d / f'checkpoints/epoch_{from_epoch:03}.pt', weights_only=False)
    net = s.MLP(); net.load_state_dict(ck['model'])
    opt = s.make_opt(net, spec); opt.load_state_dict(ck['optimizer'])
    ref = copy.deepcopy(net)
    ropt = s.make_opt(ref, spec); ropt.load_state_dict(copy.deepcopy(ck['optimizer']))
    old = {name: p.detach().clone() for name, p in net.named_parameters()}
    ix = s.order(spec['seed'], from_epoch + 1)[:spec['batch_size']]
    loss = s.F.cross_entropy(net(x[ix]), y[ix]); loss.backward()
    for p, q in zip(net.parameters(), ref.parameters()):
        q.grad = p.grad.detach().clone()
    factor = s.schedule(from_epoch + 1)
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
    return {'spec': spec, 'from_epoch': from_epoch, 'epoch': from_epoch + 1, 'batch': 0, 'layers': layers,
            'target_history': 'this checkpoint and this run gradient, not the separate SGDM trajectory'}

