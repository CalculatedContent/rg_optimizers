"""Learning rates and update counts; no framework dependencies or training."""
import math

def parent_schedule(epoch):
    if not 0 <= epoch <= 60:
        raise ValueError('Parent schedule is defined only through epoch60')
    if epoch <= 5:
        return epoch / 5
    if epoch <= 20:
        return .1 + .45 * (1 + math.cos(math.pi * (epoch - 5) / 15))
    return .01 + .045 * (1 + math.cos(math.pi * (epoch - 20) / 40))

def schedule(epoch):
    if not 60 <= epoch <= 120:
        raise ValueError('Continuation has a fixed epoch120 boundary')
    return .01

def accounting(spec, epoch):
    if not 0 <= epoch <= 120:
        raise ValueError('Unplanned endpoint')
    n, b, mult = 55000, spec['batch_size'], spec['multiplier']
    q = math.ceil(n / b)
    factors = [parent_schedule(e) if e <= 60 else schedule(e) for e in range(1, epoch+1)]
    f = factors[-1] if factors else 0.
    peaks = [.04, .2, .8] if spec['arm'] == 'sgdm' else [.001] * 3
    sums = q * sum(factors)
    squared = q * sum(x*x for x in factors)
    return {
        'epoch': epoch, 'batch_size': b, 'updates_per_epoch': q,
        'last_batch_examples': n - b*(q-1), 'cumulative_updates': epoch*q,
        'new_updates_since60': max(epoch-60, 0)*q,
        'cumulative_training_examples': epoch*n,
        'lr_factor': f, 'optimizer_layer_lrs': [r*mult*f for r in peaks],
        'sgdm_reference_layer_lrs': [r*mult*f for r in [.04,.2,.8]],
        'sum_optimizer_layer_lrs': [r*mult*sums for r in peaks],
        'sum_squared_optimizer_layer_lrs': [(r*mult)**2*squared for r in peaks],
        'sum_sgdm_reference_layer_lrs': [r*mult*sums for r in [.04,.2,.8]],
        'sum_squared_sgdm_reference_layer_lrs': [(r*mult)**2*squared for r in [.04,.2,.8]],
        'common_explicit_decay_product': math.exp(n/256*sum(math.log1p(-1e-5*x) for x in factors)),
        'augmentation': False,
    }
