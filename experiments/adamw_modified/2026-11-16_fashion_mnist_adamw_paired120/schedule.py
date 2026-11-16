"""Frozen epoch schedule and exact update/exposure accounting; no numerical dependencies."""
import math

def factor(epoch):
    if not 1 <= epoch <= 120:
        raise ValueError(epoch)
    if epoch <= 5:
        return epoch / 5
    if epoch <= 120:
        return .1 + .45 * (1 + math.cos(math.pi * (epoch - 5) / 115))
    raise AssertionError('Unreachable epoch')

def accounting(epoch):
    factors = [factor(e) for e in range(1, epoch + 1)]
    return dict(epoch=epoch, updates=215 * epoch, examples=55000 * epoch,
                lr=.001 * factor(epoch) if epoch else 0.,
                cumulative_nominal_lr=215 * .001 * sum(factors),
                cumulative_squared_nominal_lr=215 * 1e-6 * sum(f*f for f in factors),
                cumulative_decay_log=55000/256*sum(math.log1p(-1e-5*f) for f in factors))
