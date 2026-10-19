"""Create a prospective plan, without launching or selecting on test accuracy."""
import argparse
from dataclasses import replace
from pathlib import Path
from common import Config, atomic_json, config_dict, digest


def plan():
    base = Config()
    configurations = {}
    def add(c, family):
        values = config_dict(c.validate())
        key = digest(values)[:16]
        if key not in configurations:
            configurations[key] = {'id': key, 'config': values, 'families': []}
        if family not in configurations[key]['families']:
            configurations[key]['families'].append(family)
    # Matched one-factor-at-a-time comparisons around optimizer-specific centers.
    for opt, center_lr in [('adamw', 2e-5), ('adam', 2e-5), ('sgd', 1e-2)]:
        center = replace(base, optimizer=opt, lr=center_lr, weight_decay=.01 if opt == 'adamw' else 0.)
        for seed in (401, 402, 403):
            c = replace(center, seed=seed)
            add(c, opt + '/center')
            for multiplier in (.25, .5, 1, 2, 4):
                add(replace(c, lr=center_lr * multiplier), opt + '/lr')
            for decay in (0., 1e-4, 1e-3, 1e-2, .1):
                add(replace(c, weight_decay=decay), opt + '/decay')
            for field in ('dropout', 'attention_dropout', 'classifier_dropout'):
                for value in (0., .1, .2, .3):
                    add(replace(c, **{field: value}), opt + '/' + field)
            for fraction in (0., .05, .1):
                add(replace(c, warmup_fraction=fraction), opt + '/warmup')
            for schedule in ('constant', 'linear', 'cosine'):
                add(replace(c, schedule=schedule), opt + '/schedule')
            if opt == 'sgd':
                for momentum in (0., .5, .9, .95):
                    add(replace(c, momentum=momentum), 'sgd/momentum')
                add(replace(c, nesterov=True), 'sgd/nesterov')
            else:
                for beta1 in (.8, .9, .95):
                    add(replace(c, beta1=beta1), opt + '/beta1')
                for beta2 in (.95, .99, .999):
                    add(replace(c, beta2=beta2), opt + '/beta2')
    return {'schema': 1, 'purpose': 'Future matched validation-only comparisons; not automatically launched',
            'seeds': [401, 402, 403], 'jobs': list(configurations.values()),
            'job_count': len(configurations), 'initial_only': config_dict(base)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = plan()
    atomic_json(a.output, result)
    print(f'Wrote {result["job_count"]} prospective full-run configurations. No training launched.')
