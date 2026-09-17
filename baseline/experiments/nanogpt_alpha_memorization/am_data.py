"""Fixed synthetic targets and exact, paired presentation schedules."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from collections import Counter
import hashlib
import json
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Record:
    id: str
    cohort: str
    prefix: tuple[int, ...]
    target: tuple[int, ...]
    dose: int = -1
    offset: int = 0


class Dataset:
    def __init__(self, cfg, seed, batch_size):
        self.cfg, self.seed, self.batch_size = cfg, seed, batch_size
        self.steps = int(cfg['steps'])
        self.withdrawal = int(cfg.get('withdrawal_step', self.steps // 2))
        if self.withdrawal < 1 or self.withdrawal > self.steps:
            raise ValueError('withdrawal_step must be within the training horizon.')
        rng = np.random.default_rng(cfg['data_seed'])
        self.train, self.audit, self.canaries = [], [], []
        p = cfg['modulus']
        pairs = rng.permutation(p*p)
        ntrain, nval = len(pairs)//2, len(pairs)//4
        noisy = set(rng.permutation(ntrain)[:round(ntrain*cfg['noise_fraction'])])
        labels = rng.integers(p, size=ntrain)
        for i, index in enumerate(pairs):
            a, b = divmod(int(index), p)
            truth = (16+(a+b)%p,)
            record = Record(f'rule_{a}_{b}', 'train_clean', (1,16+a,2,16+b,3), truth)
            if i < ntrain:
                if i in noisy:
                    record = replace(record, cohort='train_noise', target=(16+int(labels[i]),))
                    self.audit.append(replace(record, cohort='noise_true_label', target=truth))
                self.train.append(record)
            else:
                self.audit.append(replace(record, cohort='validation_clean' if i<ntrain+nval else 'test_clean'))
        self.audit += self.train
        used = set()
        for family, count, length in [('long', cfg['long_per_dose'], cfg['suffix_tokens']),
                                       ('short', cfg['short_per_dose'], cfg['exposure_length'])]:
            for dose in cfg['doses']:
                for index in range(count):
                    prefix = tuple(map(int, rng.integers(16,272,cfg['prefix_tokens'])))
                    alphabet = range(16,272) if family=='long' else cfg['exposure_alphabet']
                    target = tuple(map(int, rng.choice(list(alphabet),size=length)))
                    if prefix in used:
                        raise ValueError('Duplicate canary key; change protocol version, not individual seed.')
                    used.add(prefix)
                    self.canaries.append(Record(f'{family}_{dose}_{index}', family, prefix, target, dose))
        self.audit += self.canaries
        copies = [r for r in self.canaries for _ in range(r.dose)]
        slots = self.withdrawal * batch_size
        if len(copies)>slots:
            raise ValueError('Requested presentations do not fit the acquisition window.')
        schedule = np.random.default_rng(np.random.SeedSequence([cfg['data_seed'],seed,99]))
        where = schedule.choice(slots,len(copies),replace=False)
        self.schedule = dict(zip(map(int,where),copies))
        self.fingerprint = digest({'train':[asdict(r) for r in self.train],
                                  'audit':[asdict(r) for r in self.audit],
                                  'schedule':{str(s):r.id for s,r in self.schedule.items()}})

    def batch(self, step):
        if not 0<=step<self.steps:
            raise ValueError('Update is outside the frozen horizon.')
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg['data_seed'],self.seed,step,7]))
        ordinary = [self.train[int(i)] for i in rng.integers(len(self.train),size=self.batch_size)]
        return [self.schedule.get(step*self.batch_size+i,r) for i,r in enumerate(ordinary)]

    def probes(self, final=False):
        counts, result = Counter(), []
        for r in self.audit:
            if r.cohort=='test_clean' and not final:
                continue
            if r.dose<0 and counts[r.cohort]>=self.cfg['rule_audit_limit']:
                continue
            result.append(r); counts[r.cohort]+=1
        return result

    def planned_counts(self, step):
        return Counter(r.id for slot,r in self.schedule.items() if slot<step*self.batch_size)

    def save(self, root):
        (root/'targets.json').write_text(json.dumps([asdict(r) for r in self.audit],indent=2)+'\n')
        (root/'presentations.json').write_text(json.dumps({str(k):r.id for k,r in self.schedule.items()},sort_keys=True)+'\n')
