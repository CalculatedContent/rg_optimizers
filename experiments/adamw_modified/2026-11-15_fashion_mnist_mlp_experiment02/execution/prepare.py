"""Validate dataset identity, freeze the 120-run protocol, and run kernel preflight."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import gzip
import hashlib
import json
import platform
import struct
import time
import datetime
from pathlib import Path
import adapter as a

ROOT=a.ROOT; s=a.s


def main():
    assert not (ROOT/'Launch.json').exists(), 'Never change a launched protocol'
    start=ROOT/'Preparation_Start.json'
    if not start.exists():
        with start.open('x') as f:json.dump({'epoch_seconds':time.time(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
    p=a.PLAN
    for f in p['data']['files']:
        path=ROOT/'data'/f['name'];raw=path.read_bytes()
        assert hashlib.md5(raw).hexdigest()==f['md5'], f['name']
        b=gzip.decompress(raw)
        if f['kind']=='images':
            assert struct.unpack('>IIII',b[:16])==(2051,f['count'],28,28)
            assert len(b)==16+f['count']*784
        else:
            assert struct.unpack('>II',b[:8])==(2049,f['count'])
            assert len(b)==8+f['count'] and max(b[8:])<=9
    assert s.sha(ROOT/'data/split.npz')==p['data']['split']['sha256']
    with s.np.load(ROOT/'data/split.npz',allow_pickle=False) as z:
        assert len(z['train'])==55000 and len(z['validation'])==5000
        assert s.np.array_equal(s.np.sort(s.np.concatenate([z['train'],z['validation']])),s.np.arange(60000))
    assert s.sha(ROOT/'study.py')==p['parent_training_source_sha256']
    assert len(p['specifications'])==120
    protocol=dict(p)
    protocol.update(status='frozen_for_execution',dataset='Fashion-MNIST',source_sha256=s.sha(ROOT/'study.py'),
        adapter_sha256=s.sha(ROOT/'adapter.py'),data_hashes={q.name:s.sha(q) for q in sorted((ROOT/'data').iterdir())},
        versions={'python':platform.python_version(),'torch':s.torch.__version__,'numpy':s.np.__version__,'weightwatcher':s.ww.__version__},
        maximum_training_seconds=16200,total_intended_seconds=18000,
        authorization='User requested setup, execution, verification, model-status analysis, plots and check-in on 2026-11-15.',
        model_failure_rule='Nonfinite training is a failure. For completed runs, verify collapse from saved model activations, constant predictions and gradients. Low accuracy or alpha far from2 alone is a review flag, not an automatic collapse label.',
        inference='All 120 planned outcomes retained; five paired seeds. A later stable-only sensitivity subset must be labeled post hoc and exclude settings symmetrically across arms.')
    if (ROOT/'Protocol.json').exists():
        assert json.loads((ROOT/'Protocol.json').read_text())==protocol
    else:
        with (ROOT/'Protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    s.preflight()
    audit=json.loads((ROOT/'Preflight_Audit.json').read_text())
    audit.update(dataset='Fashion-MNIST',data_identity_verified=True,split_hash=s.sha(ROOT/'data/split.npz'),
                 adapter_sha256=s.sha(ROOT/'adapter.py'),actual_versions=protocol['versions'],
                 preflight_is_not_a_confirmatory_run=True)
    s.writej(ROOT/'Preflight_Audit.json',audit)
    print('FASHION-MNIST PREFLIGHT PASSED',flush=True)


if __name__=='__main__':main()
