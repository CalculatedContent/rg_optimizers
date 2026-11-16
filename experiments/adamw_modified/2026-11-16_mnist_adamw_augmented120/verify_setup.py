"""Frozen design/source/step accounting checks, independent of PyTorch."""
import ast,hashlib,json,math
from pathlib import Path
from schedule import factor,accounting
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
def main():
    p=json.loads((ROOT/'Protocol.json').read_text());m=json.loads((ROOT/'Source_Manifest.json').read_text())
    for path,expected in m['repository_files'].items():
        assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==expected,path
    assert p['planned_runs']==2 and p['seeds']==[33001,33002] and p['fixed_epochs']==120
    assert {(x['seed'],x['augmentation']) for x in p['specifications']}=={(seed,aug) for seed in [33001,33002] for aug in [True]}
    assert all(x['arm']=='adamw' and x['multiplier']==1 and x['batch_size']==256 for x in p['specifications'])
    assert p['dataset']=='MNIST'
    original=json.loads((REPO/'experiments/adamw_modified/2026-11-13_mnist_mlp_experiment01/frozen/Protocol.json').read_text())
    provenance=json.loads((ROOT/'Data_Provenance.json').read_text())
    assert provenance['dataset']=='MNIST'
    assert {r['name']:r['sha256'] for r in provenance['files']}=={k:v for k,v in original['data_hashes'].items() if k!='split.npz'}
    assert provenance['split_sha256']==original['data_hashes']['split.npz']
    assert factor(1)==.2 and factor(5)==1 and factor(120)==.1
    assert p['checkpoints']==list(range(121)) and p['test_epochs']==list(range(121))
    assert accounting(120)['updates']==25800
    assert math.isclose(accounting(120)['cumulative_nominal_lr'],14.147,abs_tol=1e-10)
    try: factor(121)
    except ValueError: pass
    else: raise AssertionError('Epoch cap violated')
    for q in ROOT.glob('*.py'):ast.parse(q.read_text())
    out=dict(passed=True,native_only=True,augmented_specs=2,fixed_horizon=120,
             epoch120=accounting(120),all_source_hashes_verified=True)
    (ROOT/'Setup_Checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
