"""Fresh SGDM replay; imports the unchanged frozen training code."""
import os,sys,json,math,shutil,csv,statistics as st
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'2026-11-24_five_seeds'
sys.path.insert(0,str(SOURCE))
import run_additional_seed as launcher
run=launcher.run
sys.path.insert(0,str(SOURCE/'frozen_source/short_test_adamw_graft'))
from report import extract_spectrum
def train(seed):
    launcher.verify()
    run.runtime.DEVICE='cpu'
    run.runtime.initialize()
    data=run.base.load_data(SOURCE/'data/input.txt')
    out=ROOT/'runs'/f'seed_{seed}'
    out.mkdir(parents=True,exist_ok=True)
    design=run.design()
    run.base.write_json(out/'design.json',design)
    h=run.base.sha(out/'design.json')
    folder=out/'sgdm'/f'seed_{seed}'
    for target in (2000,20000):
        run.trial(out,data,'sgdm',seed,run.SCHEDULE,target,h)
        shutil.copyfile(folder/'last.pt',out/f'checkpoint_{target}.pt')
    r=json.loads((folder/'result.json').read_text())
    assert r['completed'] and [x['step'] for x in r['history']]==list(range(0,20001,500))
    for row in r['history']:
        summary,_=extract_spectrum(row['alpha'])
        assert summary['valid_matrix_fits']==18
    run.base.write_json(out/'execution.json',dict(seed=seed,github={k:os.getenv(k) for k in ('GITHUB_RUN_ID','GITHUB_SHA','RUNNER_OS','RUNNER_ARCH','ImageVersion')},runtime=run.runtime.environment()))
    import zipfile
    with zipfile.ZipFile(out/'checkpoints.zip','w',zipfile.ZIP_DEFLATED) as z:
        for t in (2000,20000):
            p=out/f'checkpoint_{t}.pt';z.write(p,p.name)
    with zipfile.ZipFile(out/'checkpoints.zip') as z: assert z.testzip() is None
def analyze():
    rows=[]; layers=[]; groups=defaultdict(list)
    for seed in range(1340,1345):
        r=json.loads((ROOT/'runs'/f'seed_{seed}'/'sgdm'/f'seed_{seed}'/'result.json').read_text())
        assert r['completed'] and len(r['history'])==41
        for q in r['history']:
            s,ls=extract_spectrum(q['alpha']); assert s['valid_matrix_fits']==18
            a=s['confidence_tail_weighted_alpha']; sw=sum(x['confidence_weight'] for x in ls)
            v=sum(x['confidence_weight']*(x['fit_alpha']-a)**2 for x in ls)/sw
            ce=q['validation_loss'];acc=q['validation']['next_character_accuracy']
            row=dict(seed=seed,step=q['step'],alpha=a,layer_variance=v,validation_ce=ce,validation_ppl=math.exp(ce),validation_accuracy=acc)
            rows.append(row);groups[q['step']].append(row)
            layers.extend(dict(seed=seed,step=q['step'],**x) for x in ls)
    summary=[]
    for t,g in sorted(groups.items()):
        assert len(g)==5
        a=st.mean(x['alpha'] for x in g)
        s=dict(step=t,n=5,weighted_layer_seed_sd=math.sqrt(st.mean(x['layer_variance']+(x['alpha']-a)**2 for x in g)))
        for k in ('alpha','validation_ce','validation_ppl','validation_accuracy'):
            v=[x[k] for x in g];m=st.mean(v);sd=st.stdev(v);s.update({k+'_mean':m,k+'_sd':sd,k+'_ci95_half':2.77644511*sd/math.sqrt(5)})
        summary.append(s)
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    for name,data in [('per_seed',rows),('per_matrix',layers),('trajectory',summary)]:
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in data))));w.writeheader();w.writerows(data)
    run.base.write_json(out/'status.json',dict(completed=True,n_seeds=5,n_measurements=len(rows),n_fits=len(layers)))
if __name__=='__main__':
    if sys.argv[1]=='analyze':analyze()
    else:train(int(sys.argv[1]))

