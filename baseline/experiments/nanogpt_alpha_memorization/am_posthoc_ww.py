"""Post-hoc WeightWatcher over saved memorization checkpoints.

Training stays untouched. This module reconstructs each saved CPU model, verifies
checkpoint identity against behavioral audits when available, runs the pinned
WeightWatcher analysis with fix_fingers='clip_xmax', and writes seed-level and
across-seed summaries with 95% intervals.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import t
from am_runtime import baseline,make_model,model_hash
from am_spectral import measure,save_rows

WW_CFG={
    'version':'0.7.7','ERG':True,'randomize':True,'plot':False,'min_evals':20,
    'fix_fingers':'clip_xmax','max_fingers':10,
}


def interval(values):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]; n=len(x)
    if not n: return None,None,0
    mean=float(x.mean())
    ci=float(t.ppf(.975,n-1)*x.std(ddof=1)/math.sqrt(n)) if n>1 else None
    return mean,ci,n


def checkpoints(run_dir):
    return sorted(run_dir.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[1]))


def analyze_checkpoint(path,arm,seed,output,force=False):
    step=int(path.stem.split('_')[1]); target=output/f'step_{step:08d}.csv'
    if target.exists() and not force:
        return pd.read_csv(target)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    manifest=saved['manifest']; cfg=manifest['protocol']; source=baseline(cfg)
    if manifest['arm']!=arm or int(manifest['seed'])!=seed or int(saved['step'])!=step:
        raise ValueError(f'Checkpoint identity mismatch: {path}')
    model=make_model(source,cfg,seed,'cpu'); model.load_state_dict(saved['model']); model.eval()
    state=model_hash(model)
    behavior=path.parent/'behavior'/f'step_{step:08d}.json'
    if behavior.exists():
        expected=json.loads(behavior.read_text())['model_sha256']
        if expected!=state: raise ValueError(f'Behavior/checkpoint hash mismatch: {path}')
    rows=measure(model,WW_CFG,seed,step)
    for row in rows: row.update(arm=arm,seed=seed,checkpoint=str(path))
    save_rows(target,rows)
    print(f'WW {arm} seed={seed} step={step} matrices={len(rows)} fix_fingers=clip_xmax',flush=True)
    return pd.DataFrame(rows)


def summarize(root,frames):
    out=root/'posthoc_weightwatcher'; out.mkdir(exist_ok=True)
    all_rows=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if all_rows.empty: raise ValueError('No post-hoc spectra found.')
    all_rows.to_csv(out/'alpha_all.csv',index=False)
    seed_rows=[]
    for (arm,seed,step),g in all_rows.groupby(['arm','seed','step']):
        clipped=pd.to_numeric(g.alpha_clip_xmax,errors='coerce'); raw=pd.to_numeric(g.alpha_raw,errors='coerce')
        seed_rows.append({'arm':arm,'seed':seed,'step':step,'n_matrices':len(g),
            'min_alpha_clip_xmax':float(clipped.min()),'mean_alpha_clip_xmax':float(clipped.mean()),
            'min_alpha_raw':float(raw.min()),'mean_alpha_raw':float(raw.mean()),
            'n_clip_below_2':int((clipped<2).sum()),'n_raw_below_2':int((raw<2).sum()),
            'n_supported_clip_below_2':int(((clipped<2)&g.fit_supported.astype(bool)).sum())})
    seed=pd.DataFrame(seed_rows); seed.to_csv(out/'alpha_seed_summary.csv',index=False)
    summary=[]
    for (arm,step),g in seed.groupby(['arm','step']):
        for metric in ['min_alpha_clip_xmax','mean_alpha_clip_xmax','min_alpha_raw','mean_alpha_raw','n_clip_below_2','n_raw_below_2']:
            mean,ci,n=interval(g[metric]); summary.append({'arm':arm,'step':step,'metric':metric,'mean':mean,'ci95':ci,'n_seeds':n})
    summary=pd.DataFrame(summary); summary.to_csv(out/'alpha_summary.csv',index=False)
    matrix=[]
    for (arm,step,name),g in all_rows.groupby(['arm','step','matrix']):
        for metric in ['alpha_clip_xmax','alpha_raw']:
            mean,ci,n=interval(pd.to_numeric(g[metric],errors='coerce')); matrix.append({'arm':arm,'step':step,'matrix':name,'metric':metric,'mean':mean,'ci95':ci,'n_seeds':n})
    matrix=pd.DataFrame(matrix); matrix.to_csv(out/'alpha_matrix_summary.csv',index=False)
    regimes=[]
    for (arm,seed),g in seed.groupby(['arm','seed']):
        below=g[g.n_clip_below_2>0]
        regimes.append({'arm':arm,'seed':seed,'ever_clip_below_2':bool(len(below)),
            'first_saved_step_clip_below_2':int(below.step.min()) if len(below) else None,
            'minimum_saved_alpha_clip_xmax':float(g.min_alpha_clip_xmax.min())})
    pd.DataFrame(regimes).to_csv(out/'alpha_regimes.csv',index=False)
    plots(out,summary,matrix,seed)
    return out


def plots(out,summary,matrix,seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=out/'figures'; dest.mkdir(exist_ok=True)
    for metric in ['min_alpha_clip_xmax','mean_alpha_clip_xmax']:
        fig,ax=plt.subplots(figsize=(8,5))
        for arm,g in summary[summary.metric==metric].groupby('arm'):
            g=g.sort_values('step'); ax.plot(g.step,g['mean'],label=arm)
            z=g[g.ci95.notna()]
            if len(z): ax.fill_between(z.step,z['mean']-z.ci95,z['mean']+z.ci95,alpha=.18)
        ax.axhline(2,linestyle='--',label='alpha = 2'); ax.set(xlabel='optimizer updates',ylabel=metric,title=f'{metric}: mean and 95% CI across seeds'); ax.legend(); fig.tight_layout()
        fig.savefig(dest/f'{metric}.png',dpi=160); fig.savefig(dest/f'{metric}.svg'); plt.close(fig)
    for name,g0 in matrix[matrix.metric=='alpha_clip_xmax'].groupby('matrix'):
        fig,ax=plt.subplots(figsize=(8,5))
        for arm,g in g0.groupby('arm'):
            g=g.sort_values('step'); ax.plot(g.step,g['mean'],label=arm)
            z=g[g.ci95.notna()]
            if len(z): ax.fill_between(z.step,z['mean']-z.ci95,z['mean']+z.ci95,alpha=.18)
        ax.axhline(2,linestyle='--',label='alpha = 2'); ax.set(xlabel='optimizer updates',ylabel='alpha_clip_xmax',title=name); ax.legend(); fig.tight_layout()
        safe=name.replace('.','_').replace('/','_'); fig.savefig(dest/f'matrix_{safe}.png',dpi=160); fig.savefig(dest/f'matrix_{safe}.svg'); plt.close(fig)


def run(root,force=False):
    root=Path(root).resolve(); frames=[]; expected=0
    protocol_file=root/'protocol.json'
    if not protocol_file.exists(): raise ValueError('Study protocol.json not found.')
    protocol=json.loads(protocol_file.read_text())
    arms=tuple(protocol.get('arms',()))
    if not arms: raise ValueError('Study protocol declares no arms.')
    for arm in arms:
        for run_dir in sorted((root/arm).glob('seed_*')):
            manifest_file=run_dir/'manifest.json'
            if not manifest_file.exists(): continue
            manifest=json.loads(manifest_file.read_text()); seed=int(manifest['seed'])
            cps=checkpoints(run_dir); expected+=len(cps)
            output=run_dir/'spectral_posthoc'; output.mkdir(exist_ok=True)
            for path in cps:
                frames.append(analyze_checkpoint(path,arm,seed,output,force=force))
    if not expected: raise ValueError('No saved model_*.pt checkpoints found.')
    out=summarize(root,frames)
    print(f'Post-hoc WeightWatcher complete: {expected} checkpoints -> {out}',flush=True)
    return out
