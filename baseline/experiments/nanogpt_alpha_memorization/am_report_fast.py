"""Behavior-only reports for the no-online-WeightWatcher AdamW/Muon campaign."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from am_data import digest


def interval(values):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]; n=len(x)
    if not n:return {'n_seeds':0,'mean':None,'ci95':None}
    return {'n_seeds':n,'mean':float(x.mean()),'ci95':float(t.ppf(.975,n-1)*x.std(ddof=1)/math.sqrt(n)) if n>1 else None}


def summarize(frame,keys,metrics):
    if frame.empty:return pd.DataFrame()
    rows=[]; means=frame.groupby(keys+['seed'],dropna=False)[metrics].mean().reset_index()
    for key,g in means.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        for metric in metrics:rows.append(dict(zip(keys,key),metric=metric,**interval(g[metric])))
    return pd.DataFrame(rows)


def collect(root):
    status=[]; behavior=[]; exposures=[]; compression=[]; manifests={}
    for path in sorted(Path(root).glob('*/seed_*/manifest.json')):
        m=json.loads(path.read_text()); identity=dict(m); fp=identity.pop('fingerprint')
        if digest(identity)!=fp:raise ValueError(f'Manifest fingerprint mismatch: {path.parent}')
        arm,seed=m['arm'],m['seed']; manifests[(arm,seed)]=m; run=path.parent
        complete=(run/'complete.json').exists(); failure=run/'failure.json'
        row={'arm':arm,'seed':seed,'complete':complete,'state':'complete' if complete else 'failed' if failure.exists() else 'partial_or_running'}
        if failure.exists():row['error']=json.loads(failure.read_text()).get('error','')
        if complete:row.update(json.loads((run/'complete.json').read_text()))
        status.append(row)
        for saved in sorted((run/'behavior').glob('step_*.json')):
            obj=json.loads(saved.read_text()); common={'arm':arm,'seed':seed,'step':obj['step'],'complete_run':complete,'model_sha256':obj['model_sha256']}
            behavior.extend({**r,**common} for r in obj['rows'])
            exposures.extend({**r,**common} for r in obj.get('exposure',[]))
            compression.extend({**r,**common} for r in obj.get('compression',[]))
    return pd.DataFrame(status),pd.DataFrame(behavior),pd.DataFrame(exposures),pd.DataFrame(compression),manifests


def paired(frame,keys,metrics):
    if frame.empty:return pd.DataFrame()
    g=frame.groupby(['arm','seed']+keys,dropna=False)[metrics].mean().reset_index()
    a=g[g.arm=='adamw'].drop(columns='arm'); b=g[g.arm=='muon'].drop(columns='arm')
    j=a.merge(b,on=['seed']+keys,suffixes=('_adamw','_muon'))
    if j.empty:return pd.DataFrame()
    out=j[['seed']+keys].copy()
    for m in metrics:out[m]=j[m+'_muon']-j[m+'_adamw']
    return summarize(out,keys,metrics)


def plots(directory,tables):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    dest=directory/'figures';dest.mkdir(exist_ok=True)
    stats=tables.get('behavior_summary',pd.DataFrame())
    if not stats.empty:
        for (cohort,metric,dose),g in stats.groupby(['cohort','metric','dose']):
            if metric not in ('exact_match','nll','token_match','teacher_accuracy'):continue
            fig,ax=plt.subplots(figsize=(7,4))
            for arm,z in g.groupby('arm'):
                z=z.sort_values('step');ax.plot(z.step,z['mean'],label=arm)
                v=z[z.ci95.notna()]
                if len(v):ax.errorbar(v.step,v['mean'],yerr=v.ci95,fmt='none',capsize=3)
            ax.set(xlabel='Optimizer updates',ylabel=metric,title=f'{cohort}, dose {dose} | mean and 95% seed CI');ax.legend();fig.tight_layout()
            for ext in ('png','svg'):fig.savefig(dest/f'{cohort}_{dose}_{metric}.{ext}',dpi=140)
            plt.close(fig)
    exp=tables.get('exposure_summary',pd.DataFrame())
    if not exp.empty:
        for dose,g in exp[exp.metric=='exposure_lower'].groupby('dose'):
            fig,ax=plt.subplots(figsize=(7,4))
            for arm,z in g.groupby('arm'):
                z=z.sort_values('step');ax.plot(z.step,z['mean'],label=arm)
                v=z[z.ci95.notna()]
                if len(v):ax.errorbar(v.step,v['mean'],yerr=v.ci95,fmt='none',capsize=3)
            ax.set(xlabel='Optimizer updates',ylabel='Conservative exposure (bits)',title=f'Canary exposure, dose {dose}');ax.legend();fig.tight_layout()
            for ext in ('png','svg'):fig.savefig(dest/f'exposure_{dose}.{ext}',dpi=140)
            plt.close(fig)


def generate(root,make_plots=True):
    root=Path(root);directory=root/'report';directory.mkdir(exist_ok=True)
    status,b,e,c,manifests=collect(root)
    if status.empty:raise ValueError('No study manifests found; no results were invented.')
    keys=['arm','step','cohort','dose','prefix_tokens','target_tokens'];metrics=['nll','perplexity','exact_match','token_match','teacher_accuracy','longest_prefix']
    tables={'runs':status,'behavior':b,'exposure':e,'compression':c}
    tables['behavior_summary']=summarize(b,keys,metrics)
    tables['exposure_summary']=summarize(e,['arm','step','dose'],['exposure_lower','exposure_upper'])
    tables['paired_differences']=paired(b,keys[1:],metrics)
    if not b.empty:
        complete_keys={(r.arm,int(r.seed)) for _,r in status[status.complete==True].iterrows()}
        final=b[b.apply(lambda r:(r.arm,int(r.seed)) in complete_keys and r.step==manifests[(r.arm,int(r.seed))]['protocol']['steps'],axis=1)]
        tables['final_complete_seed_summary']=summarize(final,['arm','cohort','dose','prefix_tokens','target_tokens'],metrics)
        tables['final_paired_differences']=paired(final,['cohort','dose','prefix_tokens','target_tokens'],metrics)
    if not c.empty:
        c=c.copy();c['retrieval_success']=(~c.censored).astype(float)
        tables['compression_summary']=summarize(c,['arm','step','dose'],['retrieval_success','ratio'])
    table_dir=directory/'tables';table_dir.mkdir(exist_ok=True)
    for name,frame in tables.items():frame.to_csv(table_dir/f'{name}.csv',index=False)
    if make_plots:plots(directory,tables)
    complete=int(status.complete.sum())
    text=(f'# AdamW / Muon memorization study\n\nCompleted: **{complete}/10 planned runs**.\n\n'
          'Training uses five ordinary AdamW seeds and five ordinary Muon seeds with the pinned source optimizer profiles. '
          '**WeightWatcher is OFF during training and there is no spectral guard.** Checkpoints are retained every 500 updates for post-hoc spectral analysis.\n\n'
          'Behavioral endpoints include exact and partial canary recall, teacher-forced accuracy, NLL/perplexity, finite-universe exposure, prefix dependence, random-label fitting, clean held-out performance, and withdrawal/retention.\n\n'
          'Error bars are pointwise Student-t 95% intervals across independent training seeds after averaging probes within each seed. Paired differences are Muon minus AdamW.\n')
    (directory/'summary.md').write_text(text)
    return directory
