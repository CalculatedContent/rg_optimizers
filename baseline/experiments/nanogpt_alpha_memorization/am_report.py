"""Saved-data reports; seed-level uncertainty, no fabricated experiment results."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from am_data import digest


def interval(values):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]; n=len(x)
    if not n: return {'n_seeds':0,'mean':None,'ci95':None}
    return {'n_seeds':n,'mean':float(x.mean()),
            'ci95':float(t.ppf(.975,n-1)*x.std(ddof=1)/math.sqrt(n)) if n>1 else None}


def summarize(frame,keys,metrics):
    rows=[]
    if frame.empty: return pd.DataFrame()
    # First reduce probes within each seed, then calculate across-seed intervals.
    means=frame.groupby(keys+['seed'],dropna=False)[metrics].mean().reset_index()
    for key, group in means.groupby(keys,dropna=False):
        if not isinstance(key,tuple): key=(key,)
        for metric in metrics:
            rows.append(dict(zip(keys,key),metric=metric,**interval(group[metric])))
    return pd.DataFrame(rows)


def auc(members,controls):
    a=np.asarray(members); b=np.asarray(controls)
    if not len(a) or not len(b): return None
    return float(((a[:,None]>b).astype(float)+.5*(a[:,None]==b)).mean())


def collect(root):
    root=Path(root); status=[]; behavior=[]; spectral=[]; exposures=[]; compression=[]; controls=[]; manifests={}; spectral_ids={}
    for path in sorted(root.glob('*/seed_*/manifest.json')):
        manifest=json.loads(path.read_text()); arm=manifest['arm']; seed=manifest['seed']; run=path.parent
        identity=dict(manifest); fingerprint=identity.pop('fingerprint')
        if digest(identity)!=fingerprint: raise ValueError(f'Manifest fingerprint mismatch: {run}')
        manifests[(arm,seed)]=manifest
        completed=(run/'complete.json').exists(); failure=run/'failure.json'
        if completed:
            done=json.loads((run/'complete.json').read_text())
            if done.get('state')!='complete' or done.get('step')!=manifest['protocol']['steps'] or done.get('fingerprint')!=fingerprint:
                raise ValueError(f'Invalid completion record: {run}')
        info={'arm':arm,'seed':seed,'complete':completed,'expected_steps':manifest['protocol']['steps'],
              'state':'complete' if completed else 'failed' if failure.exists() else 'partial_or_running'}
        if failure.exists(): info['error']=json.loads(failure.read_text()).get('error','')
        if completed: info.update({k:v for k,v in json.loads((run/'complete.json').read_text()).items() if k!='state'})
        status.append(info)
        for saved in sorted((run/'behavior').glob('step_*.json')):
            data=json.loads(saved.read_text()); identity={'arm':arm,'seed':seed,'step':data['step'],
                'complete_run':completed,'model_sha256':data['model_sha256']}
            behavior.extend({**r,**identity} for r in data['rows'])
            exposures.extend({**r,**identity} for r in data.get('exposure',[]))
            compression.extend({**r,**identity} for r in data.get('compression',[]))
        for saved in sorted((run/'spectral').glob('step_*.csv')):
            frame=pd.read_csv(saved); frame['arm']=arm; frame['seed']=seed; frame['complete_run']=completed
            identity=frame[['step','model_sha256']].drop_duplicates()
            if len(identity)!=1: raise ValueError(f'Inconsistent spectral identity: {saved}')
            spectral_ids[(arm,seed,int(identity.step.iloc[0]))]=identity.model_sha256.iloc[0]
            spectral.append(frame)
        for item in (r for r in behavior if r['arm']==arm and r['seed']==seed):
            if spectral_ids.get((arm,seed,item['step']))!=item['model_sha256']:
                raise ValueError(f'Behavior/spectrum checkpoint identity mismatch: {run}, step {item["step"]}')
        events=run/'control_events.jsonl'
        if events.exists():
            lines=events.read_text().splitlines()
            for i,line in enumerate(lines):
                try: controls.append(dict(json.loads(line),arm=arm,seed=seed))
                except json.JSONDecodeError:
                    if i!=len(lines)-1: raise
                    break
    return (pd.DataFrame(status),pd.DataFrame(behavior),pd.concat(spectral,ignore_index=True) if spectral else pd.DataFrame(),
            pd.DataFrame(exposures),pd.DataFrame(compression),pd.DataFrame(controls),manifests)


def paired(frame,manifests,keys,metrics):
    if frame.empty: return pd.DataFrame()
    allowed=[]
    for (arm,seed),one in manifests.items():
        if arm!='adamw' or ('muon_guarded',seed) not in manifests: continue
        two=manifests[('muon_guarded',seed)]
        if any(one[k]!=two[k] for k in ('protocol','source_model','runtime','initial_sha256','data_sha256','code_sha256')):
            raise ValueError(f'Incompatible pair for seed {seed}; no pooled comparison written.')
        allowed.append(seed)
    f=frame[frame.seed.isin(allowed)]
    grouped=f.groupby(['arm','seed']+keys,dropna=False)[metrics].mean().reset_index()
    a=grouped[grouped.arm=='adamw'].drop(columns='arm')
    b=grouped[grouped.arm=='muon_guarded'].drop(columns='arm')
    joined=a.merge(b,on=['seed']+keys,suffixes=('_adamw','_muon'))
    out=joined[['seed']+keys].copy()
    for m in metrics: out[m]=joined[m+'_muon']-joined[m+'_adamw']
    return summarize(out,keys,metrics)


def plots(directory,tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=directory/'figures'; dest.mkdir(exist_ok=True)
    stats=tables.get('behavior_summary',pd.DataFrame())
    if not stats.empty:
        for (cohort,metric,dose),group in stats.groupby(['cohort','metric','dose']):
            if metric not in ('exact_match','nll','token_match'): continue
            fig,ax=plt.subplots(figsize=(7,4))
            for arm,data in group.groupby('arm'):
                data=data.sort_values('step')
                ax.plot(data.step,data['mean'],label=arm)
                # Missing uncertainty for one seed stays missing, not a zero band.
                good=data.ci95.notna()
                if good.any():
                    z=data[good]; ax.fill_between(z.step,z['mean']-z.ci95,z['mean']+z.ci95,alpha=.18)
            ax.set(xlabel='Accepted optimizer updates',ylabel=metric,
                   title=f'{cohort}, dose {int(dose)} | pointwise mean and 95% seed CI')
            ax.legend(); fig.tight_layout()
            for ext in ('png','svg'): fig.savefig(dest/f'{cohort}_{int(dose)}_{metric}.{ext}',dpi=140)
            plt.close(fig)
    spectral=tables.get('spectral_summary',pd.DataFrame())
    if not spectral.empty:
        for matrix,group in spectral.groupby('matrix'):
            fig,ax=plt.subplots(figsize=(7,4))
            for (arm,metric),data in group.groupby(['arm','metric']):
                if metric not in ('alpha_clip_xmax','alpha_raw'): continue
                data=data.sort_values('step')
                ax.plot(data.step,data['mean'],label=f'{arm}: {metric}',linestyle='--' if metric=='alpha_raw' else '-')
                good=data.ci95.notna()
                if good.any():
                    z=data[good]; ax.fill_between(z.step,z['mean']-z.ci95,z['mean']+z.ci95,alpha=.18)
            ax.axhline(2,linestyle='--',label='alpha = 2')
            ax.set(xlabel='Accepted optimizer updates',ylabel='alpha_clip_xmax',title=matrix)
            ax.legend(); fig.tight_layout()
            for ext in ('png','svg'): fig.savefig(dest/f'alpha_{matrix}.{ext}',dpi=140)
            plt.close(fig)
    exp=tables.get('exposure_summary',pd.DataFrame())
    if not exp.empty:
        for dose,group in exp[exp.metric=='exposure_lower'].groupby('dose'):
            fig,ax=plt.subplots(figsize=(7,4))
            for arm,data in group.groupby('arm'):
                data=data.sort_values('step'); ax.plot(data.step,data['mean'],label=arm)
                valid=data.ci95.notna(); z=data[valid]
                if len(z): ax.errorbar(z.step,z['mean'],yerr=z.ci95,fmt='none',capsize=3)
            ax.set(xlabel='Accepted updates',ylabel='Conservative exact exposure (bits)',title=f'Finite-universe canaries, dose {dose}')
            ax.legend(); fig.tight_layout()
            for ext in ('png','svg'): fig.savefig(dest/f'exposure_{dose}.{ext}',dpi=140)
            plt.close(fig)

    for name,metric,xcol in [('compression_summary','retrieval_success','step'),
                              ('prefix_summary','exact_match','prefix_tokens'),
                              ('membership_summary','auc_baseline_corrected','step')]:
        table=tables.get(name,pd.DataFrame())
        if table.empty: continue
        table=table[table.metric==metric]
        if name=='prefix_summary': table=table[table.step==table.step.max()]
        for dose,group in table.groupby('dose'):
            fig,ax=plt.subplots(figsize=(7,4))
            group_keys=['arm','cohort'] if 'cohort' in group else ['arm']
            for label,z in group.groupby(group_keys):
                z=z.sort_values(xcol); ax.plot(z[xcol],z['mean'],label=str(label))
                v=z[z.ci95.notna()]
                if len(v): ax.errorbar(v[xcol],v['mean'],yerr=v.ci95,fmt='none',capsize=3)
            ax.set(xlabel=xcol,ylabel=metric,title=f'{name}, dose {dose} | 95% seed CI')
            ax.legend(); fig.tight_layout()
            for ext in ('png','svg'): fig.savefig(dest/f'{name}_{dose}.{ext}',dpi=140)
            plt.close(fig)


def generate(root,make_plots=True):
    root=Path(root); directory=root/'report'; directory.mkdir(exist_ok=True)
    status,b,s,e,c,control,manifests=collect(root)
    if not len(status): raise ValueError('No study manifests found; no results were invented.')
    keys=['arm','step','cohort','dose','prefix_tokens','target_tokens']
    metrics=['nll','perplexity','exact_match','token_match','teacher_accuracy','longest_prefix']
    tables={'runs':status,'behavior':b,'spectral':s,'exposure':e,'compression':c,'control_events':control}
    tables['behavior_summary']=summarize(b,keys,metrics)
    tables['exposure_summary']=summarize(e,['arm','step','dose'],['exposure_lower','exposure_upper'])
    tables['spectral_summary']=summarize(s,['arm','step','matrix'],['alpha_clip_xmax','alpha_raw'])
    if not c.empty:
        c['retrieval_success']=(~c.censored).astype(float)
        tables['compression_summary']=summarize(c,['arm','step','dose'],['retrieval_success','ratio'])
    tables['paired_differences']=paired(b,manifests,keys[1:],metrics)
    if not b.empty:
        final=b[b.complete_run & b.apply(lambda r:r.step==manifests[(r.arm,r.seed)]['protocol']['steps'],axis=1)]
        tables['final_complete_seed_summary']=summarize(final,keys[:1]+keys[2:],metrics)
        tables['final_paired_differences']=paired(final,manifests,keys[2:],metrics)
        onset=[]; contrasts=[]; membership=[]
        for (arm,seed,cohort),frame in b[b.cohort.isin(['long','short'])].groupby(['arm','seed','cohort']):
            initial=frame[frame.step==0].set_index('id').nll
            for (step,dose),g in frame.groupby(['step','dose']):
                zero=frame[(frame.step==step)&(frame.dose==0)]
                if len(zero): contrasts.append({'arm':arm,'seed':seed,'cohort':cohort,'step':step,'dose':dose,
                    'nll_advantage_vs_zero':float(zero.nll.mean()-g.nll.mean()),'exact_advantage_vs_zero':float(g.exact_match.mean()-zero.exact_match.mean())})
                if dose>0 and len(zero) and all(i in initial.index for i in list(g.id)+list(zero.id)):
                    membership.append({'arm':arm,'seed':seed,'cohort':cohort,'step':step,'dose':dose,
                       'auc_nll':auc(-g.nll.values,-zero.nll.values),
                       'auc_baseline_corrected':auc(initial.loc[g.id].values-g.nll.values,initial.loc[zero.id].values-zero.nll.values)})
            for key,g in frame.groupby('id'):
                hit=g[(g.exact_match==1)&(g.presentations>0)].sort_values('step')
                onset.append({'arm':arm,'seed':seed,'cohort':cohort,'id':key,'dose':int(g.dose.iloc[0]),
                              'first_observed_exact_step':int(hit.step.iloc[0]) if len(hit) else None,
                              'presentations_at_first':int(hit.presentations.iloc[0]) if len(hit) else None,
                              'observed_not_continuous_onset':True})
        tables['onsets']=pd.DataFrame(onset); tables['zero_dose_contrasts']=pd.DataFrame(contrasts); tables['membership']=pd.DataFrame(membership)
    regimes=[]
    if not s.empty:
        for (arm,seed),frame in s.groupby(['arm','seed']):
            bad=frame[(frame.alpha_clip_xmax<2)&(frame.step>0)]
            supported=bad[bad.fit_supported.astype(str).str.lower()=='true']
            invalid=~np.isfinite(frame.alpha_clip_xmax)|~np.isfinite(frame.alpha_raw)
            violations=int(((frame.alpha_clip_xmax<=2)|(frame.alpha_raw<=2)|invalid).sum())
            regimes.append({'arm':arm,'seed':seed,'numerical_below2_rows':len(bad),
                            'fit_supported_below2_rows':len(supported),'accepted_gate_violations':violations if arm=='muon_guarded' else None,
                            'note':'Fit support is not a calibrated test of correlations; no unseen steps are certified.'})
    tables['regimes']=pd.DataFrame(regimes)
    prefix_rows=[]
    for path in root.glob('*/seed_*/behavior/step_*.json'):
        run=path.parent.parent; m=json.loads((run/'manifest.json').read_text()); obj=json.loads(path.read_text())
        prefix_rows.extend(dict(r,arm=m['arm'],seed=m['seed'],step=obj['step']) for r in obj.get('prefix_rows',[]))
    tables['prefix_dependence']=pd.DataFrame(prefix_rows)
    tables['prefix_summary']=summarize(tables['prefix_dependence'],['arm','step','dose','prefix_tokens'],['exact_match','token_match','nll'])
    tables['membership_summary']=summarize(tables.get('membership',pd.DataFrame()),['arm','step','cohort','dose'],['auc_nll','auc_baseline_corrected'])
    table_dir=directory/'tables'; table_dir.mkdir(exist_ok=True)
    for name,frame in tables.items(): frame.to_csv(table_dir/f'{name}.csv',index=False)
    complete=int(status.complete.sum())
    text=(f'# Alpha / memorization study\n\nCompleted: **{complete}/10 planned runs**. '
          'Incomplete and failed seeds are retained in runs.csv; available-case intervals are not a completed five-seed result.\n\n'
          'This compares ordinary AdamW against **spectrally guarded Muon**, not ordinary Muon or an isolated alpha intervention. '
          'AdamW below-two entry is measured, never required for inclusion. Failure to enter that regime is a result.\n\n'
          'Error bars are pointwise Student-t 95% intervals across independent training seeds after averaging probes within each seed. '
          'One seed has no estimated interval. Paired effects use compatible seeds at common checkpoints; difference = guarded Muon minus AdamW. '
          'Multiple endpoints are exploratory; these intervals are not simultaneous significance tests.\n\n'
          'All eight trainable matrices are monitored (tied embedding/head counted once). Guard compliance concerns accepted measured checkpoints only. '
          'Raw/clipped alpha and rejected spectra are retained; random-like spectra require separate interpretation. '
          'No claim that the old MPS crash is fixed or that alpha causes overfitting follows from this report.\n\n'
          'Exact recall requires the entire target. Long canaries use 32-token suffixes; formal exposure uses separate 3-token secrets in a 4096-candidate universe. '
          'Prefix compression is a restricted search with known position offset, not adversarial compression. '
          'A smaller NLL gap or lower recall alone does not establish improved clean generalization.\n\n'
          'See tables/runs.csv, regimes.csv, final_complete_seed_summary.csv, final_paired_differences.csv and figures/.\n')
    (directory/'summary.md').write_text(text)
    if make_plots: plots(directory,tables)
    return directory
