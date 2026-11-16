"""All-outcome analysis; two seeds remain two replicates, never hundreds of epochs."""
import json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from schedule import accounting
ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'Protocol.json').read_text())
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
PLOTS=ROOT/'plots';PLOTS.mkdir(exist_ok=True)

def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'item'):x=x.item()
    return None if isinstance(x,float) and not np.isfinite(x) else x
def write(p,x):p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def ident(sp):return ('augmented' if sp['augmentation'] else 'unaugmented')+'/seed_'+str(sp['seed'])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def spectral(row):
    sp=row.get('spectra',{});out={}
    for n,r in sp.items():
        a=r['alpha'];out['alpha_'+n]=np.nan if a is None else a
        out['distance_'+n]=abs(out['alpha_'+n]-2)
        out['sub2_'+n]=max(0,2-out['alpha_'+n]) if a is not None else np.nan
        for k in ['supported','D','sigma','xmin','xmax','num_pl_spikes','norm','spectral_norm']:
            out[n+'_'+k]=r.get(k)
        for frac in ['0.2','0.5']:out[n+'_fixed_rank_'+frac]=r['fixed_rank'][frac]['alpha']
        if n=='fc3':out['alpha_fc3_small_n']=r.get('small_n',{}).get('alpha')
    if sp:
        a=np.array([out['alpha_fc1'],out['alpha_fc2']]);d=np.maximum(0,2-a)
        out.update(hidden_mean_alpha=a.mean(),hidden_symmetric_distance=abs(a-2).mean(),
            hidden_sub2_deficit=d.mean(),hidden_rms_sub2_deficit=np.sqrt((d*d).mean()),
            hidden_fraction_below2=(a<2).mean(),hidden_capped_alpha=np.minimum(a,2).mean(),
            hidden_parameter_weighted_alpha=np.average(a,weights=[784*256,256*256]),
            hidden_harmonic_alpha=2/np.sum(1/a) if np.all(a>0) else np.nan)
    return out

def calibration(path):
    z=np.load(path);lo=z['logits'].astype(float);y=z['labels'].astype(int)
    prob=np.exp(lo-lo.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    conf=prob.max(1);correct=prob.argmax(1)==y;b=np.minimum((15*conf).astype(int),14)
    # Stable CE from logits, not clipped probabilities.
    from scipy.special import logsumexp
    ce=float(np.mean(logsumexp(lo,axis=1)-lo[np.arange(len(y)),y]))
    return dict(test_accuracy_pct=100*correct.mean(),test_ce=ce,
        test_ece15=sum(np.mean(b==i)*abs(correct[b==i].mean()-conf[b==i].mean()) for i in range(15) if np.any(b==i)),
        test_brier=np.mean(np.sum((prob-np.eye(10)[y])**2,1)),
        test_correct_ce=float(np.mean(logsumexp(lo[correct],axis=1)-lo[correct,y[correct]])) if correct.any() else np.nan,
        test_wrong_ce=float(np.mean(logsumexp(lo[~correct],axis=1)-lo[~correct,y[~correct]])) if (~correct).any() else np.nan)

def main():
    curves=[];milestones=[];register=[];histories={};results={};observed_native=0
    verify=dict(planned=4,completed=0,independent_seeds=2,fixed_epoch120=True,
        common_initialization=True,common_order=True,equal_lr_and_step_accounting=True,
        exact_replay=True,restored_spectra_and_validation=True,checkpoint_hashes=True,
        native_adamw_statistics=True,every_epoch_records=True,every_minibatch_statistics=True,
        every_epoch_test_metrics=True,missing_runs=[],alpha_not_a_gate=True)
    for sp in PLAN['specifications']:
        key=ident(sp);d=ROOT/'runs'/key;h=read(d/'history.json',[]);r=read(d/'done.json');err=read(d/'failure.json')
        histories[key]=h;results[key]=r
        register.append(dict(**sp,status=r['status'] if r else 'ERROR' if err else 'INCOMPLETE',
            last_epoch=h[-1]['epoch'] if h else None,error=err['error'] if err else None))
        if not r:verify['missing_runs'].append(key)
        verify['every_epoch_records'] &= [z['epoch'] for z in h]==list(range(121)) and all('spectra' in z and 'train_unaugmented' in z for z in h)
        for row in h:
            for k,v in accounting(row['epoch']).items():
                if k in row and (abs(row[k]-v)>1e-10 if isinstance(v,float) else row[k]!=v):verify['equal_lr_and_step_accounting']=False
            rec=dict(seed=sp['seed'],augmentation=sp['augmentation'],epoch=row['epoch'],
                validation_accuracy_pct=100*row['validation']['accuracy'],validation_ce=row['validation']['loss'],
                online_training_loss=row['online_training_loss'],**{k:v for k,v in accounting(row['epoch']).items() if k!='epoch'},**spectral(row))
            if 'train_unaugmented' in row:
                rec.update(train_accuracy_pct=100*row['train_unaugmented']['accuracy'],train_ce=row['train_unaugmented']['loss'])
            curves.append(rec)
        stats_files=list((d/'statistics').glob('*.npz'))
        verify['every_minibatch_statistics'] &= len(stats_files)==120
        for p in stats_files:
            z=np.load(p);vals=z['values'];observed_native+=int(np.count_nonzero(vals[:,:,2]>0))
            verify['every_minibatch_statistics'] &= vals.shape==(215,3,10) and np.array_equal(z['batch_index'],np.arange(215))
            verify['native_adamw_statistics'] &= bool(np.all(vals[:,:,5]==1.) and np.array_equal(vals[:,:,2],vals[:,:,4]))
        if not r:continue
        verify['completed']+=1;verify['fixed_epoch120'] &= r['endpoint_epoch']==120
        verify['exact_replay'] &= r['verification']['passed'] and len(r['verification']['replays'])==3
        verify['restored_spectra_and_validation'] &= r['verification']['restored_validation_and_spectra_exact']
        verify['checkpoint_hashes'] &= all((d/n).exists() and sha(d/n)==v for n,v in r['checkpoint_hashes'].items()) and all(f'epoch_{e:04}.pt' in r['checkpoint_hashes'] for e in range(121))
        verify['every_epoch_test_metrics'] &= [z['epoch'] for z in r['test_milestones']]==list(range(121))
        for m in r['test_milestones']:
            e=m['epoch'];row=next(z for z in h if z['epoch']==e)
            cal=calibration(d/f'test_logits_{e:04}.npz')
            assert abs(cal['test_accuracy_pct']-100*m['test']['accuracy'])<1e-9 and abs(cal['test_ce']-m['test']['loss'])<2e-6
            milestones.append(dict(seed=sp['seed'],augmentation=sp['augmentation'],epoch=e,status=r['status'],
                validation_accuracy_pct=100*row['validation']['accuracy'],validation_ce=row['validation']['loss'],
                **{k:v for k,v in accounting(e).items() if k!='epoch'},**spectral(row),**cal))
    for seed in PLAN['seeds']:
        keys=[ident(dict(seed=seed,augmentation=aug)) for aug in [False,True]]
        a,b=[histories[k] for k in keys];ra,rb=[results[k] for k in keys]
        if ra and rb:verify['common_initialization'] &= ra['initial_hash']==rb['initial_hash']
        else:verify['common_initialization']=False
        if len(a)!=len(b) or not a:verify['common_order']=False
        else:verify['common_order'] &= all(x['batch_order_sha256']==y['batch_order_sha256'] for x,y in zip(a,b))
    pre=read(ROOT/'Preflight.json',{});contract=read(ROOT/'Native_AdamW_Contract.json',{})
    verify['remote_preflight']=pre.get('passed',False);verify['independent_native_contract']=contract.get('passed',False)
    verify['recorded_nonzero_native_matrix_checks']=observed_native
    keys=['fixed_epoch120','common_initialization','common_order','equal_lr_and_step_accounting','exact_replay',
          'restored_spectra_and_validation','checkpoint_hashes','native_adamw_statistics','remote_preflight','independent_native_contract',
          'every_epoch_records','every_minibatch_statistics','every_epoch_test_metrics']
    verify['passed']=verify['completed']==4 and observed_native>0 and all(verify[k] for k in keys)
    write(OUT/'Verification.json',verify);pd.DataFrame(register).to_csv(OUT/'model_register.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'trajectories.csv',index=False);df=pd.DataFrame(milestones);df.to_csv(OUT/'milestones.csv',index=False)
    changes=[];pairs=[];window=[]
    if not df.empty:
        metrics=[c for c in df if c not in ['seed','augmentation','epoch','status'] and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        for (seed,aug),g in df.groupby(['seed','augmentation']):
            base=g[g.epoch==20].iloc[0]
            for _,row in g[g.epoch>20].iterrows():
                change=dict(seed=seed,augmentation=aug,epoch=int(row.epoch))
                for m in metrics:change[m]=row[m]-base[m]
                # Baseline severity weights emphasize large departures at20; no seed selection.
                w=np.abs(np.array([base.alpha_fc1,base.alpha_fc2])-2)
                d=np.abs(np.array([row.alpha_fc1,row.alpha_fc2])-2)
                change['severity_weighted_rescue']=float(np.sum(w*(w-d))/w.sum()) if w.sum()>0 else np.nan
                changes.append(change)
        for (seed,e),g in df.groupby(['seed','epoch']):
            if len(g)!=2:continue
            a=g[g.augmentation].iloc[0];b=g[~g.augmentation].iloc[0]
            pairs.append(dict(seed=seed,epoch=int(e),**{m:a[m]-b[m] for m in metrics}))
        for (seed,aug),g in pd.DataFrame(curves).groupby(['seed','augmentation']):
            # Fixed disjoint windows; descriptive averages, not independent replicates.
            for lo,hi in [(1,20),(21,60),(61,120)]:
                v=g[(g.epoch>=lo)&(g.epoch<=hi)]
                fields=['alpha_fc1','alpha_fc2','hidden_symmetric_distance','hidden_sub2_deficit','validation_ce','validation_accuracy_pct']
                window.append(dict(seed=seed,augmentation=aug,start_epoch=lo,end_epoch=hi,
                    **{f:float(v[f].mean()) for f in fields if f in v}))
    pd.DataFrame(changes).to_csv(OUT/'within_run_changes_from20.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'paired_augmentation_contrasts.csv',index=False)
    pd.DataFrame(window).to_csv(OUT/'epoch_windows_descriptive.csv',index=False)
    c=pd.DataFrame(curves)
    if not c.empty:
        colors={False:'#2867a6',True:'#d07a24'};styles={33001:'-',33002:'--'}
        fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        panels=[('alpha_fc1','FC1 alpha'),('alpha_fc2','FC2 alpha'),('alpha_fc3_small_n','FC3 small-N alpha (exploratory)'),
                ('hidden_symmetric_distance','Mean hidden |alpha - 2|'),('hidden_sub2_deficit','Mean hidden sub-2 deficit'),('lr','Adam learning rate')]
        for ax,(metric,title) in zip(axes.flat,panels):
            for (seed,aug),g in c.groupby(['seed','augmentation']):
                if metric not in g:continue
                z=g.dropna(subset=[metric]);ax.plot(z.epoch,z[metric],color=colors[bool(aug)],ls=styles[int(seed)],label=f'{seed} '+('aug' if aug else 'no aug'))
                if metric in ['alpha_fc1','alpha_fc2']:
                    bad=z[z[metric[6:]+'_supported']==False]
                    ax.scatter(bad.epoch,bad[metric],s=10,facecolors='none',edgecolors=colors[bool(aug)])
            if metric.startswith('alpha'):ax.axhline(2,color='gray',lw=.8)
            ax.axvline(60,color='gray',ls=':',lw=1);ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);fig.savefig(PLOTS/'alpha_and_lr_trajectories.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        panels=[('validation_accuracy_pct','Validation accuracy (%)',c),('validation_ce','Validation CE',c),('train_ce','Unaugmented train CE',c),
                ('test_accuracy_pct','Test accuracy (%) every epoch',df),('test_ce','Test CE every epoch',df),('test_ece15','Test ECE15 every epoch',df)]
        for ax,(metric,title,data) in zip(axes.flat,panels):
            if not data.empty and metric in data:
                for (seed,aug),g in data.groupby(['seed','augmentation']):
                    z=g.dropna(subset=[metric]);ax.plot(z.epoch,z[metric],color=colors[bool(aug)],ls=styles[int(seed)],marker=None)
            ax.axvline(60,color='gray',ls=':',lw=1);ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2)
        fig.savefig(PLOTS/'performance_trajectories.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        for i,layer in enumerate(['fc1','fc2']):
            for j,(suffix,title) in enumerate([('num_pl_spikes','Fitted tail size'),('D','KS distance'),('norm','Spectral trace')]):
                ax=axes[i,j];metric=layer+'_'+suffix
                for (seed,aug),g in c.groupby(['seed','augmentation']):
                    if metric in g:ax.plot(g.epoch,g[metric],color=colors[bool(aug)],ls=styles[int(seed)])
                ax.set(title=layer.upper()+' '+title,xlabel='Epoch');ax.grid(alpha=.2)
        fig.savefig(PLOTS/'fit_support_and_norms.png',dpi=160);plt.close(fig)
    lines=['# Native AdamW paired120 Fashion-MNIST','',f"Completed {verify['completed']}/4 fixed120-epoch trajectories; numerical verification: {'PASS' if verify['passed'] else 'INCOMPLETE OR FAILED'}.",'',
        'Two paired seeds, augmentation off/on, same initialization, order, rate schedule and update counts. Native AdamW only. Historical sample-normalized decoupled decay retained; no graft. Five-epoch warmup to0.001; cosine to0.0001 at120.', '',
        'Every outcome is retained. Missing endpoints are not evidence for convergence or against it. Two seeds are exploratory: show individuals and ranges; epochs/layers are not replicates. Test metrics are computed from prespecified saved checkpoints after each120-epoch trajectory finishes; no checkpoint selection. The same test set was used historically.', '',
        'Alpha rising, alpha crossing2, smaller absolute distance to2, and performance improvements are different outcomes. FC3 has only10correlated eigenvalues; both estimators remain exploratory. These finite trajectories cannot establish asymptotic or RG convergence.', '']
    if not df.empty:
        cols=['seed','augmentation','epoch','alpha_fc1','alpha_fc2','alpha_fc3_small_n','hidden_symmetric_distance','hidden_sub2_deficit','test_accuracy_pct','test_ce']
        lines+=['## Snapshot at epochs0,20,60,120 (all121epochs in CSV)','', '| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
        for _,r in df[df.epoch.isin([0,20,60,120])].sort_values(['seed','augmentation','epoch']).iterrows():
            lines.append('| '+' | '.join(f'{r[k]:.6g}' if isinstance(r[k],(float,np.floating)) else str(r[k]) for k in cols)+' |')
        lines+=['','Detailed paired changes, support, weight norms, calibration, exact schedule exposure, model status and verification are in results/. Raw states and training histories are preserved in the checkpoint artifacts.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verify,indent=2))
    return verify

if __name__=='__main__':
    result=main()
    assert result['passed'],'Incomplete or failed numerical verification; all available outputs retained'
