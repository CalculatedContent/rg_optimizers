"""Fixed-endpoint paired analysis. Seeds, not layers or settings, are replicates."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,sys,math,itertools
from pathlib import Path
sys.path.insert(0,'/tmp/adamw_shape_deps')
import numpy as np,pandas as pd
from scipy import stats,special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import study as s
OUT=Path(__file__).resolve().parent
COLORS={'adamw':'#2d6fb7','magnitude_graft':'#df7934','sgdm':'#398958'}
LABELS={'adamw':'AdamW','magnitude_graft':'Magnitude graft','sgdm':'SGD + momentum'}
METRICS=['alpha_fc1','alpha_fc2','mean_alpha','alpha_variance','mean_abs_alpha_minus_2','rms_alpha_minus_2','fraction_below_2','test_accuracy_pct','test_loss']

def interval(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)];n=len(a)
    mean=float(a.mean()) if n else float('nan');sd=float(a.std(ddof=1)) if n>1 else float('nan')
    half=stats.t.ppf(.975,n-1)*sd/math.sqrt(n) if n>1 else float('nan')
    return {'n':n,'mean':mean,'seed_sd':sd,'ci_low':mean-half,'ci_high':mean+half}

def summarize_row(spec,row):
    a=np.array([row['spectra'][n]['alpha'] for n in ['fc1','fc2']])
    r={**spec,'epoch':row['epoch'],'alpha_fc1':a[0],'alpha_fc2':a[1],
       'alpha_head':row['spectra']['fc3']['alpha'],'mean_alpha':a.mean(),
       'alpha_variance':a.var(ddof=0),'alpha_layer_sd':a.std(ddof=0),
       'mean_abs_alpha_minus_2':abs(a-2).mean(),'rms_alpha_minus_2':np.sqrt(np.mean((a-2)**2)),
       'fraction_below_2':np.mean(a<2),'supported_hidden_layers':sum(row['spectra'][n]['supported'] for n in ['fc1','fc2']),
       'train_loss':row['train']['loss'],'train_accuracy_pct':row['train']['accuracy']*100,
       'validation_loss':row['validation']['loss'],'validation_accuracy_pct':row['validation']['accuracy']*100}
    if 'test' in row:r.update(test_loss=row['test']['loss'],test_accuracy_pct=row['test']['accuracy']*100)
    return r

def main():
    protocol=json.loads((OUT/'Protocol.json').read_text());assert protocol['source_sha256']==s.sha(OUT/'study.py')
    rows=[];trajectories=[];layers=[];problems=[];logit_checks=[];initials={};order_hashes={}
    for spec in s.specifications():
        d=OUT/'runs'/s.run_id(spec);p=d/'done.json'
        if not p.exists():problems.append({'spec':spec,'issue':'missing completion'});continue
        done=json.loads(p.read_text());assert done['spec']==spec and done['status']=='complete' and done['last_epoch']==120
        cfg=json.loads((d/'config.json').read_text());assert cfg['source_sha256']==protocol['source_sha256']
        initials.setdefault(spec['seed'],set()).add(done['initial_hash'])
        history=json.loads((d/'history.json').read_text());assert [r['epoch'] for r in history]==list(range(121))
        for r in history:
            if r['epoch']:
                order_hashes.setdefault((spec['seed'],r['epoch']),set()).add(r['batch_order_sha256'])
            if 'spectra' not in r:continue
            rr=summarize_row(spec,r);trajectories.append(rr)
            if r['epoch']==120:rows.append(rr)
            for n,v in r['spectra'].items():
                layer={**spec,'epoch':r['epoch'],'layer':n,**{k:v[k] for k in ['alpha','D','sigma','xmin','xmax','num_pl_spikes','num_evals','norm','spectral_norm','supported','eligible_hidden','status','warning']}}
                for f in ['0.2','0.5']:layer['fixed_rank_alpha_'+f]=v['fixed_rank'][f]['alpha']
                layers.append(layer)
        z=np.load(d/'test_logits_epoch_120.npz');acc=float(np.mean(z['logits'].argmax(1)==z['labels']))
        ll=float(np.mean(special.logsumexp(z['logits'].astype(float),axis=1)-z['logits'][np.arange(len(z['labels'])),z['labels']]))
        assert abs(acc-done['final']['test']['accuracy'])<1e-14
        err=abs(ll-done['final']['test']['loss']);assert err<2e-6
        logit_checks.append(err)
        for ep in [0,30,60,90,120]:assert (d/f'checkpoints/epoch_{ep:03}.pt').is_file()
        for ep in range(1,121):assert (d/f'statistics/epoch_{ep:03}.npz').is_file()
    assert not problems,problems
    assert all(len(v)==1 for v in initials.values()) and all(len(v)==1 for v in order_hashes.values())
    e=pd.DataFrame(rows);t=pd.DataFrame(trajectories);l=pd.DataFrame(layers)
    e.to_csv(OUT/'Endpoints.csv',index=False);t.to_csv(OUT/'Trajectories.csv',index=False);l.to_csv(OUT/'All_Layer_Measurements.csv',index=False)
    cell=[]
    for (batch,mult,arm),g in e.groupby(['batch_size','multiplier','arm']):
        for metric in METRICS:cell.append({'batch_size':batch,'multiplier':mult,'arm':arm,'metric':metric,**interval(g[metric])})
    cell=pd.DataFrame(cell);cell.to_csv(OUT/'Setting_Summaries.csv',index=False)
    overall=[]
    for arm,g in e.groupby('arm'):
        av=g.groupby('seed')[METRICS].mean()
        for metric in METRICS:overall.append({'arm':arm,'metric':metric,'estimand':'equal average over nine preset settings',**interval(av[metric])})
    overall=pd.DataFrame(overall);overall.to_csv(OUT/'Overall_Summaries.csv',index=False)
    paired=[];rawpairs=[]
    for treatment,reference in [('magnitude_graft','adamw'),('sgdm','adamw'),('magnitude_graft','sgdm')]:
        key=['seed','batch_size','multiplier'];a=e[e.arm==treatment].set_index(key);b=e[e.arm==reference].set_index(key)
        delta=a[METRICS]-b[METRICS];delta=delta.reset_index()
        for _,r in delta.iterrows():rawpairs.append({'treatment':treatment,'reference':reference,**r.to_dict()})
        for (batch,mult),g in delta.groupby(['batch_size','multiplier']):
            for metric in METRICS:paired.append({'treatment':treatment,'reference':reference,'scope':'setting','batch_size':batch,'multiplier':mult,'metric':metric,**interval(g[metric])})
        avg=delta.groupby('seed')[METRICS].mean()
        for metric in METRICS:paired.append({'treatment':treatment,'reference':reference,'scope':'overall','batch_size':None,'multiplier':None,'metric':metric,**interval(avg[metric])})
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'Paired_Summaries.csv',index=False);pd.DataFrame(rawpairs).to_csv(OUT/'Paired_Seed_Differences.csv',index=False)
    corr=[]
    for arm,g in e.groupby('arm'):
        for metric in ['alpha_fc1','alpha_fc2','mean_alpha','alpha_variance','mean_abs_alpha_minus_2']:
            corr.append({'arm':arm,'spectral_metric':metric,'n_settings_times_seeds':len(g),'independent_seeds':g.seed.nunique(),
                         'pearson_r':stats.pearsonr(g[metric],g.test_accuracy_pct).statistic,
                         'spearman_rho':stats.spearmanr(g[metric],g.test_accuracy_pct).statistic,
                         'interpretation':'descriptive within arm; settings share seeds; no independent-run p-value'})
    pd.DataFrame(corr).to_csv(OUT/'Accuracy_Correlations.csv',index=False)
    # Support-sensitive paired comparisons on exactly the same eligible layers.
    support=[]
    final=l[(l.epoch==120)&l.eligible_hidden]
    for treatment,reference in [('magnitude_graft','adamw'),('sgdm','adamw')]:
        a=final[final.arm==treatment].set_index(['seed','batch_size','multiplier','layer']);b=final[final.arm==reference].set_index(['seed','batch_size','multiplier','layer'])
        shared=a.supported&b.supported
        for idx in a.index:
            support.append({'treatment':treatment,'reference':reference,**dict(zip(['seed','batch_size','multiplier','layer'],idx)),
                            'both_supported':bool(shared.loc[idx]),'delta_alpha':a.loc[idx,'alpha']-b.loc[idx,'alpha'],
                            'delta_abs_distance':abs(a.loc[idx,'alpha']-2)-abs(b.loc[idx,'alpha']-2)})
    pd.DataFrame(support).to_csv(OUT/'Common_Support_Pairs.csv',index=False)
    # Numerical identity checks from the recorded updates, across every run/epoch.
    norm_errors=[];cosines=[];min_steps=[]
    for spec in s.specifications():
        if spec['arm']!='magnitude_graft':continue
        d=OUT/'runs'/s.run_id(spec)
        for ep in range(1,121):
            z=np.load(d/f'statistics/epoch_{ep:03}.npz')['values']
            denom=z[:,:,3];mask=denom>1e-20
            norm_errors.extend(np.abs(z[:,:,4][mask]/denom[mask]-1).tolist())
            cosines.extend(z[:,:,7].flatten().tolist());min_steps.extend(z[:,:,8].flatten().tolist())
    assert max(norm_errors)<3e-6
    audit={'passed':True,'completed':len(e),'expected':135,'trajectory_rows':len(t),'layer_rows':len(l),
           'checkpoint_count':135*5,'matched_initializations':True,'matched_per_epoch_example_order':True,
           'max_saved_logit_cross_entropy_difference':max(logit_checks),'all_saved_logit_accuracy_counts_match':True,
           'max_recorded_graft_magnitude_relative_error':max(norm_errors),'recorded_graft_magnitudes_checked':len(norm_errors),
           'minimum_adam_sgdm_direction_cosine':min(cosines),'median_adam_sgdm_direction_cosine':float(np.median(cosines)),
           'final_hidden_supported':int(final.supported.sum()),'final_hidden_total':len(final),
           'next_experiment_started':False,'source_sha256':protocol['source_sha256']}
    s.writej(OUT/'Analysis_Audit.json',audit)
    plots(e,t,paired)
    print(json.dumps(audit,indent=2));print(overall.to_string(index=False));print(paired[paired.scope=='overall'].to_string(index=False))

def style():
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def plots(e,t,p):
    style()
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey=True)
    for i,b in enumerate([128,256,512]):
        for j,m in enumerate([.5,1.,2.]):
            ax=axes[i,j];g=e[(e.batch_size==b)&(e.multiplier==m)]
            ax.axhline(2,color='#888888',lw=1,ls='--')
            for k,arm in enumerate(s.ARMS):
                a=g[g.arm==arm]
                for off,metric,marker in [(-.12,'alpha_fc1','o'),(.12,'alpha_fc2','s')]:
                    y=a[metric];ax.scatter(np.full(len(y),k+off),y,color=COLORS[arm],alpha=.35,s=20)
                    ax.errorbar(k+off,y.mean(),y.std(ddof=1),color=COLORS[arm],fmt=marker,markersize=6,capsize=3)
            ax.set_title(f'Batch {b} | LR × {m:g}');ax.set_xticks(range(3),['AdamW','Graft','SGDM']);ax.grid(axis='y',alpha=.15)
            if j==0:ax.set_ylabel('Layer α')
    fig.suptitle('MNIST: every setting and seed at epoch 120',fontsize=17,y=.99)
    fig.text(.5,.025,'Circles: FC1  •  Squares: FC2  •  Small points: five seeds  •  Bars: seed SD  •  Dashed line: α = 2',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.05,1,.96]);fig.savefig(OUT/'Layer_Alpha_All_Settings.png',dpi=180);plt.close(fig)
    sel=p[(p.scope=='setting')&(p.treatment=='magnitude_graft')&(p.reference=='adamw')]
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,metric,title in zip(axes,['alpha_fc1','mean_abs_alpha_minus_2','test_accuracy_pct'],['FC1 α change','Distance to α = 2 change','Test accuracy change (pp)']):
        grid=sel[sel.metric==metric].pivot(index='batch_size',columns='multiplier',values='mean').loc[[128,256,512],[.5,1.,2.]]
        lim=max(abs(grid.values).max(),1e-6);im=ax.imshow(grid,vmin=-lim,vmax=lim,cmap='coolwarm',aspect='auto')
        for i,b in enumerate(grid.index):
            for j,m in enumerate(grid.columns):
                r=sel[(sel.metric==metric)&(sel.batch_size==b)&(sel.multiplier==m)].iloc[0]
                ax.text(j,i,f'{r["mean"]:+.3f}\n[{r.ci_low:+.3f}, {r.ci_high:+.3f}]',ha='center',va='center',fontsize=10,color='black' if abs(r['mean'])<lim*.7 else 'white')
        ax.set_xticks(range(3),['0.5','1','2']);ax.set_yticks(range(3),['128','256','512']);ax.set_xlabel('Learning-rate multiplier');ax.set_ylabel('Batch size');ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.72)
    fig.suptitle('Magnitude graft minus native AdamW: paired effects',fontsize=16)
    fig.text(.5,.01,'Each cell: mean paired change [95% t interval], five seeds. Negative distance change means closer to 2.',ha='center')
    fig.tight_layout(rect=[0,.045,1,.94]);fig.savefig(OUT/'Paired_Effects_All_Settings.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
    for ax,m in zip(axes,[.5,1.,2.]):
        for arm in s.ARMS:
            for b,mk in [(128,'o'),(256,'s'),(512,'^')]:
                g=e[(e.arm==arm)&(e.multiplier==m)&(e.batch_size==b)]
                ax.scatter(g.mean_alpha,g.test_accuracy_pct,color=COLORS[arm],marker=mk,s=35,alpha=.7,label=LABELS[arm] if b==128 else None)
        ax.axvline(2,ls='--',color='#888888',lw=1);ax.set_title(f'LR multiplier {m:g}');ax.set_xlabel('Mean hidden-layer α');ax.grid(alpha=.15)
    axes[0].set_ylabel('Test accuracy (%)');axes[-1].legend(fontsize=9)
    fig.suptitle('MNIST accuracy versus α at fixed epoch 120',fontsize=16)
    fig.text(.5,.01,'Batch size: circle 128  •  square 256  •  triangle 512. Every point is one run; no selected winners.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(OUT/'Accuracy_vs_Alpha.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,metric,title in zip(axes,['alpha_fc1','alpha_fc2','test_accuracy_pct'],['FC1 α','FC2 α','Test accuracy (%)']):
        for arm in s.ARMS:
            g=t[(t.batch_size==256)&(t.multiplier==1)&(t.arm==arm)].groupby('epoch')[metric].agg(['mean','std']).dropna()
            ax.plot(g.index,g['mean'],color=COLORS[arm],label=LABELS[arm]);ax.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],color=COLORS[arm],alpha=.15)
        if metric.startswith('alpha'):ax.axhline(2,ls='--',lw=1,color='#888888')
        ax.set_xlabel('Epoch');ax.set_ylabel(title);ax.grid(alpha=.15)
    axes[-1].legend(fontsize=9)
    fig.suptitle('Prespecified central setting: batch 256, original layer rates',fontsize=16)
    fig.text(.5,.01,'Lines: five-seed means. Bands: seed SD. Test metrics evaluated at fixed quarterly checkpoints.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(OUT/'Central_Setting_Trajectories.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
