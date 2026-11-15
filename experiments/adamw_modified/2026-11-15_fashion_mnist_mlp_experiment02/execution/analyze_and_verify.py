"""Verify Fashion-MNIST checkpoints, classify outcomes, analyze and preserve results.

No new experiment or continuation of saved training occurs here. Replayed updates
operate only on restored in-memory copies. --smoke checks one finished run only.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import datetime
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, pearsonr, spearmanr
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import adapter as a
from verify_helpers import direction_check
from diagnostics import diagnose
import preserve

ROOT=a.ROOT;s=a.s
ARMS=s.ARMS
LABELS=dict(zip(ARMS,['AdamW','Magnitude graft','SGD + momentum']))
COLORS=dict(zip(ARMS,['#2867A6','#CD6B27','#35815B']))
METRICS=['alpha_fc1','alpha_fc2','abs_alpha_fc1_minus_2','abs_alpha_fc2_minus_2',
         'alpha_head','abs_alpha_head_minus_2','mean_alpha','alpha_variance','mean_abs_alpha_minus_2',
         'test_accuracy_pct','test_loss','validation_accuracy_pct','validation_loss','train_accuracy_pct']


def endpoint(spec, row):
    values=np.array([row['spectra'][n]['alpha'] if row['spectra'][n]['alpha'] is not None else np.nan for n in s.LAYERS],float)
    logs=np.log10([row['spectra'][n]['spectral_norm'] for n in ('fc1','fc2')])
    result={**spec,'epoch':row['epoch'],'alpha_fc1':values[0],'alpha_fc2':values[1],'alpha_head':values[2],
            'abs_alpha_fc1_minus_2':abs(values[0]-2),'abs_alpha_fc2_minus_2':abs(values[1]-2),
            'abs_alpha_head_minus_2':abs(values[2]-2),
            'mean_alpha':values[:2].mean(),'alpha_variance':values[:2].var(),
            'mean_abs_alpha_minus_2':np.abs(values[:2]-2).mean(),
            'alpha_hat_mean':float(np.mean(values[:2]*logs)),
            'mean_log10_spectral_norm':float(logs.mean()),
            'supported_hidden_layers':sum(row['spectra'][n]['supported'] for n in ('fc1','fc2'))}
    for split in ('train','validation','test'):
        if split in row:result.update({split+'_accuracy_pct':100*row[split]['accuracy'],split+'_loss':row[split]['loss']})
    return result


def verify_run(spec,data,expected_orders):
    x,y,xv,yv,xt,yt=data;d=ROOT/'runs'/s.run_id(spec)
    done=json.loads((d/'done.json').read_text());history=json.loads((d/'history.json').read_text())
    assert done['status']=='complete' and done['last_epoch']==120 and [r['epoch'] for r in history]==list(range(121))
    pre=json.loads((ROOT/'Preflight_Audit.json').read_text())
    for row in history[1:]:assert row['batch_order_sha256']==expected_orders[(spec['seed'],row['epoch'])]
    hashes=[]
    for epoch in (0,30,60,90,120):
        path=d/f'checkpoints/epoch_{epoch:03}.pt'
        hashes.append({'spec':spec,'epoch':epoch,'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':s.sha(path)})
    initial=s.torch.load(d/'checkpoints/epoch_000.pt',weights_only=False)
    net=s.MLP();net.load_state_dict(initial['model'])
    assert s.weight_hash(net)==pre['initial_hashes'][str(spec['seed'])]==done['initial_hash']
    ck=s.torch.load(d/'checkpoints/epoch_120.pt',weights_only=False);net.load_state_dict(ck['model'])
    assert s.weight_hash(net)==done['final_weight_hash']
    assert all(s.torch.isfinite(p).all() for p in net.parameters())
    for g,rate in zip(ck['optimizer']['param_groups'],s.RATES if spec['arm']=='sgdm' else [.001]*3):
        assert math.isclose(g['lr'],rate*spec['multiplier']*s.schedule(120),rel_tol=1e-12)
        assert g['weight_decay']==0
        if spec['arm']=='sgdm':assert g['momentum']==.9 and not g['nesterov'] and g['dampening']==0
        else:
            assert g['betas']==(.9,.999) and g['eps']==1e-8
            for pid in g['params']:assert int(ck['optimizer']['state'][pid]['step'])==120*math.ceil(55000/spec['batch_size'])
    destination=ROOT/'verification_replays'/s.run_id(spec);(destination/'spectra').mkdir(parents=True,exist_ok=True)
    rebuilt=s.spectra(net,destination,120)
    raw=np.load(d/'spectra/epoch_120.npz');new=np.load(destination/'spectra/epoch_120.npz')
    assert all(np.array_equal(raw[n],new[n]) for n in s.LAYERS)
    for n in s.LAYERS:
        old=done['final']['spectra'][n]['alpha'];current=rebuilt[n]['alpha']
        assert (old is None and current is None) or (old is not None and current is not None and abs(old-current)<1e-10)
    logits=np.load(d/'test_logits_epoch_120.npz');assert np.array_equal(logits['labels'],yt.numpy())
    predictions=logits['logits'].argmax(1)
    accuracy=float((predictions==logits['labels']).mean())
    loss=float((logsumexp(logits['logits'].astype(float),axis=1)-logits['logits'][np.arange(len(yt)),logits['labels']]).mean())
    assert abs(accuracy-done['final']['test']['accuracy'])<1e-14
    assert abs(loss-done['final']['test']['loss'])<2e-6
    ck=s.torch.load(d/'checkpoints/epoch_030.pt',weights_only=False)
    net.load_state_dict(ck['model']);opt=s.make_opt(net,spec);opt.load_state_dict(ck['optimizer']);s.torch.set_rng_state(ck['torch_rng'])
    order=s.order(spec['seed'],31);stats=np.load(d/'statistics/epoch_031.npz')
    for j in range(11):
        ix=order[j*spec['batch_size']:(j+1)*spec['batch_size']];record=j in (0,1,10)
        loss0,values=a.ORIGINAL_STEP(net,opt,spec,x[ix],y[ix],s.schedule(31),record)
        if record:
            k=int(np.flatnonzero(stats['batch_index']==j)[0])
            assert loss0==float(stats['loss'][k])
            assert np.array_equal(np.array(values,dtype=np.float32),stats['values'][k],equal_nan=True)
    activity=diagnose(spec,xt,yt);assert activity['max_restored_logit_abs_error']==0
    graft=[];norm_count=0;norm_error=0
    if spec['arm']=='magnitude_graft':
        graft=[direction_check(spec,x,y,from_epoch=ep) for ep in (0,30)]
        assert all(r['direction_defined_for_nonzero_update'] and not r['frozen_denominator_floor_active'] for r in graft[0]['layers'])
        for epoch in range(1,121):
            z=np.load(d/f'statistics/epoch_{epoch:03}.npz')['values']
            keep=z[:,:,3]>1e-20
            if keep.any():norm_error=max(norm_error,float(np.abs(z[:,:,4][keep]/z[:,:,3][keep]-1).max()))
            norm_count+=int(keep.sum())
        assert norm_error<3e-6
    constant=sum(c>0 for c in activity['prediction_counts'])==1
    dead=activity['hidden_test_activity']['fc2']['fraction_test_examples_with_zero_hidden_vector']==1
    status='FAILED' if constant or dead else 'USABLE'
    review=[]
    if accuracy<=float(np.bincount(yt.numpy(),minlength=10).max()/len(yt))+.01:review.append('near_chance_accuracy')
    if not all(done['final']['spectra'][n]['supported'] for n in ('fc1','fc2')):review.append('unsupported_hidden_spectral_fit')
    check={'spec':spec,'initial_and_final_hashes_match':True,'common_minibatches_match':True,'final_optimizer_config_matches':True,
           'raw_spectra_exact':True,'alpha_restoration_passed':True,'test_logits_exact':True,'test_accuracy_exact':True,
           'eleven_step_replay_exact':True,'graft_checks':graft,'graft_nondegenerate_magnitudes_checked':norm_count,
           'max_graft_magnitude_relative_error':norm_error,'model_status':status,'review_flags':review}
    return done,history,hashes,check,activity


def summary(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    n=len(values);mean=float(values.mean()) if n else np.nan
    sd=float(values.std(ddof=1)) if n>1 else np.nan
    half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else np.nan
    return {'n':n,'mean':mean,'seed_sd':sd,'ci_low':mean-half,'ci_high':mean+half}


def statistics(e,complete,output):
    setting=[]
    for (batch,mult,arm),group in e.groupby(['batch_size','multiplier','arm']):
        for metric in METRICS:setting.append({'batch_size':batch,'multiplier':mult,'arm':arm,'metric':metric,**summary(group[metric])})
    setting=pd.DataFrame(setting);setting.to_csv(output/'setting_summaries.csv',index=False)
    effects=[];overall=[]
    if complete:
        # Missing fits must not silently become an eight-setting estimate.
        avg=e.groupby(['seed','arm'])[METRICS].agg(lambda z: z.mean(skipna=False))
        for arm in ARMS:
            for metric in METRICS:overall.append({'arm':arm,'metric':metric,**summary(avg.xs(arm,level='arm')[metric])})
        scopes=[('all_settings',None,None,e)]+[('setting',b,m,g) for (b,m),g in e.groupby(['batch_size','multiplier'])]
        for scope,b,m,frame in scopes:
            means=frame.groupby(['seed','arm'])[METRICS].agg(lambda z: z.mean(skipna=False))
            for treatment,reference in [('magnitude_graft','adamw'),('sgdm','adamw'),('magnitude_graft','sgdm')]:
                delta=means.xs(treatment,level='arm')-means.xs(reference,level='arm')
                for metric in METRICS:effects.append({'scope':scope,'batch_size':b,'multiplier':m,'treatment':treatment,'reference':reference,'metric':metric,**summary(delta[metric])})
    effects=pd.DataFrame(effects);effects.to_csv(output/'paired_effects.csv',index=False)
    overall=pd.DataFrame(overall);overall.to_csv(output/'overall_summaries.csv',index=False)
    corr=[]
    for arm,g in e.groupby('arm'):
        for feature in ['mean_alpha','alpha_variance','mean_abs_alpha_minus_2','mean_log10_spectral_norm','alpha_hat_mean']:
            z=g[[feature,'test_accuracy_pct']].dropna()
            if len(z)>2 and z[feature].std()>0 and z.test_accuracy_pct.std()>0:
                corr.append({'arm':arm,'feature':feature,'n_rows':len(z),'independent_seeds':g.seed.nunique(),
                  'pearson_r':pearsonr(z[feature],z.test_accuracy_pct).statistic,'spearman_rho':spearmanr(z[feature],z.test_accuracy_pct).statistic,
                  'interpretation':'Exploratory pooled setting/seed association; rows are not independent seeds; no causal claim or iid p-value.'})
    pd.DataFrame(corr).to_csv(output/'accuracy_correlations.csv',index=False)
    return setting,overall,effects


def status_plot(status,destination):
    """Display every preset cell, including failed or unfinished training."""
    settings=sorted((r['batch_size'],m) for r in a.PLAN['included_settings'] for m in r['lr_multipliers'])
    scores=np.zeros((len(settings),3));labels={}
    records=[]
    for i,(batch,multiplier) in enumerate(settings):
        for j,arm in enumerate(ARMS):
            z=status[(status.batch_size==batch)&(status.multiplier==multiplier)&(status.arm==arm)]
            usable=int(z.model_status.eq('USABLE').sum())
            failed=int(z.model_status.isin(['FAILED','FAILED_TRAINING']).sum())
            unresolved=5-usable-failed
            accuracy=z.test_accuracy_pct.mean() if 'test_accuracy_pct' in z else np.nan
            scores[i,j]=usable/5
            labels[i,j]=f'{usable}/5 usable'+(f' · {failed} failed' if failed else '')+(f'\n{unresolved} unresolved' if unresolved else '')+(f'\n{accuracy:.2f}% accuracy' if np.isfinite(accuracy) else '')
            records.append({'batch_size':batch,'multiplier':multiplier,'arm':arm,'planned':5,
                            'usable':usable,'failed':failed,'unresolved':unresolved,
                            'mean_completed_test_accuracy_pct':accuracy})
    fig,ax=plt.subplots(figsize=(10,8))
    ax.imshow(scores,cmap='YlGn',vmin=0,vmax=1,aspect='auto',alpha=.55)
    for (i,j),label in labels.items():ax.text(j,i,label,ha='center',va='center',fontsize=10)
    ax.set_xticks(range(3),[LABELS[x] for x in ARMS]);ax.set_yticks(range(len(settings)),[f'Batch {b} / LR ×{m:g}' for b,m in settings])
    ax.set_title('Functional model status across all planned settings',pad=16,fontsize=15)
    ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,len(settings),1),minor=True)
    ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',bottom=False,left=False)
    fig.text(.5,.025,'Usable = finite classifier, nonconstant predictions, no complete FC2 collapse. α does not determine this label.\nAccuracy includes every completed model, including failures. No failed outcome is removed.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,1]);fig.savefig(destination/'Model_Status.png',dpi=170);plt.close(fig)
    pd.DataFrame(records).to_csv(destination.parent/'results/quality_counts_by_setting.csv',index=False)


def plots(e,trajectories,effects,destination,complete):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    order=sorted(set(zip(e.batch_size,e.multiplier)))
    fig,axs=plt.subplots(2,2,figsize=(12,8.3))
    for ax,metric,title in zip(axs.flat,['alpha_fc1','alpha_fc2','mean_abs_alpha_minus_2','test_accuracy_pct'],['FC1 α','FC2 α','Mean hidden |α − 2|','Test accuracy (%)']):
        for j,arm in enumerate(ARMS):
            for i,(b,m) in enumerate(order):
                vals=e[(e.arm==arm)&(e.batch_size==b)&(e.multiplier==m)][metric].dropna().to_numpy()
                if not len(vals):continue
                x=i+(j-1)*.23
                ax.scatter(x+np.linspace(-.055,.055,len(vals)),vals,s=16,color=COLORS[arm],alpha=.55)
                ax.errorbar(x,vals.mean(),yerr=vals.std(ddof=1) if len(vals)>1 else 0,fmt='o',capsize=3,color=COLORS[arm],label=LABELS[arm] if i==0 else None)
        if metric.startswith('alpha'):ax.axhline(2,color='gray',ls='--',lw=1)
        if metric=='test_accuracy_pct' and (e.model_status=='FAILED').any():ax.set_ylim(0,100)
        ax.set_xticks(range(len(order)),[f'{b}\n×{m:g}' for b,m in order]);ax.set_xlabel('Batch size / LR multiplier');ax.set_title(title);ax.grid(axis='y',alpha=.15)
    axs[0,0].legend(fontsize=9)
    fig.suptitle('Fashion-MNIST: all planned settings, including failures',fontsize=16)
    fig.text(.5,.015,'Points = seeds; error bars = seed SD. FC3 is exploratory and excluded from hidden-layer averages.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(destination/'All_Settings_Alpha_Accuracy.png',dpi=170);plt.close(fig)
    central=trajectories[(trajectories.batch_size==256)&(trajectories.multiplier==1)&(trajectories.epoch>=10)]
    fig,axs=plt.subplots(1,3,figsize=(12.5,4.6))
    for ax,metric,title in zip(axs,['alpha_fc1','alpha_fc2','validation_accuracy_pct'],['FC1 α','FC2 α','Validation accuracy (%)']):
        for arm,g in central.groupby('arm'):
            mean=g.groupby('epoch')[metric].mean();sd=g.groupby('epoch')[metric].std()
            ax.plot(mean.index,mean,color=COLORS[arm],label=LABELS[arm]);ax.fill_between(mean.index,mean-sd,mean+sd,color=COLORS[arm],alpha=.14)
        if metric.startswith('alpha'):ax.axhline(2,color='gray',ls='--',lw=1)
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.15)
    axs[-1].legend(fontsize=9);fig.suptitle('Central setting: batch 256 / LR ×1',fontsize=16)
    fig.text(.5,.02,'Mean ± seed SD; display starts at epoch 10. Initialization remains in the saved trajectories.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(destination/'Central_Trajectories.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.7))
    for arm,g in e.groupby('arm'):
        for ax,feature,label in zip(axs,['mean_alpha','mean_abs_alpha_minus_2'],['Mean hidden α','Mean hidden |α − 2|']):
            usable=g[g.model_status=='USABLE'];failed=g[g.model_status!='USABLE']
            ax.scatter(usable[feature],usable.test_accuracy_pct,label=LABELS[arm],color=COLORS[arm],alpha=.65)
            if len(failed):ax.scatter(failed[feature],failed.test_accuracy_pct,color=COLORS[arm],marker='x',s=80,label=LABELS[arm]+' failed')
            ax.set_xlabel(label);ax.set_ylabel('Test accuracy (%)');ax.grid(alpha=.15)
    axs[0].axvline(2,color='gray',ls='--',lw=1);axs[-1].legend(fontsize=8)
    fig.suptitle('Fashion-MNIST: spectral alignment and accuracy',fontsize=15)
    fig.text(.5,.02,'All completed models retained. Multiple settings within a seed are not independent replicates.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(destination/'Accuracy_vs_Alpha.png',dpi=170);plt.close(fig)
    if complete and len(effects):
        groups=[('setting','magnitude_graft'),('setting','sgdm'),('all_settings','magnitude_graft'),('all_settings','sgdm')]
        labels=['Central — graft','Central — SGDM','All eight — graft','All eight — SGDM']
        fig,axs=plt.subplots(1,2,figsize=(12,5.2))
        for ax,metric,title in zip(axs,['mean_abs_alpha_minus_2','test_accuracy_pct'],['Δ mean hidden |α − 2|','Δ test accuracy (percentage points)']):
            for i,(scope,arm) in enumerate(groups):
                z=effects[(effects.scope==scope)&(effects.treatment==arm)&(effects.reference=='adamw')&(effects.metric==metric)]
                if scope=='setting':z=z[(z.batch_size==256)&(z.multiplier==1)]
                r=z.iloc[0];ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o',capsize=4,color=COLORS[arm])
            ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(4),labels if ax is axs[0] else ['']*4);ax.set_ylim(3.5,-.5);ax.set_title(title);ax.grid(axis='x',alpha=.15)
        axs[0].set_xlabel('Negative = closer to 2');axs[1].set_xlabel('Positive = higher accuracy')
        fig.suptitle('Paired effects relative to AdamW',fontsize=16)
        fig.text(.5,.02,'95% paired t intervals over five seeds. All-eight effects average settings within each seed first; no failures removed.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(destination/'Paired_Effects.png',dpi=170);plt.close(fig)


def write_report(e,overall,effects,summary_data,pub):
    complete=summary_data['completed_runs']==120
    lines=['# Experiment 02: Fashion-MNIST results','',
       f"**{summary_data['completed_runs']}/120 runs completed and checked. {summary_data['usable_models']} usable classifiers; {summary_data['failed_models']} failed completed models.**",'',
       'ReLU MLP 784→256→256→10, 120 epochs, five paired seeds, eight settings, native AdamW / magnitude-only graft / SGDM. The graft retains Adam direction, moments and bias updates and changes only matrix-update magnitude using its own gradient history.','',
       '## Central setting: batch 256 / LR ×1','',
       '| Optimizer | FC1 α | FC2 α | Mean hidden \\|α−2\\| | Test accuracy | Usable / completed |','|---|---:|---:|---:|---:|---:|']
    central=e[(e.batch_size==256)&(e.multiplier==1)]
    for arm in ARMS:
        g=central[central.arm==arm]
        if len(g):lines.append(f"| {LABELS[arm]} | {g.alpha_fc1.mean():.3f} | {g.alpha_fc2.mean():.3f} | {g.mean_abs_alpha_minus_2.mean():.3f} | {g.test_accuracy_pct.mean():.3f}% | {int(g.model_status.eq('USABLE').sum())}/{len(g)} |")
    if complete:
        lines+=['','## Paired graft effects versus AdamW','','| Scope | ΔFC1 α | ΔFC2 α | Δmean \\|α−2\\| | Δaccuracy (pp) |','|---|---:|---:|---:|---:|']
        for scope,label in [('setting','Central setting'),('all_settings','All eight settings')]:
            q=effects[(effects.scope==scope)&(effects.treatment=='magnitude_graft')&(effects.reference=='adamw')]
            if scope=='setting':q=q[(q.batch_size==256)&(q.multiplier==1)]
            q=q.set_index('metric');cells=[]
            for m in ['alpha_fc1','alpha_fc2','mean_abs_alpha_minus_2','test_accuracy_pct']:
                r=q.loc[m];cells.append(f"{r['mean']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]")
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        lines+=['','Intervals are descriptive paired 95% Student t intervals over five seeds. Across settings, differences are averaged within each seed first. Settings are not independent seeds. Positive Δα is upward movement; negative Δ|α−2| is closer to2; positive Δaccuracy is higher accuracy. These outcomes are distinct. No planned failures are excluded.']
    else:lines+=['','This pass is incomplete. Do not interpret available runs as the completed five-seed experiment. Overall paired conclusions are withheld.']
    lines+=['','![Model status](plots/Model_Status.png)','','![All settings](plots/All_Settings_Alpha_Accuracy.png)','','![Central trajectories](plots/Central_Trajectories.png)','',
            '![Accuracy and alpha](plots/Accuracy_vs_Alpha.png)','']
    if complete:lines+=['![Paired effects](plots/Paired_Effects.png)','']
    lines+=['## Model status','','USABLE means a finite trained classifier with nonconstant predictions and no complete FC2 collapse. FAILED marks constant predictions or complete FC2 collapse established from the restored model. Nonfinite or interrupted training is separately recorded. Accuracy or α alone does not define a successful theoretical test. Fit-support flags and low-accuracy review flags remain visible.','',
        '[Every model](results/model_status.csv) · [Per-setting statistics](results/setting_summaries.csv) · [Paired effects](results/paired_effects.csv) · [Layer measurements](results/layer_measurements.csv) · [Trajectories](results/trajectories.csv) · [Verification](verification/Final_Verification.json) · [Saved checkpoints](ARTIFACTS.md)','',
        '## Verification and scope','',
        f"Checked {summary_data['checkpoints_hashed']} checkpoint hashes, every completed run's initial/final model hashes, actual optimizer settings, all minibatch-order hashes, raw restored spectra and α, restored test logits, and eleven-step checkpoint replay. Graft direction/moments/bias updates are checked both before and after training; nonzero target magnitudes and denominator-floor exceptions are explicit.",'',
        'WeightWatcher 0.7.7 uses default tail selection. Primary means use FC1 and FC2. FC3 has only ten eigenvalues and is exploratory. All reported endpoints are the fixed epoch-120 checkpoint, not the best test checkpoint. Validation deterioration and unsupported fits are retained in the model register. Intervals are descriptive and unadjusted for multiple comparisons. The batch128/LR×2 setting was excluded prospectively using the earlier MNIST result; no new setting was added. The original MNIST experiment remains untouched. No follow-up experiment is running.','']
    (pub/'README.md').write_text('\n'.join(lines));(pub/'RESULTS.md').write_text('\n'.join(lines))


def finalize(verification,e,setting,overall,effects,activities,pub):
    completion=json.loads((ROOT/'Completion.json').read_text())
    archive_entries=[]
    source=json.loads((ROOT/'Source_Save.json').read_text());archive_entries.append(('Source and Fashion-MNIST data',source))
    for seed in a.PLAN['seeds']:
        path=ROOT/f'Seed_{seed}_Saved.json'
        if path.exists():
            value=json.loads(path.read_text())
            if 'receipt' in value:archive_entries.append((f'Seed {seed}: 24 runs / 120 checkpoints',value))
    saved=len(archive_entries)==6
    result={'experiment':'Experiment02','dataset':'Fashion-MNIST','architecture':[784,256,256,10],
        'status':'complete_and_verified' if len(e)==120 else 'incomplete','planned_runs':120,'completed_runs':len(e),
        'usable_models':int(e.model_status.eq('USABLE').sum()),'failed_models':int(e.model_status.eq('FAILED').sum()),
        'unfinished_or_failed_training_runs':[r for r in completion['results'] if r['status']!='complete'],
        'seeds':a.PLAN['seeds'],'settings':a.PLAN['included_settings'],'source_sha256':s.sha(ROOT/'study.py'),
        'adapter_sha256':s.sha(ROOT/'adapter.py'),'checkpoints_hashed':len(verification['checkpoint_manifest']),
        'all_verification_checks_passed':True,'all_completed_seed_archives_saved':saved,'training_active':False,
        'next_experiment_started':False,'training_seconds':completion['training_seconds'],
        'overall_summaries':overall.to_dict('records'),'paired_effects':effects.to_dict('records'),
        'artifact_archives':[{'label':label,'sha256':item['sha256'],**item['receipt']['results'][0]} for label,item in archive_entries]}
    s.writej(ROOT/'Experiment02_Result_Summary.json',result)
    s.writej(pub/'results/Experiment02_Result_Summary.json',result)
    s.writej(pub/'results/STATUS.json',result)
    write_report(e,overall,effects,result,pub)
    artifact_lines=['# Saved Fashion-MNIST artifacts','']
    for label,item in archive_entries:
        r=item['receipt']['results'][0];url=f"https://chatgpt.com/library?file_id={r['library_file_id']}&account_id=personal"
        artifact_lines += [f"- [{label}]({url}) — SHA256 `{item['sha256']}`."]
    artifact_lines+=['','Archives are owner-access-controlled. Each seed archive includes data, frozen source and all states/logs for that seed. Restore all seed archives plus the results/verification bundle to one empty folder; overlapping immutable source/data files must have identical hashes. Do not rerun training to reproduce the report. Use `python analyze_and_verify.py --analyze-only` to verify and regenerate local tables without performing a second artifact save.','']
    (pub/'ARTIFACTS.md').write_text('\n'.join(artifact_lines))
    (pub/'MODEL_STATUS.md').write_text('# Model status\n\n'+f"{result['usable_models']} usable; {result['failed_models']} failed completed models. {120-len(e)} unfinished/nonfinite runs are listed separately.\n\n[Full model register](results/model_status.csv) · [Summary and plots](README.md) · [Restored-model diagnostics](verification/Model_Diagnostics.json)\n")
    for name in ['study.py','adapter.py','prepare.py','supervisor.py','preserve.py','analyze_and_verify.py','verify_helpers.py','diagnostics.py']:
        shutil.copyfile(ROOT/name,pub/'execution'/name)
    for name in ['Protocol.json','Preflight_Audit.json','Execution_Source_Manifest.json','Data_Provenance.json','Launch.json','Completion.json','Preparation_Start.json','Detached_Launch_Attempt.json','Execution_Metadata_Clarification.json']:
        if (ROOT/name).exists():shutil.copyfile(ROOT/name,pub/'verification'/name)
    if ARGS.analyze_only:return result
    if (ROOT/'Final_Save.json').exists():raise RuntimeError('Final save already exists; resolve identity before re-saving')
    for folder in ('data','plots','checkpoints'):(pub/folder).mkdir(exist_ok=True)
    (pub/'data/README.md').write_text('# Verified Fashion-MNIST data\n\nOfficial compressed-file checksums and IDX dimensions passed. The original split indices were reused byte-for-byte. Data and provenance are included in the source and seed archives. [Data hashes](../verification/Data_Provenance.json) · [Artifacts](../ARTIFACTS.md)\n')
    (pub/'plots/README.md').write_text('# Experiment 02 plots\n\n[Model status](Model_Status.png) · [All settings](All_Settings_Alpha_Accuracy.png) · [Central trajectories](Central_Trajectories.png) · [Accuracy versus alpha](Accuracy_vs_Alpha.png)'+(' · [Paired effects](Paired_Effects.png)' if len(e)==120 else '')+'\n\nSeed SDs and paired confidence intervals are labeled separately. All completed outcomes, including failures, are retained.\n')
    (pub/'checkpoints/README.md').write_text(f"# Saved checkpoints\n\n{len(verification['checkpoint_manifest'])} checkpoint files were hashed. [Manifest](../verification/Checkpoint_Manifest.json) · [Saved archives](../ARTIFACTS.md). No follow-up run is active.\n")
    bundle_files=[p for p in pub.rglob('*') if p.is_file()]+[ROOT/n for n in ['Experiment02_Result_Summary.json','Checkpoint_Manifest.json','analyze_and_verify.py','verify_helpers.py','diagnostics.py']]
    bundle=preserve.archive_and_save('Experiment02_FashionMNIST_Results_And_Verification.zip',bundle_files)
    result['results_archive']=bundle['receipt']['results'][0];result['results_archive_sha256']=bundle['sha256']
    s.writej(ROOT/'Experiment02_Result_Summary.json',result)
    attempt=ROOT/'Result_Summary_Upload_Attempt.json'
    with attempt.open('x') as f:json.dump({'path':str(ROOT/'Experiment02_Result_Summary.json')},f)
    request={'uploads':[{'local_path':str(ROOT/'Experiment02_Result_Summary.json'),'purpose':'create_library_file','library_artifact_type':'other'}]}
    proc=subprocess.run([sys.executable,str(preserve.HELPER)],input=json.dumps(request),text=True,capture_output=True,timeout=300)
    (ROOT/'Result_Summary_Upload_Receipt.json').write_text(proc.stdout)
    (ROOT/'result_summary_upload.log').write_text(proc.stderr)
    assert proc.returncode==0,'Summary upload uncertain; inspect receipt before retry'
    receipt=json.loads(proc.stdout);assert receipt['results'][0]['status']=='succeeded' and receipt['results'][0]['local_metadata_applied']
    s.writej(ROOT/'Final_Save.json',{'results_bundle':bundle,'summary_receipt':receipt,'all_checkpoint_archives_saved':saved})
    print('FINAL RESULTS SAVED',receipt['results'][0]['library_file_id'],flush=True)
    return result


def main():
    p=json.loads((ROOT/'Protocol.json').read_text());pre=json.loads((ROOT/'Preflight_Audit.json').read_text())
    assert p['dataset']=='Fashion-MNIST' and pre['passed']
    for name,sha in json.loads((ROOT/'Execution_Source_Manifest.json').read_text()).items():assert s.sha(ROOT/name)==sha
    for name,sha in p['data_hashes'].items():assert s.sha(ROOT/'data'/name)==sha
    data=s.load_data()
    orders={(seed,ep):hashlib.sha256(s.order(seed,ep).tobytes()).hexdigest() for seed in p['seeds'] for ep in range(1,121)}
    completed=[spec for spec in p['specifications'] if (ROOT/'runs'/s.run_id(spec)/'done.json').exists()]
    if ARGS.smoke:
        assert completed,'No completed run available yet'
        done,h,hashes,check,activity=verify_run(completed[0],data,orders)
        print(json.dumps({'smoke_verified':True,'spec':completed[0],'model_status':check['model_status'],'accuracy':activity['test_accuracy_pct']}));return
    assert (ROOT/'Completion.json').exists(),'Training has not finished; do not finalize partial live outputs'
    assert completed,'No completed models available for analysis'
    pub=ROOT/'publication'
    for name in ('results','plots','verification','execution'):(pub/name).mkdir(parents=True,exist_ok=True)
    rows=[];trajectories=[];layer_rows=[];hashes=[];checks=[];activities=[]
    for spec in completed:
        done,history,manifest,check,activity=verify_run(spec,data,orders)
        hashes+=manifest;checks.append(check);activities.append(activity)
        row=endpoint(spec,done['final']);row.update(model_status=check['model_status'],review_flags=';'.join(check['review_flags']),
            run_id=s.run_id(spec),validation_accuracy_drop_pp=max(r['validation']['accuracy'] for r in history if 'validation' in r)*100-done['final']['validation']['accuracy']*100,
            validation_loss_rise=done['final']['validation']['loss']-min(r['validation']['loss'] for r in history if 'validation' in r))
        rows.append(row)
        for h in history:
            if 'spectra' in h:
                trajectories.append(endpoint(spec,h))
                for name,z in h['spectra'].items():layer_rows.append({**spec,'epoch':h['epoch'],'layer':name,**{k:z[k] for k in ['alpha','D','sigma','num_pl_spikes','num_evals','stable_rank','spectral_norm','norm','supported']}})
        print('VERIFIED',len(rows),s.run_id(spec),check['model_status'],row['test_accuracy_pct'],flush=True)
    e=pd.DataFrame(rows);tr=pd.DataFrame(trajectories)
    e.to_csv(pub/'results/endpoints.csv',index=False)
    all_status=e.to_dict('records')
    for entry in json.loads((ROOT/'Completion.json').read_text())['results']:
        if entry['status']!='complete':
            all_status.append({**entry['spec'],'run_id':s.run_id(entry['spec']),
                'model_status':'FAILED_TRAINING' if entry['status']=='nonfinite' else 'ERROR' if entry['status']=='exception' else 'INCOMPLETE',
                'execution_status':entry['status'],'review_flags':entry.get('error','')})
    status=pd.DataFrame(all_status)
    status.to_csv(pub/'results/model_status.csv',index=False)
    tr.to_csv(pub/'results/trajectories.csv',index=False);pd.DataFrame(layer_rows).to_csv(pub/'results/layer_measurements.csv',index=False)
    complete=len(e)==120
    if complete:assert e.groupby(['seed','arm']).size().eq(8).all()
    setting,overall,effects=statistics(e,complete,pub/'results')
    s.writej(ROOT/'Checkpoint_Manifest.json',hashes);s.writej(pub/'verification/Checkpoint_Manifest.json',hashes)
    verification={'passed':True,'dataset':'Fashion-MNIST','completed_runs':len(e),'source_sha256':p['source_sha256'],
        'checkpoint_manifest':hashes,'run_checks':checks,'data_hashes_match':True,'paired_initialization_and_minibatches_match':True,
        'all_test_predictions_and_logits_reproduced':True,'all_completed_runs_replayed_eleven_steps':True,
        'graft_magnitudes_checked':sum(c['graft_nondegenerate_magnitudes_checked'] for c in checks),
        'maximum_graft_magnitude_relative_error':max(c['max_graft_magnitude_relative_error'] for c in checks),
        'incomplete_training_runs':120-len(e),'verification_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    s.writej(pub/'verification/Final_Verification.json',verification);s.writej(pub/'verification/Model_Diagnostics.json',activities)
    plots(e,tr,effects,pub/'plots',complete)
    status_plot(status,pub/'plots')
    result=finalize(verification,e,setting,overall,effects,activities,pub)
    print(json.dumps({k:result[k] for k in ('status','completed_runs','usable_models','failed_models','all_completed_seed_archives_saved')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--analyze-only',action='store_true');ARGS=parser.parse_args()
    main()
