from pathlib import Path
import json,math,zipfile,hashlib
import numpy as np,pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;out=p/'results';plots=p/'plots';out.mkdir(exist_ok=True);plots.mkdir(exist_ok=True)
plan=json.loads((p/'Protocol.json').read_text());rows=[];status=[];layers=[];trajectories=[]
for sp in plan['specifications']:
 id=f"b{sp['batch_size']}_lr{sp['multiplier']:g}/{sp['arm']}/seed_{sp['seed']}";d=p/'runs'/id
 if not (d/'done.json').exists():
  status.append({**sp,'status':'ERROR' if (d/'failure.json').exists() else 'INCOMPLETE'});continue
 r=json.loads((d/'done.json').read_text());a=np.array([r['spectra'][n]['alpha'] if r['spectra'][n]['alpha'] is not None else np.nan for n in ['fc1','fc2']])
 row={**sp,'status':r['status'],'endpoint_epoch':r['endpoint_epoch'],'stopped_epoch':r['stopped_epoch'],'alpha_fc1':a[0],'alpha_fc2':a[1],'mean_alpha':a.mean(),'mean_abs_alpha_minus_2':abs(a-2).mean(),'alpha_layer_variance':a.var(),'distance_fc1':abs(a[0]-2),'distance_fc2':abs(a[1]-2),'alpha_fc3':r['spectra']['fc3']['alpha'],'distance_fc3':abs(r['spectra']['fc3']['alpha']-2) if r['spectra']['fc3']['alpha'] is not None else np.nan,'validation_loss':r['validation']['loss'],'test_accuracy_pct':100*r['test']['accuracy'],'test_loss':r['test']['loss'],'supported_hidden_layers':sum(r['spectra'][n]['supported'] for n in ['fc1','fc2'])}
 rows.append(row);status.append(row)
 for n,z in r['spectra'].items():layers.append({**sp,'endpoint_epoch':r['endpoint_epoch'],'layer':n,**{k:z[k] for k in ['alpha','D','sigma','xmin','xmax','num_pl_spikes','supported','spectral_norm','norm']}})
 for h in json.loads((d/'history.json').read_text()):
  trajectories.append({**sp,'epoch':h['epoch'],'train_loss':h['train']['loss'],'validation_loss':h['validation']['loss'],'minimum_so_far':h['minimum_validation_loss_seen'],'fixed_endpoint':h['epoch']==r['endpoint_epoch']})
e=pd.DataFrame(rows);tr=pd.DataFrame(trajectories);pd.DataFrame(status).to_csv(out/'model_status.csv',index=False);e.to_csv(out/'endpoints.csv',index=False);pd.DataFrame(layers).to_csv(out/'layer_measurements.csv',index=False);tr.to_csv(out/'raw_loss_trajectories.csv',index=False)
complete=len(e)==120;effects=[];means=[]
metrics=['alpha_fc1','alpha_fc2','alpha_fc3','distance_fc1','distance_fc2','distance_fc3','mean_alpha','mean_abs_alpha_minus_2','alpha_layer_variance','validation_loss','test_accuracy_pct','test_loss']
if complete:
 assert e.groupby(['seed','arm']).size().eq(8).all()
 for scope,frame in [('all_settings',e)]+[(f'b{b}_lr{m:g}',g) for (b,m),g in e.groupby(['batch_size','multiplier'])]:
  avg=frame.groupby(['seed','arm'])[metrics].agg(lambda x:x.mean(skipna=False))
  for arm in ['adamw','magnitude_graft','sgdm']:
   for met in metrics:
    v=avg.xs(arm,level='arm')[met].dropna();means.append({'scope':scope,'arm':arm,'metric':met,'independent_seeds':len(v),'mean':v.mean(),'seed_sd':v.std(ddof=1)})
  for arm in ['magnitude_graft','sgdm']:
   delta=avg.xs(arm,level='arm')-avg.xs('adamw',level='arm')
   for met in metrics:
    x=delta[met].dropna();n=len(x);mu=x.mean();half=t.ppf(.975,n-1)*x.std(ddof=1)/math.sqrt(n) if n>1 else np.nan
    effects.append({'scope':scope,'treatment':arm,'reference':'adamw','metric':met,'independent_seeds':n,'mean_delta':mu,'ci_low':mu-half,'ci_high':mu+half})
pd.DataFrame(means).to_csv(out/'seed_summaries.csv',index=False);pd.DataFrame(effects).to_csv(out/'paired_effects.csv',index=False)
# Pairing and update-statistics audit independent of endpoint alpha.
audit={'completed':len(rows),'paired_initialization':True,'paired_minibatches':True,'fixed_epoch20':True,'restored_replay':True,'graft_nonzero_records':0,'maximum_graft_norm_relative_error':0.0}
groups={}
for sp in plan['specifications']:
 d=p/'runs'/f"b{sp['batch_size']}_lr{sp['multiplier']:g}/{sp['arm']}/seed_{sp['seed']}"
 if not (d/'done.json').exists():continue
 r=json.loads((d/'done.json').read_text());h=json.loads((d/'history.json').read_text());key=(sp['seed'],sp['batch_size'],sp['multiplier'])
 fingerprints=(r['initial_hash'],[x['batch_order_sha256'] for x in h])
 if key in groups:
  audit['paired_initialization'] &= fingerprints[0]==groups[key][0]
  audit['paired_minibatches'] &= fingerprints[1]==groups[key][1]
 else:groups[key]=fingerprints
 audit['fixed_epoch20'] &= r['endpoint_epoch']==20
 audit['restored_replay'] &= bool(r['verification']['checkpoint_replay_exact'] and r['verification']['restored_validation_and_spectra_exact'])
 if sp['arm']=='magnitude_graft':
  for q in (d/'statistics').glob('epoch_*.npz'):
   a=np.load(q)['values'];target=a[:,:,3];actual=a[:,:,4];good=(target>0)&np.isfinite(target)&np.isfinite(actual)
   audit['graft_nonzero_records']+=int(good.sum())
   if good.any():audit['maximum_graft_norm_relative_error']=max(audit['maximum_graft_norm_relative_error'],float(np.max(np.abs(actual[good]/target[good]-1))))
audit['optimizer_contract']=json.loads((p/'Optimizer_Contract.json').read_text()) if (p/'Optimizer_Contract.json').exists() else {'passed':False}
audit['passed']=len(rows)>0 and all(audit[k] for k in ['paired_initialization','paired_minibatches','fixed_epoch20','restored_replay']) and audit['optimizer_contract']['passed'] and audit['maximum_graft_norm_relative_error']<2e-4
(out/'Verification.json').write_text(json.dumps(audit,indent=2)+'\n')
summary={'dataset':'Fashion-MNIST','protocol':'separate fixed-epoch20 restart','planned':120,'completed_verified':len(e),'complete':complete,'verification_passed':bool(audit['passed']),'five_seeds_not_eight_independent_settings':True,'original120epoch_results_preserved':True,'gpt_on_hold':True,'paired_effects':effects}
(out/'Clean_Restart_Result_Summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if len(e):
 colors={'adamw':'#2867A6','magnitude_graft':'#CD6B27','sgdm':'#35815B'}
 fig,axs=plt.subplots(1,2,figsize=(11,4.8))
 for arm,g in e.groupby('arm'):
  axs[0].scatter(g.mean_alpha,g.test_accuracy_pct,label=arm,color=colors[arm],alpha=.7);axs[1].scatter(g.mean_abs_alpha_minus_2,g.test_accuracy_pct,label=arm,color=colors[arm],alpha=.7)
 for ax in axs:ax.set_ylabel('Epoch20 test accuracy (%)');ax.grid(alpha=.15)
 axs[0].set_xlabel('Mean hidden α');axs[0].axvline(2,color='gray',ls='--');axs[1].set_xlabel('Mean hidden |α − 2|');axs[1].legend();fig.suptitle('Fixed epoch20; all completed outcomes');fig.tight_layout();fig.savefig(plots/'Final_Alpha_Accuracy.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
 central=tr[(tr.batch_size==256)&(tr.multiplier==1)]
 for ax,arm in zip(axs,['adamw','magnitude_graft','sgdm']):
  for seed,g in central[central.arm==arm].groupby('seed'):
   ax.plot(g.epoch,g.validation_loss,color=colors[arm],alpha=.6);ax.plot(g.epoch,g.train_loss,color=colors[arm],alpha=.3,ls='--');best=g[g.fixed_endpoint];ax.scatter(best.epoch,best.validation_loss,color=colors[arm],marker='x')
  ax.set_title(arm);ax.set_xlabel('Epoch');ax.grid(alpha=.15)
 axs[0].set_ylabel('Cross-entropy (solid validation; dashed train)');fig.suptitle('Central setting: raw curves, fixed epoch20 marked ×');fig.tight_layout();fig.savefig(plots/'Raw_Validation_Loss.png',dpi=160);plt.close(fig)
if len(tr):
 fig,axs=plt.subplots(2,4,figsize=(16,8),sharey=True)
 for ax,((b,m),g) in zip(axs.flat,tr.groupby(['batch_size','multiplier'])):
  for arm,h in g.groupby('arm'):
   for seed,v in h.groupby('seed'):ax.plot(v.epoch,v.validation_loss,color=colors[arm],alpha=.35,lw=1)
  ax.set_title(f'Batch{b}, LR multiplier{m:g}');ax.set_xlabel('Epoch');ax.grid(alpha=.15)
 fig.suptitle('All completed runs: actual validation loss, no checkpoint selection');fig.tight_layout();fig.savefig(plots/'All_Settings_Validation_Loss.png',dpi=150);plt.close(fig)
lines=['# Clean fixed-endpoint Fashion-MNIST restart','',f'{len(e)}/120 completed and verified. All earlier outcomes remain separate.','', 'All primary endpoints are epoch20. Five-epoch warmup followed by cosine annealing over15epochs to10% of peak rate. Same architecture, frozen optimizer kernel, layer rates, five paired seeds, data split and minibatches. Validation and WeightWatcher evaluated every epoch; no loss-based or alpha-based checkpoint selection. Separate central-setting pilot seed31999 excluded from inference. Test evaluated once per final checkpoint. The test set was used in earlier experiments, so this is not a previously unseen test set.','', '[Model register](results/model_status.csv) · [Final endpoints](results/endpoints.csv) · [Layer fits](results/layer_measurements.csv) · [Paired effects](results/paired_effects.csv) · [Raw loss curves](results/raw_loss_trajectories.csv)','', 'Five seeds are independent; eight settings are averaged within seed for overall paired95%t intervals. All outcomes and unsupported fits are retained. FC3 has only10eigenvalues and is exploratory/unsupported. Functional quality requires validation accuracy>=80% with nonconstant predictions; alpha is separate. Upward alpha, smaller|alpha−2|, greater accuracy and lower cross-entropy are distinct outcomes. These descriptive paired intervals are unadjusted across multiple metrics/settings; no accuracy improvement when an interval includes0.','']
if not complete:lines+=['**Incomplete: no full five-seed conclusion is claimed.**','']
if len(e):lines+=['![Raw loss](plots/Raw_Validation_Loss.png)','','![Selected alpha and accuracy](plots/Final_Alpha_Accuracy.png)','']
(p/'RESULTS.md').write_text('\n'.join(lines))
archives=p/'archives';archives.mkdir(exist_ok=True)
for seed in plan['independent_seeds']:
 files=[f for d in (p/'runs').glob(f'*/*/seed_{seed}') for f in d.rglob('*') if f.is_file() and f.suffix!='.tmp']
 if not files:continue
 files += [f for f in p.glob('*.json') if f.is_file()]+[f for f in p.glob('*.py') if f.is_file()]
 q=archives/f'Fashion_Clean_Restart_Seed_{seed}.zip'
 with zipfile.ZipFile(q,'w',zipfile.ZIP_STORED) as z:
  for f in files:z.write(f,str(f.relative_to(p)))
 (archives/(q.name+'.json')).write_text(json.dumps({'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size},indent=2)+'\n')
print({'completed_verified':len(e),'complete':complete})

if (p/'pilot').exists():
 q=archives/'Fashion_Clean_Restart_Pilot.zip'
 with zipfile.ZipFile(q,'w',zipfile.ZIP_STORED) as z:
  for f in (p/'pilot').rglob('*'):
   if f.is_file() and f.suffix!='.tmp':z.write(f,str(f.relative_to(p)))
  for f in p.glob('*.json'):z.write(f,str(f.relative_to(p)))
 (archives/(q.name+'.json')).write_text(json.dumps({'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size},indent=2)+'\n')
