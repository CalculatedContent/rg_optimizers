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
 row={**sp,'status':r['status'],'best_epoch':r['best_epoch'],'stopped_epoch':r['stopped_epoch'],'alpha_fc1':a[0],'alpha_fc2':a[1],'mean_alpha':a.mean(),'mean_abs_alpha_minus_2':abs(a-2).mean(),'alpha_layer_variance':a.var(),'validation_loss':r['validation']['loss'],'test_accuracy_pct':100*r['test']['accuracy'],'test_loss':r['test']['loss'],'supported_hidden_layers':sum(r['spectra'][n]['supported'] for n in ['fc1','fc2'])}
 rows.append(row);status.append(row)
 for n,z in r['spectra'].items():layers.append({**sp,'best_epoch':r['best_epoch'],'layer':n,**{k:z[k] for k in ['alpha','D','sigma','xmin','xmax','num_pl_spikes','supported','spectral_norm','norm']}})
 for h in json.loads((d/'history.json').read_text()):
  trajectories.append({**sp,'epoch':h['epoch'],'train_loss':h['train']['loss'],'validation_loss':h['validation']['loss'],'best_so_far':h['best_validation_loss_so_far'],'selected_epoch':h['epoch']==r['best_epoch']})
e=pd.DataFrame(rows);tr=pd.DataFrame(trajectories);pd.DataFrame(status).to_csv(out/'model_status.csv',index=False);e.to_csv(out/'endpoints.csv',index=False);pd.DataFrame(layers).to_csv(out/'layer_measurements.csv',index=False);tr.to_csv(out/'raw_loss_trajectories.csv',index=False)
complete=len(e)==120;effects=[];means=[]
metrics=['alpha_fc1','alpha_fc2','mean_alpha','mean_abs_alpha_minus_2','alpha_layer_variance','validation_loss','test_accuracy_pct','test_loss']
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
summary={'dataset':'Fashion-MNIST','protocol':'separate validation-stopped rerun','planned':120,'completed_verified':len(e),'complete':complete,'five_seeds_not_eight_independent_settings':True,'original120epoch_results_preserved':True,'gpt_on_hold':True,'paired_effects':effects}
(out/'Validation_Rerun_Result_Summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if len(e):
 colors={'adamw':'#2867A6','magnitude_graft':'#CD6B27','sgdm':'#35815B'}
 fig,axs=plt.subplots(1,2,figsize=(11,4.8))
 for arm,g in e.groupby('arm'):
  axs[0].scatter(g.mean_alpha,g.test_accuracy_pct,label=arm,color=colors[arm],alpha=.7);axs[1].scatter(g.mean_abs_alpha_minus_2,g.test_accuracy_pct,label=arm,color=colors[arm],alpha=.7)
 for ax in axs:ax.set_ylabel('Selected-checkpoint test accuracy (%)');ax.grid(alpha=.15)
 axs[0].set_xlabel('Mean hidden α');axs[0].axvline(2,color='gray',ls='--');axs[1].set_xlabel('Mean hidden |α − 2|');axs[1].legend();fig.suptitle('Validation-selected checkpoints; all completed outcomes');fig.tight_layout();fig.savefig(plots/'Selected_Alpha_Accuracy.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
 central=tr[(tr.batch_size==256)&(tr.multiplier==1)]
 for ax,arm in zip(axs,['adamw','magnitude_graft','sgdm']):
  for seed,g in central[central.arm==arm].groupby('seed'):
   ax.plot(g.epoch,g.validation_loss,color=colors[arm],alpha=.6);best=g[g.selected_epoch];ax.scatter(best.epoch,best.validation_loss,color=colors[arm],marker='x')
  ax.set_title(arm);ax.set_xlabel('Epoch');ax.grid(alpha=.15)
 axs[0].set_ylabel('Actual validation cross-entropy');fig.suptitle('Central setting: raw curves, selected checkpoints marked ×');fig.tight_layout();fig.savefig(plots/'Raw_Validation_Loss.png',dpi=160);plt.close(fig)
lines=['# Validation-stopped Fashion-MNIST rerun','',f'{len(e)}/120 completed and verified. Original fixed120-epoch outcomes remain separate.','', 'Same architecture, optimizer kernels, layer rates, five seeds, data split and minibatches. Validation evaluated every epoch; restore minimum validation cross-entropy checkpoint; stop after10epochs without improvement, minimum10 and maximum120epochs. Raw validation loss may fluctuate and is never rewritten as a monotonic curve. Neither test accuracy nor alpha selects checkpoints.','', '[Model register](results/model_status.csv) · [Selected endpoints](results/endpoints.csv) · [Layer fits](results/layer_measurements.csv) · [Paired effects](results/paired_effects.csv) · [Raw loss curves](results/raw_loss_trajectories.csv)','', 'Five seeds are independent; eight settings are averaged within seed for overall paired95%t intervals. Pilot central cases count once in120. Failures and unsupported fits are retained. Lower |alpha−2|, upward alpha movement, accuracy and cross-entropy are distinct outcomes.','']
if not complete:lines+=['**Incomplete: no full five-seed conclusion is claimed.**','']
if len(e):lines+=['![Raw loss](plots/Raw_Validation_Loss.png)','','![Selected alpha and accuracy](plots/Selected_Alpha_Accuracy.png)','']
(p/'RESULTS.md').write_text('\n'.join(lines))
archives=p/'archives';archives.mkdir(exist_ok=True)
for seed in plan['independent_seeds']:
 files=[f for d in (p/'runs').glob(f'*/*/seed_{seed}') for f in d.rglob('*') if f.is_file() and f.suffix!='.tmp']
 if not files:continue
 files += [f for f in p.glob('*.json') if f.is_file()]+[f for f in p.glob('*.py') if f.is_file()]
 q=archives/f'Fashion_Validation_Rerun_Seed_{seed}.zip'
 with zipfile.ZipFile(q,'w',zipfile.ZIP_STORED) as z:
  for f in files:z.write(f,str(f.relative_to(p)))
 (archives/(q.name+'.json')).write_text(json.dumps({'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size},indent=2)+'\n')
print({'completed_verified':len(e),'complete':complete})
