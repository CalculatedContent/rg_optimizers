"""All-run report: five independent seeds, failures retained, no alpha selection."""
import json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent;incoming=Path('incoming');out=root/'results';plots=root/'plots';out.mkdir(exist_ok=True);plots.mkdir(exist_ok=True)
rows=[];layers=[];status=[];histories=[]
for d in sorted(incoming.glob('full-*')):
 jobs=list(d.glob('runs/*/done.json'));fail=d/'Failure.json'
 if not jobs:
  status.append({'artifact':d.name,'status':'FAILED_OR_INCOMPLETE','detail':fail.read_text() if fail.exists() else 'No completed checkpoint report'});continue
 j=json.loads(jobs[0].read_text());vpath=jobs[0].parent/'Verification.json'
 if not vpath.exists():status.append({'artifact':d.name,'status':'VERIFICATION_FAILED'});continue
 v=json.loads(vpath.read_text());spec=j['spec'];h=j['history'][-1]
 aa=[x['alpha'] for x in h['spectra'] if x['part']=='physical'];valid=all(x is not None for x in aa)
 row={**{k:spec[k] for k in ['id','seed','arm','multiplier']},'status':v['functional_status'],'test_loss':j['test']['loss'],'test_bits_per_character':j['test']['bits_per_character'],'test_accuracy_pct':100*j['test']['accuracy'],'validation_loss':h['validation']['loss'],'mean_alpha':float(np.mean(aa)) if valid else np.nan,'mean_abs_alpha_minus_2':float(np.mean(np.abs(np.array(aa)-2))) if valid else np.nan,'alpha_layer_variance':float(np.var(aa)) if valid else np.nan,'supported_physical_matrices':sum(x['supported'] for x in h['spectra'] if x['part']=='physical')}
 rows.append(row);status.append(row)
 for h in j['history']:
  histories.append({**{k:spec[k] for k in ['seed','arm','multiplier']},'step':h['step'],'validation_loss':h['validation']['loss']})
  for layer in h['spectra']:layers.append({**{k:spec[k] for k in ['seed','arm','multiplier']},'step':h['step'],**layer})
e=pd.DataFrame(rows);pd.DataFrame(status).to_csv(out/'model_status.csv',index=False);pd.DataFrame(layers).to_csv(out/'layer_spectra.csv',index=False);pd.DataFrame(histories).to_csv(out/'validation_trajectories.csv',index=False)
complete=len(e)==45 and e.groupby(['seed','arm']).size().eq(3).all()
e.to_csv(out/'endpoints.csv',index=False)
metrics=['mean_alpha','mean_abs_alpha_minus_2','alpha_layer_variance','test_loss','test_bits_per_character','test_accuracy_pct']
effects=[]
if complete:
 for scope,frame in [('all_settings',e)]+[(f'lr_{m:g}',g) for m,g in e.groupby('multiplier')]:
  means=frame.groupby(['seed','arm'])[metrics].agg(lambda x:x.mean(skipna=False))
  for arm in ['magnitude_graft','sgdm']:
   delta=means.xs(arm,level='arm')-means.xs('adamw',level='arm')
   for metric in metrics:
    x=delta[metric].dropna();n=len(x);mu=x.mean();sd=x.std(ddof=1);h=t.ppf(.975,n-1)*sd/math.sqrt(n) if n>1 else np.nan
    effects.append({'scope':scope,'treatment':arm,'reference':'adamw','metric':metric,'independent_seeds':n,'mean_delta':mu,'ci_low':mu-h,'ci_high':mu+h})
pd.DataFrame(effects).to_csv(out/'paired_effects.csv',index=False)
summary={'experiment':'Experiment03','dataset':'Tiny Shakespeare','completed_verified_runs':len(e),'planned_confirmatory_runs':45,'complete':bool(complete),'independent_seeds':5,'pilot_is_separate':True,'failures_retained':True,'paired_effects':effects}
(out/'Experiment03_Result_Summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if len(e):
 colors={'adamw':'#2867A6','magnitude_graft':'#CD6B27','sgdm':'#35815B'}
 fig,axs=plt.subplots(1,2,figsize=(11,4.5))
 for arm,g in e.groupby('arm'):
  for ax,feature in zip(axs,['mean_alpha','mean_abs_alpha_minus_2']):ax.scatter(g[feature],g.test_bits_per_character,label=arm,color=colors[arm]);ax.set_xlabel(feature);ax.set_ylabel('Held-out bits / character (lower is better)');ax.grid(alpha=.15)
 axs[0].axvline(2,color='gray',ls='--');axs[-1].legend();fig.suptitle('All verified outcomes; settings are not independent seeds');fig.tight_layout();fig.savefig(plots/'Alpha_vs_Test_Loss.png',dpi=160);plt.close(fig)
 layer=pd.DataFrame(layers);z=layer[(layer.step==10000)&(layer.part=='physical')];fig,ax=plt.subplots(figsize=(12,5))
 names=sorted(z.matrix.unique())
 for j,arm in enumerate(['adamw','magnitude_graft','sgdm']):
  g=z[z.arm==arm]
  for i,name in enumerate(names):
   vals=g[g.matrix==name].alpha.dropna();ax.scatter(np.full(len(vals),i+(j-1)*.2),vals,s=14,alpha=.55,color=colors[arm],label=arm if i==0 else None)
 ax.axhline(2,color='gray',ls='--');ax.set_xticks(range(len(names)),[n.replace('transformer.h.','').replace('.weight','') for n in names],rotation=35,ha='right');ax.set_ylabel('WeightWatcher α');ax.legend();fig.tight_layout();fig.savefig(plots/'Per_Matrix_Alpha.png',dpi=160);plt.close(fig)
lines=['# Experiment03: GPT results','',f'{len(e)}/45 confirmatory runs completed and verified. Pilot runs are excluded.','', 'The magnitude-only graft retains Adam direction/moments/bias updates on each run. It changes only the magnitudes of the eight physical attention/MLP matrix updates. Embeddings and tied output head retain native Adam updates in the graft arm.','', 'Five seeds are the independent replicates. Across three learning-rate settings, differences are averaged within each seed before paired 95% t intervals. All outcomes remain visible; unsupported fits are flagged. Positive alpha movement is not necessarily closer to2. Lower test loss is better.','', '[Every model](results/model_status.csv) · [Per-matrix spectra](results/layer_spectra.csv) · [Paired effects](results/paired_effects.csv)','']
if not complete:lines+=['**Incomplete pass: no completed five-seed conclusion is claimed.**','']
if len(e):lines+=['![Alpha and test loss](plots/Alpha_vs_Test_Loss.png)','','![Matrix exponents](plots/Per_Matrix_Alpha.png)','']
(root/'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps({'completed':len(e),'complete':bool(complete)}))
