"""All-outcome report. No checkpoint selection, training or statistical exclusions."""
from pathlib import Path
import json,math,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import softmax,logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'Protocol.json').read_text())
ARMS=['adamw','magnitude_graft','sgdm']
LAYERS=['fc1','fc2','fc3']
COLORS=['#426baf','#dd7622','#268d67']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writej(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def interval(x):
 x=np.asarray(x,float)
 if len(x)!=5 or not np.isfinite(x).all():return [np.nan]*3
 m=x.mean();h=t.ppf(.975,4)*x.std(ddof=1)/np.sqrt(5)
 return [m,m-h,m+h]
def metrics(path):
 z=np.load(path);logits=z['logits'].astype(float);y=z['labels'];prob=softmax(logits,axis=1);confidence=prob.max(1);correct=prob.argmax(1)==y
 ece=0.
 for i in range(15):
  k=(confidence>=i/15)&(confidence<(i+1)/15 if i<14 else confidence<=1)
  if k.any():ece+=k.mean()*abs(correct[k].mean()-confidence[k].mean())
 return {'accuracy_pp':100*correct.mean(),'ce':np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y]),'ece15':ece,'brier':np.mean(np.sum((prob-np.eye(10)[y])**2,axis=1))}

def main():
 out=ROOT/'results';out.mkdir(exist_ok=True);plots=ROOT/'plots';plots.mkdir(exist_ok=True)
 models=[];histories=[];checks=[];signatures={}
 for spec in P['specifications']:
  d=ROOT/'runs'/f"b256_lr1/{spec['arm']}/seed_{spec['seed']}";row=dict(spec)
  if not (d/'done.json').exists():
   row['status']='ERROR' if (d/'failure.json').exists() else 'INCOMPLETE'
   if (d/'failure.json').exists():row['error']=json.loads((d/'failure.json').read_text())['error']
   models.append(row);continue
  done=json.loads((d/'done.json').read_text());row['status']=done['status'];row['endpoint_epoch']=done['endpoint_epoch'];row['initial_hash']=done['initial_hash'];row.update(metrics(d/'final_test_logits.npz'))
  checked=done['verification']['passed'] and done['endpoint_epoch']==P['fixed_epochs']
  checked=checked and abs(row['ce']-done['test']['loss'])<2e-6 and abs(row['accuracy_pp']-100*done['test']['accuracy'])<1e-9
  checked=checked and all(sha(d/name)==value for name,value in done['checkpoint_hashes'].items())
  checks.append({'spec':spec,'passed':bool(checked),'saved_checkpoint_count':len(done['checkpoint_hashes'])})
  for l in LAYERS:
   a=done['spectra'][l]['alpha'];a=float(a) if a is not None else np.nan;row['alpha_'+l]=a;row['deficit_'+l]=max(0,2-a) if np.isfinite(a) else np.nan;row['distance_'+l]=abs(a-2);row['supported_'+l]=done['spectra'][l]['supported']
  row['fc3_native_alpha']=done['spectra']['fc3']['native_alpha'];row['fc3_tail_count']=done['spectra']['fc3']['small_n']['n_tail'];row['fc3_sigma']=done['spectra']['fc3']['small_n']['sigma']
  row['hidden_deficit']=np.mean([row['deficit_'+l] for l in LAYERS[:2]]);row['all_layer_deficit']=np.mean([row['deficit_'+l] for l in LAYERS]);row['hidden_distance']=np.mean([row['distance_'+l] for l in LAYERS[:2]])
  models.append(row)
  h=json.loads((d/'history.json').read_text());signatures[(spec['seed'],spec['arm'])]=[(x['batch_order_sha256'],x['augmentation_sha256']) for x in h]
  for x in h:histories.append(dict(spec)|{'epoch':x['epoch'],'validation_ce':x['validation']['loss'],'validation_accuracy':100*x['validation']['accuracy'],'train_ce':x['train']['loss']}|{'alpha_'+l:x['spectra'][l]['alpha'] for l in LAYERS})
 df=pd.DataFrame(models);df.to_csv(out/'models.csv',index=False);pd.DataFrame(histories).to_csv(out/'histories.csv',index=False)
 completed=sum('endpoint_epoch' in x for x in models);pairing=[]
 for seed in P['seeds']:
  rows=df[df.seed==seed];keys=[(seed,a) for a in ARMS];present=all(k in signatures for k in keys)
  pairing.append({'seed':seed,'all_arms_present':present,'paired':bool(present and rows.initial_hash.nunique()==1 and all(signatures[k]==signatures[keys[0]] for k in keys)) if present else False})
 verification={'planned':15,'completed':completed,'all_fixed_endpoints_verified':completed==15 and all(x['passed'] for x in checks),'pairing':pairing,'checks':checks,'five_independent_seeds':True,'failures_retained':True,'FC3_small_sample_uncertainty':True}
 verification['passed']=verification['all_fixed_endpoints_verified'] and all(x['paired'] for x in pairing)
 writej(out/'Verification.json',verification)
 lines=['# Fashion-MNIST: audited magnitude graft with matched translation augmentation','',f'Completed {completed}/15 fixed-epoch120 endpoints. Three pilot arms are excluded from inference. All model statuses are retained.','',
 'Same 784→256→256→10 MLP, five fresh paired seeds, batch256 and original rates. Training uses a shared 50% chance of a zero-padded translation up to two pixels. The schedule now matches the original MNIST 120-epoch schedule; augmentation and seeds also changed, so historical differences are not attributable to one factor.','',
 'Scale audit establishes conditional optimizer coordinate covariance. Grafted matrices have the SGDM target-rate scaling law; retained Adam biases have the Adam rate law. Full MLP loss invariance, a common SGDM law for every parameter, and RG trace-log preservation are not established.','',
 'FC3 uses WeightWatcher’s small-N routine directly with zero alpha prior; the native constructor estimate is also saved. Its ten eigenvalues remain a small correlated sample. Historical alpha estimates are unchanged.','']
 if histories:
  h=pd.DataFrame(histories);fig,ax=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
  for a,color in zip(ARMS,COLORS):
   hh=h[h.arm==a]
   for j,l in enumerate(LAYERS):
    v=hh.groupby('epoch')['alpha_'+l].agg(['mean','std']);ax[j].plot(v.index,v['mean'],label=a,color=color);ax[j].fill_between(v.index,v['mean']-v['std'].fillna(0),v['mean']+v['std'].fillna(0),color=color,alpha=.12)
  for j,l in enumerate(LAYERS):ax[j].axhline(2,color='black',ls='--');ax[j].set_title(l.upper()+(' · small-N' if l=='fc3' else ''));ax[j].set_xlabel('Epoch');ax[j].set_ylabel('Alpha')
  ax[0].legend(fontsize=8);fig.suptitle('Completed models; bands are seed SD, not confidence intervals');fig.savefig(plots/'alpha_trajectories.png',dpi=160);plt.close(fig)
  fig,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
  for a,color in zip(ARMS,COLORS):
   hh=h[h.arm==a]
   for j,metric in enumerate(['validation_ce','validation_accuracy']):
    for seed,g in hh.groupby('seed'):ax[j].plot(g.epoch,g[metric],color=color,alpha=.2,lw=.8)
    v=hh.groupby('epoch')[metric].mean();ax[j].plot(v.index,v,label=a,color=color)
  for a,title in zip(ax,['Validation cross-entropy','Validation accuracy (%)']):a.set_title(title);a.set_xlabel('Epoch')
  ax[0].legend(fontsize=8);fig.suptitle('Raw validation curves; no monotonicity or checkpoint selection');fig.savefig(plots/'validation_trajectories.png',dpi=160);plt.close(fig)
 effects=[]
 if completed:
  baseline=df[df.arm=='adamw'].set_index('seed').reindex(P['seeds'])
  for arm in ARMS[1:]:
   target=df[df.arm==arm].set_index('seed').reindex(P['seeds'])
   for m in ['accuracy_pp','ce','ece15','brier','hidden_deficit','all_layer_deficit','hidden_distance']+['alpha_'+l for l in LAYERS]+['distance_'+l for l in LAYERS]:
    if m not in target or m not in baseline:continue
    mu,lo,hi=interval(target[m]-baseline[m]);effects.append({'arm':arm,'metric':m,'mean':mu,'ci_low':lo,'ci_high':hi,'independent_seeds':5})
  for arm in ARMS[1:]:
   target=df[df.arm==arm].set_index('seed').reindex(P['seeds'])
   for layers,label in [(LAYERS[:2],'hidden'),(LAYERS,'all_layers')]:
    cols=['alpha_'+l for l in layers]
    if not all(c in target and c in baseline for c in cols):continue
    aa=baseline[cols].to_numpy(float);tt=target[cols].to_numpy(float)
    for metric,da,dt in [('symmetric_weighted_rescue',np.abs(aa-2),np.abs(tt-2)),('sub2_weighted_rescue',np.maximum(0,2-aa),np.maximum(0,2-tt))]:
     denom=da.sum(axis=1);scores=np.divide((da*(da-dt)).sum(axis=1),denom,out=np.full(5,np.nan),where=denom>0)
     mu,lo,hi=interval(scores);effects.append({'arm':arm,'metric':label+'_'+metric,'mean':mu,'ci_low':lo,'ci_high':hi,'independent_seeds':5})
  pd.DataFrame(effects).to_csv(out/'paired_effects.csv',index=False)
  lines+=['## Paired treatment minus AdamW','','Five seed-level differences; 95% Student-t intervals, df4. Missing endpoints produce undefined intervals. Positive accuracy is favorable; negative CE and sub-2 deficit are favorable. Alpha increases are separate from alignment.','','| Arm | Metric | Mean | 95% interval |','|---|---|---:|---:|']
  for e in effects:lines.append(f"| {e['arm']} | {e['metric']} | {e['mean']:.5f} | [{e['ci_low']:.5f}, {e['ci_high']:.5f}] |")
  lines+=['','## Endpoint alpha means','','| Arm | FC1 | FC2 | FC3 small-N | Accuracy (%) | CE |','|---|---:|---:|---:|---:|---:|']
  for arm in ARMS:
   g=df[df.arm==arm];vals=[g[m].mean() if m in g else np.nan for m in ['alpha_fc1','alpha_fc2','alpha_fc3','accuracy_pp','ce']];lines.append('| '+arm+' | '+' | '.join(f'{v:.5f}' for v in vals)+' |')
 lines+=['','![All-layer alpha trajectories](plots/alpha_trajectories.png)','','![Validation trajectories](plots/validation_trajectories.png)','',
 '[All outcomes](results/models.csv) · [Verification](results/Verification.json) · [Protocol](Protocol.json) · [Scale derivation](SCALE_DERIVATION.md) · [Literature](LITERATURE.md)','',
 'This is one prespecified configuration, not an optimum search. Prior test exposure and the hypothesis-driven follow-up make it a targeted replication, not untouched external confirmation. Accuracy improvement is asserted only when its paired interval excludes zero; spectral or loss outcomes are interpreted separately.']
 (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'completed':completed,'planned':15,'verified':verification['passed']}))

if __name__=='__main__':main()
