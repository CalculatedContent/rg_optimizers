"""Reproduce descriptive MNIST/Fashion comparison. No training or selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parents[1]
OUT=P/'comparison';OUT.mkdir(exist_ok=True)
PLOTS=P/'plots';PLOTS.mkdir(exist_ok=True)
M=pd.read_csv(P/'inputs/mnist/endpoints.csv').rename(columns={'alpha_head':'alpha_fc3','alpha_variance':'alpha_layer_variance'})
F=pd.read_csv(P/'results/endpoints.csv').rename(columns={'endpoint_epoch':'epoch'})
S=pd.read_csv(P/'inputs/mnist/model_status.csv')
keys=['seed','batch_size','multiplier','arm']
assert len(M)==135 and len(F)==120 and not M.duplicated(keys).any() and not F.duplicated(keys).any()
M=M.merge(S[S.cohort=='original'][keys+['model_status','status_reason']],on=keys,validate='one_to_one').rename(columns={'model_status':'status'})
settings=set(zip(F.batch_size,F.multiplier))
M['in_shared_grid']=[(b,m) in settings for b,m in zip(M.batch_size,M.multiplier)]
M.to_csv(OUT/'mnist_all_original_outcomes.csv',index=False)
S.to_csv(OUT/'mnist_original_and_replication_status.csv',index=False)
M=M[M.in_shared_grid].copy()
assert len(M)==120 and set(M.seed)==set(F.seed)==set(range(31001,31006))
assert (M.epoch==120).all() and (F.epoch==20).all()
for d in [M,F]:
 for n in ['fc1','fc2','fc3']:d['distance_'+n]=(d['alpha_'+n]-2).abs()
 assert np.allclose(d.mean_abs_alpha_minus_2,(d.distance_fc1+d.distance_fc2)/2)
 assert d.groupby(['seed','arm']).size().eq(8).all()
M['dataset']='MNIST';F['dataset']='Fashion-MNIST'
cols=keys+['dataset','epoch','status','alpha_fc1','alpha_fc2','alpha_fc3','distance_fc1','distance_fc2','distance_fc3','mean_alpha','mean_abs_alpha_minus_2','alpha_layer_variance','test_accuracy_pct','test_loss','validation_loss','supported_hidden_layers']
D=pd.concat([M[cols],F[cols]],ignore_index=True)
D.to_csv(OUT/'shared_setting_endpoints.csv',index=False)
METRICS=['alpha_fc1','alpha_fc2','alpha_fc3','distance_fc1','distance_fc2','distance_fc3','mean_alpha','mean_abs_alpha_minus_2','alpha_layer_variance','test_accuracy_pct','test_loss','validation_loss']
means=[];effects=[];seeds=[]
def interval(v):
 v=np.asarray(v,dtype=float);assert len(v)==5 and np.isfinite(v).all()
 mu=v.mean();sd=v.std(ddof=1);half=t.ppf(.975,4)*sd/np.sqrt(5)
 return dict(n_seeds=5,mean=float(mu),seed_sd=float(sd),ci_low=float(mu-half),ci_high=float(mu+half))
for ds,frame in D.groupby('dataset'):
 scopes=[('common_eight_settings',frame)]+[(f'b{b}_lr{m:g}',g) for (b,m),g in frame.groupby(['batch_size','multiplier'])]
 for scope,g in scopes:
  avg=g.groupby(['seed','arm'])[METRICS].mean()
  for (seed,arm),v in avg.iterrows():seeds.append(dict(dataset=ds,scope=scope,seed=seed,arm=arm,**v.to_dict()))
  for arm in ['adamw','magnitude_graft','sgdm']:
   for met in METRICS:means.append(dict(dataset=ds,scope=scope,arm=arm,metric=met,**interval(avg.xs(arm,level='arm')[met])))
  for arm in ['magnitude_graft','sgdm']:
   delta=avg.xs(arm,level='arm')-avg.xs('adamw',level='arm')
   for met in METRICS:effects.append(dict(dataset=ds,scope=scope,treatment=arm,reference='adamw',metric=met,**interval(delta[met])))
A=pd.DataFrame(means);E=pd.DataFrame(effects)
A.to_csv(OUT/'arm_means_with_95ci.csv',index=False);E.to_csv(OUT/'paired_effects_with_95ci.csv',index=False)
pd.DataFrame(seeds).to_csv(OUT/'independent_seed_summaries.csv',index=False)
# Independent agreement with both already published analyses.
maximum_error=0.0
for filename,ds,scope in [('stable_paired_effects.csv','MNIST','common_eight_settings')]:
 old=pd.read_csv(P/'inputs/mnist'/filename)
 for _,r in old[old.reference=='adamw'].iterrows():
  v=E[(E.dataset==ds)&(E.scope==scope)&(E.treatment==r.treatment)&(E.metric==r.metric)].iloc[0]
  maximum_error=max(maximum_error,max(abs(v[k]-r[k]) for k in ['mean','ci_low','ci_high']))
old=pd.read_csv(P/'results/paired_effects.csv')
for _,r in old[old.scope=='all_settings'].iterrows():
 v=E[(E.dataset=='Fashion-MNIST')&(E.scope=='common_eight_settings')&(E.treatment==r.treatment)&(E.metric==r.metric)].iloc[0]
 maximum_error=max(maximum_error,abs(v['mean']-r.mean_delta),abs(v.ci_low-r.ci_low),abs(v.ci_high-r.ci_high))
assert maximum_error<1e-10
(OUT/'Comparison_Verification.json').write_text(json.dumps({'passed':True,'mnist_original_runs':135,'mnist_shared_grid_runs':120,'fashion_shared_grid_runs':120,'independent_seeds_per_dataset':5,'maximum_difference_from_published_intervals':maximum_error,'cross_dataset_causal_claim':False,'endpoint_selection':False},indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'adamw':'#2469A4','magnitude_graft':'#D16C26','sgdm':'#348565'}
names={'adamw':'AdamW','magnitude_graft':'Magnitude graft','sgdm':'SGDM'}
labels={'MNIST':'MNIST · epoch 120','Fashion-MNIST':'Fashion-MNIST · epoch 20'}
fig,axs=plt.subplots(2,3,figsize=(13.5,7.8))
panels=[('alpha_fc1','Change in FC1 α'),('alpha_fc2','Change in FC2 α'),('mean_abs_alpha_minus_2','Change in mean |α − 2|\nNegative = better alignment'),('test_accuracy_pct','Change in test accuracy (pp)\nPositive = better accuracy'),('test_loss','Change in test cross-entropy\nNegative = better loss'),('alpha_layer_variance','Change in layer α variance')]
order=[('MNIST','magnitude_graft'),('MNIST','sgdm'),('Fashion-MNIST','magnitude_graft'),('Fashion-MNIST','sgdm')]
for ax,(metric,title) in zip(axs.flat,panels):
 for y,(ds,arm) in enumerate(order):
  r=E[(E.dataset==ds)&(E.scope=='common_eight_settings')&(E.treatment==arm)&(E.metric==metric)].iloc[0]
  ax.errorbar(r['mean'],y,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o' if arm=='magnitude_graft' else 's',color='#28598F' if ds=='MNIST' else '#B55722',capsize=4)
 ax.axvline(0,color='gray',lw=1,ls='--');ax.set_yticks(range(4),[('MNIST' if ds=='MNIST' else 'Fashion')+' · '+('graft' if a=='magnitude_graft' else 'SGDM') for ds,a in order]);ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.16)
fig.suptitle('Optimizer effects relative to AdamW within each experiment',fontsize=16,y=.995)
fig.text(.5,.005,'Eight shared settings averaged within seed; five paired seeds, 95% t intervals. Different datasets AND training schedules.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(PLOTS/'optimizer_effects_mnist_vs_fashion.png',dpi=160);plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(11.5,8.5))
for col,ds in enumerate(['MNIST','Fashion-MNIST']):
 for row,metric in enumerate(['mean_alpha','mean_abs_alpha_minus_2']):
  ax=axs[row,col]
  for arm,g in D[D.dataset==ds].groupby('arm'):ax.scatter(g[metric],g.test_accuracy_pct,label=names[arm],color=colors[arm],alpha=.7,s=28)
  ax.set_title(labels[ds]);ax.set_ylabel('Test accuracy (%)');ax.set_xlabel('Mean hidden-layer α' if row==0 else 'Mean hidden-layer |α − 2|');ax.grid(alpha=.15)
  if row==0:ax.axvline(2,color='gray',ls='--',lw=1)
axs[0,0].legend(fontsize=9,loc='best');fig.suptitle('Accuracy and spectral alignment: all shared-setting outcomes',fontsize=15)
fig.text(.5,.007,'40 outcomes per arm and dataset; five independent seeds. Vertical scales differ across datasets.',ha='center')
fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(PLOTS/'accuracy_alignment_comparison.png',dpi=160);plt.close(fig)

TR=pd.read_csv(P/'results/raw_loss_trajectories.csv')
fig,axs=plt.subplots(2,4,figsize=(14.5,7.3),sharex=True,sharey=True)
for ax,((b,m),g) in zip(axs.flat,TR.groupby(['batch_size','multiplier'])):
 for arm,h in g[g.epoch>=1].groupby('arm'):
  for seed,v in h.groupby('seed'):ax.plot(v.epoch,v.validation_loss,color=colors[arm],alpha=.45,lw=1)
 ax.set_title(f'Batch {b}, LR ×{m:g}');ax.set_xlabel('Epoch');ax.grid(alpha=.15)
for ax in axs[:,0]:ax.set_ylabel('Validation cross-entropy')
handles=[plt.Line2D([],[],color=colors[a],label=names[a],lw=2) for a in colors]
fig.legend(handles=handles,loc='upper center',ncol=3,bbox_to_anchor=(.5,.945),frameon=False)
fig.suptitle('Fashion-MNIST: actual validation loss, epochs 1–20',fontsize=16)
fig.text(.5,.008,'All 120 runs shown; initialization epoch 0 omitted for scale only and retained in the CSV. Fixed endpoint at epoch 20.',ha='center')
fig.tight_layout(rect=(0,.035,1,.90));fig.savefig(PLOTS/'fashion_validation_all_settings.png',dpi=160);plt.close(fig)
print('Comparison verified. Maximum interval difference:',maximum_error)
print(E[(E.scope=='common_eight_settings')&(E.treatment=='magnitude_graft')][['dataset','metric','mean','ci_low','ci_high']].to_string(index=False))
