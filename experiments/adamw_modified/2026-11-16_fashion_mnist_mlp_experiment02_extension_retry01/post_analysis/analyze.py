"""Exploratory read-only analysis of frozen RETRY01 archives. No network/training."""
import argparse, json, zipfile, io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from scipy.stats import t, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pa=argparse.ArgumentParser();pa.add_argument('--artifacts',type=Path,required=True);pa.add_argument('--output',type=Path,required=True);args=pa.parse_args()
P=args.artifacts;O=args.output;O.mkdir(exist_ok=True,parents=True)
E=pd.read_csv(P/'Fashion-Retry01-Audit/results/endpoints20_and60.csv')
ARMS=['adamw','magnitude_graft','sgdm'];COLORS=['#2465a8','#d66b23','#2c8760']
def interval(x):
 x=np.asarray(x,float);assert len(x)==5 and np.isfinite(x).all();m=x.mean();h=t.ppf(.975,4)*x.std(ddof=1)/np.sqrt(5);return dict(mean=m,ci_low=m-h,ci_high=m+h)
def contrasts(df,metrics,groups):
 out=[]
 for key,g in df.groupby(groups,dropna=False):
  if not isinstance(key,tuple):key=(key,)
  for arm in ARMS[1:]:
   for m in metrics:
    a=g.pivot(index='seed',columns='arm',values=m);out.append({**dict(zip(groups,key)),'arm':arm,'metric':m,**interval(a[arm]-a.adamw)})
 return pd.DataFrame(out)
metrics=['alpha_fc1','alpha_fc2','distance_fc1','distance_fc2','mean_abs_alpha_minus_2','test_accuracy_pct','test_loss']
effects=contrasts(E,metrics,['epoch','batch_size','multiplier']);effects.to_csv(O/'setting_effects.csv',index=False)
E.groupby(['epoch','batch_size','multiplier','arm'])[metrics].mean().to_csv(O/'setting_means.csv')
rows=[];updates=[];flows=[];decomp=[];preds={}
for seed in range(31001,31006):
 with zipfile.ZipFile(P/f'Fashion-Retry01-Seed{seed}'/f'Fashion_Extension_Seed_{seed}.zip') as z:
  for n in [n for n in z.namelist() if n.startswith('runs/') and n.endswith('/done.json')]:
   base=n.rsplit('/',1)[0];d=json.loads(z.read(n));s=d['spec'];parent=json.loads(z.read(base+'/parent_done.json'));hist=json.loads(z.read(base+'/history.json'))
   val=np.load(io.BytesIO(z.read(base+'/final_validation_logits.npz')));test=np.load(io.BytesIO(z.read(base+'/final_test_logits.npz')))
   vx=val['logits'].astype(float);vy=val['labels'];x=test['logits'].astype(float);y=test['labels'];ii=np.arange(len(y))
   def ce(logT):
    q=vx/np.exp(logT);return (logsumexp(q,axis=1)-q[np.arange(len(vy)),vy]).mean()
   fit=minimize_scalar(ce,bounds=(-4,4),method='bounded',options={'xatol':1e-9});T=np.exp(fit.x);assert fit.success and abs(fit.x)<3.99
   loss=logsumexp(x,axis=1)-x[ii,y];q=x/T;cal=logsumexp(q,axis=1)-q[ii,y];pr=x.argmax(1);wrong=pr!=y;conf=np.exp(x.max(1)-logsumexp(x,axis=1));assert np.array_equal(pr,q.argmax(1))
   correct_margin=x[ii,y]-np.max(np.where(np.eye(10,dtype=bool)[y],-np.inf,x),axis=1)
   r={**s,'temperature':T,'test_loss':loss.mean(),'calibrated_test_loss':cal.mean(),'calibration_ce_change':cal.mean()-loss.mean(),'test_accuracy_pct':100*(~wrong).mean(),'logit_centered_rms':np.sqrt(np.mean((x-x.mean(1,keepdims=True))**2)),'wrong_confidence':conf[wrong].mean(),'wrong_mean_loss':loss[wrong].mean(),'loss_q99':np.quantile(loss,.99),'confident_errors':(wrong&(conf>.9)).sum(),'correct_margin_mean':correct_margin[~wrong].mean(),'incorrect_margin_mean':correct_margin[wrong].mean(),'val_loss_20':parent['validation']['loss'],'val_loss_60':d['validation']['loss'],'train_loss_60':d['final_train']['loss'],'train_accuracy_60':100*d['final_train']['accuracy'],'val_loss_change':d['validation']['loss']-parent['validation']['loss'],'minimum_val_epoch_descriptive_only':min(hist,key=lambda h:h['validation']['loss'])['epoch']}
   for layer in ['fc1','fc2','fc3']:
    for ep,rr in [(20,parent),(60,d)]:
     spec=rr['spectra'][layer];r[f'{layer}_frob_{ep}']=np.sqrt(spec['norm']);r[f'{layer}_spectral_eigenvalue_{ep}']=spec['spectral_norm']
     for frac,v in spec['fixed_rank'].items():
      flows.append({**s,'epoch':ep,'layer':layer,'fraction':float(frac),'h_minus1_proxy':v['trace_log_residual'],'fixed_rank_alpha':v['alpha'],'ww_alpha':spec['alpha'],'ww_distance':abs(spec['alpha']-2)})
   rows.append(r);preds[(seed,s['batch_size'],s['multiplier'],s['arm'])]=(y,loss,wrong)
   for ep in range(21,61):
    a=np.load(io.BytesIO(z.read(base+f'/statistics/epoch_{ep:03}.npz')));v=a['values'].astype(float);cols=a['columns'].tolist()
    for li,layer in enumerate(a['layers'].tolist()):
     rr={**s,'epoch':ep,'layer':layer,'sampled_batches':len(v)}
     for ci,c in enumerate(cols):
      vals=v[:,li,ci];rr[c]=np.mean(vals) if np.isfinite(vals).all() else np.nan
     rr['relative_step_q95']=np.quantile(v[:,li,cols.index('relative_step')],.95)
     updates.append(rr)
 print('Analyzed seed',seed,flush=True)
for seed in range(31001,31006):
 for b,m in E[['batch_size','multiplier']].drop_duplicates().itertuples(index=False,name=None):
  y,a,wa=preds[(seed,b,m,'adamw')]
  for arm in ARMS[1:]:
   yy,l,w=preds[(seed,b,m,arm)];assert np.array_equal(y,yy)
   for label,mask in [('both_correct',~wa&~w),('both_wrong',wa&w),('corrected',wa&~w),('introduced',~wa&w)]:
    decomp.append(dict(seed=seed,batch_size=b,multiplier=m,arm=arm,category=label,count=int(mask.sum()),ce_contribution=float((l-a)[mask].sum()/len(y))))
R=pd.DataFrame(rows);U=pd.DataFrame(updates);F=pd.DataFrame(flows);D=pd.DataFrame(decomp)
R.to_csv(O/'model_diagnostics.csv',index=False);U.to_csv(O/'sampled_update_summaries.csv',index=False);F.to_csv(O/'fixed_rank_proxies.csv',index=False);D.to_csv(O/'paired_error_decomposition.csv',index=False)
cm=['temperature','test_loss','calibrated_test_loss','calibration_ce_change','logit_centered_rms','wrong_mean_loss','loss_q99','confident_errors','val_loss_change','fc1_frob_60','fc2_frob_60','fc3_frob_60']
rc=contrasts(R,cm,['batch_size','multiplier']);rc.to_csv(O/'diagnostic_setting_effects.csv',index=False)
pooled=R.groupby(['seed','arm'])[cm].mean().reset_index();pooled['scope']='all_settings';pc=contrasts(pooled,cm,['scope']);pc.to_csv(O/'diagnostic_overall_effects.csv',index=False)
dd=D.groupby(['seed','arm','category'])[['count','ce_contribution']].mean().reset_index();dc=[]
for (arm,c),g in dd.groupby(['arm','category']):
 for metric in ['count','ce_contribution']:dc.append(dict(arm=arm,category=c,metric=metric,**interval(g[metric])))
pd.DataFrame(dc).to_csv(O/'error_decomposition_intervals.csv',index=False)
# Setting-level heterogeneity is descriptive, not a reason to drop settings.
hetero=[]
for arm,g in E[E.epoch==60].groupby('arm'):
 for layer in ['fc1','fc2']:
  for seed,h in g.groupby('seed'):
   a=h['alpha_'+layer];hetero.append(dict(arm=arm,layer=layer,seed=seed,mean_alpha=a.mean(),mean_distance=(a-2).abs().mean(),distance_of_mean=abs(a.mean()-2),across_setting_sd=a.std(ddof=1),below_two=int((a<2).sum()),above_two=int((a>2).sum())))
pd.DataFrame(hetero).to_csv(O/'alpha_heterogeneity.csv',index=False)
# Descriptive within-seed correlations across the eight settings, not 40 IID samples.
corr=[]
for (seed,arm),g in R.groupby(['seed','arm']):
 for feature in ['fc3_frob_60','logit_centered_rms','temperature']:
  corr.append(dict(seed=seed,arm=arm,feature=feature,rho=spearmanr(g[feature],g.test_loss).statistic))
pd.DataFrame(corr).to_csv(O/'within_seed_descriptive_correlations.csv',index=False)

settings=sorted(E[['batch_size','multiplier']].drop_duplicates().itertuples(index=False,name=None));labels=[f'B{b} ×{m:g}' for b,m in settings]
fig,axs=plt.subplots(1,3,figsize=(13,5),layout='constrained')
for ax,metric,title in zip(axs,['test_accuracy_pct','test_loss','mean_abs_alpha_minus_2'],['Accuracy difference (pp)','Cross-entropy difference','Mean hidden distance difference']):
 for offset,arm,color in [(-.12,'magnitude_graft',COLORS[1]),(.12,'sgdm',COLORS[2])]:
  sub=effects[(effects.epoch==60)&(effects.arm==arm)&(effects.metric==metric)].set_index(['batch_size','multiplier']).loc[settings]
  mu=sub['mean'].values;ax.errorbar(mu,np.arange(8)+offset,xerr=np.array([mu-sub.ci_low.values,sub.ci_high.values-mu]),fmt='o',color=color,label=arm,capsize=3)
 ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks(np.arange(8),labels);ax.invert_yaxis();ax.set_title(title);ax.grid(alpha=.2)
axs[0].legend(fontsize=8);fig.suptitle('Fixed epoch60: optimizer minus AdamW; five paired seeds per setting\nAll eight settings retained; descriptive 95% t intervals');fig.savefig(O/'setting_effects.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for arm,color in zip(ARMS,COLORS):
 g=R[R.arm==arm];sg=g.groupby('seed');a=sg[['test_loss','calibrated_test_loss']].mean();means=a.mean();sem=a.std(ddof=1)/np.sqrt(5)*t.ppf(.975,4)
 axs[0].errorbar([0,1],means,yerr=sem,color=color,marker='o',label=arm,capsize=3)
 axs[1].scatter(g.fc3_frob_60,g.logit_centered_rms,color=color,alpha=.65,label=arm)
 axs[2].scatter(g.temperature,g.test_loss,color=color,alpha=.65)
axs[0].set_xticks([0,1],['Raw logits','Validation-calibrated']);axs[0].set_ylabel('Test cross-entropy');axs[0].legend(fontsize=8)
axs[1].set_xlabel('FC3 Frobenius norm');axs[1].set_ylabel('Centered logit RMS');axs[2].set_xlabel('Validation-fitted temperature');axs[2].set_ylabel('Raw test cross-entropy')
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('Exploratory calibration diagnostic at fixed epoch60\nPositive scalar temperature preserves every predicted class');fig.savefig(O/'calibration_and_norms.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for j,layer in enumerate(['fc1','fc2','fc3']):
 for arm,color in zip(ARMS,COLORS):
  g=U[(U.layer==layer)&(U.arm==arm)].groupby('epoch')[['relative_step','weight_norm']].mean()
  axs[0,j].plot(g.index,g.relative_step,color=color,label=arm);axs[1,j].plot(g.index,g.weight_norm,color=color)
 axs[0,j].set_title(layer.upper());axs[0,j].set_yscale('log');axs[1,j].set_xlabel('Epoch');axs[0,j].grid(alpha=.2);axs[1,j].grid(alpha=.2)
axs[0,0].set_ylabel('Sampled update norm / weight norm');axs[1,0].set_ylabel('Frobenius weight norm');axs[0,0].legend(fontsize=8)
fig.suptitle('Recorded continuation dynamics; all settings and seeds\nSampled optimizer updates exclude explicit decay; not total path lengths');fig.savefig(O/'update_dynamics.png',dpi=160);plt.close(fig)

summary={'models':len(R),'seeds':5,'new_training':False,'temperature_fit':'Minimize epoch60 validation CE over log T in [-4,4]; apply to saved test logits; no endpoint selection','calibration_predictions_unchanged':True,'all_settings_retained':True,'group_means':R.groupby('arm')[cm].mean().to_dict(orient='index'),'pooled_effects':pc.to_dict(orient='records'),'error_decomposition':dc,'proxy_caution':'Fixed-rank h-1 is a spectral tail-shape diagnostic; neither optimizer Jacobian logdet nor a verified RG scale-equivariance residual. Recorded norms alone do not identify those quantities.'}
(O/'Summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(R.groupby('arm')[['temperature','test_loss','calibrated_test_loss','fc3_frob_20','fc3_frob_60','logit_centered_rms']].mean().to_string())
print(pc[pc.metric.isin(['calibrated_test_loss','temperature'])].to_string(index=False))
