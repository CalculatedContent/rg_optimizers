"""Post-hoc baseline-weighted alpha improvement; existing endpoints only."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
pa=argparse.ArgumentParser();pa.add_argument('--endpoints',type=Path,required=True);pa.add_argument('--output',type=Path,required=True);args=pa.parse_args()
e=pd.read_csv(args.endpoints);out=args.output;out.mkdir(parents=True,exist_ok=True)
keys=['seed','batch_size','multiplier'];records=[];seedrows=[];summary=[]
for ep in [20,60]:
 a=e[(e.epoch==ep)&(e.arm=='adamw')].set_index(keys)
 for arm in ['magnitude_graft','sgdm']:
  g=e[(e.epoch==ep)&(e.arm==arm)].set_index(keys);assert a.index.equals(g.index)
  for kind in ['absolute_baseline_distance','sub2_baseline_deficit']:
   rows=[]
   for ix,r in a.iterrows():
    for layer in ['fc1','fc2']:
     aa=r['alpha_'+layer];gg=g.loc[ix,'alpha_'+layer];da=abs(aa-2);dg=abs(gg-2)
     w=da if kind=='absolute_baseline_distance' else max(0,2-aa)
     rows.append(dict(zip(keys,ix))|dict(epoch=ep,arm=arm,weighting=kind,layer=layer,baseline_alpha=aa,treatment_alpha=gg,weight=w,baseline_distance=da,treatment_distance=dg,weighted_baseline=w*da,weighted_treatment=w*dg,weighted_improvement=w*(da-dg)))
   r=pd.DataFrame(rows);records+=rows
   for label,ff in [('hidden_combined',r),('fc1',r[r.layer=='fc1']),('fc2',r[r.layer=='fc2'])]:
    sums=ff.groupby('seed')[['weight','weighted_baseline','weighted_treatment','weighted_improvement']].sum();assert len(sums)==5 and (sums.weight>0).all()
    vals=sums.weighted_improvement/sums.weight;half=t.ppf(.975,4)*vals.std(ddof=1)/np.sqrt(5)
    summary.append(dict(epoch=ep,arm=arm,weighting=kind,layer=label,weighted_baseline_distance=(sums.weighted_baseline/sums.weight).mean(),weighted_treatment_distance=(sums.weighted_treatment/sums.weight).mean(),improvement=vals.mean(),ci_low=vals.mean()-half,ci_high=vals.mean()+half,independent_seeds=5))
    for seed,v in vals.items():seedrows.append(dict(epoch=ep,arm=arm,weighting=kind,layer=label,seed=seed,improvement=v,total_weight=sums.loc[seed,'weight']))
pd.DataFrame(records).to_csv(out/'paired_layer_contributions.csv',index=False);pd.DataFrame(seedrows).to_csv(out/'seed_scores.csv',index=False);s=pd.DataFrame(summary);s.to_csv(out/'weighted_effects.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
for ax,layer,title in zip(axes,['fc1','fc2','hidden_combined'],['FC1','FC2','Both hidden layers']):
 for y,arm,color in [(1,'magnitude_graft','#d66b23'),(0,'sgdm','#2c8760')]:
  r=s[(s.epoch==60)&(s.arm==arm)&(s.weighting=='absolute_baseline_distance')&(s.layer==layer)].iloc[0]
  ax.errorbar(r.improvement,y,xerr=[[r.improvement-r.ci_low],[r.ci_high-r.improvement]],fmt='o',color=color,capsize=5)
 ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks([0,1],['SGDM','Magnitude graft']);ax.set_ylim(-.5,1.5);ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Weighted improvement toward 2')
fig.suptitle('Epoch60: weight = native AdamW distance from 2\nPositive favors treatment; five seed-level scores with 95% t intervals');fig.savefig(out/'weighted_alignment.png',dpi=160);plt.close(fig)
lines=['# Baseline-weighted alpha improvement','','This exploratory metric was specified after inspecting the original experiment, following Charles’s clarification that rescuing layers far below2 matters more than small movements near2. All120 models and both hidden layers remain in the source data. This does not replace the original primary metric or create independent confirmation. No training, tuning or endpoint selection was performed.','','For each paired layer and setting, let d_A=|alpha_AdamW−2| and d_T=|alpha_treatment−2|. The primary weight is w=d_A. Within each seed, compute I_s=sum[w(d_A−d_T)]/sum[w]. Positive I means improvement. The treatment uses the **same AdamW-derived weights**; never recompute weights from treatment alpha. Average the five seed scores equally and use Student-t intervals with4 degrees of freedom. The combined score sums FC1 and FC2 contributions before normalization, rather than averaging the two separately normalized layer scores.','','Example:1.70→1.90 contributes0.30×0.20=+0.060 to the numerator.1.95→1.90 contributes0.05×(−0.05)=−0.0025. The first contribution is24 times larger in magnitude. A large worsening is still penalized.','','A separately labeled sensitivity analysis uses w=max(0,2−alpha_AdamW), giving zero weight to baseline layers already at or above2. Treatment distance remains absolute, so large overshoots are penalized. Neither formula was selected based on which produces favorable results. We do not sweep weighting exponents.','','## Epoch60: primary baseline-distance weighting','','| Treatment | Layer(s) | Weighted AdamW distance | Weighted treatment distance | Improvement [95% interval] |','|---|---|---:|---:|---:|']
for _,r in s[(s.epoch==60)&(s.weighting=='absolute_baseline_distance')].iterrows():lines.append(f'| {r.arm} | {r.layer} | {r.weighted_baseline_distance:.5f} | {r.weighted_treatment_distance:.5f} | {r.improvement:+.5f} [{r.ci_low:+.5f}, {r.ci_high:+.5f}] |')
lines+=['','![Weighted alignment](weighted_alignment.png)','','The graft improves weighted FC1 alignment, but worsens weighted FC2 alignment. Its combined hidden-layer effect is unresolved (interval includes zero). SGDM improves the combined weighted score. Thus weighting changes the interpretation of FC1, while it does not establish a joint graft benefit across both hidden layers. This metric addresses spectral rescue only; it does not establish accuracy or CE improvement.','','## Sub-2-only sensitivity at epoch60','','| Treatment | Layer(s) | Improvement [95% interval] |','|---|---|---:|']
for _,r in s[(s.epoch==60)&(s.weighting=='sub2_baseline_deficit')].iterrows():lines.append(f'| {r.arm} | {r.layer} | {r.improvement:+.5f} [{r.ci_low:+.5f}, {r.ci_high:+.5f}] |')
lines+=['','[All weighted results, including epoch20](weighted_effects.csv) · [Five-seed scores](seed_scores.csv) · [Every paired-layer contribution](paired_layer_contributions.csv).','','Weights are fixed across treatments **within each endpoint**, but separately computed at20 and60. Do not interpret their score difference as a fixed-weight longitudinal effect. The weights depend on observed AdamW alpha and can amplify baseline estimation noise/regression-to-the-mean; a future confirmatory analysis should prespecify the weighting and consider independently estimated baseline weights. Five seeds are the replicates; layers/settings are not independent replicates. CIs are descriptive and unadjusted for multiplicity. FC3 remains excluded because its alpha fit is unsupported.','','Reproduce with `python analyze.py --endpoints <endpoints20_and60.csv> --output <directory>`. The input is the unchanged parent experiment results file.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(s[(s.epoch==60)&(s.weighting=='absolute_baseline_distance')].to_string(index=False))
