"""python analyze.py RUN_DIRECTORY: immutable records -> CSV and scientific plots."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = Path(sys.argv[1]); out = root / 'summaries'; out.mkdir(exist_ok=True)
m = pd.DataFrame([json.loads(p.read_text()) for p in sorted((root/'metrics').glob('*.json'))])
if m.empty: raise SystemExit('No measurements yet')
m.to_csv(out/'metrics.csv',index=False)
for metric in ('nll','perplexity','token_error'):
 fig, ax=plt.subplots()
 for split in ('train','val','test'): ax.plot(m.tokens_seen,m[f'{split}_{metric}'],label=split)
 ax.set(xlabel='Token presentations',ylabel=metric); ax.legend(); fig.tight_layout(); fig.savefig(out/f'{metric}_vs_tokens.png'); plt.close(fig)
rows=[r for p in sorted((root/'ww_metrics').glob('*.json')) for r in json.loads(p.read_text())['records']]
if not rows: raise SystemExit('Scalar plots saved; no WW snapshots yet')
w=pd.DataFrame(rows); w.to_csv(out/'ww_metrics.csv',index=False)
summary=[]
for variant in ('alpha_raw','alpha_clip_xmax'):
 grouped=w.groupby('tokens_seen')[variant].agg(['mean','min','std']).reset_index()
 for stat in ('mean','min'):
  fig,ax=plt.subplots(); ax.plot(grouped.tokens_seen,grouped[stat],'o-'); ax.set(xlabel='Token presentations',ylabel=f'{stat} {variant}'); fig.tight_layout(); fig.savefig(out/f'{stat}_{variant}_vs_tokens.png'); plt.close(fig)
  if variant=='alpha_raw':
   paired=grouped.merge(m[['tokens_seen','test_token_error']],on='tokens_seen',validate='one_to_one').dropna()
   fig,ax=plt.subplots(); ax.scatter(paired[stat],paired.test_token_error)
   if len(paired)>2 and paired[stat].nunique()>1:
    slope,intercept=np.polyfit(paired[stat],paired.test_token_error,1); x=np.sort(paired[stat]); ax.plot(x,intercept+slope*x)
   ax.set(xlabel=f'{stat} raw alpha',ylabel='Test token error'); fig.tight_layout(); fig.savefig(out/f'test_error_vs_{stat}_raw.png'); plt.close(fig)
 fig,axes=plt.subplots(3,2,figsize=(13,12))
 for ax,kind in zip(axes.flat,('Q','K','V','O','MLP_IN','MLP_OUT')):
  for name,g in w[w.matrix_type==kind].groupby('matrix_name'):
   g=g.sort_values('tokens_seen'); ax.plot(g.tokens_seen,g[variant],label=name)
  ax.axhline(2,color='gray',ls=':'); ax.set(title=kind,xlabel='Token presentations',ylabel=variant); ax.legend(fontsize=6,ncol=3)
 fig.tight_layout(); fig.savefig(out/f'all_layers_{variant}.png'); plt.close(fig)
 for name,g in w.groupby('matrix_name'):
  g=g.sort_values('tokens_seen').dropna(subset=[variant]); recent=g.tail(5)
  if g.empty: continue
  slope=np.polyfit(recent.tokens_seen,recent[variant],1)[0] if len(recent)>1 else np.nan
  summary.append({'matrix_name':name,'variant':variant,'initial':g[variant].iloc[0],'latest':g[variant].iloc[-1], 'minimum':g[variant].min(),'delta':g[variant].iloc[-1]-g[variant].iloc[0], 'recent_slope_per_token':slope,'recent_variance':recent[variant].var()})
pd.DataFrame(summary).to_csv(out/'matrix_summary.csv',index=False)
print(out)
