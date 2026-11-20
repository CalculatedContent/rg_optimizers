"""Verified checkpoint plotter. No synthetic scientific observations."""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REQUIRED={'dataset','optimizer','seed','epoch','layer','alpha','ks_d','n_tail','test_accuracy'}
def aggregate(df):
    missing=REQUIRED-set(df)
    if missing: raise ValueError(f'Missing: {sorted(missing)}')
    keys=['dataset','optimizer','seed','epoch']
    if df.duplicated(keys+['layer']).any(): raise ValueError('Duplicate layer')
    if not df.layer.isin(['FC1','FC2','FC3']).all() or not df.groupby(keys).layer.nunique().eq(3).all(): raise ValueError('Require FC1/FC2/FC3 per checkpoint')
    for k in ['alpha','ks_d','n_tail','test_accuracy','epoch']:
        df[k]=pd.to_numeric(df[k],errors='raise')
    if not np.isfinite(df[['alpha','ks_d','n_tail','test_accuracy','epoch']]).all().all(): raise ValueError('Nonfinite values')
    if (df.ks_d<0).any() or (df.n_tail<=0).any(): raise ValueError('Invalid fit data')
    if df.groupby(keys).test_accuracy.nunique().max()>1: raise ValueError('Inconsistent accuracy')
    df=df.copy()
    df['weight']=np.sqrt(df.n_tail)/(df.ks_d+.02)
    df['weighted_alpha']=df.weight*df.alpha
    a=df.groupby(keys,as_index=False).agg(weighted_alpha=('weighted_alpha','sum'),weight=('weight','sum'),test_accuracy=('test_accuracy','first'))
    a['alpha_bar']=a.weighted_alpha/a.weight
    return a
def main():
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--out',default='figures');args=p.parse_args()
    from pathlib import Path
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    a=aggregate(pd.read_csv(args.csv));a.to_csv(out/'verified_model_checkpoints.csv',index=False)
    for ds,g in a.groupby('dataset'):
        fig,ax=plt.subplots(figsize=(6.5,4))
        for opt,h in g.groupby('optimizer'): ax.scatter(h.alpha_bar,h.test_accuracy,s=14,alpha=.6,label=opt)
        ax.axvline(2,color='black',ls='--');ax.set(xlabel='Weighted model alpha',ylabel='Test accuracy',title=ds);ax.legend();fig.tight_layout();fig.savefig(out/f'{ds}_accuracy_alpha.pdf');plt.close(fig)
        for opt,h in g.groupby('optimizer'):
            fig,ax=plt.subplots(figsize=(6.5,4))
            for seed,z in h.groupby('seed'):
                z=z.sort_values('epoch');ax.plot(z.epoch,z.alpha_bar,marker='.',label=str(seed))
            ax.axhline(2,color='black',ls='--');ax.set(xlabel='Epoch',ylabel='Weighted model alpha',title=f'{ds}: {opt}');ax.legend();fig.tight_layout();fig.savefig(out/f'{ds}_{opt}_epoch_alpha.pdf');plt.close(fig)
if __name__=='__main__':main()
