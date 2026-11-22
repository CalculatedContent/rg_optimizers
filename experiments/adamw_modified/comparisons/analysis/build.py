"""Build dataset-specific weighted analyses and a reusable campaign comparison.

Reads frozen endpoints only. No training, network access, fitting or selection.
"""
from pathlib import Path
import argparse,hashlib,json,sys,os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from weighted import KEY,paired_contributions,seed_scores,interval,performance_scores

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[2]
COLORS={'magnitude_graft':'#d66b23','sgdm':'#2c8760'}
NAMES={'magnitude_graft':'Magnitude graft','sgdm':'SGDM'}

def writej(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def hashes(path):
    b=path.read_bytes()
    return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
            'git_blob_sha':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()}

def source_paths(reg):
    return sorted(set([d['status_source'] for d in reg['datasets']]+
        [e['source'] for d in reg['datasets'] for e in d['endpoints']]+
        list(reg.get('verification_references',{}).values())))

def normalized(dataset,endpoint):
    frame=pd.read_csv(REPO/endpoint['source'])
    frame=frame[frame.epoch==endpoint['epoch']].copy()
    if frame.duplicated(KEY+['arm']).any():raise ValueError('Duplicate endpoint')
    if set(frame.seed)!=set(endpoint['seeds']):raise ValueError('Unexpected seed set')
    if 'status' not in frame:
        status=pd.read_csv(REPO/dataset['status_source'])
        if dataset.get('status_cohort'):
            status=status[status.cohort==dataset['status_cohort']]
        status=status[KEY+['arm',dataset['status_column']]].rename(columns={dataset['status_column']:'status'})
        frame=frame.merge(status,on=KEY+['arm'],how='left',validate='one_to_one')
        if frame.status.isna().any():raise ValueError('Missing model-status record')
    if 'supported_hidden_layers' not in frame:
        frame['supported_hidden_layers']=sum(frame['supported_'+l].astype(int) for l in endpoint['layers'])
    frame['dataset']=dataset['id'];frame['endpoint_id']=endpoint['id']
    frame['dependence_group']=endpoint['dependence_group']
    return frame

def compute(dataset,endpoint,reg):
    frame=normalized(dataset,endpoint)
    allowed={(x['batch_size'],x['multiplier']) for x in reg['shared_settings']}
    mask=np.array([(b,m) in allowed for b,m in zip(frame.batch_size,frame.multiplier)])
    frame['in_shared_settings']=mask
    shared=frame[mask].copy()
    arms=[reg['baseline']]+reg['treatments']
    if set(shared.arm)!=set(arms):raise ValueError('Unexpected arms')
    expected={(seed,b,m,arm) for seed in endpoint['seeds'] for b,m in allowed for arm in arms}
    actual=set(shared[KEY+['arm']].itertuples(index=False,name=None))
    if actual!=expected:raise ValueError('Incomplete shared setting grid')
    cohorts=[('shared_settings',shared)]
    if endpoint.get('all_original_settings'):cohorts.append(('all_original_settings',frame))
    summaries=[];seeds=[];contributions=[];performance=[];performance_seed=[]
    common={'dataset':dataset['id'],'endpoint_id':endpoint['id'],'epoch':endpoint['epoch']}
    for cohort,cohort_frame in cohorts:
        for arm in reg['treatments']:
            for weighting in reg['weightings']:
                c=paired_contributions(cohort_frame,reg['baseline'],arm,endpoint['layers'],weighting,reg['target_alpha'])
                info=common|dict(cohort=cohort,arm=arm,weighting=weighting)
                contributions.extend([info|x for x in c.to_dict('records')])
                scopes=[('all_settings',c)]+[(f'b{b}_lr{m:g}',g) for (b,m),g in c.groupby(['batch_size','multiplier'])]
                for scope,cf in scopes:
                    for label,lf in [('hidden_combined',cf)]+[(l,cf[cf.layer==l]) for l in endpoint['layers']]:
                        scores=seed_scores(lf);desc=info|dict(scope=scope,layer=label)
                        seeds.extend([desc|x for x in scores.to_dict('records')])
                        summary=interval(scores.improvement)
                        summaries.append(desc|summary|dict(
                            weighted_baseline_distance=scores.weighted_baseline_distance.mean(skipna=False),
                            weighted_treatment_distance=scores.weighted_treatment_distance.mean(skipna=False),
                            functional_failed_pairs=int(((lf.baseline_status!='USABLE')|(lf.treatment_status!='USABLE')).groupby([lf.seed,lf.batch_size,lf.multiplier]).any().sum())))
            scopes=[('all_settings',cohort_frame)]+[(f'b{b}_lr{m:g}',g) for (b,m),g in cohort_frame.groupby(['batch_size','multiplier'])]
            for scope,cf in scopes:
                scores=performance_scores(cf,reg['baseline'],arm)
                desc=common|dict(cohort=cohort,scope=scope,arm=arm)
                performance_seed.extend([desc|x for x in scores.to_dict('records')])
                for metric in ['accuracy_change_pp','ce_change']:
                    performance.append(desc|dict(metric=metric)|interval(scores[metric]))
    return {'models':frame,'weighted':pd.DataFrame(summaries),'seed_scores':pd.DataFrame(seeds),
            'contributions':pd.DataFrame(contributions),'performance':pd.DataFrame(performance),
            'performance_seed_scores':pd.DataFrame(performance_seed)}

def concat(bundles):
    return {name:pd.concat([b[name] for b in bundles],ignore_index=True) for name in bundles[0]}

def fmt(row):
    if row['status']!='VALID':return 'undefined'
    return f'{row["mean"]:+.5f} [{row.ci_low:+.5f}, {row.ci_high:+.5f}]'

def table(bundle,reg,cohort='shared_settings'):
    s=bundle['weighted'];p=bundle['performance']
    lines=['| Endpoint | Treatment | Weighted FC1 improvement | Weighted FC2 improvement | Combined improvement | Accuracy change (pp) | CE change |',
           '|---|---|---:|---:|---:|---:|---:|']
    for eid in bundle['models'].endpoint_id.unique():
        for arm in reg['treatments']:
            q=s[(s.endpoint_id==eid)&(s.arm==arm)&(s.cohort==cohort)&(s.scope=='all_settings')&(s.weighting==reg['primary_weighting'])]
            if q.empty:continue
            perf=p[(p.endpoint_id==eid)&(p.arm==arm)&(p.cohort==cohort)&(p.scope=='all_settings')].set_index('metric')
            vals=[eid,NAMES.get(arm,arm)]+[fmt(q[q.layer==l].iloc[0]) for l in ['fc1','fc2','hidden_combined']]+[fmt(perf.loc[m]) for m in ['accuracy_change_pp','ce_change']]
            lines.append('| '+' | '.join(vals)+' |')
    return lines

def plots(bundle,reg,dest):
    dest.mkdir(parents=True,exist_ok=True)
    s=bundle['weighted'];p=bundle['performance'];eids=list(bundle['models'].endpoint_id.unique())
    labels={e['id']:e['label'] for d in reg['datasets'] for e in d['endpoints']}
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for j,(kind,metric,title) in enumerate([('weighted','hidden_combined','Weighted alignment improvement'),('performance','accuracy_change_pp','Accuracy change (pp)'),('performance','ce_change','Cross-entropy change')]):
        ax=axes[j]
        for offset,arm in zip([-.12,.12],reg['treatments']):
            vals=[]
            for eid in eids:
                f=s if kind=='weighted' else p
                q=f[(f.endpoint_id==eid)&(f.arm==arm)&(f.cohort=='shared_settings')&(f.scope=='all_settings')]
                if kind=='weighted':q=q[(q.layer==metric)&(q.weighting==reg['primary_weighting'])]
                else:q=q[q.metric==metric]
                vals.append(q.iloc[0])
            mu=np.array([r['mean'] for r in vals]);lo=np.array([r.ci_low for r in vals]);hi=np.array([r.ci_high for r in vals])
            ax.errorbar(mu,np.arange(len(eids))+offset,xerr=np.array([mu-lo,hi-mu]),fmt='o',capsize=4,color=COLORS[arm],label=NAMES[arm])
        ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks(np.arange(len(eids)),[labels[i] for i in eids]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(title)
    axes[0].legend(fontsize=8);fig.suptitle('Eight shared settings; five seed scores per endpoint; paired 95% intervals\nPositive alignment/accuracy effects and negative CE effects are favorable; schedules differ')
    fig.savefig(dest/'weighted_alignment_and_performance.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,layer in zip(axes,['fc1','fc2']):
        for offset,arm in zip([-.12,.12],reg['treatments']):
            q=s[(s.arm==arm)&(s.cohort=='shared_settings')&(s.scope=='all_settings')&(s.weighting==reg['primary_weighting'])&(s.layer==layer)].set_index('endpoint_id').loc[eids]
            ax.errorbar(q['mean'],np.arange(len(eids))+offset,xerr=np.array([q['mean']-q.ci_low,q.ci_high-q['mean']]),fmt='o',capsize=4,color=COLORS[arm],label=NAMES[arm])
        ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks(np.arange(len(eids)),[labels[i] for i in eids]);ax.invert_yaxis();ax.set_title(layer.upper());ax.set_xlabel('Baseline-weighted improvement toward 2');ax.grid(axis='x',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Per-layer spectral rescue; paired 95% intervals\nIdentical baseline weights across treatments at each endpoint')
    fig.savefig(dest/'layer_weighted_alignment.png',dpi=160);plt.close(fig)

def save_results(bundle,dest):
    dest.mkdir(parents=True,exist_ok=True)
    for key,frame in bundle.items():frame.to_csv(dest/(key+'.csv'),index=False,float_format='%.17g')

METHOD='''For each matched layer, dA=|alpha_AdamW−2| and dT=|alpha_treatment−2|. The primary weight is w=dA. Within each seed, I=sum[w(dA−dT)]/sum[w]. Positive I means rescue toward 2. Baseline weights are identical across treatments. Combined scores sum both layers before normalization, rather than averaging separately normalized layer scores. Five seed scores are averaged equally; intervals use Student-t with 4 degrees of freedom. Model accuracy and CE contrasts remain ordinary paired seed means over settings, not severity-weighted performance claims.

The separately labeled sensitivity uses w=max(0,2−alpha_AdamW); treatment distance remains absolute so overshoot is penalized. Both weightings, both treatments and all settings are retained. No weighting exponent was tuned. Zero total weight or nonfinite scores remain undefined, never silently dropped. FC3 alpha is excluded because its fit has only 10 eigenvalues; unsupported hidden fits are labeled and retained.

This weighting was chosen after seeing results. Intervals are descriptive, unadjusted for multiple comparisons, and conditional on fixed datasets/splits/settings. Baseline-derived weights may amplify estimation noise/regression to the mean. The five seeds, not settings or layers, are the replicates. Weights are recomputed from native AdamW separately at each endpoint; cross-epoch score changes are not fixed-weight longitudinal effects. Fashion 20 and 60 are the same models and test set, not independent replication. The comparison is descriptive because dataset and learning-rate schedule differ.'''

def dataset_report(dataset,bundle,reg,verification):
    folder=REPO/dataset['analysis_folder'];save_results(bundle,folder/'campaign_results');plots(bundle,reg,folder/'campaign_plots')
    status=pd.read_csv(REPO/dataset['status_source']);status.to_csv(folder/'campaign_results/source_model_status_register.csv',index=False)
    lines=[f'# {dataset["name"]}: weighted spectral rescue','','Reproducible analysis of saved endpoints only. No new training or changed primary results.','','## Metric and inference','',METHOD,'','## Eight shared settings','','All values are mean differences [paired 95% interval]. Positive weighted improvement/accuracy is favorable; negative CE is favorable.','']+table(bundle,reg)
    lines+=['','![Weighted comparison](campaign_plots/weighted_alignment_and_performance.png)','','![Layer results](campaign_plots/layer_weighted_alignment.png)','','## Status and scope','']
    for endpoint in dataset['endpoints']:
        f=bundle['models'][bundle['models'].endpoint_id==endpoint['id']]
        lines.append(f'- {endpoint["id"]}: {len(f)} original endpoint rows; status counts {f.status.value_counts().to_dict()}; schedule: {endpoint["schedule"]}.')
    if dataset['id']=='mnist':
        lines+=['','The original MNIST grid has 135 runs. Its batch 128/LR×2 graft failed in all five original seeds; the separate seed 31006 replication contains one further failed graft and two usable controls. The entire 138-model status register is preserved. The common eight-setting comparison contains 120 usable models, and is conditional on the previously identified stable settings. The fifteen original outcomes at the excluded setting are retained below and in `campaign_results/models.csv`; none were erased. Two original collapsed models also have unsupported hidden fits.','','## Full original nine-setting sensitivity','','These scores include all 135 original models, including collapsed classifiers. A numerically favorable alpha score does not make a collapsed classifier usable. Do not substitute this spectral-only statistic for the functional failure register. The incomplete sixth-seed replication is never added to the five-seed intervals.','']+table(bundle,reg,'all_original_settings')
    else:
        lines+=['','Fashion has 120 unique model trajectories with two endpoints, not 240 independent models. The continuation reused all epoch 20 model/optimizer/RNG states. Prior interrupted/superseded Fashion passes are separate historical experiments and are not pooled here. Their original files remain unchanged.']
    rel=os.path.relpath(HERE,folder)
    lines+=['','## Reproduce and inspect','','Run `python reproduce.py` from this folder. This invokes the shared code and regenerates this dataset’s campaign outputs. Original source inputs and earlier reports are read-only.','',f'[General comparison]({rel}/README.md) · [Reusable analysis code]({rel}/analysis/build.py) · [Source hashes]({rel}/sources.lock.json) · [Numerical verification](campaign_results/Verification.json) · [All derived tables](campaign_results/).']
    (folder/'CAMPAIGN_REPORT.md').write_text('\n'.join(lines)+'\n')
    writej(folder/'campaign_results/Verification.json',verification)
    (folder/'reproduce.py').write_text('"""Rebuild this dataset analysis; no training."""\nimport runpy,sys\nfrom pathlib import Path\nscript=Path(__file__).resolve().parents[3]/"comparisons/analysis/build.py"\nsys.path.insert(0,str(script.parent))\nsys.argv=[str(script),"--dataset","'+dataset['id']+'"]\nrunpy.run_path(str(script),run_name="__main__")\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registry',type=Path,default=HERE/'experiments.json');parser.add_argument('--dataset');parser.add_argument('--refresh-lock',action='store_true');parser.add_argument('--source-commit');args=parser.parse_args()
    reg=json.loads(args.registry.read_text());lockpath=HERE/'sources.lock.json'
    if args.refresh_lock:
        if not args.source_commit:raise ValueError('--source-commit is required when refreshing the input lock')
        writej(lockpath,{'source_commit':args.source_commit,'sources':{p:hashes(REPO/p) for p in source_paths(reg)}})
    lock=json.loads(lockpath.read_text())
    if set(lock['sources'])!=set(source_paths(reg)):raise ValueError('Registry and source lock differ; inspect before explicit refresh')
    for path,expected in lock['sources'].items():
        if hashes(REPO/path)!=expected:raise ValueError('Frozen input changed: '+path)
    datasets=[d for d in reg['datasets'] if not args.dataset or d['id']==args.dataset]
    if not datasets:raise ValueError('Unknown dataset id')
    bundles={};checks={'source_hashes_match':True,'source_commit':lock['source_commit'],'new_training':False,'weighting_post_hoc':True,'datasets':{},'reference_maximum_error':{}}
    for dataset in datasets:
        b=concat([compute(dataset,e,reg) for e in dataset['endpoints']]);bundles[dataset['id']]=b
        checks['datasets'][dataset['id']]={'endpoint_rows':len(b['models']),'unique_trajectories':len(b['models'][KEY+['arm']].drop_duplicates()),'paired_seed_scores':len(b['seed_scores']),'all_expected_seed_scores_present':bool(b['weighted'].independent_seeds.eq(5).all()),'functional_failures_retained':int((b['models'].status!='USABLE').sum())}
        # Independent reconstruction uses NumPy arrays per seed; it does not call the score function.
        maxerr=0.
        for endpoint in dataset['endpoints']:
            f=b['models'];f=f[(f.endpoint_id==endpoint['id'])&f.in_shared_settings]
            for arm in reg['treatments']:
                for seed in endpoint['seeds']:
                    q=f[f.seed==seed];a=q[q.arm==reg['baseline']].sort_values(KEY);g=q[q.arm==arm].sort_values(KEY)
                    da=np.abs(a[['alpha_'+l for l in endpoint['layers']]].to_numpy()-reg['target_alpha']);dt=np.abs(g[['alpha_'+l for l in endpoint['layers']]].to_numpy()-reg['target_alpha'])
                    direct=(da*(da-dt)).sum()/da.sum()
                    ref=b['seed_scores']
                    ref=ref[(ref.endpoint_id==endpoint['id'])&(ref.seed==seed)&(ref.arm==arm)&(ref.cohort=='shared_settings')&(ref.scope=='all_settings')&(ref.layer=='hidden_combined')&(ref.weighting==reg['primary_weighting'])].iloc[0]
                    maxerr=max(maxerr,abs(direct-ref.improvement))
        if maxerr>1e-12:raise AssertionError('Independent NumPy score mismatch')
        checks['datasets'][dataset['id']]['independent_array_score_max_error']=maxerr
        if dataset['id']=='mnist':
            stable=pd.read_csv(REPO/reg['verification_references']['mnist_stable']).set_index(KEY+['arm'])
            actual=b['models'][b['models'].in_shared_settings].set_index(KEY+['arm']).reindex(stable.index)
            cols=['alpha_fc1','alpha_fc2','test_accuracy_pct','test_loss'];error=float(np.max(np.abs(stable[cols].to_numpy()-actual[cols].to_numpy())))
            if error>1e-10:raise AssertionError('Original stable subset mismatch')
            checks['reference_maximum_error']['mnist_saved_stable_endpoints']=error
        if dataset['id']=='fashion':
            old=pd.read_csv(REPO/reg['verification_references']['fashion_weighted']);new=b['weighted'];errors=[]
            for _,row in old.iterrows():
                q=new[(new.epoch==row.epoch)&(new.arm==row.arm)&(new.weighting==row.weighting)&(new.layer==row.layer)&(new.cohort=='shared_settings')&(new.scope=='all_settings')].iloc[0]
                errors.extend([abs(q['mean']-row.improvement),abs(q.ci_low-row.ci_low),abs(q.ci_high-row.ci_high)])
            error=max(errors)
            if error>1e-10:raise AssertionError('Existing Fashion weighted results changed')
            checks['reference_maximum_error']['fashion_saved_weighted_results']=error
    checks['passed']=True
    for dataset in datasets:dataset_report(dataset,bundles[dataset['id']],reg,checks)
    if not args.dataset:
        combined=concat(list(bundles.values()));save_results(combined,HERE/'results');plots(combined,reg,HERE/'plots');writej(HERE/'results/Verification.json',checks)
        lines=['# Cross-experiment weighted spectral comparison','','This folder compares the original MNIST experiment and the verified Fashion-MNIST endpoints. It is an extensible analysis registry, not a training launcher. All raw historical experiments, failure records and original primary analyses remain unchanged.','','## RG paper results','','[English results and plots](rg_paper/RESULTS.md) · [Tiếng Việt](rg_paper/RESULTS.vi.md) · [Run the reproduction](rg_paper/README.md)','','## Dataset folders','']
        for d in datasets:
            rel=os.path.relpath(REPO/d['analysis_folder'],HERE);lines.append(f'- [{d["name"]}: weighted report, code entry point and plots]({rel}/CAMPAIGN_REPORT.md)')
        lines+=['','## Shared-setting results','','Eight settings × five seeds × three arms per endpoint. All values are mean [paired 95% interval]. The same native-AdamW baseline weights are used for graft and SGDM.','']+table(combined,reg)
        lines+=['','![Weighted alignment and performance](plots/weighted_alignment_and_performance.png)','','![Layer comparisons](plots/layer_weighted_alignment.png)','','## Interpretation','','MNIST shows positive weighted rescue for the graft on the eight shared stable settings. Fashion epoch 60 shows FC1 rescue but FC2 deterioration; the combined graft interval includes zero. SGDM improves the combined weighted score in both datasets. A positive spectral result is not an accuracy or CE claim: the paired performance columns must be read separately.','','MNIST ran 120 epochs; Fashion initially ran 20 and then continued the same states to 60 using a different low-rate schedule. Dataset and schedule effects cannot be separated here. Fashion 20 and 60 are dependent endpoints, not separate replications. MNIST’s shared eight-setting subset was previously chosen after discovering the failed batch 128/LR×2 graft setting. The MNIST report separately retains the full135-run grid and all138 original-plus-replication status records; full-grid spectral summaries never rehabilitate collapsed classifiers.','','## Metric and inference','',METHOD,'','## Reproduction','','From the repository root:','','```bash','python -m pip install -r experiments/adamw_modified/comparisons/requirements.txt','python -m unittest discover -s experiments/adamw_modified/comparisons/tests -v','python experiments/adamw_modified/comparisons/analysis/build.py','```','','A dataset folder also provides `reproduce.py`. Input SHA256 and Git-blob hashes are pinned in [sources.lock.json](sources.lock.json). The code fails on changed inputs, duplicate/missing pairs, unexpected seed grids or disagreement with the existing Fashion results. Zero-weight/nonfinite scores are explicitly undefined, never silently dropped. Sources are read only; no training package or network access is used.','','## Adding another completed experiment','','1. Preserve its source endpoints and model-status register in a new dated experiment folder. Export one row per seed/setting/arm/endpoint with columns `seed,batch_size,multiplier,arm,epoch,alpha_fc1,alpha_fc2,test_accuracy_pct,test_loss,status` plus hidden-fit support counts/flags. Use `USABLE`/`FAILED` as functional labels independently of alpha.','2. Add its dataset/endpoint entry to [experiments.json](experiments.json), with source paths, output analysis folder, seed list, hidden layers, schedule, and dependence group. Use the same dependence group for continuations of the same models. Do not silently pool incomplete grids or invent independent seeds.','3. Inspect the new source files and run `analysis/build.py --refresh-lock --source-commit <reviewed_Git_commit>` once to pin their current hashes and rebuild. Refreshing is explicit; normal builds refuse changed inputs. For a different setting grid or architecture, define the comparison design before interpreting results; the present registry is a matched MLP grid.','4. Inspect the new plots, verification and failure registers before committing. New entries do not start or authorize training.','','[All numerical tables](results/) · [Verification](results/Verification.json) · [Shared calculation module](analysis/weighted.py) · [Scientific invariant tests](tests/test_weighted.py).']
        (HERE/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
