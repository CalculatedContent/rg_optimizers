#!/usr/bin/env python3
"""Recalculate all published statistics from the pinned primary result JSONs.

Python standard library only. Do not substitute WeightWatcher's alpha_weighted.
Each model average includes all 18 unique matrices. Equal seed weights follow.
The five-seed 95% Student-t confidence interval uses df=4.
"""
import collections, csv, hashlib, json, math, pathlib, statistics
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'data'; RAW=OUT/'raw'
ARMS=['sgdm','native','modified']; SEEDS=list(range(1340,1345))
T95=2.7764451051977987
METRICS=['alpha','validation_ce','validation_ppl','validation_accuracy']
def stats(x):
    x=list(x); n=len(x); assert n==5,n
    m=statistics.mean(x); sd=statistics.stdev(x); hw=T95*sd/math.sqrt(n)
    return dict(n=n,mean=m,sd=sd,sem=sd/math.sqrt(n),ci95_half=hw,ci95_low=m-hw,ci95_high=m+hw)
def writecsv(name,rows):
    p=OUT/name
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def family(name):
    if name=='transformer.wte.weight':return 'Token embedding'
    if name=='transformer.wpe.weight':return 'Position embedding'
    if '.attn.c_attn.' in name:return 'QKV'
    if '.attn.c_proj.' in name:return 'Attention output'
    if '.mlp.c_fc.' in name:return 'MLP expansion'
    if '.mlp.c_proj.' in name:return 'MLP projection'
    raise AssertionError(name)
perseed=[]; matrices=[]; familyrows=[]; audit=[]; runtimes=[]
for arm in ARMS:
    paths=sorted((RAW/arm/'runs').rglob('result.json'))
    assert len(paths)==5,(arm,len(paths))
    seen=[]
    for path in paths:
        d=json.loads(path.read_text());seed=d['seed'];seen.append(seed)
        assert d['completed'] is True
        assert d['endpoint']['step']==20000
        assert [h['step'] for h in d['history']]==list(range(0,20001,500))
        assert d['identity']['seed']==seed
        ep=RAW/arm/'runs'/f'seed_{seed}'/('github_execution.json' if arm=='modified' else 'execution.json')
        meta=json.loads(ep.read_text());run_id=meta['github']['GITHUB_RUN_ID']
        expected={'sgdm':'45386899611','native':'45393623223','modified':'45375423863'}[arm]
        assert str(run_id)==expected,(arm,seed,meta)
        runtimes.append(dict(arm=arm,seed=seed,training_seconds=d['training_seconds']))
        for h in d['history']:
            step=h['step'];sp=h['alpha'];assert sp['unique_matrices']==18
            assert sp['successful_fits']==18
            fits=sp['matrices']; assert len(fits)==18
            names=[m['matrix'] for m in fits]
            assert len(set(names))==18
            assert 'lm_head.weight' not in names
            assert names.count('transformer.wte.weight')==1
            vals=[];rows=[]
            for m in fits:
                assert len(m['fits'])==1
                f=m['fits'][0]; assert f['status']=='success'
                a=float(f['alpha']);D=float(f['D']);tail=float(f['num_pl_spikes'])
                assert math.isfinite(a) and math.isfinite(D) and tail>0 and D>=0
                w=math.sqrt(tail)/(D+0.02)
                vals.append((a,w))
                name=m['matrix']
                rows.append(dict(arm=arm,seed=seed,step=step,matrix=name,family=family(name),block=int(name.split('.')[2])+1 if name.startswith('transformer.h.') else '',rows=m['shape'][0],columns=m['shape'][1],alpha=a,D=D,num_pl_spikes=tail,weight=w))
            total=sum(w for a,w in vals)
            avg=sum(a*w for a,w in vals)/total
            var=sum(w*(a-avg)**2 for a,w in vals)/total
            for r in rows:r['weight_fraction']=r['weight']/total
            matrices.extend(rows)
            for fam in sorted(set(r['family'] for r in rows)):
                fr=[r for r in rows if r['family']==fam]
                fw=sum(r['weight'] for r in fr)
                familyrows.append(dict(arm=arm,seed=seed,step=step,family=fam,n_matrices=len(fr),alpha=sum(r['alpha']*r['weight'] for r in fr)/fw,weight_fraction=fw/total))
            val=h['validation']; ce=val['loss_nats_per_character'];ppl=val['perplexity']
            assert math.isclose(ppl,math.exp(ce),rel_tol=1e-12)
            assert math.isclose(h['validation_loss'],ce,rel_tol=1e-12)
            if step==20000:
                assert val==d['endpoint']['metrics']['validation']
            perseed.append(dict(arm=arm,seed=seed,step=step,alpha=avg,validation_ce=ce,validation_ppl=ppl,validation_accuracy=val['next_character_accuracy'],weighted_layer_variance=var,weighted_layer_sd=math.sqrt(var),n_matrices=18))
        audit.append(dict(arm=arm,seed=seed,complete=True,n_measurements=len(d['history']),n_matrix_fits=sum(len(h['alpha']['matrices']) for h in d['history']),run_id=run_id,design_sha256=d['design_sha256']))
    assert sorted(seen)==SEEDS
    if arm!='modified':
        status=json.loads((RAW/arm/'analysis/status.json').read_text())
        assert status['completed'] and status['n_seeds']==5 and status['n_measurements']==205 and status['n_fits']==3690
        recorded=list(csv.DictReader((RAW/arm/'analysis/per_seed.csv').open()))
        for rr in recorded:
            r=next(x for x in perseed if x['arm']==arm and x['seed']==int(rr['seed']) and x['step']==int(rr['step']))
            for metric in METRICS:
                assert math.isclose(r[metric],float(rr[metric]),rel_tol=2e-14,abs_tol=2e-14),(arm,metric,r,rr)
    else:
        status=json.loads((RAW/arm/'cohort_status.json').read_text())
        summ=json.loads((RAW/arm/'analysis/summary.json').read_text())
        assert status['complete_five_seeds'] and status['completed_seeds']==5
        assert summ['complete_five_seeds'] and summ['n_completed']==5
        cohort=json.loads((RAW/arm/'analysis/execution_cohort.json').read_text())
        assert cohort
        recorded=list(csv.DictReader((RAW/arm/'analysis/model_alpha_history.csv').open()))
        assert len(recorded)==205
        for rr in recorded:
            r=next(x for x in perseed if x['arm']==arm and x['seed']==int(rr['seed']) and x['step']==int(rr['step']))
            for metric,col in [('alpha','confidence_tail_weighted_alpha'),('validation_ce','validation_ce'),('validation_ppl','validation_ppl')]:
                assert math.isclose(r[metric],float(rr[col]),rel_tol=2e-14,abs_tol=2e-14),(arm,metric,r,rr)
trajectory=[]
for arm in ARMS:
    for step in range(0,20001,500):
        group=[r for r in perseed if r['arm']==arm and r['step']==step]
        row=dict(arm=arm,step=step,n_seeds=5)
        for metric in METRICS:
            for k,v in stats(r[metric] for r in group).items():
                if k!='n':row[f'{metric}_{k}']=v
        # This is matrix dispersion, not seed uncertainty.
        row['mean_weighted_layer_sd']=statistics.mean(r['weighted_layer_sd'] for r in group)
        trajectory.append(row)
layerstats=[]
for arm in ARMS:
    for step in (2000,20000):
        ms=[r for r in matrices if r['arm']==arm and r['step']==step]
        for name in sorted(set(r['matrix'] for r in ms)):
            g=[r for r in ms if r['matrix']==name];row=dict(arm=arm,step=step,matrix=name,family=g[0]['family'],block=g[0]['block'])
            row.update({f'alpha_{k}':v for k,v in stats(r['alpha'] for r in g).items()})
            row['mean_weight_fraction']=statistics.mean(r['weight_fraction'] for r in g)
            layerstats.append(row)
famstats=[]
for arm in ARMS:
    for step in range(0,20001,500):
        for fam in sorted(set(r['family'] for r in familyrows)):
            g=[r for r in familyrows if r['arm']==arm and r['step']==step and r['family']==fam]
            row=dict(arm=arm,step=step,family=fam,n_matrices=g[0]['n_matrices'])
            row.update({f'alpha_{k}':v for k,v in stats(r['alpha'] for r in g).items()})
            row['mean_weight_fraction']=statistics.mean(r['weight_fraction'] for r in g)
            famstats.append(row)
paired=[]
for comparator in ('native','sgdm'):
    for step in (2000,20000):
        for metric in METRICS:
            a={r['seed']:r[metric] for r in perseed if r['arm']=='modified' and r['step']==step}
            b={r['seed']:r[metric] for r in perseed if r['arm']==comparator and r['step']==step}
            ds=[a[s]-b[s] for s in SEEDS]
            paired.append(dict(contrast='modified minus '+comparator,step=step,metric=metric,**stats(ds),all_negative=all(x<0 for x in ds),per_seed_differences=dict(zip(SEEDS,ds))))
writecsv('per_seed.csv',perseed)
writecsv('trajectory.csv',trajectory)
writecsv('matrix_fits.csv',matrices)
writecsv('matrix_endpoint_summary.csv',layerstats)
writecsv('family_per_seed.csv',familyrows)
writecsv('family_summary.csv',famstats)
writecsv('training_runtime.csv',runtimes)
endpoints=[r for r in trajectory if r['step'] in (2000,20000)]
writecsv('endpoint_summary.csv',endpoints)
summary=dict(estimator='sum ordinary fitted alpha * sqrt(num_pl_spikes)/(D+0.02) / sum weights; all18unique matrices; no matrix-size weighting; tied token/output embedding once',seed_aggregation='Five equally weighted seed model averages; sample SD(ddof1);95% Student-t CI(df4); bands are pointwise, not simultaneous',cohorts=json.loads((OUT/'cohorts.json').read_text()),n_runs=len(audit),n_model_measurements=len(perseed),n_matrix_fits=len(matrices),checks=audit,endpoints=endpoints,paired_comparisons=paired,training_seconds_by_arm={a:dict(mean=statistics.mean(r['training_seconds'] for r in runtimes if r['arm']==a),min=min(r['training_seconds'] for r in runtimes if r['arm']==a),max=max(r['training_seconds'] for r in runtimes if r['arm']==a)) for a in ARMS})
(OUT/'verified_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(n_runs=len(audit),n_measurements=len(perseed),n_fits=len(matrices),endpoints=endpoints,paired_comparisons=paired),indent=2))

