"""Read-only audit of RETRY01 downloaded archives; no training or selection."""
from pathlib import Path
import json, zipfile, hashlib, io, collections
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t

P = Path(__file__).resolve().parent
A = P / 'Fashion-Retry01-Audit'
O = P / 'postcheck'
O.mkdir(exist_ok=True)
def sha(b): return hashlib.sha256(b).hexdigest()
def ci(x):
    x=np.asarray(x,dtype=float); half=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return [float(x.mean()),float(x.mean()-half),float(x.mean()+half)]
manifest=json.loads((A/'Source_Manifest.json').read_text())
end=pd.read_csv(A/'results/endpoints20_and60.csv')
completed=json.loads((A/'Completion.json').read_text())
assert completed['completed']==completed['planned']==120
assert len(end)==240 and not end.duplicated(['seed','batch_size','multiplier','arm','epoch']).any()
assert set(end.seed)==set(range(31001,31006))
assert end.groupby(['seed','arm','epoch']).size().eq(8).all()
checks={'source_hashes_verified':True,'inner_archive_hashes_verified':True,'checkpoint_hashes_verified':0,'endpoint_records_verified':0,'exact_replay_records_verified':0,'paired_initialization':True,'paired_minibatches':True,'functional_statuses':{},'logit_accuracy_max_error':0.,'logit_ce_max_error':0.,'graft_nonzero_records':0,'maximum_graft_norm_relative_error':0.,'spectra_restoration_exact':True}
fingerprints={}; statuses=collections.Counter(); tails=[]; logits={}; checkpoints=[]
for seed in range(31001,31006):
    f=P/f'Fashion-Retry01-Seed{seed}'/f'Fashion_Extension_Seed_{seed}.zip'
    meta=json.loads(f.with_suffix('.zip.json').read_text())
    assert f.stat().st_size==meta['bytes']
    with f.open('rb') as h: assert hashlib.file_digest(h,'sha256').hexdigest()==meta['sha256']
    with zipfile.ZipFile(f) as z:
        for name in ['run.py','report.py','Protocol.json']:
            assert sha(z.read(name))==manifest['files'][name]
        for name in [n for n in z.namelist() if n.startswith('runs/') and n.endswith('/done.json')]:
            r=json.loads(z.read(name)); base=name.rsplit('/',1)[0]; sp=r['spec']
            assert sp['seed']==seed and r['endpoint_epoch']==60 and r['start_epoch']==20 and r['stop_reason']=='fixed_epoch60'
            v=r['verification']; assert all(v[k] for k in ['passed','checkpoint_replay_exact','restored_validation_and_spectra_exact','optimizer_state_restored_exact','fixed_endpoint'])
            assert v['replay_origin']==21 and not any(v[k] for k in ['selection_used_test','selection_used_validation','selection_used_alpha'])
            checks['exact_replay_records_verified']+=1
            parent=json.loads(z.read(base+'/parent_done.json'))
            assert r['parent_checkpoint_sha256']==parent['checkpoint_hashes']['final.pt']
            for ck,h in r['checkpoint_hashes'].items():
                assert sha(z.read(base+'/'+ck))==h
                checks['checkpoint_hashes_verified']+=1
            checkpoints.append({'spec':sp,'checkpoint_hashes':r['checkpoint_hashes']})
            hist=json.loads(z.read(base+'/history.json'))
            assert [h['epoch'] for h in hist]==list(range(61))
            fp=(r['initial_hash'],[h['batch_order_sha256'] for h in hist]); key=(seed,sp['batch_size'],sp['multiplier'])
            if key in fingerprints: assert fp==fingerprints[key]
            fingerprints[key]=fp
            for epoch,rr in [(20,parent),(60,r)]:
                row=end[(end.seed==seed)&(end.batch_size==sp['batch_size'])&(end.multiplier==sp['multiplier'])&(end.arm==sp['arm'])&(end.epoch==epoch)].iloc[0]
                assert abs(row.test_accuracy_pct-100*rr['test']['accuracy'])<1e-12
                assert abs(row.test_loss-rr['test']['loss'])<1e-12
                for layer in ['fc1','fc2','fc3']:
                    assert abs(row['alpha_'+layer]-rr['spectra'][layer]['alpha'])<1e-12
                checks['endpoint_records_verified']+=1
            s1=np.load(io.BytesIO(z.read(base+'/spectra/epoch_060.npz')))
            s2=np.load(io.BytesIO(z.read(base+'/restored/spectra/epoch_060.npz')))
            for k in s1.files: assert np.array_equal(s1[k],s2[k],equal_nan=True)
            if sp['arm']=='magnitude_graft':
                for ep in range(21,61):
                    a=np.load(io.BytesIO(z.read(base+f'/statistics/epoch_{ep:03}.npz')))['values']
                    target,actual=a[:,:,3],a[:,:,4]; good=(target>0)&np.isfinite(target)&np.isfinite(actual)
                    checks['graft_nonzero_records']+=int(good.sum())
                    checks['maximum_graft_norm_relative_error']=max(checks['maximum_graft_norm_relative_error'],float(np.max(abs(actual[good]/target[good]-1))))
            a=np.load(io.BytesIO(z.read(base+'/final_test_logits.npz')))
            x=a['logits'].astype(float); y=a['labels']; loss=logsumexp(x,axis=1)-x[np.arange(len(y)),y]; pred=x.argmax(1); wrong=pred!=y; conf=np.exp(x.max(1)-logsumexp(x,axis=1))
            checks['logit_accuracy_max_error']=max(checks['logit_accuracy_max_error'],abs((~wrong).mean()-r['test']['accuracy']))
            checks['logit_ce_max_error']=max(checks['logit_ce_max_error'],abs(loss.mean()-r['test']['loss']))
            tail=np.sort(loss)[-100:]
            tails.append({**sp,'loss_q95':np.quantile(loss,.95),'loss_q99':np.quantile(loss,.99),'worst1pct_mean_loss':tail.mean(),'worst1pct_share_of_total_loss':tail.sum()/loss.sum(),'confident_errors_gt90pct':int((wrong&(conf>.9)).sum()),'test_loss':loss.mean()})
            logits[(seed,sp['batch_size'],sp['multiplier'],sp['arm'])]=(y,loss,wrong)
            statuses[r['status']]+=1
    print('Verified seed',seed,flush=True)
checks['functional_statuses']=dict(statuses)
assert checks['checkpoint_hashes_verified']==600 and checks['endpoint_records_verified']==240 and checks['exact_replay_records_verified']==120
assert checks['graft_nonzero_records']==115800 and checks['maximum_graft_norm_relative_error']<=1.2e-7
assert checks['logit_accuracy_max_error']<1e-12 and checks['logit_ce_max_error']<1e-6

# Independently recompute every published overall paired interval from endpoints.
metrics=['alpha_fc1','alpha_fc2','distance_fc1','distance_fc2','mean_abs_alpha_minus_2','test_accuracy_pct','test_loss']
avg=end.groupby(['seed','epoch','arm'])[metrics].mean(); comparisons=[]; maxerr=0.
for ep in [20,60]:
    for arm in ['magnitude_graft','sgdm']:
        for m in metrics:
            vals=avg.xs((ep,arm),level=['epoch','arm'])[m]-avg.xs((ep,'adamw'),level=['epoch','arm'])[m]
            calc=ci(vals); ref=pd.read_csv(A/'results/paired_effects.csv').query('scope=="all_settings" and epoch==@ep and treatment==@arm and metric==@m').iloc[0]
            maxerr=max(maxerr,float(np.max(abs(np.array(calc)-ref[['mean','ci_low','ci_high']].to_numpy(float)))))
            comparisons.append({'kind':'arm_contrast','epoch':ep,'arm':arm,'metric':m,'mean':calc[0],'ci_low':calc[1],'ci_high':calc[2]})
for arm in ['adamw','magnitude_graft','sgdm']:
    for m in metrics:
        vals=avg.xs((60,arm),level=['epoch','arm'])[m]-avg.xs((20,arm),level=['epoch','arm'])[m]; calc=ci(vals)
        ref=pd.read_csv(A/'results/within_arm_changes.csv').query('scope=="all_settings" and arm==@arm and metric==@m').iloc[0]
        maxerr=max(maxerr,float(np.max(abs(np.array(calc)-ref[['mean','ci_low','ci_high']].to_numpy(float)))))
        comparisons.append({'kind':'epoch60_minus20','arm':arm,'metric':m,'mean':calc[0],'ci_low':calc[1],'ci_high':calc[2]})
for arm in ['magnitude_graft','sgdm']:
    for m in metrics:
        v60=avg.xs((60,arm),level=['epoch','arm'])[m]-avg.xs((60,'adamw'),level=['epoch','arm'])[m]
        v20=avg.xs((20,arm),level=['epoch','arm'])[m]-avg.xs((20,'adamw'),level=['epoch','arm'])[m]; calc=ci(v60-v20)
        ref=pd.read_csv(A/'results/effect_changes.csv').query('scope=="all_settings" and treatment==@arm and metric==@m').iloc[0]
        maxerr=max(maxerr,float(np.max(abs(np.array(calc)-ref[['mean','ci_low','ci_high']].to_numpy(float)))))
        comparisons.append({'kind':'effect_change','arm':arm,'metric':m,'mean':calc[0],'ci_low':calc[1],'ci_high':calc[2]})
assert maxerr<1e-10; checks['maximum_recomputed_interval_error']=maxerr
checks['fit_support_counts']={f'{arm}_{layer}':int(g['supported_'+layer].sum()) for arm,g in end[end.epoch==60].groupby('arm') for layer in ['fc1','fc2','fc3']}
checks['plots_visually_inspected']=['Endpoint_Comparison.png','Validation_Trajectories.png','Alpha_Accuracy.png','Alpha_Trajectories.png']
checks['passed']=True
(O/'Downloaded_Artifact_Verification.json').write_text(json.dumps(checks,indent=2)+'\n')
(O/'Checkpoint_Hash_Audit.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
pd.DataFrame(comparisons).to_csv(O/'recomputed_overall_effects.csv',index=False)
tf=pd.DataFrame(tails);tf.to_csv(O/'exploratory_loss_tail_models.csv',index=False)
ta=tf.groupby(['seed','arm']).mean(numeric_only=True);te=[]
for arm in ['magnitude_graft','sgdm']:
    for m in ['loss_q95','loss_q99','worst1pct_mean_loss','worst1pct_share_of_total_loss','confident_errors_gt90pct']:
        vals=ta.xs(arm,level='arm')[m]-ta.xs('adamw',level='arm')[m];c=ci(vals)
        te.append({'treatment':arm,'reference':'adamw','metric':m,'independent_seeds':5,'mean':c[0],'ci_low':c[1],'ci_high':c[2]})
pd.DataFrame(te).to_csv(O/'exploratory_loss_tail_effects.csv',index=False)
print(json.dumps(checks,indent=2))
print(pd.DataFrame(comparisons).query('kind=="epoch60_minus20" and metric in ["test_accuracy_pct","test_loss","mean_abs_alpha_minus_2"]').to_string(index=False))
print(tf.groupby('arm').mean(numeric_only=True).to_string())
