"""Synthetic reporting integration checks; these fixtures are NOT experimental results."""
import copy,json,tempfile,hashlib,contextlib,io
from pathlib import Path
import numpy as np
import pandas as pd
import report
from schedule import accounting

def main():
    with tempfile.TemporaryDirectory(prefix='synthetic-report-test-') as td:
        root=Path(td);report.ROOT=root;report.OUT=root/'results';report.PLOTS=root/'plots'
        report.OUT.mkdir();report.PLOTS.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):v=report.main()
        assert not v['passed'] and v['completed']==0 and len(v['missing_runs'])==2
        (root/'Preflight.json').write_text('{"passed":true}')
        (root/'Native_AdamW_Contract.json').write_text('{"passed":true}')
        (root/'Initialization_Audit.json').write_text(json.dumps(dict(passed=True,initial_hashes={str(seed):f'same-init-{seed}' for seed in report.PLAN['seeds']})))
        for sp in report.PLAN['specifications']:
            d=root/'runs'/report.ident(sp);(d/'statistics').mkdir(parents=True)
            history=[];tests=[];hashes={}
            for epoch in range(121):
                cp=d/f'epoch_{epoch:04}.pt';cp.write_bytes(b'SYNTHETIC CHECKPOINT HASH FIXTURE')
                hashes[cp.name]=report.sha(cp)
                log=np.zeros((3,10));log[0,0]=3.;log[1,1]=2.;log[2,0]=1.+epoch/120
                labels=np.array([0,1,2]);path=d/f'test_logits_{epoch:04}.npz'
                np.savez_compressed(path,logits=log,labels=labels)
                cal=report.calibration(path)
                tests.append(dict(epoch=epoch,test=dict(accuracy=cal['test_accuracy_pct']/100,loss=cal['test_ce']),checkpoint_sha256=hashes[cp.name]))
                spectra={}
                for name,alpha in [('fc1',1.6+.003*epoch+.05*sp['augmentation']),('fc2',2.1-.001*epoch+.02*sp['augmentation']),('fc3',3.)]:
                    spectra[name]=dict(alpha=alpha,supported=epoch>20 and name!='fc3',D=.1,sigma=.2,xmin=1.,xmax=3.,
                        num_pl_spikes=30 if name!='fc3' else 8,norm=10.,spectral_norm=3.,
                        fixed_rank={'0.2':{'alpha':alpha},'0.5':{'alpha':alpha}},small_n={'alpha':alpha})
                history.append(dict(**accounting(epoch),spectra=spectra,validation=dict(accuracy=.8,loss=.3),
                    online_training_loss=.2,train_unaugmented=dict(accuracy=.9,loss=.2),batch_order_sha256=report.order_hash(sp['seed'],epoch)))
                if epoch:
                    np.savez_compressed(d/f'statistics/epoch_{epoch:04}.npz',values=np.ones((215,3,10)),batch_index=np.arange(215))
            (d/'history.json').write_text(json.dumps(history))
            result=dict(spec=sp,endpoint_epoch=120,initial_hash=f'same-init-{sp["seed"]}',
                status='FUNCTIONAL_FAILURE' if sp['seed']==33001 else 'USABLE',checkpoint_hashes=hashes,test_milestones=tests,
                verification=dict(passed=True,replays=[1,60,119],restored_validation_and_spectra_exact=True))
            (d/'done.json').write_text(json.dumps(result))
        with contextlib.redirect_stdout(io.StringIO()):v=report.main()
        assert v['passed'] and v['completed']==2
        assert len(pd.read_csv(report.OUT/'milestones.csv'))==242
        assert (pd.read_csv(report.OUT/'model_register.csv').status=='FUNCTIONAL_FAILURE').sum()==1
        intervals=pd.read_csv(report.OUT/'within_run_interval_changes.csv')
        assert len(intervals)==6 and np.allclose(intervals.loc[intervals.start_epoch==60,'alpha_fc1'],.18)
        changes=pd.read_csv(report.OUT/'within_run_changes_from20.csv')
        assert np.allclose(changes.loc[changes.epoch==120,'alpha_fc1'],.3)
        assert all((report.PLOTS/n).stat().st_size>10000 for n in ['alpha_and_lr_trajectories.png','performance_trajectories.png','fit_support_and_norms.png'])
        # Missing raw statistics must invalidate otherwise complete endpoints.
        victim=next((root/'runs').glob('*/seed_*/statistics/epoch_0001.npz'));victim.unlink()
        with contextlib.redirect_stdout(io.StringIO()):bad=report.main()
        assert not bad['passed'] and not bad['every_minibatch_statistics']
    print('PASS: missing/complete outcomes, failed-model retention,242epoch records,6within-run intervals, known changes, plots and missing-statistics negative control')

if __name__=='__main__':main()
