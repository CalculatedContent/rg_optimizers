"""Post-run restoration, update replay and checkpoint integrity checks."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time
from pathlib import Path
sys.path.insert(0,'/tmp/adamw_shape_deps')
import numpy as np,torch
import study as s
OUT=Path(__file__).resolve().parent

def main():
    p=json.loads((OUT/'Protocol.json').read_text());a=json.loads((OUT/'Analysis_Audit.json').read_text())
    c=json.loads((OUT/'Completion.json').read_text())
    assert a['passed'] and c['completed']==135 and not c['failures']
    assert p['source_sha256']==s.sha(OUT/'study.py')
    x,y,*_=s.load_data();manifest=[];restored=[];replayed=[];tic=time.time()
    for spec in s.specifications():
        d=OUT/'runs'/s.run_id(spec);done=json.loads((d/'done.json').read_text())
        for ep in [0,30,60,90,120]:
            cp=d/f'checkpoints/epoch_{ep:03}.pt'
            manifest.append({'path':str(cp.relative_to(OUT)),'bytes':cp.stat().st_size,'sha256':s.sha(cp),'spec':spec,'epoch':ep})
        ck=torch.load(d/'checkpoints/epoch_120.pt',weights_only=False)
        net=s.MLP();net.load_state_dict(ck['model']);assert s.weight_hash(net)==done['final_weight_hash']
        assert ck['config']['initial_hash']==done['initial_hash'] and ck['epoch']==120
        # Recompute all three spectra from a central and an extreme setting per arm.
        selected=(spec['seed']==31001 and spec['batch_size']==256 and spec['multiplier']==1.) or (spec['seed']==31005 and spec['batch_size']==128 and spec['multiplier']==2.)
        if selected:
            dest=OUT/'verification'/s.run_id(spec);(dest/'spectra').mkdir(parents=True,exist_ok=True)
            sp=s.spectra(net,dest,120)
            err=max(abs(sp[n]['alpha']-done['final']['spectra'][n]['alpha']) for n in s.LAYERS)
            assert err<1e-10
            old=np.load(d/'spectra/epoch_120.npz');new=np.load(dest/'spectra/epoch_120.npz')
            for n in s.LAYERS:assert np.array_equal(old[n],new[n])
            restored.append({'spec':spec,'max_alpha_difference':err,'raw_spectra_identical':True})
            # Replay the first eleven next-epoch batches from a quarter checkpoint.
            ck=torch.load(d/'checkpoints/epoch_030.pt',weights_only=False)
            net.load_state_dict(ck['model']);opt=s.make_opt(net,spec);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng'])
            ix=s.order(spec['seed'],31);z=np.load(d/'statistics/epoch_031.npz')
            errors=[]
            for j,st in enumerate(range(0,len(y),spec['batch_size'])):
                if j>10:break
                ind=ix[st:st+spec['batch_size']];record=j in [0,1,10]
                loss,ss=s.train_step(net,opt,spec,x[ind],y[ind],s.schedule(31),record)
                if record:
                    k=int(np.flatnonzero(z['batch_index']==j)[0]);values=np.array(ss,dtype=np.float32)
                    assert loss==float(z['loss'][k])
                    assert np.array_equal(values,z['values'][k],equal_nan=True)
                    errors.append(0.)
            replayed.append({'spec':spec,'from_epoch':30,'through_batch':10,'three_sampled_losses_and_update_vectors_match':True})
    s.writej(OUT/'Checkpoint_Manifest.json',manifest)
    s.writej(OUT/'Final_Verification.json',{'passed':True,'completed_runs':135,'all_final_model_hashes_match':True,
       'checkpoints_hashed':len(manifest),'checkpoint_total_bytes':sum(r['bytes'] for r in manifest),
       'restored_spectrum_checks':restored,'replay_checks':replayed,'analysis_audit':a,
       'training_wall_seconds':c['seconds'],'verification_seconds':time.time()-tic,
       'next_experiment_started':False,'stop_for_user_review':True})
    print(json.dumps({'passed':True,'checkpoints':len(manifest),'spectral_restorations':len(restored),'replay_checks':len(replayed)}))

if __name__=='__main__':main()
