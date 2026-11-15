"""Resume only the 64 unfinished Fashion-MNIST runs; preserve frozen training."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,shutil,urllib.request,datetime,traceback,concurrent.futures as cf,multiprocessing
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
PLAN=json.loads((HERE/'Recovery_Plan.json').read_text())
WORK=HERE/'run_work'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)

def prepare():
 assert not WORK.exists(),'Do not duplicate/reinitialize recovery work'
 WORK.mkdir();(WORK/'data').mkdir()
 for p in (BASE/'execution').glob('*.py'):shutil.copyfile(p,WORK/p.name)
 for name in ['Protocol.json','Preflight_Audit.json','Execution_Source_Manifest.json','Data_Provenance.json']:
  shutil.copyfile(BASE/'verification'/name,WORK/name)
 shutil.copyfile(BASE/'protocol/Protocol_Plan.json',WORK/'Protocol_Plan.json')
 shutil.copyfile(HERE/'split.npz',WORK/'data/split.npz')
 assert sha(WORK/'study.py')==PLAN['original_source_sha256'] and sha(WORK/'adapter.py')==PLAN['original_adapter_sha256']
 for name,value in json.loads((WORK/'Execution_Source_Manifest.json').read_text()).items():assert sha(WORK/name)==value
 assert sha(WORK/'data/split.npz')==PLAN['split_sha256']
 for rec in json.loads((WORK/'Data_Provenance.json').read_text())['files']:
  dest=WORK/'data'/rec['name'];dest.write_bytes(urllib.request.urlopen(rec['download_url'],timeout=180).read());assert sha(dest)==rec['sha256']
 for row in PLAN['resumed_runs']:
  sp=row['spec'];id=f"b{sp['batch_size']}_lr{sp['multiplier']:g}/{sp['arm']}/seed_{sp['seed']}";src=HERE/'inputs'/id/'resume.pt';assert sha(src)==row['resume_sha256']
  dst=WORK/'runs'/id;dst.mkdir(parents=True);shutil.copyfile(src,dst/'resume.pt');write(dst/'config.json',row['config'])
 write(WORK/'Recovery_Launch.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prior_complete':56,'resuming':6,'unstarted':58,'total_pending':64,'gpt_on_hold':True,'authorization':PLAN['authorization']})


def restoration_audit(a):
 s=a.s;data=s.DATA;audit=[]
 pre=json.loads((WORK/'Preflight_Audit.json').read_text())
 for row in PLAN['resumed_runs']:
  sp=row['spec'];d=WORK/'runs'/s.run_id(sp);ck=s.torch.load(d/'resume.pt',weights_only=False)
  assert ck['epoch']==row['epoch'] and ck['config']==row['config']
  s.torch.manual_seed(sp['seed']);net=s.MLP();assert s.weight_hash(net)==pre['initial_hashes'][str(sp['seed'])]
  net.load_state_dict(ck['model']);opt=s.make_opt(net,sp);opt.load_state_dict(ck['optimizer']);s.torch.set_rng_state(ck['torch_rng'])
  ep=ck['epoch']+1;ix=s.order(sp['seed'],ep)[:sp['batch_size']]
  loss,stats=a.ORIGINAL_STEP(net,opt,sp,data[0][ix],data[1][ix],s.schedule(ep),True)
  old=s.np.asarray(row['first_continuation_statistics'],dtype=float);new=s.np.asarray(stats,dtype=float)
  assert abs(loss-row['first_continuation_loss'])<=1e-6,(sp,'loss probe')
  assert s.np.allclose(old,new,rtol=1e-5,atol=1e-6,equal_nan=True),(sp,'update probe')
  audit.append({'spec':sp,'restored_epoch':ck['epoch'],'checkpoint_hash_matches':True,'initialization_hash_matches':True,'loss_difference':abs(loss-row['first_continuation_loss']),'statistics_exact':bool(s.np.array_equal(old,new,equal_nan=True)),'first_update_model_hash_exact':s.weight_hash(net)==row['first_continuation_model_hash'],'tolerance_check_passed':True})
 write(WORK/'Recovery_Restoration_Audit.json',{'passed':True,'checks':audit,'source_sha256':s.sha(WORK/'study.py'),'training_parameters_changed':False})


def one(spec,deadline):
 import adapter as a
 r=a.train_one(spec,deadline)
 return {k:r[k] for k in ['spec','status','last_epoch','seconds'] if k in r}


def verify_new(a):
 s=a.s;checks=[];hashes=[]
 for sp in PLAN['pending_specs']:
  d=WORK/'runs'/s.run_id(sp)
  if not (d/'done.json').exists():continue
  done=json.loads((d/'done.json').read_text());ck=s.torch.load(d/'checkpoints/epoch_120.pt',weights_only=False)
  assert ck['config']['source_sha256']==PLAN['original_source_sha256']
  expected_initial=json.loads((WORK/'Preflight_Audit.json').read_text())['initial_hashes'][str(sp['seed'])]
  assert done['initial_hash']==ck['config']['initial_hash']==expected_initial
  history=json.loads((d/'history.json').read_text());assert [h['epoch'] for h in history]==list(range(121))
  for h in history[1:]:assert h['batch_order_sha256']==hashlib.sha256(s.order(sp['seed'],h['epoch']).tobytes()).hexdigest()
  net=s.MLP();net.load_state_dict(ck['model']);assert s.weight_hash(net)==done['final_weight_hash']
  test,logits=s.evaluate(net,s.DATA[4],s.DATA[5],True);saved=s.np.load(d/'test_logits_epoch_120.npz')
  assert test==done['final']['test'] and s.np.array_equal(saved['logits'],logits) and s.np.array_equal(saved['labels'],s.DATA[5].numpy())
  dest=WORK/'verification'/s.run_id(sp);(dest/'spectra').mkdir(parents=True,exist_ok=True)
  spec=s.spectra(net,dest,120);old=s.np.load(d/'spectra/epoch_120.npz');new=s.np.load(dest/'spectra/epoch_120.npz');assert all(s.np.array_equal(old[n],new[n]) for n in s.LAYERS)
  for n in s.LAYERS:assert spec[n]['alpha']==done['final']['spectra'][n]['alpha']
  replay=False
  if (d/'checkpoints/epoch_030.pt').exists():
   early=s.torch.load(d/'checkpoints/epoch_030.pt',weights_only=False);net.load_state_dict(early['model'])
   opt=s.make_opt(net,sp);opt.load_state_dict(early['optimizer']);s.torch.set_rng_state(early['torch_rng'])
   order=s.order(sp['seed'],31);saved_stats=s.np.load(d/'statistics/epoch_031.npz')
   for j in range(11):
    ix=order[j*sp['batch_size']:(j+1)*sp['batch_size']];record=j in (0,1,10)
    loss,values=a.ORIGINAL_STEP(net,opt,sp,s.DATA[0][ix],s.DATA[1][ix],s.schedule(31),record)
    if record:
     k=int(s.np.flatnonzero(saved_stats['batch_index']==j)[0]);assert loss==float(saved_stats['loss'][k])
     assert s.np.array_equal(s.np.array(values,dtype=s.np.float32),saved_stats['values'][k],equal_nan=True)
   replay=True
  for cp in (d/'checkpoints').glob('*.pt'):hashes.append({'path':str(cp.relative_to(WORK)),'sha256':s.sha(cp)})
  checks.append({'spec':sp,'initial_hash_matches':True,'all_minibatch_hashes_match':True,'final_model_hash_matches':True,'final_logits_and_accuracy_exact':True,'final_spectra_and_alpha_exact':True,'eleven_step_replay_from_new_epoch30_exact':replay,'old_epoch30_replay_pending_on_original_runtime':not replay})
 write(WORK/'Recovery_Verification.json',{'passed':True,'verified_completed_pending_runs':len(checks),'checks':checks,'checkpoint_hashes':hashes,'prior56_preserved_manifest':PLAN['completed_preserved'],'note':'Historical checkpoint files remain in original archives. Merge these new outputs with the preserved original files before the full120-model audit.'})


def main():
 prepare();sys.path.insert(0,str(WORK));import adapter as a
 s=a.s;s.DATA=s.load_data();restoration_audit(a)
 deadline=time.time()+4.5*3600;start=time.time();results=[];errors=[]
 assert len(PLAN['pending_specs'])==64
 prior={s.run_id(x['spec']) for x in PLAN['completed_preserved']};assert all(s.run_id(x) not in prior for x in PLAN['pending_specs'])
 with cf.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('fork')) as pool:
  futures={pool.submit(one,sp,deadline):sp for sp in PLAN['pending_specs']}
  while futures:
   ready,_=cf.wait(futures,timeout=20,return_when=cf.FIRST_COMPLETED)
   for f in ready:
    sp=futures.pop(f)
    try:r=f.result()
    except Exception as ex:r={'spec':sp,'status':'exception','error':repr(ex)};errors.append(r)
    results.append(r);print('RECOVERED',len(results),s.run_id(sp),r['status'],flush=True)
   write(WORK/'Recovery_Status.json',{'prior_complete_preserved':56,'pending_returned':len(results),'pending_completed':sum(x['status']=='complete' for x in results),'total_complete':56+sum(x['status']=='complete' for x in results),'results':results,'errors':errors,'training_active':bool(futures),'elapsed_seconds':time.time()-start})
 verify_new(a)
 write(WORK/'Recovery_Completion.json',{'status':'complete' if len(results)==64 and all(x['status']=='complete' for x in results) else 'incomplete','total_complete':56+sum(x['status']=='complete' for x in results),'prior_complete_preserved':56,'results':results,'errors':errors,'seconds':time.time()-start,'source_sha256':s.sha(WORK/'study.py'),'gpt_on_hold':True,'full_merged_analysis_pending':True})

if __name__=='__main__':
 try:main()
 except Exception as ex:
  write(HERE/'Recovery_Failure.json',{'error':repr(ex),'traceback':traceback.format_exc()});raise
