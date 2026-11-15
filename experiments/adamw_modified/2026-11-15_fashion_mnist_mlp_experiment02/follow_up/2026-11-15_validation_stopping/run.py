"""Separate validation-stopped Fashion-MNIST rerun; frozen optimizer kernel."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,copy,hashlib,shutil,urllib.request,traceback,concurrent.futures as cf,multiprocessing
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parent
PARENT=Path(os.environ.get('FASHION_PARENT',str(ROOT.parents[1])))
KERNEL=PARENT/'execution' if (PARENT/'execution/study.py').exists() else PARENT
sys.path.insert(0,str(KERNEL));import study as s
s.OUT=ROOT
PLAN=json.loads((ROOT/'Protocol.json').read_text())

def parentfile(name):
 for p in [PARENT/name,PARENT/'verification'/name,PARENT/'protocol'/name]:
  if p.exists():return p
 raise FileNotFoundError(name)

def prepare():
 assert s.sha(KERNEL/'study.py')==PLAN['kernel_sha256']
 (ROOT/'data').mkdir(exist_ok=True)
 provenance=json.loads(parentfile('Data_Provenance.json').read_text())
 for row in provenance['files']:
  dest=ROOT/'data'/row['name'];local=PARENT/'data'/row['name']
  if not dest.exists():
   if local.exists():shutil.copyfile(local,dest)
   else:dest.write_bytes(urllib.request.urlopen(row['download_url'],timeout=180).read())
  assert s.sha(dest)==row['sha256']
 dest=ROOT/'data/split.npz'
 if not dest.exists():shutil.copyfile(PARENT/'data/split.npz' if (PARENT/'data/split.npz').exists() else PARENT/'recovery/split.npz',dest)
 assert s.sha(dest)==provenance['split_sha256']
 z=np.load(dest);assert len(np.intersect1d(z['train'],z['validation']))==0 and len(z['train'])==55000 and len(z['validation'])==5000
 s.DATA=s.load_data();s.writej(ROOT/'Data_Audit.json',{'passed':True,'dataset':'Fashion-MNIST','split_sha256':s.sha(dest),'kernel_sha256':s.sha(KERNEL/'study.py')})


def evaluate(net,x,y):
 row,logits=s.evaluate(net,x,y,True)
 raw=logits.astype(float);loss=float(np.mean(logsumexp(raw,axis=1)-raw[np.arange(len(y)),y.numpy()]))
 assert abs(loss-row['loss'])<2e-6
 row['independent_cross_entropy_error']=abs(loss-row['loss']);row['prediction_counts']=np.bincount(logits.argmax(1),minlength=10).tolist()
 return row,logits


def fit_audit(rows,filename):
 errors=[];z=np.load(filename)
 for name in s.LAYERS:
  r=rows[name]
  if r['alpha'] is None or len(z[name])<50:continue # Output layer retained but not a supported spectral inference
  x=z[name].astype(float)*r['xmax']/z[name].max();tail=x[(x>=r['xmin']*(1-1e-10))&(x<=r['xmax']*(1+1e-10))]
  if len(tail)<2:continue
  estimate=1+len(tail)/np.log(tail/r['xmin']).sum()
  assert len(tail)==r['num_pl_spikes'] and abs(estimate-r['alpha'])<1e-8
  errors.append(abs(estimate-r['alpha']))
 return max(errors,default=0.)


def checkpoint(d,name,net,opt,spec,epoch,history,initial):
 s.savept(d/name,{'model':net.state_dict(),'optimizer':opt.state_dict(),'torch_rng':s.torch.get_rng_state(),'spec':spec,'epoch':epoch,'history':history,'initial_hash':initial,'model_hash':s.weight_hash(net),'kernel_sha256':s.sha(KERNEL/'study.py'),'runner_sha256':s.sha(ROOT/'run.py')})


def train(spec,deadline,smoke=False):
 d=ROOT/('smoke' if smoke else 'runs')/s.run_id(spec)
 assert not d.exists(),'Never replace an existing observation'
 for name in ('spectra','statistics'):(d/name).mkdir(parents=True,exist_ok=True)
 s.torch.manual_seed(spec['seed']);net=s.MLP();opt=s.make_opt(net,spec);initial=s.weight_hash(net)
 if not smoke:assert initial==json.loads(parentfile('Preflight_Audit.json').read_text())['initial_hashes'][str(spec['seed'])]
 x,y,xv,yv,xt,yt=s.DATA;best=float('inf');best_epoch=0;history=[];tic=time.time();stop='maximum_epochs';fit_error=0.
 for epoch in range((2 if smoke else PLAN['maximum_epochs'])+1):
  assert time.time()<deadline,'Bounded rerun deadline reached'
  samples=[];order_hash=None
  if epoch:
   net.train();order=s.order(spec['seed'],epoch);order_hash=hashlib.sha256(order.tobytes()).hexdigest()
   for j,start in enumerate(range(0,len(y),spec['batch_size'])):
    assert time.time()<deadline,'Bounded rerun deadline reached'
    ix=order[start:start+spec['batch_size']];record=j<2 or j%10==0
    loss,stats=s.train_step(net,opt,spec,x[ix],y[ix],s.schedule(epoch),record)
    if record:samples.append((j,loss,stats))
   np.savez_compressed(d/f'statistics/epoch_{epoch:03}.npz',batch_index=[r[0] for r in samples],loss=[r[1] for r in samples],values=np.array([r[2] for r in samples],dtype=np.float32))
  validation,_=evaluate(net,xv,yv);training=s.evaluate(net,x,y);spectra=s.spectra(net,d,epoch)
  fit_error=max(fit_error,fit_audit(spectra,d/f'spectra/epoch_{epoch:03}.npz'))
  row={'epoch':epoch,'train':training,'validation':validation,'spectra':spectra,'batch_order_sha256':order_hash}
  history.append(row)
  if validation['loss']<best:
   best=validation['loss'];best_epoch=epoch;checkpoint(d,'best.pt',net,opt,spec,epoch,history,initial)
  row['best_validation_loss_so_far']=best
  checkpoint(d,'last.pt',net,opt,spec,epoch,history,initial)
  if epoch==0:checkpoint(d,'initial.pt',net,opt,spec,epoch,history,initial)
  s.writej(d/'history.json',history)
  if epoch%5==0:print(s.run_id(spec),'epoch',epoch,'val',validation['loss'],'best_epoch',best_epoch,flush=True)
  if not smoke and epoch>=PLAN['minimum_epochs'] and epoch-best_epoch>=PLAN['patience']:
   stop='validation_patience';break
 assert all(s.torch.isfinite(p).all() for p in net.parameters())
 ck=s.torch.load(d/'best.pt',weights_only=False);assert ck['epoch']==best_epoch
 net.load_state_dict(ck['model']);assert s.weight_hash(net)==ck['model_hash']
 selected_val,val_logits=evaluate(net,xv,yv);assert selected_val==history[best_epoch]['validation']
 restored=d/'restored';(restored/'spectra').mkdir(parents=True,exist_ok=True);sp=s.spectra(net,restored,best_epoch)
 old=np.load(d/f'spectra/epoch_{best_epoch:03}.npz');new=np.load(restored/f'spectra/epoch_{best_epoch:03}.npz');assert all(np.array_equal(old[n],new[n]) for n in s.LAYERS)
 assert all(sp[n]['alpha']==history[best_epoch]['spectra'][n]['alpha'] for n in s.LAYERS)
 origin=best_epoch if best_epoch<epoch else 0
 replay=s.torch.load(d/('best.pt' if origin==best_epoch else 'initial.pt'),weights_only=False)
 net.load_state_dict(replay['model']);opt.load_state_dict(replay['optimizer']);s.torch.set_rng_state(replay['torch_rng'])
 order=s.order(spec['seed'],origin+1);recorded=np.load(d/f'statistics/epoch_{origin+1:03}.npz')
 for j in range(2):
  ix=order[j*spec['batch_size']:(j+1)*spec['batch_size']];loss,values=s.train_step(net,opt,spec,x[ix],y[ix],s.schedule(origin+1),True)
  k=int(np.flatnonzero(recorded['batch_index']==j)[0]);assert loss==float(recorded['loss'][k]);assert np.array_equal(np.array(values,dtype=np.float32),recorded['values'][k],equal_nan=True)
 net.load_state_dict(ck['model'])
 result={'spec':spec,'best_epoch':best_epoch,'stopped_epoch':epoch,'stop_reason':stop,'initial_hash':initial,'selected_model_hash':s.weight_hash(net),'initial_validation_loss':history[0]['validation']['loss'],'validation':selected_val,'selected_train':s.evaluate(net,x,y),'spectra':sp,'seconds':time.time()-tic,'status':'USABLE' if sum(v>0 for v in selected_val['prediction_counts'])>1 else 'FAILED','verification':{'passed':True,'checkpoint_replay_exact':True,'replay_origin':origin,'restored_validation_and_spectra_exact':True,'maximum_independent_alpha_error':fit_error,'selection_used_test':False,'selection_used_alpha':False,'kernel_sha256':s.sha(KERNEL/'study.py')},'checkpoint_hashes':{n:s.sha(d/n) for n in ['initial.pt','best.pt','last.pt']}}
 if not smoke:
  result['test'],test_logits=evaluate(net,xt,yt);np.savez_compressed(d/'selected_test_logits.npz',logits=test_logits,labels=yt.numpy())
 np.savez_compressed(d/'selected_validation_logits.npz',logits=val_logits,labels=yv.numpy())
 s.writej(d/'done.json',result);return result


def guarded(spec,deadline):
 try:return train(spec,deadline)
 except Exception as exc:
  d=ROOT/'runs'/s.run_id(spec);s.writej(d/'failure.json',{'spec':spec,'error':repr(exc),'traceback':traceback.format_exc()});return {'spec':spec,'status':'ERROR','error':repr(exc)}


def main(smoke=False):
 prepare()
 if smoke:
  for arm in s.ARMS:train({'seed':31999,'batch_size':256,'multiplier':1,'arm':arm},time.time()+1800,True)
  print('ALL THREE SMOKE CHECKS PASSED');return
 assert not (ROOT/'Launch.json').exists(),'Do not duplicate this rerun'
 tic=time.time();deadline=tic+4*3600;s.writej(ROOT/'Launch.json',{'started_epoch_seconds':tic,'deadline_epoch_seconds':deadline,'planned_runs':120,'new_protocol':True})
 specs=PLAN['specifications'];pilot=[x for x in specs if x['seed']==31001 and x['batch_size']==256 and x['multiplier']==1]
 assert len(pilot)==3
 with cf.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('fork')) as pool:
  first=list(pool.map(guarded,pilot,[deadline]*3))
  passed=all(r['status']=='USABLE' and r['verification']['passed'] and r['initial_validation_loss']-r['validation']['loss']>=.1 for r in first)
  s.writej(ROOT/'Pilot_Gate.json',{'passed':passed,'runs':first,'included_in120':True,'alpha_not_a_gate':True})
  if not passed:raise RuntimeError('First three runs failed checks; expansion stopped')
  remaining=[x for x in specs if x not in pilot];results=first[:];futures={pool.submit(guarded,x,deadline):x for x in remaining}
  while futures:
   ready,_=cf.wait(futures,timeout=20,return_when=cf.FIRST_COMPLETED)
   for f in ready:results.append(f.result());futures.pop(f)
   s.writej(ROOT/'Run_Register.json',{'returned':len(results),'planned':120,'completed':sum('test' in r for r in results),'errors':[r for r in results if r['status']=='ERROR'],'training_active':bool(futures)})
 s.writej(ROOT/'Completion.json',{'results':results,'planned':120,'completed':sum('test' in r for r in results),'seconds':time.time()-tic,'gpt_on_hold':True})

if __name__=='__main__':
 try:main('--smoke' in sys.argv)
 except Exception as exc:s.writej(ROOT/'Failure.json',{'error':repr(exc),'traceback':traceback.format_exc()});raise
