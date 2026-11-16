"""Independent pairing/augmentation and intentionally wrong replay controls."""
import copy,json
import numpy as np
import run as r

def replay(spec,ck,wrong=None):
 s=r.s;net=s.MLP();opt=s.make_opt(net,spec);net.load_state_dict(ck['model']);opt.load_state_dict(copy.deepcopy(ck['optimizer']));s.torch.set_rng_state(ck['torch_rng'])
 if wrong=='zero_moments':
  for st in opt.state.values():
   for key in ['momentum_buffer','exp_avg','exp_avg_sq']:
    if key in st:st[key].zero_()
 order=s.order(spec['seed'],2);values=[]
 for j in range(2):
  k=j+2 if wrong=='wrong_batch' else j;ix=order[k*256:(k+1)*256]
  xb,_=r.augment_batch(s.DATA[0][ix],spec['seed'],2,k)
  loss,stats=s.train_step(net,opt,spec,xb,s.DATA[1][ix],r.schedule(2)*(1.1 if wrong=='wrong_lr' else 1),True)
  values.append((loss,np.array(stats,np.float32)))
 return values

def main():
 r.prepare();s=r.s
 x=s.torch.zeros(256,784);x[:,14*28+14]=1
 a,ha=r.augment_batch(x,32998,2,0);b,hb=r.augment_batch(x,32998,2,0);c,hc=r.augment_batch(x,32998,2,1)
 assert s.torch.equal(a,b) and ha==hb and ha!=hc
 assert s.torch.all(a.sum(1)==1) and s.torch.all((a>=0)&(a<=1))
 nz=a.reshape(-1,28,28).nonzero();assert s.torch.all((nz[:,1:]-14).abs()<=2)
 rows=[]
 for arm in s.ARMS:
  spec={'seed':32998,'batch_size':256,'multiplier':1.,'arm':arm}
  d=r.ROOT/'smoke'/s.run_id(spec);ck=s.torch.load(d/'epoch_001.pt',weights_only=False)
  actual=replay(spec,ck);recorded=np.load(d/'statistics/epoch_002.npz')
  assert all(v[0]==float(recorded['loss'][j]) and np.array_equal(v[1],recorded['values'][j],equal_nan=True) for j,v in enumerate(actual))
  for wrong in ['wrong_batch','zero_moments','wrong_lr']:
   changed=replay(spec,ck,wrong);equal=all(a[0]==b[0] and np.array_equal(a[1],b[1],equal_nan=True) for a,b in zip(actual,changed))
   assert not equal;rows.append({'arm':arm,'perturbation':wrong,'correctly_rejected':True})
 s.writej(r.ROOT/'Replay_Negative_Controls.json',{'passed':True,'rejected':len(rows),'records':rows,'augmentation_pairing':True,'augmentation_range_and_no_wrap':True})
 print('Nine incorrect replay controls rejected; augmentation pairing verified')

if __name__=='__main__':main()
