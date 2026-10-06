"""Reference tail averages over replicated dense parameters, no CUDA collectives."""
import math
import torch
from .ngram_math import is_update_step


class TailAverages:
    def __init__(self, params, total=1194):
        self.params,self.total=params,total
        groups={'ema':('lm_head','embed'),'avg':('vo_bank','mlp_bank'),
                've':('value_embeds',),'bank':('qk_bank','vo_bank','mlp_bank')}
        # Reserve the late-run footprint before preflight measures peak memory.
        self.buffers={(tag,name):torch.zeros_like(params[name],dtype=torch.float32)
                      for tag,names in groups.items() for name in names}
        self.seeded=set()

    @torch.no_grad()
    def tick(self,step):
        n=self.total
        r=2/299
        ema_rate=(r if step==n-298 or step==n-1 else -math.expm1(2*math.log1p(-r)))
        groups=[
            ('ema',('lm_head','embed'),298,ema_rate if step%2 or step==n-298 else None),
            ('avg',('vo_bank','mlp_bank'),250,4/53 if (n-1-step)%4==0 else None),
            ('ve',('value_embeds',),250,4/53 if is_update_step(step) else None),
            ('bank',('qk_bank','vo_bank','mlp_bank'),298,2/(298//4+1) if step%4==1 or step>=n-2 else None),
        ]
        for tag,names,window,rate in groups:
            if step<n-window or rate is None: continue
            for name in names:
                key=(tag,name)
                if key not in self.seeded:
                    self.buffers[key].copy_(self.params[name].detach().float())
                    self.seeded.add(key)
                else:
                    self.buffers[key].lerp_(self.params[name].float(),rate)

    @torch.no_grad()
    def ship(self):
        names=('embed','lm_head','mlp_bank','qk_bank','vo_bank')
        before={name:self.params[name].float().norm() for name in names}
        for tag in ('ema','avg','ve','bank'):
            for (which,name),average in self.buffers.items():
                if which!=tag: continue
                p=self.params[name]
                blend=.65 if tag=='ema' else (.55 if name=='qk_bank' else .55-.2007) if tag=='bank' else None
                p.copy_((average if blend is None else torch.lerp(p.float(),average,blend)).to(p.dtype))
        required={('ema','lm_head'),('ema','embed'),('avg','vo_bank'),('avg','mlp_bank'),
                  ('ve','value_embeds'),('bank','qk_bank'),('bank','vo_bank'),('bank','mlp_bank')}
        if self.seeded!=required:
            raise RuntimeError('Tail averages incomplete; cannot report final benchmark')
        for name in names:
            p=self.params[name]; after=p.float().norm()
            ratio=torch.where(after>0,before[name]/after,torch.ones_like(after))
            p.copy_((p.float()*ratio).to(p.dtype))
