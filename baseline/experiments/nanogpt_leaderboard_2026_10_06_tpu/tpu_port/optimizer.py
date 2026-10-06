"""ANVIL twin rails, six-map cascade, lane equalizer and auxiliary cautious Adam.

Portable FP32 master parameters with BF16 round-to-nearest projections replace
CUDA's packed low-mantissa/truncated-high-half storage. See PORTING.md.
"""
import math
import torch
from .schedule import get_rail_beta
from .ngram_math import is_update_step

ANVIL_MAPS = [
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
]
# (betas, lr multiplier, decay multiplier)
ADAM = {
    'scalars': ((.9,.99),5.,0.), 'smear_gate': ((.9,.99),.01,0.),
    've_gate_bank': ((.9,.99),1.,1.), 'lm_head': ((.5,.95),1.,150.),
    'post_lambdas': ((.9,.95),1.,0.), 'resid_lambdas': ((.9,.95),5.,0.),
    'value_embeds': ((.75,.95),70.,5.), 'embed': ((.5,.95),1.,150.),
    'mudd_w1': ((.9,.99),.25,1.), 'mudd_w2': ((.9,.99),.25,1.),
    'mudd_w2g': ((.9,.99),.25,1.), 'mudd_b2': ((.9,.99),.25,0.),
    'mudd_gate_w1': ((.9,.99),.1,1.), 'mudd_gate_w2': ((.9,.99),.1,1.),
    'mudd_gate_b2': ((.9,.99),.1,0.), '_mudd_gate_scale': ((.9,.99),.1,0.),
}
BANKS = ('qk_bank', 'vo_bank', 'mlp_bank')


def scalar(value, tensor):
    # Upload scalar data rather than changing graph constants every training step.
    return torch.tensor(value, dtype=torch.float32, device='cpu').to(tensor.device)


def cascade(gradient, velocity, momentum, fast_beta, fast_weight):
    g = gradient.float()
    velocity[0].lerp_(g, 1 - fast_beta)
    velocity[1].lerp_(g, .02)
    blend = fast_weight * velocity[0] + (1-fast_weight) * velocity[1]
    # Keep the polynomial recurrence in FP32. BF16 rounding/fusion differences
    # in XLA can push the intermediate singular values outside its stable basin.
    x = torch.lerp(g, blend, momentum)
    tall = x.size(-2) > x.size(-1)
    gram = x.mT @ x if tall else x @ x.mT
    d = gram.diagonal(dim1=-2,dim2=-1).float().sum(-1)[...,None,None].sqrt() * 1.05 + 1e-6
    x = x/d
    gram = gram/d.square()
    for index,(a,b,c) in enumerate(ANVIL_MAPS):
        if index:
            gram = x.mT @ x if tall else x @ x.mT
        poly = b*gram + c*(gram @ gram)
        product = x @ poly if tall else poly @ x
        x = a*x + product
    return x.bfloat16()


class Optimizer:
    def __init__(self, model, schedule, reduce_mean=lambda x:x):
        self.params = {getattr(p,'label',name.replace('.weight','')):p for name,p in model.named_parameters()}
        if set(self.params) != set(ADAM) | set(BANKS):
            raise ValueError('Unexpected model parameter labels')
        self.schedule, self.reduce_mean = schedule, reduce_mean
        self.split = False
        self.state = {}
        for name,p in self.params.items():
            if name in BANKS:
                # Module.to(XLA) replaces Parameters and discards Python attributes.
                # Derive bank layout from actual tensor shapes, never p.reshape metadata.
                shape = ((p.shape[0]*p.shape[1],*p.shape[2:])
                         if name=='mlp_bank' and p.ndim==4 else tuple(p.shape))
                red_dim = -1 if shape[-2] >= shape[-1] else -2
                shape_lane = list(shape); shape_lane[red_dim]=1
                self.state[name] = dict(master=p.detach().view(shape).float().clone(),
                    velocity=torch.zeros((2,*shape),device=p.device),
                    energy=torch.zeros(shape_lane,device=p.device),red_dim=red_dim,shape=shape)
            else:
                self.state[name] = dict(m=torch.zeros_like(p,dtype=torch.float32),
                    v=torch.zeros_like(p,dtype=torch.float32),event=0)

    @torch.no_grad()
    def step(self, step):
        do_adam = step % 2 == 1
        if do_adam and not self.split:
            self.params['lm_head'].grad.add_(self.params['embed'].grad.T)
        for name,p in self.params.items():
            update = (name in BANKS or (is_update_step(step) if name=='value_embeds' else do_adam))
            if not update or (name=='embed' and not self.split):
                continue
            if p.grad is None:
                raise RuntimeError('Missing gradient: '+name)
            g = self.reduce_mean(p.grad)
            state = self.state[name]
            lr_mult = self.schedule.get_lr(step)
            if name in BANKS:
                g = g.view(state['shape'])
                beta = get_rail_beta(step,self.schedule.total_steps)
                momentum = scalar(beta,g)
                fast_beta = scalar(.85 if step>=514 else beta,g)
                fast_weight = scalar(.4385 if step>=514 else 1.,g)
                u = cascade(g,state['velocity'],momentum,fast_beta,fast_weight)
                axis=state['red_dim']; length=u.size(axis)
                power=u.float().square().mean(axis,keepdim=True)
                before=(power.sum((-2,-1),keepdim=True)*length).sqrt()
                state['energy'].lerp_(power,.1)
                gain=state['energy'].clamp_min(1e-10).rsqrt()
                after=(power*length*gain.square()).sum((-2,-1),keepdim=True).sqrt()
                u=u*(gain*(before/after.clamp_min(1e-10))).to(u.dtype)
                rate=.023*lr_mult
                shape_mult=max(1.,g.size(-2)/g.size(-1))**.5
                factors=torch.ones(g.size(0),1,1,device=p.device)
                if name=='mlp_bank':
                    factors[1::2]=2
                    # Reference slot 7 has no MLP: its two stored matrices are frozen.
                    factors[14:16]=0
                effective=scalar(rate*shape_mult,p)*factors
                master=state['master']
                aligned=state['velocity'][1]*master>=0
                master.sub_(master*aligned*scalar(2.25*rate,p)*effective + u.float()*effective)
                p.copy_(master.reshape(p.shape).to(p.dtype))
            else:
                (beta1,beta2),mul,wd = ADAM[name]
                if name=='value_embeds' and step>=336:
                    beta1,beta2,wd=beta1**2,beta2**2,10.
                state['event']+=1; event=state['event']
                # Upstream moments are FP32; dense gradient exchange uses parameter dtype.
                g=g.float()
                state['m'].mul_(beta1).add_(g,alpha=1-beta1)
                state['v'].mul_(beta2).addcmul_(g,g,value=1-beta2)
                rate=.008*lr_mult*mul
                step_size=scalar(rate*math.sqrt(1-beta2**event)/(1-beta1**event),p)
                u=state['m']/(state['v'].sqrt()+1e-10)*step_size
                u=u+torch.where(u*p>0,p.float()*scalar(rate*rate*.005*wd,p),0)
                p.copy_((p.float()-u).to(p.dtype))
            p.grad=None
        if do_adam and not self.split:
            self.params['embed'].copy_(self.params['lm_head'].T)
            self.params['embed'].grad=None
        if step==self.schedule.split_step:
            for key in ('m','v'):
                self.state['embed'][key].copy_(self.state['lm_head'][key].T)
            self.state['embed']['event']=self.state['lm_head']['event']
            self.split=True
