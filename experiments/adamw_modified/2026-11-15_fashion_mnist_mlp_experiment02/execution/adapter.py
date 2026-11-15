"""Isolated Fashion-MNIST adapter around the byte-identical Experiment01 kernel."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import json
import time
from pathlib import Path
import study as s

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'Protocol_Plan.json').read_text())
s.OUT = ROOT
s.SEEDS = PLAN['seeds']
s.specifications = lambda: PLAN['specifications']
s.freeze_protocol = lambda: json.loads((ROOT/'Protocol.json').read_text())
ORIGINAL_STEP = s.train_step
DEADLINE = float('inf')


class BudgetStop(Exception):
    pass


def bounded_step(net, opt, spec, xb, yb, factor, record=False):
    if time.time() >= DEADLINE:
        dest=ROOT/'runs'/s.run_id(spec)
        s.savept(dest/'deadline_state.pt',{'model':net.state_dict(),'optimizer':opt.state_dict(),
                 'torch_rng':s.torch.get_rng_state(),'spec':spec,'status':'time_limit'})
        raise BudgetStop('Training cutoff reached before the next minibatch')
    return ORIGINAL_STEP(net,opt,spec,xb,yb,factor,record)


def configure_training(deadline):
    global DEADLINE
    DEADLINE=deadline
    s.train_step=bounded_step


def train_one(spec, deadline):
    if time.time() >= deadline:
        return {'spec':spec,'status':'not_started_budget','last_epoch':-1}
    configure_training(deadline)
    try:
        return s.train(spec,deadline)
    except BudgetStop:
        history=ROOT/'runs'/s.run_id(spec)/'history.json'
        last=json.loads(history.read_text())[-1]['epoch'] if history.exists() else -1
        result={'spec':spec,'status':'time_limit','last_epoch':last}
        s.writej(ROOT/'runs'/s.run_id(spec)/'stopped.json',result)
        return result
