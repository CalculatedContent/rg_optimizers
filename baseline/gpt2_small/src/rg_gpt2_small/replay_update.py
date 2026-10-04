"""Replay one saved MuonClip update, locating the first invalid stage.

Diagnostic synchronization changes graph boundaries. A passing replay is evidence
about this instrumented path, not proof that the original continuous path is fixed.
"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from . import experiment as g, port_debug as debug
from .execution_checks import check_finite, evaluate_splits
from rg_nanogpt_one_head import runtime as rt, tpu_spmd as spmd, optimizers


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()



def source_tensors(state):
    yield from (('weight/'+name,tensor) for name,tensor in state['model'].items())
    for index,optimizer in enumerate(state['optimizers']):
        for parameter,values in optimizer['state'].items():
            for key,value in values.items():
                if torch.is_tensor(value): yield f'optimizer_{index}/{parameter}/{key}',value



def replay(checkpoint, data_root, output, device='tpu', expected_step=1):
    checkpoint=Path(checkpoint); output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    model=None; handles=None; state=None; gen=None; completed_roles=[]
    report={'status':'running','original_checkpoint':str(checkpoint),'device':device,
            'automatic_long_run':False,'instrumented_graph_boundaries':True}
    debug.environment(output)
    try:
        debug.stage(output,'loading_source_checkpoint',expected_step)
        report['source_sha256']=sha256(checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=state['config']; start=int(state['step']); update=start+1
        if start!=expected_step or state['scheduler_step']!=start:
            raise RuntimeError('Unexpected source checkpoint step/scheduler.')
        step_tokens=cfg['training']['batch_size']*cfg['training']['grad_accum_steps']*cfg['model']['block_size']
        if state['tokens_seen']!=start*step_tokens: raise RuntimeError('Checkpoint token count differs.')
        if cfg['optimizer']['family']!='muon_clip': raise RuntimeError('Expected a MuonClip checkpoint.')
        report.update(source_step=start,replayed_update=update,config=cfg)
        debug.write(output/'REPLAY_SOURCE.json',report)
        check_finite(source_tensors(state),output,'saved_cpu_state',start,'cpu')
        # Validate the ORIGINAL fingerprint before selecting a CPU diagnostic device.
        metadata,arrays=g.load_memmaps(data_root,cfg)
        versions={'torch':torch.__version__,'numpy':np.__version__}
        fingerprint=hashlib.sha256(json.dumps({'config':cfg,'data':metadata,'versions':versions},sort_keys=True).encode()).hexdigest()
        if fingerprint!=state['fingerprint']: raise RuntimeError('Source config/data/version fingerprint mismatch.')
        runtime_cfg=json.loads(json.dumps(cfg))
        if device=='cpu': runtime_cfg['runtime']['tpu_spmd']=False
        spmd.initialize(runtime_cfg,device); dev=rt.choose_device(device); rt.configure_runtime(dev,runtime_cfg)
        rt.seed_everything(cfg['seed'],dev)
        model=g.GPT(g.GPTConfig(**cfg['model'])).to(dev); model.load_state_dict(state['model'])
        spmd.replicate_model(model); handles=g.make_handles(model,cfg)
        optimizers.load_optimizer_state_dict(handles,state['optimizers'])
        gen=torch.Generator(); gen.set_state(state['data_rng'])
        torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        if device!='cpu': rt.restore_accelerator_rng_state(state['accelerator_rng'],dev)
        debug.write(output/'runtime.json',rt.runtime_metadata(dev))
        check_finite(debug.optimizer_tensors(model,handles),output,'restored_state',start,dev)
        evaluate_splits(model,arrays,cfg,dev,output,start,'before_update_evaluation')
        # Restore saved RNG streams after the comparison probe; exact next training windows.
        torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        if device!='cpu': rt.restore_accelerator_rng_state(state['accelerator_rng'],dev)
        optimizers.zero_grad(handles); t=cfg['training']
        for handle in handles:
            handle.set_lr(optimizers.cosine_learning_rate(start,total_steps=t['schedule_steps'],
                warmup_steps=t['warmup_steps'],peak_lr=handle.peak_lr,min_lr=handle.min_lr))
        debug.write(output/'learning_rates.json',{h.role:h.lr for h in handles})
        debug.stage(output,'forward_backward',update)
        losses=[]; offsets=[]
        for _ in range(t['grad_accum_steps']):
            x,y=g.batch(arrays['train'],gen,t['batch_size'],cfg['model']['block_size'],dev,trace=offsets)
            _,loss=model(x,y); losses.append(loss.detach()); (loss/t['grad_accum_steps']).backward()
        debug.write(output/'training_inputs.json',{'update':update,'offsets':offsets,'context':cfg['model']['block_size']})
        spmd.replicate_gradients(model)
        norm=rt.gradient_norm(model.parameters())
        check_finite([('gradient/'+n,p.grad) for n,p in model.named_parameters() if p.grad is not None]
                     +[('loss/'+str(i),loss) for i,loss in enumerate(losses)]+[('gradient_norm',norm)],
                     output,'before_clipping',update,dev)
        g.require_finite_update(model,norm,losses,output,update)
        torch.nn.utils.clip_grad_norm_(model.parameters(),t['grad_clip'],foreach=False)
        check_finite([('gradient/'+n,p.grad) for n,p in model.named_parameters() if p.grad is not None],
                     output,'after_clipping',update,dev)
        for handle in handles:
            debug.stage(output,'applying_'+handle.role,update)
            handle.optimizer.step(); rt.mark_step(dev); rt.synchronize(dev)
            completed_roles.append(handle.role)
            check_finite(debug.optimizer_tensors(model,handles),output,'after_'+handle.role,update,dev)
        # Capture the updated state BEFORE evaluation can fail. Original checkpoint is read-only.
        snapshot(output,model,handles,cfg,gen,update,completed_roles)
        result=evaluate_splits(model,arrays,cfg,dev,output,update,'after_update_evaluation')
        report.update(status='one_update_passed',completed_roles=completed_roles,metrics=result,
                      interpretation='Instrumented replay only. Original TPU path and long-run stability remain unproven.')
        return output
    except Exception as exc:
        report.update(status='failed',error=str(exc),completed_roles=completed_roles)
        debug.failure(output,exc)  # Save diagnosis before attempting a potentially slow state copy.
        if model is not None and handles is not None and state is not None and gen is not None and completed_roles:
            try:
                if not (output/'update_state.pt').exists():
                    snapshot(output,model,handles,state['config'],gen,int(state['step'])+1,completed_roles)
            except Exception as snapshot_error: report['snapshot_error']=str(snapshot_error)
        raise
    finally:
        debug.write(output/'REPLAY_STATUS.json',report)


def snapshot(output,model,handles,cfg,gen,update,completed_roles):
    debug.stage(output,'saving_diagnostic_state',update)
    path=Path(output)/'update_state.pt'; tmp=path.with_suffix('.tmp')
    payload={'diagnostic_only':True,'resumable':False,'update':update,'completed_roles':list(completed_roles),
             'config':cfg,'model':model.state_dict(),'optimizers':optimizers.optimizer_state_dict(handles),
             'next_data_rng':gen.get_state()}
    with tmp.open('wb') as stream:
        torch.save(rt.tree_to_cpu(payload),stream); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True); parser.add_argument('--data-root',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--device',default='tpu',choices=('tpu','cpu'))
    args=parser.parse_args(); faulthandler.dump_traceback_later(300,repeat=True)
    try: replay(args.checkpoint,args.data_root,args.output,args.device)
    finally: faulthandler.cancel_dump_traceback_later()


if __name__=='__main__': main()
