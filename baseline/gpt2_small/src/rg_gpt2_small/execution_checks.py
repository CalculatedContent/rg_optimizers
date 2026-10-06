"""Scalar finite checks and synchronized evaluation shared with the TPU replay.

The extra graph boundaries are an experimental workaround, not a diagnosed fix.
"""
import json
from pathlib import Path

import torch
from rg_nanogpt_one_head import runtime as rt
from . import port_debug as debug


def check_finite(named, output, label, update, device, *, retain=True):
    """No stack/cat, extrema, or full-tensor host copies: materialize scalar flags."""
    debug.stage(output,label+'_started',update)
    checks=[]
    for name,tensor in named:
        value=tensor.detach()
        checks.append((name,tuple(value.shape),str(value.dtype),torch.isfinite(value).all(),
                       (value<0).any() if name.endswith('/exp_avg_sq') else None))
    rt.synchronize(torch.device(device))
    records=[]
    for name,shape,dtype,finite,negative in checks:
        records.append({'tensor':name,'shape':shape,'dtype':dtype,
                        'all_finite':bool(finite.cpu().item()),
                        'negative_second_moment':bool(negative.cpu().item()) if negative is not None else False})
    bad=[row for row in records if not row['all_finite'] or row['negative_second_moment']]
    report={'stage':label,'update':update,'records':records,'invalid_tensors':bad}
    prefix=f'{update:06d}' if retain or bad else 'latest'
    debug.write(Path(output)/'diagnostics'/f'{prefix}-{label}.json',report)
    if retain or bad: debug.xla_metrics(output,f'{update:06d}-{label}',device)
    if bad:
        debug.write(Path(output)/'FIRST_INVALID.json',report)
        raise RuntimeError(f'First invalid stage: {label}; tensors: '+', '.join(row['tensor'] for row in bad[:8]))
    debug.stage(output,label+'_passed',update)



@torch.no_grad()
def evaluate_splits(model, arrays, cfg, device, output, update, label):
    from .experiment import batch
    model.eval(); result={}
    try:
        for j,split in enumerate(('train','val','test')):
            generator=torch.Generator().manual_seed(cfg['seed']+20000+j)
            losses=[]; accuracies=[]
            for index in range(cfg['eval_batches']):
                stage=f'{label}_{split}_batch_{index}'
                debug.stage(output,stage+'_forward',update)
                offsets=[]
                x,y=batch(arrays[split],generator,cfg['training']['batch_size'],
                            cfg['model']['block_size'],device,trace=offsets)
                debug.write(Path(output)/'diagnostics'/f'{update:06d}-{stage}-inputs.json',
                            {'split':split,'context':cfg['model']['block_size'],'offsets':offsets})
                logits,loss=model(x,y)
                accuracy=(logits.argmax(-1)==y).float().mean()
                rt.mark_step(device)
                check_finite([('logits',logits),('loss',loss)],output,stage,update,device)
                losses.append(float(loss.cpu())); accuracies.append(float(accuracy.cpu()))
            result.update({split+'_nll':sum(losses)/len(losses),
                           split+'_token_error':1-sum(accuracies)/len(accuracies)})
            debug.write(Path(output)/(label+'.json'),result)
            print(json.dumps({'evaluation':label,'split':split,**result}),flush=True)
        return result
    finally:
        model.train()

