#!/usr/bin/env python3
"""Read-only CPU gradient probes of trusted local nanoGPT checkpoints.

Loads model.py directly: does not initialize XLA, install packages, or change
training state. These are fresh probe gradients, NOT historical optimizer updates.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch


def load_model_module():
    path = Path(__file__).resolve().parents[1] / 'src/rg_nanogpt_one_head/model.py'
    spec = importlib.util.spec_from_file_location('gradient_probe_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_starts(length, block_size, count, seed):
    if length <= block_size:
        raise ValueError('token split is shorter than one context plus target')
    return np.random.default_rng(seed).integers(0, length-block_size, size=count)


def probe(model, tokens, starts, batch_size):
    """Gradient of mean token cross entropy over exactly these windows."""
    model.eval()
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    correct = 0
    batches = len(starts)//batch_size
    if batches < 1 or len(starts) % batch_size:
        raise ValueError('probe windows must be divisible by batch size')
    block = model.cfg.block_size
    for i in range(batches):
        positions = starts[i*batch_size:(i+1)*batch_size]
        x = torch.from_numpy(np.stack([tokens[s:s+block].astype(np.int64) for s in positions]))
        y = torch.from_numpy(np.stack([tokens[s+1:s+block+1].astype(np.int64) for s in positions]))
        logits, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite probe loss')
        (loss/batches).backward()
        total_loss += float(loss.detach())/batches
        correct += int((logits.detach().argmax(-1)==y).sum())
        del logits, loss
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'missing/nonfinite gradient: {name}')
        gradients[name] = parameter.grad.detach().clone()
    return gradients, total_loss, correct/(len(starts)*block)


def stats(weight, train, val):
    w = float(torch.linalg.vector_norm(weight))
    t = float(torch.linalg.vector_norm(train))
    v = float(torch.linalg.vector_norm(val))
    dot = float(torch.sum(train.double()*val.double()))
    return dict(weight_norm=w, train_grad_norm=t, val_grad_norm=v,
                train_grad_rms=t/math.sqrt(train.numel()),
                val_grad_rms=v/math.sqrt(val.numel()),
                train_grad_to_weight=t/w if w else float('nan'),
                train_val_dot=dot,
                train_val_cosine=dot/(t*v) if t*v else float('nan'),
                # First-order derivative along -g_train, NOT a Muon update.
                val_directional_derivative_negative_train_grad=-dot)


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--local-steps',type=int,nargs='+',default=[210000,220000,250000])
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batches',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=71903)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--save-gradients',action='store_true')
    a=p.parse_args(argv)
    if min(a.batches,a.batch_size,a.repeats,a.threads)<1:
        p.error('counts and threads must be positive')
    # Never overwrite an existing diagnostic or write into the training run.
    if a.output.resolve().is_relative_to(a.run_dir.resolve()):
        p.error('output must be outside the training run')
    paths=[]
    for step in sorted(set(a.local_steps)):
        hits=list((a.run_dir/'epoch_checkpoints').glob(f'model_epoch_*_step_{step:07d}.pt'))
        if len(hits)!=1:
            p.error(f'expected one permanent checkpoint at local step {step}; found {len(hits)}. Available: '+', '.join(f.name for f in sorted((a.run_dir/'epoch_checkpoints').glob('*.pt'))[-12:]))
        paths.append(hits[0])
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    module=load_model_module()
    metadata=json.loads((a.data_root/'meta.json').read_text())
    if metadata.get('dtype')!='uint16':
        raise ValueError('expected uint16 corpus')
    tokens={}
    identities={}
    for split in ['train','val']:
        record=metadata['files'][split]
        path=a.data_root/record['path']
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=record['sha256']:
            raise ValueError(f'{split} corpus hash mismatch')
        identities[split]=h.hexdigest()
        tokens[split]=np.memmap(path,dtype=np.uint16,mode='r')
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[];metrics=[];records=[];reference_config=None;sample_manifest=None
    for path in paths:
        print(f'[gradient-probe] loading {path.name}',flush=True)
        # Only use your own trusted experiment checkpoints (torch pickle).
        payload=torch.load(path,map_location='cpu',weights_only=False)
        cfg=payload['config']['model']
        if reference_config is not None and cfg!=reference_config:
            raise ValueError('model configuration differs between checkpoints')
        reference_config=cfg
        local=int(payload['step']);offset=int(payload['config'].get('continuation',{}).get('global_step_offset',0))
        step=local+offset
        if 'global_step' in payload and int(payload['global_step'])!=step:
            raise ValueError('checkpoint global step mismatch')
        model=module.GPT(module.GPTConfig(**cfg));model.load_state_dict(payload['model'],strict=True)
        for w in model.parameters():
            if not torch.isfinite(w).all():raise FloatingPointError('nonfinite checkpoint weight')
        records.append(dict(checkpoint=str(path),global_step=step,local_step=local,model_state_sha256=payload.get('model_state_sha256')))
        del payload
        labels={id(w):name for name,_,_,w in module.transformer_matrix_items(model)}
        samples={}
        for repeat in range(a.repeats):
            gradients={}
            for j,split in enumerate(['train','val']):
                starts=sample_starts(len(tokens[split]),cfg['block_size'],a.batches*a.batch_size,a.seed+repeat*2+j)
                samples[f'{repeat}:{split}']=starts.tolist()
                g,loss,acc=probe(model,tokens[split],starts,a.batch_size)
                gradients[split]=g
                metrics.append(dict(global_step=step,repeat=repeat,split=split,loss=loss,accuracy=acc,tokens=len(starts)*cfg['block_size']))
                print(f'[gradient-probe] global_step={step} repeat={repeat} {split} loss={loss:.6f} acc={acc:.4%}',flush=True)
            total_t=total_v=total_dot=total_w=0.0
            for name,w in model.named_parameters():
                s=stats(w.detach(),gradients['train'][name],gradients['val'][name])
                rows.append(dict(global_step=step,repeat=repeat,parameter=name,matrix=labels.get(id(w),name),numel=w.numel(),**s))
                total_t+=s['train_grad_norm']**2;total_v+=s['val_grad_norm']**2;total_dot+=s['train_val_dot'];total_w+=s['weight_norm']**2
            for r in rows:
                if r['global_step']==step and r['repeat']==repeat:
                    r['train_gradient_energy_fraction']=r['train_grad_norm']**2/total_t if total_t else float('nan')
            metrics.append(dict(global_step=step,repeat=repeat,split='global_gradient',loss=float('nan'),accuracy=float('nan'),tokens=0,
                                train_grad_norm=math.sqrt(total_t),val_grad_norm=math.sqrt(total_v),train_val_cosine=total_dot/math.sqrt(total_t*total_v) if total_t*total_v else float('nan')))
            if a.save_gradients:torch.save(gradients,a.output/f'gradients_{step}_repeat_{repeat}.pt')
            del gradients
        if sample_manifest is not None and samples!=sample_manifest:raise ValueError('probe windows changed')
        sample_manifest=samples
        del model
        write_csv(a.output/'per_parameter_gradients.csv',rows)
        fields=set().union(*(r.keys() for r in metrics))
        write_csv(a.output/'probe_metrics.csv',[{k:r.get(k,'') for k in sorted(fields)} for r in metrics])
    report=dict(checkpoints=records,model=reference_config,corpus_sha256=identities,sample_starts=sample_manifest,
                arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},torch_version=torch.__version__,
                interpretation='CPU eval-mode mean-loss gradients on fixed sampled train/validation windows. Negative train_val_cosine indicates local conflict for raw gradient descent, not necessarily MuonClip. No optimizer step is simulated; no test data used. Tied embedding/output weight counted once. Repeats are independent sampled probes, not independent training runs.')
    (a.output/'probe_manifest.json').write_text(json.dumps(report,indent=2))
    with zipfile.ZipFile(a.output/'gradient_probe_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.iterdir()):
            if f.suffix in ['.csv','.json','.pt']:z.write(f,f.name)
    print(f'[gradient-probe] DONE: {a.output / "gradient_probe_results.zip"}',flush=True)


if __name__=='__main__':main()
