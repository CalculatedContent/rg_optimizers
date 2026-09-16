"""Behavioral metrics independent of the spectral controller."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
import math
import numpy as np
import torch
import torch.nn.functional as F


def hidden(model, ids, offset=0):
    if ids.shape[1]+offset>model.cfg.block_size or offset<0:
        raise ValueError('Context/position bound exceeded.')
    positions = torch.arange(offset,offset+ids.shape[1],device=ids.device)
    x = model.drop(model.token_embedding(ids)+model.position_embedding(positions))
    for block in model.blocks:
        x = block(x)
    return model.ln_f(x)


def next_logits(model, ids, offset=0):
    return model.lm_head(hidden(model,ids,offset)[:,-1,:])


def suffix_losses(model, records):
    """No padding and no full-vocabulary logits at masked prompt positions."""
    if not records or len({(len(r.prefix),len(r.target),r.offset) for r in records})!=1:
        raise ValueError('Scoring batches need identical lengths and positions.')
    device = next(model.parameters()).device
    full = torch.tensor([r.prefix+r.target for r in records],device=device)
    start = len(records[0].prefix)-1
    logits = model.lm_head(hidden(model,full[:,:-1],records[0].offset)[:,start:,:])
    labels = full[:,start+1:]
    losses = F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),reduction='none')
    return losses.reshape(len(records),-1), logits.argmax(-1), labels


def batches(records, size):
    groups = defaultdict(list)
    for r in records:
        groups[(len(r.prefix),len(r.target),r.offset)].append(r)
    for rows in groups.values():
        for start in range(0,len(rows),size):
            yield rows[start:start+size]


@torch.inference_mode()
def greedy(model, prefixes, length, offset=0):
    ids = torch.tensor(prefixes,device=next(model.parameters()).device)
    result = []
    for _ in range(length):
        token = next_logits(model,ids,offset).argmax(-1,keepdim=True)
        result.append(token)
        ids = torch.cat([ids,token],dim=1)
    return torch.cat(result,dim=1).cpu().numpy()


def recall(predicted, target):
    if np.asarray(predicted).shape!=np.asarray(target).shape:
        raise ValueError('Sequence lengths differ.')
    hit = np.asarray(predicted)==np.asarray(target)
    if hit.ndim!=1 or not len(hit):
        raise ValueError('Nonempty one-dimensional equal-length sequences required.')
    missed = np.flatnonzero(~hit)
    return {'exact_match':float(hit.all()),'token_match':float(hit.mean()),
            'longest_prefix':int(missed[0]) if len(missed) else len(hit)}


@torch.inference_mode()
def evaluate(model, records, batch_size):
    previous = model.training; model.eval(); rows=[]
    try:
        for chunk in batches(records,batch_size):
            losses, predicted, target = suffix_losses(model,chunk)
            nll = losses.mean(1).cpu().tolist()
            teacher = predicted.eq(target).float().mean(1).cpu().tolist()
            generation = greedy(model,[r.prefix for r in chunk],len(chunk[0].target),chunk[0].offset)
            for i,r in enumerate(chunk):
                if not math.isfinite(nll[i]):
                    raise FloatingPointError('Nonfinite evaluation loss.')
                rows.append({'id':r.id,'cohort':r.cohort,'dose':r.dose,
                             'prefix_tokens':len(r.prefix),'target_tokens':len(r.target),
                             'nll':nll[i], 'perplexity':math.exp(nll[i]) if nll[i]<700 else None,
                             'teacher_accuracy':teacher[i], **recall(generation[i],r.target)})
        return rows
    finally:
        model.train(previous)


def rank_exposure(scores, index):
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all() or not 0<=index<len(scores):
        raise ValueError('Invalid complete-universe score vector.')
    low=1+int((scores<scores[index]).sum()); high=int((scores<=scores[index]).sum())
    return {'rank_min':low,'rank_max':high,'candidate_count':len(scores),
            'exposure_lower':math.log2(len(scores)/high), 'exposure_upper':math.log2(len(scores)/low)}


@torch.inference_mode()
def exposure(model, record, alphabet, batch_size):
    """Exhaustive equal-length universe via its autoregressive prefix tree.

    For a 16-token alphabet and length 3: score 1+16+256 contexts,
    yielding the exact summed conditional NLLs of all 4096 suffixes.
    """
    if len(set(alphabet))!=len(alphabet) or any(t not in alphabet for t in record.target):
        raise ValueError('Invalid declared canary universe.')
    previous=model.training; model.eval()
    try:
        tails=[()]; scores=np.zeros(1)
        for _ in range(len(record.target)):
            probabilities=[]
            for start in range(0,len(tails),batch_size):
                ids=torch.tensor([record.prefix+t for t in tails[start:start+batch_size]],
                                 device=next(model.parameters()).device)
                lp=next_logits(model,ids,record.offset).log_softmax(-1)[:,alphabet]
                probabilities.extend(lp.cpu().double().tolist())
            scores=(scores[:,None]-np.asarray(probabilities)).reshape(-1)
            tails=[t+(token,) for t in tails for token in alphabet]
        index=tails.index(record.target)
        return {**rank_exposure(scores,index),'target_index':index}, scores
    finally:
        model.train(previous)


def compression_probes(records, grid):
    # Prefix truncation preserves the absolute target position. Position offset
    # is supplied context, so this is NOT unrestricted adversarial compression.
    return [replace(r,prefix=r.prefix[-p:],offset=len(r.prefix)-p)
            for r in records if r.cohort=='long' for p in grid if p<=len(r.prefix)]


def compress_summary(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[row['id']].append(row)
    result=[]
    for key, group in groups.items():
        success=[r['prefix_tokens'] for r in group if r['exact_match']==1]
        shortest=min(success) if success else None
        result.append({'id':key,'dose':group[0]['dose'], 'shortest_prefix':shortest,
                       'ratio':group[0]['target_tokens']/shortest if shortest else None,
                       'censored':shortest is None, 'tested_prefixes':len(group)})
    return result
