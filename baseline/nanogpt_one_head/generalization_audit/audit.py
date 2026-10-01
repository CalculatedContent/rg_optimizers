"""Fixed-document, checkpoint-matched generalization audit; CPU/CUDA only."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import torch
from pinned_model import GPT, GPTConfig

VERSION = '1.0'
MATRICES = ['Q', 'K', 'V', 'O', 'MLP_IN', 'MLP_OUT']

def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def state_hash(state):
    # Byte-for-byte the repository's model_state_sha256 convention.
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        meta = json.dumps(dict(name=str(name), shape=list(t.shape), dtype=str(t.dtype)),
                          sort_keys=True, separators=(',', ':')).encode()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        for data in (meta, raw): h.update(len(data).to_bytes(8, 'big')); h.update(data)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)

def document_windows(tokens, count, width, seed, eot=50256):
    """One window per uniformly selected eligible document, never across EOT."""
    ends = np.flatnonzero(tokens == eot)
    starts = np.r_[0, ends + 1]; ends = np.r_[ends, len(tokens)]
    eligible = np.flatnonzero(ends - starts >= width)
    if len(eligible) < count:
        raise ValueError(f'Only {len(eligible)} documents of length >= {width}; requested {count}')
    rng = np.random.default_rng(seed)
    docs = rng.choice(eligible, count, replace=False)
    offsets = np.array([rng.integers(starts[i], ends[i] - width + 1) for i in docs])
    windows = np.stack([np.asarray(tokens[o:o+width], dtype=np.int64) for o in offsets])
    assert not np.any(windows == eot)
    return windows, docs, offsets

def per_document_scores(logits, targets, rare_counts, rare_threshold=10):
    """Full-vocabulary proper scores, ranks and calibration sufficient statistics."""
    z = logits.float()
    if not torch.isfinite(z).all(): raise ValueError('Nonfinite logits')
    logp = z.log_softmax(-1); p = logp.exp()
    true_logp = logp.gather(-1, targets[..., None]).squeeze(-1)
    confidence, predicted = p.max(-1)
    correct = predicted.eq(targets)
    true_logits = z.gather(-1, targets[..., None])
    # Midrank handles tied logits without giving every tied token rank 1.
    rank = 1 + (z > true_logits).sum(-1) + 0.5 * ((z == true_logits).sum(-1) - 1)
    top5 = z.topk(min(5, z.shape[-1]), dim=-1).indices.eq(targets[..., None]).any(-1)
    nll = -true_logp
    arrays = {
        'nll': nll.mean(-1),
        'token_error_pct': 100 * (~correct).float().mean(-1),
        'top5_error_pct': 100 * (~top5).float().mean(-1),
        'mrr': rank.reciprocal().mean(-1),
        'brier': (p.square().sum(-1) - 2*true_logp.exp() + 1).mean(-1),
        'entropy_nats': -(p * logp).sum(-1).mean(-1),
        'confident_wrong_pct': 100*((confidence >= 0.5) & ~correct).float().mean(-1),
        'p95_token_nll': torch.quantile(nll, .95, dim=-1),
    }
    a = {k: v.double().cpu().numpy() for k,v in arrays.items()}
    rare = torch.as_tensor(rare_counts, device=targets.device)[targets] <= rare_threshold
    a['_rare_nll_sum'] = (nll * rare).sum(-1).double().cpu().numpy()
    a['_rare_count'] = rare.sum(-1).double().cpu().numpy()
    bins = torch.clamp((confidence*15).long(), max=14)
    for b in range(15):
        mask = bins.eq(b)
        for kind, values in [('count', mask), ('confidence', confidence*mask), ('correct', correct*mask)]:
            a[f'_bin_{b}_{kind}'] = values.sum(-1).double().cpu().numpy()
    return a

@torch.inference_mode()
def teacher_metrics(model, windows, counts, batch, device):
    collected = {}
    for start in range(0, len(windows), batch):
        w = torch.as_tensor(windows[start:start+batch], device=device)
        logits, _ = model(w[:, :-1])
        scores = per_document_scores(logits, w[:, 1:], counts)
        for k,v in scores.items(): collected.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k,v in collected.items()}

def lcs_f1(a, b):
    prev = [0] * (len(b)+1)
    for token in a:
        row = [0]
        for j, other in enumerate(b):
            row.append(prev[j]+1 if token == other else max(prev[j+1], row[-1]))
        prev = row
    return 2*prev[-1] / max(1, len(a)+len(b))

def repetition3(tokens):
    grams = list(zip(tokens, tokens[1:], tokens[2:]))
    return 1-len(set(grams))/len(grams) if grams else 0.

@torch.inference_mode()
def generation_metrics(model, windows, batch, device, prompt_tokens, new_tokens, seed, encoder):
    from sacrebleu.metrics import BLEU, CHRF
    bleu = BLEU(tokenize='13a', effective_order=True)
    chrf = CHRF()
    output = {}; samples = []
    def push(key, values): output.setdefault(key, []).extend(np.asarray(values).tolist())
    rng = np.random.default_rng(seed)
    # Keep the immediate previous token intact; ablate the earlier context.
    shuffled = windows[:, :prompt_tokens].copy()
    for row in shuffled: row[:-1] = row[:-1][rng.permutation(prompt_tokens-1)]
    for start in range(0, len(windows), batch):
        w = torch.as_tensor(windows[start:start+batch], device=device)
        prompts = w[:, :prompt_tokens]; refs = w[:, prompt_tokens:prompt_tokens+new_tokens]
        predicted = model.generate_greedy(prompts, new_tokens)[:, -new_tokens:]
        correct = predicted.eq(refs).cpu().numpy()
        push('free_token_error_pct', 100*(1-correct.mean(-1)))
        push('free_exact_failure_pct', 100*(1-correct.all(-1)))
        push('free_matching_prefix_tokens', np.cumprod(correct, axis=-1).sum(-1))
        for h in sorted(set([1, min(8,new_tokens), min(16,new_tokens), new_tokens])):
            push(f'free_error_first_{h}_pct', 100*(1-correct[:, :h].mean(-1)))
        # Score identical references under correct and shuffled prompts.
        conditional = []
        for pfx in [prompts, torch.as_tensor(shuffled[start:start+batch], device=device)]:
            seq = torch.cat([pfx, refs], dim=1)
            logits, _ = model(seq[:, :-1])
            z = logits[:, prompt_tokens-1:prompt_tokens-1+new_tokens].float().log_softmax(-1)
            conditional.append(-z.gather(-1, refs[..., None]).squeeze(-1).mean(-1).cpu().numpy())
        push('reference_continuation_nll', conditional[0])
        push('shuffled_context_nll_increase', conditional[1]-conditional[0])
        for i,(pred,ref) in enumerate(zip(predicted.cpu().tolist(), refs.cpu().tolist())):
            hyp, truth = encoder.decode(pred), encoder.decode(ref)
            push('sentence_bleu', [bleu.sentence_score(hyp, [truth]).score])
            push('sentence_chrf', [chrf.sentence_score(hyp, [truth]).score])
            push('token_rouge_l_f1', [lcs_f1(pred, ref)])
            push('repeat3_fraction', [repetition3(pred)])
            push('excess_repeat3_fraction', [repetition3(pred)-repetition3(ref)])
            samples.append(dict(probe_index=start+i, prompt=encoder.decode(prompts[i].cpu().tolist()),
                                reference=truth, generated=hyp, generated_tokens=pred, reference_tokens=ref))
    return {k: np.array(v, dtype=float) for k,v in output.items()}, samples

def estimate(arrays, indices):
    """Aggregate whole-document draws, preserving within-document dependence."""
    out = {k:float(v[indices].mean()) for k,v in arrays.items() if not k.startswith('_')}
    if 'nll' in out:
        out['perplexity'] = math.exp(out['nll']); out['bits_per_token'] = out['nll']/math.log(2)
        counts = sum(arrays[f'_bin_{b}_count'][indices].sum() for b in range(15))
        out['ece_pct'] = float(100*sum(abs(arrays[f'_bin_{b}_confidence'][indices].sum()-arrays[f'_bin_{b}_correct'][indices].sum()) for b in range(15))/counts)
        nrare = arrays['_rare_count'][indices].sum()
        if nrare: out['rare_token_nll'] = float(arrays['_rare_nll_sum'][indices].sum()/nrare)
    return out

def summarize(arrays, draws, seed):
    n = len(next(iter(arrays.values())))
    point = estimate(arrays, np.arange(n)); rng = np.random.default_rng(seed)
    boot = [estimate(arrays, rng.integers(n, size=n)) for _ in range(draws)]
    results = {}
    for key,val in point.items():
        values = [r[key] for r in boot if key in r]
        lo,hi = np.percentile(values,[2.5,97.5])
        results[key] = dict(value=val, low=float(lo), high=float(hi), n_documents=n)
    return results

def corpus_overlap(samples, draws, seed):
    from sacrebleu.metrics import BLEU, CHRF
    hypotheses=[s['generated'] for s in samples];references=[s['reference'] for s in samples]
    rng=np.random.default_rng(seed);n=len(samples)
    indices=[rng.integers(n,size=n) for _ in range(min(draws,200))]
    result={}
    for name,scorer in [('corpus_bleu',BLEU(tokenize='13a',effective_order=True)),('corpus_chrf',CHRF())]:
        point=scorer.corpus_score(hypotheses,[references]).score
        values=[scorer.corpus_score([hypotheses[i] for i in ix],[[references[i] for i in ix]]).score for ix in indices]
        lo,hi=np.percentile(values,[2.5,97.5])
        result[name]=dict(value=float(point),low=float(lo),high=float(hi),n_documents=n)
    return result

def gaps(test, train, draws, seed):
    rng = np.random.default_rng(seed); result = {}
    for key in ['nll','token_error_pct','brier']:
        x,y = test[key],train[key]; values = []
        for _ in range(draws):
            values.append(x[rng.integers(len(x),size=len(x))].mean()-y[rng.integers(len(y),size=len(y))].mean())
        lo,hi = np.percentile(values,[2.5,97.5])
        result['test_minus_train_'+key] = dict(value=float(x.mean()-y.mean()),low=float(lo),high=float(hi),n_documents=len(x))
    return result

def spectral_features(rows):
    expected = {'L00_W_'+m for m in MATRICES}
    if len(rows)!=6 or set(rows.matrix_name)!=expected or not rows.status.eq('success').all():
        raise ValueError('Requires all six successful one-block matrix fits')
    out = {}
    for variant,col in [('raw','alpha_raw'),('clipped','alpha_clip_xmax')]:
        vals = rows[col].to_numpy(float)
        if not np.isfinite(vals).all() or (vals<=0).any(): raise ValueError('Invalid alpha')
        out[f'{variant}_min'] = float(vals.min()); out[f'{variant}_mean'] = float(vals.mean())
        for _,r in rows.iterrows(): out[f'{variant}_{r.matrix_name.removeprefix("L00_W_")}'] = float(r[col])
    return out

def load_checked(path, spectral_rows, manifest):
    # Only use checkpoints generated by your own training run (pickle format).
    ck = torch.load(path, map_location='cpu', weights_only=False)
    state = ck['model']; digest = state_hash(state)
    if digest != ck.get('model_state_sha256'): raise ValueError('Checkpoint tensor hash mismatch')
    if not spectral_rows.model_state_sha256.eq(digest).all(): raise ValueError('Spectral/model hash mismatch')
    if not spectral_rows.protocol_fingerprint.eq(ck['fingerprint']).all(): raise ValueError('Protocol mismatch')
    if not spectral_rows.run_seed.eq(ck['seed']).all(): raise ValueError('Seed mismatch')
    if ck['config']['model'] != manifest['model']: raise ValueError('Model config mismatch')
    local = int(ck['step']); offset = int(ck['config'].get('continuation',{}).get('global_step_offset',0))
    if not spectral_rows.step.eq(local).all(): raise ValueError('Checkpoint step mismatch')
    if offset != int(manifest.get('continuation',{}).get('global_step_offset',0)): raise ValueError('Step offset mismatch')
    if any(not torch.isfinite(v).all() for v in state.values()): raise ValueError('Nonfinite weights')
    model = GPT(GPTConfig(**ck['config']['model']))
    model.load_state_dict(state, strict=True); model.eval()
    return model, digest, local+offset

def export_tables(out):
    records = [json.loads(p.read_text()) for p in sorted((out/'checkpoints').glob('*.json'))]
    tidy=[]; wide=[]
    for rec in records:
        row = dict(global_step=rec['global_step'],model_state_sha256=rec['model_state_sha256'],**rec['alpha'])
        for key,v in rec['metrics'].items():
            row[key]=v['value']
            tidy.append(dict(global_step=rec['global_step'],metric=key,**v))
        wide.append(row)
    for name,values in [('metrics',wide),('uncertainty',tidy)]:
        tmp=out/(name+'.csv.tmp');pd.DataFrame(values).to_csv(tmp,index=False);tmp.replace(out/(name+'.csv'))

def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    d = pd.read_csv(out/'metrics.csv').sort_values('global_step')
    u = pd.read_csv(out/'uncertainty.csv')
    metrics = u.metric.unique(); root=out/'plots';root.mkdir(exist_ok=True)
    correlations=[]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(out/'all_metrics_vs_alpha.pdf') as pdf:
        for metric in metrics:
            ci=u[u.metric.eq(metric)].set_index('global_step').reindex(d.global_step)
            for variant in ['raw','clipped']:
                fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
                fig.suptitle(f'{metric} vs {variant} alpha\n95% document-bootstrap intervals; epoch 0 excluded',fontsize=13)
                for ax,feature in zip(axes.flat,['min','mean']+MATRICES):
                    x=d[f'{variant}_{feature}'];y=d[metric]
                    ax.vlines(x,ci.low,ci.high,color='#a9b6c5',alpha=.4,lw=.7)
                    sc=ax.scatter(x,y,c=d.global_step/1e6,cmap='viridis',s=23,zorder=3)
                    ax.set_xlabel(feature+' alpha');ax.set_ylabel(metric);ax.grid(alpha=.2)
                    if x.nunique()>1 and y.nunique()>1:
                        r=x.corr(y);rho=x.corr(y,method='spearman')
                        dx=x.diff();dy=y.diff()
                        dr=dx.corr(dy) if dx.nunique()>1 and dy.nunique()>1 else np.nan
                    else:r=rho=dr=np.nan
                    correlations.append(dict(metric=metric,variant=variant,alpha=feature,n=len(d),pearson=r,spearman=rho,first_difference_pearson=dr))
                fig.colorbar(sc,ax=list(axes.flat),label='Cumulative steps (millions)',shrink=.7)
                pdf.savefig(fig);fig.savefig(root/f'{metric}__{variant}.png',dpi=120);plt.close(fig)
    pd.DataFrame(correlations).to_csv(out/'exploratory_correlations.csv',index=False)

def run(args):
    import tiktoken
    import sacrebleu
    torch.set_num_threads(args.threads)
    if args.device=='cpu': torch.set_num_interop_threads(1)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    # One writer, including across accidental double launches.
    import fcntl
    with (out/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((args.run_dir/'manifest.json').read_text())
        offset=int(manifest.get('continuation',{}).get('global_step_offset',0))
        if manifest['model']['n_layer']!=1:raise ValueError('This audit is for the current one-block model')
        block=int(manifest['model']['block_size'])
        if args.prompt_tokens+args.new_tokens>block+1:raise ValueError('Continuation exceeds probe length')
        spectra=pd.read_csv(args.run_dir/'spectral/layers.csv')
        counts=None;windows={};sources={};probe_info={}
        for i,split in enumerate(['train','test']):
            path=args.data_root/(split+'.bin');actual=sha_file(path)
            if actual!=manifest['data_metadata']['files'][split]['sha256']:raise ValueError(f'{split} data hash mismatch')
            sources[split]=actual;tokens=np.memmap(path,dtype=np.uint16,mode='r')
            windows[split],docs,starts=document_windows(tokens,args.documents,block+1,args.probe_seed+i)
            probe_info[split]=dict(document_ids=docs.tolist(),token_offsets=starts.tolist())
            if split=='train':
                counts=np.zeros(manifest['model']['vocab_size'],dtype=np.int64)
                for j in range(0,len(tokens),1000000):counts+=np.bincount(tokens[j:j+1000000],minlength=len(counts))
            del tokens
        encoder=tiktoken.get_encoding('gpt2')
        sources['model_source']=sha_file(Path(__file__).with_name('pinned_model.py'))
        sources['audit_source']=sha_file(__file__)
        settings={k:v for k,v in vars(args).items() if k not in ['output','run_dir','data_root','command']}
        protocol=dict(version=VERSION,sources=sources,settings=settings,probes=probe_info,
                      torch_version=torch.__version__,sacrebleu_version=sacrebleu.__version__,
                      numpy_version=np.__version__,tiktoken_version=tiktoken.__version__)
        protocol_path=out/'protocol.json'
        if protocol_path.exists():
            old=json.loads(protocol_path.read_text())
            if old['protocol']!=protocol:raise ValueError('Audit settings/data/code changed; choose a new output directory')
            selected=old['checkpoints']
        else:
            available={}
            for p in sorted((args.run_dir/'epoch_checkpoints').glob('model_epoch_*_step_*.pt')):
                step=int(re.search(r'_step_(\d+)\.pt$',p.name).group(1))
                if step+offset<=0 or step+offset<args.min_step or step+offset>args.max_step:continue
                rows=spectra[spectra.step.eq(step)]
                if len(rows)!=6 or rows.model_state_sha256.nunique()!=1:continue
                spectral_features(rows)
                available[step]=p
            steps=sorted(available)
            if not steps:raise ValueError('No retained epoch checkpoints with matching spectra')
            take=np.unique(np.linspace(0,len(steps)-1,min(len(steps),args.max_checkpoints)).round().astype(int))
            selected=[dict(step=steps[i],path=str(available[steps[i]].resolve())) for i in take]
            atomic_json(protocol_path,dict(protocol=protocol,checkpoints=selected,
                selection='Evenly spaced by checkpoint index, independent of alpha and performance',
                available_checkpoint_count=len(steps),note='Exploratory monitoring; not a fresh confirmatory test.'))
        print(f'{len(selected)} checkpoints; {args.documents} documents/split; {args.generation_documents} generation documents; device={args.device}',flush=True)
        for number,item in enumerate(selected,1):
            step=item['step'];global_step=step+offset
            dest=out/'checkpoints'/f'{global_step:010d}.json'
            if dest.exists():print(f'[{number}/{len(selected)}] resume: {global_step} complete',flush=True);continue
            begin=time.monotonic();rows=spectra[spectra.step.eq(step)]
            model,digest,checked_step=load_checked(item['path'],rows,manifest)
            assert checked_step==global_step
            model.to(args.device)
            arrays={};result={}
            for split in ['train','test']:
                print(f'[{number}/{len(selected)}] step={global_step} {split} teacher metrics',flush=True)
                arrays[split]=teacher_metrics(model,windows[split],counts,args.batch_size,args.device)
                for k,v in summarize(arrays[split],args.bootstrap,args.probe_seed+100).items():result[f'{split}_{k}']=v
            result.update(gaps(arrays['test'],arrays['train'],args.bootstrap,args.probe_seed+101))
            print(f'[{number}/{len(selected)}] step={global_step} free generation',flush=True)
            gen,samples=generation_metrics(model,windows['test'][:args.generation_documents],args.batch_size,args.device,args.prompt_tokens,args.new_tokens,args.probe_seed+200,encoder)
            for k,v in summarize(gen,args.bootstrap,args.probe_seed+201).items():result[f'test_{k}']=v
            for k,v in corpus_overlap(samples,args.bootstrap,args.probe_seed+202).items():result[f'test_{k}']=v
            (out/'per_document').mkdir(exist_ok=True)
            np.savez_compressed(out/'per_document'/f'{global_step:010d}.npz',
                **{f'{split}__{k}':v for split,a in arrays.items() for k,v in a.items()},
                **{f'generation__{k}':v for k,v in gen.items()})
            atomic_json(out/'generations'/f'{global_step:010d}.json',samples)
            atomic_json(dest,dict(global_step=global_step,model_state_sha256=digest,alpha=spectral_features(rows),metrics=result,elapsed_seconds=time.monotonic()-begin))
            export_tables(out)
            print(f'[{number}/{len(selected)}] saved step={global_step}; elapsed={time.monotonic()-begin:.1f}s',flush=True)
            del model
        export_tables(out);plot(out)
        atomic_json(out/'DONE.json',dict(checkpoints=len(selected),completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('run');q.add_argument('--run-dir',type=Path,required=True);q.add_argument('--data-root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--device',choices=['cpu','cuda'],default='cpu');q.add_argument('--threads',type=int,default=2)
    q.add_argument('--max-checkpoints',type=int,default=16);q.add_argument('--documents',type=int,default=128)
    q.add_argument('--generation-documents',type=int,default=64);q.add_argument('--batch-size',type=int,default=2)
    q.add_argument('--prompt-tokens',type=int,default=64);q.add_argument('--new-tokens',type=int,default=32)
    q.add_argument('--bootstrap',type=int,default=500);q.add_argument('--probe-seed',type=int,default=83001)
    q.add_argument('--min-step',type=int,default=1);q.add_argument('--max-step',type=int,default=2**63-1)
    q=sub.add_parser('plot');q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='plot':plot(a.output);return
    for k in ['threads','max_checkpoints','documents','generation_documents','batch_size','prompt_tokens','new_tokens','bootstrap']:
        if getattr(a,k)<1:p.error(k+' must be positive')
    if a.generation_documents>a.documents:p.error('generation-documents must not exceed documents')
    run(a)

if __name__=='__main__':main()
