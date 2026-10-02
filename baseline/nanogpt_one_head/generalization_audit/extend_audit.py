"""Extend a completed audit using its exact scorer, data, probes and settings."""
import argparse, fcntl, json, os, re, shutil, subprocess, sys, tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch, sacrebleu, tiktoken
import audit
from python_csv import install, snapshot_bytes
install()

SELECTION = {'min_step', 'max_step', 'max_checkpoints'}

def verify_reference(ref, data):
    p = json.loads((ref/'protocol.json').read_text())['protocol']
    for name, version in [('torch',torch.__version__),('numpy',np.__version__),
                          ('sacrebleu',sacrebleu.__version__),('tiktoken',tiktoken.__version__)]:
        if p[name+'_version'] != version:
            raise ValueError(f'{name} version differs from original audit')
    for name, path in [('audit_source',Path(audit.__file__)),
                       ('model_source',Path(audit.__file__).with_name('pinned_model.py')),
                       ('train',data/'train.bin'),('test',data/'test.bin')]:
        if audit.sha_file(path) != p['sources'][name]:
            raise ValueError(f'{name} hash differs from original audit')
    for i, split in enumerate(['train','test']):
        tokens=np.memmap(data/(split+'.bin'),dtype=np.uint16,mode='r')
        _, ids, starts=audit.document_windows(tokens,p['settings']['documents'],257,p['settings']['probe_seed']+i)
        if dict(document_ids=ids.tolist(),token_offsets=starts.tolist()) != p['probes'][split]:
            raise ValueError(f'{split} probe differs from original audit')
    return p

def verify_protocol(original, actual):
    for key in original:
        if key == 'settings':
            a={k:v for k,v in original[key].items() if k not in SELECTION}
            b={k:v for k,v in actual[key].items() if k not in SELECTION}
            if a != b: raise ValueError('Evaluation settings differ')
        elif original[key] != actual[key]:
            raise ValueError(f'Audit protocol differs: {key}')

def discover(root, cutoff, maximum):
    available={}
    for run in sorted(root.glob('segments/segment_*/muon_clip/seed_1337')):
        man=json.loads((run/'manifest.json').read_text()); offset=man['continuation']['global_step_offset']
        if offset > maximum:continue
        print(f'Reading spectra: {run}',flush=True)
        spectra=pd.read_csv(run/'spectral/layers.csv')
        for file in sorted((run/'epoch_checkpoints').glob('model_epoch_*_step_*.pt')):
            step=int(re.search(r'_step_(\d+)\.pt$',file.name).group(1)); glob=offset+step
            if not cutoff < glob <= maximum:continue
            rows=spectra[spectra.step.eq(step)]
            if len(rows)!=6 or rows.model_state_sha256.nunique()!=1:continue
            audit.spectral_features(rows)
            digest=rows.model_state_sha256.iloc[0]
            if glob in available and available[glob]['model_hash']!=digest:raise ValueError('Conflicting boundary model hashes')
            available.setdefault(glob,dict(global_step=glob,local_step=step,run=str(run.resolve()),path=str(file.resolve()),model_hash=digest))
    return [available[k] for k in sorted(available)]

def merge(ref, outputs, out, cutoff):
    out.mkdir(exist_ok=True)
    for src in [ref]+outputs:
        for folder in ['checkpoints','per_document','generations']:
            dst=out/folder;dst.mkdir(exist_ok=True)
            for file in sorted((src/folder).glob('*')):
                target=dst/file.name
                if target.exists() and target.read_bytes()!=file.read_bytes():raise ValueError(f'Conflicting result: {file.name}')
                if not target.exists():shutil.copy2(file,target)
    audit.export_tables(out)
    d=pd.read_csv(out/'metrics.csv').sort_values('global_step')
    records=[]
    for label,g in [('previous',d[d.global_step<=cutoff]),('new',d[d.global_step>cutoff]),('combined',d)]:
        for metric in [c for c in d if c.startswith('test_')]:
            r=g.raw_mean.corr(g[metric]) if g[metric].nunique()>1 and len(g)>2 else None
            records.append(dict(period=label,metric=metric,n=len(g),pearson_raw_mean=r))
    pd.DataFrame(records).to_csv(out/'unadjusted_correlations.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    metrics=['test_token_error_pct','test_nll','test_mrr','test_top5_error_pct','test_corpus_bleu','test_corpus_chrf']
    u=pd.read_csv(out/'uncertainty.csv')
    for ax,metric in zip(axes.flat,metrics):
        ci=u[u.metric.eq(metric)].set_index('global_step').reindex(d.global_step)
        ax.vlines(d.raw_mean,ci.low,ci.high,color='lightgray',lw=.7)
        for label,mask,color in [('Previous',d.global_step<=cutoff,'#777777'),('New',d.global_step>cutoff,'#167c80')]:
            g=d[mask];ax.scatter(g.raw_mean,g[metric],label=label,color=color,s=25)
        x=np.linspace(d.raw_mean.min(),d.raw_mean.max(),100)
        ax.plot(x,np.polyval(np.polyfit(d.raw_mean,d[metric],1),x),color='black',lw=1)
        ax.set(title=f'{metric}\nCombined r={d.raw_mean.corr(d[metric]):+.3f}',xlabel='Mean raw alpha',ylabel=metric)
        ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Exact original document probe; no detrending\nBars: original 95% document-bootstrap intervals; line: combined OLS')
    fig.savefig(out/'raw_alpha_regression.png',dpi=160);fig.savefig(out/'raw_alpha_regression.pdf');plt.close(fig)
    for row in records:
        if row['metric']=='test_token_error_pct':print(row,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,default=Path('/mnt/disks/rg-data/generalization_audit/results'))
    p.add_argument('--root',type=Path,default=Path('/mnt/disks/rg-data/muonclip-extended'))
    p.add_argument('--data',type=Path,default=Path('/mnt/disks/rg-data/rg-nanogpt-one-head/data'))
    p.add_argument('--output',type=Path,default=Path('/mnt/disks/rg-data/generalization_audit/exact_extension_20261002'))
    p.add_argument('--max-step',type=int,default=5100000)
    p.add_argument('--count',type=int,default=16)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.count<1:raise ValueError('count must be positive')
    with (out/'extension.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (args.reference/'DONE.json').exists():raise ValueError('Original audit must be complete')
        print('Verifying original probe and scorer; CSV parser: Python, snapshot reads.',flush=True)
        original=verify_reference(args.reference,args.data)
        old=pd.read_csv(args.reference/'metrics.csv');cutoff=int(old.global_step.max())
        planfile=out/'selection.json'
        identity=dict(reference_protocol_sha256=audit.sha_file(args.reference/'protocol.json'),
                      reference_metrics_sha256=audit.sha_file(args.reference/'metrics.csv'),
                      cutoff=cutoff,max_step=args.max_step,count=args.count)
        if planfile.exists():
            plan=json.loads(planfile.read_text())
            if plan['identity']!=identity:raise ValueError('Selection changed; use new output directory')
        else:
            available=discover(args.root,cutoff,args.max_step)
            if not available:raise ValueError('No newer retained checkpoints')
            take=np.unique(np.linspace(0,len(available)-1,min(args.count,len(available))).round().astype(int))
            plan=dict(identity=identity,selected=[available[i] for i in take],available=len(available))
            # Snapshot metadata and hard-link selected weights before training can prune them.
            for item in plan['selected']:
                run=Path(item['run']);stage=out/'staged'/run.parents[1].name
                (stage/'epoch_checkpoints').mkdir(parents=True,exist_ok=True);(stage/'spectral').mkdir(exist_ok=True)
                shutil.copy2(run/'manifest.json',stage/'manifest.json')
                (stage/'spectral/layers.csv').write_bytes(snapshot_bytes(run/'spectral/layers.csv'))
                target=stage/'epoch_checkpoints'/Path(item['path']).name
                if not target.exists():os.link(item['path'],target)
                item['staged_run']=str(stage)
            audit.atomic_json(planfile,plan)
        print('Exact original source, versions, data hashes, document IDs and offsets verified.',flush=True)
        print('Selected new steps:',[x['global_step'] for x in plan['selected']],flush=True)
        outputs=[]
        for stage in sorted({x['staged_run'] for x in plan['selected']}):
            run=Path(stage);dest=out/'segments'/run.name;dest.parent.mkdir(exist_ok=True)
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('python_csv.py').resolve()),'run','--run-dir',str(run),
                 '--data-root',str(args.data),'--output',str(dest)]
            settings=dict(original['settings']);settings.update(min_step=cutoff+1,max_step=args.max_step,max_checkpoints=args.count)
            for key,value in settings.items():cmd += ['--'+key.replace('_','-'),str(value)]
            subprocess.run(cmd,check=True)
            verify_protocol(original,json.loads((dest/'protocol.json').read_text())['protocol'])
            outputs.append(dest)
        combined=out/'combined';merge(args.reference,outputs,combined,cutoff)
        # Include provenance plus per-document outputs, never checkpoint weights.
        audit.atomic_json(combined/'csv_reader.json',dict(engine='python',snapshot_reads=True,scorer_source_unchanged=True))
        archive=out.parent/'exact_probe_results.tgz';temporary=archive.with_suffix('.partial')
        with tarfile.open(temporary,'w:gz') as tf:
            tf.add(combined,arcname='combined');tf.add(planfile,arcname='selection.json')
            tf.add(args.reference/'protocol.json',arcname='reference_protocol.json')
            for result in outputs:tf.add(result/'protocol.json',arcname=f'protocols/{result.name}.json')
        temporary.replace(archive)
        audit.atomic_json(out/'DONE.json',dict(archive=str(archive),new_checkpoints=len(plan['selected'])))
        print(f'COMPLETE: {archive}',flush=True)
if __name__=='__main__':main()
