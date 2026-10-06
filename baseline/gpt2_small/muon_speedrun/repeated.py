"""Fixed-budget, paired-seed comparisons using the existing speedrun worker."""
from collections import defaultdict
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from stock_config import ARCHITECTURE, GPTConfig
from benchmark_config import BENCHMARK, TOTAL_STEPS, MEASUREMENT_INTERVAL, protocol
from cloudshell import cloud_uri, EXPERIMENT

SEEDS = (1337, 1338, 1339)
JOB_SECONDS = 12 * 3600
SUITE_SECONDS = 6 * JOB_SECONDS + 900
VAL_TOKENS = 10485760
METRICS = ('val_nll', 'val_perplexity', 'val_token_error')
ALPHAS = ('alpha_raw_mean', 'alpha_raw_min', 'alpha_clip_xmax_mean', 'alpha_clip_xmax_min')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def plan():
    jobs = []
    for index, seed in enumerate(SEEDS):
        # Alternate order to avoid putting one optimizer systematically first.
        for optimizer in (('muon_clip', 'adamw') if index % 2 == 0 else ('adamw', 'muon_clip')):
            jobs.append({'seed':seed, 'optimizer':optimizer, 'name':f'{optimizer}-s{seed}'})
    return {'jobs':jobs, 'experiment':EXPERIMENT, 'architecture':ARCHITECTURE, 'config':asdict(GPTConfig()),
            'benchmark':BENCHMARK, 'protocol':protocol(),
            'steps_per_run':TOTAL_STEPS, 'tokens_per_run':TOTAL_STEPS*524288,
            'microbatch':64, 'accumulation':8, 'batch_tokens':524288,
            'measurement_interval':MEASUREMENT_INTERVAL, 'full_validation_tokens':VAL_TOKENS,
            'full_budget':True, 'automatic_restart':False,
            'per_run_seconds_cap':JOB_SECONDS, 'suite_seconds_cap':SUITE_SECONDS,
            'randomness':'Initialization varies across seeds. All runs use the same fixed token order.',
            'schedule':'700-update linear warmup, cosine decay to zero at 19560 updates; global gradient clipping at 1.0.',
            'comparison':'Stock GPT-2 Small; MuonClip hidden LR 0.02, auxiliary AdamW LR 0.0006; AdamW control LR 0.0006. AdamW matrix decay 0.1; MuonClip hidden decay 0.1; per-head QK clip threshold 100.',
            'target_note':'3.28 is the original GPT-2/FineWeb target; this TPU/Muon port has not demonstrated convergence.',
            'error_bars':'Sample standard deviation across seeds, not across matrices or updates.'}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def stats(values):
    return {'n':len(values), 'mean':statistics.mean(values) if values else None,
            'std':statistics.stdev(values) if len(values)>1 else None}


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def table(path, rows, fields):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def report(root):
    root = Path(root)
    p = read(root/'PLAN.json')
    curves, layers = defaultdict(list), defaultdict(list)
    runs, finals, times = [], {}, defaultdict(list)
    observations=[]
    for job in p['jobs']:
        run = root/(root.name+'-'+job['name'])
        status = read(run/'status.json', {})
        supervisor = read(run/'RUN_STATUS.json', {})
        manifest = read(run/'manifest.json', {})
        launch = read(run/'launch.json', {})
        metrics = run/'metrics.jsonl'
        rows = []
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue # A live trailing line may not yet be complete.
                if row.get('kind')=='validation' and row.get('full_benchmark_evaluation') and row.get('evaluation_tokens')==VAL_TOKENS:
                    rows.append(row)
        identity = manifest.get('seed')==job['seed'] and manifest.get('optimizer')==job['optimizer'] and manifest.get('full_budget') is True
        if p.get('architecture') is not None:
            identity = identity and (manifest.get('architecture') == p['architecture']
                                     and manifest.get('config') == p['config'])
        if p.get('benchmark') is not None:
            identity = identity and (manifest.get('benchmark') == p['benchmark']
                                     and manifest.get('protocol') == p['protocol'])
        final_step = p['steps_per_run']
        complete = (identity and status.get('step')==final_step and supervisor.get('exit_code')==0
                    and supervisor.get('tracking',{}).get('status')=='complete'
                    and supervisor.get('backup',{}).get('exit_code')==0
                    and any(r.get('step')==final_step for r in rows))
        entry = {**job, 'root':str(run), 'complete':complete,
                 'status':supervisor.get('status','not_started'), 'step':status.get('step'),
                 'target_first_observed_seconds':None}
        if identity:
            paired = {r['step']:r for r in rows}
            for row in paired.values():
                end_to_end=(row['recorded_unix']-launch['started_unix']
                            if finite(row.get('recorded_unix')) and finite(launch.get('started_unix')) else None)
                observations.append(dict(optimizer=job['optimizer'],seed=job['seed'],step=row['step'],
                    tokens_seen=row.get('tokens_seen'),training_elapsed_seconds=row.get('elapsed_seconds'),
                    end_to_end_seconds=end_to_end,**{m:row.get(m) for m in METRICS}))
                for metric in METRICS:
                    if finite(row.get(metric)):
                        curves[job['optimizer'],row['step'],metric].append(float(row[metric]))
            hit = next((r for r in rows if finite(r.get('val_nll')) and r['val_nll']<=3.28),None)
            if hit:
                entry['target_first_observed_training_seconds'] = hit.get('elapsed_seconds')
                if finite(hit.get('recorded_unix')) and finite(launch.get('started_unix')):
                    elapsed=hit['recorded_unix']-launch['started_unix']
                    entry['target_first_observed_seconds']=elapsed
                    times[job['optimizer']].append(elapsed)
            for filename, fields, destination in [('summary.csv',ALPHAS,curves),
                                                   ('layers.csv',('alpha_raw','alpha_clip_xmax'),layers)]:
                path = run/'tracking'/filename
                if path.exists():
                    with path.open(newline='') as f:
                        for row in csv.DictReader(f):
                            step = int(row['step'])
                            if step not in paired:
                                continue
                            # Pair by exact checkpoint and the recorded validation token error.
                            if not finite(row.get('val_token_error')) or abs(float(row['val_token_error'])-paired[step]['val_token_error'])>1e-12:
                                raise ValueError('Spectrum/validation mismatch: '+str(path))
                            for metric in fields:
                                if finite(row.get(metric)):
                                    key = (job['optimizer'],step,metric) if destination is curves else (job['optimizer'],step,row['matrix_name'],metric)
                                    destination[key].append(float(row[metric]))
            if complete:
                finals[job['optimizer'],job['seed']] = paired[final_step]
        runs.append(entry)
    curve_rows = [dict(optimizer=o,step=s,metric=m,**stats(v)) for (o,s,m),v in sorted(curves.items())]
    layer_rows = [dict(optimizer=o,step=s,matrix_name=n,metric=m,**stats(v)) for (o,s,n,m),v in sorted(layers.items())]
    paired_seeds = [s for s in SEEDS if ('muon_clip',s) in finals and ('adamw',s) in finals]
    differences = {metric:stats([finals['adamw',s][metric]-finals['muon_clip',s][metric] for s in paired_seeds]) for metric in METRICS}
    out = {'benchmark':p.get('benchmark', 'historical-unspecified'), 'architecture':p.get('architecture', 'historical-unspecified'), 'config':p.get('config'),
           'complete_runs':sum(r['complete'] for r in runs), 'expected_runs':6, 'runs':runs,
           'paired_final_seeds':paired_seeds, 'final_adamw_minus_muonclip':differences,
           'target_times':{o:{**stats(times[o]), 'note':'End-to-end from each worker launch to first observed crossing; noncrossing runs are not successes.'} for o in ('muon_clip','adamw')},
           'error_bars':'Sample SD across available seeds; n is reported at every point. No independence assumption across training steps.',
           'interpretation':'Final differences require completed, backed-up, tracked pairs. Positive AdamW-minus-MuonClip NLL/error favors Muon. Recipes also differ in weight decay.'}
    table(root/'observations.csv',observations,['optimizer','seed','step','tokens_seen',
          'training_elapsed_seconds','end_to_end_seconds',*METRICS])
    write(root/'COMPARISON.json',out)
    table(root/'curves.csv',curve_rows,['optimizer','step','metric','n','mean','std'])
    table(root/'layers.csv',layer_rows,['optimizer','step','matrix_name','metric','n','mean','std'])
    return out


def plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (root/'curves.csv').open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for ax,metric in zip(axes.flat,('val_nll','val_token_error','alpha_raw_mean','alpha_raw_min')):
        for optimizer in ('muon_clip','adamw'):
            series = [r for r in rows if r['optimizer']==optimizer and r['metric']==metric]
            if not series: continue
            x=[int(r['step']) for r in series]; y=[float(r['mean']) for r in series]
            ax.plot(x,y,label=optimizer)
            band=[r for r in series if r['std']]
            if band:
                ax.fill_between([int(r['step']) for r in band],
                                [float(r['mean'])-float(r['std']) for r in band],
                                [float(r['mean'])+float(r['std']) for r in band],alpha=.18)
        ax.set(title=metric,xlabel='Training update'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Fixed-recipe nanoGPT speedrun — mean ± SD across available seeds')
    fig.tight_layout(); fig.savefig(root/'comparison.png',dpi=160); plt.close(fig)


def worker_command(root, job, deadline):
    return [sys.executable,'-u',str(Path(__file__).with_name('worker.py')),str(root),str(deadline),
            '--optimizer',job['optimizer'],'--seed',str(job['seed']),
            '--microbatch','64','--attention','flash','--full-budget']


def execute(root, deadline):
    root=Path(root); p=read(root/'PLAN.json')
    expected = plan()
    if any(p.get(key) != expected[key] for key in ('jobs', 'architecture', 'config', 'benchmark', 'protocol', 'steps_per_run', 'tokens_per_run', 'measurement_interval', 'per_run_seconds_cap', 'suite_seconds_cap')):
        raise ValueError('Suite plan changed; refusing to mix experiments')
    for job in p['jobs']:
        run=root/(root.name+'-'+job['name'])
        # Never rerun or overwrite a partially completed job.
        run.mkdir()
        write(root/'SUITE_STATUS.json',{'status':'running','job':job,'deadline_unix':deadline})
        job_deadline=min(time.time()+JOB_SECONDS,deadline-600)
        if job_deadline-time.time()<JOB_SECONDS-30:
            raise RuntimeError('Insufficient time for the next complete run; suite stopped')
        write(run/'launch.json',{**job,'suite':root.name,'started_unix':time.time(),
                                 'deadline_unix':job_deadline,'full_budget':True,
                                 'architecture':p['architecture'], 'benchmark':p['benchmark'],
                                 'experiment':EXPERIMENT, 'cloud_uri':cloud_uri(run)})
        (run/'commit.txt').write_text((root/'commit.txt').read_text())
        with (run/'run.log').open('w') as log:
            subprocess.run(worker_command(run,job,job_deadline),stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=JOB_SECONDS+5)
        result=report(root)
        if not next(r for r in result['runs'] if r['name']==job['name'])['complete']:
            raise RuntimeError('Incomplete run; suite stopped at '+job['name'])
    plots(root)
    write(root/'SUITE_STATUS.json',{'status':'training_complete','completed_runs':6})
    # Report artifacts are small; every individual run already verified its own backup.
    from rg_nanogpt_one_head.continuous_support import CloudPublisher
    publisher=CloudPublisher(cloud_uri(root))
    receipts=[publisher.file(path,path.name) for path in sorted(root.iterdir())
              if path.name!='SUITE_STATUS.json' and path.is_file() and path.suffix in ('.json','.csv','.png','.txt')]
    write(root/'SUITE_STATUS.json',{'status':'complete','completed_runs':6,'cloud_backup':'verified'})
    receipts.append(publisher.file(root/'SUITE_STATUS.json','SUITE_STATUS.json'))
    publisher.json({'status':'verified','files':receipts},'CLOUD_BACKUP_VERIFIED.json')
    write(root/'CLOUD_BACKUP_VERIFIED.json',{'status':'verified','files':receipts})


def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('deadline',type=float,nargs='?'); p.add_argument('--report',action='store_true')
    a=p.parse_args()
    if a.report:
        print(json.dumps(report(a.root),indent=2)); plots(a.root); return
    if a.deadline is None: p.error('deadline required for execution')
    try:
        execute(a.root,a.deadline)
    except Exception as exc:
        try: report(a.root)
        except Exception: pass
        write(a.root/'SUITE_STATUS.json',{'status':'failed_or_incomplete','error':str(exc),'automatic_restart':False})
        raise


if __name__=='__main__': main()
