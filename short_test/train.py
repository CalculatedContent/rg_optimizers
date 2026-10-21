"""One resumable CPU seed, following the original two-stage pilot recipe."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import time
PROCESS_START = time.perf_counter()
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import resource
from common import *

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--seed', type=int, required=True)
    a=p.parse_args(); out=a.output; out.mkdir(parents=True, exist_ok=True)
    lock=(out/'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'result.json').exists(): return
    torch.set_num_interop_threads(1)
    manifest=json.loads((out.parent.parent/'manifest.json').read_text())
    assert manifest['source_sha256']==source_hashes(), 'Frozen source changed'
    assert a.seed in manifest['seeds']
    data=load_data(a.data); model, optimizer, rng=setup(a.seed)
    config=dict(seed=a.seed, manifest_sha256=sha(out.parent.parent/'manifest.json'))
    step=0; timings=[]; history=[]; sessions=[]; lr=.0001
    anchor=None; bad=0; best_loss=None; best_step=None; best_state=None; reductions=[]; stop_reason=None
    last=out/'last.pt'
    if last.exists():
        ck=torch.load(last, map_location='cpu', weights_only=False)
        assert ck['config']==config
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); restore_rng(ck,rng)
        step=ck['step']; timings=ck['timings']; history=ck['history']; sessions=ck['sessions']
        lr=ck['lr']; anchor=ck['anchor']; bad=ck['bad']; best_loss=ck['best_loss']; best_step=ck['best_step']
        best_state=ck['best_state']; reductions=ck['reductions']; stop_reason=ck['stop_reason']; del ck
    session_start=time.perf_counter(); session_first=step; started=datetime.now(timezone.utc).isoformat()
    def save():
        session=dict(started_utc=started, first_step=session_first, last_step=step,
                     elapsed_seconds=time.perf_counter()-session_start)
        state=dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   step=step, timings=timings, history=history, sessions=sessions+[session], lr=lr,
                   anchor=anchor, bad=bad, best_loss=best_loss, best_step=best_step,
                   best_state=best_state, reductions=reductions, stop_reason=stop_reason, **rng_state(rng))
        tmp=last.with_suffix('.pt.tmp'); torch.save(state,tmp); tmp.replace(last)
        write_json(out/'progress.json',dict(seed=a.seed,step=step,max_steps=20000,training_seconds=sum(timings),
                   last_evaluation=history[-1] if history else None,utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(event='start',seed=a.seed,step=step,utc=started)),flush=True)
    if step==0 and not history:
        history.append(dict(step=0,train_sample_loss=sampled_loss(model,data['train']),
                            validation_sample_loss=sampled_loss(model,data['validation'],10002)))
        save()
    while step<20000 and stop_reason is None:
        t=time.perf_counter(); loss=update(model,optimizer,data['train'],rng,step,lr)
        step+=1; timings.append(time.perf_counter()-t)
        if step<=2000 and step%250==0:
            history.append(dict(step=step,train_sample_loss=sampled_loss(model,data['train']),
                                validation_sample_loss=sampled_loss(model,data['validation'],10002)))
        if step>=2000 and step%500==0:
            val=full_eval(model,data['validation']); value=val['loss_nats_per_character']
            if anchor is None or anchor-value>=.005: anchor,bad=value,0
            else: bad+=1
            history.append(dict(step=step,train_sample_loss=sampled_loss(model,data['train']),validation=val,
                                lr=lr,anchor=anchor,bad_checks=bad))
            if best_loss is None or value<best_loss:
                best_loss,best_step,best_state=value,step,copy.deepcopy(model.state_dict())
            if bad>=3:
                if lr>.00005:
                    reductions.append(dict(step=step,old_lr=lr,new_lr=.00005)); lr,anchor,bad=.00005,value,0
                else: stop_reason='validation_plateau_confirmed_at_lower_learning_rate'
        if step%250==0:
            if step==20000 and stop_reason is None: stop_reason='hard_cap_reached_without_confirmed_plateau'
            save(); print(json.dumps(dict(seed=a.seed,step=step,loss=loss,training_seconds=sum(timings))),flush=True)
    monitor_seconds=sum(s['elapsed_seconds'] for s in sessions)+time.perf_counter()-session_start
    # Freeze both choices before evaluating test data; all ten seeds were fixed before launch.
    write_json(out/'selection.json',dict(seed=a.seed,endpoint_step=step,selected_step=best_step,
                validation_loss=best_loss,stop_reason=stop_reason,checkpoint_sha256=sha(last)))
    evaluation_start=time.perf_counter()
    endpoint=dict(step=step,metrics={k:full_eval(model,v) for k,v in data.items()},alpha=spectrum(model.state_dict()))
    if best_step==step: selected=endpoint
    else:
        model.load_state_dict(best_state)
        selected=dict(step=best_step,metrics={k:full_eval(model,v) for k,v in data.items()},alpha=spectrum(model.state_dict()))
    assert abs(selected['metrics']['validation']['loss_nats_per_character']-best_loss)<1e-7
    result=dict(completed=True,seed=a.seed,stop_reason=stop_reason,
                practical_validation_plateau_confirmed=stop_reason.startswith('validation_plateau'),
                endpoint=endpoint,validation_selected=selected,history=history,lr_reductions=reductions,
                timing_seconds=dict(training_updates=sum(timings),training_with_monitoring=monitor_seconds,
                                    final_evaluation_and_alpha=time.perf_counter()-evaluation_start,
                                    current_process=time.perf_counter()-PROCESS_START),
                timing_scope='Measured active work; current_process includes imports/setup in this invocation only. Training and monitoring sum checkpointed sessions; interruption gaps excluded.',
                checkpoint_sha256=sha(last),manifest_sha256=config['manifest_sha256'],
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                finished_utc=datetime.now(timezone.utc).isoformat(),
                test_disclosure=manifest['test_policy'])
    write_json(out/'result.json',result)
    print(json.dumps(dict(event='complete',seed=a.seed,step=step,alpha=endpoint['alpha']['mean_alpha'])),flush=True)

if __name__=='__main__': main()
