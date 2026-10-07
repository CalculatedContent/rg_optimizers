"""Standard-library probe reporting; safe to import in the launch grandparent."""
import json
from pathlib import Path
import resource
import sys
import time

FLAGS={'full_table_allocated':False,'tpu_execution_verified':False,'this_is_not_preflight':True}
STEP_STAGES={'train_128_compile','train_128_steady','train_49152_compile',
             'train_49152_steady','eval_262144_compile'}


def host_memory():
    available=None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): available=int(line.split()[1])*1024
    return {'mem_available_bytes':available,
            'peak_host_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024}


def write_json(path,value):
    path=Path(path); temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)


def summarize(root,*,final=False,exception=None):
    root=Path(root); ranks=[]
    for rank in range(8):
        path=root/f'probe-rank-{rank:02d}.json'
        value=json.loads(path.read_text()) if path.exists() else {'rank':rank,'status':'missing',**FLAGS}
        if final and value.get('status') not in ('complete','failed'):
            value.update(status='failed',failed_stage=value.get('current_stage','worker_initialization'),
                         exception=exception or 'Worker exited or was stopped before completing the probe')
            if value.get('stages') and value['stages'][-1]['status']=='running':
                value['stages'][-1]['status']='failed'
            # The launcher only finalizes after stopping the process group.
            write_json(path,value)
        ranks.append(value)
    complete=all(r.get('status')=='complete' and r.get('finite_loss') is True and
                 STEP_STAGES<={s['stage'] for s in r.get('stages',[])
                               if s.get('status')=='complete' and s.get('finite_loss') is True}
                 for r in ranks)
    failed=exception is not None or any(r.get('status')=='failed' for r in ranks) or (final and not complete)
    failures=[r for r in ranks if r.get('status')=='failed']
    summary={'status':'failed' if failed else 'complete' if complete else 'pending',
             'expected_world_size':8,'ranks':ranks,
             'exception':exception or (failures[0].get('exception') if failures else None),
             'failed_stage':failures[0].get('failed_stage') if failures else ('incomplete_workers' if failed else None),
             'finite_loss':all(r.get('finite_loss') is True for r in ranks) if complete else None,**FLAGS}
    for key in ('python','torch','torch_xla','world_size','device','runtime_device_attributes'):
        summary[key]=ranks[0].get(key)
    write_json(root/'PROBE_SUMMARY.json',summary)
    return summary


class ProbeReport:
    def __init__(self,root,rank):
        self.root,self.rank=Path(root),rank
        self.started=time.perf_counter(); self.stage='worker_initialization'
        self.data={'rank':rank,'status':'running','python':sys.version,'torch':None,'torch_xla':None,
                   'world_size':None,'device':None,'runtime_device_attributes':None,
                   'stages':[],'finite_loss':None,'failed_stage':None,'exception':None,**FLAGS}
        self.flush()

    def flush(self):
        self.data.update(host_memory(),current_stage=self.stage,
                         elapsed_seconds=time.perf_counter()-self.started)
        write_json(self.root/f'probe-rank-{self.rank:02d}.json',self.data)

    def begin(self,stage,**metadata):
        self.stage=stage
        event={'stage':stage,'status':'running','hbm':None,**metadata}
        self.data['stages'].append(event); self.flush()
        return event

    def memory(self,xm,device,event):
        event['hbm']={key:int(value) for key,value in xm.get_memory_info(device).items()}
        event.update(host_memory()); self.flush()

    def finish(self,error=None):
        if error is not None:
            self.data.update(status='failed',failed_stage=self.stage,
                             exception=f'{type(error).__name__}: {error}')
            if self.data['stages']:
                self.data['stages'][-1]['status']='failed'
        else:
            self.data['status']='complete'
        # Never query XLA here: an OOM can leave the runtime unusable.
        self.flush()
        if self.rank==0: summarize(self.root)
