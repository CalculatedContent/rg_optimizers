"""Use existing TPU only. No allocation, formatting, disk deletion or corpus download."""
from pathlib import Path
import datetime
import json
import shlex
import subprocess

PROJECT='tpu-builders-504820'; ZONE='us-west4-a'
QUEUE='ww-continuous8-24h-20261003-s1337'; NODE=QUEUE+'-node'
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
def run(*args,capture=False):
    return subprocess.run(args,check=True,text=True,stdout=subprocess.PIPE if capture else None).stdout
commit=run('git','-C',str(REPO),'rev-parse','HEAD',capture=True).strip()
if run('git','-C',str(REPO),'status','--porcelain',capture=True).strip(): raise SystemExit('Use a clean checkout of the pushed commit')
args=['--project='+PROJECT,'--zone='+ZONE]
queue=json.loads(run('gcloud','alpha','compute','tpus','queued-resources','describe',QUEUE,*args,'--format=json',capture=True))
if queue.get('state',{}).get('state')!='ACTIVE': raise SystemExit('Existing TPU is not ACTIVE. No replacement will be allocated.')
print(json.dumps(queue,indent=2),flush=True)
node=json.loads(run('gcloud','compute','tpus','tpu-vm','describe',NODE,*args,'--format=json',capture=True))
print(json.dumps(node,indent=2),flush=True)
root='/mnt/disks/rg-data/gpt2small/validation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
remote='/tmp/rg-gpt2-validation-'+commit[:12]
run('gcloud','compute','tpus','tpu-vm','ssh',NODE,*args,'--worker=0','--command=mkdir -p '+shlex.quote(remote))
run('gcloud','compute','tpus','tpu-vm','scp',str(HERE/'on_tpu.sh'),str(HERE/'prepare_existing.py'),NODE+':'+remote+'/',*args,'--worker=0')
command=shlex.join(['bash',remote+'/on_tpu.sh',commit,root])
run('gcloud','compute','tpus','tpu-vm','ssh',NODE,*args,'--worker=0','--command='+command)
print('Saved validation:',root)
