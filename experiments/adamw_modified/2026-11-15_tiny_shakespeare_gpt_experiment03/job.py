"""One immutable GitHub Actions job, with failure evidence retained."""
import os,json,time,sys,traceback
from pathlib import Path
import runner as r
for name,expected in json.loads((r.ROOT/'Source_Manifest.json').read_text()).items():
    assert r.sha(r.ROOT/name)==expected,('Frozen source mismatch',name)
phase,arm=sys.argv[1:3];seed=int(sys.argv[3]);mult=float(sys.argv[4])
id=f'{phase}_{seed}_lr{mult:g}_{arm}'
started=time.time()
deadline=min(float(os.environ.get('CAMPAIGN_DEADLINE',started+96*3600)),started+4.5*3600)
spec={'id':id,'phase':phase,'arm':arm,'seed':seed,'multiplier':mult,'steps':r.PLAN['pilot_steps'] if phase=='pilot' else r.PLAN['full_steps'],'deadline':deadline}
r.write(r.ROOT/'Job.json',{'spec':spec,'started_epoch':started,'github_run_id':os.environ.get('GITHUB_RUN_ID'),'github_sha':os.environ.get('GITHUB_SHA'),'torch_version':r.torch.__version__})
try:
 if phase=='pilot':r.preflight()
 r.run(spec)
except Exception as exc:
 r.write(r.ROOT/'Failure.json',{'spec':spec,'error':repr(exc),'traceback':traceback.format_exc()})
 raise
