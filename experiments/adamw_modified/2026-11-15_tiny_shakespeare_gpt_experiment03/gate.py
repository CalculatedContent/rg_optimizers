"""Proceed only after all paired pilot arms pass correctness and viability checks."""
import json,time,os
from pathlib import Path
p=Path('incoming');checks=[];ids=[];durations=[];starts=[];errors=[]
for arm in ['adamw','magnitude_graft','sgdm']:
 paths=list(p.glob(f'pilot-{arm}/runs/*/done.json'))
 if len(paths)!=1:errors.append(f'{arm}: missing or ambiguous completed pilot');continue
 d=paths[0].parent;j=json.loads(paths[0].read_text());v=json.loads((d/'Verification.json').read_text());base=d.parent.parent
 pre=json.loads((base/'Preflight.json').read_text());audit=json.loads((base/'Data_Audit.json').read_text());job=json.loads((base/'Job.json').read_text())
 initial=j['history'][0]['validation']['loss'];final=j['history'][-1]['validation']['loss'];passed=v['passed'] and pre['passed'] and audit['passed'] and v['functional_status']=='USABLE' and initial-final>=.1
 checks.append({'arm':arm,'passed':passed,'initial_validation_loss':initial,'final_validation_loss':final,'validation_loss_drop':initial-final,'seconds':j['seconds'],'initial_hash':j['initial_hash'],'data_hashes':audit['array_hashes']})
 ids.append(j['initial_hash']);durations.append(j['seconds']);starts.append(job['started_epoch'])
 if not passed:errors.append(f'{arm}: correctness or learning gate failed')
if len(set(ids))!=1:errors.append('initialization mismatch or missing')
if checks and any(c['data_hashes']!=checks[0]['data_hashes'] for c in checks):errors.append('data mismatch')
# Conservative throughput projection: 45 longer runs / 3 simultaneous jobs, plus 25% overhead.
projected=max(durations,default=float('inf'))*10*15*1.25
if projected>90*3600:errors.append('projected full grid exceeds remaining 96-hour budget')
if max(durations,default=float('inf'))*10*1.25>4.5*3600:errors.append('projected individual run exceeds bounded job budget')
passed=len(checks)==3 and not errors
deadline=min(starts,default=time.time())+96*3600
result={'passed':passed,'checks':checks,'errors':errors,'projected_grid_hours':projected/3600 if durations else None,'campaign_deadline':deadline,'alpha_used_as_gate':False,'pilot_excluded_from_confirmatory_analysis':True}
Path('Pilot_Gate.json').write_text(json.dumps(result,indent=2)+'\n')
with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write(f'passed={str(passed).lower()}\ndeadline={deadline}\n')
print(json.dumps(result,indent=2))
