"""Check the planned experiment, boundary conditions and source AST before training."""
import ast, hashlib, json, math
from pathlib import Path
from step_accounting import schedule, parent_schedule, accounting

root=Path(__file__).resolve().parent
p=json.loads((root/'Protocol.json').read_text())
specs=p['specifications']
assert len(specs)==120
assert len({(x['seed'],x['batch_size'],x['multiplier'],x['arm']) for x in specs})==120
settings={(128,.5),(128,1),(256,.5),(256,1),(256,2),(512,.5),(512,1),(512,2)}
for seed in range(31001,31006):
 for arm in ['adamw','magnitude_graft','sgdm']:
  assert {(x['batch_size'],x['multiplier']) for x in specs if x['seed']==seed and x['arm']==arm}==settings
assert p['start_epoch']==60 and p['fixed_epochs']==120 and p['augmentation'] is False
assert p['maximum_training_seconds']==14400 and p['maximum_workflow_seconds']==18000
assert schedule(60)==parent_schedule(60)==.01
assert all(schedule(e)==.01 for e in range(61,121))
for e in [59,121]:
 try:schedule(e)
 except ValueError:pass
 else:raise AssertionError('Endpoint guard failed')
for b,steps in [(128,430),(256,215),(512,108)]:
 sp={'batch_size':b,'multiplier':1,'arm':'adamw'}
 a,z=accounting(sp,60),accounting(sp,120)
 assert a['cumulative_updates']==60*steps and z['cumulative_updates']==120*steps
 assert z['new_updates_since60']==60*steps and z['last_batch_examples'] in range(1,b+1)
 assert math.isclose(z['sum_optimizer_layer_lrs'][0]-a['sum_optimizer_layer_lrs'][0],60*steps*1e-5,abs_tol=1e-12)
 assert math.isclose(z['sum_squared_optimizer_layer_lrs'][0]-a['sum_squared_optimizer_layer_lrs'][0],60*steps*1e-10,abs_tol=1e-12)
 assert z['cumulative_training_examples']==6_600_000
for f in root.glob('*.py'):compile(f.read_text(),str(f),'exec')
r=ast.parse((root/'run.py').read_text())
tr=next(n for n in r.body if isinstance(n,ast.FunctionDef) and n.name=='train')
assert 'range(61, 121)' in ast.unparse(tr)
assert 'epoch_061.pt' in ast.unparse(tr) and 'statistics/epoch_062.npz' in ast.unparse(tr)
assert 'early_stop' not in ast.unparse(tr)
print('PASS:120 original states, fixed60-to120 continuation, constant terminal LR, exact update/exposure counts and syntax.')
