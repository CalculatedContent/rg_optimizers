"""Dependency-free checks of the fixed continuation and exact planned pairing."""
import ast
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
p = json.loads((root / 'Protocol.json').read_text())
specs = p['specifications']
assert len(specs) == 120
assert len({(x['seed'], x['batch_size'], x['multiplier'], x['arm']) for x in specs}) == 120
settings = {(128, .5), (128, 1), (256, .5), (256, 1), (256, 2), (512, .5), (512, 1), (512, 2)}
for seed in range(31001, 31006):
    for arm in ['adamw', 'magnitude_graft', 'sgdm']:
        assert {(x['batch_size'], x['multiplier']) for x in specs if x['seed'] == seed and x['arm'] == arm} == settings
tree = ast.parse((root / 'run.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
scope = {'math': math}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'schedule_only', 'exec'), scope)
rates = [scope['schedule'](e) for e in range(20, 61)]
assert abs(rates[0] - .1) < 1e-15 and abs(rates[-1] - .01) < 1e-15
assert all(.01 <= b <= a <= .1 for a, b in zip(rates, rates[1:]))
assert p['start_epoch'] == 20 and p['fixed_epochs'] == 60
assert p['maximum_training_seconds'] == 14400 and p['maximum_workflow_seconds'] == 18000
for f in root.glob('*.py'):
    compile(f.read_text(), str(f), 'exec')
print('PASS: 120 unique paired continuations, exact settings, fixed endpoints, continuous nonincreasing schedule and syntax.')
