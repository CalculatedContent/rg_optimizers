from __future__ import annotations
import copy, json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from am_data import Dataset

CFG=json.loads((HERE/'protocol.json').read_text())


def test_extended_horizon_preserves_original_canary_schedule_and_fingerprint():
    batch_size=32
    original=Dataset(CFG,1337,batch_size)
    extended_cfg=copy.deepcopy(CFG)
    extended_cfg['steps']=100000
    extended_cfg['withdrawal_step']=CFG['steps']//2
    extended=Dataset(extended_cfg,1337,batch_size)
    assert original.withdrawal==extended.withdrawal==5000
    assert original.fingerprint==extended.fingerprint
    assert {k:v.id for k,v in original.schedule.items()}=={k:v.id for k,v in extended.schedule.items()}
    assert extended.planned_counts(100000)==original.planned_counts(10000)


def test_extended_batches_after_10k_contain_no_canaries():
    extended_cfg=copy.deepcopy(CFG)
    extended_cfg['steps']=100000
    extended_cfg['withdrawal_step']=5000
    data=Dataset(extended_cfg,1337,32)
    canary_ids={r.id for r in data.canaries}
    for step in (10000,25000,50000,99999):
        assert not any(r.id in canary_ids for r in data.batch(step))
