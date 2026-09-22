import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import run_study

def test_qkclip_ten_seed_replication_is_frozen():
    cfg=json.loads((HERE/'protocol_four_head_muon_qkclip_10seeds.json').read_text())
    assert len(cfg['seeds'])==10
    assert len(set(cfg['seeds']))==10
    assert cfg['arms']==['muon_qkclip']
    assert cfg['steps']==100000
    assert cfg['model_overrides']['n_head']==4
    assert cfg['qk_clip']=={'threshold':100.0,'balance':0.5}
    assert cfg['data_seed']==20260918
    assert cfg['noise_fraction']==0.25
    jobs=run_study.jobs(cfg)
    assert len(jobs)==10
    assert jobs==[('muon_qkclip',seed) for seed in cfg['seeds']]
