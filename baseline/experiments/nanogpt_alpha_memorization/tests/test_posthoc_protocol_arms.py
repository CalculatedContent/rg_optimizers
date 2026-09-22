from pathlib import Path
import json
import pandas as pd

import am_posthoc_ww as posthoc


def test_posthoc_uses_arms_declared_by_study_protocol(monkeypatch,tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps({'arms':['muon_qkclip']})+'\n')
    run=tmp_path/'muon_qkclip'/'seed_2027'
    run.mkdir(parents=True)
    (run/'manifest.json').write_text(json.dumps({'seed':2027})+'\n')
    checkpoint=run/'model_00100000.pt'
    checkpoint.write_bytes(b'placeholder')

    seen=[]
    def fake_analyze(path,arm,seed,output,force=False):
        seen.append((path,arm,seed))
        return pd.DataFrame([{
            'arm':arm,'seed':seed,'step':100000,'matrix':'m',
            'alpha_clip_xmax':3.0,'alpha_raw':3.0,'fit_supported':True
        }])

    def fake_summarize(root,frames):
        assert len(frames)==1
        return root/'posthoc_weightwatcher'

    monkeypatch.setattr(posthoc,'analyze_checkpoint',fake_analyze)
    monkeypatch.setattr(posthoc,'summarize',fake_summarize)

    out=posthoc.run(tmp_path)
    assert out==tmp_path/'posthoc_weightwatcher'
    assert seen==[(checkpoint,'muon_qkclip',2027)]
