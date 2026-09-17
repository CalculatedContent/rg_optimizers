from pathlib import Path
import sys
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import am_posthoc_ww as post


def test_pinned_posthoc_weightwatcher_contract():
    assert post.WW_CFG['version']=='0.7.7'
    assert post.WW_CFG['fix_fingers']=='clip_xmax'
    assert post.WW_CFG['ERG'] is True and post.WW_CFG['randomize'] is True


def test_checkpoint_order(tmp_path):
    for step in (1000,0,500): (tmp_path/f'model_{step:08d}.pt').touch()
    assert [int(p.stem.split('_')[1]) for p in post.checkpoints(tmp_path)]==[0,500,1000]


def test_posthoc_summary_uses_seed_error_bars_and_below_two(tmp_path):
    rows=[]
    for arm in ('adamw','muon'):
        for seed,alpha in zip((1,2,3,4,5),(1.8,1.9,2.0,2.1,2.2) if arm=='adamw' else (2.2,2.3,2.4,2.5,2.6)):
            for matrix in ('a','b'):
                rows.append({'arm':arm,'seed':seed,'step':500,'matrix':matrix,
                    'alpha_clip_xmax':alpha,'alpha_raw':alpha+.1,'fit_supported':True})
    out=post.summarize(tmp_path,[pd.DataFrame(rows)])
    seed=pd.read_csv(out/'alpha_seed_summary.csv'); summary=pd.read_csv(out/'alpha_summary.csv')
    assert seed[(seed.arm=='adamw')].n_clip_below_2.sum()==4
    row=summary[(summary.arm=='adamw')&(summary.metric=='min_alpha_clip_xmax')].iloc[0]
    assert row.n_seeds==5 and pd.notna(row.ci95)
    assert (out/'figures'/'min_alpha_clip_xmax.png').exists()
