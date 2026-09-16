from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_study


def test_generated_results_root_is_canonical_tmp_child(monkeypatch):
    monkeypatch.setattr(run_study, 'timestamp', lambda: '20990101_010101')
    root = run_study.resolve_root(create=True)
    canonical_tmp = Path('/tmp').resolve()
    assert root == root.resolve()
    assert root.parent == canonical_tmp
    assert root.name == 'nanogpt_alpha_memorization_20990101_010101'
