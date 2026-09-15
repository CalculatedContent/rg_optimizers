"""Launcher/analysis tests use recorded-data fixtures, not training evidence."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import study
import analyze

CFG = json.loads((HERE / 'configs/suite.json').read_text())


def arguments(**kwargs):
    return SimpleNamespace(**dict(command='run', root=None, latest=False, resume=False,
                                  optimizer='both', stage='full', recipe='repository',
                                  seed=1337, device='mps', dry_run=False, **kwargs))


def fixture_run(root, optimizer='adamw', *, steps=(0, 1, 2), completed=False, gains=None):
    path = root / 'full/repository/verbatim' / optimizer / 'seed_1337'
    path.mkdir(parents=True)
    suite = {'stages': {'full': {'steps': 4}}, 'canary_prefix_tokens': 64}
    manifest = dict(stage='full', recipe='repository', condition='verbatim', seed=1337,
                    profile={'family': optimizer}, suite=suite, fingerprint=f'hash-{optimizer}',
                    source_model={'block_size': 256}, input_tokens_per_step=8192,
                    initial_model_sha256='same-init', data_sha256='same-data',
                    runner_sha256='same-source', device={'device': 'test-cpu'})
    (path / 'manifest.json').write_text(json.dumps(manifest))
    inventory = [dict(eid=f'canary_{d}', group=f'dose_{d}', prompt=[1]*64, target=[2]*32)
                 for d in [0, 4]]
    (path / 'probe_inventory.json').write_text(json.dumps(inventory))
    # Slots 0 and 31 are seen at completed update 1; 32 at update 2.
    (path / 'injection_schedule.json').write_text(json.dumps({'0':'canary_4','31':'canary_4','32':'canary_4','33':'canary_4'}))
    gains = gains or {}
    rows = []
    for step in steps:
        audit = {}
        for d in [0, 4]:
            exact = gains.get((step,d), int(step == 1 and d == 4))
            means = dict(exact_match=float(exact), continuation_token_accuracy=.25 if d else 0.,
                         teacher_forced_accuracy=.5 if d else 0., nll=4. if d else 6.)
            audit[f'dose_{d}/prefix_64'] = dict(n=1, mean=means, examples=[dict(eid=f'canary_{d}',**means)])
        if step == 4:
            audit['dose_4/prefix_8'] = copy.deepcopy(audit['dose_4/prefix_64'])
        rows.append(dict(step=step, audit=audit, model_sha256=f'model-{step}'))
    (path / 'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (path / 'spectral').mkdir()
    for step in steps:
        (path / 'spectral' / f'step_{step:08d}.csv').write_text(
            'step,matrix_name,alpha_clip_xmax,alpha_raw,D,ERG_gap,num_traps,rand_distance,num_fingers\n'
            f'{step},L00_W_Q,2.1,2.4,0.1,1,0,0.2,1\n')
    if completed:
        (path / 'complete.json').write_text(json.dumps(dict(status='complete',steps=4,fingerprint=manifest['fingerprint'])))
    return path


def load(path):
    return analyze.read_run(path, {}, [])


def test_default_is_exactly_two_full_runs(tmp_path):
    jobs = study.commands(arguments(), CFG, tmp_path)
    assert len(jobs) == 4  # two smoke, two full
    assert [name for name,_ in jobs if name.startswith('full/')] == [
        'full/repository/verbatim/adamw/seed_1337','full/repository/verbatim/muon/seed_1337']
    assert all('--resume' not in cmd for _,cmd in jobs)
    assert all(cmd[cmd.index('--condition')+1] == 'verbatim' for _,cmd in jobs)


def test_single_optimizer_and_explicit_resume(tmp_path):
    args = arguments(); args.optimizer = 'muon'
    assert len(study.commands(args, CFG, tmp_path)) == 2
    args.resume = True
    assert all('--resume' in cmd for _,cmd in study.commands(args,CFG,tmp_path))


def test_completed_smoke_reuse_not_full_reuse(tmp_path):
    p = tmp_path/'smoke/repository/verbatim/adamw/seed_1337'
    p.mkdir(parents=True); (p/'complete.json').write_text('{}')
    jobs = study.commands(arguments(),CFG,tmp_path)
    assert '--resume' in jobs[0][1] and '--resume' not in jobs[2][1]


def test_dry_run_creates_no_results_or_latest(tmp_path, monkeypatch):
    latest = tmp_path/'latest.txt'
    monkeypatch.setattr(study,'LATEST',latest)
    root = tmp_path/'fresh'
    assert study.main(['run','--root',str(root),'--dry-run']) == 0
    assert not root.exists() and not latest.exists()


def test_existing_run_is_never_overwritten(tmp_path):
    fixture_run(tmp_path)
    with pytest.raises(ValueError, match='already exists'):
        study.launch(arguments(),CFG,tmp_path)


def test_failure_stops_sequence_keeps_error_and_runs_partial_analysis(tmp_path, monkeypatch):
    calls, reports = [], []
    monkeypatch.setattr(study,'LATEST',tmp_path/'latest.txt')
    def failed(command, log):
        calls.append(command)
        return 3
    monkeypatch.setattr(study,'stream_job',failed)
    monkeypatch.setattr(study,'report',lambda root,**kw: reports.append(root))
    assert study.launch(arguments(),CFG,tmp_path) == 3
    assert len(calls) == len(reports) == 1
    status = json.loads(next(tmp_path.glob('smoke/**/launcher_status.json')).read_text())
    assert status['state'] == 'failed' and status['returncode'] == 3


def test_interrupt_stops_without_later_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(study,'LATEST',tmp_path/'latest.txt')
    monkeypatch.setattr(study,'stream_job',lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert study.launch(arguments(),CFG,tmp_path) == 130
    assert json.loads(next(tmp_path.glob('smoke/**/launcher_status.json')).read_text())['state'] == 'interrupted'


def test_real_subprocess_streaming_preserves_failure_code(tmp_path, capsys):
    logfile = tmp_path/'job.log'
    code = study.stream_job([sys.executable,'-c','print("visible failure"); raise SystemExit(7)'],logfile)
    assert code == 7
    assert 'visible failure' in logfile.read_text() and 'visible failure' in capsys.readouterr().out


def test_live_tail_is_only_tolerated_malformed_line():
    warning=[]
    good='{"step":0,"audit":{}}\n'
    assert len(analyze.history(good+'{"step":',warning)) == 1 and warning
    with pytest.raises(ValueError): analyze.history(good+'{bad}\n',[])
    with pytest.raises(ValueError): analyze.history(good+good,[])


def test_exposure_reconstruction_uses_completed_updates(tmp_path):
    run=load(fixture_run(tmp_path))
    r=[r for r in run['examples'] if r['dose']==4]
    assert [x['presentations_from_schedule'] for x in r] == [0,2,4]
    assert [x['last_presentation_update_from_schedule'] for x in r] == [None,1,2]


def test_partial_result_peak_and_first_observation_are_not_final(tmp_path):
    run=load(fixture_run(tmp_path))
    doses,canaries,contrasts=analyze.summarize([run])
    row=next(r for r in doses if r['dose']==4)
    assert row['status']=='incomplete_or_running' and row['latest_step']==2
    assert row['first_observed_exact_step']==1 and row['peak_observed_exact_match']==1
    assert row['latest_exact_match']==0
    assert canaries[-1]['presentations_at_first_exact_from_schedule']==2
    assert contrasts[1]['exact_match_minus_zero_dose']==1
    assert contrasts[0]['zero_dose_nll_minus_exposed_nll']==2


def test_prefix_only_at_final_has_one_observation(tmp_path):
    run=load(fixture_run(tmp_path,steps=(0,1,4),completed=True))
    rows=analyze.summarize([run])[0]
    short=next(r for r in rows if r['prefix_tokens']==8)
    assert short['checkpoints']==1 and short['status']=='complete'


def test_no_exact_match_is_not_zero_onset(tmp_path):
    r=load(fixture_run(tmp_path,steps=(0,2)))
    assert all(x['first_observed_exact_step'] is None for x in analyze.summarize([r])[0])


def test_missing_schedule_counts_stay_missing(tmp_path):
    path=fixture_run(tmp_path)
    (path/'injection_schedule.json').unlink()
    assert all(r['presentations_from_schedule'] is None for r in load(path)['examples'])


def test_comparison_only_common_checkpoints(tmp_path):
    a=load(fixture_run(tmp_path,'adamw',steps=(0,1,2)))
    b=load(fixture_run(tmp_path,'muon',steps=(0,2,4)))
    rows=analyze.compare([a,b],[])
    assert {r['step'] for r in rows}=={0,2}
    assert all('muon_minus_adamw' in k for k in rows[0] if k.startswith('exact_match'))


def test_mismatched_hardware_cannot_be_paired(tmp_path):
    a=load(fixture_run(tmp_path,'adamw')); b=load(fixture_run(tmp_path,'muon'))
    b['manifest']['device']={'device':'other'}
    warning=[]
    assert analyze.compare([a,b],warning)==[] and warning


@pytest.mark.parametrize('fault',['n','mean','nan','duplicate'])
def test_bad_behavior_fails_closed(tmp_path,fault):
    path=fixture_run(tmp_path)
    p=path/'metrics.jsonl'; rows=[json.loads(x) for x in p.read_text().splitlines()]
    value=rows[0]['audit']['dose_4/prefix_64']
    if fault=='n': value['n']=2
    elif fault=='mean': value['mean']['nll']=999
    elif fault=='nan': value['examples'][0]['nll']=float('nan')
    else: value['examples']*=2; value['n']=2
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError): load(path)


def test_spectral_step_mismatch_is_rejected(tmp_path):
    p=fixture_run(tmp_path)
    (p/'spectral/step_00000000.csv').write_text('step,matrix_name\n7,L00_W_Q\n')
    with pytest.raises(ValueError,match='Spectral step'): load(p)


def test_report_writes_tables_without_altering_training_files(tmp_path):
    path=fixture_run(tmp_path)
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}
    out=analyze.analyze(tmp_path,plots=False)
    assert (out/'summary.md').is_file() and (out/'behavior.csv').is_file()
    assert 'incomplete_or_running' in (out/'summary.md').read_text()
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert 'num_fingers' in next(csv.reader((out/'spectral.csv').open()))


def test_report_plots_are_real_files(tmp_path):
    pytest.importorskip('matplotlib')
    fixture_run(tmp_path)
    out=analyze.analyze(tmp_path,plots=True)
    assert len(list((out/'figures').rglob('*.png')))==9


def test_no_results_reports_unknown_not_zero(tmp_path):
    out=analyze.analyze(tmp_path,plots=False)
    assert 'No non-smoke behavioral measurements' in (out/'summary.md').read_text()


def test_frozen_training_files_unchanged():
    pins={'run.py':'ed773a6f5fad23554031c5f577f28c12f185d6ca',
          'configs/suite.json':'1527fdb59a3d65da8f083f35ed8e04dddd2194b3'}
    for name,expected in pins.items():
        b=(HERE/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==expected


def test_shell_wrapper_has_no_shell_options_or_80_run_loop():
    source=(HERE/'run_full.sh').read_text()
    assert 'set -' not in source and 'for ' not in source
    assert 'study.py' in source


def test_schedule_dose_mismatch_rejected(tmp_path):
    path=fixture_run(tmp_path)
    (path/'injection_schedule.json').write_text('{}')
    with pytest.raises(ValueError,match='Schedule total'): load(path)


def test_completion_requires_matching_horizon_and_fingerprint(tmp_path):
    path=fixture_run(tmp_path,steps=(0,4),completed=True)
    study.verify_completion(path,4)
    with pytest.raises(RuntimeError): study.verify_completion(path,5)


def test_zero_exit_without_completion_is_not_success(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'LATEST',tmp_path/'latest.txt')
    monkeypatch.setattr(study,'stream_job',lambda *a: 0)
    monkeypatch.setattr(study,'report',lambda *a,**kw: None)
    assert study.launch(arguments(),CFG,tmp_path)==1
