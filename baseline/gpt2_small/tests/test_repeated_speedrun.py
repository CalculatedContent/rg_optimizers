import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'muon_speedrun'))
import repeated as suite


def launcher():
    spec=importlib.util.spec_from_file_location('repeated_launcher',BASE/'speedrun.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def populate(root, omit=None):
    suite.write(root/'PLAN.json',suite.plan())
    for job in suite.plan()['jobs']:
        r=root/(root.name+'-'+job['name']); r.mkdir()
        complete=job['name']!=omit
        suite.write(r/'manifest.json',dict(seed=job['seed'],optimizer=job['optimizer'],full_budget=True,
                    architecture=suite.plan()['architecture'],config=suite.plan()['config'],
                    benchmark=suite.BENCHMARK,protocol=suite.protocol()))
        suite.write(r/'status.json',dict(step=suite.TOTAL_STEPS if complete else 125))
        suite.write(r/'RUN_STATUS.json',dict(status='schedule_complete_target_not_met',exit_code=0,
                    tracking={'status':'complete'},backup={'exit_code':0}))
        offset=(job['seed']-1337)*.1+(.2 if job['optimizer']=='adamw' else 0)
        v=dict(kind='validation',step=suite.TOTAL_STEPS if complete else 125,evaluation_tokens=suite.VAL_TOKENS,
               full_benchmark_evaluation=True,val_nll=3.3+offset,val_perplexity=27+offset,
               val_token_error=.6+offset/100,elapsed_seconds=1000)
        (r/'metrics.jsonl').write_text(json.dumps(v)+'\n')
        (r/'tracking').mkdir()
        (r/'tracking/summary.csv').write_text('step,val_token_error,alpha_raw_mean\n'+f'{v["step"]},{v["val_token_error"]},{4+offset}\n')
        (r/'tracking/layers.csv').write_text('step,val_token_error,matrix_name,alpha_raw,alpha_clip_xmax\n'+f'{v["step"]},{v["val_token_error"]},L00_W_Q,{3+offset},{2+offset}\n')


def test_plan_has_matched_seeds_and_equal_budgets():
    p=suite.plan()
    assert p['architecture']=='gpt2-small-stock-v1'
    assert p['config']==dict(vocab_size=50257,block_size=1024,n_layer=12,n_head=12,n_embd=768)
    assert len(p['jobs'])==6
    assert [(j['seed'],j['optimizer']) for j in p['jobs']]==[
        (1337,'muon'),(1337,'adamw'),(1338,'adamw'),(1338,'muon'),(1339,'muon'),(1339,'adamw')]
    assert p['steps_per_run']*p['batch_tokens']==p['tokens_per_run']
    for j in p['jobs']:
        command=suite.worker_command(Path('/run'),j,123.)
        assert command[command.index('--seed')+1]==str(j['seed'])
        assert command[command.index('--optimizer')+1]==j['optimizer']
        assert '--full-budget' in command


def test_statistics_use_seeds_and_paired_differences(tmp_path):
    populate(tmp_path)
    result=suite.report(tmp_path)
    assert result['complete_runs']==6 and result['paired_final_seeds']==list(suite.SEEDS)
    diff=result['final_adamw_minus_muon']['val_nll']
    assert diff['n']==3 and diff['mean']==pytest.approx(.2) and diff['std']==pytest.approx(0,abs=1e-12)
    import csv
    with (tmp_path/'curves.csv').open() as f: rows=list(csv.DictReader(f))
    row=next(r for r in rows if r['optimizer']=='muon' and r['metric']=='val_nll')
    assert int(row['n'])==3 and float(row['mean'])==pytest.approx(3.4)
    assert float(row['std'])==pytest.approx(.1) # Sample SD across seeds.
    assert result['target_times']['adamw']['n']==0


def test_failed_or_partial_run_is_not_a_completed_pair(tmp_path):
    populate(tmp_path,omit='adamw-s1339')
    result=suite.report(tmp_path)
    assert result['complete_runs']==5 and result['paired_final_seeds']==[1337,1338]
    assert result['final_adamw_minus_muon']['val_nll']['n']==2


def test_old_architecture_cannot_enter_stock_model_comparison(tmp_path):
    populate(tmp_path)
    path=tmp_path/(tmp_path.name+'-muon-s1337')/'manifest.json'
    value=json.loads(path.read_text()); value.pop('architecture'); suite.write(path,value)
    result=suite.report(tmp_path)
    assert result['complete_runs']==5 and result['paired_final_seeds']==[1338,1339]


def test_old_suite_plan_cannot_start_stock_jobs(tmp_path):
    old=suite.plan(); old.pop('architecture'); suite.write(tmp_path/'PLAN.json',old)
    with pytest.raises(ValueError,match='refusing to mix'):
        suite.execute(tmp_path,suite.time.time()+suite.SUITE_SECONDS)


def test_spectral_checkpoint_mismatch_is_rejected(tmp_path):
    populate(tmp_path)
    path=tmp_path/(tmp_path.name+'-muon-s1337')/'tracking/summary.csv'
    path.write_text(f'step,val_token_error,alpha_raw_mean\n{suite.TOTAL_STEPS},0.5,4\n')
    with pytest.raises(ValueError,match='Spectrum/validation mismatch'): suite.report(tmp_path)


def test_short_or_stale_lease_never_starts_suite(monkeypatch):
    m=launcher(); monkeypatch.setattr(m.time,'time',lambda:1000.)
    with pytest.raises(RuntimeError,match='72h45m'): m.require_time(dict(checked_unix=1000,termination_unix=1000+7200))
    with pytest.raises(RuntimeError,match='expired'): m.require_time(dict(checked_unix=0,termination_unix=1e9))
    m.require_time(dict(checked_unix=1000,termination_unix=1000+suite.SUITE_SECONDS+1800))


def test_job_failure_stops_suite_without_retry(tmp_path,monkeypatch):
    suite.write(tmp_path/'PLAN.json',suite.plan()); (tmp_path/'commit.txt').write_text('a'*40)
    calls=[]
    def fail(command,**kwargs):
        calls.append(command)
        raise suite.subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(suite.subprocess,'run',fail)
    with pytest.raises(suite.subprocess.CalledProcessError): suite.execute(tmp_path,suite.time.time()+suite.SUITE_SECONDS)
    assert len(calls)==1
    assert calls[0][calls[0].index('--optimizer')+1]=='muon'
    with pytest.raises(FileExistsError): suite.execute(tmp_path,suite.time.time()+suite.SUITE_SECONDS)
    assert len(calls)==1


def test_status_does_not_launch_training(monkeypatch,tmp_path):
    m=launcher(); calls=[]
    suite.write(tmp_path/'launch.json',{'unit':'existing'})
    (tmp_path/'suite.log').write_text('existing')
    monkeypatch.setattr(sys,'argv',['speedrun.py','status','--remote','--root',str(tmp_path)])
    monkeypatch.setattr(m,'run',lambda cmd,**kw:calls.append(cmd))
    m.main()
    assert calls==[['tail','-n','15',str(tmp_path/'suite.log')]]


def test_target_time_includes_setup_and_does_not_count_non_crossers(tmp_path):
    populate(tmp_path)
    r=tmp_path/(tmp_path.name+'-muon-s1337')
    suite.write(r/'launch.json',{'started_unix':100.})
    row=json.loads((r/'metrics.jsonl').read_text())
    row.update(val_nll=3.2,recorded_unix=1600.,elapsed_seconds=1000.)
    (r/'metrics.jsonl').write_text(json.dumps(row)+'\n')
    result=suite.report(tmp_path)
    assert result['target_times']['muon']['n']==1
    assert result['target_times']['muon']['mean']==1500.
    assert result['target_times']['adamw']['n']==0


def test_old_3000_step_protocol_cannot_enter_new_comparison(tmp_path):
    populate(tmp_path)
    path=tmp_path/(tmp_path.name+'-muon-s1337')/'manifest.json'
    value=json.loads(path.read_text()); value.pop('benchmark'); suite.write(path,value)
    result=suite.report(tmp_path)
    assert result['complete_runs']==5 and result['paired_final_seeds']==[1338,1339]


def test_stock_baseline_schedule_matches_pinned_reference_for_every_update(tmp_path):
    from benchmark_config import lr_factor, TOTAL_STEPS, WARMUP
    from data import reference, required_shards, FineWeb
    assert TOTAL_STEPS==reference.TOTAL_STEPS==19560 and WARMUP==reference.WARMUP==700
    for step in range(TOTAL_STEPS+1):
        assert .0006*lr_factor(step)==pytest.approx(reference.learning_rate(step),abs=1e-16)
    assert lr_factor(0)==1/700 and lr_factor(699)==lr_factor(700)==1
    assert lr_factor(TOTAL_STEPS)==0
    source=FineWeb(tmp_path,0)
    names=required_shards(source,64,updates=TOTAL_STEPS)
    assert len(names)==104 and set(names)==set(source.manifest['files'])


def test_single_run_lease_uses_requested_cap(monkeypatch):
    m=launcher(); monkeypatch.setattr(m.time,'time',lambda:1000.)
    m.require_time(dict(checked_unix=1000,termination_unix=1000+13*3600),12*3600+600)
    with pytest.raises(RuntimeError,match='12h10m'):
        m.require_time(dict(checked_unix=1000,termination_unix=1000+12*3600),12*3600+600)
