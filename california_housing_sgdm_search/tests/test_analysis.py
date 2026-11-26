"""Scientific analysis tests; fixtures are synthetic and never experiment data."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from california_housing_sgdm_search import analysis as a


def metrics(rmse=.5):
    return {"rmse":rmse,"mse":rmse**2,"mae":.3,"r2":.8,"mape_percent":18.}


def observation(step, rmse=.5, alpha=2.4, weight_hash="a"*64):
    return {"step":step,"train":metrics(rmse),"validation":metrics(rmse),"test":metrics(rmse),
            "weights_sha256":weight_hash,"spectral_weights_sha256":weight_hash,
            "spectral":{"n_eligible":1,"eligible_matrices":["backbone.blocks.1.0.weight"],
                "confidence_tail_alpha":alpha,"layer_alpha_sd":None,
                "matrices":[{"matrix":"backbone.blocks.1.0.weight","shape":[784,784],
                    "fit":{"status":"success","alpha":alpha,"D":.1,"num_pl_spikes":100,"num_evals":784}}]}}


def test_student_t_over_seeds_and_missing_observations():
    stat=a.seed_statistics([1.,2.,3.,4.,5.])
    assert stat["mean"]==3
    assert stat["sample_sd"]==pytest.approx(math.sqrt(2.5))
    assert stat["ci95_half_width"]==pytest.approx(1.9632431614775607)
    assert stat["degrees_of_freedom"]==4
    assert stat["complete_expected_seeds"]
    missing=a.seed_statistics([1.,None,float("nan"),True])
    assert missing["n"]==1 and missing["ci95_half_width"] is None
    assert not missing["complete_expected_seeds"]


def test_paired_horizon_contrast_differs_from_independent_cohorts():
    before={seed:.5+i*.01 for i,seed in enumerate(a.PARENT_SEEDS)}
    after={seed:value-.03 for seed,value in before.items()}
    paired=a.paired_effect(after,before,a.PARENT_SEEDS)
    assert paired["n"]==5 and paired["mean"]==pytest.approx(-.03)
    assert paired["ci95_half_width"]==pytest.approx(0,abs=1e-12)
    independent=a.welch_effect(list(after.values()),list(before.values()))
    assert independent["mean_difference"]==pytest.approx(-.03)
    assert independent["ci95_half_width"]>0
    assert a.paired_effect(after,{a.PARENT_SEEDS[0]:.5},a.PARENT_SEEDS)["n"]==1


def test_measurement_pair_requires_the_same_weights_and_raw_alpha():
    measured=observation(20000)
    issues=[]
    row=a.measurement_row(measured,cohort="confirmation",seed=3340,endpoint="fixed_updates",issues=issues)
    assert not issues
    assert row["confidence_tail_alpha"]==2.4 and row["same_checkpoint_verified"]
    assert row["layer_alpha_sd"] is None
    measured["spectral_weights_sha256"]="b"*64
    issues=[]
    row=a.measurement_row(measured,cohort="confirmation",seed=3340,endpoint="fixed_updates",issues=issues)
    assert row["confidence_tail_alpha"] is None and not row["same_checkpoint_verified"]
    assert any("matching checkpoint weight hashes" in issue for issue in issues)
    measured=observation(20000)
    measured["spectral"]["confidence_tail_alpha"]=9.
    issues=[]
    a.measurement_row(measured,cohort="confirmation",seed=3340,endpoint="fixed_updates",issues=issues)
    assert any("raw fit" in issue for issue in issues)


def test_no_survivor_average_or_fabricated_layer_dispersion():
    spectrum=observation(2000)["spectral"]
    assert a.spectral_alpha(spectrum)==2.4
    spectrum["matrices"].append(deepcopy(spectrum["matrices"][0]))
    assert a.spectral_alpha(spectrum) is None
    spectrum=observation(2000)["spectral"]
    spectrum["matrices"][0]["fit"]["num_pl_spikes"]=0
    assert a.spectral_alpha(spectrum) is None


def test_validation_selection_excludes_arbitrary_pause_and_old_window():
    best=observation(52,.4)
    window={"start":0,"end":100000,"observed_through":100000,"complete":True}
    best["selection_window"]=window
    record={"best_validation":best,"best_weights_sha256":best["weights_sha256"],
            "selection_window":window,
            "history":[observation(0,.9),observation(2000,.1),observation(100000,.6)],
            "epoch_history":[{"step":52,"validation":metrics(.4)}]}
    issues=[]
    a.verify_best_selection(record,0,100000,issues)
    assert not issues
    record["best_validation"]["step"]=2000
    issues=[]
    a.verify_best_selection(record,0,100000,issues)
    assert any("earliest minimum" in issue for issue in issues)
    issues=[]
    a.verify_best_selection(record,20000,100000,issues)
    assert any("outside" in issue for issue in issues)


def test_continuation_compares_every_saved_state_fingerprint():
    fields={"model_sha256":"a","optimizer_sha256":"b","torch_rng_sha256":"c",
            "numpy_rng_sha256":"d","python_rng_sha256":"e","batch_generator_sha256":"f",
            "epoch_permutation_sha256":"g","next_offset":8192,"completed_updates":20000,
            "completed_epochs":384,"minibatch_sha256":"h","epoch_loss_sum":4.,"epoch_examples":8192}
    source={"sha256":"checkpoint","fingerprints":fields}
    record={"parent_checkpoint_sha256":"checkpoint","continuity":{"parent":deepcopy(fields),"start":deepcopy(fields),"verified":True}}
    issues=[]
    a.verify_continuity(record,source,issues)
    assert not issues
    for key in fields:
        altered=deepcopy(record)
        altered["continuity"]["start"][key]="changed"
        issues=[]
        a.verify_continuity(altered,source,issues)
        assert any("exact continuation" in issue for issue in issues), key


def test_grid_configs_are_never_pooled_as_independent_seeds():
    points=[]
    for config in range(32):
        for offset,seed in enumerate(a.DEV_SEEDS):
            points.append({"candidate_id":f"sgdm_{config:02d}","seed":seed,"endpoint":"best_validation",
                           "step":100+seed,"validation_rmse":.5+offset*.1,"confidence_tail_alpha":2.+offset})
    rows=a.grid_summary(points)
    assert len(rows)==32
    assert all(row["validation_rmse_n"]==2 and row["confidence_tail_alpha_n"]==2 for row in rows)
    assert all(row["validation_rmse_sample_sd"]==pytest.approx(math.sqrt(.005)) for row in rows)
    assert all("ci95_half_width" not in row for row in rows)
    with pytest.raises(ValueError,match="duplicate development seed"):
        a.grid_summary(points+[points[0]])


def test_development_test_leakage_is_found_recursively():
    assert not a.has_test_measurement({"history":[{"test":None}],"best_validation":{"test":{}}})
    assert a.has_test_measurement({"best_validation":{"test":{"rmse":.4}}})
    assert a.has_test_measurement({"epoch_history":[{"test_rmse":.4}]})


def test_retained_development_weights_match_metric_hash(tmp_path):
    import torch
    from california_housing_benchmark.train import model_hash
    state={"weight":torch.tensor([[1.,2.],[3.,4.]])}
    digest=model_hash(state)
    best=observation(52,weight_hash=digest)
    final=observation(50000,weight_hash=digest)
    record={"best_validation":best,"best_weights_sha256":digest,"final_weights_sha256":digest,"history":[final]}
    trial={"retained_files":{}}
    for name,key,step in (("best_checkpoint.pt","selected_step",52),("final_weights.pt","completed_updates",50000)):
        torch.save({"model":state,"weights_sha256":digest,key:step},tmp_path/name)
        trial["retained_files"][name]=a.file_sha256(tmp_path/name)
    issues=[]
    a.audit_development_weights(record,tmp_path,trial,issues)
    assert not issues
    altered={"weight":torch.tensor([[1.,2.],[3.,5.]])}
    torch.save({"model":altered,"weights_sha256":digest,"completed_updates":50000},tmp_path/"final_weights.pt")
    trial["retained_files"]["final_weights.pt"]=a.file_sha256(tmp_path/"final_weights.pt")
    issues=[]
    a.audit_development_weights(record,tmp_path,trial,issues)
    assert any("retained model" in issue for issue in issues)


def test_development_plot_records_all_settings_and_failure_counts(tmp_path):
    plan=json.loads(Path(a.__file__).with_name("plan.json").read_text())
    points,statuses=[],[]
    for i,candidate in enumerate(plan["candidates"]):
        for offset,seed in enumerate(a.DEV_SEEDS):
            failure=i==31 and offset==1
            statuses.append({"candidate_id":candidate["id"],"seed":seed,"numerical_failure":failure})
            for endpoint,step in a.GRID_ENDPOINTS:
                points.append({"candidate_id":candidate["id"],"seed":seed,"endpoint":endpoint,
                    "step":step if step is not None else 1234,"validation_rmse":None if failure else .4+i*.003+offset*.002,
                    "confidence_tail_alpha":None if failure else 1.7+i*.03+offset*.01,
                    "same_checkpoint_verified":not failure,**candidate["config"]})
    a._plot_grid(points,statuses,tmp_path)
    assert len(list(tmp_path.glob("*.pdf")))==4
    assert len(list(tmp_path.glob("*.png")))==4
    assert sum(row["numerical_failure"] for row in statuses)==1
    assert len(points)==256


def test_full_development_scan_keeps_numerical_failures_and_selects_validation_only(tmp_path,monkeypatch):
    plan=json.loads(Path(a.__file__).with_name("plan.json").read_text())
    contract={"plan":plan,"frozen":{"selected":plan["candidates"][0]["config"]}}
    monkeypatch.setattr(a,"audit_execution",lambda *args:None)
    monkeypatch.setattr(a,"audit_development_weights",lambda *args:None)
    for i,candidate in enumerate(plan["candidates"]):
        for offset,seed in enumerate(a.DEV_SEEDS):
            failure=i==31 and offset==1
            directory=tmp_path/"development"/candidate["id"]/f"seed_{seed}"
            directory.mkdir(parents=True)
            value=.4+i*.01+offset*.001
            best=observation(52,value)
            best["test"]=None
            window={"start":0,"end":50000,"observed_through":50000,"complete":True}
            best["selection_window"]=window
            history=[observation(step,value+.01) for step in ((0,2000,20000) if failure else (0,2000,20000,50000))]
            for row in history:
                row["test"]=None
            record={"seed":seed,"phase":"development","config":candidate["config"],
                    "status":"failed" if failure else "complete","completed_updates":30000 if failure else 50000,
                    "history":history,"best_validation":None if failure else best,
                    "selection_window":window,"best_weights_sha256":best["weights_sha256"],
                    "epoch_history":[{"step":52,"validation":metrics(value)}]}
            if failure:
                record["error"]="FloatingPointError: test fixture"
            (directory/"result.json").write_text(json.dumps(record))
            trial={"seed":seed,"status":"numerical_failure" if failure else "complete",
                   "result_sha256":a.file_sha256(directory/"result.json")}
            if failure:
                trial["error"]=record["error"]
            (directory/"trial.json").write_text(json.dumps(trial))
    errors,warnings=[],[]
    points,statuses=a.audit_development(tmp_path,contract,errors,warnings)
    assert not errors, errors
    assert len(statuses)==64 and len(points)==256
    assert sum(row["numerical_failure"] for row in statuses)==1
    failed=[p for p in points if p["status"]=="numerical_failure"]
    assert len(failed)==4
    assert len([p for p in failed if p["validation_rmse"] is not None])==2
    assert len(warnings)==1
    contract["frozen"]["selected"]=plan["candidates"][1]["config"]
    errors=[]
    a.audit_development(tmp_path,contract,errors,[])
    assert any("frozen winner" in error for error in errors)
