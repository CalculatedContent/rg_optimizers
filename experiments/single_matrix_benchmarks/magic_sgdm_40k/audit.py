"""Audit fixed-40k SGDM continuations and compare validation metrics, no training."""
import csv,json,math,os,pathlib,statistics
import torch
from run import D,ORIGINAL,P,CAP,START,sha,whash
R=D/"results";A=R/"analysis";A.mkdir(parents=True,exist_ok=True)
def save(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def table(path,rows):
    if not rows:return
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
keys=["accuracy","ce","auc","brier","ece15","mean_confidence","confidence_correct","confidence_incorrect","high_confidence_fraction","high_confidence_error_fraction"]
errors=[];rows=[];complete=0;fits_failed=0
for bound in P["source_runs"]:
    seed=bound["seed"]
    try:
        d=R/f"s0-{seed}";j=json.loads((d/"result.json").read_text())
        assert j["status"]=="complete" and j["seed"]==seed and j["config"]==P["config"] and j["phase"]=="continuation" and j["case"]=="s0"
        assert j["start_step"]==START and j["planned_updates"]==j["completed_updates"]==CAP-START and j["final_step"]==CAP and j["dropout"]==0
        assert j["source_commit"]==os.environ["GITHUB_SHA"] and j["primary_archive_commit"]==P["primary_archive_commit"]
        assert j["plan_sha256"]==sha(D/"plan.json") and j["data_manifest_sha256"]==sha(ORIGINAL/"prepared/manifest.json")
        assert j["source_resume_sha256"]==bound["final_resume_sha256"] and j["source_endpoint_sha256"]==bound["endpoint_checkpoint_sha256"] and j["initial_weights_sha256"]==bound["endpoint_weights_sha256"]
        assert j["selection_sha256"]==sha(ORIGINAL/"results/frozen_selection.json")
        assert j["threads"]==1 and j["versions"]["torch"]=="2.7.1+cpu" and j["versions"]["weightwatcher"]=="0.7.7"
        assert [h["step"] for h in j["history"]]==P["checkpoints"]
        assert j["history"][0]["weights_sha256"]==bound["endpoint_weights_sha256"]
        case_rows=[]
        for h in j["history"]:
            assert set(h["metrics"])=={"train","validation"}
            assert sha(d/h["checkpoint_file"])==h["checkpoint_sha256"]
            q=torch.load(d/h["checkpoint_file"],map_location="cpu",weights_only=True)
            assert q["step"]==h["step"] and q["weights_sha256"]==h["weights_sha256"]==h["spectral_weights_sha256"]==whash(q["model"])
            assert h["spectral"]["matrix"]=="fc2" and h["spectral"]["shape"]==[784,784] and len(h["spectral"]["eigenvalues"])==784
            f=h["spectral"]["raw"]
            r={"seed":seed,"step":h["step"],"weights_sha256":h["weights_sha256"],"alpha":f["alpha"],"D":f["D"],"tail":f["tail"],"fit_status":f["status"]}
            for part in ("train","validation"):
                m=h["metrics"][part]
                for k in keys:r[part+"_"+k]=m[k]
                for t in P["roc_fpr"]:r[part+"_gamma_at_fpr_"+str(t)]=m["gamma_efficiency"][str(t)]
            r["accuracy_gap"]=r["train_accuracy"]-r["validation_accuracy"]
            r["ce_gap"]=r["validation_ce"]-r["train_ce"]
            case_rows.append(r)
        q=torch.load(d/"final_resume.pt",map_location="cpu",weights_only=True)
        assert sha(d/"final_resume.pt")==j["final_resume_sha256"] and q["step"]==CAP and whash(q["model"])==j["history"][-1]["weights_sha256"]
        complete+=1;rows.extend(case_rows);fits_failed+=sum(r["fit_status"]!="success" for r in case_rows)
    except Exception as e:errors.append(f"{seed}: {e!r}")
table(A/"trajectories.csv",rows)
def stats(v):
    v=[x for x in v if x is not None and math.isfinite(x)]
    if len(v)!=2:return {"mean":None,"ci95_half_width":None,"n":len(v)}
    return {"mean":statistics.mean(v),"ci95_half_width":12.7062047364*statistics.stdev(v)/math.sqrt(2),"n":2}
metric_keys=["alpha","D","tail","accuracy_gap","ce_gap"]+[s+"_"+k for s in ("train","validation") for k in keys]+["validation_gamma_at_fpr_"+str(t) for t in P["roc_fpr"]]
aggregates=[{"step":s,"metric":k,**stats([r[k] for r in rows if r["step"]==s])} for s in P["checkpoints"] for k in metric_keys]
table(A/"statistics.csv",aggregates)
pairs=[]
for seed in P["seeds"]:
    v={r["step"]:r for r in rows if r["seed"]==seed}
    if START in v and CAP in v:pairs.append({"seed":seed,**{k+"_change":v[CAP][k]-v[START][k] if v[CAP][k] is not None and v[START][k] is not None else None for k in metric_keys}})
table(A/"paired_changes.csv",pairs)
delta={k:stats([r[k+"_change"] for r in pairs]) for k in metric_keys}
result={"complete":complete==2 and not errors,"n_complete":complete,"audit_errors":errors,"verified_additional_updates":complete*(CAP-START),"failed_fits":fits_failed,"primary_archive_commit":P["primary_archive_commit"],"launch_commit":os.getenv("GITHUB_SHA"),"plan_sha256":sha(D/"plan.json"),"statistics":aggregates,"paired_changes":pairs,"paired_change_statistics":delta,"note":"Two paired development seeds, exploratory nominal 95% t intervals (df=1); validation only, no new test access."}
save(A/"summary.json",result)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plots=["alpha","validation_accuracy","validation_ce","validation_auc","validation_ece15","validation_brier","validation_mean_confidence","accuracy_gap","ce_gap"]
for k in plots:
    fig,ax=plt.subplots(figsize=(7,4))
    for seed in P["seeds"]:
        z=[r for r in rows if r["seed"]==seed and r[k] is not None]
        if z:ax.plot([r["step"] for r in z],[r[k] for r in z],marker="o",label=f"seed {seed}")
    if k=="alpha":ax.axhline(2,ls="--",color="gray")
    ax.set(xlabel="Total updates",ylabel=k,title="MAGIC SGDM continuation; validation only")
    if rows:ax.legend()
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(A/f"{k}_trajectory.{ext}",dpi=160)
    plt.close(fig)
def fmt(v):return "NA" if v["mean"] is None else f'{v["mean"]:.5f} ± {v["ci95_half_width"]:.5f}'
md=["# MAGIC SGDM continuation: 20k to 40k","",f"Complete: {result['complete']}; audited {complete}/2; failed fits {fits_failed}; new updates {result['verified_additional_updates']}.","","Validation only; two paired development seeds, nominal 95% t intervals (df=1). The 20k weights and momentum/batch RNG state are bound to the original frozen archive. No test reuse or AdamW training.","","| Metric | 20k | 40k | Paired change |","|---|---:|---:|---:|"]
for k in plots+["D","tail"]:
    a=next(x for x in aggregates if x["step"]==START and x["metric"]==k);b=next(x for x in aggregates if x["step"]==CAP and x["metric"]==k)
    md.append(f"| {k} | {fmt(a)} | {fmt(b)} | {fmt(delta[k])} |")
md+=["","[Full trajectories](trajectories.csv) · [Statistics](statistics.csv) · [Paired changes](paired_changes.csv) · [Alpha](alpha_trajectory.png) · [Accuracy](validation_accuracy_trajectory.png) · [CE](validation_ce_trajectory.png) · [Calibration](validation_ece15_trajectory.png)","","Audit errors: "+json.dumps(errors)]
(A/"RESULTS.md").write_text("\n".join(md)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("statistics","paired_changes","paired_change_statistics")},indent=2))
if not result["complete"]:raise RuntimeError("Incomplete; preserve records")
