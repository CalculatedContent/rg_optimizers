"""Strict validation-only selection and permanent summary, including partial runs."""
import csv, json, math, statistics, sys
from run import D,P,CAP,TRACKS,sha,dump,whash,ORIGINAL
R=D/"results"; A=R/"analysis"
def expected(phase):
    return [(c["id"],s,c) for c in P["grid"] for s in P["dev_seeds"]] if phase=="grid" else [(t,s,None) for t in TRACKS for s in P["confirmation_seeds"]]
def load(phase,selection=None):
    import torch
    records=[];errors=[]
    for case,seed,cfg in expected(phase):
        folder=R/phase/f"{case}-{seed}"
        try:
            p=folder/"result.json";r=json.loads(p.read_text());records.append(r)
            assert (r["phase"],r["case"],r["seed"])==(phase,case,seed)
            assert r["plan_sha256"]==sha(D/"plan.json") and r["data_manifest_sha256"]==sha(ORIGINAL/"prepared/manifest.json")
            assert r["status"]=="complete" and r["completed_updates"]==CAP and r["start_step"]==0 and r["planned_updates"]==CAP
            assert r["config"]==(cfg if phase=="grid" else selection["selected"][case])
            assert r["selection_sha256"]==(None if phase=="grid" else sha(R/"frozen_selection.json"))
            assert [h["step"] for h in r["history"]]==P["checkpoints"]
            initial_gains=None
            control=json.loads((D.parent/"magic_scale_invariant"/"results"/"grid"/f"{case}-{seed}"/"result.json").read_text())
            assert control["initial_weights_sha256"]==r["initial_weights_sha256"]
            for h in r["history"]:
                path=folder/h["checkpoint_file"];assert sha(path)==h["checkpoint_sha256"]
                cp=torch.load(path,map_location="cpu",weights_only=True)
                assert cp["step"]==h["step"] and whash(cp["model"])==cp["weights_sha256"]==h["weights_sha256"]==h["spectral_weights_sha256"]
                gains=[float(cp["model"][f"{i}.gain"]) for i in (0,2,4)]
                if initial_gains is None:initial_gains=gains
                assert initial_gains==gains
                assert "calibrated_ce" in h["metrics"]["validation"]
                expected_keys={"train","validation"} if phase=="grid" else ({"train","test"} if h["step"]==CAP else {"train"})
                assert set(h["metrics"])==expected_keys
                assert set(h["radial_fixed_train_probe"])=={"fc1","fc2","fc3"}
                assert all(x["relative_radial_residual"] is None or x["relative_radial_residual"]<2e-5 for x in h["radial_fixed_train_probe"].values())
            assert sha(folder/"final_resume.pt")==r["final_resume_sha256"]
            resume=torch.load(folder/"final_resume.pt",map_location="cpu",weights_only=True)
            assert resume["step"]==CAP and whash(resume["model"])==r["history"][-1]["weights_sha256"]
        except Exception as e:errors.append(f"{phase}/{case}-{seed}: {type(e).__name__}: {e}")
    return records,errors
def rows(records):
    for r in records:
        for h in r.get("history",[]):
            s=h["spectral"];raw=s["raw"]
            q={"phase":r["phase"],"case":r["case"],"seed":r["seed"],"track":r["config"]["track"],"lr":r["config"]["lr"],"status":r["status"],"step":h["step"],"alpha":raw["alpha"],"D":raw["D"],"tail":raw["tail"],"fit_status":raw["status"],"raw_spectral_norm":s["raw_spectral_norm"],"effective_spectral_norm":s["effective_spectral_norm"],"effective_eigenvalue_scale":s["effective_eigenvalue_scale"]}
            for part,m in h["metrics"].items():
                for k in ("accuracy","ce","auc"):q[part+"_"+k]=m[k]
            for k,v in h["metrics"].get("validation",{}).items():
                if k in ("calibrated_ce","full_temperature","logit_rms","brier","calibrated_brier","ece15","calibrated_ece15","auc_logit"):q["validation_"+k]=v
            for k,v in h.get("clipping",{}).items():q["clip_"+k]=v
            for k,v in (h.get("last_sampled_angular") or {}).items():q["sampled_"+k]=v
            for part in ("validation","test"):
                if part in h["metrics"]:
                    q[part+"_accuracy_gap"]=q["train_accuracy"]-q[part+"_accuracy"]
                    q[part+"_ce_gap"]=q[part+"_ce"]-q["train_ce"]
            for layer,z in h["radial_fixed_train_probe"].items():
                for k,v in z.items():q[layer+"_"+k]=v
            for layer,z in (h.get("last_update_preclip_radial") or {}).items():
                q[layer+"_last_update_preclip_residual"]=z["relative_radial_residual"]
            yield q
def write_csv(path,rr):
    keys=list(dict.fromkeys(k for r in rr for k in r))
    with path.open("w") as f:
        if keys:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A.mkdir(parents=True,exist_ok=True)
    g,errors=load("grid")
    rr=list(rows(g));end=[q for q in rr if q["step"]==CAP]
    write_csv(A/"trajectories.csv",rr);write_csv(A/"fixed_20k.csv",end)
    ordinary=list(csv.DictReader((ORIGINAL/"results/analysis/trajectory.csv").open()))
    learned=list(csv.DictReader((D.parent/"magic_scale_invariant"/"results/analysis/trajectories.csv").open()))
    cal=list(csv.DictReader((D.parents[2]/"calibration_archive"/"experiments/single_matrix_benchmarks/magic_scale_invariant/calibration/calibration.csv").open()))
    def key(q):return (q["case"],int(q["seed"]),int(q["step"]))
    o={key(q):q for q in ordinary if q["phase"]=="grid" and q["case"] in [c["id"] for c in P["grid"]]}
    l={key(q):q for q in learned if q["phase"]=="grid"}
    ca={key(q):q for q in cal}
    metrics=("alpha","D","tail","train_accuracy","train_ce","validation_accuracy","validation_ce","validation_auc","validation_accuracy_gap","validation_ce_gap","fc2_raw_frobenius","fc2_effective_frobenius","fc2_gain","fc3_gain","validation_calibrated_ce","validation_full_temperature","validation_logit_rms","validation_ece15","validation_calibrated_ece15","clip_mean_factor","sampled_angle")
    def num(v):
        try:return float(v) if v is not None and v!="" else None
        except (ValueError,TypeError):return None
    paired=[]
    for q in rr:
        k=key(q)
        oq=o.get(k,{});lq=l.get(k,{});cq=ca.get(k,{})
        z={"case":q["case"],"seed":q["seed"],"step":q["step"],"lr":q["lr"],"status":q["status"]}
        for m in metrics:
            a=num(oq.get(m));b=num(lq.get(m));c=num(q.get(m))
            if m=="validation_calibrated_ce":b=num(cq.get("oof_ce"))
            elif m=="validation_full_temperature":b=num(cq.get("temperature_full"))
            elif m=="validation_logit_rms":b=num(cq.get("logit_rms"))
            elif m=="validation_ece15":b=num(cq.get("raw_ece15"))
            elif m=="validation_calibrated_ece15":b=num(cq.get("oof_ece15"))
            z.update({f"ordinary_{m}":a,f"learned_{m}":b,f"fixed_{m}":c,f"fixed_minus_learned_{m}":c-b if c is not None and b is not None else None,f"fixed_minus_ordinary_{m}":c-a if c is not None and a is not None else None})
        paired.append(z)
        if not (oq and lq and cq):errors.append(f"missing matched control {k}")
    write_csv(A/"paired_trajectories.csv",paired)
    endpoints=[q for q in paired if q["step"]==CAP]
    write_csv(A/"paired_endpoints.csv",endpoints)
    stats=[]
    for c in P["grid"]:
        v=[q for q in endpoints if q["case"]==c["id"] and q["status"]=="complete"]
        for metric in metrics:
            for arch in ("ordinary","learned","fixed","fixed_minus_learned","fixed_minus_ordinary"):
                x=[q[arch+"_"+metric] for q in v if q[arch+"_"+metric] is not None]
                stats.append(dict(case=c["id"],optimizer=c["optimizer"],lr=c["lr"],metric=metric,architecture=arch,n=len(x),mean=statistics.mean(x) if x else None,ci95_half_width=12.706204736*statistics.stdev(x)/math.sqrt(2) if len(x)==2 else None))
    write_csv(A/"statistics.csv",stats)
    for metric in ("alpha","validation_accuracy","validation_ce","validation_calibrated_ce","fc2_effective_frobenius","fc3_gain"):
        fig,axs=plt.subplots(2,4,figsize=(13,6),sharex=True)
        for ax,c in zip(axs.flat,P["grid"]):
            for arch in ("ordinary","learned","fixed"):
                x=[];y=[]
                for step in P["checkpoints"]:
                    vals=[q[arch+"_"+metric] for q in paired if q["case"]==c["id"] and q["step"]==step and q[arch+"_"+metric] is not None]
                    if vals:x.append(step);y.append(statistics.mean(vals))
                if x:ax.plot(x,y,marker="o",label=arch)
            if metric=="alpha":ax.axhline(2,color="gray",ls="--")
            ax.set_title(f'{c["optimizer"]} {c["lr"]}');ax.grid(alpha=.2)
        axs.flat[-1].axis("off")
        handles,labels=axs.flat[0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc="lower right")
        fig.supxlabel("Updates");fig.supylabel(metric);fig.tight_layout();fig.savefig(A/(metric+"_trajectory.png"),dpi=160);plt.close(fig)
    for metric in ("validation_accuracy","validation_ce","validation_calibrated_ce"):
        fig,ax=plt.subplots()
        for arch in ("ordinary","learned","fixed"):
            q=[v for v in endpoints if v[arch+"_alpha"] is not None and v[arch+"_"+metric] is not None]
            if q:ax.scatter([v[arch+"_alpha"] for v in q],[v[arch+"_"+metric] for v in q],label=arch)
        ax.axvline(2,color="gray",ls="--");ax.set(xlabel="Raw FC2 alpha",ylabel=metric);ax.legend();fig.tight_layout();fig.savefig(A/(metric+"_vs_alpha.png"),dpi=160);plt.close(fig)
    complete=len(g)==14 and len(end)==14 and len(rr)==84 and not errors
    fit_failures=sum(q["fit_status"]!="success" for q in rr)
    out=dict(complete=complete,n_grid=len(g),n_endpoints=len(end),n_checkpoints=len(rr),audit_errors=errors,raw_fit_failures=fit_failures,statistics=stats,controls={"ordinary":P["primary_archive"],"learned":P["learned_gain_control"],"calibration":P["calibration_control"]},test_protocol="validation only; no new test predictions; nominal paired two-seed intervals")
    dump(A/"summary.json",out)
    lines=["# MAGIC fixed-gain scale-invariant MLP","",f"Complete: {complete}; grid {len(g)}/14; endpoints {len(end)}/14; checkpoints {len(rr)}/84; raw fit failures {fit_failures}.","", "All three gains frozen at matched initialization. Same training partition, seeds, learning rates, clipping, no WD, no dropout. No new test access or confirmation runs. Calibration cross-fitted on validation. Two-seed exploratory results.","","| Optimizer | LR | Ordinary alpha | Learned alpha | Fixed alpha | Ordinary val acc | Learned val acc | Fixed val acc | Learned raw CE | Fixed raw CE | Learned calibrated CE | Fixed calibrated CE |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in P["grid"]:
        def fmt(arch,metric):
            q=next(z for z in stats if z["case"]==c["id"] and z["architecture"]==arch and z["metric"]==metric)
            return f'{q["mean"]:.4f}' if q["mean"] is not None else "—"
        lines.append("| "+" | ".join([c["optimizer"],str(c["lr"])]+[fmt(a,m) for m,aa in (("alpha",("ordinary","learned","fixed")),("validation_accuracy",("ordinary","learned","fixed")),("validation_ce",("learned","fixed")),("validation_calibrated_ce",("learned","fixed"))) for a in aa])+" |")
    lines+=["","Audit errors: "+str(errors),"","[Paired endpoints](paired_endpoints.csv) · [Paired trajectories](paired_trajectories.csv) · [Statistics](statistics.csv) · [Alpha trajectory](alpha_trajectory.png) · [Accuracy](validation_accuracy_trajectory.png) · [Calibrated CE](validation_calibrated_ce_trajectory.png)","", "Intervals across two seeds have df=1 and are descriptive; not independent test population confidence intervals. Raw alpha fits not selected by performance; short tails and alpha>6 warrant separate sensitivity analysis. Fixed gains constrain effective matrix norms and model capacity; this is not merely a gauge transformation."]
    (A/"RESULTS.md").write_text("\n".join(lines)+"\n")
    assert complete,errors
if __name__=="__main__":summarize()
