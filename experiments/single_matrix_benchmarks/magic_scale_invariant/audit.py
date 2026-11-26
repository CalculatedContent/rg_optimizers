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
            for h in r["history"]:
                path=folder/h["checkpoint_file"];assert sha(path)==h["checkpoint_sha256"]
                cp=torch.load(path,map_location="cpu",weights_only=True)
                assert cp["step"]==h["step"] and whash(cp["model"])==cp["weights_sha256"]==h["weights_sha256"]==h["spectral_weights_sha256"]
                expected_keys={"train","validation"} if phase=="grid" else ({"train","test"} if h["step"]==CAP else {"train"})
                assert set(h["metrics"])==expected_keys
                assert set(h["radial_fixed_train_probe"])=={"fc1","fc2","fc3"}
                assert all(x["relative_radial_residual"] is None or x["relative_radial_residual"]<2e-5 for x in h["radial_fixed_train_probe"].values())
            assert sha(folder/"final_resume.pt")==r["final_resume_sha256"]
            resume=torch.load(folder/"final_resume.pt",map_location="cpu",weights_only=True)
            assert resume["step"]==CAP and whash(resume["model"])==r["history"][-1]["weights_sha256"]
        except Exception as e:errors.append(f"{phase}/{case}-{seed}: {type(e).__name__}: {e}")
    return records,errors
def selection():
    g,errors=load("grid")
    R.mkdir(parents=True,exist_ok=True)
    dump(R/"grid_audit.json",{"n_grid":len(g),"audit_errors":errors})
    assert not errors and len(g)==14,errors
    selected={};scores=[]
    for c in P["grid"]:
        runs=[r for r in g if r["case"]==c["id"]]
        assert len(runs)==2
        metrics=[r["history"][-1]["metrics"]["validation"] for r in runs]
        scores.append({"config":c,"accuracy":statistics.mean(x["accuracy"] for x in metrics),"ce":statistics.mean(x["ce"] for x in metrics)})
    for t in TRACKS:
        selected[t]=min((q for q in scores if q["config"]["track"]==t),key=lambda q:(-q["accuracy"],q["ce"],q["config"]["id"]))["config"]
    dump(R/"frozen_selection.json",{"complete_grid":True,"audit_errors":[],"plan_sha256":sha(D/"plan.json"),"selected":selected,"scores":scores})
def rows(records):
    for r in records:
        for h in r.get("history",[]):
            s=h["spectral"];raw=s["raw"]
            q={"phase":r["phase"],"case":r["case"],"seed":r["seed"],"track":r["config"]["track"],"lr":r["config"]["lr"],"status":r["status"],"step":h["step"],"alpha":raw["alpha"],"D":raw["D"],"tail":raw["tail"],"fit_status":raw["status"],"raw_spectral_norm":s["raw_spectral_norm"],"effective_spectral_norm":s["effective_spectral_norm"],"effective_eigenvalue_scale":s["effective_eigenvalue_scale"]}
            for part,m in h["metrics"].items():
                for k in ("accuracy","ce","auc"):q[part+"_"+k]=m[k]
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
    sel=None
    try:sel=json.loads((R/"frozen_selection.json").read_text())
    except Exception as e:errors.append("selection missing: "+str(e))
    c,ce=load("confirm",sel);errors+=ce
    rr=list(rows(g+c));end=[q for q in rr if q["step"]==CAP]
    write_csv(A/"trajectories.csv",rr);write_csv(A/"fixed_20k.csv",end)
    stats=[]
    for t in TRACKS:
        group=[q for q in end if q["phase"]=="confirm" and q["track"]==t and q["status"]=="complete"]
        for k in ("test_accuracy","test_ce","test_auc","alpha","D","tail","train_accuracy","train_ce","test_accuracy_gap","test_ce_gap","fc2_raw_frobenius","fc2_effective_frobenius","fc2_gain","fc2_relative_radial_residual","raw_spectral_norm","effective_spectral_norm"):
            vals=[q[k] for q in group if q.get(k) is not None]
            n=len(vals);sd=statistics.stdev(vals) if n>1 else None
            from scipy.stats import t as student
            stats.append({"track":t,"metric":k,"mean":statistics.mean(vals) if n else None,"sd":sd,"ci95_half_width":float(student.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None,"n_valid":n,"n_expected":5})
    write_csv(A/"statistics.csv",stats)
    controls=list(csv.DictReader((ORIGINAL/"results/analysis/statistics.csv").open()))
    controls=[q for q in controls if q["track"] in TRACKS and q["metric"] in ("test_accuracy","test_ce","test_auc","alpha","D","tail")]
    write_csv(A/"ordinary_controls.csv",controls)
    def plot_traj(metric,path):
        fig,ax=plt.subplots()
        for t in TRACKS:
            for step in P["checkpoints"]:
                pass
            x=[];y=[];e=[]
            for step in P["checkpoints"]:
                v=[q[metric] for q in rr if q["phase"]=="confirm" and q["track"]==t and q["step"]==step and q.get(metric) is not None]
                if v:x.append(step);y.append(statistics.mean(v));e.append(statistics.stdev(v) if len(v)>1 else 0)
            if x:ax.errorbar(x,y,yerr=e,marker="o",label=t)
        if metric=="alpha":ax.axhline(2,ls="--",color="gray")
        ax.set(xlabel="Updates",ylabel=metric);ax.legend();fig.tight_layout();fig.savefig(A/path,dpi=160);plt.close(fig)
    for metric in ("alpha","train_ce","fc2_raw_frobenius","fc2_effective_frobenius","fc2_gain","fc2_relative_radial_residual"):plot_traj(metric,metric+"_trajectory.png")
    for metric in ("validation_accuracy","validation_ce"):
        fig,ax=plt.subplots()
        for t in TRACKS:
            v=[q for q in end if q["phase"]=="grid" and q["track"]==t and q.get(metric) is not None and q["alpha"] is not None]
            if v:ax.scatter([q["alpha"] for q in v],[q[metric] for q in v],label=t)
        ax.axvline(2,ls="--",color="gray");ax.set(xlabel="Raw FC2 alpha",ylabel=metric);ax.legend();fig.tight_layout();fig.savefig(A/("grid_"+metric+"_vs_alpha.png"),dpi=160);plt.close(fig)
    fit_failures=sum(q["fit_status"]!="success" for q in rr)
    out={"complete":not errors and len(g)==14 and len(c)==10,"n_grid":len(g),"n_confirm":len(c),"audit_errors":errors,"raw_fit_failures":fit_failures,"selection":sel,"statistics":stats,"ordinary_controls":controls,"accelerated_control_archive":"d19ede2339394c62eebdb04417b94aa5e55d4af0","test_protocol":"exploratory reused test, confirmation refits on outer_train; no independent validation on those refits"}
    dump(A/"summary.json",out)
    lines=["# MAGIC scale-invariant MLP results","",f'Complete: {out["complete"]}; grid {len(g)}/14; confirm {len(c)}/10; raw fit failures {fit_failures}.',"","| Track | Metric | Mean | SD | 95% CI half-width | n |","|---|---|---:|---:|---:|---:|"]
    for q in stats:lines.append("| "+" | ".join(str(q[k]) for k in ("track","metric","mean","sd","ci95_half_width","n_valid"))+" |")
    lines+=["","Ordinary controls: [original statistics](ordinary_controls.csv); accelerated SGDM archive: d19ede2339394c62eebdb04417b94aa5e55d4af0.","","CIs are across training seeds, not test population uncertainty. Reused test is exploratory. Validation-only development selection; confirmation train+validation refits have no separate validation. Raw FC2 alpha is not enforced; gain changes effective scale, not spectral shape for nonzero gain. LR/norm² is only a nominal angular scale for plain SGD, not actual momentum or AdamW dynamics. Radial gradient symmetry does not prove volume preservation or alpha=2.","","Audit errors: "+str(errors),"","[Endpoints](fixed_20k.csv) · [Trajectories](trajectories.csv) · [Alpha plot](alpha_trajectory.png)"]
    (A/"RESULTS.md").write_text("\n".join(lines)+"\n")
    assert out["complete"],errors
if __name__=="__main__":
    {"select":selection,"summarize":summarize}[sys.argv[1]]()
