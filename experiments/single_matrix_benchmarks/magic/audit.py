"""Audit complete/partial records; select by validation only; no training."""
import argparse,csv,hashlib,json,math,pathlib,statistics
from run import D,P,CAP,TRACKS,sha,whash
R=D/"results"
def dump(p,j):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2,allow_nan=False)+"\n")
def csvout(p,rows):
    if rows:
        with p.open("w") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(phase,case,seed,cfg,selsha=None):
    import torch
    d=R/phase/f"{case}-{seed}";j=json.loads((d/"result.json").read_text())
    assert j["status"]=="complete" and j["phase"]==phase and j["case"]==case and j["seed"]==seed and j["config"]==cfg and j["dropout"]==0
    assert j["start_step"]==0 and j["planned_updates"]==j["completed_updates"]==CAP and j["selection_sha256"]==selsha
    assert j["plan_sha256"]==sha(D/"plan.json") and j["data_manifest_sha256"]==sha(D/"prepared/manifest.json")
    assert j["threads"]==1 and j["versions"]["weightwatcher"]=="0.7.7" and j["versions"]["torch"]=="2.7.1+cpu"
    assert [h["step"] for h in j["history"]]==P["checkpoints"]
    for h in j["history"]:
        assert h["spectral"]["matrix"]=="fc2" and h["spectral"]["shape"]==[784,784] and len(h["spectral"]["eigenvalues"])==784
        path=d/h["checkpoint_file"];assert sha(path)==h["checkpoint_sha256"]
        q=torch.load(path,map_location="cpu",weights_only=True)
        assert q["step"]==h["step"] and q["weights_sha256"]==h["weights_sha256"]==h["spectral_weights_sha256"]==whash(q["model"])
        assert ("test" in h["metrics"])==(phase=="confirm" and h["step"]==CAP)
        assert ("validation" in h["metrics"])==(phase=="grid")
    path=d/"final_resume.pt";assert sha(path)==j["final_resume_sha256"]
    q=torch.load(path,map_location="cpu",weights_only=True);assert q["step"]==CAP and whash(q["model"])==j["history"][-1]["weights_sha256"]
    return j
def grid():
    g=[];errors=[]
    for c in P["grid"]:
        for seed in P["dev_seeds"]:
            try:g.append(read("grid",c["id"],seed,c))
            except Exception as e:errors.append(f"{c['id']}/{seed}: {e!r}")
    return g,errors
def select(g):
    scores=[]
    for c in P["grid"]:
        v=[j["history"][-1]["metrics"]["validation"] for j in g if j["config"]==c]
        assert len(v)==2 and all(math.isfinite(x[k]) for x in v for k in ("accuracy","ce"))
        scores.append({"case":c["id"],"track":c["track"],"mean_validation_accuracy":statistics.mean(x["accuracy"] for x in v),"mean_validation_ce":statistics.mean(x["ce"] for x in v),"config":c})
    best={t:min((s for s in scores if s["track"]==t),key=lambda s:(-s["mean_validation_accuracy"],s["mean_validation_ce"],s["case"]))["config"] for t in TRACKS}
    return best,scores
def freeze():
    g,e=grid();best,scores=select(g) if not e else ({},[])
    j={"complete_grid":not e,"n_grid":len(g),"audit_errors":e,"plan_sha256":sha(D/"plan.json"),"selected":best,"scores":scores}
    out=R/"frozen_selection.json"
    if out.exists() and json.loads(out.read_text())!=j:raise ValueError("Refuse changing frozen selection")
    dump(out,j);print(json.dumps(j,indent=2))
    if e:raise RuntimeError("Incomplete grid")
def stats(vals):
    v=[x for x in vals if x is not None and math.isfinite(x)]
    if len(v)!=5:return {"mean":None,"sd":None,"ci95_half_width":None,"n_valid":len(v),"n_expected":5}
    sd=statistics.stdev(v);return {"mean":statistics.mean(v),"sd":sd,"ci95_half_width":2.7764451052*sd/math.sqrt(5),"n_valid":5,"n_expected":5}
def summary():
    g,e=grid();records=list(g);sel=None
    try:
        sp=R/"frozen_selection.json";sel=json.loads(sp.read_text())
        assert not e and sel["complete_grid"] and not sel["audit_errors"] and sel["plan_sha256"]==sha(D/"plan.json") and sel["selected"]==select(g)[0]
        for t in TRACKS:
            for seed in P["confirmation_seeds"]:
                try:records.append(read("confirm",t,seed,sel["selected"][t],sha(sp)))
                except Exception as ex:e.append(f"{t}/{seed}: {ex!r}")
    except Exception as ex:e.append(f"selection: {ex!r}")
    rows=[]
    for j in records:
        for h in j["history"]:
            r={"phase":j["phase"],"track":j["config"]["track"],"case":j["case"],"seed":j["seed"],"step":h["step"],"weights_sha256":h["weights_sha256"],"alpha":h["spectral"]["raw"]["alpha"],"D":h["spectral"]["raw"]["D"],"tail":h["spectral"]["raw"]["tail"]}
            for part in ("train","validation","test"):
                m=h["metrics"].get(part,{})
                for k in ("accuracy","ce","auc"):r[part+"_"+k]=m.get(k)
                for t in P["roc_fpr"]:r[part+"_gamma_at_fpr_"+str(t)]=m.get("gamma_efficiency",{}).get(str(t))
            rows.append(r)
    out=R/"analysis";out.mkdir(parents=True,exist_ok=True)
    fixed=[r for r in rows if r["step"]==CAP];csvout(out/"trajectory.csv",rows);csvout(out/"fixed_20k.csv",fixed)
    s=[]
    for t in TRACKS:
        z=[r for r in fixed if r["phase"]=="confirm" and r["track"]==t]
        for k in ["test_accuracy","test_ce","test_auc","alpha","D","tail"]+["test_gamma_at_fpr_"+str(x) for x in P["roc_fpr"]]:
            s.append({"track":t,"metric":k,**stats([r[k] for r in z])})
    csvout(out/"statistics.csv",s)
    baseline=None
    try:
        baseline=json.loads((R/"xgb/summary.json").read_text())
        assert baseline["data_manifest_sha256"]==sha(D/"prepared/manifest.json") and baseline["plan_sha256"]==sha(D/"plan.json") and baseline["model_sha256"]==sha(R/"xgb/model.ubj")
    except Exception as ex:e.append(f"XGBoost: {ex!r}")
    result={"complete":not e,"n_grid":len(g),"n_confirm":sum(j["phase"]=="confirm" for j in records),"audit_errors":e,"failed_fits":sum(h["spectral"]["raw"]["status"]!="success" for j in records for h in j["history"]),"selection":sel,"statistics":s,"xgb":baseline}
    dump(out/"summary.json",result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for phase,metric in (("grid","validation_accuracy"),("confirm","test_accuracy")):
        fig,ax=plt.subplots(figsize=(7,5))
        for t in TRACKS:
            z=[r for r in fixed if r["phase"]==phase and r["track"]==t and r["alpha"] is not None and r[metric] is not None]
            ax.scatter([r["alpha"] for r in z],[r[metric]*100 for r in z],label=t)
        ax.axvline(2,ls="--",color="gray");ax.set(xlabel="FC2 alpha",ylabel=metric+" (%)");ax.legend();fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(out/f"{phase}_{metric}_vs_alpha.{ext}",dpi=160)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for t in TRACKS:
        xx=[];yy=[]
        for step in P["checkpoints"]:
            v=[r["alpha"] for r in rows if r["phase"]=="confirm" and r["track"]==t and r["step"]==step and r["alpha"] is not None]
            if len(v)==5:xx.append(step);yy.append(statistics.mean(v))
        ax.plot(xx,yy,marker="o",label=t)
    ax.axhline(2,ls="--",color="gray");ax.set(xlabel="Updates",ylabel="FC2 alpha");ax.legend();fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(out/f"confirmation_alpha_trajectory.{ext}",dpi=160)
    plt.close(fig)
    md=["# MAGIC fixed-20k primary results","",f"Complete: {result['complete']}; development {len(g)}/24; confirmation {result['n_confirm']}/15; failed raw fits {result['failed_fits']}.","","| Track | Metric | Mean | SD | 95% CI half width | n |","|---|---|---:|---:|---:|---:|"]
    md += [f"| {v['track']} | {v['metric']} | {v['mean']} | {v['sd']} | {v['ci95_half_width']} | {v['n_valid']} |" for v in s]
    md += ["","XGBoost same split: "+json.dumps(baseline),"","References: ICLR 2025 TabM reported ordinary MLP 85.39%, XGBoost 85.50%; different split, context only.","","[Grid plot](grid_validation_accuracy_vs_alpha.png) · [Confirmation plot](confirm_test_accuracy_vs_alpha.png) · [Alpha trajectory](confirmation_alpha_trajectory.png) · [Endpoint data](fixed_20k.csv) · [Statistics](statistics.csv) · [Full trajectories](trajectory.csv)","", "Audit errors: "+json.dumps(e)]
    (out/"RESULTS.md").write_text("\n".join(md)+"\n")
    print(json.dumps(result,indent=2))
    if e:raise RuntimeError("Incomplete; preserve archive")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["select","summarize"]);a=ap.parse_args()
    freeze() if a.mode=="select" else summary()
