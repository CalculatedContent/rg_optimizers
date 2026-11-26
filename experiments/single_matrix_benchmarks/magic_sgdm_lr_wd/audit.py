"""Frozen validation-only selection and complete/partial audit; never trains."""
import argparse,csv,json,math,pathlib,statistics
from run import D, ORIGINAL as O, P, sha, dump, whash
import numpy as np
import torch
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=D/"results"; A=R/"analysis"; S=R/"frozen_selection.json"
CFG={q["id"]:q for q in P["grid"]}
M=O/"prepared/manifest.json"; PH=sha(D/"plan.json"); MH=sha(M)
def audit(folder,phase,case,seed,cfg,reused=False,selection=None):
    f=folder/"result.json"; r=json.loads(f.read_text())
    assert r["phase"]==phase and r["seed"]==seed and r["start_step"]==0 and r["planned_updates"]==20000
    assert r["case"]==("s0" if phase=="grid" else "sgdm_wd0") if reused else r["case"]==case
    assert r["dropout"]==0 and r["data_manifest_sha256"]==MH
    assert r["plan_sha256"]==(P["primary_plan_sha256"] if reused else PH)
    assert all(r["config"][k]==cfg[k] for k in ("optimizer","lr","momentum","weight_decay"))
    if not reused: assert r["config"]==cfg and r["selection_sha256"]==selection
    else:
        assert r["source_commit"]=="9571f954a2070e717123f499ec9f41340a56b9ef"
        assert r["selection_sha256"]==(None if phase=="grid" else sha(O/"results/frozen_selection.json"))
    hist=r["history"]; steps=[h["step"] for h in hist]
    assert steps==P["checkpoints"][:len(steps)] and 0<=r["completed_updates"]<=20000
    for h in hist:
        path=folder/h["checkpoint_file"]
        assert sha(path)==h["checkpoint_sha256"]
        ck=torch.load(path,map_location="cpu",weights_only=True)
        assert ck["step"]==h["step"] and whash(ck["model"])==ck["weights_sha256"]==h["weights_sha256"]==h["spectral_weights_sha256"]
        assert set(h["metrics"])==({"train","validation"} if phase=="grid" else ({"train","test"} if h["step"]==20000 else {"train"}))
        e=np.asarray(h["spectral"]["eigenvalues"]); assert e.shape==(784,) and np.isfinite(e).all() and (e>=0).all()
        for k,v in (("spectral_norm",np.sqrt(e.max())),("frobenius_norm",np.sqrt(e.sum()))):
            if k in h["spectral"]:assert math.isclose(v,h["spectral"][k],rel_tol=1e-10)
    if r["status"]=="complete":
        assert steps==P["checkpoints"] and r["completed_updates"]==20000
        p=folder/"final_resume.pt"; assert sha(p)==r["final_resume_sha256"]
        ck=torch.load(p,map_location="cpu",weights_only=True); assert ck["step"]==20000 and whash(ck["model"])==hist[-1]["weights_sha256"]
    else:assert r["status"]=="failed" and "error" in r
    return r,{"path":str(f.relative_to(D.parent)),"result_sha256":sha(f),"status":r["status"],"reused":reused,"case":case,"seed":seed,"phase":phase}
def collect_grid():
    out=[]; refs=[]; errors=[]
    for case,cfg in CFG.items():
        reused=case=="g00"; seed=3470
        folder=O/"results/grid/s0-3470" if reused else R/"grid"/f"{case}-{seed}"
        try:
            r,ref=audit(folder,"grid",case,seed,cfg,reused)
            out.append((case,r));refs.append(ref)
        except Exception as e:errors.append({"phase":"grid","case":case,"seed":seed,"error":repr(e)})
    return out,refs,errors
def selection_from(grid,errors):
    scores=[]
    for case,r in grid:
        if r["status"]=="complete":
            v=r["history"][-1]["metrics"]["validation"]
            if math.isfinite(v["accuracy"]) and math.isfinite(v["ce"]):
                scores.append({"case":case,"validation_accuracy":v["accuracy"],"validation_ce":v["ce"],"config":CFG[case]})
    scores.sort(key=lambda a:(-a["validation_accuracy"],a["validation_ce"],a["case"]))
    return {"complete_grid":len(grid)==16 and not errors,"n_grid":len(grid),"audit_errors":errors,"plan_sha256":PH,"data_manifest_sha256":MH,"scores":scores,"selected":{"sgdm":scores[0]["config"]} if scores and len(grid)==16 and not errors else {}}
def select():
    grid,refs,errors=collect_grid(); sel=selection_from(grid,errors)
    R.mkdir(parents=True,exist_ok=True);dump(S,sel);dump(R/"grid_audit.json",refs)
    assert sel["complete_grid"] and sel["selected"],errors
    print(json.dumps(sel,indent=2))
def collect_confirm(sel):
    out=[];refs=[];errors=[]
    if not sel.get("selected"):return out,refs,errors
    cfg=sel["selected"]["sgdm"]; reused=cfg["id"]=="g00"
    for seed in P["confirmation_seeds"]:
        folder=O/"results/confirm"/f"sgdm_wd0-{seed}" if reused else R/"confirm"/f"sgdm-{seed}"
        try:
            r,ref=audit(folder,"confirm","sgdm",seed,cfg,reused,None if reused else sha(S))
            out.append(("sgdm",r));refs.append(ref)
        except Exception as e:errors.append({"phase":"confirm","seed":seed,"error":repr(e)})
    return out,refs,errors
def flatten(case,r,phase):
    rows=[]
    for h in r["history"]:
        sp=h["spectral"]; raw=sp["raw"]; e=np.asarray(sp["eigenvalues"])
        row={"phase":phase,"case":case,"seed":r["seed"],"status":r["status"],"step":h["step"],"lr":r["config"]["lr"],"weight_decay":r["config"]["weight_decay"],"alpha":raw["alpha"],"D":raw["D"],"tail":raw["tail"],"fit_status":raw["status"],"spectral_norm":float(np.sqrt(e.max())),"frobenius_norm":float(np.sqrt(e.sum()))}
        for split,m in h["metrics"].items():
            for k in ("accuracy","ce","auc"):row[f"{split}_{k}"]=m[k]
            for k,v in m["gamma_efficiency"].items():row[f"{split}_gamma_at_fpr_{k}"]=v
        for split in ("validation","test"):
            if split+"_ce" in row:
                row[split+"_ce_gap"]=row[split+"_ce"]-row["train_ce"]
                row[split+"_accuracy_gap"]=row["train_accuracy"]-row[split+"_accuracy"]
        rows.append(row)
    return rows
def csvout(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def stats(rows):
    out=[]
    for k in rows[0] if rows else []:
        if k in ("seed","lr","weight_decay","step"):continue
        a=[r[k] for r in rows if isinstance(r.get(k),(float,int)) and math.isfinite(r[k])]
        if a:
            n=len(a);sd=statistics.stdev(a) if n>1 else None
            out.append({"metric":k,"mean":statistics.mean(a),"sd":sd,"ci95_half_width":float(t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None,"n":n})
    return out
def plots(rows,ends):
    grid=[r for r in ends if r["phase"]=="grid" and r["status"]=="complete"]
    for k in ("validation_accuracy","validation_ce"):
        fig,ax=plt.subplots()
        for wd in [0,.0001,.001,.01]:
            a=[r for r in grid if r["weight_decay"]==wd and r["alpha"] is not None]
            ax.scatter([r["alpha"] for r in a],[r[k] for r in a],label=f"WD={wd:g}")
            for r in a:ax.annotate(f'{r["lr"]:g}',(r["alpha"],r[k]),fontsize=7)
        ax.axvline(2,ls="--",color="gray");ax.set(xlabel="Raw FC2 WW alpha",ylabel=k);ax.legend();fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(A/f"grid_{k}_vs_alpha.{ext}")
        plt.close(fig)
    for phase,metrics in (("grid",("alpha","validation_accuracy","validation_ce","validation_ce_gap","validation_accuracy_gap","spectral_norm","frobenius_norm","D","tail")),("confirm",("alpha","train_accuracy","train_ce","spectral_norm","frobenius_norm"))):
        group=[r for r in rows if r["phase"]==phase]
        if not group:continue
        for k in metrics:
            fig,ax=plt.subplots()
            for case,seed in sorted({(r["case"],r["seed"]) for r in group}):
                a=[r for r in group if r["case"]==case and r["seed"]==seed and r.get(k) is not None]
                if a:ax.plot([r["step"] for r in a],[r[k] for r in a],marker=".",label=case if phase=="grid" else str(seed))
            if k=="alpha":ax.axhline(2,ls="--",color="gray")
            ax.set(xlabel="Updates",ylabel=k);ax.legend(fontsize=6,ncol=2);fig.tight_layout()
            for ext in ("png","pdf"):fig.savefig(A/f"{phase}_{k}_trajectory.{ext}")
            plt.close(fig)
def summarize():
    A.mkdir(parents=True,exist_ok=True)
    grid,refs,errors=collect_grid(); computed=selection_from(grid,errors)
    sel=json.loads(S.read_text()) if S.exists() else {}
    if sel!=computed:errors.append({"phase":"selection","error":"missing or not identical to recomputed validation-only selection"})
    confirm,crefs,cerrors=collect_confirm(sel if sel==computed else {})
    errors+=cerrors; refs+=crefs
    rows=[row for phase,items in (("grid",grid),("confirm",confirm)) for case,r in items for row in flatten(case,r,phase)]
    ends=[r for r in rows if r["step"]==20000]
    cends=[r for r in ends if r["phase"]=="confirm" and r["status"]=="complete"]
    failed=[ref for ref in refs if ref["status"]!="complete"]
    summary={"complete":len(grid)==16 and len(cends)==5 and not errors,"n_grid":len(grid),"n_confirm_complete":len(cends),"audit_errors":errors,"failed_training":failed,"failed_fits":sum(r["fit_status"]!="success" for r in rows),"selection":sel,"references":refs,"confirmation_statistics":stats(cends),"original_reference":{"test_accuracy":.8761829652996845,"alpha":4.43919651632,"test_ce":.318897956609726,"test_auc":.9264236334275798,"xgboost_accuracy":.882229232386961},"test_protocol":"Exploratory reuse of previously inspected fixed test split; no selection by test or alpha."}
    csvout(A/"trajectories.csv",rows);csvout(A/"fixed_20k.csv",ends);csvout(A/"statistics.csv",summary["confirmation_statistics"])
    dump(A/"summary.json",summary);plots(rows,ends)
    lines=["# Accelerated MAGIC SGDM fixed-20k sweep","",f'Complete: {summary["complete"]}; grid {len(grid)}/16; confirmation {len(cends)}/5; failed fits {summary["failed_fits"]}.', "", "Selection: fixed-20k validation accuracy, then CE, never alpha/test. One development seed. Reused test is exploratory.", "", "Winner: "+str(sel.get("selected",{})), "", "| Metric | Mean | 95% CI half-width | n |","|---|---:|---:|---:|"]
    for x in summary["confirmation_statistics"]:lines.append(f'| {x["metric"]} | {x["mean"]:.8g} | {x["ci95_half_width"]} | {x["n"]} |')
    lines+=["","Original five-seed SGDM test accuracy 87.62%, alpha 4.439, CE .31890, AUC .92642; same-split XGBoost 88.22%.","", "Reused baseline files are referenced by path and SHA-256 in summary.json at immutable primary archive "+P["primary_archive"]+".","",f"Audit errors: {errors}",f"Failed training: {failed}","","[Endpoint CSV](fixed_20k.csv) · [All trajectories](trajectories.csv) · [Statistics](statistics.csv) · [Grid accuracy vs alpha](grid_validation_accuracy_vs_alpha.png) · [Grid CE vs alpha](grid_validation_ce_vs_alpha.png)"]
    (A/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"complete":summary["complete"],"errors":errors,"failed":failed}))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("mode",choices=["select","summarize"]);args=a.parse_args()
    torch.set_num_threads(1); select() if args.mode=="select" else summarize()
