"""Audit the fixed endpoints; freeze selection before test; summarize and plot."""
import argparse,csv,hashlib,json,math,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent; ROOT=HERE/"results"
PLAN=json.loads((HERE/"plan.json").read_text()); CAP=20000
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")
def csvout(p,rows):
    if not rows:return
    with p.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def verify(phase,job,cfg,seed,selection_sha=None):
    folder=ROOT/phase/job;r=json.loads((folder/"result.json").read_text())
    if (r["job"]!=job or r["phase"]!=phase or r["seed"]!=seed or r["config"]!=cfg or r["status"]!="complete"
        or r["planned_updates"]!=CAP or r["completed_updates"]!=CAP or r["selection_sha256"]!=selection_sha
        or r["plan_sha256"]!=sha(HERE/"plan.json") or r["data_manifest_sha256"]!=sha(ROOT/"data"/"manifest.json")
        or r["prepared_sha256"]!=sha(ROOT/"data"/"prepared.npz")):raise ValueError("Identity or cap mismatch")
    if [h["step"] for h in r["history"]]!=PLAN["checkpoints"]:raise ValueError("Incomplete history")
    for h in r["history"]:
        if h["weights_sha256"]!=h["spectral_weights_sha256"] or sha(folder/h["checkpoint_file"])!=h["checkpoint_sha256"]:raise ValueError("Weight binding mismatch")
        if phase=="grid" and "test" in h["metrics"]:raise ValueError("Development test leakage")
        if phase=="confirm" and "test" not in h["metrics"]:raise ValueError("Missing confirmation test")
    if sha(folder/"resume_final.pt")!=r["final_resume_sha256"]:raise ValueError("Final resume hash mismatch")
    return r
def read_grid():
    records=[];errors=[]
    for cfg in PLAN["cases"]:
        for seed in PLAN["development_seeds"]:
            job=f"grid-{cfg['id']}-{seed}"
            try:records.append(verify("grid",job,cfg,seed))
            except Exception as e:errors.append(f"{job}: {e}")
    return records,errors
def selection(records):
    scores=[]
    for lr in sorted(set(c["lr"] for c in PLAN["cases"])):
        for momentum in sorted(set(c["momentum"] for c in PLAN["cases"])):
            group=[r for r in records if r["config"]["lr"]==lr and r["config"]["momentum"]==momentum]
            if len(group)!=4:raise ValueError("Incomplete paired setting")
            scores.append(dict(lr=lr,momentum=momentum,mean_validation_loss=statistics.mean(r["history"][-1]["metrics"]["validation"]["loss"] for r in group),
                mean_validation_accuracy=statistics.mean(r["history"][-1]["metrics"]["validation"]["accuracy"] for r in group)))
    return sorted(scores,key=lambda x:(x["mean_validation_loss"],x["lr"],x["momentum"]))
def select():
    records,errors=read_grid();out=ROOT/"selection";out.mkdir(parents=True,exist_ok=True)
    scores=selection(records) if not errors else []
    s=dict(complete_grid=not errors,n_grid=len(records),audit_errors=errors,plan_sha256=sha(HERE/"plan.json"),
        selected=scores[0] if scores else None,all_scores=scores,selection_rule=PLAN["selection"])
    dump(out/"frozen_selection.json",s);csvout(out/"grid_pair_scores.csv",scores);print(json.dumps(s,indent=2))
    if errors:raise RuntimeError("Incomplete grid; confirmation refused")
def endpoint_rows(records):
    rows=[]
    for r in records:
        cfg=r["config"]
        for h in r["history"]:
            s=h["spectral"];m=h["metrics"]
            row=dict(phase=r["phase"],job=r["job"],seed=r["seed"],case=cfg["id"],lr=cfg["lr"],momentum=cfg["momentum"],dropout=cfg["dropout"],step=h["step"],
              train_loss=m["train"]["loss"],train_accuracy=m["train"]["accuracy"],validation_loss=m["validation"]["loss"],validation_accuracy=m["validation"]["accuracy"],
              test_loss=m.get("test",{}).get("loss"),test_accuracy=m.get("test",{}).get("accuracy"),
              alpha_raw=s["raw_alpha"],alpha_corrected=s["corrected_alpha"],alpha_raw_all_four=s["raw_alpha_all_four"],alpha_corrected_all_four=s["corrected_alpha_all_four"],weights_sha256=h["weights_sha256"])
            for l in s["layers"]:
                name=l["layer"];row[name+"_alpha"]=l["raw"]["alpha"];row[name+"_alpha_corrected"]=l["corrected_alpha"]
                row[name+"_raw_D"]=l["raw"]["D"];row[name+"_raw_tail"]=l["raw"]["tail"];row[name+"_mask_sd"]=l["corrected_alpha_sd"]
            rows.append(row)
    return rows
def stats(values):
    if any(v is None or not math.isfinite(v) for v in values) or not values:return dict(n_valid=sum(v is not None and math.isfinite(v) for v in values),n_expected=len(values),mean=None,sd=None,ci95_half_width=None)
    n=len(values);s=statistics.stdev(values) if n>1 else None
    return dict(n_valid=n,n_expected=n,mean=statistics.mean(values),sd=s,ci95_half_width=2.7764451052*s/math.sqrt(n) if n==5 else None)
def plots(grid,conf,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name,xkey in [("raw","alpha_raw"),("corrected","alpha_corrected")]:
        fig,ax=plt.subplots(figsize=(7,5))
        for p in PLAN["dropout_rates"]:
            g=[r for r in grid if r["step"]==CAP and r["dropout"]==p and r[xkey] is not None]
            ax.scatter([r[xkey] for r in g],[100*r["validation_accuracy"] for r in g],label=f"dropout {p:.3f}",alpha=.8)
        ax.axvline(2,ls="--",color="gray");ax.set(xlabel=f"{name} confidence-tail alpha (FC1-FC3)",ylabel="Validation accuracy (%)",title="SGDM grid at 20,000 updates");ax.legend();fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(out/f"grid_validation_accuracy_vs_{name}_alpha.{ext}",dpi=160)
        plt.close(fig)
    if conf:
        for metric in ("test_accuracy","alpha_raw","alpha_corrected"):
            fig,ax=plt.subplots(figsize=(7,5))
            for p in PLAN["dropout_rates"]:
                for step in []:pass
                steps=[];means=[];err=[]
                for step in PLAN["checkpoints"]:
                    g=[r[metric] for r in conf if r["step"]==step and r["dropout"]==p]
                    if len(g)==5 and all(v is not None for v in g):
                        f=100 if metric=="test_accuracy" else 1;steps.append(step);means.append(f*statistics.mean(g));err.append(f*2.7764451052*statistics.stdev(g)/math.sqrt(5))
                if steps:ax.errorbar(steps,means,yerr=err,marker="o",label=f"dropout {p:.3f}")
            if "alpha" in metric:ax.axhline(2,ls="--",color="gray")
            ax.set(xlabel="Training updates",ylabel="Test accuracy (%)" if metric=="test_accuracy" else metric,title="Five paired confirmation seeds");ax.legend();fig.tight_layout()
            for ext in ("png","pdf"):fig.savefig(out/f"confirmation_{metric}_trajectory.{ext}",dpi=160)
            plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5))
        for p in PLAN["dropout_rates"]:
            for metric,marker in [("alpha_raw","o"),("alpha_corrected","x")]:
                g=[r for r in conf if r["step"]==CAP and r["dropout"]==p and r[metric] is not None]
                ax.scatter([r[metric] for r in g],[100*r["test_accuracy"] for r in g],label=f"p={p:.3f} {metric}",marker=marker)
        ax.axvline(2,ls="--",color="gray");ax.set(xlabel="Confidence-tail alpha",ylabel="Test accuracy (%)",title="Fixed 20k confirmation");ax.legend();fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(out/f"confirmation_test_accuracy_vs_alpha.{ext}",dpi=160)
        plt.close(fig)
def summarize():
    out=ROOT/"analysis";out.mkdir(parents=True,exist_ok=True)
    grid,errors=read_grid();conf=[];sel=None
    sp=ROOT/"selection"/"frozen_selection.json"
    try:
        sel=json.loads(sp.read_text())
        if errors or not sel["complete_grid"] or sel["audit_errors"] or sel["selected"]!=selection(grid)[0] or sel["plan_sha256"]!=sha(HERE/"plan.json"):raise ValueError("Frozen selection mismatch")
        for pidx in (0,1):
            cfg=next(c for c in PLAN["cases"] if c["lr"]==sel["selected"]["lr"] and c["momentum"]==sel["selected"]["momentum"] and c["dropout_index"]==pidx)
            for seed in PLAN["confirmation_seeds"]:
                job=f"confirm-p{pidx}-{seed}"
                try:conf.append(verify("confirm",job,cfg,seed,sha(sp)))
                except Exception as e:errors.append(f"{job}: {e}")
    except Exception as e:errors.append(f"selection: {e}")
    gr=endpoint_rows(grid);cr=endpoint_rows(conf);csvout(out/"grid_trajectory.csv",gr);csvout(out/"confirmation_trajectory.csv",cr)
    csvout(out/"grid_fixed_20k.csv",[r for r in gr if r["step"]==CAP]);csvout(out/"confirmation_fixed_20k.csv",[r for r in cr if r["step"]==CAP])
    summaries=[]
    metrics=["test_accuracy","test_loss","alpha_raw","alpha_corrected","alpha_raw_all_four","alpha_corrected_all_four"]+[f"{l}_{m}" for l in ("fc1","fc2","fc3","fc4") for m in ("alpha","alpha_corrected")]
    for p in PLAN["dropout_rates"]:
        g=[r for r in cr if r["dropout"]==p and r["step"]==CAP]
        for key in metrics:
            v=[r[key] for r in g]+[None]*(5-len(g))
            summaries.append(dict(dropout=p,metric=key,**stats(v)))
    csvout(out/"confirmation_statistics.csv",summaries)
    pair_rows=[];zero={r["seed"]:r for r in cr if r["step"]==CAP and r["dropout"]==0}
    for r in cr:
        if r["step"]!=CAP or not r["dropout"] or r["seed"] not in zero:continue
        b=zero[r["seed"]]
        for layer in ("model","fc2","fc3","fc4"):
            rawkey="alpha_raw" if layer=="model" else layer+"_alpha";corrkey="alpha_corrected" if layer=="model" else layer+"_alpha_corrected"
            raw,correct,target=r[rawkey],r[corrkey],b[rawkey]
            pair_rows.append(dict(seed=r["seed"],layer=layer,p=r["dropout"],no_dropout_alpha=target,dropout_raw_alpha=raw,dropout_corrected_alpha=correct,
                raw_error=abs(raw-target) if raw is not None and target is not None else None,corrected_error=abs(correct-target) if correct is not None and target is not None else None))
    csvout(out/"paired_dropout_comparison.csv",pair_rows)
    failed_fits=sum(l["raw"]["status"]!="success" for r in grid+conf for h in r["history"] for l in h["spectral"]["layers"])
    failed_mask_fits=sum(f["status"]!="success" for r in grid+conf for h in r["history"] for l in h["spectral"]["layers"] for f in l["corrected_fits"])
    complete=len(grid)==len(PLAN["cases"])*2 and len(conf)==10 and not errors
    result=dict(complete_grid=len(grid)==len(PLAN["cases"])*2,complete_confirmation=len(conf)==10,n_grid=len(grid),n_confirmation=len(conf),cap=CAP,audit_errors=errors,
        selected=sel.get("selected") if sel else None,raw_fit_failures=failed_fits,corrected_fit_failures=failed_mask_fits,statistics=summaries,paired_dropout=pair_rows,complete=complete)
    try:
        if gr:plots(gr,cr,out)
    except Exception as e:result["audit_errors"].append(f"plot: {e}");result["complete"]=False
    dump(out/"summary.json",result)
    md=["# Mini Speech Commands SGDM results","",f"Complete: {result['complete']}. Grid {len(grid)}/24; confirmation {len(conf)}/10. Fixed cap: 20,000 updates.",
        "",f"Selected paired LR/momentum: {result['selected']}.", "", "| Dropout | Metric | Mean | Sample SD | 95% CI half-width |","|---:|---|---:|---:|---:|"]
    for v in summaries:md.append(f"| {v['dropout']:.6f} | {v['metric']} | {v['mean']} | {v['sd']} | {v['ci95_half_width']} |")
    md+=["","No alpha or test selection. Invalid fits retained as null; primary average requires FC1-FC3, secondary includes small-N FC4.",
        f"Raw fit failures: {failed_fits}; corrected fit failures: {failed_mask_fits}.","", "Audit errors: "+json.dumps(result["audit_errors"]),
        "", "See CSVs, plots, all run directories, data manifest, and frozen selection. Square-root masking is an empirical diagnostic, not an exact inversion of dropout training."]
    (out/"RESULTS.md").write_text("\n".join(md)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("statistics","paired_dropout")},indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["select","summarize"]);a=p.parse_args()
    select() if a.mode=="select" else summarize()
if __name__=="__main__":main()
