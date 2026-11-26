"""Audit full cohorts, freeze selection, and report fixed endpoints. No training."""
import argparse,csv,hashlib,json,math,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent;DS=HERE/"power_plant";R=DS/"results";P=json.loads((DS/"plan.json").read_text());CAP=20000
TRACKS=["sgdm_wd0","adamw_wd0","adamw_wd001"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,j):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2,allow_nan=False)+"\n")
def csvout(p,rows):
    if rows:
        with p.open("w") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(phase,case,seed,cfg,selection_sha=None):
    d=R/phase/f"{case}-{seed}";j=json.loads((d/"result.json").read_text())
    if (j["status"]!="complete" or j["planned_updates"]!=CAP or j["completed_updates"]!=CAP or j["start_step"]!=0 or j["config"]!=cfg or j["seed"]!=seed or j["phase"]!=phase or j["case"]!=case or j["selection_sha256"]!=selection_sha or j["plan_sha256"]!=sha(DS/"plan.json") or j["data_manifest_sha256"]!=sha(DS/"prepared"/"manifest.json")):raise ValueError("Run identity/cap differs")
    if [x["step"] for x in j["history"]]!=P["checkpoints"]:raise ValueError("Incomplete history")
    for h in j["history"]:
        if sha(d/h["checkpoint_file"])!=h["checkpoint_sha256"] or h["weights_sha256"]!=h["spectral_weights_sha256"]:raise ValueError("Weight binding differs")
        if "test" in h["metrics"] and (phase=="grid" or h["step"]!=CAP):raise ValueError("Test access outside protocol")
    if sha(d/"final_resume.pt")!=j["final_resume_sha256"]:raise ValueError("Resume hash differs")
    if (phase=="grid" and "validation" not in j["history"][-1]["metrics"]) or (phase!="grid" and "test" not in j["history"][-1]["metrics"]):raise ValueError("Missing primary metric")
    return j
def grid():
    rows=[];errors=[]
    for c in P["grid"]:
        for seed in P["dev_seeds"]:
            try:rows.append(read("grid",c["id"],seed,c))
            except Exception as e:errors.append(f"{c['id']}/{seed}: {e}")
    return rows,errors
def select(g):
    scores=[]
    for c in P["grid"]:
        v=[j["history"][-1]["metrics"]["validation"]["rmse"] for j in g if j["config"]==c]
        if len(v)!=2 or not all(math.isfinite(x) for x in v):raise ValueError("Incomplete grid")
        scores.append({"case":c["id"],"track":c["track"],"mean_val_rmse":statistics.mean(v),"config":c})
    return {t:min([s for s in scores if s["track"]==t],key=lambda s:(s["mean_val_rmse"],s["case"]))["config"] for t in TRACKS},scores
def freeze():
    g,e=grid();selected,scores=select(g) if not e else ({},[])
    j={"complete_grid":not e,"n_grid":len(g),"audit_errors":e,"plan_sha256":sha(DS/"plan.json"),"selected":selected,"scores":scores}
    out=R/"frozen_selection.json"
    if out.exists() and json.loads(out.read_text())!=j:raise ValueError("Refuse changing frozen selection")
    dump(out,j);print(json.dumps(j,indent=2))
    if e:raise RuntimeError("Incomplete grid; no confirmation")
def values(records):
    ans=[]
    for j in records:
        for h in j["history"]:
            m=h["metrics"];s=h["spectral"]
            ans.append(dict(phase=j["phase"],case=j["case"],track=j["config"]["track"],seed=j["seed"],dropout=j["dropout"],step=h["step"],
                train_rmse=m["train"]["rmse"],validation_rmse=m.get("validation",{}).get("rmse"),test_rmse=m.get("test",{}).get("rmse"),test_mae=m.get("test",{}).get("mae"),
                alpha=s["raw"]["alpha"],D=s["raw"]["D"],tail=s["raw"]["tail"],corrected_alpha=s["corrected_alpha"],weights_sha256=h["weights_sha256"]))
    return ans
def stats(v):
    good=[x for x in v if x is not None and math.isfinite(x)]
    if len(good)!=5 or len(v)!=5:return {"mean":None,"sd":None,"ci95_half_width":None,"n_valid":len(good),"n_expected":5}
    sd=statistics.stdev(good);return {"mean":statistics.mean(good),"sd":sd,"ci95_half_width":2.7764451052*sd/math.sqrt(5),"n_valid":5,"n_expected":5}
def summary(include_dropout):
    g,e=grid();records=list(g);sel=None
    try:
        sp=R/"frozen_selection.json";sel=json.loads(sp.read_text())
        if e or not sel["complete_grid"] or sel["audit_errors"] or sel["selected"]!=select(g)[0] or sel["plan_sha256"]!=sha(DS/"plan.json"):raise ValueError("Frozen selection mismatch")
        for phase in (["confirm","dropout"] if include_dropout else ["confirm"]):
            for t in TRACKS:
                for seed in P["confirmation_seeds"]:
                    try:records.append(read(phase,t,seed,sel["selected"][t],sha(sp)))
                    except Exception as ex:e.append(f"{phase}/{t}/{seed}: {ex}")
    except Exception as ex:e.append(f"selection: {ex}")
    rows=values(records);out=R/"analysis";out.mkdir(exist_ok=True)
    csvout(out/"trajectory.csv",rows);fixed=[r for r in rows if r["step"]==CAP];csvout(out/"fixed_20k.csv",fixed)
    s=[]
    for phase in (["confirm","dropout"] if include_dropout else ["confirm"]):
        for t in TRACKS:
            for metric in ("test_rmse","test_mae","alpha","corrected_alpha"):
                a=[r[metric] for r in fixed if r["phase"]==phase and r["track"]==t]
                s.append(dict(phase=phase,track=t,metric=metric,**stats(a)))
    csvout(out/"statistics.csv",s)
    failed=sum(h["spectral"]["raw"]["status"]!="success" for j in records for h in j["history"])
    result={"complete":not e,"n_grid":len(g),"n_confirm":sum(j["phase"]=="confirm" for j in records),"n_dropout":sum(j["phase"]=="dropout" for j in records),"audit_errors":e,"raw_fit_failures":failed,"selection":sel,"statistics":s}
    dump(out/"summary.json",result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for phase,metric in [("grid","validation_rmse"),("confirm","test_rmse"),("dropout","test_rmse")]:
        if not any(r["phase"]==phase for r in fixed):continue
        fig,ax=plt.subplots(figsize=(7,5))
        for t in TRACKS:
            a=[r for r in fixed if r["phase"]==phase and r["track"]==t and r["alpha"] is not None and r[metric] is not None]
            ax.scatter([r["alpha"] for r in a],[r[metric] for r in a],label=t)
            if phase=="dropout":
                b=[r for r in a if r["corrected_alpha"] is not None];ax.scatter([r["corrected_alpha"] for r in b],[r[metric] for r in b],marker="x",label=t+" corrected")
        ax.axvline(2,color="gray",linestyle="--");ax.set(xlabel="FC2 alpha",ylabel=metric+" (MW)",title=f"Power plant fixed 20k {phase}");ax.legend(fontsize=8);fig.tight_layout()
        for ext in ("pdf","png"):fig.savefig(out/f"{phase}_{metric}_vs_alpha.{ext}",dpi=160)
        plt.close(fig)
    md=["# Power Plant fixed-20k results","",f"Complete: {result['complete']}; development {result['n_grid']}/24; confirmation {result['n_confirm']}/15; optional dropout {result['n_dropout']}/15.","","| Phase | Track | Metric | Mean | SD | 95% CI half width |","|---|---|---|---:|---:|---:|"]
    md += [f"| {v['phase']} | {v['track']} | {v['metric']} | {v['mean']} | {v['sd']} | {v['ci95_half_width']} |" for v in s]
    md+=["","Published outer split 0 only; 11 identical full test rows overlap outer train. Literature 20-split MC-dropout aggregate is context, not same estimand.","","Audit errors: "+json.dumps(e)]
    (out/"RESULTS.md").write_text("\n".join(md)+"\n")
    print(json.dumps(result,indent=2))
    if e:raise RuntimeError("Incomplete results; preserve archive")
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("mode",choices=["select","summarize"]);a.add_argument("--include-dropout",action="store_true");args=a.parse_args()
    freeze() if args.mode=="select" else summary(args.include_dropout)
