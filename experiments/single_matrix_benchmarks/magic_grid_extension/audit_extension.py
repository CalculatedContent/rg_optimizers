"""Audit original endpoint weights without training; compare full expanded validation grid."""
import argparse,csv,json,math,os,pathlib,statistics
import numpy as np
import torch
from torch import nn
from run import D,ORIGINAL,P,CAP,TRACKS,sha,whash
from calibration import metrics_from_logits
R=D/"results"
FPRS=P["roc_fpr"]
OLD_PLAN=json.loads((ORIGINAL/"plan.json").read_text())
METS=["accuracy","ce","auc","brier","ece15","mean_confidence","confidence_correct","confidence_incorrect","high_confidence_fraction","high_confidence_error_fraction"]
def dump(p,j):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2,allow_nan=False)+"\n")
def csvout(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def data_check(root):
    mf=root/"prepared/manifest.json";dp=root/"prepared/data.npz"
    m=json.loads(mf.read_text())
    assert sha(root/"plan.json")==P["primary_plan_sha256"]==m["plan_sha256"]
    assert sha(dp)==P["primary_prepared_sha256"]==m["prepared_sha256"]
    assert m["n_rows"]==19020 and m["features"]==10 and m["classes"]==[6688,12332] and m["cross_partition_duplicate_feature_groups"]==0
    return mf,dp
def check_record(j,case,seed,manifest,plan_hash,source):
    assert j["status"]=="complete" and j["phase"]=="grid" and j["case"]==case["id"] and j["config"]==case and j["seed"]==seed
    assert j["start_step"]==0 and j["planned_updates"]==j["completed_updates"]==CAP and j["dropout"]==0 and j["selection_sha256"] is None
    assert j["plan_sha256"]==plan_hash and j["data_manifest_sha256"]==sha(manifest) and j["source_commit"]==source
    assert j["threads"]==1 and j["versions"]["weightwatcher"]=="0.7.7" and j["versions"]["torch"]=="2.7.1+cpu"
    assert [h["step"] for h in j["history"]]==P["checkpoints"]
    for h in j["history"]:
        assert set(h["metrics"])=={"train","validation"}
        assert h["spectral"]["matrix"]=="fc2" and h["spectral"]["shape"]==[784,784] and len(h["spectral"]["eigenvalues"])==784
        assert h["weights_sha256"]==h["spectral_weights_sha256"]
def check_weight(d,h):
    path=d/h["checkpoint_file"];assert sha(path)==h["checkpoint_sha256"]
    q=torch.load(path,map_location="cpu",weights_only=True)
    assert q["step"]==h["step"] and q["weights_sha256"]==h["weights_sha256"]==whash(q["model"])
    return q
def trim(j,origin):
    return {"origin":origin,"case":j["case"],"seed":j["seed"],"config":j["config"],"source_commit":j["source_commit"],"history":[{"step":h["step"],"weights_sha256":h["weights_sha256"],"checkpoint_sha256":h["checkpoint_sha256"],"metrics":h["metrics"],"spectral":h["spectral"]["raw"]} for h in j["history"]]}
def baseline(root,primary_sha):
    m,dp=data_check(root);m0,_=data_check(ORIGINAL);assert sha(m)==sha(m0)
    s=json.loads((root/"results/analysis/summary.json").read_text())
    assert s["complete"] and s["n_grid"]==24 and s["n_confirm"]==15 and not s["audit_errors"]
    frozen=root/"results/frozen_selection.json";sel=json.loads(frozen.read_text())
    assert sel["complete_grid"] and sel["n_grid"]==24 and not sel["audit_errors"] and sel["plan_sha256"]==P["primary_plan_sha256"]
    with np.load(dp) as a:x,y,train,val=[a[k] for k in ("x","y","train","val")]
    mean=x[train].mean(0);sd=np.maximum(x[train].std(0),1e-12)
    sets={k:(torch.from_numpy(((x[ix]-mean)/sd).astype(np.float32)),torch.from_numpy(y[ix].astype(np.int64))) for k,ix in {"train":train,"validation":val}.items()}
    records=[];errors=[]
    for cfg in OLD_PLAN["grid"]:
        for seed in OLD_PLAN["dev_seeds"]:
            try:
                folder=root/"results/grid"/f"{cfg['id']}-{seed}";j=json.loads((folder/"result.json").read_text())
                check_record(j,cfg,seed,m,P["primary_plan_sha256"],P["primary_launch_commit"])
                assert np.allclose(j["normalization"]["mean"],mean,rtol=0,atol=1e-12) and np.allclose(j["normalization"]["sd"],sd,rtol=0,atol=1e-12)
                h=j["history"][-1];q=check_weight(folder,h)
                net=nn.Sequential(nn.Linear(10,784),nn.ReLU(),nn.Linear(784,784),nn.ReLU(),nn.Linear(784,2))
                net.load_state_dict(q["model"]);net.eval()
                for k,(xx,yy) in sets.items():
                    with torch.no_grad():logits=torch.cat([net(b) for b in xx.split(1024)])
                    v=metrics_from_logits(logits,yy,FPRS);old=h["metrics"][k]
                    for metric in ("accuracy","ce","auc"):
                        assert abs(v[metric]-old[metric])<=1e-6,(k,metric,v[metric],old[metric])
                    h["metrics"][k]={**old,**{a:b for a,b in v.items() if a not in old}}
                records.append(trim(j,"original"))
            except Exception as e:errors.append(f"{cfg['id']}/{seed}: {e!r}")
    if not errors:
        for t in TRACKS:
            def rank(c):
                vals=[r["history"][-1]["metrics"]["validation"] for r in records if r["case"]==c["id"]]
                return (-statistics.mean(v["accuracy"] for v in vals),statistics.mean(v["ce"] for v in vals),c["id"])
            winner=min((c for c in OLD_PLAN["grid"] if c["track"]==t),key=rank)
            assert winner==sel["selected"][t],(winner,sel["selected"][t])
    out={"primary_archive_commit":primary_sha,"primary_summary_sha256":sha(root/"results/analysis/summary.json"),"primary_selection_sha256":sha(frozen),"manifest_sha256":sha(m),"records":records,"errors":errors,"n_original":len(records),"complete":not errors and len(records)==24,"training_updates":0}
    dump(R/"baseline.json",out)
    print(json.dumps({k:v for k,v in out.items() if k!="records"},indent=2))
    if not out["complete"]:raise RuntimeError("Original endpoint audit failed; no new training")
def new_records(launch_sha):
    mf,_=data_check(ORIGINAL);rows=[];errors=[]
    for cfg in P["grid"]:
        for seed in P["dev_seeds"]:
            try:
                d=R/"grid"/f"{cfg['id']}-{seed}";j=json.loads((d/"result.json").read_text())
                check_record(j,cfg,seed,mf,sha(D/"plan.json"),launch_sha)
                for h in j["history"]:check_weight(d,h)
                q=torch.load(d/"final_resume.pt",map_location="cpu",weights_only=True)
                assert sha(d/"final_resume.pt")==j["final_resume_sha256"] and q["step"]==CAP and whash(q["model"])==j["history"][-1]["weights_sha256"]
                rows.append(trim(j,"extension"))
            except Exception as e:errors.append(f"{cfg['id']}/{seed}: {e!r}")
    return rows,errors
def stats(v):
    v=[x for x in v if x is not None and math.isfinite(x)]
    if len(v)!=2:return {"mean":None,"sd":None,"ci95_half_width":None,"n_valid":len(v),"n_expected":2}
    sd=statistics.stdev(v);return {"mean":statistics.mean(v),"sd":sd,"ci95_half_width":12.7062047364*sd/math.sqrt(2),"n_valid":2,"n_expected":2}
def summarize(launch_sha,primary_sha):
    errors=[];old=[]
    try:
        b=json.loads((R/"baseline.json").read_text())
        assert b["complete"] and b["n_original"]==24 and b["primary_archive_commit"]==primary_sha and b["training_updates"]==0 and not b["errors"]
        assert b["manifest_sha256"]==sha(ORIGINAL/"prepared/manifest.json")
        old=b["records"]
        assert {(r["case"],r["seed"]) for r in old}=={(c["id"],s) for c in OLD_PLAN["grid"] for s in P["dev_seeds"]}
    except Exception as e:errors.append(f"Baseline: {e!r}")
    new,e=new_records(launch_sha);errors.extend(e)
    out=R/"analysis";out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for j in old+new:
        for h in j["history"]:
            c=j["config"];r={"origin":j["origin"],"case":j["case"],"track":c["track"],"lr":c["lr"],"momentum":c.get("momentum"),"weight_decay":c["weight_decay"],"seed":j["seed"],"step":h["step"],"weights_sha256":h["weights_sha256"],"checkpoint_sha256":h["checkpoint_sha256"],"alpha":h["spectral"]["alpha"],"D":h["spectral"]["D"],"tail":h["spectral"]["tail"],"fit_status":h["spectral"]["status"]}
            for split in ("train","validation"):
                m=h["metrics"][split]
                for k in METS:r[split+"_"+k]=m.get(k)
                for fpr in FPRS:r[split+"_gamma_at_fpr_"+str(fpr)]=m.get("gamma_efficiency",{}).get(str(fpr))
            r["accuracy_gap"]=r["train_accuracy"]-r["validation_accuracy"]
            r["ce_gap"]=r["validation_ce"]-r["train_ce"]
            rows.append(r)
    fixed=[r for r in rows if r["step"]==CAP]
    csvout(out/"trajectories.csv",rows);csvout(out/"endpoints.csv",fixed)
    metric_keys=["alpha","D","tail","accuracy_gap","ce_gap"]+[s+"_"+k for s in ("train","validation") for k in METS]+["validation_gamma_at_fpr_"+str(f) for f in FPRS]
    cases=[];long=[]
    for cfg in OLD_PLAN["grid"]+P["grid"]:
        z=[r for r in fixed if r["case"]==cfg["id"]]
        case={"case":cfg["id"],"origin":"extension" if cfg in P["grid"] else "original","track":cfg["track"],"lr":cfg["lr"],"momentum":cfg.get("momentum"),"weight_decay":cfg["weight_decay"],"metrics":{k:stats([r[k] for r in z]) for k in metric_keys}}
        cases.append(case)
        for k,v in case["metrics"].items():long.append({a:b for a,b in case.items() if a!="metrics"}|{"metric":k}|v)
    csvout(out/"case_statistics.csv",long)
    def get(c,k):return c["metrics"][k]["mean"]
    eligible=[c for c in cases if get(c,"validation_accuracy") is not None and get(c,"validation_ce") is not None]
    winners={}
    for t in TRACKS:
        z=[c for c in eligible if c["track"]==t]
        if not z:continue
        a=min(z,key=lambda c:(-get(c,"validation_accuracy"),get(c,"validation_ce"),c["case"]))
        b=min(z,key=lambda c:(get(c,"validation_ce"),-get(c,"validation_accuracy"),c["case"]))
        frontier=[c["case"] for c in z if not any(get(o,"validation_accuracy")>=get(c,"validation_accuracy") and get(o,"validation_ce")<=get(c,"validation_ce") and (get(o,"validation_accuracy")>get(c,"validation_accuracy") or get(o,"validation_ce")<get(c,"validation_ce")) for o in z)]
        winners[t]={"best_accuracy":a["case"],"best_ce":b["case"],"pareto_frontier":frontier}
    result={"complete":not errors and len(old)==24 and len(new)==24,"audit_errors":errors,"n_original":len(old),"n_new":len(new),"max_new_updates":480000,"verified_new_updates":len(new)*CAP,"primary_archive_commit":primary_sha,"launch_commit":launch_sha,"plan_sha256":sha(D/"plan.json"),"failed_fits_original":sum(r["fit_status"]!="success" for r in rows if r["origin"]=="original"),"failed_fits_new":sum(r["fit_status"]!="success" for r in rows if r["origin"]=="extension"),"cases":cases,"winners":winners,"note":"Validation only, two paired seeds per case, nominal t intervals df=1; no new test access or confirmation training."}
    dump(out/"summary.json",result)
    dump(out/"selection_by_metric.json",winners)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots=[("alpha","validation_accuracy","accuracy_vs_alpha"),("alpha","validation_ce","cross_entropy_vs_alpha"),("alpha","validation_ece15","calibration_vs_alpha"),("validation_accuracy","validation_ce","accuracy_vs_cross_entropy"),("validation_accuracy","validation_mean_confidence","confidence_vs_accuracy")]
    for x,y,name in plots:
        fig,ax=plt.subplots(figsize=(7,5))
        for t in TRACKS:
            for o,marker in (("original","o"),("extension","x")):
                z=[r for r in fixed if r["track"]==t and r["origin"]==o and r[x] is not None and r[y] is not None]
                if z:ax.scatter([r[x] for r in z],[r[y] for r in z],marker=marker,label=t+" "+o,alpha=.85)
        if x=="alpha":ax.axvline(2,ls="--",color="gray")
        ax.set(xlabel=x,ylabel=y,title="MAGIC fixed-20k validation grid; individual seeds");ax.legend(fontsize=7);fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(out/f"{name}.{ext}",dpi=160)
        plt.close(fig)
    for metric in ("validation_accuracy","validation_ce","alpha","validation_ece15","validation_mean_confidence"):
        fig,ax=plt.subplots(figsize=(8,5))
        for t in TRACKS:
            for cfg in P["grid"]:
                if cfg["track"]!=t:continue
                z=[(s,stats([r[metric] for r in rows if r["case"]==cfg["id"] and r["step"]==s])["mean"]) for s in P["checkpoints"]]
                ax.plot([s for s,v in z if v is not None],[v for s,v in z if v is not None],marker=".",label=cfg["id"]+" "+t)
        if metric=="alpha":ax.axhline(2,ls="--",color="gray")
        ax.set(xlabel="Updates",ylabel=metric,title="New grid: two-seed means");ax.legend(fontsize=6,ncol=2);fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(out/f"trajectory_{metric}.{ext}",dpi=160)
        plt.close(fig)
    def fmt(c,k,scale=1):
        v=c["metrics"][k];return "NA" if v["mean"] is None else f'{scale*v["mean"]:.4f} ± {scale*v["ci95_half_width"]:.4f}'
    md=["# MAGIC expanded fixed-20k validation grid","",f"Complete: {result['complete']}. Original 24-run cohort reused: {len(old)}/24; new: {len(new)}/24. No test evaluation or new confirmation. Failed original/new WW fits: {result['failed_fits_original']}/{result['failed_fits_new']}.","",f"Original immutable archive: https://github.com/CalculatedContent/rg_optimizers/tree/{primary_sha}/experiments/single_matrix_benchmarks/magic/results","", "Nominal two-seed Student-t 95% confidence intervals (df=1); broad and exploratory, not test-population intervals. Mean confidence is predictive confidence, not statistical uncertainty. ECE uses 15 equal-width bins; Brier is binary squared probability error. No selection by alpha.","","| Case | Track | LR | Momentum | Val acc (%) | Val CE | ROC-AUC | ECE | Brier | Mean confidence | FC2 α |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cases:
        md.append(f"| {c['case']} | {c['track']} | {c['lr']} | {c['momentum']} | {fmt(c,'validation_accuracy',100)} | {fmt(c,'validation_ce')} | {fmt(c,'validation_auc')} | {fmt(c,'validation_ece15')} | {fmt(c,'validation_brier')} | {fmt(c,'validation_mean_confidence')} | {fmt(c,'alpha')} |")
    md+=["","Validation-only winners and Pareto frontier: "+json.dumps(winners),"","[Endpoint rows](endpoints.csv) · [All case statistics](case_statistics.csv) · [Trajectories](trajectories.csv) · [Accuracy–α](accuracy_vs_alpha.png) · [CE–α](cross_entropy_vs_alpha.png) · [ECE–α](calibration_vs_alpha.png) · [Accuracy–CE](accuracy_vs_cross_entropy.png) · [Confidence–accuracy](confidence_vs_accuracy.png)","","Audit errors: "+json.dumps(errors)]
    (out/"RESULTS.md").write_text("\n".join(md)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))
    if not result["complete"]:raise RuntimeError("Incomplete; preserve archive")
if __name__=="__main__":
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    a=argparse.ArgumentParser();a.add_argument("mode",choices=["baseline","summarize"]);a.add_argument("--primary-root",type=pathlib.Path);a.add_argument("--primary-sha",required=True);a.add_argument("--launch-sha",default=os.getenv("GITHUB_SHA"));args=a.parse_args()
    if args.mode=="baseline":baseline(args.primary_root,args.primary_sha)
    else:summarize(args.launch_sha,args.primary_sha)
