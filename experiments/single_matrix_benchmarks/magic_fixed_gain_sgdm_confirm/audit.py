"""Archive completed or partial fixed-gain confirmation, without rerunning training."""
import csv,json,math,statistics
import torch
from run import D,P,CAP,sha,whash,dump
R=D/"results";A=R/"analysis"; A.mkdir(parents=True,exist_ok=True)
seeds=P["confirmation_seeds"];rows=[];errors=[];endpoints=[]
metrics=("accuracy","ce","auc","brier","ece15","logit_rms")
for seed in seeds:
    folder=R/"confirm"/f"sgdm_wd0-{seed}"
    try:
        j=json.loads((folder/"result.json").read_text())
        assert j["seed"]==seed and j["config"]==P["grid"][0] and j["plan_sha256"]==sha(D/"plan.json")
        assert j["data_manifest_sha256"]==sha(D.parent/"magic/prepared/manifest.json")
        initial=None
        for h in j["history"]:
            p=folder/h["checkpoint_file"]
            assert sha(p)==h["checkpoint_sha256"]
            c=torch.load(p,map_location="cpu",weights_only=True)
            assert c["step"]==h["step"] and whash(c["model"])==h["weights_sha256"]==c["weights_sha256"]==h["spectral_weights_sha256"]
            gains=[float(c["model"][f"{i}.gain"]) for i in (0,2,4)]
            if initial is None: initial=gains
            assert gains==initial
            assert set(h["metrics"])==({"train","test"} if h["step"]==CAP else {"train"})
            raw=h["spectral"]["raw"]; q=dict(seed=seed,step=h["step"],status=j["status"],alpha=raw["alpha"],D=raw["D"],tail=raw["tail"],fit_status=raw["status"],gain_fc2=gains[1],gain_fc3=gains[2],clip_count=h["clipping"]["count"],clip_mean=h["clipping"]["mean_factor"])
            for part,m in h["metrics"].items():
                for k in metrics:q[part+"_"+k]=m.get(k)
            for k,v in (h.get("last_sampled_angular") or {}).items():q["angular_"+k]=v
            rows.append(q)
        assert j["status"]=="complete" and j["completed_updates"]==CAP and [h["step"] for h in j["history"]]==P["checkpoints"]
        p=folder/"final_resume.pt";assert sha(p)==j["final_resume_sha256"]
        c=torch.load(p,map_location="cpu",weights_only=True);assert c["step"]==CAP and whash(c["model"])==j["history"][-1]["weights_sha256"]
        endpoints.append(rows[-1])
    except Exception as e:errors.append(f"{seed}: {type(e).__name__}: {e}")
def write(name,rr):
    keys=list(dict.fromkeys(k for r in rr for k in r))
    with (A/name).open("w") as f:
        if keys:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
write("trajectories.csv",rows);write("fixed_20k.csv",endpoints)
controls=[]
for arch,dirname in (("ordinary","magic"),("learned_gain","magic_scale_invariant")):
    for seed in seeds:
        try:
            p=D.parent/dirname/"results/confirm"/f"adamw_wd0-{seed}"/"result.json"
            j=json.loads(p.read_text());h=j["history"][-1]
            assert j["status"]=="complete" and j["seed"]==seed and h["step"]==CAP and j["config"]["lr"]==.0001 and j["config"]["weight_decay"]==0
            q=dict(architecture=arch,seed=seed,alpha=h["spectral"]["raw"]["alpha"],D=h["spectral"]["raw"]["D"],tail=h["spectral"]["raw"]["tail"],source_file_sha256=sha(p))
            for k in ("accuracy","ce","auc"):q["test_"+k]=h["metrics"]["test"][k]
            controls.append(q)
        except Exception as e:errors.append(f"control {arch}/{seed}: {e}")
write("controls.csv",controls)
stats=[];paired=[]
from scipy.stats import t
def agg(arm,metric,vals):
    n=len(vals); sd=statistics.stdev(vals) if n>1 else None
    return dict(arm=arm,metric=metric,n=n,mean=statistics.mean(vals) if n else None,ci95_half_width=float(t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None)
for k in ("alpha","D","tail","test_accuracy","test_ce","test_auc","test_brier","test_ece15","train_accuracy","train_ce"):
    stats.append(agg("fixed_gain_sgdm",k,[q[k] for q in endpoints if q.get(k) is not None]))
for arch in ("ordinary","learned_gain"):
    for k in ("alpha","D","tail","test_accuracy","test_ce","test_auc"):
        cc=[q for q in controls if q["architecture"]==arch and q[k] is not None]
        stats.append(agg(arch+"_adamw",k,[q[k] for q in cc]))
        idx={q["seed"]:q[k] for q in cc}
        ds=[q[k]-idx[q["seed"]] for q in endpoints if q["seed"] in idx and q.get(k) is not None]
        paired.append(agg("fixed_sgdm_minus_"+arch+"_adamw",k,ds))
write("statistics.csv",stats);write("paired_differences.csv",paired)
complete=len(endpoints)==5 and len(rows)==30 and len(controls)==10 and not errors
dump(A/"summary.json",dict(complete=complete,n_completed=len(endpoints),n_checkpoints=len(rows),raw_fit_failures=sum(q["fit_status"]!="success" for q in rows),errors=errors,statistics=stats,paired=paired,protocol=P["test_protocol"]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
for seed in seeds:
    rr=[q for q in rows if q["seed"]==seed and q["alpha"] is not None]
    if rr:ax.plot([q["step"] for q in rr],[q["alpha"] for q in rr],marker="o",label=str(seed))
ax.axhline(2,color="gray",ls="--");ax.set(xlabel="Updates",ylabel="Raw FC2 alpha");ax.legend();fig.tight_layout();fig.savefig(A/"alpha_trajectory.png",dpi=160);plt.close(fig)
lines=["# MAGIC fixed-gain SGDM five-seed exploratory confirmation","",f"Complete: {complete}; completed {len(endpoints)}/5; checkpoints {len(rows)}/30; errors: {errors}","", "Configuration chosen by user after observing development alpha; not the validation accuracy winner. Fixed gains, LR .03, momentum .9, zero decay, 20k updates. Same original outer-train/test split and confirmation seeds. Test reused; this is exploratory. AdamW controls are archived models with different parameterization (ordinary or learned gain), not a matched fixed-gain AdamW confirmation.","","| Arm | Metric | n | Mean | 95% CI half-width |","|---|---|---:|---:|---:|"]
for q in stats+paired:lines.append("| "+" | ".join(str(q[k]) for k in ("arm","metric","n","mean","ci95_half_width"))+" |")
lines+=["","Intervals are across training seeds, not independent test-population uncertainty. Paired differences use matching seeds. No alpha-based selection among new runs. Report failures and short tails without excluding them silently.","","[Endpoints](fixed_20k.csv) · [Trajectories](trajectories.csv) · [Paired differences](paired_differences.csv) · [Alpha plot](alpha_trajectory.png)"]
(A/"RESULTS.md").write_text("\n".join(lines)+"\n")
assert complete,errors
