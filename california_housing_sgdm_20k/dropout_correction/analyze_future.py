"""Read-only checkpoint analysis for the prepared, independently authorized future study."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
CAP=20000
def fit(a):
    a=np.sort(np.asarray(a,dtype=np.float64))
    a=a[a>0]
    logs=np.log(a); n=len(a)-np.arange(len(a)-1)
    s=np.cumsum(logs[::-1])[::-1][:-1]-n*logs[:-1]
    alphas=1+np.divide(n,s,out=np.full_like(s,np.inf),where=s>0)
    ds=np.full(len(n),np.inf)
    for i,(k,alpha) in enumerate(zip(n,alphas)):
        if np.isfinite(alpha):
            ds[i]=np.max(np.abs(np.arange(k)/k-(1-(a[i:]/a[i])**(1-alpha))))
    i=int(np.argmin(ds))
    return float(alphas[i]),float(ds[i]),int(n[i]),float(a[i])

def model_hash(state):
    d=hashlib.sha256()
    for name,value in sorted(state.items()):
        d.update(name.encode()); d.update(str(tuple(value.shape)).encode())
        d.update(str(value.dtype).encode()); d.update(value.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()
def ev(w):
    return np.linalg.svd(w,compute_uv=False)**2
def safe_fit(w):
    try:
        alpha,D,n,xmin=fit(ev(w))
        if not (math.isfinite(alpha) and alpha>1 and math.isfinite(D) and n>=2):
            raise ValueError("Invalid fit")
        return {"alpha":alpha,"D":D,"tail":n,"status":"success"}
    except Exception as e:
        return {"alpha":None,"D":None,"tail":None,"status":"failed:"+str(e)}
def main():
    torch.set_num_threads(1)
    plan=json.loads((HERE/"future_plan.json").read_text())
    rates,seeds=plan["dropout_probabilities"],plan["fresh_seeds"]
    assert plan["primary_step"]==CAP and rates[0]==0
    rows=[]; raw=[]; baseline={}
    for i,p in enumerate(rates):
        for seed in seeds:
            folder=HERE/"future_results"/f"p{i:02d}"/f"seed_{seed}"
            record=json.loads((folder/"result.json").read_text())
            assert record["status"]=="complete" and record["completed_updates"]==CAP and record["start_step"]==0
            assert record["config"]=={**plan["optimizer"],"dropout":p}
            checkpoint=torch.load(folder/"checkpoint_step_20000.pt",map_location="cpu",weights_only=False)
            assert checkpoint["completed_updates"]==CAP
            state=checkpoint["model"]
            measured=next(h for h in record["history"] if h["step"]==CAP)
            assert model_hash(state)==record["final_weights_sha256"]==measured["weights_sha256"]==measured["spectral_weights_sha256"]
            w=state[plan["primary_layer"]].detach().double().numpy().copy()
            orig=measured["spectral"]["matrices"][0]
            a=safe_fit(w)
            if orig["status"]!="success":
                a={"alpha":None,"D":None,"tail":None,"status":"archived_fit_failed"}
            else:
                assert a["status"]=="success" and abs(a["alpha"]-orig["alpha"])<1e-6 and abs(a["D"]-orig["D"])<1e-6 and a["tail"]==orig["num_pl_spikes"]
            if i==0:
                baseline[seed]=a["alpha"]
            for method,r in [("original",0),("full_p_mask",p),("sqrt_mask",1-math.sqrt(1-p))]:
                fits=[]
                for rep in range(1 if r==0 else plan["mask_repetitions"]):
                    rng=np.random.default_rng(np.random.SeedSequence([20261128,i,seed,rep]))
                    f=a.copy() if r==0 else safe_fit(w*(rng.random(w.shape)>=r))
                    fits.append(f)
                    raw.append(dict(rate_index=i,p=p,seed=seed,method=method,r=r,rep=rep,**f))
                valid=[f for f in fits if f["status"]=="success"]
                mean=statistics.mean(f["alpha"] for f in valid) if len(valid)==len(fits) else None
                target=baseline[seed]
                rows.append(dict(rate_index=i,p=p,seed=seed,method=method,r=r,alpha=mean,
                    control_alpha=target,paired_error=abs(mean-target) if mean is not None and target is not None else None,
                    n_valid=len(valid),n_requested=len(fits),test_rmse=measured["test"]["rmse"],test_mae=measured["test"]["mae"],
                    weights_sha256=record["final_weights_sha256"]))
    summary=[]
    for i,p in enumerate(rates):
        for method in ("original","full_p_mask","sqrt_mask"):
            g=[z for z in rows if z["rate_index"]==i and z["method"]==method]
            valid=[z for z in g if z["paired_error"] is not None]
            complete=len(valid)==len(seeds)
            summary.append(dict(rate_index=i,p=p,method=method,n_valid=len(valid),n_expected=len(seeds),
                mean_alpha=statistics.mean(z["alpha"] for z in valid) if complete else None,
                mean_absolute_paired_error=statistics.mean(z["paired_error"] for z in valid) if complete else None,
                sd_paired_error=statistics.stdev(z["paired_error"] for z in valid) if complete else None,
                mean_test_rmse=statistics.mean(z["test_rmse"] for z in g),
                mean_test_mae=statistics.mean(z["test_mae"] for z in g)))
    out=HERE/"future_results"/"analysis";out.mkdir(exist_ok=True)
    for name,data in [("raw_mask_fits",raw),("paired_seed_values",rows),("summary",summary)]:
        with (out/(name+".csv")).open("w") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(summary,indent=2,allow_nan=False))
if __name__=="__main__":
    main()
