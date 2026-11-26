import csv, hashlib, json, math, pathlib, statistics, contextlib, io
import numpy as np
import torch
import weightwatcher as ww

torch.set_num_threads(1)
root=pathlib.Path("california_housing_sgdm_20k/results/2026-11-26_sgdm_20k/runs")
out=pathlib.Path("california_housing_sgdm_20k/dropout_correction/reproduced/scan")
out.mkdir(exist_ok=True)
p=0.3690073279296028
rs=[0.025,0.05,0.075,0.094,0.1,0.15,0.2,0.25,0.3,p,0.5,0.6,0.65]
reps=10
rows=[]; layers=[]; audit=[]
def model_hash(state):
    d=hashlib.sha256()
    for name,value in sorted(state.items()):
        d.update(name.encode()); d.update(str(tuple(value.shape)).encode())
        d.update(str(value.dtype).encode()); d.update(value.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()
def fit(a):
    a=np.asarray(a,dtype=np.float64); a=np.sort(a[a>0])
    logs=np.log(a); n=len(a)-np.arange(len(a)-1)
    s=np.cumsum(logs[::-1])[::-1][:-1]-n*logs[:-1]
    alphas=1+np.divide(n,s,out=np.full_like(s,np.inf),where=s>0)
    ds=np.full(len(n),np.inf)
    for i,(k,alpha) in enumerate(zip(n,alphas)):
        if np.isfinite(alpha):
            ds[i]=np.max(np.abs(np.arange(k)/k - (1-(a[i:]/a[i])**(1-alpha))))
    i=int(np.argmin(ds))
    return float(alphas[i]),float(ds[i]),int(n[i]),float(a[i])
def ev(w):
    return np.linalg.eigvalsh(w.T@w)
def add(seed,method,r,rep,a):
    alpha,D,n,xmin=fit(a)
    vals=np.maximum(a,0); s=vals.sum()
    rows.append(dict(seed=seed,method=method,r=r,rep=rep,alpha=alpha,D=D,num_pl_spikes=n,xmin=xmin,
        effective_rank=float(np.exp(-np.sum((vals[vals>0]/s)*np.log(vals[vals>0]/s)))),
        participation_rank=float(s*s/np.sum(vals*vals))))
for seed in range(3340,3345):
    folder=root/f"seed_{seed}"
    state=torch.load(folder/"final_weights.pt",map_location="cpu",weights_only=True)
    if "model" in state and isinstance(state["model"],dict): state=state["model"]
    record=json.loads((folder/"result.json").read_text())
    assert record["completed_updates"]==20000 and record["config"]["dropout"]==p
    assert model_hash(state)==record["final_weights_sha256"]
    measured=next(h for h in record["history"] if h["step"]==20000)
    assert measured["weights_sha256"]==measured["spectral_weights_sha256"]==record["final_weights_sha256"]
    w=state["backbone.blocks.1.0.weight"].detach().double().numpy().copy()
    assert w.shape==(784,784)
    orig=measured["spectral"]["matrices"][0]
    x=w.T@w; a=np.linalg.eigvalsh(x)
    alpha,D,n,xmin=fit(a)
    assert abs(alpha-orig["alpha"])<1e-6 and abs(D-orig["D"])<1e-6 and n==orig["num_pl_spikes"], (seed,alpha,D,n,orig["alpha"])
    audit.append(dict(seed=seed,weight_hash=model_hash(state),original_alpha=alpha,original_D=D,original_tail=n))
    add(seed,"original",0,0,a)
    # FC1: shipped small-N analysis, not a hand-set exponent.
    tensor=state["backbone.blocks.0.0.weight"]
    try:
        copied=torch.nn.Sequential(torch.nn.Linear(tensor.shape[1],tensor.shape[0],bias=False))
        with torch.no_grad(): copied[0].weight.copy_(tensor)
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            frame=ww.WeightWatcher(copied).analyze(min_evals=8)
        layers.append(dict(seed=seed,matrix="FC1",fits=json.loads(frame.to_json(orient="records"))))
    except Exception as exc:
        layers.append(dict(seed=seed,matrix="FC1",error=repr(exc)))
    for ri,r in enumerate(rs):
        c=r/(1-r)
        add(seed,"isotropic_expectation",r,0,a+c*np.trace(x)/len(a))
        add(seed,"exact_mask_expectation",r,0,np.linalg.eigvalsh(x+c*np.diag(np.diag(x))))
        for rep in range(reps):
            rng=np.random.default_rng(np.random.SeedSequence([seed,ri,rep,20261126]))
            masked=w*(rng.random(w.shape)>=r)
            add(seed,"entry_mask",r,rep,ev(masked))
            flat=w.ravel().copy()
            indices=rng.choice(flat.size,size=round(r*flat.size),replace=False)
            flat[indices]=rng.permutation(flat[indices])
            add(seed,"partial_permutation",r,rep,ev(flat.reshape(w.shape)))
    print("completed seed",seed,flush=True)
with (out/"raw_fits.csv").open("w") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary=[]
for method,r in sorted(set((z["method"],z["r"]) for z in rows)):
    group=[z for z in rows if z["method"]==method and z["r"]==r]
    seed_means=[statistics.mean(z["alpha"] for z in group if z["seed"]==seed) for seed in range(3340,3345)]
    summary.append(dict(method=method,r=r,mean_alpha=statistics.mean(seed_means),
        sd_across_seed_means=statistics.stdev(seed_means),seed_means=seed_means,
        mean_D=statistics.mean(z["D"] for z in group),mean_tail=statistics.mean(z["num_pl_spikes"] for z in group),
        mean_effective_rank=statistics.mean(z["effective_rank"] for z in group),n=len(group)))
report=dict(source_commit="95cf2d49400c258588398383a9a041e5603e5837",updates=0,training_started=False,
    original_dropout=p,randomization_repeats=reps,audit_errors=[],weight_audit=audit,fc1=layers,summary=summary)
(out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({"closest":sorted(summary,key=lambda z:abs(z["mean_alpha"]-2))[:12],"fc1":layers},indent=2))

