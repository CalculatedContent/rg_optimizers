import csv, hashlib, json, math, pathlib, statistics
import numpy as np
import torch
torch.set_num_threads(1)
root=pathlib.Path("california_housing_sgdm_20k/results/2026-11-26_sgdm_20k")
out=pathlib.Path("california_housing_sgdm_20k/dropout_correction/reproduced/formula")
out.mkdir(exist_ok=True)
p=0.3690073279296028; q=1-p; r=1-math.sqrt(q)
reps=20
rows=[]; dev=[]; audit=[]
def model_hash(state):
    d=hashlib.sha256()
    for name,value in sorted(state.items()):
        d.update(name.encode()); d.update(str(tuple(value.shape)).encode())
        d.update(str(value.dtype).encode()); d.update(value.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()
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
def ev(w):
    g=w.T@w if w.shape[1]<=w.shape[0] else w@w.T
    return np.linalg.eigvalsh(g)
def add(seed,method,rep,a):
    a=np.asarray(a); a=a[a>0]
    alpha,D,n,xmin=fit(a); z=a/a.sum()
    rows.append(dict(seed=seed,method=method,rep=rep,alpha=alpha,D=D,tail=n,rank=len(a),
        participation_rank=float(1/np.sum(z*z)),effective_rank=float(np.exp(-np.sum(z*np.log(z))))))
for seed in range(3340,3345):
    folder=root/"runs"/f"seed_{seed}"
    state=torch.load(folder/"final_weights.pt",map_location="cpu",weights_only=True)
    if "model" in state and isinstance(state["model"],dict): state=state["model"]
    record=json.loads((folder/"result.json").read_text())
    assert record["completed_updates"]==20000 and record["config"]["dropout"]==p
    assert model_hash(state)==record["final_weights_sha256"]
    measured=next(h for h in record["history"] if h["step"]==20000)
    assert measured["weights_sha256"]==measured["spectral_weights_sha256"]==record["final_weights_sha256"]
    w=state["backbone.blocks.1.0.weight"].detach().double().numpy().copy()
    x=w.T@w; a=np.linalg.eigvalsh(x); orig=measured["spectral"]["matrices"][0]
    alpha,D,n,_=fit(a)
    assert abs(alpha-orig["alpha"])<1e-6 and abs(D-orig["D"])<1e-6 and n==orig["num_pl_spikes"]
    audit.append(dict(seed=seed,weight_hash=model_hash(state),baseline_alpha=alpha))
    add(seed,"original",0,a)
    add(seed,"exact_expected_gram_p",0,np.linalg.eigvalsh(q*x+p*np.diag(np.diag(x))))
    add(seed,"fractional_expected_gram_sqrt_q",0,np.linalg.eigvalsh((1-r)*x+r*np.diag(np.diag(x))))
    for rep in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([20261127,seed,rep]))
        # Actual activation dropout: remove columns for input dropout; rows for output dropout.
        cols=rng.random(w.shape[1])<q; rr=rng.random(w.shape[0])<q
        add(seed,"input_neuron_dropout_p",rep,ev(w[:,cols]))
        add(seed,"output_neuron_dropout_p",rep,ev(w[rr,:]))
        add(seed,"two_sided_neuron_dropout_p_per_side",rep,ev(w[np.ix_(rr,cols)]))
        # Marginal expectation equivalence, not the same spectral distribution as neuron dropout.
        uniform=rng.random(w.shape)
        add(seed,"entry_mask_p",rep,ev(w*(uniform<q)))
        # Prespecified equal composition of two dropout channels: hypothesis only.
        add(seed,"entry_mask_sqrt_q",rep,ev(w*(uniform<math.sqrt(q))))
        # Mean-zero independent noise with per-column second moment fixed by W.
        noise=rng.standard_normal(w.shape)*np.sqrt(np.diag(x)/w.shape[0])[None,:]
        add(seed,"variance_matched_noise_p",rep,ev(math.sqrt(q)*w+math.sqrt(p)*noise))
        # Replacement probability from pairwise survival, matching expected off-diagonal attenuation.
        # Zero-mean replacement with column second moments, independent draws.
        add(seed,"independent_replacement_sqrt_q",rep,ev(np.where(uniform<math.sqrt(q),w,noise)))
    print("completed",seed,flush=True)
# Evaluate spectrum-only isotropic corrections against all paired, valid original development endpoints.
inventory=json.loads((root/"source_inventory.json").read_text())
by={}
for d in inventory["development"]:
    path=root/d["result"]["local_path"]
    if not path.is_file(): path=pathlib.Path(d["result"]["repository_path"])
    assert path.is_file(), str(path)
    j=json.loads(path.read_text())
    h=next(z for z in j["history"] if z["step"]==20000)
    assert h["spectral_weights_sha256"]==h["weights_sha256"]
    m=h["spectral"]["matrices"][0]
    if m["status"]!="success": continue
    by[(int(d["candidate_id"].split("_")[-1]),d["seed"])]=(j["config"]["dropout"],m)
for (i,seed),(drop,m) in by.items():
    if i%2!=1 or (i-1,seed) not in by: continue
    zero,control=by[(i-1,seed)]
    assert zero==0 and drop==p
    a=np.asarray(m["eigenvalues"]); mean=a.mean()
    for method,prob in [("original",0),("isotropic_full_p",p),("isotropic_half_channel",r)]:
        aa=a+prob/(1-prob)*mean; alpha,D,n,_=fit(aa)
        dev.append(dict(candidate=f"sgdm_{i:02d}",seed=seed,method=method,dropout=p,
                        alpha=alpha,no_dropout_alpha=control["alpha"],absolute_error=abs(alpha-control["alpha"]),
                        target2_error=abs(alpha-2),D=D,tail=n))
def summarize(group):
    seedmeans=[statistics.mean(z["alpha"] for z in group if z["seed"]==seed) for seed in range(3340,3345)]
    return dict(mean_alpha=statistics.mean(seedmeans),sd_seed_means=statistics.stdev(seedmeans),seed_means=seedmeans,
        mean_D=statistics.mean(z["D"] for z in group),mean_tail=statistics.mean(z["tail"] for z in group),
        mean_rank=statistics.mean(z["rank"] for z in group),mean_participation_rank=statistics.mean(z["participation_rank"] for z in group),
        n=len(group))
summary={m:summarize([z for z in rows if z["method"]==m]) for m in sorted(set(z["method"] for z in rows))}
devsummary={}
for m in sorted(set(z["method"] for z in dev)):
    g=[z for z in dev if z["method"]==m]
    near=[z for z in g if abs(z["no_dropout_alpha"]-2)<=.25]
    devsummary[m]=dict(n=len(g),mean_absolute_paired_error=statistics.mean(z["absolute_error"] for z in g),
        n_controls_near_2=len(near),mean_absolute_paired_error_near_2=statistics.mean(z["absolute_error"] for z in near) if near else None)
for name,data in [("confirmation_fits",rows),("development_spectrum_comparison",dev)]:
    with (out/(name+".csv")).open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
report=dict(training_updates=0,training_started=False,p=p,q=q,half_channel_probability=r,repetitions=reps,
    audit_errors=[],weight_audit=audit,confirmation=summary,development=devsummary,
    limitations="One nonzero training dropout rate; half channel is a hypothesis, not implied by single activation dropout; spectrum-only development controls cannot assess elementwise or neuron masking.")
(out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps(report,indent=2))

