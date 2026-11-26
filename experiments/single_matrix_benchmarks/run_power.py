"""Single-matrix regression/classification trainer; power-plant adapter first. Dry-run by default."""
import argparse, contextlib, hashlib, importlib.metadata, io, json, math, os, pathlib, platform, random, statistics, sys
HERE=pathlib.Path(__file__).resolve().parent
DS=HERE/"power_plant"
CAP=20000
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(path,value):
    tmp=path.with_name(path.name+".tmp");tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");os.replace(tmp,path)
def plan():
    p=json.loads((DS/"plan.json").read_text())
    if p["cap"]!=CAP or p["checkpoints"]!=[0,2000,5000,10000,15000,CAP] or p["architecture"]!=[4,784,784,1]:
        raise ValueError("Unexpected frozen cap or model")
    if len(p["grid"])!=12 or p["development_runs"]!=24 or p["confirmation_runs"]!=15:raise ValueError("Unexpected budget")
    return p
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--execute-training",action="store_true")
    a.add_argument("--phase",choices=["grid","confirm","dropout"])
    a.add_argument("--case")
    a.add_argument("--seed",type=int)
    a.add_argument("--authorize-dropout-extension",action="store_true")
    args=a.parse_args();p=plan()
    if not args.execute_training:
        print(json.dumps({"status":"dry_run_no_training","grid":[{"case":c["id"],"seed":s,"updates":CAP} for c in p["grid"] for s in p["dev_seeds"]],
           "confirmation":{"tracks":["sgdm_wd0","adamw_wd0","adamw_wd001"],"seeds":p["confirmation_seeds"],"updates":CAP},
           "optional_dropout_not_launched":p["optional_dropout_runs"]},indent=2));return
    if args.phase=="dropout" and not args.authorize_dropout_extension:raise ValueError("Separate dropout authorization required")
    if args.phase not in ("grid","confirm","dropout"):raise ValueError("Specify phase")
    selection_hash=None
    if args.phase=="grid":
        if args.seed not in p["dev_seeds"]:raise ValueError("Unexpected development seed")
        cfg=next(c for c in p["grid"] if c["id"]==args.case)
    else:
        if args.seed not in p["confirmation_seeds"]:raise ValueError("Unexpected confirmation seed")
        sp=DS/"results"/"frozen_selection.json";sel=json.loads(sp.read_text())
        if sel["complete_grid"] is not True or sel["audit_errors"] or sel["plan_sha256"]!=sha(DS/"plan.json"):raise ValueError("Selection not verified")
        cfg=sel["selected"][args.case]
        if cfg not in p["grid"]:raise ValueError("Selected configuration differs")
        selection_hash=sha(sp)
        if args.phase=="dropout":
            counterpart=DS/"results"/"confirm"/f"{cfg['track']}-{args.seed}"/"result.json"
            base=json.loads(counterpart.read_text())
            if base["status"]!="complete" or base["completed_updates"]!=CAP or base["config"]!=cfg or base["dropout"]!=0:raise ValueError("Missing matched no-dropout confirmation")
    prob=p["dropout_extension"] if args.phase=="dropout" else 0.
    folder=DS/"results"/args.phase/f"{args.case}-{args.seed}"
    if folder.exists():raise FileExistsError("Refuse overwrite/restart/resume")
    import numpy as np
    import torch
    from torch import nn
    import weightwatcher as ww
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    manifest=DS/"prepared"/"manifest.json";data=DS/"prepared"/"data.npz"
    dm=json.loads(manifest.read_text())
    if sha(data)!=dm["prepared_sha256"]:raise ValueError("Prepared data mismatch")
    with np.load(data) as d:
        x,y=d["x"],d["y"];idx=d["train"] if args.phase=="grid" else d["outer_train"]
        subsets={"train":idx}
        if args.phase=="grid":subsets["validation"]=d["val"]
        else:subsets["test"]=d["test"]
    mean=x[idx].mean(0);sd=np.maximum(x[idx].std(0),1e-12);ym=float(y[idx].mean());ys=max(float(y[idx].std()),1e-12)
    sets={k:(torch.from_numpy(((x[v]-mean)/sd).astype(np.float32)),torch.from_numpy(((y[v]-ym)/ys).astype(np.float32))) for k,v in subsets.items()}
    seed=args.seed;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    m=nn.Sequential(nn.Linear(4,784),nn.ReLU(),nn.Dropout(prob),nn.Linear(784,784),nn.ReLU(),nn.Dropout(prob),nn.Linear(784,1))
    opt=(torch.optim.SGD(m.parameters(),lr=cfg["lr"],momentum=cfg["momentum"],dampening=0,nesterov=False,weight_decay=cfg["weight_decay"])
         if cfg["optimizer"]=="sgdm" else torch.optim.AdamW(m.parameters(),lr=cfg["lr"],betas=tuple(cfg["betas"]),eps=cfg["eps"],weight_decay=cfg["weight_decay"]))
    def whash(state):
        h=hashlib.sha256()
        for k,v in sorted(state.items()):
            h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def fit(w):
        try:
            z=nn.Sequential(nn.Linear(784,784,bias=False))
            with torch.no_grad():z[0].weight.copy_(torch.as_tensor(w,dtype=torch.float32))
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                frame=ww.WeightWatcher(z).analyze(min_evals=8,randomize=False,plot=False)
            if len(frame)!=1:raise ValueError("Expected one matrix")
            row=json.loads(frame.to_json(orient="records"))[0];a,d,k=row.get("alpha"),row.get("D"),row.get("num_pl_spikes")
            valid=row.get("status")=="success" and isinstance(a,(int,float)) and math.isfinite(a) and a>1 and isinstance(d,(int,float)) and math.isfinite(d) and 0<=d<=1 and isinstance(k,(int,float)) and 2<=k<=784
            return {"alpha":a if valid else None,"D":d if valid else None,"tail":k if valid else None,"status":"success" if valid else "failed","ww_row":row}
        except Exception as e:return {"alpha":None,"D":None,"tail":None,"status":"failed","error":repr(e)}
    def spectral(step):
        st=torch.get_rng_state();ns=np.random.get_state();ps=random.getstate()
        try:
            w=m[3].weight.detach().cpu().numpy().copy();f=fit(w)
            eig=np.sort(np.linalg.svd(w.astype(np.float64),compute_uv=False)**2)
            r=1-math.sqrt(1-prob);fs=[]
            for k in range(p["mask_repetitions"] if prob else 1):
                rng=np.random.default_rng(np.random.SeedSequence([20261126,seed,step,k]))
                fs.append(fit(w*(rng.random(w.shape)>=r)) if prob else f)
            vals=[g["alpha"] for g in fs]
            return {"matrix":"fc2","shape":[784,784],"raw":f,"eigenvalues":eig.tolist(),"mask_probability":r,
               "corrected_alpha":statistics.mean(vals) if all(v is not None for v in vals) else None,"corrected_fits":fs}
        finally:torch.set_rng_state(st);np.random.set_state(ns);random.setstate(ps)
    def evaluate(name):
        xx,yy=sets[name];old=m.training;m.eval()
        with torch.no_grad():pred=torch.cat([m(b).squeeze(-1) for b in xx.split(1024)])
        m.train(old);err=(pred-yy).double()*ys
        return {"rmse":float(err.square().mean().sqrt()),"mae":float(err.abs().mean()),"n":len(yy)}
    folder.mkdir(parents=True)
    gen=torch.Generator().manual_seed(seed)
    report={"status":"running","phase":args.phase,"case":args.case,"seed":seed,"config":cfg,"dropout":prob,"start_step":0,"planned_updates":CAP,"completed_updates":0,
      "plan_sha256":sha(DS/"plan.json"),"data_manifest_sha256":sha(manifest),"selection_sha256":selection_hash,"source_commit":os.getenv("GITHUB_SHA"),
      "versions":{k:importlib.metadata.version(k) for k in ("torch","numpy","weightwatcher","powerlaw")},
      "cpu":next((line.split(":",1)[1].strip() for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),platform.processor()) if pathlib.Path("/proc/cpuinfo").exists() else platform.processor(),
      "threads":1,"python":platform.python_version(),"initial_weights_sha256":whash(m.state_dict()),"normalization":{"x_mean":mean.tolist(),"x_sd":sd.tolist(),"y_mean":ym,"y_sd":ys},"history":[]}
    xx,yy=sets["train"];n=len(yy);offset=n;perm=None;step=0
    def observe():
        h=whash(m.state_dict());path=folder/f"checkpoint_{step:05d}.pt";torch.save({"step":step,"model":m.state_dict(),"weights_sha256":h},path)
        metrics={name:evaluate(name) for name in sets if name!="test" or step==CAP}
        s=spectral(step)
        if whash(m.state_dict())!=h:raise ValueError("Measurement mutated weights")
        row={"step":step,"weights_sha256":h,"spectral_weights_sha256":h,"checkpoint_file":path.name,"checkpoint_sha256":sha(path),"metrics":metrics,"spectral":s}
        report["history"].append(row);report["completed_updates"]=step;atomic_json(folder/"result.json",report)
        print(json.dumps({"step":step,"metrics":metrics,"alpha":s["raw"]["alpha"],"corrected":s["corrected_alpha"]}),flush=True)
    try:
        observe();m.train()
        while step<CAP:
            if offset==n:perm=torch.randperm(n,generator=gen);offset=0
            ids=perm[offset:offset+p["batch_size"]];offset+=len(ids)
            opt.zero_grad(set_to_none=True);loss=(m(xx[ids]).squeeze(-1)-yy[ids]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),p["clip"],error_if_nonfinite=True);opt.step();step+=1
            if step in p["checkpoints"]:observe()
        assert step==CAP and report["history"][-1]["step"]==CAP
        torch.save({"step":step,"model":m.state_dict(),"optimizer":opt.state_dict(),"batch_rng":gen.get_state(),"torch_rng":torch.get_rng_state(),"permutation":perm,"offset":offset},folder/"final_resume.pt")
        report.update(status="complete",completed_updates=step,final_resume_sha256=sha(folder/"final_resume.pt"));atomic_json(folder/"result.json",report)
    except Exception as e:
        report.update(status="failed",completed_updates=step,error=repr(e));atomic_json(folder/"result.json",report);raise
if __name__=="__main__":main()
