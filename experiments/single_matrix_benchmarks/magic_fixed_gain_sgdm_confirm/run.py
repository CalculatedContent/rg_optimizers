"""Fixed-20k single-matrix classification. Dry-run default."""
import argparse, contextlib, hashlib, importlib.metadata, io, json, math, os, pathlib, random
import statistics
D=pathlib.Path(__file__).resolve().parent
P=json.loads((D/"plan.json").read_text()); CAP=20000
TRACKS=["sgdm_wd0"]
ORIGINAL=D.parent/"magic"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
    tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n");os.replace(tmp,p)
def whash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--execute-training",action="store_true");ap.add_argument("--phase",choices=["confirm"]);ap.add_argument("--case");ap.add_argument("--seed",type=int);a=ap.parse_args()
    assert P["cap"]==CAP and P["architecture"]==[10,784,784,2] and P["dropout"]==0 and len(P["grid"])==1 and len(P["dev_seeds"])==0 and P["confirmation_seeds"]==[3472,3473,3474,3475,3476]
    if not a.execute_training:
        print(json.dumps({"status":"dry_run_no_training","grid":[{"case":c["id"],"seed":s,"updates":CAP} for c in P["grid"] for s in P["dev_seeds"]],"confirmation":[{"track":t,"seed":s,"updates":CAP} for t in TRACKS for s in P["confirmation_seeds"]]},indent=2));return
    assert a.phase=="confirm" and a.case=="sgdm_wd0" and a.seed in P["confirmation_seeds"]
    cfg=P["grid"][0]
    assert cfg=={"id":"s2","track":"sgdm_wd0","optimizer":"sgdm","lr":.03,"momentum":.9,"weight_decay":0}
    selsha=sha(D/"plan.json")
    folder=D/"results"/a.phase/f"{a.case}-{a.seed}"
    if folder.exists():raise FileExistsError("Refuse to restart or overwrite")
    import numpy as np, torch, weightwatcher as ww
    from torch import nn
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold
    from scipy.optimize import minimize_scalar
    from scipy.special import expit
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    mf=ORIGINAL/"prepared/manifest.json";dp=ORIGINAL/"prepared/data.npz";manifest=json.loads(mf.read_text())
    assert manifest["prepared_sha256"]==sha(dp)==P["primary_prepared_sha256"] and manifest["plan_sha256"]==sha(ORIGINAL/"plan.json")==P["primary_plan_sha256"]
    with np.load(dp) as d:
        x,y=d["x"],d["y"];idx=d["outer_train"]
        assert a.phase=="confirm"
        subsets={"train":idx,"test":d["test"]}
    mean=x[idx].mean(0);sd=np.maximum(x[idx].std(0),1e-12)
    sets={k:(torch.from_numpy(((x[v]-mean)/sd).astype(np.float32)),torch.from_numpy(y[v].astype(np.int64))) for k,v in subsets.items()}
    seed=a.seed;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    from model import build
    m=build()
    fixed_gains={i:m[i].gain.detach().clone() for i in (0,2,4)}
    assert all(not m[i].gain.requires_grad for i in fixed_gains)
    opt=(torch.optim.SGD(m.parameters(),lr=cfg["lr"],momentum=cfg["momentum"],weight_decay=cfg["weight_decay"]) if cfg["optimizer"]=="sgdm" else torch.optim.AdamW(m.parameters(),lr=cfg["lr"],betas=tuple(cfg["betas"]),eps=cfg["eps"],weight_decay=cfg["weight_decay"]))
    def spectral():
        st=torch.get_rng_state();ns=np.random.get_state();ps=random.getstate()
        try:
            w=m[2].weight.detach().cpu().numpy().copy()
            eig=np.sort(np.linalg.svd(w.astype(np.float64),compute_uv=False)**2)
            f={"alpha":None,"D":None,"tail":None,"status":"failed"}
            try:
                z=nn.Sequential(nn.Linear(784,784,bias=False))
                with torch.no_grad():z[0].weight.copy_(torch.as_tensor(w,dtype=torch.float32))
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    frame=ww.WeightWatcher(z).analyze(min_evals=8,randomize=False,plot=False)
                if len(frame)!=1:raise ValueError("Expected one matrix")
                row=json.loads(frame.to_json(orient="records"))[0];al,ks,k=row.get("alpha"),row.get("D"),row.get("num_pl_spikes")
                valid=row.get("status")=="success" and isinstance(al,(int,float)) and math.isfinite(al) and al>1 and isinstance(ks,(int,float)) and math.isfinite(ks) and 0<=ks<=1 and isinstance(k,(int,float)) and 2<=k<=784
                f={"alpha":al if valid else None,"D":ks if valid else None,"tail":k if valid else None,"status":"success" if valid else "failed","ww_row":row}
            except Exception as e:f["error"]=repr(e)
            scale=abs(float(m[2].gain.detach()))*math.sqrt(784)/float(m[2].weight.detach().norm())
            return {"matrix":"fc2","shape":[784,784],"raw":f,"eigenvalues":eig.tolist(),"effective_eigenvalue_scale":scale**2,"raw_spectral_norm":float(np.sqrt(eig[-1])),"effective_spectral_norm":scale*float(np.sqrt(eig[-1]))}
        finally:torch.set_rng_state(st);np.random.set_state(ns);random.setstate(ps)
    def evaluate(name):
        xx,yy=sets[name];old=m.training;m.eval()
        with torch.no_grad():logits=torch.cat([m(b) for b in xx.split(1024)])
        m.train(old);prob=logits.softmax(-1)[:,1].numpy();target=yy.numpy()
        fpr,tpr,_=roc_curve(target,prob,drop_intermediate=False)
        result={"ce":float(nn.functional.cross_entropy(logits,yy)),"accuracy":float((logits.argmax(-1)==yy).double().mean()),"auc":float(roc_auc_score(target,prob)),"gamma_efficiency":{str(t):float(tpr[fpr<=t].max()) for t in P["roc_fpr"]},"n":len(yy)}
        z=(logits.double()[:,1]-logits.double()[:,0]).numpy()
        p=expit(z);y=target
        ece=0.
        for i in range(15):
            mask=(p>=i/15)&((p<(i+1)/15) if i<14 else (p<=1))
            if mask.any():ece+=float(mask.mean())*abs(float(p[mask].mean()-y[mask].mean()))
        result.update(brier=float(np.mean((p-y)**2)),ece15=ece,logit_rms=float(np.sqrt(np.mean(z*z))),auc_logit=float(roc_auc_score(y,z)))
        return result
    folder.mkdir(parents=True)
    gen=torch.Generator().manual_seed(seed);step=0
    report={"phase":a.phase,"case":a.case,"seed":seed,"config":cfg,"status":"running","start_step":0,"planned_updates":CAP,"completed_updates":0,"dropout":0,"plan_sha256":sha(D/"plan.json"),"data_manifest_sha256":sha(mf),"selection_sha256":selsha,"source_commit":os.getenv("GITHUB_SHA"),"versions":{k:importlib.metadata.version(k) for k in ("torch","numpy","weightwatcher","powerlaw","scikit-learn")},"threads":1,"initial_weights_sha256":whash(m.state_dict()),"normalization":{"mean":mean.tolist(),"sd":sd.tolist()},"history":[]}
    xx,yy=sets["train"];n=len(yy);offset=n;perm=None
    def radial_stats(grads):
        out={}
        for name,layer,g in zip(("fc1","fc2","fc3"),(m[0],m[2],m[4]),grads):
            w=layer.weight.detach().double(); g=g.detach().double(); n=float(w.norm()); gn=float(g.norm())
            out[name]={"raw_frobenius":n,"effective_frobenius":float(layer.effective_weight().detach().double().norm()),"gain":float(layer.gain.detach()),"grad_norm":gn,"relative_radial_residual":abs(float((w*g).sum()))/(n*gn) if n*gn>0 else None,"nominal_lr_over_norm_squared":cfg["lr"]/n**2}
        return out
    last_batch_radial=None
    clip_count=0;clip_sum=0.;angular_samples=[]
    def observe():
        assert all(torch.equal(m[i].gain.detach(),fixed_gains[i]) for i in fixed_gains)
        h=whash(m.state_dict());path=folder/f"checkpoint_{step:05d}.pt";torch.save({"step":step,"model":m.state_dict(),"weights_sha256":h},path)
        metrics={k:evaluate(k) for k in sets if k!="test" or step==CAP}
        spec=spectral()
        probe_loss=nn.functional.cross_entropy(m(xx[:P["batch_size"]]),yy[:P["batch_size"]])
        probe_grads=torch.autograd.grad(probe_loss,tuple(m[i].weight for i in (0,2,4)))
        radial=radial_stats(probe_grads)
        assert whash(m.state_dict())==h
        report["history"].append({"step":step,"weights_sha256":h,"spectral_weights_sha256":h,"checkpoint_file":path.name,"checkpoint_sha256":sha(path),"metrics":metrics,"spectral":spec,"radial_fixed_train_probe":radial,"last_update_preclip_radial":last_batch_radial,"clipping":{"count":clip_count,"mean_factor":clip_sum/step if step else None,"updates":step},"last_sampled_angular":angular_samples[-1] if angular_samples else None})
        report["completed_updates"]=step;dump(folder/"result.json",report)
        print(json.dumps({"step":step,"metrics":metrics,"alpha":spec["raw"]["alpha"],"D":spec["raw"]["D"],"tail":spec["raw"]["tail"]}),flush=True)
    try:
        observe();m.train()
        while step<CAP:
            if offset==n:perm=torch.randperm(n,generator=gen);offset=0
            ids=perm[offset:offset+P["batch_size"]];offset+=len(ids)
            opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(m(xx[ids]),yy[ids])
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
            loss.backward()
            if step+1 in P["checkpoints"]:last_batch_radial=radial_stats([m[i].weight.grad for i in (0,2,4)])
            total_norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),P["clip"],error_if_nonfinite=True));factor=min(1.,P["clip"]/(total_norm+1e-6));clip_sum+=factor;clip_count+=int(factor<1.)
            sample=(step+1 in P["checkpoints"]);prev=m[2].weight.detach().clone() if sample else None
            opt.step();step+=1
            if sample:
                u=prev.double();r=u.norm();u=u/r;delta=m[2].weight.detach().double()-prev.double();radial=(u*delta).sum();tangent=(delta-radial*u).norm();angular_samples.append({"step":step,"angle":float(torch.atan2(tangent,r+radial)),"relative_tangential":float(tangent/r),"radial_cosine":float(radial/delta.norm()) if delta.norm()>0 else None,"clip_factor":factor})
            if step in P["checkpoints"]:observe()
        assert step==CAP and report["history"][-1]["step"]==CAP
        torch.save({"step":step,"model":m.state_dict(),"optimizer":opt.state_dict(),"batch_rng":gen.get_state(),"torch_rng":torch.get_rng_state(),"permutation":perm,"offset":offset},folder/"final_resume.pt")
        report.update(status="complete",final_resume_sha256=sha(folder/"final_resume.pt"));dump(folder/"result.json",report)
    except Exception as e:
        report.update(status="failed",completed_updates=step,error=repr(e));dump(folder/"result.json",report);raise
if __name__=="__main__":main()
