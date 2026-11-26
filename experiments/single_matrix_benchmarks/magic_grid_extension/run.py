"""Fixed-20k single-matrix classification. Dry-run default."""
import argparse, contextlib, hashlib, importlib.metadata, io, json, math, os, pathlib, random
import statistics
D=pathlib.Path(__file__).resolve().parent
ORIGINAL=D.parent/"magic"
from calibration import metrics_from_logits
P=json.loads((D/"plan.json").read_text()); CAP=20000
TRACKS=["sgdm_wd0","adamw_wd0","adamw_wd001"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
    tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n");os.replace(tmp,p)
def whash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--execute-training",action="store_true");ap.add_argument("--phase",choices=["grid"]);ap.add_argument("--case");ap.add_argument("--seed",type=int);a=ap.parse_args()
    assert P["cap"]==CAP and P["architecture"]==[10,784,784,2] and P["dropout"]==0 and len(P["grid"])==12 and len(P["dev_seeds"])==2 and P["confirmation_runs"]==0
    if not a.execute_training:
        print(json.dumps({"status":"dry_run_no_training","grid":[{"case":c["id"],"seed":s,"updates":CAP} for c in P["grid"] for s in P["dev_seeds"]]},indent=2));return
    selsha=None
    if a.phase=="grid":
        assert a.seed in P["dev_seeds"];cfg=next(c for c in P["grid"] if c["id"]==a.case)
    else:raise ValueError("Phase required")
    folder=D/"results"/a.phase/f"{a.case}-{a.seed}"
    if folder.exists():raise FileExistsError("Refuse to restart or overwrite")
    import numpy as np, torch, weightwatcher as ww
    from torch import nn
    from sklearn.metrics import roc_auc_score, roc_curve
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    mf=ORIGINAL/"prepared/manifest.json";dp=ORIGINAL/"prepared/data.npz";manifest=json.loads(mf.read_text())
    assert manifest["prepared_sha256"]==sha(dp)==P["primary_prepared_sha256"] and manifest["plan_sha256"]==sha(ORIGINAL/"plan.json")==P["primary_plan_sha256"]
    with np.load(dp) as d:
        x,y=d["x"],d["y"];idx=d["train"]
        subsets={"train":idx,"validation":d["val"]}
    mean=x[idx].mean(0);sd=np.maximum(x[idx].std(0),1e-12)
    sets={k:(torch.from_numpy(((x[v]-mean)/sd).astype(np.float32)),torch.from_numpy(y[v].astype(np.int64))) for k,v in subsets.items()}
    seed=a.seed;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    m=nn.Sequential(nn.Linear(10,784),nn.ReLU(),nn.Linear(784,784),nn.ReLU(),nn.Linear(784,2))
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
            return {"matrix":"fc2","shape":[784,784],"raw":f,"eigenvalues":eig.tolist()}
        finally:torch.set_rng_state(st);np.random.set_state(ns);random.setstate(ps)
    def evaluate(name):
        xx,yy=sets[name];old=m.training;m.eval()
        with torch.no_grad():logits=torch.cat([m(b) for b in xx.split(1024)])
        m.train(old)
        return metrics_from_logits(logits,yy,P["roc_fpr"])
    folder.mkdir(parents=True)
    gen=torch.Generator().manual_seed(seed);step=0
    report={"phase":a.phase,"case":a.case,"seed":seed,"config":cfg,"status":"running","start_step":0,"planned_updates":CAP,"completed_updates":0,"dropout":0,"plan_sha256":sha(D/"plan.json"),"data_manifest_sha256":sha(mf),"selection_sha256":selsha,"source_commit":os.getenv("GITHUB_SHA"),"versions":{k:importlib.metadata.version(k) for k in ("torch","numpy","weightwatcher","powerlaw","scikit-learn")},"threads":1,"initial_weights_sha256":whash(m.state_dict()),"normalization":{"mean":mean.tolist(),"sd":sd.tolist()},"history":[]}
    xx,yy=sets["train"];n=len(yy);offset=n;perm=None
    def observe():
        h=whash(m.state_dict());path=folder/f"checkpoint_{step:05d}.pt";torch.save({"step":step,"model":m.state_dict(),"weights_sha256":h},path)
        metrics={k:evaluate(k) for k in sets}
        spec=spectral();assert whash(m.state_dict())==h
        report["history"].append({"step":step,"weights_sha256":h,"spectral_weights_sha256":h,"checkpoint_file":path.name,"checkpoint_sha256":sha(path),"metrics":metrics,"spectral":spec})
        report["completed_updates"]=step;dump(folder/"result.json",report)
        print(json.dumps({"step":step,"metrics":metrics,"alpha":spec["raw"]["alpha"],"D":spec["raw"]["D"],"tail":spec["raw"]["tail"]}),flush=True)
    try:
        observe();m.train()
        while step<CAP:
            if offset==n:perm=torch.randperm(n,generator=gen);offset=0
            ids=perm[offset:offset+P["batch_size"]];offset+=len(ids)
            opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(m(xx[ids]),yy[ids])
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),P["clip"],error_if_nonfinite=True);opt.step();step+=1
            if step in P["checkpoints"]:observe()
        assert step==CAP and report["history"][-1]["step"]==CAP
        torch.save({"step":step,"model":m.state_dict(),"optimizer":opt.state_dict(),"batch_rng":gen.get_state(),"torch_rng":torch.get_rng_state(),"permutation":perm,"offset":offset},folder/"final_resume.pt")
        report.update(status="complete",final_resume_sha256=sha(folder/"final_resume.pt"));dump(folder/"result.json",report)
    except Exception as e:
        report.update(status="failed",completed_updates=step,error=repr(e));dump(folder/"result.json",report);raise
if __name__=="__main__":main()
