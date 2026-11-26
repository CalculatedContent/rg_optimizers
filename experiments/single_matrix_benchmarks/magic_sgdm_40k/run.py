"""Resume two frozen SGDM development runs from 20k to 40k, validation only."""
import argparse, contextlib, hashlib, importlib.metadata, io, json, math, os, pathlib, random
import statistics
D=pathlib.Path(__file__).resolve().parent
ORIGINAL=D.parent/"magic"
from calibration import metrics_from_logits
P=json.loads((D/"plan.json").read_text()); CAP=40000; START=20000
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
    tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n");os.replace(tmp,p)
def whash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--execute-training",action="store_true");ap.add_argument("--seed",type=int,required=True);a=ap.parse_args()
    assert a.seed in P["seeds"] and P["cap"]==CAP and P["start_step"]==START and P["architecture"]==[10,784,784,2] and P["dropout"]==0
    assert P["checkpoints"]==[20000,25000,30000,35000,40000] and len(P["seeds"])==2 and P["max_additional_updates"]==40000
    cfg=P["config"]
    assert cfg=={"id":"s0","track":"sgdm_wd0","optimizer":"sgdm","lr":.003,"momentum":.9,"weight_decay":0}
    src=ORIGINAL/"results/grid"/f"s0-{a.seed}"
    manifest_plan=ORIGINAL/"plan.json"
    assert sha(manifest_plan)==P["primary_plan_sha256"]
    summary=json.loads((ORIGINAL/"results/analysis/summary.json").read_text())
    assert summary["complete"] and summary["n_grid"]==24 and summary["n_confirm"]==15 and not summary["audit_errors"]
    selection=ORIGINAL/"results/frozen_selection.json"
    sel=json.loads(selection.read_text())
    assert sel["complete_grid"] and sel["selected"]["sgdm_wd0"]==cfg and not sel["audit_errors"]
    original=json.loads((src/"result.json").read_text());end=original["history"][-1]
    bound=next(x for x in P["source_runs"] if x["seed"]==a.seed)
    assert original["status"]=="complete" and original["phase"]=="grid" and original["config"]==cfg and original["seed"]==a.seed and original["source_commit"]==P["primary_launch_commit"]
    assert original["start_step"]==0 and original["completed_updates"]==original["planned_updates"]==START and end["step"]==START
    assert original["plan_sha256"]==P["primary_plan_sha256"] and original["selection_sha256"] is None and original["dropout"]==0
    assert original["final_resume_sha256"]==bound["final_resume_sha256"]==sha(src/"final_resume.pt")
    assert end["weights_sha256"]==end["spectral_weights_sha256"]==bound["endpoint_weights_sha256"]
    assert end["checkpoint_sha256"]==bound["endpoint_checkpoint_sha256"]==sha(src/end["checkpoint_file"])
    assert all(set(h["metrics"])=={"train","validation"} for h in original["history"])
    folder=D/"results"/f"s0-{a.seed}"
    if folder.exists():raise FileExistsError("Refuse replay/overwrite")
    import numpy as np, torch, weightwatcher as ww
    from torch import nn
    from sklearn.metrics import roc_auc_score, roc_curve
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    mf=ORIGINAL/"prepared/manifest.json";dp=ORIGINAL/"prepared/data.npz";manifest=json.loads(mf.read_text())
    assert manifest["prepared_sha256"]==sha(dp)==P["primary_prepared_sha256"] and manifest["plan_sha256"]==P["primary_plan_sha256"] and original["data_manifest_sha256"]==sha(mf)
    with np.load(dp) as d:
        x,y=d["x"],d["y"];idx=d["train"]
        subsets={"train":idx,"validation":d["val"]}
    mean=x[idx].mean(0);sd=np.maximum(x[idx].std(0),1e-12)
    assert np.array_equal(mean,np.asarray(original["normalization"]["mean"])) and np.array_equal(sd,np.asarray(original["normalization"]["sd"]))
    sets={k:(torch.from_numpy(((x[v]-mean)/sd).astype(np.float32)),torch.from_numpy(y[v].astype(np.int64))) for k,v in subsets.items()}
    seed=a.seed;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    m=nn.Sequential(nn.Linear(10,784),nn.ReLU(),nn.Linear(784,784),nn.ReLU(),nn.Linear(784,2))
    opt=torch.optim.SGD(m.parameters(),lr=cfg["lr"],momentum=cfg["momentum"],weight_decay=cfg["weight_decay"])
    saved=torch.load(src/"final_resume.pt",map_location="cpu",weights_only=True)
    endpoint=torch.load(src/end["checkpoint_file"],map_location="cpu",weights_only=True)
    assert saved["step"]==endpoint["step"]==START and endpoint["weights_sha256"]==end["weights_sha256"]
    assert whash(saved["model"])==whash(endpoint["model"])==end["weights_sha256"]
    m.load_state_dict(saved["model"]);opt.load_state_dict(saved["optimizer"])
    assert len(opt.param_groups)==1 and all(opt.param_groups[0][k]==v for k,v in {"lr":.003,"momentum":.9,"weight_decay":0,"dampening":0,"nesterov":False}.items())
    assert len(opt.state)==len(list(m.parameters())) and all("momentum_buffer" in opt.state[v] and opt.state[v]["momentum_buffer"].shape==v.shape for v in m.parameters())
    gen=torch.Generator();gen.set_state(saved["batch_rng"]);torch.set_rng_state(saved["torch_rng"])
    perm=saved["permutation"];offset=saved["offset"];n=len(idx)
    assert isinstance(offset,int) and 0<=offset<=n and perm.numel()==n and torch.equal(torch.sort(perm).values,torch.arange(n))
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
    # Confirm the archived baseline is reproduced before a single new update.
    initial={k:evaluate(k) for k in sets}
    for k in initial:
        for metric in ("accuracy","ce","auc"):
            assert abs(initial[k][metric]-end["metrics"][k][metric])<=1e-6,(k,metric,initial[k][metric],end["metrics"][k][metric])
    assert whash(m.state_dict())==end["weights_sha256"]
    if not a.execute_training:
        print(json.dumps({"status":"verified_zero_training","seed":seed,"start":START,"cap":CAP,"source_resume_sha256":bound["final_resume_sha256"],"weights_sha256":end["weights_sha256"],"metrics":initial},indent=2));return
    folder.mkdir(parents=True)
    step=START
    report={"phase":"continuation","case":"s0","seed":seed,"config":cfg,"status":"running","start_step":START,"planned_updates":CAP-START,"completed_updates":0,"final_step":START,"dropout":0,"plan_sha256":sha(D/"plan.json"),"data_manifest_sha256":sha(mf),"primary_archive_commit":P["primary_archive_commit"],"source_commit":os.getenv("GITHUB_SHA"),"source_resume_sha256":bound["final_resume_sha256"],"source_endpoint_sha256":bound["endpoint_checkpoint_sha256"],"initial_weights_sha256":end["weights_sha256"],"selection_sha256":sha(selection),"versions":{k:importlib.metadata.version(k) for k in ("torch","numpy","weightwatcher","powerlaw","scikit-learn")},"threads":1,"normalization":{"mean":mean.tolist(),"sd":sd.tolist()},"history":[]}
    xx,yy=sets["train"]
    def observe():
        h=whash(m.state_dict());path=folder/f"checkpoint_{step:05d}.pt";torch.save({"step":step,"model":m.state_dict(),"weights_sha256":h},path)
        metrics={k:evaluate(k) for k in sets}
        spec=end["spectral"] if step==START else spectral();assert whash(m.state_dict())==h
        report["history"].append({"step":step,"weights_sha256":h,"spectral_weights_sha256":h,"checkpoint_file":path.name,"checkpoint_sha256":sha(path),"metrics":metrics,"spectral":spec})
        report["completed_updates"]=step-START;report["final_step"]=step;dump(folder/"result.json",report)
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
        report.update(status="failed",completed_updates=step-START,final_step=step,error=repr(e));dump(folder/"result.json",report);raise
if __name__=="__main__":main()
