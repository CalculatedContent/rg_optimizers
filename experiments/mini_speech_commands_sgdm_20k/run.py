"""SGDM only. Fixed 20k cap. Each measurement is bound to saved weights."""
import argparse, contextlib, hashlib, importlib.metadata, io, json, logging, math, os, pathlib, platform, random, statistics, time
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import weightwatcher as ww

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE/"results"
CAP=20000
PLAN=json.loads((HERE/"plan.json").read_text())
NAMES=("fc1","fc2","fc3","fc4")
logging.getLogger("weightwatcher").setLevel(logging.ERROR)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def atomic_json(path,value):
    tmp=path.with_name(path.name+".tmp");tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");os.replace(tmp,path)
def atomic_torch(path,value):
    tmp=path.with_name(path.name+".tmp");torch.save(value,tmp);os.replace(tmp,path)
def model(p):
    return nn.Sequential(OrderedDict([
        ("fc1",nn.Linear(1024,128)),("relu1",nn.ReLU()),("drop1",nn.Dropout(p)),
        ("fc2",nn.Linear(128,128)),("relu2",nn.ReLU()),("drop2",nn.Dropout(p)),
        ("fc3",nn.Linear(128,128)),("relu3",nn.ReLU()),("drop3",nn.Dropout(p)),
        ("fc4",nn.Linear(128,8))]))
def fit(w):
    # Direct shipped WW call, including its small-N route for FC4.
    try:
        m=nn.Sequential(nn.Linear(w.shape[1],w.shape[0],bias=False))
        with torch.no_grad(): m[0].weight.copy_(torch.as_tensor(w,dtype=torch.float32))
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            df=ww.WeightWatcher(m).analyze(min_evals=8,randomize=False,plot=False)
        if len(df)!=1: raise ValueError(f"Expected one WW fit, got {len(df)}")
        row=json.loads(df.to_json(orient="records"))[0]
        a,d,n=row.get("alpha"),row.get("D"),row.get("num_pl_spikes")
        good=(row.get("status")=="success" and isinstance(a,(float,int)) and math.isfinite(a) and a>1
              and isinstance(d,(float,int)) and math.isfinite(d) and 0<=d<=1
              and isinstance(n,(float,int)) and 2<=n<=min(w.shape))
        return dict(alpha=float(a) if good else None,D=float(d) if good else None,tail=int(n) if good else None,
            status="success" if good else "failed",ww_status=row.get("status"),warning=row.get("warning"),
            xmin=row.get("xmin"),matrix_rank=row.get("matrix_rank"),num_evals=row.get("num_evals"),sigma=row.get("sigma"))
    except Exception as e: return dict(alpha=None,D=None,tail=None,status="failed",error=repr(e))
def aggregate(fits):
    if not all(f["status"]=="success" for f in fits): return None
    weights=[math.sqrt(f["tail"])/(f["D"]+0.02) for f in fits]
    return sum(f["alpha"]*w for f,w in zip(fits,weights))/sum(weights)
def mean_complete(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None
def spectra(m,p,seed,step):
    torch_state=torch.get_rng_state(); np_state=np.random.get_state(); py_state=random.getstate()
    try:
        raw=[];masked=[];layers=[]; reps=PLAN["mask_reps"] if p>0 else 1
        for li,name in enumerate(NAMES):
            w=getattr(m,name).weight.detach().cpu().numpy().copy()
            f=fit(w);raw.append(f)
            # FC1 receives no dropout; FC2-FC4 receive one activation-dropout input.
            r=0 if li==0 else 1-math.sqrt(1-p)
            fs=[]
            for k in range(reps):
                if r==0: f2=dict(f)
                else:
                    rng=np.random.default_rng(np.random.SeedSequence([20261126,seed,step,li,k]))
                    f2=fit(w*(rng.random(w.shape)>=r))
                fs.append(f2)
            masked.append(fs)
            vals=[f2["alpha"] for f2 in fs]
            avg=mean_complete(vals)
            layers.append(dict(layer=name,shape=list(w.shape),input_dropout=0 if li==0 else p,mask_probability=r,
                raw=f,corrected_alpha=avg,corrected_alpha_sd=statistics.stdev(vals) if avg is not None and len(vals)>1 else (0. if avg is not None else None),
                corrected_fits=fs))
        robust=[aggregate([masked[i][k] for i in range(3)]) for k in range(reps)]
        all_four=[aggregate([masked[i][k] for i in range(4)]) for k in range(reps)]
        return dict(raw_alpha=aggregate(raw[:3]),raw_alpha_all_four=aggregate(raw),
            corrected_alpha=mean_complete(robust),corrected_alpha_all_four=mean_complete(all_four),
            corrected_aggregate_replicates=robust,corrected_all_four_replicates=all_four,layers=layers,
            primary_layers=list(NAMES[:3]),mask_reps=reps,method="direct WeightWatcher 0.7.7; ordinary alpha, sqrt(tail)/(D+0.02); no size factor")
    finally:
        torch.set_rng_state(torch_state);np.random.set_state(np_state);random.setstate(py_state)
def evaluate(m,x,y):
    old=m.training;m.eval();loss=0.;right=0
    with torch.no_grad():
        for a,b in zip(x.split(512),y.split(512)):
            z=m(a);loss+=F.cross_entropy(z,b,reduction="sum").item();right+=(z.argmax(1)==b).sum().item()
    m.train(old)
    return dict(loss=loss/len(y),accuracy=right/len(y),n=len(y))
def verify_plan():
    if (PLAN["optimizer"]!="sgdm" or PLAN["max_updates"]!=CAP or PLAN["checkpoints"]!=[0,2000,5000,10000,15000,CAP]
        or PLAN["dampening"]!=0 or PLAN["nesterov"] is not False or PLAN["mask_reps"]!=20
        or any(c["weight_decay"]!=0 for c in PLAN["cases"])): raise ValueError("Unexpected plan or cap")
def smoke():
    verify_plan()
    torch.manual_seed(123)
    m=model(PLAN["dropout_rates"][1])
    before=state_hash(m.state_dict());rng=torch.get_rng_state().clone()
    s=spectra(m,0.,123,0)
    assert state_hash(m.state_dict())==before and torch.equal(rng,torch.get_rng_state())
    assert [r["shape"] for r in s["layers"]]==[[128,1024],[128,128],[128,128],[8,128]]
    assert abs((1-math.sqrt(1-PLAN["dropout_rates"][1]))-0.20564952818645776)<1e-12
    # A known nondegenerate eight-value diagonal spectrum confirms small-N routing.
    w=np.diag(np.geomspace(1,16,8)).astype(np.float32)
    f=fit(w);assert f["status"]=="success",f
    assert f["num_evals"]==8 and f["tail"]==8,f
    assert m(torch.zeros(2,1024)).shape==(2,8)
    print("Smoke passed; no optimizer updates executed",flush=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true");parser.add_argument("--phase",choices=["grid","confirm"])
    parser.add_argument("--case");parser.add_argument("--seed",type=int);parser.add_argument("--dropout-index",type=int)
    a=parser.parse_args();verify_plan()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if a.smoke: smoke();return
    if a.phase=="grid":
        if a.seed not in PLAN["development_seeds"]: raise ValueError("Invalid grid seed")
        matches=[c for c in PLAN["cases"] if c["id"]==a.case]
        if len(matches)!=1: raise ValueError("Invalid case")
        cfg=matches[0]; job=f"grid-{cfg['id']}-{a.seed}"; selected_sha=None
    elif a.phase=="confirm":
        if a.seed not in PLAN["confirmation_seeds"] or a.dropout_index not in (0,1): raise ValueError("Invalid confirmation seed/rate")
        sp=ROOT/"selection"/"frozen_selection.json";sel=json.loads(sp.read_text())
        if sel["complete_grid"] is not True or sel["plan_sha256"]!=sha(HERE/"plan.json") or sel["audit_errors"]!=[]: raise ValueError("Selection not complete")
        cfg=next(c for c in PLAN["cases"] if c["lr"]==sel["selected"]["lr"] and c["momentum"]==sel["selected"]["momentum"] and c["dropout_index"]==a.dropout_index)
        job=f"confirm-p{a.dropout_index}-{a.seed}";selected_sha=sha(sp)
    else: raise ValueError("Specify phase or smoke")
    folder=ROOT/a.phase/job
    if folder.exists(): raise FileExistsError("No resume/restart/overwrite")
    folder.mkdir(parents=True)
    manifest=ROOT/"data"/"manifest.json"; dm=json.loads(manifest.read_text()); data=ROOT/"data"/"prepared.npz"
    if sha(data)!=dm["prepared_sha256"] or sha(ROOT/"data"/"split.csv")!=dm["split_csv_sha256"]: raise ValueError("Prepared data identity differs")
    with np.load(data) as z:
        sets={s:(torch.from_numpy(z["x_"+s]),torch.from_numpy(z["y_"+s])) for s in ("train","validation","test") if s!="test" or a.phase=="confirm"}
    seed=a.seed;p=cfg["dropout"]
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    m=model(p);opt=torch.optim.SGD(m.parameters(),lr=cfg["lr"],momentum=cfg["momentum"],weight_decay=0,dampening=0,nesterov=False)
    gen=torch.Generator().manual_seed(seed)
    report=dict(schema_version=1,job=job,phase=a.phase,seed=seed,config=cfg,planned_updates=CAP,completed_updates=0,status="running",
        plan_sha256=sha(HERE/"plan.json"),data_manifest_sha256=sha(manifest),prepared_sha256=sha(data),selection_sha256=selected_sha,
        initial_weights_sha256=state_hash(m.state_dict()),source_commit=os.getenv("GITHUB_SHA"),run_id=os.getenv("GITHUB_RUN_ID"),
        versions={k:importlib.metadata.version(k) for k in ("torch","numpy","scipy","pandas","weightwatcher","powerlaw")},
        cpu=next((v.split(":",1)[1].strip() for v in pathlib.Path("/proc/cpuinfo").read_text().splitlines() if v.startswith("model name")),platform.processor()) if pathlib.Path("/proc/cpuinfo").exists() else platform.processor(),
        platform=platform.platform(),python=platform.python_version(),threads=torch.get_num_threads(),parameter_count=sum(x.numel() for x in m.parameters()),history=[])
    started=time.perf_counter();x,y=sets["train"];perm=None;offset=len(y);step=0
    def checkpoint():
        h=state_hash(m.state_dict());path=folder/f"checkpoint_step_{step}.pt"
        atomic_torch(path,dict(step=step,model=m.state_dict(),weights_sha256=h))
        metrics={s:evaluate(m,*data) for s,data in sets.items()}
        spectral=spectra(m,p,seed,step)
        if state_hash(m.state_dict())!=h: raise RuntimeError("Diagnostic changed weights")
        row=dict(step=step,weights_sha256=h,spectral_weights_sha256=h,checkpoint_file=path.name,checkpoint_sha256=sha(path),
            metrics=metrics,spectral=spectral,elapsed_seconds=time.perf_counter()-started)
        report["history"].append(row);report["completed_updates"]=step
        atomic_json(folder/"result.json",report)
        with (folder/"history.jsonl").open("a") as f: f.write(json.dumps(row,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(step=step,validation=metrics["validation"],raw=spectral["raw_alpha"],corrected=spectral["corrected_alpha"])),flush=True)
    try:
        checkpoint();m.train()
        while step<CAP:
            if offset==len(y): perm=torch.randperm(len(y),generator=gen);offset=0
            idx=perm[offset:offset+PLAN["batch_size"]];offset+=len(idx)
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x[idx]),y[idx])
            if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite loss before update {step+1}")
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),PLAN["grad_clip"],error_if_nonfinite=True)
            opt.step();step+=1
            if step in PLAN["checkpoints"]: checkpoint()
        if step!=CAP or report["history"][-1]["step"]!=CAP: raise RuntimeError("Cap mismatch")
        atomic_torch(folder/"resume_final.pt",dict(step=step,model=m.state_dict(),optimizer=opt.state_dict(),
            batch_generator=gen.get_state(),torch_rng=torch.get_rng_state(),permutation=perm,offset=offset))
        report["final_resume_sha256"]=sha(folder/"resume_final.pt")
        report["status"]="complete";report["completed_updates"]=step
        atomic_json(folder/"result.json",report)
    except Exception as e:
        report["status"]="failed";report["completed_updates"]=step;report["error"]=repr(e)
        atomic_json(folder/"result.json",report);raise
if __name__=="__main__":main()
