"""Zero-training verification of exact archived inputs and reusable runs."""
import csv, hashlib, json, pathlib, zipfile
import numpy as np
D=pathlib.Path(__file__).resolve().parent
O=D.parent/"magic"
P=json.loads((D/"plan.json").read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check():
    m=O/"prepared/manifest.json"; M=json.loads(m.read_text()); dp=O/"prepared/data.npz"
    assert sha(dp)==M["prepared_sha256"]==P["primary_prepared_sha256"]
    assert sha(O/"plan.json")==M["plan_sha256"]==P["primary_plan_sha256"]
    assert sha(m)=="26cc2d30485301f72087e88a59c0b044f374b4987abe1e176a308a32e2cc2473"
    src=O/"source/magic04.data"; z=O/"source/uci_source.zip"
    assert sha(src)==M["source_sha256"] and sha(z)==M["zip_sha256"]
    with zipfile.ZipFile(z) as f:
        names=[n for n in f.namelist() if pathlib.PurePosixPath(n).name=="magic04.data"]
        assert len(names)==1 and f.read(names[0])==src.read_bytes()
    with np.load(dp,allow_pickle=False) as d:
        x,y=d["x"],d["y"]; tr,va,te,ot=(d[k] for k in ("train","val","test","outer_train"))
        rows=list(csv.reader(src.read_text().splitlines())); rows=[r for r in rows if r]
        assert x.shape==(19020,10) and np.isfinite(x).all() and np.bincount(y).tolist()==[6688,12332]
        assert np.array_equal(x,np.array([r[:10] for r in rows],dtype=np.float64)) and np.array_equal(y,[P["labels"][r[10].strip()] for r in rows])
        assert np.array_equal(np.sort(np.r_[tr,va,te]),np.arange(19020)) and np.array_equal(np.sort(np.r_[tr,va]),ot)
        _,g=np.unique(x,axis=0,return_inverse=True); a,b,c=(set(g[i]) for i in (tr,va,te))
        assert not (a&b or a&c or b&c)
        for k,i in zip(("train","val","test","outer_train"),(tr,va,te,ot)):
            q=M["counts"][k]
            assert len(i)==q["n"] and hashlib.sha256(i.astype("<i8").tobytes()).hexdigest()==q["index_sha256"]
            assert np.bincount(y[i],minlength=2).tolist()==q["classes"]
    assert P["cap"]==20000 and P["dev_seeds"]==[3470] and P["confirmation_seeds"]==list(range(3472,3477))
    assert len(P["grid"])==16 and {(q["lr"],q["weight_decay"]) for q in P["grid"]}=={(lr,wd) for lr in [.003,.006,.012,.024] for wd in [0,.0001,.001,.01]}
    assert all(q["optimizer"]=="sgdm" and q["momentum"]==.9 for q in P["grid"])
    assert P["batch_size"]==256 and P["clip"]==1 and P["dropout"]==0 and P["architecture"]==[10,784,784,2]
    verified=[]
    for phase,case,seed in [("grid","s0",3470)]+[("confirm","sgdm_wd0",s) for s in P["confirmation_seeds"]]:
        folder=O/"results"/phase/f"{case}-{seed}"
        f=folder/"result.json"; r=json.loads(f.read_text())
        assert r["phase"]==phase and r["case"]==case and r["seed"]==seed
        assert r["status"]=="complete" and r["completed_updates"]==20000 and r["start_step"]==0
        assert r["plan_sha256"]==P["primary_plan_sha256"] and r["data_manifest_sha256"]==sha(m)
        assert r["config"]["lr"]==.003 and r["config"]["momentum"]==.9 and r["config"]["weight_decay"]==0
        assert [h["step"] for h in r["history"]]==P["checkpoints"]
        for h in r["history"]:
            assert sha(folder/h["checkpoint_file"])==h["checkpoint_sha256"] and h["weights_sha256"]==h["spectral_weights_sha256"]
        assert sha(folder/"final_resume.pt")==r["final_resume_sha256"]
        verified.append({"phase":phase,"case":case,"seed":seed,"result_sha256":sha(f)})
    return {"status":"verified_zero_training","plan_sha256":sha(D/"plan.json"),"data_sha256":sha(dp),"manifest_sha256":sha(m),"reusable":verified,"new_grid":15,"max_new_confirmation":5,"max_new_updates":400000}
if __name__=="__main__":
    print(json.dumps(check(),indent=2))
