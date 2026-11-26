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
    assert P["cap"]==20000 and P["dev_seeds"]==[3470,3471] and P["confirmation_seeds"]==list(range(3472,3477))
    assert len(P["grid"])==7 and all(q["weight_decay"]==0 for q in P["grid"])
    assert {q["lr"] for q in P["grid"] if q["optimizer"]=="sgdm"}=={.003,.01,.03,.1}
    assert all(q["momentum"]==.9 for q in P["grid"] if q["optimizer"]=="sgdm")
    assert {q["lr"] for q in P["grid"] if q["optimizer"]=="adamw"}=={.0001,.0003,.001}
    assert P["batch_size"]==256 and P["clip"]==1 and P["dropout"]==0 and P["architecture"]==[10,784,784,2]
    import torch, math
    from torch import nn
    from model import build
    torch.set_num_threads(1)
    measurements={}
    for dtype in (torch.float32,torch.float64):
        old=torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            torch.manual_seed(3470)
            ordinary=nn.Sequential(nn.Linear(10,784),nn.ReLU(),nn.Linear(784,784),nn.ReLU(),nn.Linear(784,2))
            torch.manual_seed(3470)
            net=build()
            x=torch.randn(32,10); y=torch.randint(0,2,(32,))
            atol,rtol=(2e-6,2e-5) if dtype==torch.float32 else (1e-12,1e-11)
            initial_error=0.
            for i in (0,2,4):
                assert torch.equal(net[i].weight,ordinary[i].weight) and torch.equal(net[i].bias,ordinary[i].bias)
                ew=net[i].effective_weight()
                torch.testing.assert_close(ew,ordinary[i].weight,rtol=rtol,atol=atol)
                initial_error=max(initial_error,float((ew-ordinary[i].weight).detach().abs().max()))
            logits=net(x).detach(); target=nn.functional.cross_entropy(logits,y)
            torch.testing.assert_close(logits,ordinary(x),rtol=rtol,atol=atol)
            max_logit=max_loss=max_residual=0.
            for i in (0,2,4):
                layer=net[i]
                for b in (.125,.3,2.,7.):
                    with torch.no_grad():
                        saved=layer.weight.clone();layer.weight.mul_(b)
                        z=net(x); loss=nn.functional.cross_entropy(z,y)
                        torch.testing.assert_close(z,logits,rtol=rtol,atol=atol)
                        torch.testing.assert_close(loss,target,rtol=rtol,atol=atol)
                        max_logit=max(max_logit,float((z-logits).abs().max()))
                        max_loss=max(max_loss,float((loss-target).abs()))
                        layer.weight.copy_(saved)
            nn.functional.cross_entropy(net(x),y).backward()
            for i in (0,2,4):
                w=net[i].weight.detach().double();g=net[i].weight.grad.detach().double()
                r=abs(float((w*g).sum()))/float(w.norm()*g.norm())
                assert math.isfinite(r) and r<(2e-6 if dtype==torch.float32 else 1e-11)
                max_residual=max(max_residual,r)
            measurements[str(dtype)]={"max_weight_initial_error":initial_error,"max_logit_rescale_error":max_logit,"max_loss_rescale_error":max_loss,"max_radial_residual":max_residual}
        finally:torch.set_default_dtype(old)
    return {"status":"verified_zero_training","plan_sha256":sha(D/"plan.json"),"data_sha256":sha(dp),"manifest_sha256":sha(m),"symmetry":measurements,"new_grid":14,"new_confirmation":10,"max_new_updates":480000}
if __name__=="__main__":
    print(json.dumps(check(),indent=2))
