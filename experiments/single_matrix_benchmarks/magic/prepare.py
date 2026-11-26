"""Official UCI data and a frozen leakage-controlled split. No training."""
import csv, hashlib, io, json, pathlib, urllib.request, zipfile
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
D=pathlib.Path(__file__).resolve().parent
P=json.loads((D/"plan.json").read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def hash_file(p):return sha(p.read_bytes())
def main():
    out=D/"prepared"; src=D/"source"
    if out.exists() or src.exists():raise FileExistsError("Refuse to overwrite prepared/source archive")
    req=urllib.request.Request(P["source"],headers={"User-Agent":"WeightWatcher research reproducibility"})
    with urllib.request.urlopen(req,timeout=90) as r:raw=r.read(10_000_001)
    if len(raw)>10_000_000:raise ValueError("Unexpected archive size")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=[n for n in z.namelist() if pathlib.PurePosixPath(n).name=="magic04.data"]
        if len(names)!=1:raise ValueError("Expected exactly one data file")
        data=z.read(names[0])
        ns=[n for n in z.namelist() if pathlib.PurePosixPath(n).name=="magic04.names"]
        docs=z.read(ns[0]) if len(ns)==1 else None
    rows=list(csv.reader(io.StringIO(data.decode("ascii"))))
    rows=[r for r in rows if r]
    assert len(rows)==19020 and all(len(r)==11 for r in rows)
    x=np.array([r[:10] for r in rows],dtype=np.float64)
    y=np.array([P["labels"][r[10].strip()] for r in rows],dtype=np.int64)
    assert x.shape==(19020,10) and np.isfinite(x).all()
    assert np.bincount(y).tolist()==[6688,12332]
    _,groups=np.unique(x,axis=0,return_inverse=True)
    folds=[b for _,b in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=P["split"]["seed"]).split(x,y,groups)]
    test,val=folds[0],folds[1]
    train=np.sort(np.concatenate(folds[2:]))
    test,val=np.sort(test),np.sort(val)
    outer=np.sort(np.r_[train,val])
    assert np.array_equal(np.sort(np.r_[train,val,test]),np.arange(len(y)))
    g=[set(groups[i]) for i in (train,val,test)]
    assert not (g[0]&g[1] or g[0]&g[2] or g[1]&g[2])
    src.mkdir();out.mkdir()
    (src/"magic04.data").write_bytes(data)
    (src/"uci_source.zip").write_bytes(raw)
    if docs is not None:(src/"magic04.names").write_bytes(docs)
    np.savez_compressed(out/"data.npz",x=x,y=y,train=train,val=val,outer_train=outer,test=test)
    manifest={"source_url":P["source"],"doi":P["doi"],"license":P["license"],"source_sha256":sha(data),"zip_sha256":sha(raw),"prepared_sha256":hash_file(out/"data.npz"),"plan_sha256":hash_file(D/"plan.json"),"n_rows":len(y),"features":10,"classes":np.bincount(y).tolist(),"unique_feature_rows":int(len(np.unique(groups))),"cross_partition_duplicate_feature_groups":0,"split":P["split"],"counts":{k:{"n":len(v),"classes":np.bincount(y[v],minlength=2).tolist(),"index_sha256":sha(v.astype("<i8").tobytes())} for k,v in {"train":train,"val":val,"test":test,"outer_train":outer}.items()}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
