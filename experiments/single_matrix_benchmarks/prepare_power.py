"""Validate archived published power-plant split. No training."""
import argparse, hashlib, json, pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
DS=HERE/"power_plant"
PLAN=json.loads((DS/"plan.json").read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blob(p):
    b=p.read_bytes();return hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()
def prepare():
    raw=DS/"source";out=DS/"prepared"
    for name,spec in PLAN["source_files"].items():
        if blob(raw/name)!=spec["git_blob_sha"]:raise ValueError(f"Source Git blob mismatch: {name}")
    d=np.loadtxt(raw/"data.txt",dtype=np.float64)
    f=np.loadtxt(raw/"index_features.txt",dtype=int,ndmin=1)
    t=np.loadtxt(raw/"index_target.txt",dtype=int,ndmin=1)
    outer=np.loadtxt(raw/"index_train_0.txt",dtype=int,ndmin=1)
    test=np.loadtxt(raw/"index_test_0.txt",dtype=int,ndmin=1)
    assert d.shape==(9568,5) and np.isfinite(d).all()
    assert f.tolist()==[0,1,2,3] and t.tolist()==[4]
    assert len(outer)==8611 and len(test)==957
    assert np.array_equal(np.sort(np.r_[outer,test]),np.arange(len(d)))
    rng=np.random.default_rng(PLAN["inner_split_seed"])
    perm=rng.permutation(outer);nval=round(len(outer)*PLAN["inner_validation_fraction"])
    val,train=perm[:nval],perm[nval:]
    assert np.array_equal(np.sort(np.r_[train,val]),np.sort(outer))
    keys={tuple(d[i]) for i in outer};featurekeys={tuple(d[i,f]) for i in outer}
    duplicates=sum(tuple(d[i]) in keys for i in test)
    feature_overlap=sum(tuple(d[i,f]) in featurekeys for i in test)
    assert duplicates==11
    out.mkdir(exist_ok=True)
    if (out/"data.npz").exists() or (out/"manifest.json").exists():raise FileExistsError("Refuse to overwrite prepared archive")
    np.savez_compressed(out/"data.npz",x=d[:,f],y=d[:,t[0]],train=train,val=val,outer_train=outer,test=test)
    manifest={"source_commit":PLAN["source_commit"],"source_files":{n:{"sha256":sha(raw/n),"git_blob_sha":blob(raw/n)} for n in PLAN["source_files"]},
       "prepared_sha256":sha(out/"data.npz"),"n_rows":len(d),"features":len(f),"outer_train":len(outer),"inner_train":len(train),
       "inner_val":len(val),"outer_test":len(test),"outer_test_full_row_overlap":int(duplicates),"outer_test_feature_overlap":int(feature_overlap),
       "unique_full_rows":len(np.unique(d,axis=0)),"outer_fold":0,"inner_split_seed":PLAN["inner_split_seed"],
       "preprocessing":"Fit normalization only on actual training subset per phase, inside trainer."}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":prepare()
