"""Reproducible, speaker-disjoint mini Speech Commands features."""
import argparse, csv, hashlib, io, json, pathlib, urllib.request, zipfile
import numpy as np
from scipy.io import wavfile

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE/"results"/"data"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def speaker_split(s):
    h=int(hashlib.sha256(("mini-speech-commands-v1:"+s).encode()).hexdigest()[:8],16)%1000
    return "train" if h<800 else "validation" if h<900 else "test"
def mel(f): return 2595*np.log10(1+f/700)
def hz(m): return 700*(10**(m/2595)-1)
freq=np.fft.rfftfreq(512,1/16000)
edges=hz(np.linspace(mel(80),mel(7600),34))
bank=np.stack([np.maximum(0,np.minimum((freq-l)/(c-l),(r-freq)/(r-c))) for l,c,r in zip(edges[:-2],edges[1:-1],edges[2:])])
window=np.hanning(400)
def features(raw):
    sr,x=wavfile.read(io.BytesIO(raw))
    if sr!=16000 or x.ndim!=1 or x.dtype!=np.int16: raise ValueError(f"Unexpected audio {sr} {x.shape} {x.dtype}")
    y=np.zeros(16000,dtype=np.float64)
    n=min(len(x),16000);y[:n]=x[:n]/32768.
    frames=np.lib.stride_tricks.sliding_window_view(y,400)[::160]*window
    power=np.abs(np.fft.rfft(frames,n=512,axis=1))**2
    z=np.log(power@bank.T+1e-6)
    return np.stack([t.mean(axis=0) for t in np.array_split(z,32,axis=0)]).astype(np.float32).ravel()
def main():
    plan=json.loads((HERE/"plan.json").read_text()); spec=plan["dataset"]
    ROOT.mkdir(parents=True,exist_ok=True)
    prepared=ROOT/"prepared.npz"; manifest=ROOT/"manifest.json"
    if prepared.exists() or manifest.exists():
        raise FileExistsError("Restore committed prepared data; refuse overwrite")
    tmp=ROOT/"mini_speech_commands.zip"
    with urllib.request.urlopen(spec["url"],timeout=90) as src,tmp.open("wb") as dst:
        while block:=src.read(1024*1024): dst.write(block)
    archive_hash=sha(tmp); labels=spec["classes"]; rows=[]; xs=[]; ys=[]
    with zipfile.ZipFile(tmp) as z:
        names=sorted(n for n in z.namelist() if n.endswith(".wav") and pathlib.PurePosixPath(n).parent.name in labels and not pathlib.PurePosixPath(n).name.startswith("._"))
        if len(names)!=spec["expected_clips"]: raise ValueError(f"Expected 8000 clips, got {len(names)}")
        for name in names:
            path=pathlib.PurePosixPath(name); speaker=path.stem.split("_nohash_")[0]
            if "_nohash_" not in path.stem: raise ValueError(f"Unexpected speaker id {path.name}")
            raw=z.read(name); split=speaker_split(speaker); label=labels.index(path.parent.name)
            xs.append(features(raw)); ys.append(label)
            rows.append(dict(path=name,speaker=speaker,split=split,label=label,wav_sha256=hashlib.sha256(raw).hexdigest()))
    x=np.stack(xs); y=np.asarray(ys,dtype=np.int64); split=np.asarray([r["split"] for r in rows])
    train=x[split=="train"]
    mean=train.mean(axis=0,dtype=np.float64); sd=np.maximum(train.std(axis=0,dtype=np.float64),1e-5)
    x=((x-mean)/sd).astype(np.float32)
    if not np.isfinite(x).all() or x.shape!=(8000,1024): raise ValueError("Invalid features")
    arrays={}; counts={}; speakers={}
    for s in ("train","validation","test"):
        idx=split==s; arrays["x_"+s]=x[idx]; arrays["y_"+s]=y[idx]
        counts[s]=np.bincount(y[idx],minlength=8).tolist()
        if min(counts[s])<=0: raise ValueError(f"Missing class in {s}")
        speakers[s]=set(r["speaker"] for r in rows if r["split"]==s)
    if any(speakers[a]&speakers[b] for a,b in [("train","validation"),("train","test"),("validation","test")]): raise RuntimeError("Speaker leakage")
    np.savez_compressed(prepared,**arrays,feature_mean=mean,feature_std=sd)
    with (ROOT/"split.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(source=spec["url"],license="CC BY; Google Speech Commands, Pete Warden (2018)",archive_sha256=archive_hash,
        archive_bytes=tmp.stat().st_size,prepared_sha256=sha(prepared),split_csv_sha256=sha(ROOT/"split.csv"),
        labels=labels,counts_by_class=counts,speaker_counts={k:len(v) for k,v in speakers.items()},
        features=spec["feature"],split_rule=spec["split"],preprocessor_sha256=sha(pathlib.Path(__file__)),n_clips=len(rows))
    manifest.write_text(json.dumps(report,indent=2)+"\n");tmp.unlink()
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
