# High-alpha spectral-only sensitivity

This is **not** an execution of WeightWatcher 0.7.7's fix_fingers='clip_xmax' implementation. The pinned Python package was not available in the diagnostic environment. Instead, this independent, deterministic clipping sensitivity uses the full recorded double-precision eigenvalues, no model loading, optimizer updates or test evaluation. It covers every recorded alpha > 6 in the fixed and matched controls, with duplicate step-zero spectra represented by s0 for each architecture and seed. The source commits and paths are in the CSV. The same initial state is duplicated across architectures for transparency. The original WW outputs remain unchanged.

For each spectrum, remove exactly k=0,1,2,3,5 largest eigenvalues (fixed grid, not chosen by alpha), retain positive eigenvalues, and fit an unbounded continuous Pareto density for every observed xmin with at least 20 retained tail eigenvalues. Compute alpha=1+n/sum(log(lambda/xmin)), minimize two-sided empirical CDF KS distance (including both sides of every jump). This KS convention and tail floor can differ from the original WW implementation; the k=0 baseline is therefore separately reported. No p-values or claims of a valid power-law population follow from KS minimization. These are diagnostics of sensitivity to high-end trimming, not replacements or a determination of actual fingers.

Reproduction (JavaScript; pass the sorted archived eigenvalue array):

```js
function fit(e,k,minTail=20){const x=e.slice(0,e.length-k).filter(v=>v>0),ln=x.map(Math.log);let sum=ln.reduce((s,v)=>s+v,0),best=null;for(let i=0;i<=x.length-minTail;i++){const n=x.length-i,den=sum-n*ln[i];if(den>0){const a=1+n/den;let D=0;for(let j=i;j<x.length;j++){const F=-Math.expm1((1-a)*(ln[j]-ln[i]));D=Math.max(D,Math.abs((j-i)/n-F),Math.abs((j-i+1)/n-F))}if(!best||D<best.D)best={alpha:a,D,tail:n,xmin:x[i]}}sum-=ln[i]}return best}
```

[CSV](high_alpha_trim_sensitivity.csv). Every trim is reported; do not select a trim by proximity to 2.
