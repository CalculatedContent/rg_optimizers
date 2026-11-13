# Experiment 01: MNIST magnitude-only AdamW replacement

**Partial results: 14 of 135 runs complete and verified, all seed 31001. No training is active.** The remaining 121 runs are unstarted. This is not a completed five-seed comparison, and no across-seed confidence intervals can yet be reported.

## Design

MLP 784→256→256→10 with ReLU; 120 epochs; five paired seeds 31001–31005; batches 128/256/512; LR multipliers 0.5/1/2; native AdamW, magnitude graft, and SGDM. Adam peak layer rates are [0.001,0.001,0.001]; SGDM target rates are [0.04,0.2,0.8]. All parameters receive common decay per processed example. See the exact [frozen protocol](frozen/Protocol.json).

For an Adam update A and first moment m with beta1=0.9, the graft applies `D = norm(-eta_S*m/(1-beta1))*A/norm(A)` per matrix. It keeps Adam's direction, moments and bias updates. The target momentum is taken from the graft's own history, not a separate SGDM run.

## Preliminary findings

Five matched AdamW/graft settings are available. FC2 moved toward 2 in all five; FC1 increased in four but sometimes overshot 2. Mean hidden-layer absolute distance to 2 improved in four settings and worsened at batch 512. Test accuracy improved slightly in four settings (0.02–0.11 percentage points), and test cross-entropy improved in all five. SGDM was also setting-dependent; the strongest observed accuracy was 98.58% at batch128 / LR multiplier0.5, with alpha values 2.072 and 1.918.

These are one-seed observations, not confirmation of a universal optimizer advantage or an RG mechanism. See [all 14 endpoints](results/endpoints.csv), [full per-run results](results/completed_runs.json) and the [135-run status register](results/status.json).

## Verification

All 14 completed model hashes match; saved layer alphas reproduce; one recorded update per run replays exactly from an epoch30 checkpoint. [Verification evidence](verification/Completed_Run_Verification.json). The preflight also independently checks AdamW/SGDM updates, graft magnitude/direction, deterministic replay and scalar-dilation invariance of the fitted exponent.

## Source and reproduction

`frozen/study.py` is the exact historical training source (SHA256 `1bfd7032e00ad87d729670197925fb40e19bc802a8eb716db2571e56b538d3d8`). Do not edit it for a continuation. The source and analysis scripts retain their original runtime paths and are preserved as provenance, not advertised as a turnkey fresh-clone launcher. `analyze.py` and `verify.py` require the completed 135-run grid; they must not be used to label these 14 runs a complete experiment.

For exact continuation, restore the linked archive into an isolated working directory, retain its `study.py`, `Protocol.json`, `data/` and `runs/` layout, and use the versions recorded in the protocol. Existing `done.json` runs must be skipped; resume unfinished runs from saved optimizer/model/RNG states. Verify the source and data hashes before execution. A portable launcher and reliable long-lived executor still need to be configured before further training; no new training is launched by this repository addition.

[Checkpoint and data archive](ARTIFACTS.md). Subsequent daily experiments belong in sibling dated folders under `experiments/adamw_modified/`.
