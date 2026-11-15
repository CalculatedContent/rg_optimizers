# Experiment02: Fashion-MNIST — recovery running

**Authorized recovery is running: preserve56 completed runs, resume6 saved states and complete58 unstarted runs. Final120-run verification remains pending.**

[Live recovery](https://github.com/CalculatedContent/rg_optimizers/actions/runs/43706454292) · [Recovery protocol](recovery/README.md)

Charles explicitly requested this continuation before GPT on15Nov2026. GPT is cancelled/on hold. The recovery source and optimization settings are unchanged.

The last supervisor update was approximately2026-11-15 10:22UTC. Its execution session subsequently became inaccessible. No completion or final verification summary was produced. The original execution was interrupted; the separately documented authorized recovery now continues from saved states.

Two complete seed blocks (31001 and31002) and all available seed31003 checkpoints/logs have been preserved. Execution completion does not mean every model has passed the final functional/spectral audit. No five-seed effect, confidence interval, or theoretical conclusion is claimed from this partial pass.

[Actual status](results/STATUS.json) · [Preserved checkpoints/data](ARTIFACTS.md) · [Frozen protocol](verification/Protocol.json) · [Preflight](verification/Preflight_Audit.json) · [Historical setup plan](protocol/Protocol_Plan.json)

The frozen design was Fashion-MNIST, ReLU MLP784→256→256→10,120epochs, seeds31001–31005, eight settings and three arms: native AdamW, exact magnitude-only graft and SGDM. The graft retains Adam direction/moments/bias updates and changes only matrix magnitude using momentum from its own gradient history. Batch128/LR×2 was excluded prospectively for all arms based on the prior MNIST failure.

The original kernel SHA256 remains `1bfd7032e00ad87d729670197925fb40e19bc802a8eb716db2571e56b538d3d8`. Preflight and initial saved AdamW/graft smoke checks passed, but these do not replace full-pass verification. No failures or incomplete runs are hidden. The MNIST sibling is unchanged.

The separately authorized [GPT Experiment03](../2026-11-15_tiny_shakespeare_gpt_experiment03/) uses GitHub-hosted execution with a gated three-arm pilot. It does not resume or replace these Fashion-MNIST observations.
