# Experiment 02: Fashion-MNIST magnitude-only graft

**Running since 2026-11-15 09:57:34 UTC. Final results are not yet available.**

This is the authorized Fashion-MNIST replication of the MNIST experiment in the sibling folder. ReLU MLP 784→256→256→10; 120 epochs; five paired seeds 31001–31005; eight settings; 120 runs. Native AdamW and SGDM are the controls. The magnitude-only graft retains Adam direction, moments, bias correction and bias updates, replacing only each matrix update's Frobenius magnitude with the SGDM target from that run's own gradient history.

The planned settings are batch 128 at LR ×0.5/×1 and batches 256/512 at LR ×0.5/×1/×2. The earlier MNIST failure setting, batch128/LR×2, was excluded prospectively for all arms. No other settings, datasets or architectures are authorized in this pass.

Training cutoff: 14:25:33 UTC. Total experiment deadline: 14:55:33 UTC. Six workers execute one fixed pass. Completed seed blocks are preserved incrementally. No duplicate or follow-up run is active.

[Execution protocol](verification/Protocol.json) · [Launch record](verification/Launch.json) · [Preflight checks](verification/Preflight_Audit.json) · [Data provenance](verification/Data_Provenance.json) · [Saved artifacts](ARTIFACTS.md) · [Historical setup plan](protocol/Protocol_Plan.json)

The training kernel remains byte-identical to MNIST (SHA256 `1bfd7032e00ad87d729670197925fb40e19bc802a8eb716db2571e56b538d3d8`). The separate adapter selects Fashion-MNIST data and the approved 120-run matrix and enforces the deadline. [Metadata clarification](verification/Execution_Metadata_Clarification.json) distinguishes historical setup-only fields from live execution records.

Independent PyTorch update checks, shared initialization and minibatch checks, and preflight checkpoint replay passed. One finished AdamW and one finished graft also passed restored-logit, spectral and eleven-step replay checks. This is partial verification, not a final result.

The final report will retain all outcomes, flag collapsed/nonfinite/unfinished models, and report alpha movement, distance to2 and test accuracy separately. Five seeds—not eight settings—are the independent replicates. Hidden FC1/FC2 measurements are primary; the ten-eigenvalue output layer is exploratory. No alpha-targeting fit or post-hoc test-selected checkpoint is used.

Results, plots, checkpoint links and verification will appear in this folder. The complete MNIST sibling remains unchanged.
