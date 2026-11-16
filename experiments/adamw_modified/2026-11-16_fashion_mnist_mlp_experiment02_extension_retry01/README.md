# Verified Fashion-MNIST continuation retry01

Retry the same120 epoch20-to60 continuations after the previous attempt failed before training. The parent epoch20 checkpoints and failed attempt remain unchanged. No new model, dataset, seed or setting is introduced. GPT stays on hold.

All24 models for seed31001 passed a local full restoration/multiprocessing audit: 48 recorded minibatch losses and update-statistic arrays matched exactly. The independent15-matrix-step resumed graft contract passed. Nine deliberately incorrect minibatch/momentum/learning-rate controls were rejected. Local verification uses Python3.12 and pinned Torch2.15CPU; GitHub uses Python3.11 and the same numerical packages.

The parent job used Azure westus2 and the failed retry used AWS us-west-2, with different runner image builds. Platform-dependent floating-point behavior is a plausible explanation, not an established root cause: the failed job omitted the actual loss difference. [PyTorch numerical accuracy documentation](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) does not guarantee bitwise equality across platforms.

## Verification change

Before any new training, all120 parent states must pass hash, model/optimizer/RNG, saved hyperparameter, spectrum and loss checks. The old-machine replay logs both losses and every statistic, admitting only loss absolute error<=1e-6 and statistic error<=1e-7+1e-5*abs(reference). NaN locations must agree. These limits are fixed before the retry. The result separately reports exact historical matches. Two independent restorations on the current runtime must produce bitwise-identical losses, statistics, model and optimizer states. All9 negative controls must fail. No numerical thresholds can be relaxed automatically.

These checks are published as an early preflight artifact. Only then can the fixed continuation begin. The three central continuations verify exact epoch21-to22 replay and count once in120. No outcome-based gate or checkpoint selection.

## Training and inference

The frozen optimizer kernel and continuation training functions are unchanged. Same five seeds31001–31005, eight settings and three arms. AdamW direction/moments/bias updates remain intact in the graft, with matrix Frobenius magnitudes taken from SGDM history on its own trajectory. Resume epoch20 through fixed60; LR factor falls smoothly from.1 to.01 without restarting warmup. Validation everyepoch, WW every5, test only60. Four-hour training/five-hour workflow cap.

Report all outcomes, functional failures and unsupported fits. Five seeds are replicates; average eight settings within seed for paired95%t intervals. Report alpha separately from|alpha−2| and accuracy separately from CE. This is a follow-up chosen after the epoch20 results, not a fresh independent test. FC3 is exploratory because its spectrum has only10 eigenvalues.

[Protocol](Protocol.json) · [Local verification](verification/Retry_Local_Verification.json) · [Negative controls](verification/Local_Replay_Negative_Controls.json) · [Previous failed attempt](../2026-11-16_fashion_mnist_mlp_experiment02_extension/RESULTS.md) · [Epoch20 results](../2026-11-15_fashion_mnist_mlp_experiment02_clean/RESULTS.md)
