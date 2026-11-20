# Scale covariance: SGDM, AdamW and modified AdamW

Research organization for matched MNIST/Fashion-MNIST three-layer MLP experiments. This branch provides the protocol and folders; historical raw runs have NOT yet been recovered into this repository.

Target architecture: 784 -> 1024 -> 1024 -> 10, ReLU; keep 256/512-width historical results separately. Each optimizer gets matched seeds, data splits, batch sizes, epochs and validation-selected hyperparameters. Record epoch-resolved FC1/FC2/FC3 WeightWatcher alpha, KS D and fitted tail count, train/validation/test metrics, optimizer state and configuration. Never tune on test accuracy.

Model-level statistic: alpha_bar = sum_l(alpha_l * sqrt(n_tail_l)/(KS_D_l+0.02)) / sum_l(sqrt(n_tail_l)/(KS_D_l+0.02)).

Required plots: test accuracy vs alpha_bar; each layer and alpha_bar vs epoch (long horizons); paired modified-vs-native AdamW alpha and accuracy changes; hyperparameter sensitivity. Independent runs are statistical replicates, not checkpoints.

Status: protocol scaffold only; no training data, runnable training implementation or validated long-horizon convergence results are claimed. Manuscript source supplied separately as PDF and LaTeX in the conversation.
