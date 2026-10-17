# Stock GPT-2 Small / FineWeb10B — SGD + momentum target run

This experiment uses the unchanged SHA256-verified upstream GPT-2 Small model: 12 layers, 12 heads, width 768, MLP 3072, context 1024, vocabulary 50257, and 124439808 unique parameters.

## Optimizer

- Classical `torch.optim.SGD`
- peak LR **0.3**
- momentum **0.90**
- Nesterov off
- matrix/embedding weight decay **3e-5**
- bias/LayerNorm weight decay 0
- global gradient clipping 1.0

## Full target-reaching schedule

The previous 6200-update short profile cooled down before the model was converged. It is retained only as historical output and must not be treated as a converged SGD control.

The replacement run starts from fresh initialization and uses the full original GPT-2/FineWeb horizon: **19560 updates**, **700-update linear warmup**, then **cosine decay to zero**. Validation and WeightWatcher run every **250 updates** and at exit. With `--stop-at-target`, training stops only after a full 10485760-token validation reaches **NLL <= 3.28**; otherwise the 19560-update schedule is the safety cap.

This combines the published SGDM optimizer center (LR 0.3, momentum 0.9, small matrix decay) with the repository's original full GPT-2/FineWeb training horizon. It is not claimed to be a published end-to-end SGDM schedule.

## Launch

```bash
python3 baseline/gpt2_small/muon_speedrun/cloudshell.py start \
  --node YOUR_TPU_NODE --optimizer sgd_momentum --hours 12 \
  --stop-at-target --replace-current
```

The launcher performs dependency, attention, and full-model optimizer preflights before training. The target is selected solely by validation NLL; alpha is diagnostic and is never used for stopping.
