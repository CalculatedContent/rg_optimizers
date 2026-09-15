# Execution ledger — September 14, 2026

| Check | Actual status |
|---|---|
| Python source compilation | Passed |
| Local unit/contract tests | 14 passed |
| Plan generation | Passed; smoke/pilot 16 commands, full 80 commands |
| Real WeightWatcher 0.7.7 numerical integration | NOT RUN; unavailable in the construction environment |
| Real repository GPT + AdamW/Muon smoke runs | NOT RUN |
| MPS/CUDA execution and restart integration | NOT RUN |
| Pilot/full training and scientific results | NOT RUN |

The WeightWatcher contract test uses a mock that checks the analysis arguments,
raw/clipped field preservation, matrix binding, and training-weight/RNG
isolation. It does not establish the actual library's numerical behavior.

The evaluation test uses a uniform-logit test double. The remaining tests check
suffix masking, exact exposure counts, the counterfactual schedule, independent
RNG sampling, split disjointness, fixed label corruption, template controls,
forgetting/interference arms, context ablations, matched decay factors, and
interrupted log handling. None is a performance experiment.

No optimizer superiority, alpha/memorization association, grokking, or forgetting
result is claimed. Append actual run paths, fingerprints, device blocks, test
results and complete seed outcomes only after the corresponding executions.
