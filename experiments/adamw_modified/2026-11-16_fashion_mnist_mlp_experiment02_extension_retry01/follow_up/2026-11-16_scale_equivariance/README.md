# Fashion-MNIST scale-audited follow-up

Execution status: see [STATUS.json](STATUS.json) and the exact GitHub Actions run linked there. The new protocol is frozen before launch; prior experiments remain unchanged.

- Same dense 784→256→256→10 ReLU MLP and frozen magnitude-only kernel.
- Batch256, original layer rates, fixed120 epochs with five-epoch warmup and cosine to10% peak.
- A shared50% chance of a zero-padded translation by up to two pixels; independent of optimizer trajectory.
- Three excluded pilot arms, then five fresh paired seeds32001–32005 × AdamW/graft/SGDM:15 confirmatory runs.
- All three alpha estimates, including explicitly audited small-N FC3; retain native-constructor estimates and fit support.
- No early stopping, alpha gate, test-based tuning or selected endpoint. Four-hour training/five-hour workflow cap.

The scale audit verifies a stated coordinate-covariance law, including optimizer state and epsilon. Graft matrices use the SGDM target-rate exponent; native Adam biases retain their different exponent. It does not establish full-MLP loss invariance or an RG trace-log identity.

[Protocol](Protocol.json) · [Scale derivation](SCALE_DERIVATION.md) · [Literature](LITERATURE.md) · [Results](RESULTS.md) · [Parent comparison](../../post_analysis/weighted_alignment/CAMPAIGN_REPORT.md)

After installing the pinned workflow dependencies, `python run.py --smoke` creates the excluded development check; `python verify_setup.py` tests incorrect replay controls; `python run.py` launches the bounded pass once. Never execute these commands in a folder containing a completed launch. `python report.py` and `python archive.py` only analyze/preserve existing outcomes.
