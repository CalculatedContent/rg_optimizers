# Saved checkpoints and raw data

[Experiment01_Batch128_Seed31006_Replication_Complete.zip](https://chatgpt.com/library?file_id=libfile_fde3_f-1f813bd49d2c81918ff8b46daae7c9f4_3360a23133288191af30e7bb737808b0&account_id=personal)

- Complete new-seed replication: three runs, 120 epochs each, 15 checkpoints.
- Includes MNIST data and fixed split, model/optimizer/RNG states, per-epoch histories, sampled update statistics, saved spectra, test logits, analysis and verification code, and both figures.
- Archive size: 54,385,506 bytes.
- SHA256: `5dbe0f62006d5d1059699adbf2e1ca475323ecb5c558ccc448e8f4bbabc5279f`.
- Archive CRC verification passed before upload.
- This owner-access-controlled download does not grant public access merely because its link appears in GitHub.

Extract the archive into an empty directory. It contains the replication at its root. Use the package versions in `Replication_Protocol.json`, then run `python analyze_replication.py` from that directory to replay saved checkpoints and regenerate the analysis. No additional training is needed. Do not run the historical training source's default sweep.

The nonzero initial-step check verifies Adam direction, moments and bias updates, and the SGDM target magnitudes. The epoch-30 replay also verifies the collapsed zero-update regime, explicitly preserving the frozen denominator floor. Original Experiment 01 outcomes are not replaced or pooled into new confidence intervals.
