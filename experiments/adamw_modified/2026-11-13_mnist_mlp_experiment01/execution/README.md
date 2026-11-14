# Execution and reanalysis

The 2026-11-14 continuation recovered 14 completed runs and trained only the remaining 121. `complete_pass.py` records that historical execution; it is not a launcher for another experiment. Do not run training to regenerate the plots.

1. Obtain the common analysis/data archive and the five seed checkpoint archives from [ARTIFACTS.md](../ARTIFACTS.md). Extract them into the same empty working directory. The resulting layout contains `study.py`, `Protocol.json`, `data/`, and `runs/`.
2. Match the numerical packages in [Completion_Environment.json](../verification/Completion_Environment.json). The original scripts retain `/tmp/adamw_shape_deps` as their preferred dependency path. Their output directory is the directory containing the restored `study.py`; do not execute the copy in `frozen/` against an unrelated directory.
3. Check the frozen source SHA256 and dataset hashes. `Checkpoint_Manifest.json` provides the 675 individual checkpoint checksums.
4. From the restored directory, run `python analyze.py`, `python verify_pass.py`, `python diagnose_low_accuracy.py`, and `python plot_pass.py` in that order. These commands analyze or replay saved states without continuing training. `plot_pass.py` refreshes the heatmap layout and adds explicitly labeled detail views.

Exact floating-point replay depends on the numerical runtime. The historical 14-run restoration uses the documented tolerances; current-runtime selected checkpoints were verified exactly in the completion environment. A different machine may produce small numerical differences that require explicit investigation and documentation. Do not silently loosen checks or alter the frozen training source.

`assemble_repository.py` and `write_result_tables.py` stage the numerical evidence and tables. `preserve_completed_seeds.py` and `package_complete_evidence.py` package finished evidence; they do not start training or background supervisors. Upload receipts and authentication helpers are not part of this repository.

The original `frozen/verify.py` is retained as provenance. The completion verification adapter handles the documented cross-runtime precision differences and the existing 10⁻³⁰ denominator floor. The floor affects four of the 135 replayed layer checks in already collapsed/tiny-update cases; nonzero, nondegenerate update directions and magnitudes are checked separately. Complete numerical details are in `Final_Verification.json`.
