# Completed run records

`run_experiment.py archive` creates one immutable, review-ready directory here
after all ten replicates and the report notebook validate successfully. The
directory name is `<UTC timestamp>_<source commit>`.

Each record contains the exact protocol and environment provenance, aggregate
tables and figures, the HTML/Markdown report, the executed notebook, a
SHA-256 file manifest, and lightweight per-replicate manifests/test outcomes.
Tokenized corpus files, caches, training logs, and model checkpoints remain
beneath `RG_NANOGPT_EXPERIMENT_ROOT` and are never copied here.
