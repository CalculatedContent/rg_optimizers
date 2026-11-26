# Accelerated MAGIC SGDM

Parent archive: 09dbaa4e095ff21fc8b1e130798d247b75c15913. Same data and splits, 10→784→784→2, no dropout, batch 256, gradient clip 1, momentum .9. Learning rates .003/.006/.012/.024 crossed with coupled SGD weight decay 0/.0001/.001/.01. One common original development seed; reuse exact existing configurations where available. Select by validation accuracy then CE only; five original confirmation seeds for winner, reuse exact matching runs. Fixed 20,000 update cap, max five parallel workers, preserve partial and failed runs. Record raw FC2 alpha, KS D, tail, eigenvalues, norms, training and validation metrics and gaps; archive checkpoints, hashes, summaries, CSVs, figures and five-seed confidence intervals. Test only after selection; reused test is exploratory. Verify archived data bytes and manifest before training; no overlapping training cohorts. Do not replay completed or partial jobs.

Status: scope staged; workflow not yet installed.
