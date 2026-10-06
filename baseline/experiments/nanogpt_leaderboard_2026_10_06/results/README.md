# Leaderboard experiment results

No GPU training results have been produced here. Each launch creates a unique
UTC timestamp/UUID directory containing the source snapshot, manifest, data
hashes, launch command, console log, upstream logs, and completion status.

With `--save-weights`, `weights/rank-00/model.pt` contains CPU-readable dense
weights and inference buffers. All `weights/rank-*/ngram-*.pt` chunks collectively
contain the sparse n-gram table. `weights/WEIGHTS_COMPLETE.json` exists only after
all eight exports have completed and their hashes and row coverage have passed.

Use a persistent `--results-root` on the GPU host and archive the entire run
directory before releasing it. This CUDA runner does not use the stock GPT-2 TPU
cloud-backup service. Generated outputs are ignored by Git.
