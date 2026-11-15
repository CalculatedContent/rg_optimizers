# Checkpoints: pending

Planned checkpoint epochs: 0, 30, 60, 90 and 120 for each of 120 runs, giving 600 scheduled checkpoints if all runs complete. No checkpoint exists yet.

Save model parameters, optimizer state, RNG state, configuration and source hashes, epoch and history. Use unique run IDs beneath this experiment's output root. Record file SHA256s and persistent archive identifiers in a manifest. Preserve the latest recoverable state if the budget expires or training fails.

Keep large binary checkpoints in saved artifact archives and check their manifests and download references into GitHub. Never write checkpoints into the MNIST sibling folder, overwrite a completed run, or reuse a checkpoint from another dataset as initialization.
