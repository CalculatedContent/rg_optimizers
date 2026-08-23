# Environment locks

The campaign launcher writes the actual `pip freeze`, Python executable and
version, Torch / accelerator metadata, CUDA/cuDNN or XLA versions, and cache
paths into the `/tmp` provenance directory at run time. The archive command
copies those observed locks here only as part of a dated completed run.

Launcher child processes also receive an ephemeral `HOME` below the required
campaign root. This catches libraries that ignore their dedicated cache or XDG
variables; production commands do not write experiment data or caches to the
user's real home directory.

Each archive retains the raw `pip freeze`, creates a version-pinned
`requirements_replay.txt` for the complete installed campaign dependency
closure, and records `dependency_lock.json`. Archive fails on a dependency
installed from an opaque direct/VCS/file origin rather than pretending that a
name/version pin is equivalent. The archived reproduction sequence runs
`verify-lock` after installing the checked-out project and before touching the
corpus or starting training.

The Git archive does not vendor large wheels or conda packages. Preserve the
public package-channel configuration (or an external wheelhouse) needed to
resolve the recorded builds. Replay fails closed if those artifacts are no
longer resolvable or the recreated dependency closure differs.

No synthetic Darwin, CUDA, or TPU lock file is checked in before that platform
has actually run the experiment.
