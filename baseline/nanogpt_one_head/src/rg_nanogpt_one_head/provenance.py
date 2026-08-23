from __future__ import annotations

"""Source-tree provenance used by manifests and restart fingerprints."""

import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import re
import subprocess
from typing import Any


_SCIENTIFIC_DISTRIBUTIONS = {
    "torch": "torch",
    "torch-xla": "torch-xla",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "PyYAML": "PyYAML",
    "weightwatcher": "weightwatcher",
    "powerlaw": "powerlaw",
    "datasets": "datasets",
    "tiktoken": "tiktoken",
    "sacrebleu": "sacrebleu",
    "matplotlib": "matplotlib",
    "jupyter": "jupyter",
    "ipykernel": "ipykernel",
    "nbformat": "nbformat",
    "papermill": "papermill",
    "packaging": "packaging",
}


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _git_optional(root: Path, *arguments: str) -> str:
    try:
        return _git(root, *arguments)
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def repository_provenance() -> dict[str, Any]:
    """Return a deterministic source identity without assuming a clone path.

    A clean commit is the normal production identity.  For an explicitly
    allowed development run, the SHA-256 of the tracked diff is also included
    so restart compatibility cannot silently survive a code change.
    """

    start = Path(__file__).resolve().parent
    try:
        root = Path(_git(start, "rev-parse", "--show-toplevel"))
        commit = _git(root, "rev-parse", "HEAD")
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        describe = _git(root, "describe", "--tags", "--always", "--dirty")
        tags = tuple(
            value
            for value in _git_optional(
                root, "tag", "--points-at", "HEAD"
            ).splitlines()
            if value and value != "unknown"
        )
        status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        tracked_diff = _git(root, "diff", "--binary", "HEAD", "--")
        return {
            "available": True,
            "repository_name": root.name,
            "commit": commit,
            "branch": branch,
            "describe": describe,
            "tags_at_commit": list(tags),
            "tag_status": ",".join(tags) if tags else "untagged",
            "origin_url": _git_optional(root, "remote", "get-url", "origin"),
            "dirty": bool(status),
            "tracked_diff_sha256": hashlib.sha256(
                tracked_diff.encode("utf-8")
            ).hexdigest(),
            "untracked_file_count": sum(
                line.startswith("??") for line in status.splitlines()
            ),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {
            "available": False,
            "repository_name": "unknown",
            "commit": "unknown",
            "branch": "unknown",
            "describe": "unknown",
            "tags_at_commit": [],
            "tag_status": "unknown",
            "origin_url": "unknown",
            "dirty": None,
            "tracked_diff_sha256": "unknown",
            "untracked_file_count": None,
        }


def source_fingerprint_payload() -> dict[str, Any]:
    """Return only source fields that change executable semantics."""

    source = repository_provenance()
    return {
        "available": source["available"],
        "commit": source["commit"],
        "dirty": source["dirty"],
        "tracked_diff_sha256": source["tracked_diff_sha256"],
        "untracked_file_count": source["untracked_file_count"],
    }


def scientific_dependency_versions() -> dict[str, str]:
    """Return exact direct and transitive campaign dependency identities.

    Including the installed dependency closure in every protocol fingerprint
    prevents seeds run weeks apart from being pooled after a silent transitive
    library upgrade. Optional requirements are included only when installed on
    the current hardware block.
    """

    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError as exc:
        raise RuntimeError(
            "packaging is required to inventory the dependency closure"
        ) from exc

    versions = {"python": platform.python_version()}
    pending: list[str] = []
    for name, distribution in _SCIENTIFIC_DISTRIBUTIONS.items():
        try:
            versions[name] = importlib_metadata.version(distribution)
            pending.append(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"

    normalized = lambda value: re.sub(  # noqa: E731 - compact local invariant
        r"[-_.]+", "-", str(value).strip()
    ).lower()
    visited: set[str] = set()
    while pending:
        requested = pending.pop()
        requested_key = normalized(requested)
        if requested_key in visited:
            continue
        visited.add(requested_key)
        try:
            distribution = importlib_metadata.distribution(requested)
        except importlib_metadata.PackageNotFoundError:
            continue
        actual_name = str(distribution.metadata.get("Name", requested)).strip()
        if not actual_name:
            raise RuntimeError(
                f"installed dependency {requested!r} has no distribution name"
            )
        versions.setdefault(actual_name, str(distribution.version))
        for requirement_text in distribution.requires or ():
            try:
                requirement = Requirement(requirement_text)
            except InvalidRequirement as exc:
                raise RuntimeError(
                    f"installed dependency {actual_name} has an invalid "
                    f"requirement: {requirement_text!r}"
                ) from exc
            dependency_key = normalized(requirement.name)
            if dependency_key in visited:
                continue
            try:
                importlib_metadata.version(requirement.name)
            except importlib_metadata.PackageNotFoundError:
                continue
            pending.append(requirement.name)
    return versions
