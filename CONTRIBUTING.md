# Contributing

This repository holds **reference baselines** and **independent RG optimizer
experiments**. Contributions should keep those lanes separate and avoid
unsubstantiated efficacy claims.

This guide is for people sending **pull requests**. How you get a branch onto
GitHub depends on your access (see below). The **layout and PR expectations**
apply to external and write-access contributors alike; maintainers may use a
shorter internal path for trivial fixes.

## Who you are (access paths)

| Role | Typical access | How you open a PR |
|---|---|---|
| **External contributor** | No push access to this repo | **Fork** under your account, push a branch to *your* fork, open a PR into `CalculatedContent/rg_optimizers` `main` |
| **Collaborator / write access** | Can push branches to this repo | Branch on **this** repo from current `main`, push here, open a PR into `main` (no personal fork required) |
| **Maintainer** | Admin / merge rights | Same as write access for reviewable changes; direct commits to `main` only per the project’s usual maintainer practice (prefer PRs for non-trivial work) |

**Fork is not a social rank.** It is only the usual GitHub path when you cannot
push to this repository. If you already have write access, do not fork just to
satisfy a ritual.

### External contributor (no write access)

1. Fork `CalculatedContent/rg_optimizers`.
2. Branch from current upstream **`main`**.
3. Push the branch to **your fork**.
4. Open a PR against `CalculatedContent/rg_optimizers` `main`.

### Collaborator with write access

1. Clone this repository (or add it as `origin`).
2. Branch from current **`main`**.
3. Push the branch to **this** repo.
4. Open a PR against `main`.

Prefer **small, single-theme** PRs (one package, or docs-only) in either path.

## Layout (read before editing)

| Path | Role |
|---|---|
| [`baseline/`](baseline/) | **No RG correction.** Matched baseline recipes and notebooks (MLP3, ViT, nanoGPT one-head, nanochat, …). |
| [`optimizers/`](optimizers/) | **Independent** experiment packages (own notebooks, tests, hypotheses). Not drop-in replacements for baseline without reading each package README. |
| [`OPTIMIZER_VARIANTS.md`](OPTIMIZER_VARIANTS.md) | Read-only map of actuators, support/geometry, and cadence for every optimizer package. |

Do not mix baseline recipe changes and optimizer-intervention changes in the same PR unless the PR description explicitly justifies the coupling.

## Before you open a PR

From the repo root (with your usual venv/conda env):

```bash
# example: baseline-oriented syntax/tests (see CI for the full matrix)
PYTHONPATH=src pytest -q
```

Also run the package tests that match the files you touched (for example CI
workflows under `.github/workflows/` such as `baseline-tests.yml` or
`wwpgd-local-delta-tests.yml`). Name what you ran in the PR body.

## PR expectations

These expectations apply to **any** PR author (external or write-access), unless
a maintainer explicitly says otherwise for a given change.

| Do | Don't |
|---|---|
| State docs-only vs behavior vs logging | Claim an optimizer “wins” without a stated baseline protocol |
| Keep package defaults unchanged unless the PR is about defaults | Treat α = 2 as a universal point optimum (prefer **boundary** language when α is discussed) |
| Point at `OPTIMIZER_VARIANTS.md` when adding a new optimizer folder | Unify all optimizers into one package without discussion |
| Keep baseline test metrics **monitoring-only** (no test-set model selection) | Bundle unrelated refactors with a one-line fix |

Suggested PR body checklist:

```text
## Type
- [ ] docs only
- [ ] logging / provenance fields
- [ ] behavior (describe default impact)

## Summary
...

## Test plan
- [ ] pytest / named CI workflow
- [ ] no scientific efficacy claim

## Out of scope
...
```

## Where to look

| Topic | Path |
|---|---|
| Root overview + quick start | [`README.md`](README.md) |
| Optimizer catalog | [`OPTIMIZER_VARIANTS.md`](OPTIMIZER_VARIANTS.md) |
| Baseline qualification trail | [`baseline/FINAL_BASELINE_QUALIFICATION.md`](baseline/FINAL_BASELINE_QUALIFICATION.md) and sibling audit docs under `baseline/` |

## License

By contributing, you agree that your contributions are licensed under the same
terms as this repository.
