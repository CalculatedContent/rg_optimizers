# Contributing

This repository holds **reference baselines** and **independent RG optimizer
experiments**. Contributions should keep those lanes separate and avoid
unsubstantiated efficacy claims.

## Workflow

1. Fork `CalculatedContent/rg_optimizers` (or use your existing fork).
2. Branch from current **`main`**.
3. Open a pull request against `CalculatedContent/rg_optimizers` `main`.
4. Prefer **small, single-theme** PRs (one package, or docs-only).

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
