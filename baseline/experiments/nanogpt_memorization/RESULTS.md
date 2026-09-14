# Execution ledger

Date: 2026-09-14.

## Completed in this preparation session

- Read the repository's current baseline recipe, model, optimizer implementation,
  spectral-monitor implementation, qualification document, and dependency file.
- Pinned source commit and Git blob identities in `study.json`.
- Created this separate experimental-design/measurement folder.
- Ran `python -m pytest -q tests`: **32 passed in 2.78 seconds**, CPU.
- Tests cover exact-versus-token recall, teacher-forced-versus-free-running
  behavior, suffix alignment, exhaustive finite-universe exposure and ties,
  prefix-search censoring, candidate batching, deterministic probes, exact
  scheduled presentations, source-drift checks, plan counts, and WW guards.

## Not completed / no results claimed

- No end-to-end memorization trainer or natural/association data adapter has been
  implemented; the integration contract is in `TRAINER_CONTRACT.md`.
- No FineWeb-Edu download, canary-injected training, or AdamW/Muon training
  campaign was run in this session.
- No actual WeightWatcher call was executed here; WeightWatcher is not installed
  in this execution environment. The hook reuses the inspected repository code.
- No pinned-nanoGPT optimizer-step, resume, or target-hardware integration test
  was executed. The controlled test model is not nanoGPT.
- No empirically optimal configuration was established; the repository's
  source-backed recipe is inherited without relabeling it as a frozen optimum.
- There are no measured optimizer advantages, memorization rates, spectral
  trends, or significance claims yet.
