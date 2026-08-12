# Optimizer provenance field inventory

**Class B/C companion.** Snapshot of whether packages emit local-delta-style
provenance fields (`actuator_id`, `ecs_backend`, `dose_definition`, `dose_value`,
`is_first_apply`, and where present `is_first_due`). Update when packages gain
logging.

| Package | Provenance fields | Notes |
|---|---|---|
| `wwpgd_local_delta` | **yes** | SoT grammar (#34): null dose on no-op; schedule-aware first apply |
| `trace_log_tracker` | **pending / open PR** | Midpoint sibling: open collab PR **#49** (F1 `is_first_apply` vs `is_first_due`) when not yet on main |
| `self_consistent_trace_log_tracker` | **yes** (step stats) | This package: `ecs_backend=self_consistent_F_m`; first **successful** apply per param (F1); null dose on no-op |
| `adaptive_spectral_guard` | no | Different cadence model |
| `ecs_probe_loss_trace_wall` | no | |
| `spectral_rg_flow_projector` | no | |
| `full_matrix_log_rg` | no | |

**Rule:** extend one package at a time; do not invent dose when no correction ran.

**Field meanings (joint tables):**

| Field | Meaning |
|---|---|
| `actuator_id` | Package / actuator name |
| `ecs_backend` | Support lineage label (`midpoint_pl_detx`, `self_consistent_F_m`, …) |
| `dose_definition` | Named ratio definition string |
| `dose_value` | Realized dose or **null** if no correction committed |
| `is_first_apply` | First **successful** correction for that parameter (not first schedule tick) |
| `is_first_due` | First schedule-due step (clock); may have null dose |
