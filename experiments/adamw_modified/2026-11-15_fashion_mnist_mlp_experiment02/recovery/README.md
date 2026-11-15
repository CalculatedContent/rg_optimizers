# Authorized Fashion-MNIST recovery

Charles requested resuming this experiment before continuing GPT on15Nov2026. This supersedes the prior instruction not to resume. GPT was cancelled and its launch workflow disabled.

Preserve56 completed runs. Resume six runs from epochs112,75,41,57,25,9 using saved model, optimizer, RNG and history; begin only58 unstarted runs. No replacement seeds or settings. Frozen study.py and adapter.py remain byte-identical. The recovery gets a bounded4.5-hour training window, with5hours total including preservation.

The six continuation probes were measured on the original runtime. GitHub must reproduce their loss and update statistics within the documented floating-point tolerances before training. Exact equality is recorded separately. This is a runtime restoration audit, not a model-selection test.

Recovery outputs are separate. Merge the resulting archives with the original preserved files, verifying that56 finished runs remain unchanged, before the full120-model analysis and600-checkpoint audit. Partial histories/checkpoints remain available; failed outcomes are retained. Do not call recovery complete until the merged final analysis and verification are done.

[Recovery plan](Recovery_Plan.json) · [Frozen input hashes](Recovery_File_Manifest.json)
