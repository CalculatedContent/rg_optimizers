NANOGPT GENERALIZATION AUDIT

Purpose
Measure multiple aspects of held-out prediction and greedy generation, and plot
every metric against minimum, mean and each of six layer-matrix alphas. Raw and
clip_xmax alpha have separate plots. No model retraining or optimizer edits.

Quick start (Cloud Shell)
1. Paste:
   cd "$HOME"
   git clone --single-branch --branch codex/generalization-audit https://github.com/CalculatedContent/rg_optimizers.git rg_optimizers_generalization
   bash rg_optimizers_generalization/baseline/nanogpt_one_head/generalization_audit/cloudshell.sh start
2. Progress:
   bash "$HOME/rg_optimizers_generalization/baseline/nanogpt_one_head/generalization_audit/cloudshell.sh" status
3. After COMPLETE appears:
   bash "$HOME/rg_optimizers_generalization/baseline/nanogpt_one_head/generalization_audit/cloudshell.sh" fetch
4. Upload generalization_results.tgz to this conversation for analysis.

For subsequent updates, use git pull --ff-only origin codex/generalization-audit
inside the separate rg_optimizers_generalization checkout. The start command
clones or updates the same branch in a separate checkout on the TPU VM, too.

If cloudshell download fails, download /home/charles/generalization_results.tgz
with Cloud Shell's Download menu. The archive already exists in Cloud Shell.
If interrupted, run start again. Completed checkpoints are skipped. The original
selected checkpoint set and probe samples are frozen in results/protocol.json.
If settings, package versions, model source, or probe inputs change, use a new
output directory; existing measurements will not be silently mixed.

Execution
CPU-only, two CPU threads, nice priority 10, separate from live TPU training.
May still compete for host CPU/RAM/I/O. It does not stop or modify training.
Uses existing Python dependencies; the launcher does not upgrade the training
environment. Missing dependencies cause an explicit import failure.
No XLA initialization is performed. An isolated copy of the exact model.py from
the training commit is bundled, avoiding package imports that might initialize
TPU runtime. Checkpoint tensors load strictly with CPU map_location.

Inputs and identity
Current continuation run: /mnt/disks/rg-data/muonclip-extended/segments/
segment_000001/muon_clip/seed_1337
Data: /mnt/disks/rg-data/rg-nanogpt-one-head/data/{train,test}.bin
Select up to 16 evenly spaced retained epoch checkpoints with complete successful
six-matrix spectra, independent of outcomes. Skip initialization globally, not
the trained continuation step zero. Use the config's global step offset.
Verify checkpoint tensor SHA256, spectral tensor SHA256, protocol fingerprint,
seed, step, and model config before evaluation; verify binary data file hashes
against the run manifest. Checkpoints must be your trusted own .pt files because
torch.load(weights_only=False) uses pickle.
An initial snapshot of available checkpoints is used; this is not a live monitor.
Pruned or missing selected checkpoints cause a clear failure; results already
saved remain usable. Neither missing checkpoints nor missing spectra are imputed.

Probe design
128 unique eligible documents per split, one 257-token window per document;
documents sampled uniformly among those long enough. Windows cannot cross EOT.
This estimates performance on the eligible long-document population, with equal
document weight, not the exact token-weighted metric from historical CSVs.
Fixed seeds, document IDs, offsets, source hashes and library versions are saved.
All checkpoints see exactly the same examples. First 64 held-out probe documents
also provide 64-token prompts with fixed 32-token reference continuations.
Greedy decoding is deterministic. In generated output, EOT is treated as a token
and decoding continues to the fixed 32-token horizon for reproducibility.

Metrics (train and test unless explicitly generation-only)
- NLL / cross-entropy in nats per token; perplexity=exp(mean NLL); bits per token.
- Top-1 token error (%) and top-5 error (%).
- Mean reciprocal rank of the correct token in the full vocabulary (midranks
  for ties); higher is better.
- Multiclass Brier score: sum_j (p_j - 1[j=y])^2; lower is better.
- Predictive entropy, nats: diagnostic, not itself a generalization error.
- ECE (%): 15 fixed equal-width confidence bins; top-label calibration.
- Confident-wrong rate (%): fraction of ALL tokens both wrong and max p >= 0.5.
- Mean per-document 95th percentile token NLL, for difficult-token behavior.
- Rare-token NLL: test targets with <=10 occurrences in the TRAIN corpus;
  pooled over eligible tokens. Omitted if the fixed probe has no such tokens.
- Test-minus-train gaps for NLL, token error, and Brier; separate split samples.

Generation-only metrics (test)
- Token error (%) over free-running continuations, exact-continuation failure (%),
  matching prefix length, and token error over horizons 1, 8, 16 and 32.
- Corpus BLEU and corpus chrF (0-100) via sacrebleu. Also macro-average sentence
  BLEU and sentence chrF, explicitly labeled separately from corpus versions.
- Token ROUGE-L F1: LCS F1 on GPT-2 TOKEN IDs, not standard word-level ROUGE-L.
- Repeated-trigram fraction and its difference from the reference's fraction.
- Teacher-forced NLL of the SAME 32-token references given the prompt.
- Context-ablation NLL increase: reference NLL with earlier prompt tokens
  shuffled minus reference NLL with correct prompt. Keep prompt's last token
  intact. Shuffles are fixed per document. This measures context sensitivity,
  not generalization error by itself; shuffled context is off-distribution.

Interpretation
BLEU/chrF/ROUGE/reference match scores penalize valid alternative continuations.
They are lexical overlap diagnostics, not factuality or hallucination detectors.
Without labeled confusable examples, this audit does NOT measure factual recall,
prototype substitution, or retrieval of the wrong example.
Perplexity and bits/token are transformations of NLL, not independent evidence.
Entropy, repetition and context sensitivity need task-specific interpretation.
The repeatedly monitored test set is exploratory. Confirm any selected metric's
relationship on fresh documents and independent training seeds.

Uncertainty and correlations
95% percentile intervals from 500 bootstrap draws over WHOLE documents; corpus
BLEU/chrF use at most 200 draws for runtime. Recompute nonlinear ECE, perplexity,
and pooled rare-token ratios inside each draw. Reuse draw seeds across checkpoints
to support paired comparisons. These are document-sampling intervals, not seed
variation and not alpha-fit uncertainty. Train/test gaps resample the two splits
independently. Per-document records are retained for later paired analyses.
All metrics, including unfavorable results, are exported. exploratory_correlations.csv gives
descriptive Pearson, Spearman, and first-difference Pearson; no naive p-values
are reported. Checkpoints are autocorrelated, and trying many metrics introduces
multiple-testing risk. First differences do not by themselves solve that issue.

Outputs in /mnt/disks/rg-data/generalization_audit/results
  protocol.json                   frozen probes, settings, checkpoints, hashes
  checkpoints/*.json              atomic checkpoint summaries; resume markers
  per_document/*.npz              scores and calibration sufficient statistics
  generations/*.json              prompts, references, model continuations
  metrics.csv                     every metric and alpha feature by checkpoint
  uncertainty.csv                 estimates and document-bootstrap intervals
  plots/*.png                     every metric vs min/mean/six individual alphas
  all_metrics_vs_alpha.pdf        all plots in one document
  exploratory_correlations.csv    descriptive levels/differences, no p-values
  DONE.json                       written only after successful completion

Plot existing completed audit results independently:
  python3 audit.py plot --output /path/to/results

Code provenance
pinned_model.py is copied without changes from CalculatedContent/rg_optimizers,
commit 7e811f17ee9b41b174bf9b8938646cffedfd0903,
baseline/nanogpt_one_head/src/rg_nanogpt_one_head/model.py.
The tensor hashing convention matches checkpoints.py at the same commit.
Local tests use tiny generated checkpoints with the same schema, not your actual
TPU checkpoint weights, which are not present in this workspace.

Validation performed before delivery
Five pytest tests passed: analytic proper-score cases; reproducible document
sampling; overlap metrics/bootstrap; rejection of spectral/checkpoint mismatch;
and end-to-end evaluation/CSV export/resume/config-change refusal.
Plotting smoke check generated raw/clipped eight-panel figures and PDF; image
layout inspected. Both shell launchers passed bash -n.

Defaults evaluate 16 checkpoints to keep the first audit manageable. To evaluate
every retained checkpoint, change --max-checkpoints to a sufficiently large
number and choose a new AUDIT_OUT in run_on_tpu.sh. The protocol is frozen on
first launch; changing the selection requires a fresh output directory.
