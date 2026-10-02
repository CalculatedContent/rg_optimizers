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


COMPLETED AUDIT RESULTS (2026-10-02 UTC)

GENERALIZATION AUDIT: RAW VS CLIPPED ALPHA

Data and scope
16 verified checkpoint summaries, 2.15M–3.01M steps, one MuonClip seed (1337).
128 fixed held-out documents for teacher metrics, 64 for 32-token greedy
continuations. Original epoch 0 excluded; trained continuation step zero retained.
31 test/gap metrics × 8 alpha summaries × 2 variants = 496 exploratory pairs;
several metrics and alpha variants are duplicates or highly dependent.

Main result
The new audit shows descriptive associations that were absent or weaker in the
previous 63-checkpoint monitoring view. Mean raw alpha versus token error has
Pearson r=+0.707 and Spearman rho=+0.776. Mean clipped alpha has r=-0.165.
Raw mean versus reciprocal rank has r=-0.702 (higher rank score is better).
Raw mean versus NLL has r=+0.426; clipped mean r=-0.140.
Raw mean versus reference-continuation NLL has r=+0.614; clipped mean r=-0.037.
Therefore this dataset does not support the blanket claim that alpha relates to
nothing. Larger RAW mean alpha accompanies worse prediction on these probes.
It does not demonstrate a causal mechanism or predictive validity on new runs.

Robustness
Raw mean/token-error r after linear step adjustment: +0.660.
First-difference r: +0.547 (unequal ~50–60k step spacing; differences, not rates).
Leave-one-checkpoint-out range: +0.650 to +0.763.
Joint resampling of the same documents across all checkpoints gives a conditional
95% percentile interval of approximately [+0.177,+0.776] for that correlation.
This conditions on this checkpoint trajectory: it is not a seed-level interval,
does not correct metric selection, and is not a temporal-independence test.

What clipping changes
Raw and clipped Q, K and MLP-input alphas are identical here. Clipping changes O
at 11/16 checkpoints, V at 1/16 and MLP output at 3/16. Minimum alpha changes at
only 1/16. Thus raw-versus-clipped minimum results are almost identical.
Raw O ranges 2.68–9.47. Its covariance contribution accounts for about 81% of the
raw mean's variance. The raw mean with O excluded still correlates +0.518 with
token error, so O explains much, but not all, of the association. The difference
(raw mean minus clipped mean) correlates +0.710 with token error. The component
removed by clipping carries information in this sample; whether it reflects
fit instability or a meaningful spectral feature needs spectrum-level study.

Layer-specific leads
K: raw=clipped, r=+0.613 with top-5 error, -0.619 with reciprocal rank.
V: clipped r=-0.646 with difficult-token NLL (mean per-document 95th percentile),
raw r=+0.412 is unstable (leave-one-out changes sign). For clipped V,
first-difference r=-0.826, leave-one-out -0.739 to -0.533.
This is an exploratory lead, not a proven generalization predictor.
MLP output: clipped r=-0.497 with NLL but first-difference r=-0.035, suggesting
its level correlation does not track local changes reliably.
Raw V versus calibration error r=-0.633 collapses to about -0.089 after removing
one influential checkpoint. Do not interpret that as a robust calibration link.

Generation metrics and measurement limitations
Corpus BLEU 0.469–1.037: raw mean r=-0.020; clipped mean r=+0.066. No mean-alpha
relationship. Minimum alpha/BLEU r about +0.33, weaker exploratory association.
Corpus chrF 14.31–15.54: raw mean r=-0.509; clipped mean r=-0.003.
Free token error 97.02–97.85%; raw mean r=-0.380 and clipped mean r=+0.266.
All exact-continuation failures are 100%: zero exact matches for every checkpoint.
This metric is saturated and cannot distinguish checkpoints.
Generated repeated-trigram fraction is 32.7–42.9%, compared with 0.625% in the
fixed references. Samples show repetitive loops such as repeated "be more likely
to". This is a real generation pathology, but its mean-alpha correlations are
weak (raw -0.117; clipped +0.108). It does not establish memorization or hallucination.
Rare-token NLL uses only TWO test tokens in two documents (six train tokens).
Discard it as a reliable generalization diagnostic for this probe size.
Reference-overlap scores penalize legitimate alternative continuations, while
entropy and shuffled-context sensitivity are diagnostics, not direct errors.

Overfitting and changes over time
Test NLL rises 5.22817 -> 5.24364 (+0.01547 nats/token); paired-document 95% CI
[+0.00456,+0.02600]. Training NLL also rises +0.02019, CI [+0.00713,+0.03274].
The test-minus-train NLL gap falls from 0.10831 to 0.10360. This is not the
characteristic pattern of improving train fit and worsening test performance.
Token-error change is only +0.10376 percentage points; paired CI [-0.17090,+0.37537].
Minimum raw alpha stays above 2 (2.1425–2.5069); no below-2 transition is tested.

Why this differs from the previous plots
This audit spans 2.15–3.01M, rather than ending at 2.77M, samples 16 rather than
63 checkpoints, and uses a new document-balanced fixed probe. Previous monitoring
used a different token-window probe. Their absolute errors and correlations
should not be treated as directly interchangeable estimates.

Next discriminating check
Keep all results exploratory. Retest the raw-mean/token-error and clipped-V/
difficult-token-loss leads on new documents with the metrics fixed in advance,
then an independent seed. Increase and redesign the generation probes for exact
recall; arbitrary natural-text exact continuation is saturated here. Inspect
O's raw/clipped fit changes at matched checkpoints before treating clipping as
removing either noise or useful signal. Do not choose metrics by the largest r.


Fresh overnight export (run in Cloud Shell)
  git -C "$HOME/rg_optimizers_generalization" pull --ff-only origin codex/generalization-audit
  bash "$HOME/rg_optimizers_generalization/baseline/nanogpt_one_head/generalization_audit/export_training.sh"

The exporter snapshots CSV/YAML/JSON metadata across all continuation segments
and the original long run. It includes checkpoint filenames/sizes/mtimes, not
checkpoint tensors or corpus files. Live files are read separately, not as an
atomic training snapshot; downstream analysis must use complete matched
six-matrix measurements and config offsets. A trailing incomplete CSV line is
removed in the exported copy. The export manifest records time and omissions.

Upload the resulting muonclip_overnight_metrics.tgz for analysis. If the browser
download fails, use Cloud Shell's Download menu with the printed local path.
Do not rerun start expecting new checkpoints: its results/protocol.json freezes
the original 16-checkpoint selection, and run_on_tpu.sh targets segment_000001.
For a new audit, select the intended retained segment and a fresh output
directory; keep document seeds and evaluation settings fixed for comparison.
Do not update the live training checkout to install audit changes.


EXACT-PROBE EXTENSION (2026-10-02)

Use exact_cloudshell.sh start to evaluate 16 evenly spaced NEW retained
checkpoints after the original audit endpoint through 5,100,000 global steps.
Run from the updated separate Cloud Shell checkout:
  bash baseline/nanogpt_one_head/generalization_audit/exact_cloudshell.sh start
  bash baseline/nanogpt_one_head/generalization_audit/exact_cloudshell.sh status
  bash baseline/nanogpt_one_head/generalization_audit/exact_cloudshell.sh fetch

The evaluator audit.py and pinned_model.py remain byte-identical. extend_audit.py
verifies source hashes, data hashes, torch/numpy/sacrebleu/tiktoken versions,
document IDs and exact token offsets against the original results/protocol.json.
All evaluation settings come from that original protocol. Only checkpoint
selection changes. A mismatch fails visibly. It does not replace the probe with
the training monitor, detrend alpha or errors, or overwrite the original audit.

New output: /mnt/disks/rg-data/generalization_audit/exact_extension_20261002
Log: /mnt/disks/rg-data/generalization_audit/exact_extension.log
Download: /mnt/disks/rg-data/generalization_audit/exact_probe_results.tgz

Selected model-only checkpoints are hard-linked into staged run directories on
the same data disk, protecting them from pruning during CPU evaluation. Metadata
is copied; live training code/configuration is not changed. These links retain
the selected weights until the extension directory is explicitly cleaned up.
The selection is frozen for restart. Interrupted runs resume completed results.
If training has already pruned older candidates, selection uses the remaining
available checkpoints and records that inventory and the selected steps.

The archive contains original and new per-document scores, generation samples,
checkpoint summaries, uncertainty, verified protocols, unadjusted correlations
for previous/new/combined periods, and mean-raw-alpha regression figures. It
does not contain checkpoint weights. Error bars remain document-bootstrap
intervals, not seed variation. Temporal dependence and exploratory selection
still limit inference; the requested primary comparison keeps the linear trend.

Validation: syntax checks; original scorer hash checked against the completed
audit; mismatched document/data/settings rejection; merge/deduplication and
regression-plot smoke checks using actual old results reproduce r=0.70660064.
Full new-checkpoint inference must run on the TPU host with its original Python
environment; torch/pytest are unavailable in this editing workspace.
