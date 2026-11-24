# Paper notes

Durable source bundles and working-paper archives for the Spectral RG project.
These are versioned research drafts and inserts; an archived draft is not a declaration that an experiment has finished.

## Shakespeare GPT

- [Current pre-extension source bundle (v10)](shakespeare/archives/Shakespeare_GPT_Section_v10_Source.zip): LaTeX, PDFs, figures, data, and reproduction appendix.
- [Earlier Shakespeare drafts and source bundles](shakespeare/archives/): ten preserved versions, including the v7 comparison draft.
- [Five-seed modified AdamW background experiment](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45375423863).
- [Experiment code and durable results](https://github.com/CalculatedContent/rg_optimizers/tree/shakespeare-graft-five-seeds-20261124/short_test_adamw_graft/experiments/2026-11-24_five_seeds).

A new two-page placeholder revision will use optimizer-level five-seed means and uncertainty, with pending modified AdamW values clearly identified until the experiment is complete.

## Related RG paper archives

[RG theory source snapshots and inserts](rg_theory/archives/) retain two full-paper source snapshots and three smaller theory inserts. The original ZIP names and bytes are preserved.

## Recover a bundle

Open its ZIP link and choose **Download raw file**, or clone this repository and copy the file from `paper_notes/`. Unzip to recover the complete original bundle.

[ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json) records each filename, byte size, SHA-256, Git blob hash, and file count. [SHA256SUMS.txt](SHA256SUMS.txt) can be checked from this directory with `sha256sum -c SHA256SUMS.txt`.

Add new revisions under a new filename or version directory; preserve old archives. Keep the latest draft status explicit and link experimental claims to measured results.
