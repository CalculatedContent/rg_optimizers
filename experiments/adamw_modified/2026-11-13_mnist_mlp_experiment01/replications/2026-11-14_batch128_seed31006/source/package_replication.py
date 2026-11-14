"""Stage a separately labeled replication and archive all three saved runs."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
REL = Path('experiments/adamw_modified/2026-11-13_mnist_mlp_experiment01/replications/2026-11-14_batch128_seed31006')
DEST = WORKSPACE / 'replication_repository_update' / REL


def main():
    summary = json.loads((ROOT / 'Replication_Result_Summary.json').read_text())
    verification = json.loads((ROOT / 'Replication_Verification.json').read_text())
    assert summary['completed_runs'] == 3 and verification['passed']
    for source, directory in [
        ('study.py', 'source'), ('run_replication.py', 'source'), ('analyze_replication.py', 'source'),
        ('verification_helpers.py', 'source'), ('checkpoint_diagnostics.py', 'source'), ('package_replication.py', 'source'),
        ('Replication_Protocol.json', 'protocol'), ('Parent_Protocol.json', 'protocol'),
        ('Endpoints.csv', 'results'), ('Reference_Endpoints.csv', 'results'), ('Trajectories.csv', 'results'),
        ('Layer_Measurements.csv', 'results'), ('Paired_Differences.csv', 'results'), ('Replication_Result_Summary.json', 'results'),
        ('Replication_Verification.json', 'verification'), ('Checkpoint_Manifest.json', 'verification'),
        ('Replication_Launch.json', 'verification'), ('Replication_Completion.json', 'verification'),
        ('Replication_vs_Original.png', 'plots'), ('Replication_Trajectories.png', 'plots')]:
        target = DEST / directory / source; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
    labels = {'adamw': 'AdamW', 'magnitude_graft': 'Magnitude graft', 'sgdm': 'SGD + momentum'}
    graft = next(r for r in summary['endpoints'] if r['arm'] == 'magnitude_graft')
    activity = next(r for r in summary['test_activity'] if r['spec']['arm'] == 'magnitude_graft')
    collapsed = graft['test_accuracy_pct'] < 50
    finding = ('The fresh seed reproduces the low-accuracy graft outcome. Together with the original five seeds, this is 6/6 tested seeds with collapse at batch 128 / LR ×2.'
               if collapsed else 'The fresh seed does not reproduce the original low-accuracy graft outcome; the table reports the complete new comparison.')
    lines = ['# Batch-128 replication: fresh seed 31006', '',
             '**Complete and verified: three paired runs, 120 epochs each.** Only the seed changed from the failing batch128 / LR×2 condition. Native AdamW and SGDM remain the controls.', '',
             finding, '', '| Optimizer | FC1 α | FC2 α | Mean hidden \\|α−2\\| | Test accuracy |',
             '|---|---:|---:|---:|---:|']
    for r in summary['endpoints']:
        lines.append(f"| {labels[r['arm']]} | {r['alpha_fc1']:.3f} | {r['alpha_fc2']:.3f} | {r['mean_abs_alpha_minus_2']:.3f} | {r['test_accuracy_pct']:.2f}% |")
    lines += ['', 'This is one additional paired seed, so no confidence interval is assigned to these three new outcomes. The original five-seed primary analysis is unchanged. The original batch-128 runs at lower learning rates did not collapse; this replication targets the batch128 / LR×2 combination.', '',
              '![Replication compared with original seeds](plots/Replication_vs_Original.png)', '',
              '![Replication trajectories](plots/Replication_Trajectories.png)', '', '## Checkpoint diagnostics', '',
              f"In the graft checkpoint, the fraction of test examples with an all-zero FC2 vector is {activity['hidden_test_activity']['fc2']['fraction_test_examples_with_zero_hidden_vector']:.3f}; the maximum per-class logit range across test inputs is {activity['max_per_class_logit_range_across_test_examples']:.3g}. Prediction counts are {activity['prediction_counts']}. These are direct measurements from the saved model, with no retraining.", '',
              '## Protocol and verification', '',
              'MNIST; ReLU MLP 784→256→256→10; batch128; LR multiplier2; seed31006; 120 epochs. The new seed changes initialization and minibatch order together, shared across arms. The data split, schedule, matched decay, layer rates, and magnitude-only graft remain unchanged. Adam direction, moments, and bias updates are retained; the target magnitude uses the graft’s own gradient history.', '',
              'Frozen training source SHA256: `1bfd7032e00ad87d729670197925fb40e19bc802a8eb716db2571e56b538d3d8`.', '',
              f"All 15 checkpoints are hashed. All three initial/final hashes, saved spectra, alpha estimates, saved test-accuracy counts, and eleven-step checkpoint replays pass. {verification['recorded_nondegenerate_magnitudes_checked']} recorded nondegenerate graft magnitudes were checked, with maximum relative error {verification['max_recorded_magnitude_relative_error']:.3g}. The existing denominator floor is handled explicitly in the graft check. Training wall time was {summary['training_wall_seconds']/60:.2f} minutes.", '',
              'A separate replay from initialization checks all three nonzero matrix updates: Adam direction, moments and bias updates are preserved, and magnitudes match their SGDM targets. The epoch-30 replay also checks the collapsed, zero-update regime; direction is undefined there.', '',
              '[Verification details](verification/Replication_Verification.json) · [Protocol](protocol/Replication_Protocol.json) · [Endpoints](results/Endpoints.csv) · [Paired differences](results/Paired_Differences.csv) · [Saved states and raw data](ARTIFACTS.md).', '',
              'FC3 fits are exploratory because the output matrix has only ten eigenvalues. The primary means and distances use FC1 and FC2. No additional settings or datasets were trained.', '',
              'To reproduce the analysis, extract the complete archive, match the package versions in the protocol, and run `python analyze_replication.py` from its root. That command reads and replays saved states without continuing training. Do not invoke the training source’s historical default sweep. The completed replication does not authorize another run.', '']
    (DEST / 'README.md').write_text('\n'.join(lines))
    out = WORKSPACE / 'artifacts/replication_seed31006'; out.mkdir(parents=True, exist_ok=True)
    archive = out / 'Experiment01_Batch128_Seed31006_Replication_Complete.zip'
    assert not archive.exists(), 'Resolve an existing archive before retrying packaging'
    with zipfile.ZipFile(archive.with_suffix('.zip.part'), 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in sorted(ROOT.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.tmp', '.pyc'):
                z.write(f, str(f.relative_to(ROOT)))
        z.writestr('README.md', '\n'.join(lines))
    archive.with_suffix('.zip.part').replace(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if '/checkpoints/' in n and n.endswith('.pt')]) == 15
    metadata = {'path': str(archive), 'bytes': archive.stat().st_size,
                'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'zip_crc_passed': True,
                'completed_runs': 3, 'checkpoints': 15, 'source_sha256': summary['source_sha256']}
    (out / 'Archive_Metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
