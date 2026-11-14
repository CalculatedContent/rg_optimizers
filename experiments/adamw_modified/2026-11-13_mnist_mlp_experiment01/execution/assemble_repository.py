"""Stage completed evidence for the existing experiment's GitHub folder."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import datetime
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
from analyze import interval
import study as s

ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parents[1] / 'repository_update'
REL = Path('experiments/adamw_modified/2026-11-13_mnist_mlp_experiment01')
DEST = STAGE / REL


def copy(name, target):
    target = DEST / target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)


def main():
    verified = json.loads((ROOT / 'Final_Verification.json').read_text())
    assert verified['passed'] and verified['completed_runs'] == 135
    assert s.sha(ROOT / 'study.py') == '1bfd7032e00ad87d729670197925fb40e19bc802a8eb716db2571e56b538d3d8'
    e = pd.read_csv(ROOT / 'Endpoints.csv')
    assert len(e) == 135 and e.seed.nunique() == 5
    for name in ('Endpoints.csv', 'Trajectories.csv', 'All_Layer_Measurements.csv',
                 'Setting_Summaries.csv', 'Overall_Summaries.csv', 'Paired_Summaries.csv',
                 'Paired_Seed_Differences.csv', 'Accuracy_Correlations.csv', 'Common_Support_Pairs.csv'):
        copy(name, 'results/' + name.lower())
    for name in ('Final_Verification.json', 'Analysis_Audit.json', 'Checkpoint_Manifest.json',
                 'Pass_Restoration_Audit.json', 'Completion_Pass_Launch.json', 'Completion.json',
                 'Completion_Environment.json', 'Low_Accuracy_Diagnostics.json', 'Restored_Spectrum_Audit.json'):
        copy(name, 'verification/' + name)
    for name in ('complete_pass.py', 'verify_pass.py', 'plot_pass.py', 'preserve_completed_seeds.py', 'assemble_repository.py', 'write_result_tables.py', 'package_complete_evidence.py', 'diagnose_low_accuracy.py'):
        copy(name, 'execution/' + name)
    for f in sorted(ROOT.glob('*.png')):
        copy(f.name, 'plots/' + f.name)
    done = [json.loads((ROOT / 'runs' / s.run_id(spec) / 'done.json').read_text()) for spec in s.specifications()]
    s.writej(DEST / 'results/completed_runs.json', done)
    status = {'status': 'complete_and_verified', 'planned': 135, 'completed': 135,
              'seeds': s.SEEDS, 'batch_sizes': [128, 256, 512], 'lr_multipliers': [.5, 1, 2],
              'arms': s.ARMS, 'epoch': 120, 'training_active': False,
              'source_sha256': verified['source_sha256'], 'next_experiment_started': False,
              'runs': [{'run_id': s.run_id(r['spec']), **r['spec'], 'status': r['status'], 'last_epoch': r['last_epoch']} for r in done]}
    s.writej(DEST / 'results/status.json', status)
    head = []
    for arm, g in e.groupby('arm'):
        head.append({'arm': arm, 'metric': 'alpha_head_exploratory',
                     **interval(g.groupby('seed').alpha_head.mean())})
    pd.DataFrame(head).to_csv(DEST / 'results/head_summaries.csv', index=False)
    accuracy_counts = [{'arm': arm, 'n_runs': len(g),
                        'test_accuracy_below_50_pct': int((g.test_accuracy_pct < 50).sum()),
                        'min_test_accuracy_pct': float(g.test_accuracy_pct.min()),
                        'max_test_accuracy_pct': float(g.test_accuracy_pct.max())}
                       for arm, g in e.groupby('arm')]
    summary = {'experiment': 'Experiment01', 'dataset': 'MNIST', 'architecture': [784, 256, 256, 10],
               'status': 'complete_and_verified', 'completed_runs': 135, 'epochs': 120,
               'seeds': s.SEEDS, 'batch_sizes': [128, 256, 512], 'lr_multipliers': [.5, 1, 2],
               'arms': s.ARMS, 'source_sha256': verified['source_sha256'],
               'independent_replicates': 5,
               'confidence_interval_method': 'Average the nine fixed settings within each paired seed, then Student t 95% interval across five seed averages. Setting-specific intervals also use five paired seeds. Descriptive, not multiplicity adjusted.',
               'overall_summaries': pd.read_csv(ROOT / 'Overall_Summaries.csv').to_dict('records'),
               'overall_paired_effects': pd.read_csv(ROOT / 'Paired_Summaries.csv').query('scope == "overall"').to_dict('records'),
               'head_exploratory_summaries': head, 'accuracy_range_and_counts': accuracy_counts,
               'verification': {'passed': True, 'checkpoints_hashed': verified['checkpoints_hashed'],
                                'all_final_model_hashes_match': True,
                                'current_runtime_exact_replay_runs': len(verified['exact_current_runtime_replay_checks']),
                                'historical_tolerance_replay_runs': len(verified['historical_runtime_replay_audit']['checks']),
                                'graft_checkpoint_checks': len(verified['graft_direction_magnitude_moment_bias_checks']),
                                'denominator_floor_layer_checks': verified['denominator_floor_layer_checks'],
                                'matched_initializations': verified['analysis_audit']['matched_initializations'],
                                'matched_example_order': verified['analysis_audit']['matched_per_epoch_example_order']},
               'training_wall_seconds_including_prior_segments': verified['training_wall_seconds'],
               'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'next_experiment_started': False, 'stop_for_user_review': True}
    s.writej(ROOT / 'Experiment01_Result_Summary.json', summary)
    copy('Experiment01_Result_Summary.json', 'results/Experiment01_Result_Summary.json')
    print(json.dumps({'stage': str(STAGE), 'files': len(list(DEST.rglob('*'))), 'accuracy_ranges': accuracy_counts}, indent=2))


if __name__ == '__main__':
    main()
