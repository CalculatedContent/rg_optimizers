"""Descriptive checkpoint diagnostics for the explicitly flagged low-accuracy runs.

No training, parameter changes, exclusions, or replacement runs are performed.
"""
import json
from pathlib import Path
import numpy as np
import torch
import study as s

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def diagnose(spec, xt, yt):
    d = ROOT / 'runs' / s.run_id(spec)
    done = json.loads((d / 'done.json').read_text())
    ck = torch.load(d / 'checkpoints/epoch_120.pt', weights_only=False)
    net = s.MLP(); net.load_state_dict(ck['model']); net.eval()
    assert s.weight_hash(net) == done['final_weight_hash']
    z = np.load(d / 'test_logits_epoch_120.npz')
    predictions = z['logits'].argmax(1)
    active = {n: np.zeros(256, dtype=np.int64) for n in ('fc1', 'fc2')}
    all_zero = {'fc1': 0, 'fc2': 0}
    restored_logits = []
    for start in range(0, len(yt), 1024):
        h1 = torch.relu(net.fc1(xt[start:start + 1024]))
        h2 = torch.relu(net.fc2(h1))
        restored_logits.append(net.fc3(h2).numpy())
        for n, h in [('fc1', h1), ('fc2', h2)]:
            positive = h > 0
            active[n] += positive.sum(0).numpy()
            all_zero[n] += int((~positive.any(1)).sum())
    restored_logits = np.concatenate(restored_logits)
    max_logit_error = float(np.abs(restored_logits - z['logits']).max())
    assert np.array_equal(restored_logits.argmax(1), predictions)
    stats = np.load(d / 'statistics/epoch_120.npz')['values']
    return {'spec': spec, 'test_accuracy_pct': float(100 * (predictions == z['labels']).mean()),
            'prediction_counts': np.bincount(predictions, minlength=10).tolist(),
            'max_per_class_logit_range_across_test_examples': float(np.ptp(z['logits'], axis=0).max()),
            'max_restored_logit_abs_error': max_logit_error,
            'hidden_test_activity': {n: {'units_never_positive_on_test': int((active[n] == 0).sum()),
                                         'mean_fraction_of_units_active': float(active[n].sum() / (len(yt) * 256)),
                                         'fraction_test_examples_with_zero_hidden_vector': all_zero[n] / len(yt)}
                                     for n in ('fc1', 'fc2')},
            'final_epoch_recorded_gradient_norm_max': {n: float(stats[:, j, 0].max()) for j, n in enumerate(s.LAYERS)},
            'final_alpha': {n: done['final']['spectra'][n]['alpha'] for n in s.LAYERS}}


def main():
    completion = json.loads((ROOT / 'Completion.json').read_text())
    assert completion['completed'] == 135
    *_, xt, yt = s.load_data()
    results = []
    for spec in s.specifications():
        done = json.loads((ROOT / 'runs' / s.run_id(spec) / 'done.json').read_text())
        if done['final']['test']['accuracy'] < .5:
            results.append(diagnose(spec, xt, yt))
    s.writej(ROOT / 'Low_Accuracy_Diagnostics.json', {
        'selection_rule': 'Descriptive inspection of every final run with test accuracy below 50%; all remain in all primary estimates.',
        'training_performed': False, 'source_sha256': s.sha(ROOT / 'study.py'), 'runs': results})
    print(json.dumps({'inspected_runs': len(results), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
