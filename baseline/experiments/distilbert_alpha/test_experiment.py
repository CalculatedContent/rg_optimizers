"""CPU engineering checks; no IMDb or test-set performance claims."""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from common import (Config, atomic_json, batch_indices, config_dict, make_optimizer,
                    stratified_split, lr_at, TensorSGD)
from prepare import prepare
from train import worker
from spectra import summarize
from accuracy_report import correlate, seed_means, AGGREGATE_METRICS, report
from sweep import plan
from cloudshell import bounded_deadline
import time


def assert_tree(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        test.assertTrue(np.array_equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a:
            assert_tree(test, a[k], b[k])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree(test, x, y)
    else:
        test.assertEqual(a, b)


class ExperimentTests(unittest.TestCase):
    def test_masked_eight_rank_coverage(self):
        for size in (3, 64, 65, 2500, 22500):
            actual = []
            counts = []
            for rank in range(8):
                batches = list(batch_indices(size, 16, 8, rank, 1, 401, True))
                counts.append(len(batches))
                for indices, weights, real in batches:
                    self.assertEqual(len(indices), 16)
                    actual.extend(indices[weights == 1].tolist())
            self.assertEqual(sorted(actual), list(range(size)))
            self.assertEqual(len(set(counts)), 1)

    def test_split(self):
        labels = np.tile(np.arange(2), 12500)
        a, b = stratified_split(labels, 1701, 1250)
        self.assertEqual((len(a), len(b)), (22500, 2500))
        self.assertFalse(set(a) & set(b))
        self.assertEqual(np.bincount(labels[b]).tolist(), [1250, 1250])
        a2, b2 = stratified_split(labels, 1701, 1250)
        self.assertTrue(np.array_equal(a, a2) and np.array_equal(b, b2))

    def test_optimizer_groups(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        for name in ('adamw', 'adam', 'sgd'):
            optimizer = make_optimizer(model, replace(Config(), optimizer=name))
            self.assertEqual(len(optimizer.param_groups[0]['params']), 1)
            self.assertEqual(len(optimizer.param_groups[1]['params']), 3)
            self.assertEqual(optimizer.param_groups[1]['weight_decay'], 0)

    def test_tensor_sgd_matches_reference(self):
        for momentum, nesterov in ((0., False), (.9, False), (.9, True)):
            a = torch.nn.Parameter(torch.arange(8, dtype=torch.float32) / 10)
            b = torch.nn.Parameter(a.detach().clone())
            x = TensorSGD([{'params': [a], 'weight_decay': .01}], torch.tensor(.1), momentum, nesterov)
            y = torch.optim.SGD([b], lr=.1, weight_decay=.01, momentum=momentum, nesterov=nesterov, foreach=False)
            for step in range(7):
                lr = .1 / (step + 1)
                x.param_groups[0]['lr'] = torch.tensor(lr)
                y.param_groups[0]['lr'] = lr
                a.grad = torch.sin(torch.arange(8, dtype=torch.float32) + step)
                b.grad = a.grad.clone()
                x.step()
                y.step()
                torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)

    def test_plan_and_budget(self):
        result = plan()
        ids = [job['id'] for job in result['jobs']]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertGreater(len(ids), 100)
        now = time.time()
        deadline = bounded_deadline({'checked_unix': now, 'termination_unix': now + 7200}, 6)
        self.assertAlmostEqual(deadline, now + 6900, places=1)
        with self.assertRaises(ValueError):
            bounded_deadline({'checked_unix': now, 'termination_unix': now + 7200}, 7)
        with self.assertRaises(RuntimeError):
            bounded_deadline({'checked_unix': now - 700, 'termination_unix': now + 7200}, 1)

    def test_exact_resume_all_optimizers(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for opt in ('adamw', 'adam', 'sgd'):
                c = replace(Config(), synthetic=True, optimizer=opt, epochs=2, max_length=8,
                            per_device_batch=8, precision='fp32', validation_per_class=4, checkpoint_steps=3)
                config = folder / (opt + '.json')
                atomic_json(config, config_dict(c))
                data = folder / 'data'
                prepare(data, c)
                args = dict(config=str(config), data=str(data), backend='cpu', deadline=math.inf)
                full, resumed = folder / (opt + '_full'), folder / (opt + '_resumed')
                worker(0, SimpleNamespace(**args, root=str(full), stop_after=0))
                worker(0, SimpleNamespace(**args, root=str(resumed), stop_after=4))
                self.assertEqual(json.loads((resumed / 'status.json').read_text())['step'], 4)
                worker(0, SimpleNamespace(**args, root=str(resumed), stop_after=0))
                def state(root):
                    generation = json.loads((root / 'latest_checkpoint.json').read_text())['generation']
                    return torch.load(root / generation / 'state.pt', weights_only=False), torch.load(root / generation / 'rng_rank0.pt', weights_only=False)
                a, ar = state(full)
                b, br = state(resumed)
                assert_tree(self, a, b)
                assert_tree(self, ar, br)
                for metric in sorted((full / 'measurements').glob('*.json')):
                    x = json.loads(metric.read_text())
                    y = json.loads((resumed / 'measurements' / metric.name).read_text())
                    self.assertEqual(x['train'], y['train'])
                    self.assertEqual(x['validation'], y['validation'])
                    self.assertEqual(x['test'], y['test'])
                    self.assertEqual(x['test']['examples'], 32)
                # Corrupted checkpoints are refused, never silently restarted.
                generation = json.loads((resumed / 'latest_checkpoint.json').read_text())['generation']
                with (resumed / generation / 'rng_rank0.pt').open('ab') as f:
                    f.write(b'corruption')
                with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                    worker(0, SimpleNamespace(**args, root=str(resumed), stop_after=0))

    def test_fit_quality_and_missing_values(self):
        rows = [dict(name='one', shape=[16, 16], alpha=1.8, included_in_mean=True,
                     fit=dict(D=.05, sigma=.1, xmin=.01, xmax=1., num_pl_spikes=8, num_evals=16, status='success')),
                dict(name='two', shape=[16, 16], alpha=2.2, included_in_mean=True,
                     fit=dict(D=.15, sigma=.2, xmin=.02, xmax=2., num_pl_spikes=4, num_evals=16, status='success')),
                dict(name='classifier', shape=[2, 16], included_in_mean=False)]
        measured = dict(step=1, epoch=1, train={'accuracy': 1.}, validation={'accuracy': .8},
                        test={'accuracy': .75}, generalization_gap_pp=20., synthetic_engineering_only=True)
        result = summarize(rows, measured)
        self.assertAlmostEqual(result['mean_D'], .1)
        self.assertEqual(result['max_D'], .15)
        self.assertAlmostEqual(result['mean_alpha'], 2.)
        self.assertEqual(rows[0]['tail_fraction'], .5)
        self.assertFalse(rows[2]['D_valid'])
        rows[0]['fit']['D'] = None
        missing = summarize(rows, measured)
        self.assertIsNone(missing['mean_D'])
        self.assertEqual(missing['valid_D_layers'], 1)
        self.assertEqual(missing['test'], measured['test'])

    def test_correlation_units(self):
        self.assertEqual(correlate([{'D': i, 'acc': 5-i} for i in range(4)], 'D', 'acc')['spearman_rho'], -1.)
        self.assertIsNone(correlate([{'D': i, 'acc': 1.} for i in range(4)], 'D', 'acc')['spearman_rho'])
        rows = [dict(protocol='same', optimizer='adamw', setting='one', seed=s, synthetic=False,
                     completed_endpoint=True, mean_D=v, test_accuracy_pct=90.+v)
                for s, v in zip((401, 402, 403), (.1, .2, .3))]
        means, excluded = seed_means(rows, AGGREGATE_METRICS)
        self.assertEqual(len(means), 1)
        self.assertAlmostEqual(means[0]['mean_D'], .2)
        self.assertAlmostEqual(means[0]['mean_D_sd'], .1)
        self.assertFalse(excluded)
        self.assertFalse(seed_means(rows[:2], AGGREGATE_METRICS)[0])
        self.assertFalse(seed_means(rows + [rows[0]], AGGREGATE_METRICS)[0])

    def test_test_labels_do_not_change_training(self):
        from common import file_hash
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            c = replace(Config(), synthetic=True, epochs=1, max_length=8, per_device_batch=8,
                        precision='fp32', validation_per_class=4)
            config = folder / 'config.json'
            atomic_json(config, config_dict(c))
            data = folder / 'data'
            prepare(data, c)
            train_sha = file_hash(data / 'prepared.json')
            args = dict(config=str(config), data=str(data), backend='cpu', deadline=math.inf, stop_after=0)
            a, b = folder / 'normal', folder / 'changed_test'
            worker(0, SimpleNamespace(**args, root=str(a)))
            changed = dict(np.load(data / 'test_tokens.npz'))
            changed['labels'] = 1 - changed['labels']
            np.savez(data / 'test_tokens.npz', **changed)
            manifest = json.loads((data / 'test_prepared.json').read_text())
            manifest['sha256'] = file_hash(data / 'test_tokens.npz')
            atomic_json(data / 'test_prepared.json', manifest)
            worker(0, SimpleNamespace(**args, root=str(b)))
            self.assertEqual(file_hash(data / 'prepared.json'), train_sha)
            def payload(root):
                gen = json.loads((root / 'latest_checkpoint.json').read_text())['generation']
                return torch.load(root / gen / 'state.pt', weights_only=False), torch.load(root / gen / 'rng_rank0.pt', weights_only=False)
            assert_tree(self, payload(a), payload(b))
            for p in (a / 'measurements').glob('*.json'):
                x, y = json.loads(p.read_text()), json.loads((b / 'measurements' / p.name).read_text())
                self.assertEqual(x['train'], y['train'])
                self.assertEqual(x['validation'], y['validation'])


if __name__ == '__main__':
    unittest.main()
