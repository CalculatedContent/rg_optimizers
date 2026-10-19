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
                # Corrupted checkpoints are refused, never silently restarted.
                generation = json.loads((resumed / 'latest_checkpoint.json').read_text())['generation']
                with (resumed / generation / 'rng_rank0.pt').open('ab') as f:
                    f.write(b'corruption')
                with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                    worker(0, SimpleNamespace(**args, root=str(resumed), stop_after=0))


if __name__ == '__main__':
    unittest.main()
