"""Engineering checks on tiny random models; never evidence about alpha < 2."""
from dataclasses import asdict, replace
import copy
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import unittest
import numpy as np
import torch
from common import Config, atomic_json, batch_indices, masked_batch, build_model, encoder_matrices
from prepare import synthetic
from spectra import summarize
from train import LossModel

HERE = Path(__file__).parent


def run_job(root, name, c, world=1, resume=False, stop=None):
    config = root / (name + '.json')
    atomic_json(config, asdict(c))
    cmd = [sys.executable]
    if world > 1:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        cmd += ['-m', 'torch.distributed.run', '--master-addr=127.0.0.1',
                f'--master-port={port}', '--nnodes=1', f'--nproc-per-node={world}']
    cmd += [str(HERE / 'train.py'), '--backend=cpu', f'--expected-world={world}',
            '--config', str(config), '--data', str(root / 'data'), '--output', str(root / name)]
    if resume:
        cmd += ['--resume']
    if stop is not None:
        cmd += ['--stop-after', str(stop)]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    latest = json.loads((root / name / 'latest_checkpoint.json').read_text())
    return torch.load(root / name / latest['path'] / 'state.pt', weights_only=False)


class ProtocolTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(Config(), synthetic=True, sequence_length=16, per_device_batch=4,
                       steps=4, warmup_steps=1, measure_steps=2, checkpoint_steps=2,
                       train_eval_sequences=11, precision='fp32', compile=False, **kwargs)

    def test_eight_shards_exactly_once(self):
        order = np.arange(37)
        seen = []
        for j in range(2):
            for rank in range(8):
                idx, valid = batch_indices(order, j, rank, 8, 4)
                self.assertEqual(len(idx), 4)
                seen.extend(idx[valid])
        self.assertEqual(sorted(seen), list(range(37)))

    def test_masking_fixed_per_example_no_special_or_padding_targets(self):
        with tempfile.TemporaryDirectory() as d:
            meta = synthetic(d)
            x = np.memmap(Path(d) / 'train.bin', dtype=np.int32, shape=(19, 16))
            c = self.config()
            ids, labels = masked_batch(x, [2, 3], [True, False], meta, c, 992)
            solo, one = masked_batch(x, [2], [True], meta, c, 992)
            torch.testing.assert_close(ids[0], solo[0])
            torch.testing.assert_close(labels[0], one[0])
            self.assertTrue((labels[1] == -100).all())
            self.assertTrue((labels[:, [0, -1]] == -100).all())

    def test_missing_fit_does_not_bias_mean(self):
        rows = [dict(valid=True, alpha=1.7, D=.04), dict(valid=False)]
        self.assertIsNone(summarize(rows, 2)['mean_alpha'])
        self.assertFalse(summarize(rows, 2)['all_fits_valid'])

    def test_full_architecture_and_random_init(self):
        torch.manual_seed(401)
        model = build_model(Config(), 30522)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 66985530)
        self.assertEqual(len(encoder_matrices(model)), 36)
        self.assertEqual(model.distilbert.transformer.layer[0].attention.q_lin.weight.shape, (768, 768))
        self.assertIs(model.vocab_projector.weight, model.distilbert.embeddings.word_embeddings.weight)

    def test_resume_matches_uninterrupted_adamw(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            synthetic(root / 'data')
            c = self.config()
            full = run_job(root, 'full', c)
            run_job(root, 'resume', c, stop=2)
            resumed = run_job(root, 'resume', c, resume=True)
            self.assertEqual(full['step'], resumed['step'])
            for name in full['model']:
                torch.testing.assert_close(full['model'][name], resumed['model'][name], rtol=0, atol=0)
            for key, state in full['optimizer']['state'].items():
                for field, value in state.items():
                    torch.testing.assert_close(value, resumed['optimizer']['state'][key][field], rtol=0, atol=0)
            for step in (2, 4):
                a = json.loads((root / 'full/measurements' / f'step_{step:08d}.json').read_text())
                b = json.loads((root / 'resume/measurements' / f'step_{step:08d}.json').read_text())
                for split in ('train', 'validation', 'test'):
                    self.assertEqual(a[split], b[split])
            # Corrupt the checkpoint and verify that resume refuses it.
            latest = json.loads((root / 'resume/latest_checkpoint.json').read_text())
            with (root / 'resume' / latest['path'] / 'state.pt').open('ab') as f:
                f.write(b'corruption')
            with self.assertRaisesRegex(AssertionError, 'Checkpoint hash mismatch'):
                run_job(root, 'resume', c, resume=True)

    def test_eight_shard_gradients_match_serial_token_mean(self):
        # In-process gradient oracle also runs where networking is restricted.
        with tempfile.TemporaryDirectory() as d:
            meta = synthetic(d)
            tokens = np.memmap(Path(d) / 'train.bin', dtype=np.int32, shape=(19, 16))
            c = self.config(dropout=0, attention_dropout=0)
            torch.manual_seed(c.seed)
            serial = LossModel(build_model(c, meta['vocab_size']))
            sharded = copy.deepcopy(serial)
            ids, labels = masked_batch(tokens, np.arange(19), np.ones(19, dtype=bool), meta, c, 901)
            loss, _, count = serial(ids, labels)
            (loss / count).backward()
            gradients = {n: torch.zeros_like(p) for n, p in sharded.named_parameters()}
            for rank in range(8):
                idx, valid = batch_indices(np.arange(19), 0, rank, 8, 4)
                x, y = masked_batch(tokens, idx, valid, meta, c, 901)
                sharded.zero_grad(set_to_none=True)
                rank_loss, _, _ = sharded(x, y)
                (rank_loss * 8 / count).backward()
                for n, p in sharded.named_parameters():
                    gradients[n] += p.grad / 8
            for n, p in serial.named_parameters():
                torch.testing.assert_close(gradients[n], p.grad, rtol=1e-4, atol=2e-6)

    @unittest.skipUnless(os.environ.get('RG_RUN_DISTRIBUTED_TESTS') == '1',
                         'Opt-in real Gloo test requires socket/interface permissions')
    def test_distributed_matches_global_token_weighted_update(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            synthetic(root / 'data')
            c = self.config(optimizer='sgd', momentum=0, dropout=0, attention_dropout=0)
            serial = run_job(root, 'serial', replace(c, per_device_batch=8))
            distributed = run_job(root, 'distributed', c, world=2)
            # Includes the uneven final batch (19 examples, global batch 8).
            for name in serial['model']:
                torch.testing.assert_close(serial['model'][name], distributed['model'][name], rtol=1e-5, atol=2e-7)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
