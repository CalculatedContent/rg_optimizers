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
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from common import (Config, atomic_json, batch_indices, masked_batch, build_model,
                    encoder_matrices, file_hash, make_optimizer, read_config)
from cloudshell import discover_node
from prepare import synthetic
from spectra import summarize, watch
from supervisor import terminate
from train import LossModel, Runtime, checkpoint, preserve_best_validation, train

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

    def test_invalid_config_is_rejected_before_training(self):
        for change in (dict(lr=float('inf')), dict(eps=float('nan')),
                       dict(grad_clip=float('inf')), dict(weight_decay=float('inf')),
                       dict(steps=4.5), dict(seed=-1), dict(seed=2**32),
                       dict(data_seed=-1), dict(compile='false'), dict(synthetic=1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config(), **change).validate()

    def test_tpu_rng_uses_supported_getter_and_restores_all_states(self):
        rt = Runtime.__new__(Runtime)
        rt.backend = 'tpu'
        fake_tpu = SimpleNamespace(get_rng_state=Mock(side_effect=RuntimeError('Internal API')),
                                   set_rng_state=Mock())
        value = torch.tensor([12, 34], dtype=torch.uint8)
        getter = Mock(return_value=value)
        with patch.object(torch, 'tpu', fake_tpu, create=True), patch.object(
                torch.accelerator, 'random', SimpleNamespace(get_rng_state=getter), create=True):
            saved = rt.rng()
            torch.rand(8)
            np.random.random(8)
            rt.restore_rng(saved)
            getter.assert_called_once_with()
            fake_tpu.get_rng_state.assert_not_called()
            fake_tpu.set_rng_state.assert_called_once_with(value)
            torch.testing.assert_close(torch.get_rng_state(), saved['torch'], rtol=0, atol=0)
            self.assertTrue(np.array_equal(np.random.get_state()[1], saved['numpy'][1]))
        # Older supported torch releases lack the replacement getter.
        fake_tpu.get_rng_state = Mock(return_value=value)
        with patch.object(torch, 'tpu', fake_tpu, create=True), patch.object(
                torch.accelerator, 'random', SimpleNamespace(), create=True):
            torch.testing.assert_close(rt.rng()['tpu'], value, rtol=0, atol=0)
            fake_tpu.get_rng_state.assert_called_once_with()

    def test_spectral_worker_drains_measurement_published_during_last_scan(self):
        with tempfile.TemporaryDirectory() as d:
            root, seen = Path(d), []
            def scan(_):
                if (root / 'final.json').exists():
                    seen.append('final')
                else:
                    # Training publishes just after this scan's file listing.
                    (root / 'final.json').write_text('{}')
                    (root / 'TRAINING_DONE.json').write_text('{}')
            with patch('spectra.analyze', side_effect=scan) as analyze, patch('spectra.time.sleep'):
                watch(root)
            self.assertEqual(seen, ['final'])
            self.assertEqual(analyze.call_count, 2)

    def test_status_can_discover_existing_tpu_near_lease_expiry(self):
        listed = SimpleNamespace(stdout=json.dumps([
            dict(name='projects/p/locations/z/nodes/existing', state='READY',
                 acceleratorType='v5litepod-8')]))
        with patch('cloudshell.run', return_value=listed), patch(
                'cloudshell.checked_lease', side_effect=RuntimeError('Lease expired')) as lease:
            self.assertEqual(discover_node('z', require_lease=False), 'existing')
            lease.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'Expected one eligible'):
                discover_node('z')

    def test_timeout_kills_process_that_ignores_sigterm(self):
        code = ('import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                'print("ready", flush=True); time.sleep(60)')
        proc = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                                stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), 'ready')
            terminate(proc, grace=.05)
            self.assertEqual(proc.returncode, -9)
        finally:
            if proc.poll() is None:
                terminate(proc, grace=.05)
            proc.stdout.close()

    def test_unpublished_checkpoint_is_replaced_and_partial_does_not_affect_retention(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = build_model(self.config(), 128)
            opt = make_optimizer(model, self.config(), torch.device('cpu'))
            rt = Runtime('cpu', 1)
            checkpoint(root, 0, model, opt, rt, {})
            orphan = root / 'checkpoints/step_00000001'
            orphan.mkdir()
            (orphan / 'state.pt').write_bytes(b'unpublished previous replay')
            (root / 'checkpoints/step_00000003').mkdir()  # Future unpublished generation.
            (root / 'checkpoints/step_99999999.partial').mkdir()
            checkpoint(root, 1, model, opt, rt, {})
            saved = torch.load(orphan / 'state.pt', weights_only=False)
            self.assertEqual(saved['step'], 1)
            self.assertTrue((root / 'checkpoints/step_00000000').exists())
            checkpoint(root, 2, model, opt, rt, {})
            self.assertFalse((root / 'checkpoints/step_00000000').exists())
            self.assertTrue(orphan.exists())

    def test_measurements_never_precede_checkpoint_and_interrupted_measurement_recovers(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            synthetic(root / 'data')
            c = replace(self.config(), checkpoint_steps=3)
            atomic_json(root / 'config.json', asdict(c))
            args = SimpleNamespace(config=root / 'config.json', data=root / 'data',
                                   output=root / 'run', resume=False, stop_after=None,
                                   deadline=float('inf'))
            import train as training
            real_measure = training.measure
            observed, interrupted = [], False
            def checked_measure(runroot, step, *values):
                nonlocal interrupted
                pointer = json.loads((runroot / 'latest_checkpoint.json').read_text())
                self.assertEqual(pointer['step'], step)
                observed.append(step)
                if step == 2 and not interrupted:
                    interrupted = True
                    raise RuntimeError('Simulated interruption before diagnostic publication')
                return real_measure(runroot, step, *values)
            with patch('train.measure', side_effect=checked_measure):
                with self.assertRaisesRegex(RuntimeError, 'Simulated interruption'):
                    train(args, Runtime('cpu', 1))
                args.resume = True
                train(args, Runtime('cpu', 1))
            self.assertEqual(observed, [0, 2, 2, 4])
            self.assertTrue((root / 'run/measurements/step_00000002.json').exists())

    def test_training_mask_rate_does_not_change_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            meta = synthetic(d)
            tokens = np.memmap(Path(d) / 'train.bin', dtype=np.int32, shape=(19, 16))
            c = self.config()
            indices, valid = np.arange(19), np.ones(19, dtype=bool)
            higher = replace(c, mask_probability=.30)
            for salt in (991, 992, 993):
                low = masked_batch(tokens, indices, valid, meta, c, salt, evaluation=True)
                high = masked_batch(tokens, indices, valid, meta, higher, salt, evaluation=True)
                for a, b in zip(low, high):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            low = masked_batch(tokens, indices, valid, meta, c, 901)[1]
            high = masked_batch(tokens, indices, valid, meta, higher, 901)[1]
            self.assertGreater((high != -100).sum(), (low != -100).sum())

    def test_best_weights_follow_validation_only_and_survive_later_regression(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = build_model(self.config(), 128)
            metrics = dict(validation=dict(mlm_loss=3.0), test=dict(mlm_loss=4.0))
            preserve_best_validation(root, 1, model, metrics, {'run': 'test'})
            first = json.loads((root / 'best_validation.json').read_text())
            with torch.no_grad():
                model.distilbert.transformer.layer[0].attention.q_lin.weight.add_(.1)
            metrics['validation']['mlm_loss'], metrics['test']['mlm_loss'] = 3.1, 1.0
            preserve_best_validation(root, 2, model, metrics, {'run': 'test'})
            self.assertEqual(first, json.loads((root / 'best_validation.json').read_text()))
            metrics['validation']['mlm_loss'], metrics['test']['mlm_loss'] = 2.9, 5.0
            preserve_best_validation(root, 3, model, metrics, {'run': 'test'})
            best = json.loads((root / 'best_validation.json').read_text())
            self.assertEqual(best['step'], 3)
            self.assertEqual(file_hash(root / best['path']), best['sha256'])
            saved = torch.load(root / best['path'], weights_only=False)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(saved['model'][name], tensor, rtol=0, atol=0)
            self.assertEqual(len(list((root / 'best_validation').glob('*.pt'))), 1)
            preserve_best_validation(root, 4, model, metrics, {'run': 'test'})
            self.assertEqual(best, json.loads((root / 'best_validation.json').read_text()))

    def test_adamw_decay_preserves_norm_bias_and_does_not_duplicate_tied_weights(self):
        c = self.config()
        model = build_model(c, 128)
        opt = make_optimizer(model, c, torch.device('cpu'))
        parameters = [p for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        with torch.no_grad():
            for p in parameters:
                p.fill_(1.)
                p.grad = torch.zeros_like(p)
        opt.step()
        block = model.distilbert.transformer.layer[0]
        self.assertTrue(torch.equal(block.sa_layer_norm.weight, torch.ones_like(block.sa_layer_norm.weight)))
        self.assertTrue(torch.equal(block.attention.q_lin.bias, torch.ones_like(block.attention.q_lin.bias)))
        for p in (block.attention.q_lin.weight, model.vocab_projector.weight):
            torch.testing.assert_close(p, torch.full_like(p, 1 - c.lr * c.weight_decay), rtol=0, atol=0)

    def test_compiled_bf16_loss_and_gradients_match_eager(self):
        c = self.config(dropout=0, attention_dropout=0)
        torch.manual_seed(42)
        eager = LossModel(build_model(c, 128))
        compiled = torch.compile(copy.deepcopy(eager), backend='aot_eager', fullgraph=True, dynamic=False)
        ids = torch.randint(5, 128, (4, 16))
        labels = ids.clone()
        labels[:, :8], labels[-1] = -100, -100
        outputs = []
        for net in (eager, compiled):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                out = net(ids, labels)
            (out[0] / out[2]).backward()
            outputs.append(out)
        for a, b in zip(*outputs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(eager.parameters(), compiled.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)

    def test_pilot_grid_changes_only_lr_and_warmup(self):
        baseline = read_config(HERE / 'configs/initial.json')
        configs = [baseline] + [read_config(p) for p in sorted((HERE / 'configs').glob('pilot_*.json'))]
        self.assertEqual(len(configs), 6)
        self.assertEqual({(c.lr, c.warmup_steps) for c in configs},
                         {(lr, warmup) for lr in (1e-4, 3e-4, 5e-4) for warmup in (300, 1000)})
        for c in configs:
            self.assertEqual(replace(c, lr=baseline.lr, warmup_steps=baseline.warmup_steps), baseline)

    def test_full_architecture_and_random_init(self):
        torch.manual_seed(401)
        model = build_model(Config(), 30522)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 66985530)
        self.assertEqual(len(encoder_matrices(model)), 36)
        self.assertEqual(model.distilbert.transformer.layer[0].attention.q_lin.weight.shape, (768, 768))
        self.assertIs(model.vocab_projector.weight, model.distilbert.embeddings.word_embeddings.weight)

    def assert_resume_matches(self, c):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            synthetic(root / 'data')
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

    def test_resume_matches_uninterrupted_for_all_optimizers(self):
        for name in ('adamw', 'adam', 'sgd'):
            with self.subTest(optimizer=name):
                self.assert_resume_matches(self.config(optimizer=name))

    def test_compiled_bf16_resume_matches_uninterrupted(self):
        self.assert_resume_matches(replace(self.config(), compile=True, precision='bf16'))

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
