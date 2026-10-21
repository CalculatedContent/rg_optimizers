"""Run on the intended host before launching the full comparison."""
import argparse
import json
import os
import time
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import comparison as c
import runtime

class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime.initialize()
        torch.set_num_threads(1)

    def test_unavailable_accelerators_fail_without_fallback(self):
        for name, available in [('cuda', 'torch.cuda.is_available'), ('mps', 'torch.backends.mps.is_available')]:
            with self.subTest(device=name), patch.object(runtime, 'DEVICE', name), patch(available, return_value=False):
                with self.assertRaises(RuntimeError): runtime.initialize()

    def test_parallel_assignment(self):
        from tpu_parallel import assigned
        jobs=list(range(12))
        for cores in range(1,9):
            parts=[assigned(jobs,rank,cores) for rank in range(8)]
            self.assertEqual(sorted(x for part in parts for x in part),jobs)
            self.assertTrue(all(not part for part in parts[cores:]))
        with self.assertRaises(ValueError):assigned(jobs,0,9)

    def test_timing_dispatch(self):
        for name, api in [('cuda', 'torch.cuda.synchronize'), ('mps', 'torch.mps.synchronize')]:
            with self.subTest(device=name), patch.object(runtime, 'DEVICE', name), patch(api) as sync:
                runtime.synchronize()
                sync.assert_called_once_with()

    def test_memory_admission_without_linux_cgroups(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            self.assertTrue(runtime.admission(p))
            (p/'memory.max').write_text(str(2 * 2**30))
            (p/'memory.stat').write_text('anon 2000000000\nkernel 100000000\n')
            self.assertFalse(runtime.admission(p))

    def test_both_optimizers_checkpoint_resume_and_full_tail(self):
        data = np.arange(1601, dtype=np.int64) % 65
        design = dict(warmup_steps=100, cosine_end=17000, floor_fraction=.1, grad_clip=1.)
        for recipe in c.candidates()[:2]:
            with self.subTest(optimizer=recipe['optimizer']):
                model, opt, rng = c.setup(1340, recipe)
                self.assertIs(model.lm_head.weight, model.transformer.wte.weight)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 804096)
                c.update(model,opt,rng,data,0,recipe,design)
                state = runtime.cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),**c.base.rng_state(rng)))
                with tempfile.TemporaryDirectory() as folder:
                    path=Path(folder)/'last.pt';torch.save(state,path)
                    state=torch.load(path,map_location='cpu',weights_only=False)
                c.update(model,opt,rng,data,1,recipe,design)
                expected=runtime.cpu_tree(model.state_dict())
                resumed, opt2, rng2=c.setup(1340,recipe)
                resumed.load_state_dict(state['model']);opt2.load_state_dict(state['optimizer']);c.base.restore_rng(state,rng2)
                c.update(resumed,opt2,rng2,data,1,recipe,design)
                runtime.synchronize()
                for key, value in expected.items():
                    actual=resumed.state_dict()[key].detach().cpu()
                    if runtime.DEVICE=='cpu':self.assertTrue(torch.equal(value,actual),key)
                    else:torch.testing.assert_close(value,actual,rtol=1e-5,atol=1e-6)
                before=rng2.get_state().clone()
                metrics=c.base.full_eval(resumed,data)
                self.assertEqual(metrics['evaluated_targets'],1600)
                self.assertTrue(np.isfinite(metrics['loss_nats_per_character']))
                self.assertTrue(torch.equal(before,rng2.get_state()))
                self.assertTrue(resumed.training)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu','mps','cuda','tpu'], default=os.environ.get('SHORT_GPT_DEVICE','cpu'))
    args=parser.parse_args()
    runtime.DEVICE=args.device
    os.environ['SHORT_GPT_DEVICE']=args.device
    runtime.initialize()
    print(json.dumps(runtime.environment()), flush=True)
    start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PortableTests))
    runtime.synchronize()
    print(json.dumps(dict(device=args.device,preflight_seconds=time.perf_counter()-start,passed=result.wasSuccessful())),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
