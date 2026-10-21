"""Run on the intended host before launching the full comparison."""
import copy
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

if __name__=='__main__':unittest.main()
