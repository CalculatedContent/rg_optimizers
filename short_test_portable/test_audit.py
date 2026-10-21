"""CPU audit of guards and worker behavior; mocked checks do not validate hardware."""
import copy
import importlib.metadata
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
import comparison as c
import runtime
import tpu_parallel as parallel


def tiny_design():
    return dict(execution=runtime.environment(),source_sha256=c.hashes(),
        versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','weightwatcher','powerlaw']},
        calibration_seed=2027,confirmation_seeds=[1340,1341],candidates=c.candidates(),
        screen_steps=2,finalists_per_optimizer=2,max_steps=3,warmup_steps=100,
        cosine_end=17000,floor_fraction=.1,eval_every=1,checkpoint_every=2,grad_clip=1.,
        test_policy='Synthetic CPU integration test, not scientific results')


def tiny_data(_):
    return {k:(np.arange(n,dtype=np.int64)%65) for k,n in [('train',130),('validation',80),('test',90)]}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.device=patch.object(runtime,'DEVICE','cpu');self.device.start()
        c.STOP=False
        torch.set_num_threads(1)
    def tearDown(self):
        self.device.stop();c.STOP=False

    def test_nested_checkpoints_are_independent_cpu_snapshots(self):
        source=torch.tensor([1.],requires_grad=True)
        saved=runtime.cpu_tree({'x':[source, (source,)]})
        with torch.no_grad():source.add_(4)
        self.assertEqual(saved['x'][0].item(),1.)
        self.assertFalse(saved['x'][0].requires_grad)
        self.assertEqual(saved['x'][1][0].device.type,'cpu')

    def test_identical_initialization_and_minibatch_for_paired_optimizers(self):
        a,_,ra=c.setup(1340,c.candidates()[0]);s,_,rs=c.setup(1340,c.candidates()[1])
        for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,s.state_dict()[key]))
        data=tiny_data(None)['train']
        for _ in range(3):
            for x,y in zip(c.base.batch(data,ra),c.base.batch(data,rs)):self.assertTrue(torch.equal(x,y))

    def test_manifest_rejects_source_and_environment_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=tiny_design();c.write(root/'design.json',d);c.verify(root)
            for key in ['source_sha256','execution','versions']:
                bad=copy.deepcopy(d);bad[key]={'changed':True};c.write(root/'design.json',bad)
                with self.assertRaises((AssertionError,importlib.metadata.PackageNotFoundError)):c.verify(root)

    def test_frozen_recipe_survives_json_tuple_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'recipes.json'
            recipe=dict(ranking=(1.2,1.3,'candidate'))
            c.freeze(path,recipe);c.freeze(path,recipe)
            with self.assertRaises(AssertionError):c.freeze(path,dict(ranking=(1.1,1.3,'candidate')))

    def test_duplicate_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            first=c.lock(Path(folder)/'run.lock')
            try:
                with self.assertRaises(BlockingIOError):c.lock(Path(folder)/'run.lock')
            finally:first.close()

    def test_worker_final_checkpoint_and_resume(self):
        for recipe in c.candidates()[:2]:
            with self.subTest(optimizer=recipe['optimizer']),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);out=root/'trial';out.mkdir()
                c.write(root/'design.json',tiny_design())
                c.write(out/'job.json',dict(phase='calibration',recipe=recipe,seed=1340))
                with patch.object(c.base,'load_data',side_effect=tiny_data),patch.object(c.base,'sampled_loss',side_effect=lambda m,d:c.base.full_eval(m,d)['loss_nats_per_character']):
                    self.assertEqual(c.worker(root,Path('unused'),out/'job.json',2),0)
                    self.assertEqual(c.worker(root,Path('unused'),out/'job.json',3),0)
                    checkpoint=torch.load(out/'last.pt',weights_only=False)
                    self.assertEqual(checkpoint['step'],3)
                    self.assertEqual(c.read(out/'summary_3.json')['checkpoint_sha256'],c.sha(out/'last.pt'))
                    before=c.sha(out/'last.pt')
                    self.assertEqual(c.worker(root,Path('unused'),out/'job.json',3),0)
                    self.assertEqual(before,c.sha(out/'last.pt'))
                self.assertFalse((out/'result.json').exists())

    def test_launcher_uses_all_topology_processes_and_limits_active_trials(self):
        fake=types.SimpleNamespace(distributed=types.SimpleNamespace(set_tpu_launch_env=lambda:None))
        args=types.SimpleNamespace(cores=4,output=Path('/tmp/out'),data=Path('/tmp/input'),preflight=True)
        with patch.dict('sys.modules',{'torch_tpu':types.ModuleType('torch_tpu')}),patch.object(torch,'tpu',fake,create=True),patch.dict(os.environ,{'WORLD_SIZE':'8','NNODES':'1','NODE_RANK':'0'},clear=True),patch.object(parallel.subprocess,'call',return_value=0) as call:
            self.assertEqual(parallel.launch(args),0)
            command=call.call_args.args[0]
            self.assertIn('--nproc_per_node=8',command)
            self.assertIn('--preflight',command)
            self.assertEqual(call.call_args.kwargs['env']['SHORT_GPT_DEVICE'],'tpu')
            args.cores=9
            with self.assertRaises(RuntimeError):parallel.launch(args)
            args.cores=4;os.environ['NNODES']='2'
            with self.assertRaises(RuntimeError):parallel.launch(args)

if __name__=='__main__':unittest.main(verbosity=2)
