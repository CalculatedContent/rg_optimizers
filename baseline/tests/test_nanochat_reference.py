import tempfile
import unittest
from pathlib import Path

from rg_baselines.nanochat_reference import (
    NanoChatD12Config,
    NanoChatMacConfig,
    _install_seed_patch,
    collect_metrics,
    find_resume_step,
    resolve_profile,
    training_command,
)


class NanoChatReferenceTests(unittest.TestCase):
    def _checkout(self, root: Path) -> Path:
        checkout = root / "checkout"
        (checkout / ".venv" / "bin").mkdir(parents=True)
        (checkout / ".venv" / "bin" / "python").write_text("")
        (checkout / ".venv" / "bin" / "torchrun").write_text("")
        return checkout

    def test_auto_profile_separates_canonical_and_mac_baselines(self):
        self.assertIsInstance(resolve_profile("auto", device_type="cuda"), NanoChatD12Config)
        self.assertIsInstance(resolve_profile("auto", device_type="mps"), NanoChatMacConfig)
        self.assertEqual(resolve_profile("mac", device_type="mps").window_pattern, "L")
        self.assertEqual(resolve_profile("d12", device_type="cuda").depth, 12)

    def test_mps_command_is_single_process_and_uses_native_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = self._checkout(Path(temporary))
            config = NanoChatMacConfig()
            command = training_command(
                checkout,
                config,
                seed=17,
                device_type="mps",
                nproc_per_node=1,
            )
            self.assertEqual(command[0], str(checkout / ".venv" / "bin" / "python"))
            joined = " ".join(command)
            self.assertIn("--device-type=mps", joined)
            self.assertIn("--depth=4", joined)
            self.assertIn("--window-pattern=L", joined)
            self.assertIn("--total-batch-size=32768", joined)
            with self.assertRaisesRegex(ValueError, "one process"):
                training_command(
                    checkout,
                    config,
                    seed=17,
                    device_type="mps",
                    nproc_per_node=2,
                )

    def test_cuda_d12_command_uses_torchrun_and_resume_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = self._checkout(Path(temporary))
            command = training_command(
                checkout,
                NanoChatD12Config(),
                seed=29,
                device_type="cuda",
                nproc_per_node=8,
                resume_from_step=250,
            )
            self.assertEqual(command[0], str(checkout / ".venv" / "bin" / "torchrun"))
            joined = " ".join(command)
            self.assertIn("--nproc_per_node=8", joined)
            self.assertIn("--depth=12", joined)
            self.assertIn("--resume-from-step=250", joined)

    def test_resume_requires_all_optimizer_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = NanoChatD12Config()
            directory = (
                root
                / "base_checkpoints"
                / "rg_d12_seed17"
            )
            directory.mkdir(parents=True)
            (directory / "model_000250.pt").write_bytes(b"model")
            (directory / "meta_000250.json").write_text("{}")
            for rank in range(7):
                (directory / f"optim_000250_rank{rank}.pt").write_bytes(b"optim")
            self.assertIsNone(
                find_resume_step(
                    root,
                    config=config,
                    seed=17,
                    nproc_per_node=8,
                )
            )
            (directory / "optim_000250_rank7.pt").write_bytes(b"optim")
            self.assertEqual(
                find_resume_step(
                    root,
                    config=config,
                    seed=17,
                    nproc_per_node=8,
                ),
                250,
            )

    def test_seed_patch_changes_only_global_seed_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            common = checkout / "nanochat" / "common.py"
            common.parent.mkdir(parents=True)
            common.write_text(
                "import os\nimport torch\n"
                "def compute_init(device_type):\n"
                "    torch.manual_seed(42)\n"
                "    if device_type == \"cuda\":\n"
                "        torch.cuda.manual_seed(42)\n"
            )
            _install_seed_patch(checkout)
            text = common.read_text()
            self.assertIn("NANOCHAT_SEED", text)
            self.assertIn("torch.mps.manual_seed(seed)", text)

    def test_resumed_log_is_deduplicated_by_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.log"
            path.write_text(
                "step 00010/00100 (10.0%) | loss: 3.0 | lrm: 1.0 | "
                "dt: 1ms | tok/sec: 1,000 | total time: 1.0m\n"
                "Step 00010 | Validation bpb: 2.0\n"
                "# RESUME FROM STEP 10\n"
                "step 00010/00100 (10.0%) | loss: 2.9 | lrm: 1.0 | "
                "dt: 1ms | tok/sec: 1,100 | total time: 1.1m\n"
                "step 00011/00100 (11.0%) | loss: 2.8 | lrm: 1.0 | "
                "dt: 1ms | tok/sec: 1,200 | total time: 1.2m\n"
            )
            metrics = collect_metrics([(17, path)], profile_name="mac_d4")
            self.assertEqual(metrics["step"].tolist(), [10, 11])
            self.assertAlmostEqual(float(metrics.iloc[0]["train_loss"]), 2.9)
            self.assertAlmostEqual(float(metrics.iloc[0]["validation_bpb"]), 2.0)


if __name__ == "__main__":
    unittest.main()
