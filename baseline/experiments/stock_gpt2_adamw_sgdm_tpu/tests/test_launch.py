"""Launch boundaries: no accidental start, overwrite, cloud write, or false success."""
import contextlib
import io
import json
from pathlib import Path
import shlex
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
from .. import launch
from .. import preflight


class LaunchBoundaries(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def args(self, *extra):
        return ["--config", str(HERE / "configs/smoke.json"),
                "--data-dir", str(self.root / "data"), "--run-dir", str(self.root / "run"),
                "--device", "cpu", "--skip-spectral", "--min-free-gib", "0", *extra]

    def spec(self):
        return launch.new_spec(launch.parser().parse_args(self.args()))

    def test_default_plan_never_starts_or_writes(self):
        with patch.object(launch.subprocess, "Popen", side_effect=AssertionError("started")), \
             patch.object(launch, "run_preflight", side_effect=AssertionError("probed")), \
             patch.object(launch, "gcloud", side_effect=AssertionError("cloud")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launch.main(self.args()), 0)
        report = json.loads(output.getvalue())
        self.assertFalse(report["executes"])
        self.assertFalse((self.root / "run").exists())
        self.assertIsNone(report["commands"]["tracking"])

    def test_existing_run_rejected_without_resume(self):
        spec = self.spec()
        output = Path(spec["run_dir"])
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("historical result")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            launch.ensure_run_available(spec)
        self.assertEqual(sentinel.read_text(), "historical result")

    def test_venv_interpreter_symlink_is_preserved_for_children(self):
        interpreter = self.root / "venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        with patch.object(launch.sys, "executable", str(interpreter)):
            spec = self.spec()
            launch.ensure_current_interpreter(spec)
        self.assertEqual(spec["python"], str(interpreter))
        self.assertEqual(launch.commands(spec)["training"][0], str(interpreter))
        self.assertNotEqual(spec["python"], str(interpreter.resolve()))

    def test_local_interpreter_mismatch_fails_before_preflight_or_start(self):
        for action in ("preflight", "start"):
            with self.subTest(action=action), \
                 patch.object(launch, "run_preflight", side_effect=AssertionError("wrong environment checked")), \
                 patch.object(launch.subprocess, "Popen", side_effect=AssertionError("started")):
                with self.assertRaisesRegex(RuntimeError, "same Python"):
                    launch.main([action, *self.args(), "--python", str(self.root / "other-venv/bin/python")])
        self.assertFalse((self.root / "run").exists())

    def test_resume_requires_checkpoint_identity(self):
        spec = {**self.spec(), "resume": True}
        root = Path(spec["run_dir"])
        (root / "checkpoints").mkdir(parents=True)
        (root / "checkpoints/latest.pt").write_bytes(b"not loaded in boundary test")
        (root / "checkpoints/latest.json").write_text(json.dumps({"config_sha256": "different"}))
        with self.assertRaisesRegex(RuntimeError, "checkpoint metadata"):
            launch.ensure_run_available(spec)

    def test_preflight_failure_leaves_no_run(self):
        spec = self.spec()
        with patch.object(launch, "run_preflight", side_effect=RuntimeError("not ready")), \
             patch.object(launch.subprocess, "Popen", side_effect=AssertionError("started")):
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                launch.start_local(spec)
        self.assertFalse(Path(spec["run_dir"]).exists())

    def test_gcs_prefix_rejects_bucket_root_and_wildcards(self):
        for value in ("gs://bucket", "gs://bucket/a/*", "gs://bucket/../other", "gs://bucket/a//b"):
            with self.assertRaises(ValueError):
                launch.validate_gcs_uri(value)
        self.assertEqual(launch.validate_gcs_uri("gs://bucket/unique-run/"), "gs://bucket/unique-run")

    def test_permission_failure_never_attempts_cloud_write(self):
        result = SimpleNamespace(returncode=1, stdout="", stderr="403 Permission denied")
        with patch.object(launch.subprocess, "run", return_value=result), \
             patch.object(launch, "gcloud", side_effect=AssertionError("write attempted")):
            with self.assertRaisesRegex(RuntimeError, "Cannot inspect"):
                launch.gcs_probe("gs://bucket/new-run", self.spec())

    def test_existing_cloud_prefix_never_overwritten_by_new_run(self):
        result = SimpleNamespace(returncode=0, stdout="gs://bucket/old-run/LAUNCH.json\n", stderr="")
        with patch.object(launch.subprocess, "run", return_value=result), \
             patch.object(launch, "gcloud", side_effect=AssertionError("write attempted")):
            with self.assertRaisesRegex(RuntimeError, "already contains"):
                launch.gcs_probe("gs://bucket/old-run", self.spec())

    def test_explicit_probe_roundtrips_and_removes_only_its_object(self):
        calls = []
        payload = {}
        def fake_gcloud(command, **_kwargs):
            calls.append(command)
            if command[1] == "cp":
                payload["text"] = Path(command[2]).read_text()
            return payload["text"] if command[1] == "cat" else ""
        missing = SimpleNamespace(returncode=1, stdout="", stderr="One or more URLs matched no objects.")
        with patch.object(launch.subprocess, "run", return_value=missing), \
             patch.object(launch, "gcloud", side_effect=fake_gcloud):
            outcome = launch.gcs_probe("gs://bucket/new-run", self.spec())
        self.assertEqual(outcome["status"], "write_read_delete_verified")
        self.assertEqual([x[1] for x in calls], ["cp", "cat", "rm"])
        self.assertEqual(calls[0][-1], calls[1][-1])
        self.assertEqual(calls[1][-1], calls[2][-1])
        self.assertIn("/__write_probe_", calls[2][-1])

    def test_remote_command_quotes_paths_and_uses_only_worker_zero(self):
        values = self.args() + ["--node", "provided-node", "--project", "provided-project",
                               "--zone", "provided-zone", "--remote-repo", "/mnt/user checkout",
                               "--python", "/mnt/venv/bin/python"]
        args = launch.parser().parse_args(values)
        command = launch.remote_command(args)
        self.assertIn("--worker=0", command)
        remote = shlex.split(command[-1])
        self.assertEqual(remote[0], "/mnt/venv/bin/python")
        self.assertEqual(remote[1], "/mnt/user checkout/baseline/experiments/stock_gpt2_adamw_sgdm_tpu/launch.py")
        self.assertNotIn("create", command)
        self.assertNotIn("delete", command)

    def test_backup_excludes_pending_weights_and_caches(self):
        names = ("checkpoints/latest.pt", "checkpoints/latest.json", "checkpoints/first_target.pt", "checkpoints/first_target.json", "checkpoints/step1.pt",
                 "tracking/snapshots/0001.pt", "tracking/snapshots/0001.json",
                 "tracking/measurements/0001.json", "xla-cache/cache.json", "metrics.jsonl", "TRAINING_DONE",
                 "reports/curves.png", "reports/curves.pdf", "unrelated.png")
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        actual = {str(p.relative_to(self.root)) for p in launch.selected_backup_files(self.root)}
        self.assertEqual(actual, {"checkpoints/latest.pt", "checkpoints/latest.json", "checkpoints/first_target.pt", "checkpoints/first_target.json",
                                  "tracking/measurements/0001.json", "metrics.jsonl", "TRAINING_DONE",
                                  "reports/curves.png", "reports/curves.pdf"})

    def test_staged_checkpoint_pair_must_have_matching_content_and_config(self):
        source = self.root / "run/checkpoints/latest.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"old checkpoint generation")
        receipt = {"file": "latest.pt", "config_sha256": "config", "sha256": preflight.sha256(source)}
        source.with_suffix(".json").write_text(json.dumps(receipt))
        staging = self.root / "staging-good"
        staging.mkdir()
        report = launch.stage_backup(source.parents[1], staging, "config")
        self.assertEqual(report["files_staged"], 2)
        self.assertFalse(report["skipped_checkpoints"])
        self.assertEqual(preflight.sha256(staging / "checkpoints/latest.pt"), receipt["sha256"])
        # Model replacement can precede sidecar replacement. Never upload either
        # half of this mismatched generation over a prior remote checkpoint.
        replacement = source.with_name("replacement.tmp")
        replacement.write_bytes(b"new checkpoint generation")
        replacement.replace(source)
        racing = self.root / "staging-race"
        racing.mkdir()
        report = launch.stage_backup(source.parents[1], racing, "config")
        self.assertEqual(report["files_staged"], 0)
        self.assertEqual(report["skipped_checkpoints"][0]["checkpoint"], "latest")
        self.assertFalse((racing / "checkpoints/latest.pt").exists())
        self.assertFalse((racing / "checkpoints/latest.json").exists())
        # The already-staged inode remains the old, complete checkpoint.
        self.assertEqual(preflight.sha256(staging / "checkpoints/latest.pt"), receipt["sha256"])

    def test_first_target_checkpoint_uses_verified_pair_staging(self):
        source = self.root / "run/checkpoints/first_target.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"first official target checkpoint")
        receipt = {"file": source.name, "config_sha256": "config", "sha256": preflight.sha256(source)}
        source.with_suffix(".json").write_text(json.dumps(receipt))
        staging = self.root / "first-target-staging"
        staging.mkdir()
        report = launch.stage_backup(source.parents[1], staging, "config")
        self.assertEqual(report["files_staged"], 2)
        self.assertEqual((staging / "checkpoints/first_target.pt").read_bytes(), source.read_bytes())
        self.assertEqual(json.loads((staging / "checkpoints/first_target.json").read_text()), receipt)

    def test_backup_failure_does_not_rewrite_training_outcome(self):
        trained = {"status": "budget_complete", "target_met": False}
        status = launch.final_status(0, 0, {"status": "failed", "error": "403"}, trained)
        self.assertTrue(status["training_process_succeeded"])
        self.assertEqual(status["training_status"], trained)
        self.assertEqual(status["backup_status"]["status"], "failed")

    def test_cpu_smoke_preflight_preserves_identity_and_needs_no_corpus(self):
        config = preflight.load_config(HERE / "configs/smoke.json")
        with patch.object(preflight, "check_versions", return_value={"test": "version check mocked"}):
            proof = preflight.run_preflight(config_path=HERE / "configs/smoke.json",
                                            data_dir=self.root / "data", run_dir=self.root / "run",
                                            device="cpu", skip_spectral=True, min_free_gib=0)
        self.assertEqual(proof["config_sha256"], config.config_sha256)
        self.assertEqual(proof["data_dir"], str((self.root / "data").resolve()))
        self.assertEqual(proof["device"], "cpu")
        self.assertEqual(proof["status"], "passed")
        self.assertFalse((self.root / "run").exists())
        self.assertEqual(proof["data"]["files"], [])

    def test_data_hash_mismatch_is_fatal_and_does_not_replace_shard(self):
        files = {}
        for name in ("fineweb_train_000001.bin", "fineweb_val_000000.bin"):
            header = [20240520, 1, 10] + [0] * 253
            path = self.root / name
            path.write_bytes(struct.pack("<256i", *header) + struct.pack("<10H", *range(10)))
            files[name] = {"size": path.stat().st_size, "sha256": preflight.sha256(path)}
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"repo": "kjj0/fineweb10B-gpt2",
                                        "revision": preflight.DATA_REVISION, "files": files}))
        config = SimpleNamespace(mode="production", budget_tokens=8, validation_offset=0, validation_tokens=4)
        with patch.object(preflight, "MANIFEST", manifest):
            self.assertEqual(len(preflight.check_data(config, self.root)["files"]), 2)
            corrupted = self.root / "fineweb_train_000001.bin"
            changed = corrupted.read_bytes()[:-1] + b"\xff"
            corrupted.write_bytes(changed)
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                preflight.check_data(config, self.root)
            self.assertEqual(corrupted.read_bytes(), changed)


if __name__ == "__main__":
    unittest.main()
