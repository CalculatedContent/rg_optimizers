from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rg_baselines.tangent_rg.capture import (
    CAPTURE_SCHEMA_VERSION,
    format_capture_name,
    load_step_capture,
    parse_capture_name,
)
from rg_baselines.tangent_rg.checkpoints import (
    ensure_tail_checkpoint_cache,
    finalize_tail_checkpoint_cache,
    format_analysis_checkpoint_name,
    inspect_full_checkpoint,
    load_analysis_checkpoint,
    load_verified_tail_checkpoint_refs,
    parse_analysis_checkpoint_name,
    quarantine_tail_checkpoint_cache_after_boundary,
    save_analysis_checkpoint,
    save_full_checkpoint,
    save_tail_checkpoint,
    verify_tail_checkpoint_cache,
    verify_tail_checkpoint_cache_prefix,
)
from rg_baselines.tangent_rg.config import TangentRGConfig, config_from_mapping
from rg_baselines.tangent_rg.muonclip import (
    MuonClipRMSWithAuxAdamW,
    matrix_update_components,
)
from rg_baselines.tangent_rg.protocol import (
    build_analysis_plan,
    make_tail_checkpoint_layout,
    tail_checkpoint_epochs,
    validate_disjoint_checkpoint_layouts,
)
from rg_baselines.tangent_rg.protocol import make_run_layout


class StrictConfigurationContractTests(unittest.TestCase):
    def test_unknown_keys_are_rejected_at_every_schema_level(self):
        cases = (
            ({"protcol": {}}, "configuration root"),
            ({"protocol": {"suite_nam": "typo"}}, "protocol"),
            ({"training": {"epocs": 10}}, "training"),
            ({"analysis": {"log_ponts": 10}}, "analysis"),
            ({"runtime": {"devce": "cpu"}}, "runtime"),
            ({"optimizers": {"muonn": {}}}, "optimizers"),
            (
                {"optimizers": {"adamw": {"learnin_rate": 1.0e-3}}},
                "optimizers.adamw",
            ),
            (
                {"optimizers": {"muon": {"newton_schul_steps": 5}}},
                "optimizers.muon",
            ),
            (
                {"optimizers": {"muonclip_rms": {"rms_sclae": 0.2}}},
                "optimizers.muonclip_rms",
            ),
        )
        for payload, section in cases:
            with self.subTest(section=section):
                with self.assertRaisesRegex(ValueError, section.replace(".", r"\.")):
                    config_from_mapping(payload)

    def test_empty_mapping_resolves_and_validates_declared_defaults(self):
        config = config_from_mapping({})
        self.assertEqual(config.epochs, 1_000)
        self.assertEqual(config.lr_schedule_epochs, 30.0)

    def test_suite_name_must_be_one_safe_path_component(self):
        for unsafe in ("", ".", "..", "../escape", "/tmp/escape", "a/b", r"a\b"):
            with self.subTest(suite_name=unsafe):
                with self.assertRaisesRegex(ValueError, "suite_name"):
                    TangentRGConfig(suite_name=unsafe).validate()
        TangentRGConfig(suite_name="mnist_mlp3-tangent.rg_v1").validate()

    def test_tail_checkpoint_root_is_restricted_beneath_tmp(self):
        for unsafe in ("relative/cache", "/tmp", "/var/tmp/cache"):
            with self.subTest(root=unsafe):
                with self.assertRaisesRegex(ValueError, "tail_checkpoint_cache_root"):
                    TangentRGConfig(tail_checkpoint_cache_root=unsafe).validate()
        TangentRGConfig(
            tail_checkpoint_cache_root="/tmp/declared-tail-cache"
        ).validate()

    def test_runtime_mapping_accepts_declared_tail_checkpoint_root(self):
        config = config_from_mapping(
            {
                "runtime": {
                    "tail_checkpoint_cache_root": "/tmp/custom-tail-cache"
                }
            }
        )
        self.assertEqual(
            config.tail_checkpoint_cache_root,
            "/tmp/custom-tail-cache",
        )


class ScheduleAndCheckpointContractTests(unittest.TestCase):
    def test_tail_checkpoint_epochs_are_exactly_final_trained_boundaries(self):
        self.assertEqual(tail_checkpoint_epochs(2), (1, 2))
        self.assertEqual(tail_checkpoint_epochs(99), tuple(range(1, 100)))
        self.assertEqual(tail_checkpoint_epochs(100), tuple(range(1, 101)))
        self.assertEqual(tail_checkpoint_epochs(1_000), tuple(range(901, 1_001)))
        self.assertEqual(len(tail_checkpoint_epochs(10_000)), 100)
        self.assertEqual(tail_checkpoint_epochs(10_000)[0], 9_901)
        with self.assertRaisesRegex(ValueError, "positive"):
            tail_checkpoint_epochs(0)

    def test_tail_checkpoint_layout_is_namespaced_by_run_identity(self):
        config = TangentRGConfig(
            suite_name="declared-suite",
            optimizer="muonclip_rms",
            seed=2027,
            tail_checkpoint_cache_root="/tmp/declared-tail-cache",
        )
        layout = make_tail_checkpoint_layout(config)
        self.assertEqual(
            layout.root,
            Path("/tmp/declared-tail-cache/declared-suite/muonclip_rms/seed_2027"),
        )
        self.assertEqual(layout.checkpoints, layout.root / "checkpoints")

    def test_persistent_and_temporary_seed_layouts_cannot_overlap(self):
        config = TangentRGConfig(
            suite_name="overlap-test",
            optimizer="adamw",
            run_root="/tmp/shared-root",
            tail_checkpoint_cache_root="/tmp/shared-root",
        )
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_disjoint_checkpoint_layouts(
                make_run_layout(config),
                make_tail_checkpoint_layout(config),
            )

    def test_learning_rate_horizon_is_decoupled_from_long_training_horizon(self):
        config = TangentRGConfig(
            optimizer="muon",
            epochs=10_000,
            lr_schedule_epochs=30.0,
            log_analysis_points=24,
            dense_burst_anchor_epochs=(0, 10, 1_000, 9_999, 10_000),
        )
        config.validate()
        plan = build_analysis_plan(config, steps_per_epoch=7)
        self.assertEqual(plan.total_steps, 70_000)
        self.assertEqual(plan.lr_schedule_steps, 210)
        self.assertIn(10_000, plan.analysis_epochs)
        self.assertNotIn(70_001, plan.capture_completed_steps)
        self.assertTrue(all(burst.anchor_epoch < 10_000 for burst in plan.dense_bursts))

    def test_analysis_checkpoint_parser_accepts_four_and_five_digit_epochs(self):
        for epoch, step in ((1_000, 430_000), (10_000, 4_300_000)):
            name = format_analysis_checkpoint_name(epoch, step)
            self.assertEqual(parse_analysis_checkpoint_name(name), (epoch, step))
        self.assertEqual(
            parse_analysis_checkpoint_name("analysis_epoch_10000_step_4300000.pt"),
            (10_000, 4_300_000),
        )

    def test_capture_step_names_round_trip(self):
        for step in (1, 999_999_999, 1_000_000_000):
            self.assertEqual(parse_capture_name(format_capture_name(step)), step)

    def test_analysis_checkpoint_loader_rejects_fingerprint_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = torch.nn.Linear(3, 2)
            path = save_analysis_checkpoint(
                Path(temporary),
                model=model,
                epoch=10_000,
                global_step=4_300_000,
                protocol_fingerprint="declared-protocol",
                optimizer_name="muon",
                seed=1337,
            )
            payload = load_analysis_checkpoint(
                path, expected_fingerprint="declared-protocol"
            )
            self.assertEqual(payload["epoch"], 10_000)
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                load_analysis_checkpoint(path, expected_fingerprint="different")
            with torch.no_grad():
                model.weight.add_(1.0)
            with self.assertRaisesRegex(RuntimeError, "model state"):
                save_analysis_checkpoint(
                    Path(temporary),
                    model=model,
                    epoch=10_000,
                    global_step=4_300_000,
                    protocol_fingerprint="declared-protocol",
                    optimizer_name="muon",
                    seed=1337,
                )

    def test_capture_loader_rejects_fingerprint_and_filename_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / format_capture_name(123)
            torch.save(
                {
                    "schema_version": CAPTURE_SCHEMA_VERSION,
                    "capture_kind": "one_step_tangent_update",
                    "completed_step": 123,
                    "protocol_fingerprint": "declared-protocol",
                },
                path,
            )
            self.assertEqual(
                load_step_capture(
                    path, expected_fingerprint="declared-protocol"
                )["completed_step"],
                123,
            )
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                load_step_capture(path, expected_fingerprint="different")
            wrong_name = path.with_name(format_capture_name(124))
            path.replace(wrong_name)
            with self.assertRaisesRegex(RuntimeError, "filename"):
                load_step_capture(wrong_name)

    def test_full_checkpoint_inspection_enforces_role_and_final_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            generator = torch.Generator().manual_seed(12)
            path = Path(temporary) / "checkpoint_final.pt"
            save_full_checkpoint(
                path,
                config={},
                model=model,
                optimizer=optimizer,
                epoch=10_000,
                global_step=4_300_000,
                best_validation_loss=0.1,
                best_validation_epoch=9_000,
                train_generator=generator,
                protocol_fingerprint="declared-protocol",
                checkpoint_role="final",
            )
            inspected = inspect_full_checkpoint(
                path,
                expected_fingerprint="declared-protocol",
                expected_role="final",
                expected_epoch=10_000,
                expected_global_step=4_300_000,
            )
            self.assertEqual(inspected["epoch"], 10_000)
            with self.assertRaisesRegex(RuntimeError, "role"):
                inspect_full_checkpoint(path, expected_role="latest")

    def test_tail_checkpoint_cache_lifecycle_and_strict_analysis_loader(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            config = TangentRGConfig(
                suite_name="tail-cache-test",
                optimizer="muon",
                seed=1337,
                epochs=3,
                lr_schedule_epochs=3,
                tail_checkpoint_cache_root=temporary,
            )
            layout = make_tail_checkpoint_layout(config)
            ensure_tail_checkpoint_cache(
                layout,
                suite_name=config.suite_name,
                optimizer_name=config.optimizer,
                seed=config.seed,
                total_epochs=config.epochs,
                steps_per_epoch=7,
                protocol_fingerprint="declared-protocol",
            )
            model = torch.nn.Linear(3, 2)
            self.assertIsNone(
                save_tail_checkpoint(
                    layout,
                    model=model,
                    epoch=0,
                    global_step=0,
                    total_epochs=config.epochs,
                    protocol_fingerprint="declared-protocol",
                    optimizer_name=config.optimizer,
                    seed=config.seed,
                )
            )
            for epoch in (1, 2, 3):
                with torch.no_grad():
                    model.weight.add_(1.0)
                save_tail_checkpoint(
                    layout,
                    model=model,
                    epoch=epoch,
                    global_step=epoch * 7,
                    total_epochs=config.epochs,
                    protocol_fingerprint="declared-protocol",
                    optimizer_name=config.optimizer,
                    seed=config.seed,
                )
            with self.assertRaisesRegex(RuntimeError, "completion marker"):
                verify_tail_checkpoint_cache(
                    layout,
                    expected_fingerprint="declared-protocol",
                )
            refs = finalize_tail_checkpoint_cache(
                layout,
                expected_fingerprint="declared-protocol",
            )
            self.assertEqual([ref.epoch for ref in refs], [1, 2, 3])
            loaded = load_verified_tail_checkpoint_refs(
                layout.root,
                expected_suite_name=config.suite_name,
                expected_optimizer_name=config.optimizer,
                expected_seed=config.seed,
                expected_fingerprint="declared-protocol",
                expected_epochs=(1, 2, 3),
            )
            self.assertEqual(loaded, refs)
            with self.assertRaisesRegex(RuntimeError, "identity"):
                load_verified_tail_checkpoint_refs(
                    layout.root,
                    expected_suite_name=config.suite_name,
                    expected_optimizer_name="adamw",
                    expected_seed=config.seed,
                    expected_fingerprint="declared-protocol",
                    expected_epochs=(1, 2, 3),
                )
            tampered = torch.load(refs[0].path, map_location="cpu", weights_only=False)
            tampered["model"]["weight"].add_(1.0)
            torch.save(tampered, refs[0].path)
            with self.assertRaisesRegex(RuntimeError, "completion marker"):
                load_verified_tail_checkpoint_refs(
                    layout.root,
                    expected_suite_name=config.suite_name,
                    expected_optimizer_name=config.optimizer,
                    expected_seed=config.seed,
                    expected_fingerprint="declared-protocol",
                    expected_epochs=(1, 2, 3),
                )

    def test_tail_checkpoint_resume_quarantines_future_and_completion(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            config = TangentRGConfig(
                suite_name="tail-resume-test",
                optimizer="adamw",
                seed=2027,
                epochs=3,
                lr_schedule_epochs=3,
                tail_checkpoint_cache_root=temporary,
            )
            layout = make_tail_checkpoint_layout(config)
            ensure_tail_checkpoint_cache(
                layout,
                suite_name=config.suite_name,
                optimizer_name=config.optimizer,
                seed=config.seed,
                total_epochs=config.epochs,
                steps_per_epoch=5,
                protocol_fingerprint="declared-protocol",
            )
            model = torch.nn.Linear(2, 2)
            for epoch in (1, 2, 3):
                save_tail_checkpoint(
                    layout,
                    model=model,
                    epoch=epoch,
                    global_step=epoch * 5,
                    total_epochs=config.epochs,
                    protocol_fingerprint="declared-protocol",
                    optimizer_name=config.optimizer,
                    seed=config.seed,
                )
            finalize_tail_checkpoint_cache(
                layout,
                expected_fingerprint="declared-protocol",
            )
            moved = quarantine_tail_checkpoint_cache_after_boundary(
                layout,
                epoch=2,
                global_step=10,
                expected_fingerprint="declared-protocol",
            )
            self.assertIn("cache_complete.json", moved)
            self.assertTrue(any("epoch_00003" in value for value in moved))
            remaining = verify_tail_checkpoint_cache(
                layout,
                expected_fingerprint="declared-protocol",
                require_complete=False,
            )
            self.assertEqual([ref.epoch for ref in remaining], [1, 2])
            self.assertEqual(
                [
                    ref.epoch
                    for ref in verify_tail_checkpoint_cache_prefix(
                        layout,
                        through_epoch=2,
                        expected_fingerprint="declared-protocol",
                    )
                ],
                [1, 2],
            )
            remaining[0].path.unlink()
            with self.assertRaisesRegex(RuntimeError, "historical prefix"):
                verify_tail_checkpoint_cache_prefix(
                    layout,
                    through_epoch=2,
                    expected_fingerprint="declared-protocol",
                )

    def test_empty_tail_prefix_is_valid_only_before_window_begins(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            config = TangentRGConfig(
                suite_name="tail-prefix-test",
                optimizer="adamw",
                seed=31415,
                epochs=101,
                lr_schedule_epochs=30,
                tail_checkpoint_cache_root=temporary,
            )
            layout = make_tail_checkpoint_layout(config)
            ensure_tail_checkpoint_cache(
                layout,
                suite_name=config.suite_name,
                optimizer_name=config.optimizer,
                seed=config.seed,
                total_epochs=config.epochs,
                steps_per_epoch=5,
                protocol_fingerprint="declared-protocol",
            )
            self.assertEqual(
                verify_tail_checkpoint_cache_prefix(
                    layout,
                    through_epoch=1,
                    expected_fingerprint="declared-protocol",
                ),
                (),
            )
            with self.assertRaisesRegex(RuntimeError, "historical prefix"):
                verify_tail_checkpoint_cache_prefix(
                    layout,
                    through_epoch=2,
                    expected_fingerprint="declared-protocol",
                )


class MuonClipRMSAlgebraTests(unittest.TestCase):
    def test_direction_has_exact_declared_rms_for_rectangular_shapes(self):
        generator = torch.Generator().manual_seed(71)
        for shape in ((4, 4), (6, 3), (3, 6)):
            gradient = torch.randn(shape, generator=generator)
            parts = matrix_update_components(
                gradient,
                None,
                momentum=0.95,
                nesterov=False,
                newton_schulz_steps=5,
                epsilon=1.0e-7,
                rms_scale=0.20,
            )
            observed = parts["direction"].float().square().mean().sqrt().item()
            self.assertAlmostEqual(observed, 0.20, places=6)
            self.assertAlmostEqual(float(parts["effective_rms"]), 0.20, places=6)

    def test_zero_gradient_does_not_invent_a_direction(self):
        parts = matrix_update_components(
            torch.zeros(3, 5),
            None,
            momentum=0.95,
            nesterov=False,
            newton_schulz_steps=5,
            epsilon=1.0e-7,
            rms_scale=0.20,
        )
        self.assertEqual(float(parts["effective_rms"]), 0.0)
        self.assertEqual(torch.count_nonzero(parts["direction"]).item(), 0)

    def test_one_step_matches_declared_decay_and_rms_direction_algebra(self):
        parameter = torch.nn.Parameter(
            torch.tensor([[1.0, -0.5], [0.3, 0.8]], dtype=torch.float32)
        )
        bias = torch.nn.Parameter(torch.zeros(2))
        optimizer = MuonClipRMSWithAuxAdamW(
            [("fc1.weight", parameter), ("fc1.bias", bias)],
            muon_parameter_names=("fc1.weight",),
            muon_lr=2.0e-3,
            muon_momentum=0.95,
            muon_nesterov=False,
            muon_weight_decay=1.0e-2,
            newton_schulz_steps=5,
            muon_eps=1.0e-7,
            rms_scale=0.20,
            auxiliary_lr=1.0e-3,
            auxiliary_betas=(0.9, 0.95),
            auxiliary_eps=1.0e-8,
            auxiliary_weight_decay=1.0e-2,
        )
        gradient = torch.tensor([[0.4, -0.2], [0.1, 0.6]])
        parameter.grad = gradient.clone()
        before = parameter.detach().clone()
        preview = matrix_update_components(
            gradient,
            None,
            momentum=0.95,
            nesterov=False,
            newton_schulz_steps=5,
            epsilon=1.0e-7,
            rms_scale=0.20,
        )
        optimizer.step()
        expected = (1.0 - 2.0e-3 * 1.0e-2) * before - 2.0e-3 * preview["direction"]
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"],
            preview["momentum_buffer"],
        )


if __name__ == "__main__":
    unittest.main()
