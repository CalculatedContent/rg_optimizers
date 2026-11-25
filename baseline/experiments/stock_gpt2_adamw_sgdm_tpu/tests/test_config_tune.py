"""Scientific-contract tests, using synthetic result records and no training."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ..config import (BENCHMARK_VALIDATION_TOKENS, FULL_BUDGET_TOKENS,
                      TrainingConfig, smoke_config)
from ..tune import (DEFAULT_STAGES, candidate_configs, check_result,
                  expected_snapshot_steps, make_plan, rank_results,
                  read_immutable_json, select_stage, stage_config, maximum_training_tokens)


def valid_record(config, loss=4.0):
    result = {
        "config_sha256": config.config_sha256, "candidate_id": config.candidate_id,
        "stage_id": config.stage_id, "seed": config.seed,
        "status": "schedule_complete_target_not_met", "terminal": True,
        "full_training_recipe_completed": True,
        "tokens_seen": config.budget_tokens, "step": config.max_steps,
        "validation": {"validation_kind": config.validation_kind,
                       "validation_offset": config.validation_offset,
                       "evaluation_tokens": config.validation_tokens,
                       "tokens_seen": config.budget_tokens,
                       "step": config.max_steps, "val_nll": loss},
    }
    steps = expected_snapshot_steps(config)
    tracking = {"complete": True, "training_done": True, "completed": len(steps),
                "failed": 0, "pending": 0, "completed_steps": steps,
                "failed_steps": [], "pending_steps": [], "missing_expected_steps": [],
                "config_sha256": config.config_sha256}
    return result, tracking


class ConfigurationTests(unittest.TestCase):
    def test_stock_dimensions_and_prior_graft_convention_are_enforced(self):
        config = candidate_configs()[0]
        for changes in ({"n_layer": 2}, {"context_length": 512}, {"optimizer": "sgdm"},
                        {"graft_scope": "logical_qkv_separately"}, {"tokenizer": "cl100k_base"},
                        {"target_val_nll": 3.7}, {"beta1": 1.0}, {"beta2": 0.95},
                        {"eps": 1e-6}, {"norm_epsilon": 1e-20}, {"reference_lr_ratio": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)
        self.assertEqual(smoke_config().mode, "smoke")

    def test_equal_tokens_rescale_updates_and_schedule(self):
        candidates = candidate_configs()
        small = replace(candidates[0], budget_tokens=524_288_000)
        larger = replace(candidates[5], global_batch_sequences=256, budget_tokens=524_288_000)
        self.assertEqual(small.max_steps, 4 * larger.max_steps)
        self.assertEqual(small.max_steps * small.tokens_per_step,
                         larger.max_steps * larger.tokens_per_step)
        self.assertAlmostEqual(small.schedule_factor(small.max_steps // 2),
                               larger.schedule_factor(larger.max_steps // 2), places=12)

    def test_cosine_has_warmup_positive_final_update_and_terminal_zero(self):
        config = stage_config(candidate_configs()[0], DEFAULT_STAGES[0])
        self.assertGreater(config.schedule_factor(0), 0)
        self.assertEqual(config.schedule_factor(config.warmup_steps - 1), 1.0)
        self.assertEqual(config.schedule_factor(config.warmup_steps), 1.0)
        self.assertGreater(config.schedule_factor(config.max_steps - 1), 0)
        self.assertEqual(config.schedule_factor(config.max_steps), 0)
        with self.assertRaises(ValueError):
            config.schedule_factor(config.max_steps + 1)
        self.assertEqual(candidate_configs()[-1].warmup_fraction, 0.1)

    def test_grid_tunes_native_lr_and_reference_ratio_with_fixed_prior_rule(self):
        configs = candidate_configs()
        self.assertEqual(len(configs), 9)
        self.assertEqual(len({config.candidate_id for config in configs}), 9)
        self.assertEqual({(c.peak_lr, c.reference_lr_ratio) for c in configs},
                         {(lr, ratio) for lr in (0.0003, 0.001, 0.003) for ratio in (100, 400, 1000)})
        self.assertIn((0.001, 400), {(c.peak_lr, c.reference_lr_ratio) for c in configs})
        for config in configs:
            self.assertEqual((config.global_batch_sequences, config.microbatch_sequences), (64, 16))
            self.assertEqual((config.beta1, config.beta2, config.eps, config.norm_epsilon),
                             (0.9, 0.99, 1e-8, 1e-30))
            self.assertEqual(config.weight_decay, 0)
            self.assertFalse({"momentum", "dampening", "nesterov"}.intersection(config.as_dict()))
        self.assertEqual(maximum_training_tokens(2), 35_194_404_864)
        self.assertEqual(maximum_training_tokens(1), 14_684_258_304)

    def test_qualified_sister_and_sgdm_configs_cannot_cross_load(self):
        from baseline.experiments.stock_gpt2_sgdm_tpu.config import TrainingConfig as SGDMConfig
        self.assertNotEqual(TrainingConfig.__module__, SGDMConfig.__module__)
        with self.assertRaises(ValueError):
            TrainingConfig.from_dict(SGDMConfig().as_dict())
        with self.assertRaises(ValueError):
            SGDMConfig.from_dict(TrainingConfig().as_dict())
        self.assertNotEqual(TrainingConfig().config_sha256,
                            replace(TrainingConfig(), reference_lr_ratio=100).config_sha256)

    def test_wsd_final_decay_is_explicit_and_cannot_be_omitted(self):
        config = replace(candidate_configs()[0], schedule="wsd")
        self.assertEqual(config.schedule_factor(config.max_steps // 2), 1)
        self.assertGreater(config.schedule_factor(config.max_steps - 1), 0)
        self.assertEqual(config.schedule_factor(config.max_steps), 0)
        with self.assertRaises(ValueError):
            replace(config, stable_fraction=0.9)

    def test_incomplete_batches_and_mismatched_validation_are_rejected(self):
        config = candidate_configs()[0]
        for changes in ({"budget_tokens": FULL_BUDGET_TOKENS + 1},
                        {"microbatch_sequences": 3},
                        {"validation_offset": 0},
                        {"validation_kind": "benchmark"},
                        {"stop_at_target": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)

    def test_hashed_configs_detect_edits_and_refuse_replacement(self):
        config = candidate_configs()[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            config.write_json(path)
            self.assertEqual(TrainingConfig.from_json(path), config)
            config.write_json(path)
            with self.assertRaises(FileExistsError):
                replace(config, peak_lr=0.2).write_json(path)
            payload = json.loads(path.read_text())
            payload["peak_lr"] = 0.2
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                TrainingConfig.from_json(path)

    def test_promoted_horizon_changes_hash_and_uses_official_window_only_for_full(self):
        base = candidate_configs()[0]
        first = stage_config(base, DEFAULT_STAGES[0])
        full = stage_config(first, DEFAULT_STAGES[-1])
        self.assertNotEqual(first.config_sha256, full.config_sha256)
        self.assertEqual(full.validation_offset, 0)
        self.assertEqual(full.validation_tokens, BENCHMARK_VALIDATION_TOKENS)
        self.assertEqual(first.validation_offset, BENCHMARK_VALIDATION_TOKENS)
        self.assertEqual(first.schedule_factor(first.max_steps), 0)
        self.assertGreater(full.schedule_factor(0), 0)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.configs = [stage_config(c, DEFAULT_STAGES[0]) for c in candidate_configs()[:2]]

    def test_selects_final_nll_at_equal_tokens_not_more_steps_or_alpha(self):
        first, second = self.configs
        records = {first.candidate_id: valid_record(first, 4.0),
                   second.candidate_id: valid_record(second, 3.9)}
        ranking = rank_results(self.configs, records)
        self.assertEqual(ranking["ranking"][0]["candidate_id"], second.candidate_id)
        self.assertFalse(ranking["optimum_claim"])

    def test_different_budgets_or_seeds_cannot_be_compared(self):
        first, second = self.configs
        for changed in (replace(second, budget_tokens=2 * second.budget_tokens),
                        replace(second, seed=43)):
            records = {first.candidate_id: valid_record(first), changed.candidate_id: valid_record(changed)}
            with self.assertRaises(ValueError):
                rank_results([first, changed], records)

    def test_incomplete_endpoint_stale_eval_and_bad_diagnostics_are_not_promoted(self):
        config = self.configs[0]
        result, tracking = valid_record(config)
        cases = []
        bad = copy.deepcopy(result); bad["tokens_seen"] -= config.tokens_per_step
        cases.append((bad, tracking))
        bad = copy.deepcopy(result); bad["validation"]["step"] -= 1
        cases.append((bad, tracking))
        bad = copy.deepcopy(result); bad["validation"]["validation_offset"] = 0
        cases.append((bad, tracking))
        bad = copy.deepcopy(result); bad["validation"]["val_nll"] = float("nan")
        cases.append((bad, tracking))
        bad = copy.deepcopy(tracking); bad["pending"] = 1
        cases.append((result, bad))
        bad = copy.deepcopy(tracking); bad["failed"] = 1
        cases.append((result, bad))
        bad = copy.deepcopy(tracking); bad["completed_steps"] = bad["completed_steps"][:-1]
        cases.append((result, bad))
        for i, record in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                rank_results([config], {config.candidate_id: record})

    def test_explicit_divergence_is_reported_not_ranked_or_silently_dropped(self):
        first, second = self.configs
        divergent, _ = valid_record(first)
        divergent.update(status="failed_nonfinite", full_training_recipe_completed=False,
                         failure_reason="nonfinite training loss")
        records = {first.candidate_id: (divergent, None), second.candidate_id: valid_record(second)}
        result = rank_results(self.configs, records)
        self.assertEqual(len(result["ranking"]), 1)
        self.assertEqual(result["excluded_divergence"][0]["candidate_id"], first.candidate_id)

    def test_missing_results_block_selection_and_official_window_is_not_tuned(self):
        first = self.configs[0]
        with self.assertRaises(ValueError):
            rank_results(self.configs, {first.candidate_id: valid_record(first)})
        full = stage_config(first, DEFAULT_STAGES[-1])
        with self.assertRaises(ValueError):
            rank_results([full], {full.candidate_id: valid_record(full)})
        smoke = smoke_config()
        with self.assertRaises(ValueError):
            rank_results([smoke], {smoke.candidate_id: valid_record(smoke)})

    def test_extra_paused_snapshot_is_accepted_only_when_all_required_steps_exist(self):
        config = self.configs[0]
        result, tracking = valid_record(config)
        extra_step = 1
        tracking["completed_steps"] = sorted(tracking["completed_steps"] + [extra_step])
        tracking["completed"] += 1
        result["expected_snapshot_steps"] = tracking["completed_steps"][:]
        self.assertEqual(check_result(config, result, tracking), [])
        result["expected_snapshot_steps"].append(extra_step + 1)
        self.assertTrue(check_result(config, result, tracking))

    def test_terminal_failure_cannot_override_completed_flag_or_target_flag(self):
        config = self.configs[0]
        result, tracking = valid_record(config)
        result.update(status="failed", target_met=True)
        with self.assertRaises(ValueError):
            rank_results([config], {config.candidate_id: (result, tracking)})

    def test_plan_to_full_export_is_fresh_and_hash_audited(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_plan(root / "plan")
            stage_dir = root / "plan" / "screen"
            for next_name, expected_count in (("refine", 3), ("full", 1)):
                stage = read_immutable_json(stage_dir / "stage.json")
                results_dir = root / "runs" / stage["stage"]["stage_id"]
                for index, entry in enumerate(stage["candidates"]):
                    config = TrainingConfig.from_json(stage_dir / entry["config"])
                    result, tracking = valid_record(config, 4.0 + index / 100)
                    run_dir = results_dir / config.candidate_id
                    run_dir.mkdir(parents=True)
                    (run_dir / "TRAINING_STATUS.json").write_text(json.dumps(result))
                    (run_dir / "TRACKING_STATUS.json").write_text(json.dumps(tracking))
                output = root / "plan" / next_name
                select_stage(stage_dir, results_dir, output)
                promoted = read_immutable_json(output / "stage.json")
                self.assertEqual(len(promoted["candidates"]), expected_count)
                self.assertTrue(promoted["fresh_initialization_required"])
                stage_dir = output
            winner = TrainingConfig.from_json(stage_dir / "winner.json")
            self.assertEqual(winner.budget_tokens, FULL_BUDGET_TOKENS)
            self.assertEqual(winner.validation_kind, "benchmark")
            self.assertGreater(winner.schedule_factor(0), 0)


if __name__ == "__main__":
    unittest.main()
