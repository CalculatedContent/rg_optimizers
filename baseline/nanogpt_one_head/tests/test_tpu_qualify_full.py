import csv
from pathlib import Path

import pytest

from rg_nanogpt_one_head.tpu_qualify_full import (
    EXPECTED_STEPS, MATRICES, artifact_check, project_hours,
)


def grid():
    return [{"chip": i % 4} for i in range(25)]


def test_eta_uses_slowest_queue_not_total_divided_by_four():
    result = project_hours(grid(), {0: 2000, 1: 2000, 2: 2000, 3: 2000})
    assert result["jobs_per_chip"] == {"0": 7, "1": 6, "2": 6, "3": 6}
    assert result["projected_sweep_hours"] == pytest.approx(7 * EXPECTED_STEPS / 3600)
    assert result["planning_hours_with_25pct_margin"] == pytest.approx(1.25 * 7 * EXPECTED_STEPS / 3600)


def test_eta_can_be_limited_by_slower_chip_with_fewer_jobs():
    result = project_hours(grid(), {0: 1000, 1: 2000, 2: 1000, 3: 1000})
    assert result["projected_sweep_hours"] == pytest.approx(6 * EXPECTED_STEPS / 3600)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_rejects_unmeasured_or_invalid_timings(value):
    with pytest.raises(ValueError):
        project_hours(grid(), {0: value, 1: 1, 2: 1, 3: 1})


def test_all_four_chips_must_have_a_measurement():
    with pytest.raises(ValueError):
        project_hours(grid(), {0: 1, 1: 1, 2: 1})


def make_artifacts(root: Path, steps=(0, 2441)):
    (root / "checkpoint_latest.pt").write_bytes(b"test-only-placeholder")
    (root / "spectral").mkdir()
    with (root / "spectral/layers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "matrix_name", "alpha_raw", "alpha_clip_xmax"])
        writer.writeheader()
        for step in steps:
            for name in MATRICES:
                writer.writerow(dict(step=step, matrix_name=name, alpha_raw=3, alpha_clip_xmax=3))
    with (root / "random_canary_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "id", "dose"])
        writer.writeheader()
        for step in steps:
            for dose in [0, 1, 4, 16, 64]:
                for index in range(8):
                    writer.writerow(dict(step=step, id=f"dose{dose}_{index}", dose=dose))


def test_acceptance_inventory_includes_trained_spectrum_and_all_canaries(tmp_path):
    make_artifacts(tmp_path)
    result = artifact_check(tmp_path, 2500)
    assert result["verified_spectral_steps"] == [0, 2441]
    assert result["canaries_per_evaluation"] == 40


def test_short_prefix_is_not_misrepresented_as_full_experiment(tmp_path):
    make_artifacts(tmp_path)
    (tmp_path / "run_complete.json").write_text('{"completed": true}')
    with pytest.raises(RuntimeError, match="completed run"):
        artifact_check(tmp_path, 2500)


def test_second_phase_requires_trained_weightwatcher_output(tmp_path):
    make_artifacts(tmp_path, steps=(0,))
    with pytest.raises(RuntimeError, match="2441"):
        artifact_check(tmp_path, 2500)


def test_first_phase_accepts_initial_spectrum(tmp_path):
    make_artifacts(tmp_path, steps=(0,))
    assert artifact_check(tmp_path, 500)["verified_spectral_steps"] == [0]
