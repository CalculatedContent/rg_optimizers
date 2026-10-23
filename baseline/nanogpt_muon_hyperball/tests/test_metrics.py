from rg_nanogpt_muon_hyperball.run_utils import METRIC_FIELDS


def test_hyperball_diagnostics_are_persisted() -> None:
    required = {
        "hyperball_relative_radius",
        "hyperball_matrix_updates_since_eval",
        "hyperball_active_fraction",
        "hyperball_mean_scale",
        "hyperball_min_scale",
        "hyperball_mean_radius",
        "hyperball_max_proposed_update_to_weight_ratio",
        "hyperball_max_applied_update_to_weight_ratio",
        "hyperball_max_proposed_update_norm",
        "hyperball_max_applied_update_norm",
    }
    assert required.issubset(METRIC_FIELDS)
