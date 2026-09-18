"""
test_backaction_trajectory.py -- V2.1 Phase 10: time-resolved D_M(t),
replacing V1/V2's single-final-timestep back-action check.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.backaction_trajectory import compute_backaction_trajectory  # noqa: E402
from decoupled_qrc.directional_dqrc import DirectionalConfig  # noqa: E402
from decoupled_qrc.validation_utils import make_nested_seeds  # noqa: E402


def _cfgs(theta_off=0.0, phi_off=0.0):
    cfg_on = DirectionalConfig(memory_variant="protected_integrable", N_M=2, N_P=5, g_processor=0.5,
                                J_processor=0.33, epsilon_M=0.5, theta=0.2, phi=0.8, ap_kind="xy")
    cfg_off = DirectionalConfig(memory_variant="protected_integrable", N_M=2, N_P=5, g_processor=0.5,
                                 J_processor=0.33, epsilon_M=0.5, theta=theta_off, phi=phi_off, ap_kind="xy")
    return cfg_on, cfg_off


def test_backaction_trajectory_has_one_entry_per_step():
    cfg_on, cfg_off = _cfgs()
    seeds = make_nested_seeds(0, 0)
    traj = compute_backaction_trajectory(cfg_on, cfg_off, T=6, seeds=seeds)
    assert len(traj.steps) == 6
    assert len(traj.trace_distance) == 6
    assert len(traj.fidelity) == 6


def test_backaction_trajectory_zero_when_interface_configs_identical():
    """cfg_on == cfg_off (both theta=0.2, phi=0.8) -> the two trajectories
    are literally the same circuit, so D_M(t) must be ~0 at every step."""
    cfg_on, cfg_off = _cfgs(theta_off=0.2, phi_off=0.8)
    seeds = make_nested_seeds(1, 0)
    traj = compute_backaction_trajectory(cfg_on, cfg_off, T=6, seeds=seeds)
    assert traj.max_trace_distance < 1e-6
    assert all(f > 1 - 1e-6 for f in traj.fidelity)


def test_backaction_trajectory_summary_stats_consistent_with_series():
    cfg_on, cfg_off = _cfgs()
    seeds = make_nested_seeds(2, 0)
    traj = compute_backaction_trajectory(cfg_on, cfg_off, T=8, seeds=seeds)
    assert traj.mean_trace_distance == pytest.approx(float(np.mean(traj.trace_distance)), abs=1e-9)
    assert traj.max_trace_distance == pytest.approx(float(np.max(traj.trace_distance)), abs=1e-9)
    assert traj.final_trace_distance == pytest.approx(traj.trace_distance[-1], abs=1e-9)


def test_backaction_trajectory_values_are_valid_trace_distances():
    cfg_on, cfg_off = _cfgs()
    seeds = make_nested_seeds(3, 0)
    traj = compute_backaction_trajectory(cfg_on, cfg_off, T=6, seeds=seeds)
    assert all(0.0 <= td <= 1.0 + 1e-6 for td in traj.trace_distance)
    assert all(0.0 <= f <= 1.0 + 1e-6 for f in traj.fidelity)


def test_backaction_trajectory_rejects_statevector_method():
    cfg_on, cfg_off = _cfgs()
    seeds = make_nested_seeds(0, 0)
    with pytest.raises(ValueError, match="density_matrix"):
        compute_backaction_trajectory(cfg_on, cfg_off, T=5, seeds=seeds, method="statevector")


def test_backaction_trajectory_common_random_numbers_deterministic():
    cfg_on, cfg_off = _cfgs()
    seeds = make_nested_seeds(4, 0)
    traj1 = compute_backaction_trajectory(cfg_on, cfg_off, T=6, seeds=seeds)
    traj2 = compute_backaction_trajectory(cfg_on, cfg_off, T=6, seeds=seeds)
    assert traj1.trace_distance == pytest.approx(traj2.trace_distance, abs=1e-12)
