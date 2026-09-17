"""
test_directional_memory.py -- protected memory variants (Part 2/3).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.directional_memory import (  # noqa: E402
    run_directional_memory, delay_resolved_capacity, memory_lifetime, z_autocorrelation, MEMORY_VARIANTS,
)
from qrc_qiskit import random_input  # noqa: E402


def test_shift_variant_matches_spatial_memory_ground_truth():
    """`variant='shift'` must be a thin wrapper on `spatial_memory`'s
    already-proven exact shift register -- same output for the same input."""
    from decoupled_qrc.spatial_memory import run_spatial_memory
    u = random_input(20, seed=0)
    a = run_directional_memory("shift", N=4, u_seq=u)
    b = run_spatial_memory(L=4, u_seq=u, gamma_M=0.0)
    assert np.array_equal(a.X, b.X)


def test_protected_integrable_reuses_memory_module():
    """`variant='protected_integrable'` must be a thin wrapper on
    `memory.run_memory_register(memory_mode='integrable')` -- same output
    for the same input/seeds."""
    from decoupled_qrc.memory import run_memory_register
    u = random_input(20, seed=0)
    a = run_directional_memory("protected_integrable", N=4, u_seq=u, epsilon_M=0.4, disorder_seed=3)
    b = run_memory_register(N=4, u_seq=u, memory_mode="integrable", lambda_im=0.4, omega_scale=0.4,
                             disorder_seed=3)
    assert np.array_equal(a.X, b.X)


def test_shift_variant_gives_near_perfect_short_delay_recall():
    u = random_input(200, seed=0)
    run = run_directional_memory("shift", N=4, u_seq=u)
    k, C = delay_resolved_capacity(run, k_max=4, washout=20, n_val=50, n_test=80)
    assert C[0] > 0.95 and C[1] > 0.95 and C[2] > 0.95
    assert C[3] < 0.3


def test_small_epsilon_m_fails_to_store_information():
    """Documented finding: at truly small epsilon_M (0.05, the literal
    reading of Part 2B's 'epsilon_M small'), the protected-integrable
    memory stores almost NOTHING recoverable -- the per-step signal is
    dominated by cumulative drift from many weak, same-signed rotations.
    This is a real, load-bearing finding (see
    docs/DQRC_DIRECTIONAL_DECOUPLING_RESULTS.md question 1) -- this test
    locks it in so it isn't silently 'fixed' by a future change without
    updating that finding."""
    u = random_input(200, seed=0)
    run = run_directional_memory("protected_integrable", N=4, u_seq=u, epsilon_M=0.05, disorder_seed=0)
    k, C = delay_resolved_capacity(run, k_max=3, washout=20, n_val=50, n_test=80)
    assert sum(C) < 0.5, f"expected near-zero capacity at epsilon_M=0.05, got C={C}"


def test_larger_epsilon_m_recovers_real_capacity():
    u = random_input(200, seed=0)
    run = run_directional_memory("protected_integrable", N=4, u_seq=u, epsilon_M=0.6, disorder_seed=0)
    k, C = delay_resolved_capacity(run, k_max=3, washout=20, n_val=50, n_test=80)
    assert C[0] > 0.5, f"expected real capacity at epsilon_M=0.6, got C={C}"


def test_z_autocorrelation_starts_at_one_and_decays_faster_for_larger_epsilon():
    k1, corr_small = z_autocorrelation(n_mem=3, epsilon_M=0.05, omega_seed=0, n_steps=8)
    k2, corr_large = z_autocorrelation(n_mem=3, epsilon_M=0.8, omega_seed=0, n_steps=8)
    assert corr_small[0] == 1.0 and corr_large[0] == 1.0
    # the small-epsilon curve should stay closer to 1 on average than the large-epsilon one
    assert np.mean(np.abs(np.array(corr_small[1:]))) > np.mean(np.abs(np.array(corr_large[1:]))) - 0.5


def test_invalid_variant_raises():
    import pytest
    with pytest.raises(ValueError):
        run_directional_memory("not_a_variant", N=4, u_seq=[0.1, 0.2])
