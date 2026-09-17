"""
test_directional_dqrc.py -- the combined M -> A -> P circuit builder and
resource accounting.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.directional_dqrc import DirectionalConfig, run_directional_dqrc  # noqa: E402


def test_resource_accounting_matches_config():
    cfg = DirectionalConfig(memory_variant="shift", N_M=2, N_P=5)
    run = run_directional_dqrc(cfg, T=10, master_seed=0)
    assert run.resources.n_qubits_physical == cfg.n_qubits_total == 2 + 1 + 5  # N_M + N_A + N_P = 8
    assert run.resources.n_ancilla == 1
    assert run.resources.n_features == run.X_mem.shape[1] + run.X_proc.shape[1]


def test_reproducible_same_seed():
    cfg = DirectionalConfig(memory_variant="shift", N_M=2, N_P=5)
    r1 = run_directional_dqrc(cfg, T=15, master_seed=3)
    r2 = run_directional_dqrc(cfg, T=15, master_seed=3)
    assert np.array_equal(r1.X_mem, r2.X_mem)
    assert np.array_equal(r1.X_proc, r2.X_proc)


def test_theta_zero_and_phi_zero_gives_finite_well_formed_features():
    cfg = DirectionalConfig(memory_variant="shift", N_M=2, N_P=4, theta=0.0, phi=0.0)
    run = run_directional_dqrc(cfg, T=10, master_seed=0)
    assert np.all(np.isfinite(run.X_mem))
    assert np.all(np.isfinite(run.X_proc))


def test_rejects_statevector_method():
    import pytest
    cfg = DirectionalConfig(memory_variant="shift", N_M=2, N_P=4)
    with pytest.raises(ValueError, match="density_matrix"):
        run_directional_dqrc(cfg, T=10, master_seed=0, method="statevector")


def test_protected_integrable_variant_runs():
    cfg = DirectionalConfig(memory_variant="protected_integrable", N_M=2, N_P=4, epsilon_M=0.5)
    run = run_directional_dqrc(cfg, T=10, master_seed=0)
    assert run.X_mem.shape[0] == 10


def test_reference_direct_coupling_run_reuses_id_memory_eoc():
    from decoupled_qrc.directional_dqrc import reference_direct_coupling_run
    run = reference_direct_coupling_run("heisenberg", N_M=2, N_P=5, kappa_processor=1.0, lambda_mp=0.5, T=10)
    assert run.X_combined.shape[0] == 10
