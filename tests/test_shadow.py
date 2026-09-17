"""
test_shadow.py -- the classical-shadow estimator (density-matrix Born
sampling + shadow_measurements' verified inversion formula) must converge to
the exact Pauli expectation values as shots grow, on a small system where
both can be computed cheaply (Part 6's required test).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.shadows import (  # noqa: E402
    reservoir_density_matrix, exact_pauli_features, shadow_readout, shadow_convergence_sweep,
)
from qiskit.quantum_info import DensityMatrix, random_density_matrix  # noqa: E402


def test_shadow_matches_exact_for_a_known_product_state():
    """rho = |0><0| tensor |+><+| (N=2): exact <Z0>=1, <X1>=1, <Z1>=<X0>=0 --
    a hand-computable ground truth, not just 'looks plausible'. The shadow
    estimate at a generous shot count must land close to these exact values."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(1)
    rho = DensityMatrix(qc).data

    labels, ops, exact_vals = exact_pauli_features(rho, N=2, max_weight=1)
    exact = dict(zip(labels, exact_vals))
    assert exact["Z0"] == pytest.approx(1.0, abs=1e-9)
    assert exact["X1"] == pytest.approx(1.0, abs=1e-9)
    assert exact["Z1"] == pytest.approx(0.0, abs=1e-9)

    rng = np.random.RandomState(0)
    labels2, ops2, shadow_vals = shadow_readout(rho, N=2, max_weight=1, n_snapshots=20000, rng=rng)
    shadow = dict(zip(labels2, shadow_vals))
    assert abs(shadow["Z0"] - 1.0) < 0.05
    assert abs(shadow["X1"] - 1.0) < 0.05
    assert abs(shadow["Z1"] - 0.0) < 0.08


def test_shadow_converges_monotonically_with_shots():
    """RMSE(shadow, exact) should trend DOWN as n_snapshots grows (classical
    shadow variance shrinks as 1/K) -- checked on a genuinely mixed reservoir
    state (purity < 1), not a pure toy state, since that's what DQRC's
    memory-register readout actually faces."""
    rho = reservoir_density_matrix(N=3, kappa=1.0, T=10, reps=1, term_seed=1, disorder_seed=1, input_seed=0)
    purity = float(np.real(np.trace(rho @ rho)))
    assert purity < 0.999, "expected a genuinely mixed state for this test to be meaningful"

    sweep = shadow_convergence_sweep(rho, N=3, max_weight=2, shot_grid=[100, 1000, 8000],
                                      n_repeats=4, seed=0)
    rmses = [p.rmse for p in sweep]
    assert rmses[0] > rmses[-1], f"expected RMSE to shrink with more shots, got {rmses}"
    assert rmses[2] < 0.15, f"RMSE at 8000 shots should be reasonably small, got {rmses[2]}"


def test_shadow_readout_seeded_reproducibility():
    """Same measurement seed -> identical shadow estimate (no hidden global
    RNG state)."""
    rho = reservoir_density_matrix(N=3, kappa=1.0, T=8, reps=1, term_seed=2, disorder_seed=2, input_seed=1)
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    _, _, v1 = shadow_readout(rho, N=3, max_weight=2, n_snapshots=500, rng=rng1)
    _, _, v2 = shadow_readout(rho, N=3, max_weight=2, n_snapshots=500, rng=rng2)
    assert np.array_equal(v1, v2)


def test_shadow_and_exact_feature_dictionaries_match():
    """`exact_pauli_features` and `shadow_readout` must target the SAME
    observable dictionary (same labels, same order) -- required for the
    exact-vs-shadow comparison to be well-defined."""
    rho = np.array(random_density_matrix(2 ** 3, seed=0).data)
    labels_exact, _, _ = exact_pauli_features(rho, N=3, max_weight=2)
    rng = np.random.RandomState(0)
    labels_shadow, _, _ = shadow_readout(rho, N=3, max_weight=2, n_snapshots=10, rng=rng)
    assert labels_exact == labels_shadow
