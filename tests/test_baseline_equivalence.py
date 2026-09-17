"""
test_baseline_equivalence.py -- `decoupled_qrc.baseline.monolithic_qrc` must
WRAP, not reimplement, the finalized `mixed_syk_core`/`qrc_qiskit` reservoir:
calling it with a given config must reproduce EXACTLY what calling
`mixed_syk_core.run_reservoir_mixed` directly (with the matching g/J derived
the same way) produces, bit-for-bit (both run under the same deterministic
Aer path). This is the direct proof for
docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 9's claim that baseline.py
wraps rather than reimplements the finalized reservoir.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.baseline import monolithic_qrc  # noqa: E402
import mixed_syk_core as msc  # noqa: E402
from qrc_qiskit import ReservoirConfig, random_input  # noqa: E402


def test_monolithic_qrc_matches_mixed_syk_core_directly():
    N, kappa, T, reps, max_weight = 4, 1.0, 20, 1, 2
    term_seed, disorder_seed, input_seed = 17, 23, 0

    run = monolithic_qrc(N=N, kappa=kappa, T=T, reps=reps, max_weight=max_weight,
                          term_seed=term_seed, disorder_seed=disorder_seed, input_seed=input_seed)

    g, J = msc.kappa_to_gJ(kappa, 0.6, 0.6)
    cfg = ReservoirConfig(N=N, g=0.0, reps=reps, input_qubit=0, seed=disorder_seed)
    u = random_input(T, seed=input_seed)
    labels_ref, X_ref, info_ref, _ = msc.run_reservoir_mixed(
        cfg, u, g=g, J=J, reps=reps, term_seed=term_seed, method="density_matrix", max_weight=max_weight)

    assert run.labels == labels_ref
    assert np.array_equal(run.u, u)
    assert np.allclose(run.X, X_ref, atol=1e-12)


def test_monolithic_qrc_at_eoc_uses_notebook4_stored_kappa():
    from decoupled_qrc.baseline import monolithic_qrc_at_eoc
    import eoc_config as ec
    run = monolithic_qrc_at_eoc(N=4, T=10, reps=1, max_weight=1, term_seed=0, disorder_seed=0, input_seed=0)
    assert "kappa=0.96" in run.resources.notes
    assert ec.NOTEBOOK4_QELM_EOC_KAPPA == 0.960


def test_classical_delay_qrc_has_zero_quantum_resources():
    from decoupled_qrc.baseline import classical_delay_qrc
    run = classical_delay_qrc(T=20, m=5)
    assert run.resources.n_qubits_physical == 0
    assert run.resources.two_qubit_gates == 0
    assert run.X.shape == (20, 6)
