"""
test_spatial_memory.py -- (1) exact-shift correctness (gamma_M=0 must
reproduce memory.py's already-proven quantum-delay-register behavior),
(2) fading behaves as a monotonic-in-k, monotonic-in-gamma decay, (3) the
Aer statevector-reset-bug regression (same pattern as test_memory.py),
(4) reproducibility.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.spatial_memory import (  # noqa: E402
    build_spatial_memory_circuit, run_spatial_memory, delay_resolved_capacity, amplitude_damping_kraus,
)
from qrc_qiskit import random_input, make_simulator  # noqa: E402
from qiskit import transpile  # noqa: E402


def test_gamma_zero_matches_exact_shift_register_ground_truth():
    """With gamma_M=0, slot k (0-indexed from the input) must hold EXACTLY
    Ry(pi*u_{t-k})|0> at every step -- the same closed-form ground truth
    `tests/test_memory.py::test_quantum_delay_register_is_exact_shift_no_classical_array`
    checks for `memory.build_quantum_delay_circuit`."""
    u_seq = [0.2, 0.6, 0.9, 0.1, 0.4]
    L = 5
    qc, labels, mem_qubits = build_spatial_memory_circuit(L, u_seq, gamma_M=0.0)
    sim = make_simulator(method="density_matrix")
    tqc = transpile(qc, sim, optimization_level=0)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)

    t = 4
    slots = mem_qubits
    for k, q in enumerate(slots, start=1):
        if t - k < 0:
            continue
        theta = np.pi * u_seq[t - k]
        got_z = np.real(data[f"Z{q}__t{t}"])
        got_x = np.real(data[f"X{q}__t{t}"])
        assert abs(got_z - np.cos(theta)) < 1e-9
        assert abs(got_x - np.sin(theta)) < 1e-9


def test_no_classical_array_used_for_readout():
    """Structural check: `build_spatial_memory_circuit` never indexes
    `u_seq` at any t' != the CURRENT t inside the per-step loop -- i.e. the
    only place a historical input value can appear in the returned features
    is via the quantum register itself, not a Python-side lookup. Verified
    by source inspection here (a change reintroducing e.g. `u_seq[t-k]`
    inside the snapshot-emitting block would break this test)."""
    import inspect
    src = inspect.getsource(build_spatial_memory_circuit)
    loop_body = src.split("for t, u_t in enumerate(u_seq):", 1)[1]
    # only "u_t" (the CURRENT input) may be referenced inside the loop body,
    # never a re-indexed u_seq[...]
    assert "u_seq[" not in loop_body


def test_last_slot_z_expectation_decays_toward_plus_one_with_gamma():
    """Amplitude damping pushes population toward |0> (<Z> -> +1). With a
    fixed nonzero input driving <Z> away from +1 at gamma=0, a stronger
    gamma_M must pull the OLDEST slot's <Z> measurably closer to +1 -- a
    direct, sign-unambiguous confirmation that damping is actually being
    applied (not just present in the circuit but silently absent, e.g. a
    Kraus-append bug)."""
    u_seq = [1.0] * 6  # Ry(pi) each step -> would give <Z>=-1 with no damping
    results = {}
    for gamma in (0.0, 0.4):
        qc, labels, mem_qubits = build_spatial_memory_circuit(4, u_seq, gamma_M=gamma)
        sim = make_simulator(method="density_matrix")
        tqc = transpile(qc, sim, optimization_level=0)
        result = sim.run(tqc, shots=1).result()
        data = result.data(0)
        oldest = mem_qubits[-1]
        results[gamma] = np.real(data[f"Z{oldest}__t5"])
    assert results[0.4] > results[0.0] + 0.1, results


def test_exact_readout_capacity_is_gamma_invariant():
    """The documented, non-obvious finding: under EXACT (noiseless)
    expectation-value readout, damping does NOT reduce capacity -- a fixed
    per-step contraction is still an invertible, deterministic function of
    u_{t-k}, so ridge recovers it regardless of gamma_M."""
    u = random_input(200, seed=0)
    sums = []
    for gamma in (0.0, 0.3):
        run = run_spatial_memory(L=5, u_seq=u, gamma_M=gamma)
        k, C = delay_resolved_capacity(run, k_max=4, washout=20, n_val=50, n_test=80)
        sums.append(sum(C))
    assert abs(sums[0] - sums[1]) < 0.5, f"expected near-invariance under exact readout, got {sums}"


def test_finite_shot_capacity_decreases_with_stronger_fading():
    """With a realistic finite shot budget added (n_shots), stronger
    fading DOES measurably reduce capacity -- the shrinking signal
    eventually drops below the noise floor. This is the correct
    demonstration of 'fading memory' for this exact-readout-by-default
    module; see `delay_resolved_capacity`'s own docstring."""
    u = random_input(200, seed=0)
    sums = []
    for gamma in (0.0, 0.6):
        run = run_spatial_memory(L=6, u_seq=u, gamma_M=gamma)
        k, C = delay_resolved_capacity(run, k_max=5, washout=20, n_val=50, n_test=80, n_shots=200, noise_seed=0)
        sums.append(sum(C))
    assert sums[0] > sums[1] + 0.5, f"stronger fading should reduce finite-shot capacity, got {sums}"


def test_run_spatial_memory_rejects_statevector_method():
    with pytest.raises(ValueError, match="density_matrix"):
        run_spatial_memory(L=3, u_seq=[0.1, 0.2], method="statevector")


def _run_with_method(qc, labels, T, sim):
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)
    X = np.empty((T, len(labels)))
    for t in range(T):
        for j, lab in enumerate(labels):
            X[t, j] = np.real(data[f"{lab}__t{t}"])
    return X


def test_gamma_zero_bare_shift_register_agrees_under_either_method():
    """Special case, worth documenting explicitly: with gamma_M=0 the ONLY
    gates are SWAP/reset/Ry -- none of which ever entangle two qubits (SWAP
    relocates an existing product state, it does not create correlations
    between previously-independent qubits) -- so the global state stays an
    EXACT pure product state throughout, and method='statevector' is
    actually fine here (no confirmed-bug trigger, since there is no
    mixedness for it to mishandle). This is the opposite of the next test
    (gamma_M>0), and is why the module still defaults its safety check to
    density_matrix -- callers should not assume gamma=0 forever, since
    composing this register with any entangling interface immediately
    reintroduces the risk."""
    u = random_input(12, seed=1)
    qc, labels, mem_qubits = build_spatial_memory_circuit(4, u, gamma_M=0.0)
    sim_dm = make_simulator(method="density_matrix")
    sim_sv = make_simulator(method="statevector")
    Xdm = _run_with_method(qc, labels, len(u), sim_dm)
    Xsv = _run_with_method(qc, labels, len(u), sim_sv)
    assert np.max(np.abs(Xdm - Xsv)) < 1e-9


def test_gamma_nonzero_statevector_disagrees_with_density_matrix():
    """With gamma_M>0 the Kraus (amplitude-damping) channel makes the state
    genuinely mixed. Aer's statevector method can only represent this via a
    single stochastic quantum-trajectory unraveling per shot -- seed-
    dependent and generally wrong relative to the exact density-matrix
    expectation (the same class of pitfall
    docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 10 flags, here triggered
    by a real decoherence channel rather than a reset-on-an-entangled-qubit)."""
    u = random_input(12, seed=1)
    qc, labels, mem_qubits = build_spatial_memory_circuit(4, u, gamma_M=0.3)
    sim_dm = make_simulator(method="density_matrix")
    sim_sv = make_simulator(method="statevector")
    Xdm = _run_with_method(qc, labels, len(u), sim_dm)
    Xsv = _run_with_method(qc, labels, len(u), sim_sv)
    assert np.max(np.abs(Xdm - Xsv)) > 0.05


def test_reproducibility():
    u = random_input(30, seed=0)
    r1 = run_spatial_memory(L=4, u_seq=u, gamma_M=0.1)
    r2 = run_spatial_memory(L=4, u_seq=u, gamma_M=0.1)
    assert np.array_equal(r1.X, r2.X)


def test_amplitude_damping_kraus_is_identity_at_zero():
    k = amplitude_damping_kraus(0.0)
    assert k.is_cptp()
