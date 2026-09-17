"""
test_reset_ablation.py -- Part 12/17 (tests #14/#15): processor reset as a
valid trace-preserving channel, and reset_period=None reproducing the
original persistent-processor architecture exactly.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.reset_ablation import run_directional_dqrc_with_reset, build_directional_circuit_with_reset  # noqa: E402
from decoupled_qrc.directional_dqrc import DirectionalConfig, run_directional_dqrc  # noqa: E402
from decoupled_qrc.utils import make_seed_bundle  # noqa: E402
from qiskit.quantum_info import DensityMatrix  # noqa: E402


def _cfg():
    return DirectionalConfig(memory_variant="shift", N_M=2, N_P=4, theta=0.2, phi=0.8, kappa_processor=1.0)


def test_reset_period_none_reproduces_persistent_architecture_exactly():
    """The whole point of building this as an ADDITIVE module: with
    reset_period=None, output must be bit-identical to the untouched
    `directional_dqrc.run_directional_dqrc`."""
    cfg = _cfg()
    r_new = run_directional_dqrc_with_reset(cfg, T=15, reset_period=None, master_seed=0)
    r_orig = run_directional_dqrc(cfg, T=15, master_seed=0)
    assert np.array_equal(r_new.X_proc, r_orig.X_proc)
    assert np.array_equal(r_new.X_mem, r_orig.X_mem)


def test_reset_every_step_gives_pure_processor_state_each_step():
    """With reset_period=1, the processor register (as a whole) must be
    reset to |0...0> immediately after readout each step -- i.e. right
    BEFORE the next step's processor layer runs, the processor alone
    starts pure and unentangled with everything else. Verified by
    inspecting the full state right after the reset instructions inserted
    for the LAST timestep (the circuit ends with a reset, so the final
    state's processor block must be exactly |0><0|...|0><0|)."""
    from qiskit.quantum_info import partial_trace
    cfg = _cfg()
    seeds = make_seed_bundle(0)
    from qrc_qiskit import random_input
    u = random_input(10, seed=seeds.dataset_seed)
    qc, groups, mem_qubits, ancilla, proc_qubits = build_directional_circuit_with_reset(cfg, u, seeds, reset_period=1)
    qc.save_density_matrix(label="rho")
    from qrc_qiskit import make_simulator
    from qiskit import transpile
    sim = make_simulator(method="density_matrix")
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    rho = DensityMatrix(np.asarray(result.data(0)["rho"]))
    other = [q for q in range(qc.num_qubits) if q not in proc_qubits]
    rho_proc = partial_trace(rho, other)
    expected = np.zeros((2 ** len(proc_qubits), 2 ** len(proc_qubits)), dtype=complex)
    expected[0, 0] = 1.0
    assert np.allclose(np.asarray(rho_proc.data), expected, atol=1e-9)


def test_reset_channel_is_trace_preserving():
    """Sanity check that the overall state stays a valid (trace-1,
    Hermitian, positive-semidefinite) density matrix with resets inserted
    -- proving qc.reset() composed with everything else stays a genuine
    CPTP evolution, not an ad hoc non-physical operation."""
    cfg = _cfg()
    seeds = make_seed_bundle(0)
    from qrc_qiskit import random_input, make_simulator
    from qiskit import transpile
    u = random_input(8, seed=seeds.dataset_seed)
    qc, groups, mem_qubits, ancilla, proc_qubits = build_directional_circuit_with_reset(cfg, u, seeds, reset_period=2)
    qc.save_density_matrix(label="rho")
    sim = make_simulator(method="density_matrix")
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    rho = np.asarray(DensityMatrix(np.asarray(result.data(0)["rho"])).data)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-9)
    assert np.allclose(rho, rho.conj().T, atol=1e-9)
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals > -1e-9)


def test_reset_period_two_differs_from_reset_period_one():
    cfg = _cfg()
    r1 = run_directional_dqrc_with_reset(cfg, T=12, reset_period=1, master_seed=0)
    r2 = run_directional_dqrc_with_reset(cfg, T=12, reset_period=2, master_seed=0)
    assert not np.array_equal(r1.X_proc, r2.X_proc)


def test_reset_ablation_rejects_statevector_method():
    cfg = _cfg()
    with pytest.raises(ValueError, match="density_matrix"):
        run_directional_dqrc_with_reset(cfg, T=10, reset_period=1, master_seed=0, method="statevector")
