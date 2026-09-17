"""
test_no_classical_delay_leakage.py -- Part 23 item 1/2's explicit failure
modes: the PRIMARY id_memory_eoc result must never use Python-stored
historical inputs or re-encode the complete input history every timestep;
`classical_delay_plus_processor` (the deliberate classical-memory CONTROL)
must stay structurally separate, never silently merged into the primary
circuit-building path.
"""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc import id_memory_eoc as idme  # noqa: E402
from decoupled_qrc.id_memory_eoc import IDMemoryEOCConfig, run_id_memory_eoc, classical_delay_plus_processor  # noqa: E402
from qrc_qiskit import random_input  # noqa: E402


def test_build_circuit_never_reindexes_u_seq():
    """Source-inspection check (same pattern as
    `tests/test_spatial_memory.py::test_no_classical_array_used_for_readout`):
    inside the per-step loop, only the CURRENT `u_t` may be referenced --
    never `u_seq[t-k]` or any other re-indexing that would mean a classical
    Python-side history buffer is doing the memory's job."""
    src = inspect.getsource(idme.build_id_memory_eoc_circuit)
    loop_body = src.split("for t, u_t in enumerate(u_seq):", 1)[1]
    assert "u_seq[" not in loop_body


def test_primary_module_does_not_import_delay_taps_into_circuit_builder():
    """`delay_taps` may be imported by this MODULE (for the explicit
    classical-delay CONTROL function), but `build_id_memory_eoc_circuit`
    itself must never call it -- the primary result's feature vector comes
    only from `save_expectation_value` snapshots of the quantum state."""
    src = inspect.getsource(idme.build_id_memory_eoc_circuit)
    assert "delay_taps" not in src


def test_classical_control_is_a_separate_named_function_not_merged_silently():
    """`classical_delay_plus_processor` must exist as its own, separately
    callable, clearly-named function (not folded into
    `run_id_memory_eoc`'s own return value) -- callers must explicitly opt
    into the classical-memory control rather than get it by default."""
    assert "classical_delay_plus_processor" in dir(idme)
    primary_src = inspect.getsource(run_id_memory_eoc)
    assert "classical_delay_plus_processor" not in primary_src
    assert "delay_taps" not in primary_src


def test_classical_control_and_quantum_result_share_the_same_input_but_different_features():
    """The classical-delay control and the primary quantum result, given
    the SAME master_seed, must be driven by the SAME input trajectory (a
    fair comparison) but must NOT produce identical feature matrices (the
    classical control's features are literally raw delayed inputs, the
    quantum result's are Pauli expectation values -- they must differ)."""
    master_seed = 3
    ctrl = classical_delay_plus_processor(m_delay=5, N_P=3, kappa_processor=1.0, T=25, reps=1,
                                           max_weight=2, master_seed=master_seed)
    cfg = IDMemoryEOCConfig(L=4, N_P=3, kappa_processor=1.0, lambda_mp=0.3, reps_processor=1, n_taps=1)
    quantum = run_id_memory_eoc(cfg, T=25, master_seed=master_seed)
    assert np.array_equal(ctrl["u"], quantum.u), "control and primary result should share the same input trajectory"
    assert ctrl["X"].shape != quantum.X_combined.shape or not np.allclose(
        ctrl["X"][:, :ctrl["X"].shape[1]], quantum.X_combined[:, :ctrl["X"].shape[1]])
