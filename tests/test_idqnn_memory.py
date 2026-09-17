"""
test_idqnn_memory.py -- ID-SQM (IDQNN-inspired spatial transform), NEVER
called "IDQNN" outright in this module or its tests.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.idqnn_memory import (  # noqa: E402
    block_partition, apply_idqnn_spatial_transform, build_id_sqm_circuit, run_id_sqm,
)
from decoupled_qrc.spatial_memory import run_spatial_memory, delay_resolved_capacity  # noqa: E402
from decoupled_qrc import diagnostics as diag  # noqa: E402
from decoupled_qrc import utils  # noqa: E402
from qrc_qiskit import random_input  # noqa: E402


def test_block_partition_sizes():
    assert block_partition([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert block_partition([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_entangle_strength_zero_is_not_required_to_be_identity_but_ry_rz_are_fixed_seed():
    """The block subcircuit's Ry/Rz angles are seeded, not random-per-call --
    reproducibility check (a silent `np.random.seed`-free RNG leak would
    break this)."""
    u = random_input(10, seed=0)
    qc1, labels1, _ = build_id_sqm_circuit(5, u, gamma_M=0.0, block_size=2, transform_seed=3)
    qc2, labels2, _ = build_id_sqm_circuit(5, u, gamma_M=0.0, block_size=2, transform_seed=3)
    assert qc1 == qc2


def test_run_id_sqm_reproducible():
    u = random_input(20, seed=0)
    r1 = run_id_sqm(L=5, u_seq=u, gamma_M=0.1, block_size=2, transform_seed=1)
    r2 = run_id_sqm(L=5, u_seq=u, gamma_M=0.1, block_size=2, transform_seed=1)
    assert np.array_equal(r1["X"], r2["X"])


def test_run_id_sqm_rejects_statevector_method():
    with pytest.raises(ValueError, match="density_matrix"):
        run_id_sqm(L=5, u_seq=[0.1, 0.2], method="statevector")


def test_id_sqm_preserves_reasonable_ipc1_relative_to_bare_shift_register():
    """Part 6's own success criterion: the shallow-wide transform should
    not badly destroy the register's own accessible memory (IPC1) relative
    to the bare shift register at the same L/gamma -- a small, weak
    (entangle_strength=0.3) transform should leave IPC1 in the same
    ballpark, not collapse it."""
    CFG = utils.active_config()
    u = random_input(CFG.T_ipc, seed=0)
    bare = run_spatial_memory(L=5, u_seq=u, gamma_M=0.0)
    M_bare, NL_bare, _, _ = diag.ipc_MN(bare.u, bare.X, CFG, seed=0)

    transformed = run_id_sqm(L=5, u_seq=u, gamma_M=0.0, block_size=2, transform_seed=0, entangle_strength=0.3)
    M_t, NL_t, _, _ = diag.ipc_MN(transformed["u"], transformed["X"], CFG, seed=0)

    assert M_t > 0.5 * M_bare, f"ID-SQM transform destroyed too much IPC1: bare={M_bare}, transformed={M_t}"
