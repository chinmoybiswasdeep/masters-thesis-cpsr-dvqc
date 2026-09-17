"""
test_memory_persistence.py -- the shift-register memory, once embedded in
the full id_memory_eoc circuit, must still behave EXACTLY like the
standalone `spatial_memory` register when disconnected from the processor
(lambda_mp=0) -- i.e. embedding must not silently disturb or reset memory
that the interface isn't supposed to touch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.id_memory_eoc import IDMemoryEOCConfig, run_id_memory_eoc  # noqa: E402
from decoupled_qrc.spatial_memory import run_spatial_memory  # noqa: E402
from qrc_qiskit import random_input  # noqa: E402


def test_disconnected_embedded_memory_matches_standalone_spatial_memory():
    """With lambda_mp=0 (no interface at all), the memory register's OWN
    features inside the combined circuit must be IDENTICAL (same seeds,
    same u_seq) to running `spatial_memory.run_spatial_memory` alone -- the
    processor's presence, with zero coupling, must have zero effect on
    memory (this is the same principle the DQRC repair's Jacobian
    experiment relies on: lambda_mp=0 must be a true, not approximate,
    disconnection)."""
    u = random_input(20, seed=1)
    # max_weight_mem=1 so the feature dictionary matches `spatial_memory`'s own
    # (single-qubit-only) convention exactly, label-for-label.
    cfg = IDMemoryEOCConfig(L=4, N_P=3, kappa_processor=1.0, lambda_mp=0.0, reps_processor=1, n_taps=1,
                             max_weight_mem=1)
    embedded = run_id_memory_eoc(cfg, T=20, master_seed=1)

    # standalone must use the SAME dataset seed the embedded run derives
    # internally (`make_seed_bundle(1).dataset_seed`) to get the same u_seq
    from decoupled_qrc.utils import make_seed_bundle
    dataset_seed = make_seed_bundle(1).dataset_seed
    u_standalone = random_input(20, seed=dataset_seed)
    standalone = run_spatial_memory(L=4, u_seq=u_standalone, gamma_M=0.0)

    assert np.array_equal(embedded.u, standalone.u)
    assert np.allclose(embedded.X_mem, standalone.X, atol=1e-9), (
        "embedded memory features diverged from standalone at lambda_mp=0 -- "
        "the interface is not a true identity at zero coupling")


def test_memory_persists_across_many_steps_with_nonzero_coupling():
    """Even WITH the processor connected, the memory register's own
    labeled features must still be present and finite at every timestep
    (a basic smoke/persistence check -- memory features should not become
    NaN/degenerate once the processor starts acting back on them)."""
    u = random_input(40, seed=2)
    cfg = IDMemoryEOCConfig(L=5, N_P=4, kappa_processor=1.0, interface_kind="xy", lambda_mp=0.3,
                             reps_processor=1, n_taps=1)
    run = run_id_memory_eoc(cfg, T=40, master_seed=2)
    assert np.all(np.isfinite(run.X_mem))
    assert run.X_mem.shape == (40, run.X_mem.shape[1])
    assert np.any(np.abs(run.X_mem) > 1e-3), "memory features collapsed to (near) zero everywhere"
