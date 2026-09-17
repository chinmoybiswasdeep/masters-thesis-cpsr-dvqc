"""
test_reproducibility.py -- same seed -> identical results, for every layer
of the stack (memory, processor, DQRC, IPC), following `eoc_config.py`'s own
"assert reproducibility rather than merely intend it" pattern.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.experiments import DQRCConfig, run_dqrc, run_ablation  # noqa: E402
from decoupled_qrc.memory import run_memory_register  # noqa: E402
from decoupled_qrc.processor import sample_processor_params, processor_chaos_diagnostics  # noqa: E402
from decoupled_qrc.ipc import compute_ipc  # noqa: E402
from decoupled_qrc.utils import make_seed_bundle  # noqa: E402
from qrc_qiskit import random_input, chrono_split  # noqa: E402


def test_seed_bundle_derivation_is_deterministic():
    a = make_seed_bundle(7)
    b = make_seed_bundle(7)
    assert a == b
    c = make_seed_bundle(8)
    assert a.reservoir_seed != c.reservoir_seed
    assert len({a.reservoir_seed, a.dataset_seed, a.measurement_seed}) == 3, "streams must not collide"


def test_run_memory_register_reproducible():
    u = random_input(30, seed=0)
    r1 = run_memory_register(N=4, u_seq=u, memory_mode="integrable", disorder_seed=5)
    r2 = run_memory_register(N=4, u_seq=u, memory_mode="integrable", disorder_seed=5)
    assert np.array_equal(r1.X, r2.X)


def test_processor_diagnostics_reproducible():
    p1 = sample_processor_params(4, kappa_processor=1.0, term_seed=3, disorder_seed=3)
    p2 = sample_processor_params(4, kappa_processor=1.0, term_seed=3, disorder_seed=3)
    d1 = processor_chaos_diagnostics(p1, n_ref_trials=5, ref_seed=0)
    d2 = processor_chaos_diagnostics(p2, n_ref_trials=5, ref_seed=0)
    assert d1 == d2


def test_run_dqrc_reproducible_same_master_seed():
    cfg = DQRCConfig(N_M=2, N_P=3, reps_processor=1, max_weight_readout=2)
    r1 = run_dqrc(cfg, T=12, master_seed=11)
    r2 = run_dqrc(cfg, T=12, master_seed=11)
    assert np.array_equal(r1.X, r2.X)
    assert np.array_equal(r1.u, r2.u)


def test_run_dqrc_different_master_seed_differs():
    cfg = DQRCConfig(N_M=2, N_P=3, reps_processor=1, max_weight_readout=2)
    r1 = run_dqrc(cfg, T=12, master_seed=11)
    r2 = run_dqrc(cfg, T=12, master_seed=12)
    assert not np.array_equal(r1.X, r2.X)


def test_compute_ipc_reproducible():
    u = random_input(200, seed=0)
    X = np.random.RandomState(0).normal(0, 1, size=(200, 5))
    train, val, test = chrono_split(200, washout=10, n_val=50, n_test=80, gap=7)
    r1 = compute_ipc(u, X, train, val, test, max_delay=4, max_degree=2, max_targets_per_degree=10,
                      n_surrogates=4, seed=1)
    r2 = compute_ipc(u, X, train, val, test, max_delay=4, max_degree=2, max_targets_per_degree=10,
                      n_surrogates=4, seed=1)
    assert r1.ipc_by_degree == r2.ipc_by_degree
    assert r1.total == r2.total
