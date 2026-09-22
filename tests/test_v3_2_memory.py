"""V3.2 memory: both mechanisms, channel spectrum, synthetic memory recovery."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v3_2_encoder import is_density_matrix  # noqa: E402
from decoupled_qrc.v3_2_memory import (  # noqa: E402
    CONTROL_RANGE, MemorySpec, build_channel, channel_spectrum, delay_profile_descriptors,
    fractional_swap_matrix, memory_feature_ops, run_memory)


def test_fractional_swap_matches_the_v3_memory_bank_exactly():
    """Mechanism A must be the SAME unitary V3 used, not a re-derivation."""
    from decoupled_qrc.memory_bank import fractional_swap_matrix as v3_version
    for m in (0.0, 0.25, 0.5, 0.77, 1.0):
        assert np.allclose(fractional_swap_matrix(m), v3_version(m), atol=1e-15)


def test_fractional_swap_endpoints():
    assert np.allclose(fractional_swap_matrix(0.0), np.eye(4))
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    assert np.allclose(fractional_swap_matrix(1.0), swap)


@pytest.mark.parametrize("mechanism", ["fractional_swap", "leaky_collision"])
def test_states_stay_valid_density_matrices(mechanism):
    spec = MemorySpec(mechanism=mechanism, L=3)
    u = np.random.default_rng(1).uniform(0, 1, 40)
    _, _, rho = run_memory(spec, 0.45, u, disorder_seed=2)
    audit = is_density_matrix(rho)
    assert audit["ok"], audit


@pytest.mark.parametrize("mechanism", ["fractional_swap", "leaky_collision"])
def test_amplitude_damping_channels_compose_rather_than_sum(mechanism):
    """A flat Kraus list over several rails would give Tr rho = L - 1."""
    spec = MemorySpec(mechanism=mechanism, L=4)
    chan = build_channel(spec, 0.4, disorder_seed=3)
    rho = np.zeros((spec.dim, spec.dim), dtype=complex)
    rho[0, 0] = 1.0
    for _ in range(5):
        rho = chan.step(rho, 0.7)
    assert abs(np.trace(rho).real - 1.0) < 1e-12


def test_feature_set_does_not_depend_on_the_control():
    a, _ = memory_feature_ops(4, 2)
    b, _ = memory_feature_ops(4, 2)
    assert a == b and len(a) == 4 * 3 + 6 * 9


def test_control_outside_range_is_rejected():
    spec = MemorySpec()
    lo, hi = CONTROL_RANGE[spec.mechanism]
    with pytest.raises(ValueError):
        build_channel(spec, hi + 0.5, disorder_seed=1)


@pytest.mark.parametrize("mechanism", ["fractional_swap", "leaky_collision"])
def test_channel_spectrum_has_a_fixed_point_and_a_decaying_mode(mechanism):
    spec = MemorySpec(mechanism=mechanism, L=3)
    s = channel_spectrum(spec, 0.45, disorder_seed=4)
    assert abs(s["spectral_radius"] - 1.0) < 1e-8      # trace-preserving fixed point
    assert 0.0 <= s["lambda_2"] < 1.0 + 1e-9
    assert s["retention_time"] > 0.0


def test_leak_rate_changes_the_retention_time():
    """Mechanism B's control must alter the channel spectrum, not just scale it."""
    spec = MemorySpec(mechanism="leaky_collision", L=3)
    low = channel_spectrum(spec, 0.15, disorder_seed=4)["lambda_2"]
    high = channel_spectrum(spec, 0.9, disorder_seed=4)["lambda_2"]
    assert high > low + 1e-3


def test_synthetic_linear_memory_is_recovered_from_a_shift_register():
    """A full SWAP shift register must expose delays 0..L-1 and nothing more."""
    from decoupled_qrc.v3_2_ipc import RidgeBank
    from qrc_qiskit import chrono_split
    spec = MemorySpec(mechanism="fractional_swap", L=3)
    rng = np.random.default_rng(7)
    u = rng.uniform(0, 1, 700)
    X, _, _ = run_memory(spec, 0.95, u, disorder_seed=1)
    tr, va, te = chrono_split(len(u), washout=40, n_val=150, n_test=180, gap=8)
    bank = RidgeBank(X, tr, va, te)
    v = 2 * u - 1
    caps = []
    for tau in range(5):
        y = np.concatenate([np.zeros(tau), v[:len(v) - tau]])
        caps.append(bank.score(y)[0])
    assert caps[0] > 0.9 and caps[1] > 0.9          # within the register
    assert caps[4] < 0.3                            # beyond it


def test_delay_profile_descriptors_split_short_and_long():
    caps = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2, 4: 0.05}
    d = delay_profile_descriptors(caps, tau_min=2)
    assert abs(d["M_total"] - 2.55) < 1e-12
    assert abs(d["M_long"] - 0.75) < 1e-12
    assert abs(d["M_short"] - 1.8) < 1e-12
    assert d["fitted_retention_time"] > 0
    assert d["tau_min"] == 2


def test_memory_resources_are_reported():
    r = MemorySpec(L=4).resources()
    assert r["memory_qubits"] == 4 and r["input_copies"] == 1 and r["observables"] == 66
