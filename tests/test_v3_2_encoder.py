"""V3.2 encoder: the properties the whole study rests on."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v3_2_encoder import (  # noqa: E402
    EncoderSpec, embed, expectation_values, is_density_matrix, local_pauli_ops, to_signed)


def test_encoder_state_is_a_valid_density_matrix():
    enc = EncoderSpec(n_copies=3)
    for u in (0.0, 0.25, 0.5, 0.9, 1.0):
        audit = is_density_matrix(enc.state(u))
        assert audit["ok"], (u, audit)


def test_expectation_of_Z_is_exactly_the_signed_input():
    """<Z_i> = s = 2u-1 on every copy: the encoder is affine by construction."""
    enc = EncoderSpec(n_copies=4)
    _, ops = local_pauli_ops(4, ("Z",))
    for u in np.linspace(0, 1, 11):
        vals = expectation_values(enc.state(u), ops)
        assert np.allclose(vals, to_signed(u), atol=1e-14)


def test_transverse_expectations_are_exactly_zero():
    enc = EncoderSpec(n_copies=3)
    _, ops = local_pauli_ops(3, ("X", "Y"))
    for u in (0.1, 0.5, 0.83):
        assert np.allclose(expectation_values(enc.state(u), ops), 0.0, atol=1e-15)


def test_single_qubit_channels_keep_local_observables_affine_in_u():
    """Any product of single-qubit maps -- which is what the processor reduces
    to at g=J=0 -- cannot create nonlinearity. This is why NL_0(0,0) = 0."""
    from scipy.linalg import expm
    rng = np.random.default_rng(4)
    n = 3
    enc = EncoderSpec(n_copies=n)
    H = sum(rng.normal() * embed(n, {q: p}) for q in range(n) for p in "XYZ")
    U = expm(-1j * 0.7 * H)
    _, ops = local_pauli_ops(n)
    us = np.linspace(0, 1, 40)
    F = np.array([expectation_values(U @ enc.state(u) @ U.conj().T, ops) for u in us])
    design = np.stack([np.ones_like(us), to_signed(us)], axis=1)
    resid = F - design @ np.linalg.lstsq(design, F, rcond=None)[0]
    assert np.max(np.abs(resid)) < 1e-12


def test_resources_report_the_structural_degree_bound():
    r = EncoderSpec(n_copies=5).resources()
    assert r["input_copies"] == 5 and r["max_instantaneous_degree"] == 5


def test_embed_rejects_out_of_range_qubits():
    with pytest.raises(ValueError):
        embed(3, {5: "X"})


def test_qubit_ordering_is_leftmost_first_not_qiskit_little_endian():
    """V3.2 uses qubit 0 = LEFTMOST tensor factor; Qiskit uses little-endian.
    Pin the relationship so the convention cannot drift silently."""
    qi = pytest.importorskip("qiskit.quantum_info")
    ours = embed(3, {0: "Z"})
    assert np.allclose(ours, qi.Operator(qi.Pauli("ZII")).data)
    assert not np.allclose(ours, qi.Operator(qi.Pauli("IIZ")).data)


def test_density_matrix_audit_detects_a_bad_matrix():
    bad = np.array([[0.6, 0.0], [0.0, 0.6]], dtype=complex)     # trace 1.2
    assert not is_density_matrix(bad)["ok"]
    neg = np.array([[1.4, 0.0], [0.0, -0.4]], dtype=complex)    # negative eigenvalue
    assert not is_density_matrix(neg)["ok"]
