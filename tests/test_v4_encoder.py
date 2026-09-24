"""V4 encoder: the polynomial claims, and the multilinearity the design rests on."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v4_encoder import (  # noqa: E402
    AffineInjection, ReuploadInjection, audit_encoder_state, density_matrix_audit, embed,
    expectations, multilinearity_report, polynomial_degrees, reachable_monomial,
    single_qubit_state)


def test_product_encoder_is_NOT_globally_affine():
    """The central correction: ((I+uZ)/2)^(x)n carries degrees up to n."""
    assert audit_encoder_state(1)["state_is_globally_affine"] is True
    for n in (2, 3, 4):
        a = audit_encoder_state(n)
        assert a["state_is_globally_affine"] is False, n
        assert a["global_readout"]["max_degree"] == n
        # but the LOCAL readout is affine
        assert a["local_readout"]["degrees_present"] == [1]
        assert a["local_readout"]["is_affine"] is True


def test_global_readout_degree_content_is_exact():
    """Gauss-Legendre is exact for polynomials, so a reported zero IS zero."""
    a2 = audit_encoder_state(2)
    assert a2["global_readout"]["degrees_present"] == [0, 2]
    a3 = audit_encoder_state(3)
    assert a3["global_readout"]["degrees_present"] == [1, 3]


def test_affine_injection_is_exactly_affine():
    inj = AffineInjection()
    rho0, drho = inj.rho0_drho()
    for u in (-1.0, -0.3, 0.0, 0.42, 1.0):
        assert np.allclose(inj.state(u), rho0 + u * drho, atol=1e-15)
    assert inj.n_copies == 1
    assert inj.resources()["nonlinear_degree_supplied"] == 1


def test_single_qubit_expectation_is_the_input():
    Z = embed(1, {0: "Z"})
    for u in np.linspace(-1, 1, 9):
        assert abs(expectations(single_qubit_state(u), [Z])[0] - u) < 1e-14


def test_reupload_injection_counts_its_copies_as_a_nonlinear_resource():
    inj = ReuploadInjection(n_copies=4)
    r = inj.resources()
    assert r["input_copies"] == 4 and r["nonlinear_degree_supplied"] == 4
    with pytest.raises(ValueError):
        ReuploadInjection(n_copies=0)


def test_polynomial_degrees_detects_a_known_polynomial():
    a = polynomial_degrees(lambda u: 3.0 * u ** 2 - 1.0)
    assert set(a["degrees_present"]) <= {0, 2}
    assert a["max_degree"] == 2 and a["is_affine"] is False
    b = polynomial_degrees(lambda u: 2.0 * u + 1.0)
    assert b["is_affine"] is True


def test_affine_channel_chain_is_multilinear_in_the_inputs():
    """Quantum channels are linear in rho, so one affine injection per step
    gives degree <= 1 in EVERY individual past input."""
    from decoupled_qrc.v4_architecture import MemorySpec, MemoryRoute

    def feat(us):
        r = MemoryRoute(MemorySpec(L_R=3), 0.8, 3)
        f = None
        for u in us:
            f = r.step(float(u))
        return float(f[0])

    rep = multilinearity_report(feat, 3, base=[0.3, -0.5, 0.15])
    assert rep["multilinear"] is True
    assert all(d <= 1 for d in rep["max_degree_per_input"].values())


def test_reachability_rule_matches_the_capability_boundary():
    assert reachable_monomial({3: 1})["reachable"] is True
    assert reachable_monomial({0: 2, 3: 1})["reachable"] is True
    assert reachable_monomial({0: 4, 2: 1, 5: 1})["reachable"] is True
    bad = reachable_monomial({3: 2})
    assert bad["reachable"] is False and bad["blocking_delays"] == [3]
    assert reachable_monomial({2: 2, 4: 1})["reachable"] is False


def test_density_matrix_audit_detects_bad_matrices():
    assert density_matrix_audit(single_qubit_state(0.3))["ok"] is True
    assert not density_matrix_audit(np.diag([0.6, 0.6]).astype(complex))["ok"]
    assert not density_matrix_audit(np.diag([1.4, -0.4]).astype(complex))["ok"]


def test_embed_rejects_bad_indices():
    with pytest.raises(ValueError):
        embed(2, {5: "X"})
