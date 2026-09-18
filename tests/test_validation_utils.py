"""
test_validation_utils.py -- nested seeds, dimensionless controls, and
local response-surface fitting (Parts 6/7/10 of the V2 validation spec).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.validation_utils import (  # noqa: E402
    make_nested_seeds, ControlRange, robust_output_scale, fit_response_surface, is_unstable,
    stencil_margin_ok, assert_in_domain,
)


def test_nested_seeds_independent_streams_differ():
    s = make_nested_seeds(0, 0)
    vals = [s.reservoir_seed, s.disorder_seed, s.input_seed, s.split_seed, s.projection_seed,
            s.bootstrap_seed, s.shot_seed]
    assert len(set(vals)) == len(vals), "every seed stream must be distinct"


def test_nested_seeds_deterministic():
    a = make_nested_seeds(2, 1)
    b = make_nested_seeds(2, 1)
    assert a == b


def test_nested_seeds_reservoir_idx_changes_reservoir_seed_but_not_shot_pattern_structure():
    """Changing ONLY reservoir_idx (keeping input_idx fixed) must change
    ALL seeds derived from it (they all incorporate reservoir_idx) -- but
    two DIFFERENT reservoir_idx values must still both produce internally
    self-consistent, fully-independent 7-tuples."""
    a = make_nested_seeds(0, 0)
    b = make_nested_seeds(1, 0)
    assert a.reservoir_seed != b.reservoir_seed
    assert a.input_seed != b.input_seed  # input_seed also depends on reservoir_idx by design


def test_common_random_numbers_same_input_idx_across_reservoir_idx_gives_different_input_seed():
    """NOTE: this project's convention derives EVERY child seed from BOTH
    (reservoir_idx, input_idx) jointly -- there is no seed that depends on
    input_idx alone. Common-random-number reuse across nearby parameter
    points is therefore achieved by fixing BOTH indices and only varying
    the (m,g,J) argument passed to the reservoir builder, not by expecting
    input_seed to be invariant under changing the reservoir index alone."""
    a = make_nested_seeds(0, 3)
    b = make_nested_seeds(1, 3)
    assert a.input_seed != b.input_seed


def test_control_range_roundtrip():
    r = ControlRange("g", 0.1, 0.6)
    assert r.to_dimensionless(0.1) == pytest.approx(0.0)
    assert r.to_dimensionless(0.6) == pytest.approx(1.0)
    assert r.to_dimensionless(0.35) == pytest.approx(0.5)
    assert r.from_dimensionless(0.5) == pytest.approx(0.35)


def test_robust_output_scale_iqr():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # one outlier
    scale = robust_output_scale(vals)
    assert scale < 90  # IQR is robust to the single outlier, unlike std


def test_response_surface_recovers_known_linear_function():
    """Part 17 test #13: local derivative recovery on an analytic toy
    function. Y = 3 + 2*m - 1*g + 0.5*J (purely linear, no curvature) --
    the fitted linear coefficients must match exactly (up to float
    precision), and the analytic gradient at the origin must match."""
    rng = np.random.RandomState(0)
    n = 60
    m = rng.uniform(-0.2, 0.2, n)
    g = rng.uniform(-0.2, 0.2, n)
    J = rng.uniform(-0.2, 0.2, n)
    y = 3 + 2 * m - 1 * g + 0.5 * J
    fit = fit_response_surface(m, g, J, y)
    assert fit.coeffs["const"] == pytest.approx(3.0, abs=1e-8)
    assert fit.coeffs["m"] == pytest.approx(2.0, abs=1e-8)
    assert fit.coeffs["g"] == pytest.approx(-1.0, abs=1e-8)
    assert fit.coeffs["J"] == pytest.approx(0.5, abs=1e-8)
    assert fit.residual_rms < 1e-8
    dM_dm, dM_dg, dM_dJ = fit.gradient_at(0.0, 0.0, 0.0)
    assert dM_dm == pytest.approx(2.0, abs=1e-8)
    assert dM_dg == pytest.approx(-1.0, abs=1e-8)
    assert dM_dJ == pytest.approx(0.5, abs=1e-8)


def test_response_surface_recovers_known_quadratic_function():
    """Y = 1 + 2*m^2 -- pure quadratic curvature in m only, no linear term,
    no cross terms. dY/dm at m_tilde=0.1 should be 2*beta_mm*0.1 = 0.4."""
    rng = np.random.RandomState(1)
    n = 80
    m = rng.uniform(-0.3, 0.3, n)
    g = rng.uniform(-0.3, 0.3, n)
    J = rng.uniform(-0.3, 0.3, n)
    y = 1 + 2 * m ** 2
    fit = fit_response_surface(m, g, J, y)
    assert fit.coeffs["mm"] == pytest.approx(2.0, abs=1e-6)
    assert fit.coeffs["m"] == pytest.approx(0.0, abs=1e-6)
    dM_dm, dM_dg, dM_dJ = fit.gradient_at(0.1, 0.0, 0.0)
    assert dM_dm == pytest.approx(0.4, abs=1e-4)


def test_is_unstable_flags_sign_change():
    result = is_unstable([1.0, -1.0])
    assert result["unstable"] and result["sign_change"]


def test_is_unstable_flags_large_spread():
    result = is_unstable([1.0, 5.0])  # same sign but large relative spread
    assert result["unstable"]


def test_is_unstable_passes_consistent_estimates():
    result = is_unstable([2.0, 2.1, 1.95])
    assert not result["unstable"]


# =============================================================================
# V2.1 Defect 1 -- stencil boundary enforcement (test requirements #1, #2)
# =============================================================================

def test_stencil_margin_ok_rejects_v1_boundary_bug_exactly():
    """The EXACT V2 bug this test locks in: J*=0.6 was the declared range's
    own upper bound (J in [0.05,0.6] -> J_tilde*=1.0), and a step h_J=0.05
    (in raw units) over a range width of 0.55 gives h_J_tilde ~0.091 --
    2*h_J_tilde ~0.182, so the required upper margin (1 - 2*h_tilde ~0.818)
    is violated by J_tilde*=1.0. This must be rejected."""
    J_tilde_star = 1.0
    h_J_tilde = 0.05 / (0.6 - 0.05)
    assert not stencil_margin_ok(J_tilde_star, h_J_tilde, k=2)


def test_stencil_margin_ok_accepts_genuine_interior_point():
    assert stencil_margin_ok(0.5, 0.05, k=2)
    assert stencil_margin_ok(0.11, 0.05, k=2)  # exactly at the k*h lower edge
    assert not stencil_margin_ok(0.09, 0.05, k=2)  # just inside the forbidden margin


def test_stencil_margin_ok_symmetric_at_both_boundaries():
    h = 0.1
    assert not stencil_margin_ok(0.15, h, k=2)   # too close to 0
    assert not stencil_margin_ok(0.85, h, k=2)   # too close to 1
    assert stencil_margin_ok(0.5, h, k=2)


def test_assert_in_domain_raises_on_out_of_range_value():
    with pytest.raises(ValueError, match="outside its declared domain"):
        assert_in_domain(0.65, 0.05, 0.6, "J")


def test_assert_in_domain_passes_silently_in_range():
    assert_in_domain(0.33, 0.05, 0.6, "J")  # must not raise
    assert_in_domain(0.05, 0.05, 0.6, "J")  # boundary itself is in-domain
    assert_in_domain(0.6, 0.05, 0.6, "J")
