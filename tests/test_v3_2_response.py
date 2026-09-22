"""V3.2 response estimation: normalisation, stencils, derivative recovery, CRN."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v3_2_response import (  # noqa: E402
    DerivativeVerdict, Jacobian, assess_stability, denormalize, derivative_estimates,
    equivalence_bound, interior_margin, normalize, stencil_points)


def test_normalisation_round_trips():
    for p in (0.1, 0.45, 0.9):
        assert abs(denormalize(normalize(p, 0.05, 0.95), 0.05, 0.95) - p) < 1e-12
    assert normalize(0.05, 0.05, 0.95) == 0.0
    assert normalize(0.95, 0.05, 0.95) == 1.0


def test_empty_range_is_rejected():
    with pytest.raises(ValueError):
        normalize(0.5, 0.7, 0.7)


def test_interior_margin_is_zero_at_a_boundary():
    assert interior_margin(0.95, 0.05, 0.95) == 0.0
    assert abs(interior_margin(0.5, 0.05, 0.95) - 0.5) < 1e-12


def test_boundary_candidate_is_rejected_by_the_stencil():
    """V3.1 selected m* = 0.95 with m in [0.1, 0.95]: no symmetric five-point
    stencil fits there, which is why its derivative was unstable."""
    with pytest.raises(ValueError):
        stencil_points(0.95, 0.1, 0.95, h=0.08)
    with pytest.raises(ValueError):
        stencil_points(0.1, 0.1, 0.95, h=0.08)


def test_interior_candidate_gives_a_symmetric_stencil():
    pts = stencil_points(0.5, 0.05, 0.95, h=0.1)
    assert set(pts) == {"m2h", "m1h", "c", "p1h", "p2h"}
    assert abs(pts["c"] - 0.5) < 1e-12
    assert abs((pts["p1h"] - pts["c"]) - (pts["c"] - pts["m1h"])) < 1e-12
    assert abs((pts["p2h"] - pts["c"]) - 2 * (pts["p1h"] - pts["c"])) < 1e-12


def test_local_derivative_recovery_on_an_analytic_function():
    """All four estimators must recover a known derivative."""
    lo, hi, h = 0.0, 1.0, 0.05
    f = lambda x: np.sin(3.0 * x) + 0.5 * x ** 2          # noqa: E731
    dfdx = lambda x: 3.0 * np.cos(3.0 * x) + x            # noqa: E731
    for p in (0.3, 0.5, 0.7):
        pts = stencil_points(p, lo, hi, h)
        vals = {k: f(v) for k, v in pts.items()}
        est = derivative_estimates(vals, h)
        # central differences are O(h^2); central_large uses 2h so it is ~4x worse
        for name, v in est.items():
            assert abs(v - dfdx(p)) < 4e-2, (name, p, v, dfdx(p))
        # the O(h^4) rule must be orders of magnitude sharper than the O(h^2) ones
        assert abs(est["five_point"] - dfdx(p)) < 1e-4
        assert abs(est["five_point"] - dfdx(p)) < abs(est["central_small"] - dfdx(p)) / 20


def test_derivative_of_an_exactly_linear_function_is_exact():
    pts = stencil_points(0.5, 0.0, 1.0, 0.1)
    est = derivative_estimates({k: 3.0 * v + 1.0 for k, v in pts.items()}, 0.1)
    for v in est.values():
        assert abs(v - 3.0) < 1e-9


def test_stability_flags_sign_disagreement():
    v = assess_stability("d", {"a": 1.0, "b": -1.1, "c": 0.9})
    assert not v.stable and any("sign" in r for r in v.reasons)


def test_stability_flags_wide_spread():
    v = assess_stability("d", {"a": 1.0, "b": 4.0, "c": 1.1})
    assert not v.stable and any("spread" in r for r in v.reasons)


def test_stability_flags_a_ci_containing_zero():
    nested = {0: [0.5, -0.4], 1: [-0.3, 0.6], 2: [0.1, -0.2]}
    v = assess_stability("d", {"a": 0.05, "b": 0.06, "c": 0.055}, nested, n_boot=400)
    assert not v.stable and any("CI contains zero" in r for r in v.reasons)


def test_stability_flags_single_seed_dominance():
    nested = {0: [10.0, 10.2], 1: [0.01, 0.02], 2: [0.0, 0.01]}
    v = assess_stability("d", {"a": 3.3, "b": 3.4, "c": 3.35}, nested, n_boot=400)
    assert v.seed_dominated and not v.stable


def test_a_clean_derivative_is_stable():
    nested = {0: [2.0, 2.1], 1: [1.9, 2.05], 2: [2.05, 1.95]}
    v = assess_stability("d", {"a": 2.0, "b": 2.02, "c": 1.99}, nested, n_boot=600)
    assert v.stable, v.reasons
    assert v.upper_abs_bound > 0


def test_equivalence_bound_uses_the_upper_bound_not_significance():
    """A wide, non-significant derivative must NOT count as suppressed."""
    wide = assess_stability("dNL_dm", {"a": 0.01, "b": 0.02},
                            {0: [0.5, -0.5], 1: [-0.6, 0.6]}, n_boot=400)
    eb = equivalence_bound(wide, reference=1.0, margin_fraction=0.25)
    assert not eb["suppressed"]

    tight = assess_stability("dNL_dm", {"a": 0.01, "b": 0.012},
                             {0: [0.011, 0.010], 1: [0.012, 0.009]}, n_boot=400)
    assert equivalence_bound(tight, reference=1.0, margin_fraction=0.25)["suppressed"]


def _verdict(name, value, stable=True):
    v = DerivativeVerdict(name=name, estimates={"a": value}, point_estimate=value)
    v.stable = stable
    return v


def test_jacobian_geometry_is_orthogonal_for_an_ideal_response():
    J = Jacobian(dM_dm=_verdict("dM_dm", 2.0), dM_dg=_verdict("dM_dg", 0.0),
                 dM_dJ=_verdict("dM_dJ", 0.0), dNL_dm=_verdict("dNL_dm", 0.0),
                 dNL_dg=_verdict("dNL_dg", 1.5), dNL_dJ=_verdict("dNL_dJ", 1.0))
    geo = J.geometry()
    assert abs(geo["angle_deg"] - 90.0) < 1e-6
    assert geo["off_diagonal_norm"] < 1e-12


def test_selectivity_is_not_evaluable_from_unstable_derivatives():
    J = Jacobian(dM_dm=_verdict("dM_dm", 2.0, stable=False), dM_dg=_verdict("dM_dg", 0.1),
                 dM_dJ=_verdict("dM_dJ", 0.1), dNL_dm=_verdict("dNL_dm", 0.1),
                 dNL_dg=_verdict("dNL_dg", 1.5), dNL_dJ=_verdict("dNL_dJ", 1.0))
    assert J.selectivity_ratios()["R_M"]["kind"] == "not_evaluable"
