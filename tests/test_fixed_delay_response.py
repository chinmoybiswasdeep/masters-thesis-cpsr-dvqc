"""
test_fixed_delay_response.py -- V2.2 Phase 9/10 / test requirements #14,
#15: five-point derivatives and local quadratic derivatives, computed via
the frozen protocol.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.fixed_delay_response import _derivative_methods, run_stencil_for_seed, aggregate_stencils  # noqa: E402
from decoupled_qrc.validation_utils import ControlRange, make_nested_seeds  # noqa: E402


def test_derivative_methods_recover_known_linear_function():
    """f(x) = 3 + 5x -- every method (small CD, large CD, 5-point,
    quadratic) must recover slope=5 exactly (up to float precision) for a
    purely linear function."""
    h_small, h_large = 0.05, 0.10
    f = lambda x: 3 + 5 * x
    values = {-h_large: f(-h_large), -h_small: f(-h_small), 0.0: f(0.0), h_small: f(h_small), h_large: f(h_large)}
    methods = _derivative_methods(values, h_small, h_large)
    for name, val in methods.items():
        assert val == pytest.approx(5.0, abs=1e-8), f"{name} gave {val}, expected 5.0"


def test_derivative_methods_recover_known_quadratic_curvature():
    """f(x) = 2 + 3x + 4x^2 -- the derivative AT x=0 is exactly 3 (the
    linear coefficient); the quadratic-fit method must recover this
    despite real curvature present, and the 5-point stencil (which is
    exact for cubics, hence exact for quadratics) must too."""
    h_small, h_large = 0.05, 0.10
    f = lambda x: 2 + 3 * x + 4 * x ** 2
    values = {-h_large: f(-h_large), -h_small: f(-h_small), 0.0: f(0.0), h_small: f(h_small), h_large: f(h_large)}
    methods = _derivative_methods(values, h_small, h_large)
    # central differences have zero bias for a pure quadratic (the bias term depends on the THIRD
    # derivative, which is 0 here) -- every method should recover the exact linear coefficient.
    for name, val in methods.items():
        assert val == pytest.approx(3.0, abs=1e-6), f"{name} gave {val}, expected 3.0"


def test_derivative_methods_five_point_nan_when_steps_not_uniform():
    """The classic 5-point formula assumes h_large == 2*h_small -- if that
    relationship doesn't hold, it must be reported as NaN (not a silently
    wrong number)."""
    h_small, h_large = 0.05, 0.30  # NOT 2*h_small
    f = lambda x: 1 + 2 * x
    values = {-h_large: f(-h_large), -h_small: f(-h_small), 0.0: f(0.0), h_small: f(h_small), h_large: f(h_large)}
    methods = _derivative_methods(values, h_small, h_large)
    assert np.isnan(methods["five_point"])
    assert methods["small_cd"] == pytest.approx(2.0, abs=1e-8)  # still valid


def test_run_stencil_for_seed_produces_center_and_all_axes():
    m_range, g_range, J_range = ControlRange("m", 0.1, 1.0), ControlRange("g", 0.05, 0.6), ControlRange("J", 0.05, 0.6)
    seeds = make_nested_seeds(0, 0)
    stencil = run_stencil_for_seed(2, 5, 0.2, 0.8, "xy", 0.5, 0.3, 0.3, m_range, g_range, J_range,
                                    h_small=0.05, h_large=0.10, seeds=seeds, T=70, washout=12, n_val=15,
                                    n_test=20, max_delay=3, max_degree=1, max_targets_per_degree=4,
                                    n_surrogates=6)
    assert set(stencil.per_axis.keys()) == {"m", "g", "J"}
    for axis, pts in stencil.per_axis.items():
        assert len(pts) == 4
        offsets = sorted(p.offset_tilde for p in pts)
        assert offsets == pytest.approx([-0.10, -0.05, 0.05, 0.10], abs=1e-9)


def test_aggregate_stencils_returns_stability_for_every_axis_and_target():
    m_range, g_range, J_range = ControlRange("m", 0.1, 1.0), ControlRange("g", 0.05, 0.6), ControlRange("J", 0.05, 0.6)
    seeds = make_nested_seeds(1, 0)
    stencil = run_stencil_for_seed(2, 5, 0.2, 0.8, "xy", 0.5, 0.3, 0.3, m_range, g_range, J_range,
                                    h_small=0.05, h_large=0.10, seeds=seeds, T=70, washout=12, n_val=15,
                                    n_test=20, max_delay=3, max_degree=1, max_targets_per_degree=4,
                                    n_surrogates=6)
    agg = aggregate_stencils([stencil], target_delay=0)
    assert set(agg.keys()) == {("m", "M"), ("g", "M"), ("J", "M"), ("m", "NL_tau0"), ("g", "NL_tau0"), ("J", "NL_tau0")}
    for key, est in agg.items():
        assert "within_seed" in est.stability
        assert len(est.stability["within_seed"]) == 1  # one seed here
