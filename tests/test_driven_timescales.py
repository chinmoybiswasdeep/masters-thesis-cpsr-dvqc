"""
test_driven_timescales.py -- V2.2 Phase 12 / test requirement #21:
driven-system correlation descriptors (as opposed to V2.1's isolated
memory/processor dynamics).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.driven_timescales import feature_series_autocorrelation, driven_system_timescales  # noqa: E402
from decoupled_qrc.directional_dqrc import DirectionalConfig  # noqa: E402
from decoupled_qrc.seeded_runner import run_directional_dqrc_seeded  # noqa: E402
from decoupled_qrc.validation_utils import make_nested_seeds  # noqa: E402


def test_autocorrelation_starts_at_one():
    x = np.random.RandomState(0).normal(size=50)
    steps, corr = feature_series_autocorrelation(x, max_lag=10)
    assert corr[0] == pytest.approx(1.0)
    assert len(corr) == 11


def test_autocorrelation_recovers_known_ar1_structure():
    """An AR(1)-like series x[t] = rho*x[t-1] + noise should show a
    decaying autocorrelation roughly consistent with rho -- a basic
    sanity check that the empirical estimator behaves as expected on
    synthetic data with KNOWN correlation structure."""
    rng = np.random.RandomState(1)
    rho = 0.7
    T = 2000
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + rng.normal(scale=0.5)
    steps, corr = feature_series_autocorrelation(x, max_lag=10)
    assert corr[1] == pytest.approx(rho, abs=0.1)
    assert corr[1] > corr[5] > corr[9]


def test_autocorrelation_flat_series_gives_zero():
    x = np.ones(30)
    steps, corr = feature_series_autocorrelation(x, max_lag=5)
    assert corr[0] == 1.0
    assert all(c == 0.0 for c in corr[1:])


def test_driven_system_timescales_on_real_circuit_run():
    cfg = DirectionalConfig(memory_variant="protected_integrable", N_M=2, N_P=5, g_processor=0.5,
                             J_processor=0.33, epsilon_M=0.5, theta=0.2, phi=0.8, ap_kind="xy")
    seeds = make_nested_seeds(0, 0)
    run = run_directional_dqrc_seeded(cfg, T=60, seeds=seeds, reset_period=None)
    result = driven_system_timescales(run, max_lag=15)
    assert result["memory"].identifiable in ("EXPONENTIAL", "NOT IDENTIFIABLE")
    assert result["processor"].identifiable in ("EXPONENTIAL", "NOT IDENTIFIABLE")
    assert not np.isnan(result["memory"].integrated_abs_time)
    assert not np.isnan(result["processor"].integrated_abs_time)
