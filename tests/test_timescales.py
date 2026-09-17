"""test_timescales.py -- tau_M/tau_P estimation (Part 13)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.timescales import processor_z_autocorrelation, fit_exponential_decay_timescale  # noqa: E402
from decoupled_qrc import directional_processor as dproc  # noqa: E402


def test_processor_autocorrelation_starts_at_one():
    params = dproc.sample_params(5, kappa_processor=1.0, term_seed=0, disorder_seed=0, reps=1)
    k, corr = processor_z_autocorrelation(params, n_steps=5)
    assert corr[0] == pytest.approx(1.0, abs=1e-9)


def test_fit_exponential_decay_recovers_known_tau():
    steps = np.arange(20)
    true_tau = 3.0
    corr = np.exp(-steps / true_tau)
    fit = fit_exponential_decay_timescale(steps, corr)
    assert fit.tau == pytest.approx(true_tau, rel=0.05)
    assert fit.r_squared > 0.99


def test_fit_exponential_decay_flags_poor_fit_on_noise():
    rng = np.random.RandomState(0)
    steps = np.arange(20)
    corr = rng.uniform(-1, 1, 20)  # pure noise, no decay structure
    fit = fit_exponential_decay_timescale(steps, corr)
    assert fit.r_squared < 0.8 or fit.fit_failed
