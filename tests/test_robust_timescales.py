"""
test_robust_timescales.py -- V2.1 Phase 9 / test requirement #19: invalid
timescale-fit rejection, plus the model-free descriptors that must remain
well-defined even when the exponential model fails.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.robust_timescales import compute_robust_timescale_descriptors  # noqa: E402


def test_clean_exponential_decay_is_identified_and_gated_open():
    steps = np.arange(30)
    true_tau = 4.0
    corr = np.exp(-steps / true_tau)
    d = compute_robust_timescale_descriptors(steps, corr)
    assert d.identifiable == "EXPONENTIAL"
    assert d.exponential_fit_quality_ok
    assert d.exponential_tau == pytest.approx(true_tau, rel=0.05)
    assert d.exponential_r2 > 0.95


def test_pure_noise_is_rejected_as_not_identifiable():
    """This is exactly V2's own failure case (r2=0.177): a poor fit must be
    gated OFF, never reported as a usable tau."""
    rng = np.random.RandomState(0)
    steps = np.arange(20)
    corr = rng.uniform(-1, 1, 20)
    d = compute_robust_timescale_descriptors(steps, corr)
    assert d.identifiable == "NOT IDENTIFIABLE"
    assert np.isnan(d.exponential_tau)
    assert not d.exponential_fit_quality_ok


def test_oscillatory_never_fully_decaying_signal_is_rejected_but_still_descriptive():
    """V2's OTHER own failure case (tau_M=inf, r2=nan): a persistently
    oscillating, non-decaying correlator (no monotone envelope decay at
    all). The exponential fit must be rejected, but the model-free
    descriptors (integrated time, plateau, recurrence) must still be
    finite and informative -- never silently dropped just because the
    exponential model does not apply."""
    steps = np.arange(40)
    corr = np.cos(0.5 * steps)  # undamped oscillation, |C(t)| never decays toward 0
    d = compute_robust_timescale_descriptors(steps, corr)
    assert d.identifiable == "NOT IDENTIFIABLE"
    assert np.isnan(d.exponential_tau)
    assert not np.isnan(d.integrated_abs_time)
    assert not np.isnan(d.long_time_plateau)
    assert d.long_time_plateau > 0.3, "an undamped oscillation must show a substantial long-time plateau"
    assert not np.isnan(d.recurrence_amplitude)
    assert d.recurrence_amplitude > 0.3


def test_first_zero_crossing_recovered_for_damped_oscillation():
    steps = np.arange(30)
    corr = np.exp(-steps / 8.0) * np.cos(0.9 * steps)
    d = compute_robust_timescale_descriptors(steps, corr)
    assert not np.isnan(d.first_zero_crossing)
    assert 0 < d.first_zero_crossing < 5


def test_first_1e_crossing_matches_known_exponential():
    steps = np.arange(30)
    true_tau = 5.0
    corr = np.exp(-steps / true_tau)
    d = compute_robust_timescale_descriptors(steps, corr)
    assert d.first_1e_crossing == pytest.approx(true_tau, abs=1.0)


def test_envelope_decay_rate_negative_for_decaying_signal_and_nan_for_flat_signal():
    steps = np.arange(30)
    decaying = np.exp(-steps / 6.0)
    d_decay = compute_robust_timescale_descriptors(steps, decaying)
    assert d_decay.envelope_decay_rate < 0

    flat = np.ones(30)
    d_flat = compute_robust_timescale_descriptors(steps, flat)
    assert np.isnan(d_flat.envelope_decay_rate)


def test_integrated_abs_time_is_monotone_in_correlation_persistence():
    steps = np.arange(30)
    short = np.exp(-steps / 1.0)
    long_lived = np.exp(-steps / 15.0)
    d_short = compute_robust_timescale_descriptors(steps, short)
    d_long = compute_robust_timescale_descriptors(steps, long_lived)
    assert d_long.integrated_abs_time > d_short.integrated_abs_time
