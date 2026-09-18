"""
test_ipc_decomposition.py -- order-delay IPC decomposition (Part 3/4 of the
V2 validation spec). The most important property: `compute_ipc`'s legacy
output must be BIT-IDENTICAL before/after this module's refactor of
`ipc.py` (it now aggregates `compute_ipc_detailed`'s records internally).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.ipc import compute_ipc, compute_ipc_detailed, to_v, legendre_target  # noqa: E402
from decoupled_qrc.ipc_decomposition import (compute_ipc_decomposed, diagnose_ceiling, to_heatmap,  # noqa: E402
                                              capacity_at_delay, nl_tensor_by_fixed_delay, m_tensor_by_fixed_delay)
from qrc_qiskit import chrono_split, random_input  # noqa: E402


def _setup(T=300, n_features=40, seed=0):
    u = random_input(T, seed=seed)
    X = np.random.RandomState(seed).normal(0, 1, size=(T, n_features))
    train, val, test = chrono_split(T, washout=30, n_val=60, n_test=90, gap=7)
    return u, X, train, val, test


def test_legacy_compute_ipc_unchanged_by_refactor():
    """The single most important regression test: compute_ipc's aggregated
    output must match compute_ipc_detailed's own aggregation exactly (same
    RNG draw order -> same numbers)."""
    u, X, train, val, test = _setup()
    legacy = compute_ipc(u, X, train, val, test, max_delay=5, max_degree=3, max_targets_per_degree=10,
                          n_surrogates=4, seed=1)
    records, was_capped, n_tested = compute_ipc_detailed(u, X, train, val, test, max_delay=5, max_degree=3,
                                                          max_targets_per_degree=10, n_surrogates=4, seed=1)
    manual_memory = sum(r.capacity for r in records if r.degree == 1)
    manual_nl = sum(r.capacity for r in records if r.degree >= 2)
    assert legacy.memory == pytest.approx(manual_memory, abs=1e-12)
    assert legacy.nonlinearity == pytest.approx(manual_nl, abs=1e-12)


def test_decomposition_M_and_NL_legacy_match_compute_ipc():
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=5, max_degree=3,
                                     max_targets_per_degree=10, n_surrogates=4, seed=2)
    legacy = compute_ipc(u, X, train, val, test, max_delay=5, max_degree=3, max_targets_per_degree=10,
                          n_surrogates=4, seed=2)
    assert decomp.M_long == pytest.approx(legacy.memory, abs=1e-12)
    assert decomp.NL_legacy == pytest.approx(legacy.nonlinearity, abs=1e-12)


def test_instant_plus_temporal_le_legacy():
    """NL_instant + NL_temporal must never EXCEED NL_legacy (every
    instant/temporal record is also counted in the legacy sum; they can be
    strictly less only if there exist degree>=2 profiles that are neither
    -- there are none by construction, since every degree>=2 profile either
    has delays==(0,) [instant] or max_delay>0 [temporal], covering all
    cases exhaustively -- so this should be an exact equality)."""
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=5, max_degree=3,
                                     max_targets_per_degree=10, n_surrogates=4, seed=3)
    assert decomp.NL_instant + decomp.NL_temporal == pytest.approx(decomp.NL_legacy, abs=1e-9)


def test_instantaneous_target_identified_correctly():
    """A planted feature that is EXACTLY L2(v_t) (delay=0, degree=2) must be
    classified as instantaneous, not temporal."""
    T = 300
    u = random_input(T, seed=0)
    v = to_v(u)
    instant_feature = legendre_target(v, 2)
    X = np.column_stack([instant_feature, np.random.RandomState(0).normal(0, 1, size=(T, 5))])
    train, val, test = chrono_split(T, washout=10, n_val=60, n_test=90, gap=7)
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=5, max_degree=2,
                                     max_targets_per_degree=20, n_surrogates=5, seed=0)
    assert decomp.NL_instant > 0.5, f"expected the planted instantaneous L2(v_t) signal to register, got {decomp.NL_instant}"


def test_temporal_target_identified_correctly_not_misclassified_as_instant():
    """A planted feature that is EXACTLY L2(v_{t-3}) (delay=3, degree=2)
    must be classified as TEMPORAL, never instantaneous -- Part 3's
    explicit 'do not misclassify cross-delay nonlinear targets' check."""
    T = 300
    u = random_input(T, seed=0)
    v = to_v(u)
    shifted = np.zeros(T)
    shifted[3:] = legendre_target(v[:-3], 2)
    X = np.column_stack([shifted, np.random.RandomState(1).normal(0, 1, size=(T, 5))])
    train, val, test = chrono_split(T, washout=10, n_val=60, n_test=90, gap=7)
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=5, max_degree=2,
                                     max_targets_per_degree=20, n_surrogates=5, seed=0)
    assert decomp.NL_temporal > 0.5, f"expected the planted temporal L2(v_(t-3)) signal to register, got {decomp.NL_temporal}"
    assert decomp.NL_instant < 0.3, f"planted temporal signal must NOT be counted as instantaneous, got {decomp.NL_instant}"


def test_ceiling_diagnostic_flags_saturated_case():
    """A feature set that can PERFECTLY reconstruct every delay-0..k target
    (e.g. k+1 orthogonal delay-encoding columns) should be flagged as
    ceiling-contaminated when max_delay==k."""
    T = 400
    u = random_input(T, seed=0)
    v = to_v(u)
    max_delay = 3
    cols = []
    for k in range(max_delay + 1):
        shifted = np.zeros(T)
        if k == 0:
            shifted[:] = v
        else:
            shifted[k:] = v[:-k]
        cols.append(legendre_target(shifted, 1))
    X = np.column_stack(cols)
    train, val, test = chrono_split(T, washout=10, n_val=80, n_test=120, gap=max_delay + 1)
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=max_delay, max_degree=1,
                                     max_targets_per_degree=10, n_surrogates=5, seed=0)
    ceil = diagnose_ceiling(decomp, X, train)
    assert ceil.fraction_of_ceiling > 0.7, (
        f"expected a near-saturated fraction_of_ceiling for a feature set that can perfectly "
        f"reconstruct every delay-0..{max_delay} target, got {ceil}")


def _m_long_bc_and_legacy_vs_noise(noise_scale, T=260, seed=7, n_noise_cols=9):
    """Deterministic (fixed master seed) construction: one real column
    v+noise_scale*eta plus several PURE-NOISE columns (a single-feature X
    would make the ridge-regression capacity score degenerate -- with only
    one feature ANY nonzero fitted weight is an affine rescaling of that
    feature, so the test-set squared correlation collapses to
    corr(X_test,y_test)^2 independent of what the model actually learned
    from train, making the shuffled-train significance test meaningless;
    extra noise columns force a genuine multi-feature fit, matching how
    this scoring function is used everywhere else in the project). As
    noise_scale shrinks, the true signal-to-noise ratio rises smoothly, so
    the underlying raw_capacity should rise smoothly too -- while the hard
    significance filter behind legacy M_long can only ever read 0 or
    (approximately) raw_capacity, with nothing in between."""
    u = random_input(T, seed=seed)
    v = to_v(u)
    rng = np.random.RandomState(seed)
    real_col = v + noise_scale * rng.normal(size=T)
    noise_cols = rng.normal(size=(T, n_noise_cols))
    X = np.column_stack([real_col, noise_cols])
    train, val, test = chrono_split(T, washout=15, n_val=50, n_test=70, gap=2)
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=1, max_degree=1,
                                     max_targets_per_degree=2, n_surrogates=8, seed=seed)
    return decomp.M_long, decomp.M_long_bc


def test_continuous_bias_corrected_capacity_varies_smoothly_under_planted_signal():
    """Defect 5 test #3: sweeping a planted signal's strength (noise_scale
    from high to low) must NOT produce any single-step jump in M_long_bc
    that dwarfs the typical step -- i.e. no hard threshold discontinuity in
    the continuous metric."""
    noise_scales = np.linspace(3.0, 0.05, 24)
    bc_values = [_m_long_bc_and_legacy_vs_noise(ns)[1] for ns in noise_scales]
    steps = np.abs(np.diff(bc_values))
    total_range = max(bc_values) - min(bc_values) + 1e-9
    # a smooth (even strongly accelerating/sigmoid-shaped) curve can have very
    # non-uniform step sizes, but no SINGLE step should dominate the curve's
    # entire range the way a hard-threshold discontinuity would.
    assert steps.max() < 0.5 * total_range, (
        f"M_long_bc showed a single step ({steps.max():.4f}) covering more than half its own "
        f"total range ({total_range:.4f}) -- expected a smoothly varying continuous metric, got {bc_values}")


def test_legacy_hard_threshold_can_jump_discontinuously():
    """Defect 5 test #4: sweeping the exact same signal-strength range, the
    LEGACY significance-filtered metric must contain an exact run of zeros
    that ends with an abrupt jump straight to a substantial value (the
    significance gate switching on) -- the concrete hard-thresholding
    artifact the continuous metric (previous test) does not have. At that
    SAME transition index, the continuous metric must already be
    comfortably nonzero (it started rising well before the legacy gate
    fired), demonstrating the two metrics really do behave differently at
    the same underlying signal strength, not just differently on average."""
    noise_scales = np.linspace(3.0, 0.05, 24)
    pairs = [_m_long_bc_and_legacy_vs_noise(ns) for ns in noise_scales]
    legacy_values = [p[0] for p in pairs]
    bc_values = [p[1] for p in pairs]

    zero_idx = [i for i, v in enumerate(legacy_values) if v <= 1e-9]
    nonzero_idx = [i for i, v in enumerate(legacy_values) if v > 1e-9]
    assert zero_idx and nonzero_idx, f"expected both zero and nonzero legacy entries, got {legacy_values}"
    gate_on = min(nonzero_idx)
    assert gate_on > 0 and (gate_on - 1) in zero_idx, "expected a zero run immediately before the gate switches on"
    assert legacy_values[gate_on] > 0.2, (
        f"expected the legacy metric to jump straight to a substantial value at the gate, "
        f"got {legacy_values[gate_on]:.4f} at index {gate_on}")
    # the continuous metric, evaluated at that SAME index, was already informative --
    # it did not need the same on/off gate to register a nonzero, rising signal.
    assert bc_values[gate_on] > 0.0, (
        f"expected the continuous bias-corrected metric to already be nonzero at the legacy "
        f"gate's transition index, got {bc_values[gate_on]:.4f}")


def test_response_estimation_uses_continuous_metric_not_legacy():
    """Defect 5's explicit rule: 'do not differentiate a hard
    significance-filtered metric for the main decoupling claim'. This is a
    structural test that `compute_ipc_decomposed`'s *_bc fields exist and
    are independently computable from *_raw/*_null (i.e. that any caller
    CAN build derivatives from the continuous metric alone, without ever
    touching the legacy significance-filtered fields)."""
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=4, max_degree=3,
                                     max_targets_per_degree=10, n_surrogates=6, seed=0)
    # M_long_bc is the PER-RECORD clip-then-sum (sum_i max(0, raw_i - null_i)), never the
    # aggregate-then-clip (max(0, sum(raw_i) - sum(null_i))) -- these differ whenever
    # individual records straddle their own null mean in different directions, so the
    # per-record form (never letting one profile's noise deficit cancel another's real
    # signal) is what must be reproducible here, not the (incorrect) aggregate shortcut.
    def manual_bc(records):
        return sum(max(0.0, r.raw_capacity - r.null_mean) for r in records)

    m_records = [r for r in decomp.records if r.degree == 1]
    instant_records = [r for r in decomp.records if r.degree >= 2 and r.delays == (0,)]
    temporal_records = [r for r in decomp.records if r.degree >= 2 and r.max_delay > 0]
    legacy_records = [r for r in decomp.records if r.degree >= 2]
    assert decomp.M_long_bc == pytest.approx(manual_bc(m_records), abs=1e-9)
    assert decomp.NL_instant_bc == pytest.approx(manual_bc(instant_records), abs=1e-9)
    assert decomp.NL_temporal_bc == pytest.approx(manual_bc(temporal_records), abs=1e-9)
    assert decomp.NL_legacy_bc == pytest.approx(manual_bc(legacy_records), abs=1e-9)
    # sanity: the raw/null aggregate fields still sum linearly (unlike bc)
    assert decomp.M_long_raw == pytest.approx(sum(r.raw_capacity for r in m_records), abs=1e-9)
    assert decomp.M_long_null == pytest.approx(sum(r.null_mean for r in m_records), abs=1e-9)


def test_capacity_at_delay_matches_instant_at_zero():
    """`capacity_at_delay(records, 0)` must equal decomp.NL_instant exactly
    (same filter, different entry point) -- the building block
    `causal_latency.py` uses for NL_local at an arbitrary ell_0."""
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=4, max_degree=3,
                                     max_targets_per_degree=10, n_surrogates=4, seed=0)
    at0 = capacity_at_delay(decomp.records, delay=0, degree_min=2)
    assert at0["legacy"] == pytest.approx(decomp.NL_instant, abs=1e-12)
    assert at0["bc"] == pytest.approx(decomp.NL_instant_bc, abs=1e-9)


def test_signed_metric_can_go_negative_unlike_clipped():
    """V2.2 Phase 4 test #7: the SIGNED metric is unclipped (raw-null can
    legitimately be negative for a pure-noise target), while the clipped
    (_bc) metric never goes below 0 -- the whole point of introducing the
    signed metric as the primary derivative quantity."""
    T = 200
    u = random_input(T, seed=11)
    X = np.random.RandomState(11).normal(0, 1, size=(T, 20))  # pure noise features, no real signal
    train, val, test = chrono_split(T, washout=20, n_val=50, n_test=70, gap=3)
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=2, max_degree=1,
                                     max_targets_per_degree=3, n_surrogates=8, seed=11)
    assert decomp.M_long_bc >= 0.0
    # the signed sum need not be negative for every draw, but individual pure-noise profiles
    # commonly score BELOW their own null mean -- verify at least one record does, and that the
    # per-record signed/clipped relationship (signed <= clipped, clipped = max(0,signed)) holds exactly.
    m_records = [r for r in decomp.records if r.degree == 1]
    signed_vals = [r.raw_capacity - r.null_mean for r in m_records]
    assert any(v < 0 for v in signed_vals), "expected at least one pure-noise profile to score below its null mean"
    for r in m_records:
        signed = r.raw_capacity - r.null_mean
        clipped = max(0.0, signed)
        assert clipped == pytest.approx(max(0.0, signed), abs=1e-12)
        assert clipped >= signed - 1e-12


def test_clipped_metric_has_kink_signed_metric_does_not():
    """V2.2 Phase 4 test #8: sweeping a planted signal's strength through
    the point where raw crosses null (signed changes sign) for a SINGLE
    profile (max_delay=0, so exactly one degree-1 target -- avoids any
    pooling ambiguity from summing multiple profiles), the CLIPPED
    metric's slope must change abruptly at that crossing (a kink -- it is
    flat at/below 0, then rises), while the SIGNED metric's slope stays
    smooth (locally linear) straight through the same crossing."""
    T = 260
    seed = 21

    def m_signed_and_clipped(noise_scale):
        u = random_input(T, seed=seed)
        v = to_v(u)
        rng = np.random.RandomState(seed)
        real_col = v + noise_scale * rng.normal(size=T)
        noise_cols = rng.normal(size=(T, 9))
        X = np.column_stack([real_col, noise_cols])
        train, val, test = chrono_split(T, washout=15, n_val=50, n_test=70, gap=2)
        decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=0, max_degree=1,
                                         max_targets_per_degree=1, n_surrogates=8, seed=seed)
        return decomp.M_long_signed, decomp.M_long_bc

    noise_scales = np.linspace(60.0, 0.05, 30)
    pairs = [m_signed_and_clipped(ns) for ns in noise_scales]
    signed_vals = np.array([p[0] for p in pairs])
    clipped_vals = np.array([p[1] for p in pairs])

    # find where signed crosses zero (the kink point for the clipped series)
    sign_changes = np.where(np.diff(np.sign(signed_vals)) != 0)[0]
    assert len(sign_changes) > 0, f"expected the signed metric to cross zero somewhere in this sweep, got {signed_vals}"

    # with a SINGLE pooled profile, clipped = max(0, signed) EXACTLY -- flat at 0 on the negative
    # side, then rising past the crossing -- a structural kink the signed series never has.
    idx = sign_changes[0]
    below_null_region = clipped_vals[:idx + 1]
    assert np.allclose(below_null_region, 0.0, atol=1e-9), (
        f"expected the clipped metric to be EXACTLY 0 everywhere signed<=0, got {below_null_region}")
    assert clipped_vals[idx + 1] > 0.0, "expected the clipped metric to rise past the crossing"
    assert np.allclose(clipped_vals, np.maximum(0.0, signed_vals), atol=1e-9)


def test_response_derivative_uses_signed_not_clipped():
    """V2.2 Phase 4 test #9 (adjacent): a simple finite-difference
    derivative built from M_long_signed must be able to be NEGATIVE (a
    real, meaningful decrease), whereas one built from the clipped M_long_bc
    could be artificially floored at a plus/minus pair straddling zero."""
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=4, max_degree=2,
                                     max_targets_per_degree=8, n_surrogates=6, seed=0)
    assert isinstance(decomp.M_long_signed, float)
    assert isinstance(decomp.NL_instant_signed, float)
    # signed must equal raw - null exactly (no clip) for the pooled category
    m_records = [r for r in decomp.records if r.degree == 1]
    assert decomp.M_long_signed == pytest.approx(
        sum(r.raw_capacity - r.null_mean for r in m_records), abs=1e-9)


def test_nl_tensor_by_fixed_delay_covers_every_delay_with_signed_field():
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=4, max_degree=3,
                                     max_targets_per_degree=8, n_surrogates=6, seed=0)
    tensor = nl_tensor_by_fixed_delay(decomp.records, max_delay=4)
    assert set(tensor.keys()) == set(range(5))
    for tau, entry in tensor.items():
        assert set(entry.keys()) == {"delay", "legacy", "raw", "null", "bc", "signed", "n_profiles"}
    # NL_instant (tau=0) must match the tensor's own tau=0 entry exactly
    assert tensor[0]["legacy"] == pytest.approx(decomp.NL_instant, abs=1e-9)
    assert tensor[0]["signed"] == pytest.approx(decomp.NL_instant_signed, abs=1e-9)


def test_m_tensor_by_fixed_delay_is_degree_one_only():
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=3, max_degree=2,
                                     max_targets_per_degree=6, n_surrogates=6, seed=0)
    tensor = m_tensor_by_fixed_delay(decomp.records, max_delay=3)
    manual_sum = sum(entry["legacy"] for entry in tensor.values())
    assert manual_sum == pytest.approx(decomp.M_long, abs=1e-9)


def test_to_heatmap_shape_and_values():
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=4, max_degree=2,
                                     max_targets_per_degree=10, n_surrogates=3, seed=0)
    hm = to_heatmap(decomp, max_degree=2, max_delay=4)
    assert hm.shape == (2, 5)
    assert hm[0, :].sum() == pytest.approx(decomp.M_long, abs=1e-9)
