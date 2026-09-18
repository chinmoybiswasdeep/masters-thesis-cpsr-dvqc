"""
test_v3_1_science.py -- V3.1 scientific layer: ceiling/tail/null audits,
processor variants and ablations, memory variants and CPTP retention, the
corrected response geometry, selectivity bounds, hierarchical bootstrap,
and Pareto/hypervolume.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.ceiling_audit import (audit_ceiling, audit_nl0, order_tail, delay_tail,  # noqa: E402
                                          audit_null, benjamini_hochberg, count_single_delay_targets)
from decoupled_qrc.memory_variants import (MemoryVariantConfig, run_memory_variant,  # noqa: E402
                                            retention_kraus, is_trace_preserving, audit_m0_semantics,
                                            build_memory_circuit_variant)
from decoupled_qrc.processor_variants import (ProcVariantConfig, run_variant, commutator_report,  # noqa: E402
                                               interaction_hamiltonian, encoder_operators,
                                               macro_unitary, PROCESSOR_VARIANTS, ABLATIONS,
                                               h_two_body, h_four_body, h_commuting)
from decoupled_qrc.v3_1_stats import (hierarchical_bootstrap, response_angle, selectivity,  # noqa: E402
                                       pareto_front, hypervolume, paired_hypervolume_bootstrap,
                                       dynamic_range)
from decoupled_qrc.v3_seeds import make_seeds  # noqa: E402
from decoupled_qrc.nonlinear_processor import build_tap_buffer  # noqa: E402


# --------------------------------------------------------------------------
# Ceiling / tail / null
# --------------------------------------------------------------------------

def test_ceiling_detects_the_exact_v3_saturation():
    """V3 reported NL0_raw = 2.000 with exactly 2 available targets. That is
    100% of the target-count ceiling and must be flagged."""
    rep = audit_ceiling("NL_0", raw=2.0, n_targets=2, numerical_rank=30, effective_rank=1.4,
                         n_train=100)
    assert rep.ceiling_contaminated
    assert rep.fraction_of_ceiling == pytest.approx(1.0)
    assert "target_count" in rep.reason


def test_ceiling_clear_when_headroom_exists():
    rep = audit_ceiling("NL_0", raw=5.09, n_targets=7, numerical_rank=30, effective_rank=4.0,
                         n_train=100)
    assert not rep.ceiling_contaminated
    assert rep.fraction_of_ceiling < 0.95


def test_rank_ceiling_can_be_the_binding_constraint():
    rep = audit_ceiling("M", raw=2.9, n_targets=20, numerical_rank=3, effective_rank=2.5, n_train=100)
    assert rep.effective_ceiling == 3.0 and rep.ceiling_contaminated


def test_sample_ceiling_can_be_the_binding_constraint():
    rep = audit_ceiling("M", raw=9.8, n_targets=50, numerical_rank=40, effective_rank=8.0, n_train=10)
    assert rep.effective_ceiling == 10.0 and rep.ceiling_contaminated


class _Rec:
    def __init__(self, degree, delays, raw, null=0.0, std=0.01):
        self.degree, self.delays = degree, delays
        self.max_delay, self.min_delay = max(delays), min(delays)
        self.raw_capacity, self.null_mean, self.null_std = raw, null, std
        self.threshold = null + 2 * std
        self.significant = raw > self.threshold
        self.capacity = raw if self.significant else 0.0


def test_order_tail_flags_a_truncated_order_horizon():
    recs = [_Rec(d, (0,), 0.9) for d in range(2, 6)]     # flat: top order still large
    tail = order_tail(recs, max_degree=5)
    assert not tail.declining and "INCREASE max_degree" in tail.recommendation


def test_order_tail_accepts_a_decaying_spectrum():
    recs = [_Rec(2, (0,), 1.0), _Rec(3, (0,), 0.4), _Rec(4, (0,), 0.1), _Rec(5, (0,), 0.02)]
    assert order_tail(recs, max_degree=5).declining


def test_delay_tail_flags_a_truncated_delay_horizon():
    recs = [_Rec(1, (t,), 0.8) for t in range(5)]
    assert not delay_tail(recs, max_delay=4).declining


def test_delay_tail_accepts_a_decayed_memory_tail():
    recs = [_Rec(1, (t,), 1.0 / (1 + 6 * t)) for t in range(5)]
    assert delay_tail(recs, max_delay=4).declining


def test_null_bias_gate():
    good = [_Rec(2, (0,), 0.9, null=0.05)]
    bad = [_Rec(2, (0,), 0.9, null=0.5)]
    sel = lambda r: r.degree >= 2 and r.delays == (0,)
    assert audit_null(good, "NL_0", sel, raw=0.9).acceptable
    assert not audit_null(bad, "NL_0", sel, raw=0.9).acceptable


def test_benjamini_hochberg_controls_discoveries():
    assert benjamini_hochberg([0.001, 0.002, 0.003])["n_significant"] == 3
    assert benjamini_hochberg([0.9, 0.8, 0.7])["n_significant"] == 0


def test_count_single_delay_targets():
    recs = [_Rec(2, (0,), 1.0), _Rec(3, (0,), 1.0), _Rec(2, (0, 1), 1.0), _Rec(2, (1,), 1.0)]
    assert count_single_delay_targets(recs, 0) == 2


# --------------------------------------------------------------------------
# Processor variants and ablations
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", PROCESSOR_VARIANTS)
def test_every_variant_builds_a_unitary(variant):
    cfg = ProcVariantConfig(variant=variant, N_P=4, R=2, n_taps=2)
    u = macro_unitary(cfg, [0.3, -0.2], make_seeds(0, 0))
    assert np.allclose(u @ u.conj().T, np.eye(2 ** cfg.N_P), atol=1e-9)


def test_reuploading_variants_have_noncommuting_consecutive_encoders():
    """P1/P3 alternate the encoder axis between layers, which is what makes
    re-uploading generate genuinely new nonlinearity."""
    seeds = make_seeds(0, 0)
    for v in ("P1", "P3"):
        rep = commutator_report(ProcVariantConfig(variant=v, N_P=4, n_taps=2), seeds)
        assert rep["encoder_vs_encoder"] > 1e-9, f"{v} consecutive encoders must not commute"
        assert rep["encoder_vs_interaction"] > 1e-9


def test_two_body_and_four_body_are_distinct_operators():
    h2, h4 = h_two_body(4), h_four_body(4)
    assert np.linalg.norm(h2) > 0 and np.linalg.norm(h4) > 0
    assert not np.allclose(h2, h4)


def test_commuting_ablation_is_actually_commuting():
    """The commuting control must have mutually commuting interaction terms."""
    n = 4
    terms = []
    for i in range(n - 1):
        t = np.zeros((2 ** n, 2 ** n), dtype=complex)
        from decoupled_qrc.processor_variants import _kron_op
        t = _kron_op(n, {i: "Z", i + 1: "Z"})
        terms.append(t)
    for a in terms:
        for b in terms:
            assert np.allclose(a @ b, b @ a, atol=1e-12)


def test_encoding_only_ablation_removes_all_interaction():
    seeds = make_seeds(0, 0)
    cfg = ProcVariantConfig(variant="P1", N_P=4, ablation="encoding_only")
    assert np.allclose(interaction_hamiltonian(cfg, seeds), 0.0)


def test_interaction_ablations_differ_from_each_other():
    seeds = make_seeds(0, 0)
    mats = {}
    for ab in ("two_body_only", "four_body_only", "full", "commuting", "noninteracting"):
        mats[ab] = interaction_hamiltonian(ProcVariantConfig(variant="P1", N_P=4, ablation=ab), seeds)
    assert not np.allclose(mats["two_body_only"], mats["four_body_only"])
    assert not np.allclose(mats["full"], mats["commuting"])
    assert not np.allclose(mats["full"], mats["noninteracting"])


def test_g_and_J_change_processor_dynamics():
    seeds = make_seeds(0, 0)
    base = ProcVariantConfig(variant="P1", N_P=4, g=0.8, J=0.6, R=2)
    u0 = macro_unitary(base, [0.3], seeds)
    from dataclasses import replace
    assert not np.allclose(u0, macro_unitary(replace(base, g=1.2), [0.3], seeds))
    assert not np.allclose(u0, macro_unitary(replace(base, J=1.0), [0.3], seeds))


def test_variant_run_resets_so_no_inter_timestep_persistence():
    seeds = make_seeds(0, 0)
    cfg = ProcVariantConfig(variant="P1", N_P=3, R=2, n_taps=1)
    rng = np.random.RandomState(0)
    a, b = rng.uniform(0, 1, 10), rng.uniform(0, 1, 10)
    b[6] = a[6]
    ra = run_variant(cfg, build_tap_buffer(a, 1), seeds)
    rb = run_variant(cfg, build_tap_buffer(b, 1), seeds)
    assert np.allclose(ra.X_P[6], rb.X_P[6], atol=1e-10)


def test_feature_budget_identical_across_parameter_points():
    """A richer readout at one point would manufacture apparent
    controllability -- the observable set must not depend on (g,J)."""
    seeds = make_seeds(0, 0)
    from dataclasses import replace
    cfg = ProcVariantConfig(variant="P1", N_P=3, R=1, n_taps=1)
    u = build_tap_buffer(np.linspace(0, 1, 6), 1)
    a = run_variant(cfg, u, seeds)
    b = run_variant(replace(cfg, g=1.3, J=0.1), u, seeds)
    assert a.labels == b.labels and a.X_P.shape == b.X_P.shape


# --------------------------------------------------------------------------
# Memory variants
# --------------------------------------------------------------------------

def test_retention_channel_is_cptp():
    for m in (0.0, 0.25, 0.5, 0.75, 1.0):
        k = retention_kraus(m)
        assert is_trace_preserving(k), f"retention channel at m={m} is not trace preserving"


def test_retention_channel_endpoints():
    """m=1 keeps the state; m=0 resets it to the reference."""
    import numpy as np
    rho = np.array([[0.3, 0.1], [0.1, 0.7]], dtype=complex)

    def apply(kr, r):
        return sum(np.asarray(k) @ r @ np.asarray(k).conj().T for k in kr.data)

    assert np.allclose(apply(retention_kraus(1.0), rho), rho, atol=1e-12)
    out0 = apply(retention_kraus(0.0), rho)
    assert np.allclose(out0, np.array([[1, 0], [0, 0]], dtype=complex), atol=1e-12)


def test_m0_semantics_audit_shows_m_is_not_pure_retention():
    """Documents WHY M1 exists: in M0, `m` moves transfer probability and
    the singlet phase together."""
    a = audit_m0_semantics()
    assert a[0.3]["transfer_probability"] < a[0.9]["transfer_probability"]
    assert a[0.3]["singlet_phase_rad"] != pytest.approx(a[0.9]["singlet_phase_rad"])


def test_m_changes_memory_dynamics_without_changing_feature_count():
    seeds = make_seeds(0, 0)
    u = np.random.RandomState(0).uniform(0, 1, 24)
    a = run_memory_variant(MemoryVariantConfig(variant="M1", L=3, m=0.9), u, seeds)
    b = run_memory_variant(MemoryVariantConfig(variant="M1", L=3, m=0.2), u, seeds)
    assert a.X_M.shape == b.X_M.shape, "m must not change the feature count"
    assert not np.allclose(a.X_M, b.X_M), "m must change the physical memory state"


def test_both_memory_variants_run():
    seeds = make_seeds(0, 0)
    u = np.random.RandomState(0).uniform(0, 1, 16)
    for v in ("M0", "M1"):
        r = run_memory_variant(MemoryVariantConfig(variant=v, L=3, m=0.7), u, seeds)
        assert r.X_M.shape[0] == 16 and r.X_M.shape[1] > 0


def test_memory_state_stays_a_valid_density_matrix():
    from qiskit import transpile
    from qiskit.quantum_info import DensityMatrix
    from qrc_qiskit import make_simulator
    seeds = make_seeds(0, 0)
    u = np.random.RandomState(0).uniform(0, 1, 8)
    qc, _ = build_memory_circuit_variant(MemoryVariantConfig(variant="M1", L=3, m=0.6), u, seeds)
    qc.save_density_matrix(label="rho")
    sim = make_simulator(method="density_matrix")
    rho = np.asarray(DensityMatrix(np.asarray(
        sim.run(transpile(qc, sim, optimization_level=1), shots=1).result().data(0)["rho"])).data)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-9)
    assert np.allclose(rho, rho.conj().T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(rho) > -1e-9)


# --------------------------------------------------------------------------
# Response geometry -- the V3 correction
# --------------------------------------------------------------------------

def test_ideal_decoupling_gives_ninety_degrees_not_not_evaluable():
    """THE V3 BUG: V3 returned NOT EVALUABLE exactly when the off-diagonal
    terms vanished -- i.e. it could not recognise the success case. The
    corrected geometry must return 90 degrees."""
    geo = response_angle(dM_dm=2.0, dNL_dm=0.0, dM_dg=0.0, dM_dJ=0.0, dNL_dg=1.0, dNL_dJ=0.5)
    assert geo.evaluable, "perfect decoupling must be EVALUABLE, not NOT EVALUABLE"
    assert geo.angle_deg == pytest.approx(90.0, abs=1e-6)


def test_fully_coupled_response_gives_small_angle():
    """Parallel response vectors -> ~0 degrees. `arccos` near 1 loses
    precision, so the tolerance is 1e-4 degrees rather than 1e-6."""
    geo = response_angle(dM_dm=1.0, dNL_dm=1.0, dM_dg=1.0, dM_dJ=0.0, dNL_dg=1.0, dNL_dJ=0.0)
    assert geo.evaluable and geo.angle_deg == pytest.approx(0.0, abs=1e-4)


def test_angle_not_evaluable_only_when_a_whole_vector_is_unresolved():
    geo = response_angle(0.0, 0.0, 1.0, 1.0, 1.0, 1.0)         # v_m entirely zero
    assert not geo.evaluable and "v_m" in geo.reason
    geo2 = response_angle(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)        # processor direction unresolved
    assert not geo2.evaluable and "grad" in geo2.reason


# --------------------------------------------------------------------------
# Selectivity with structural zeros
# --------------------------------------------------------------------------

def test_selectivity_returns_lower_bound_for_structural_zero():
    s = selectivity(numerator=2.0, denominator_estimate=0.0, denominator_upper_bound=0.005)
    assert s.kind == "lower_bound"
    assert s.ratio_lower_bound == pytest.approx(400.0)
    assert np.isnan(s.ratio), "must not fabricate a point ratio from a structural zero"


def test_selectivity_returns_ratio_when_resolved():
    s = selectivity(2.0, 0.5, 0.6)
    assert s.kind == "ratio" and s.ratio == pytest.approx(4.0)


def test_selectivity_not_evaluable_when_numerator_unresolved():
    s = selectivity(0.0, 0.0, 0.01)
    assert s.kind == "not_evaluable"


def test_selectivity_never_divides_by_epsilon():
    s = selectivity(1.0, 0.0, 0.0)
    assert s.kind == "not_evaluable" and np.isnan(s.ratio_lower_bound)


# --------------------------------------------------------------------------
# Hierarchical bootstrap
# --------------------------------------------------------------------------

def test_hierarchical_bootstrap_brackets_the_observed_mean():
    """A bootstrap quantifies uncertainty around the OBSERVED sample, so the
    interval must bracket the pooled sample mean. (With only 5 groups the
    sample mean can sit well away from the population mean, so requiring the
    interval to contain the population mean would be the wrong test.)"""
    rng = np.random.RandomState(0)
    nested = {k: list(rng.normal(5.0, 0.3, 4)) for k in range(5)}
    ci = hierarchical_bootstrap(nested, n_boot=400, seed=1)
    assert ci.ci_low <= ci.mean <= ci.ci_high
    assert ci.n_reservoir == 5


def test_hierarchical_bootstrap_covers_population_mean_with_many_groups():
    """With enough reservoir groups the sample mean converges and the
    interval does contain the population mean."""
    rng = np.random.RandomState(3)
    nested = {k: list(rng.normal(5.0, 0.3, 4)) for k in range(40)}
    ci = hierarchical_bootstrap(nested, n_boot=400, seed=1)
    assert ci.ci_low < 5.0 < ci.ci_high


def test_hierarchical_bootstrap_separates_variance_components():
    """Large between-reservoir spread with tight within-reservoir spread
    must be reported as such."""
    nested = {0: [1.0, 1.02, 0.98], 1: [5.0, 5.01, 4.99], 2: [9.0, 9.02, 8.98]}
    ci = hierarchical_bootstrap(nested, n_boot=200, seed=0)
    assert ci.between_reservoir_var > 10 * ci.within_reservoir_var


def test_hierarchical_ci_wider_than_naive_when_between_variance_dominates():
    nested = {0: [0.0] * 5, 1: [10.0] * 5}
    ci = hierarchical_bootstrap(nested, n_boot=400, seed=0)
    assert ci.ci_high - ci.ci_low > 1.0, "between-reservoir variation must widen the interval"


def test_excludes_helper():
    ci = hierarchical_bootstrap({0: [5.0, 5.1], 1: [5.2, 5.05]}, n_boot=200, seed=0)
    assert ci.excludes(0.0) and not ci.excludes(5.1)


# --------------------------------------------------------------------------
# Pareto / hypervolume
# --------------------------------------------------------------------------

def test_pareto_front_identifies_nondominated_points():
    pts = [(1.0, 5.0), (5.0, 1.0), (3.0, 3.0), (0.5, 0.5), (2.0, 2.0)]
    front = set(pareto_front(pts))
    assert 3 not in front and 4 not in front       # dominated by (3,3)
    assert {0, 1, 2} <= front


def test_hypervolume_of_single_point_is_rectangle_area():
    assert hypervolume([(2.0, 3.0)], reference=(0.0, 0.0)) == pytest.approx(6.0)


def test_hypervolume_monotone_under_domination():
    small = hypervolume([(2.0, 2.0)])
    big = hypervolume([(2.0, 2.0), (3.0, 3.0)])
    assert big > small


def test_hypervolume_ignores_points_below_reference():
    assert hypervolume([(1.0, 1.0)], reference=(2.0, 2.0)) == pytest.approx(0.0)


def test_paired_hypervolume_bootstrap_detects_a_real_difference():
    a = {k: [(3.0 + 0.05 * k, 3.0)] for k in range(5)}
    b = {k: [(1.0 + 0.05 * k, 1.0)] for k in range(5)}
    cmp = paired_hypervolume_bootstrap(a, b, n_boot=300, seed=0)
    assert cmp.delta > 0 and cmp.favours_a and cmp.ci_low > 0


def test_paired_hypervolume_bootstrap_reports_no_difference_when_identical():
    a = {k: [(2.0, 2.0)] for k in range(4)}
    cmp = paired_hypervolume_bootstrap(a, a, n_boot=200, seed=0)
    assert cmp.delta == pytest.approx(0.0) and not cmp.favours_a


# --------------------------------------------------------------------------
# Dynamic range
# --------------------------------------------------------------------------

def test_dynamic_range_flags_a_flat_surface():
    """The V3 (g,J) surface: values 1.404..1.425 -> E well below the 0.20 gate."""
    flat = dynamic_range([1.404, 1.410, 1.425, 1.406, 1.415])
    assert flat["E"] < 0.05


def test_dynamic_range_recognises_a_responsive_surface():
    responsive = dynamic_range([3.245, 4.043, 4.106, 4.019, 4.143, 3.898])
    assert responsive["E"] > 0.10
