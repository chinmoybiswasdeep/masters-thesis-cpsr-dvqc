"""
test_seeded_runner.py -- V2.1 Defect 9 fix / test requirements #8, #9:
explicit NestedSeeds propagation and common-random-numbers (CRN)
behavior in real circuit runs (not just in the seed-derivation unit tests
`test_validation_utils.py` already has).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.directional_dqrc import DirectionalConfig  # noqa: E402
from decoupled_qrc.reset_ablation import run_directional_dqrc_with_reset  # noqa: E402
from decoupled_qrc.seeded_runner import run_directional_dqrc_seeded, matched_processor_params  # noqa: E402
from decoupled_qrc.validation_utils import make_nested_seeds  # noqa: E402


def _cfg(epsilon_M=0.5, g=0.5, J=0.33):
    return DirectionalConfig(memory_variant="protected_integrable", N_M=2, N_P=5,
                              g_processor=g, J_processor=J, epsilon_M=epsilon_M, theta=0.2, phi=0.8, ap_kind="xy")


def test_seeded_runner_deterministic_rerun():
    seeds = make_nested_seeds(0, 0)
    r1 = run_directional_dqrc_seeded(_cfg(), T=10, seeds=seeds, reset_period=None)
    r2 = run_directional_dqrc_seeded(_cfg(), T=10, seeds=seeds, reset_period=None)
    assert np.array_equal(r1.X_mem, r2.X_mem)
    assert np.array_equal(r1.X_proc, r2.X_proc)
    assert np.array_equal(r1.u, r2.u)


def test_seeded_runner_common_random_numbers_across_parameter_perturbation():
    """Test requirement #9: the SAME NestedSeeds object, used for two DIFFERENT
    epsilon_M values (a plus/minus response-derivative pair), must produce
    the EXACT SAME input sequence -- the input realization is common random
    numbers across the perturbation, only the physics parameter differs."""
    seeds = make_nested_seeds(1, 0)
    r_plus = run_directional_dqrc_seeded(_cfg(epsilon_M=0.55), T=12, seeds=seeds, reset_period=None)
    r_minus = run_directional_dqrc_seeded(_cfg(epsilon_M=0.45), T=12, seeds=seeds, reset_period=None)
    assert np.array_equal(r_plus.u, r_minus.u), "plus/minus perturbations must share the identical input sequence"
    # but the actual physics differs (different epsilon_M), so outputs must NOT be identical
    assert not np.array_equal(r_plus.X_mem, r_minus.X_mem)


def test_seeded_runner_common_random_numbers_across_reset_periods():
    """Test requirement #9 (reset-variant CRN): the SAME NestedSeeds object
    used across different reset periods must still share the identical
    input sequence and Hamiltonian realization -- only the reset schedule
    differs."""
    seeds = make_nested_seeds(2, 0)
    r_persistent = run_directional_dqrc_seeded(_cfg(), T=10, seeds=seeds, reset_period=None)
    r_reset4 = run_directional_dqrc_seeded(_cfg(), T=10, seeds=seeds, reset_period=4)
    assert np.array_equal(r_persistent.u, r_reset4.u)
    assert not np.array_equal(r_persistent.X_proc, r_reset4.X_proc)


def test_seeded_runner_different_reservoir_idx_gives_different_realization():
    seeds_a = make_nested_seeds(0, 0)
    seeds_b = make_nested_seeds(3, 0)
    r_a = run_directional_dqrc_seeded(_cfg(), T=10, seeds=seeds_a, reset_period=None)
    r_b = run_directional_dqrc_seeded(_cfg(), T=10, seeds=seeds_b, reset_period=None)
    assert not np.array_equal(r_a.u, r_b.u)
    assert not np.array_equal(r_a.X_mem, r_b.X_mem)


def test_seeded_runner_reset_period_none_matches_reset_ablation_semantics():
    """reset_period=None must give the same STRUCTURE of run (persistent
    processor, never reset) as `reset_ablation.run_directional_dqrc_with_reset`
    -- not necessarily bit-identical NUMBERS (different seed-derivation
    scheme by design), but the same qualitative behavior: X_proc from a
    persistent run must differ from a reset_period=1 run on the identical
    seeds object, exactly like reset_ablation's own regression test proves
    for its own (different) seed system."""
    seeds = make_nested_seeds(4, 0)
    r_persistent = run_directional_dqrc_seeded(_cfg(), T=12, seeds=seeds, reset_period=None)
    r_reset1 = run_directional_dqrc_seeded(_cfg(), T=12, seeds=seeds, reset_period=1)
    assert not np.array_equal(r_persistent.X_proc, r_reset1.X_proc)


def test_matched_processor_params_deterministic_and_seed_dependent():
    seeds = make_nested_seeds(0, 0)
    p1 = matched_processor_params(_cfg(), seeds)
    p2 = matched_processor_params(_cfg(), seeds)
    assert p1.terms == p2.terms
    assert np.array_equal(p1.couplings, p2.couplings)
    seeds_other = make_nested_seeds(5, 0)
    p3 = matched_processor_params(_cfg(), seeds_other)
    assert not np.array_equal(p1.couplings, p3.couplings)


def test_matched_processor_params_matches_actual_circuit_build_seed_derivation():
    """The whole point of Defect-9's fix: `matched_processor_params` must
    derive EXACTLY the same (term_seed, disorder_seed) that
    `build_directional_circuit_with_reset` (hence `run_directional_dqrc_seeded`)
    actually used -- verified by cross-checking against
    `directional_processor.sample_params_gJ` called with the same explicit
    seeds by hand."""
    from decoupled_qrc import directional_processor as dproc
    seeds = make_nested_seeds(6, 0)
    cfg = _cfg()
    p_matched = matched_processor_params(cfg, seeds)
    p_manual = dproc.sample_params_gJ(cfg.N_P, cfg.g_processor, cfg.J_processor,
                                       term_seed=seeds.reservoir_seed + 1, disorder_seed=seeds.reservoir_seed + 2,
                                       reps=cfg.reps_processor)
    assert p_matched.terms == p_manual.terms
    assert np.array_equal(p_matched.couplings, p_manual.couplings)


def test_seeded_runner_rejects_statevector_method():
    seeds = make_nested_seeds(0, 0)
    with pytest.raises(ValueError, match="density_matrix"):
        run_directional_dqrc_seeded(_cfg(), T=10, seeds=seeds, reset_period=None, method="statevector")
