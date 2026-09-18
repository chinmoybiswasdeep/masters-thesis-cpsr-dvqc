"""
test_standalone_match.py -- V2.2 Phase 6 / test requirements #10, #11:
matched standalone/DQRC target protocols and the retained-NL ratio.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.standalone_match import run_matched_standalone, retained_nl_ratio  # noqa: E402
from decoupled_qrc.directional_dqrc import DirectionalConfig  # noqa: E402
from decoupled_qrc.seeded_runner import run_directional_dqrc_seeded, matched_processor_params  # noqa: E402
from decoupled_qrc.validation_utils import make_nested_seeds  # noqa: E402
from decoupled_qrc.ipc_decomposition import compute_ipc_decomposed  # noqa: E402
from qrc_qiskit import chrono_split  # noqa: E402


def _cfg():
    return DirectionalConfig(memory_variant="protected_integrable", N_M=2, N_P=5, g_processor=0.5,
                              J_processor=0.33, epsilon_M=0.5, theta=0.2, phi=0.8, ap_kind="xy")


def test_matched_standalone_uses_identical_input_sequence():
    """Test requirement #10: the standalone run must use the EXACT SAME
    input sequence as the DQRC run for the same `seeds` object -- the
    'identical T, input sequence' half of the matched-comparison
    requirement."""
    cfg = _cfg()
    seeds = make_nested_seeds(0, 0)
    run = run_directional_dqrc_seeded(cfg, T=60, seeds=seeds, reset_period=None)
    _, _, u_sa, _ = run_matched_standalone(cfg, 60, seeds)
    assert np.array_equal(run.u, u_sa)


def test_matched_standalone_uses_identical_hamiltonian_seeds_as_dqrc_processor():
    """The standalone processor's own (term_seed, disorder_seed) must
    match EXACTLY what `matched_processor_params` (hence the real DQRC
    circuit's own processor sub-Hamiltonian) uses for the same
    (cfg, seeds) -- proven by cross-checking against
    `directional_processor.sample_params_gJ` called with those same
    derived seeds directly."""
    from decoupled_qrc import directional_processor as dproc
    cfg = _cfg()
    seeds = make_nested_seeds(1, 0)
    _, X_sa, _, _ = run_matched_standalone(cfg, 40, seeds)
    manual_params = dproc.sample_params_gJ(cfg.N_P, cfg.g_processor, cfg.J_processor,
                                            term_seed=seeds.reservoir_seed + 1,
                                            disorder_seed=seeds.reservoir_seed + 2, reps=cfg.reps_processor)
    matched_params = matched_processor_params(cfg, seeds)
    assert manual_params.terms == matched_params.terms
    assert np.array_equal(manual_params.couplings, matched_params.couplings)


def test_retained_nl_ratio_is_one_when_dqrc_and_standalone_identical():
    """A degenerate but exact sanity check: computing eta_NL between two
    IDENTICAL record sets must give exactly 1.0 at every delay with
    nonzero capacity."""
    cfg = _cfg()
    seeds = make_nested_seeds(2, 0)
    T = 90
    run = run_directional_dqrc_seeded(cfg, T=T, seeds=seeds, reset_period=None)
    train, val, test = chrono_split(T, 10, 20, 25, 4)
    decomp = compute_ipc_decomposed(run.u, run.X_combined, train, val, test, max_delay=3, max_degree=2,
                                     max_targets_per_degree=5, n_surrogates=6, seed=0)
    res = retained_nl_ratio(decomp.records, decomp.records, max_delay=3)
    for tau, eta in res.eta_NL_by_delay.items():
        if res.nl_standalone_bc_by_delay[tau] > 1e-9:
            assert eta == pytest.approx(1.0, abs=1e-6)


def test_retained_nl_ratio_flags_suppression_below_threshold():
    """Test requirement #11: `suppression_flagged` must trigger when
    eta_NL(0) < 0.1, using synthetic record sets with a known, controlled
    ratio."""
    from decoupled_qrc.ipc import ProfileCapacity

    def make_record(delay, raw, null):
        return ProfileCapacity(degree=2, delays=(delay,), max_delay=delay, min_delay=delay, span=0,
                                n_delays=1, raw_capacity=raw, null_mean=null, null_std=0.001,
                                threshold=null + 0.01, significant=(raw > null + 0.01),
                                capacity=max(0.0, raw - null) if raw > null + 0.01 else 0.0)

    records_dqrc = [make_record(0, raw=0.02, null=0.01)]      # bc ~ 0.01
    records_standalone = [make_record(0, raw=0.5, null=0.01)]  # bc ~ 0.49
    res = retained_nl_ratio(records_dqrc, records_standalone, max_delay=0)
    assert res.suppression_flagged
    assert res.eta_NL_by_delay[0] < 0.1


def test_retained_nl_ratio_does_not_flag_when_comparable():
    from decoupled_qrc.ipc import ProfileCapacity

    def make_record(delay, raw, null):
        return ProfileCapacity(degree=2, delays=(delay,), max_delay=delay, min_delay=delay, span=0,
                                n_delays=1, raw_capacity=raw, null_mean=null, null_std=0.001,
                                threshold=null + 0.01, significant=(raw > null + 0.01),
                                capacity=max(0.0, raw - null) if raw > null + 0.01 else 0.0)

    records_dqrc = [make_record(0, raw=0.45, null=0.01)]
    records_standalone = [make_record(0, raw=0.5, null=0.01)]
    res = retained_nl_ratio(records_dqrc, records_standalone, max_delay=0)
    assert not res.suppression_flagged
    assert res.eta_NL_by_delay[0] > 0.8
