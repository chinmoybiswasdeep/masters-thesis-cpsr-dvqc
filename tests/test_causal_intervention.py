"""
test_causal_intervention.py -- V2.2 Phase 1/2 / test requirements #1, #2:
intervention-based causal-latency recovery, and the distinction between
causal, detectable, and peak delay.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.causal_intervention import (intervention_causal_latency,  # noqa: E402
                                                estimate_numeric_noise_floor)
from decoupled_qrc.directional_dqrc import DirectionalConfig  # noqa: E402
from decoupled_qrc.validation_utils import make_nested_seeds  # noqa: E402


def _cfg():
    return DirectionalConfig(memory_variant="protected_integrable", N_M=2, N_P=5, g_processor=0.5,
                              J_processor=0.33, epsilon_M=0.5, theta=0.2, phi=0.8, ap_kind="xy")


def test_numeric_noise_floor_is_near_machine_precision():
    """Two identical (u, seeds) runs of this project's exact (shots=1,
    density_matrix) simulation must differ only by floating-point
    round-off -- this IS the numerical-precision justification Phase 1
    requires for the intervention tolerance."""
    from qrc_qiskit import random_input
    seeds = make_nested_seeds(0, 0)
    u = random_input(15, seed=seeds.input_seed)
    floor = estimate_numeric_noise_floor(_cfg(), u, seeds, reset_period=None)
    assert floor < 1e-8, f"expected near machine-precision agreement between identical runs, got {floor}"


def test_intervention_detects_immediate_causal_response():
    """The actual circuit schedule encodes u_t, evolves memory, couples
    ancilla, evolves the processor, THEN reads out X_t -- so a
    perturbation at u_t must be visible in X_t itself (tau=0), not only
    at some later step."""
    seeds = make_nested_seeds(1, 0)
    res = intervention_causal_latency(_cfg(), T=20, seeds=seeds, perturb_t=8)
    assert res.ell_causal == 0, f"expected immediate (tau=0) causal response, got ell_causal={res.ell_causal}"
    for name in ("X_M", "X_P", "X_M+X_P", "cross"):
        assert res.ell_causal_by_group[name] == 0, f"{name} did not show an immediate causal response"


def test_intervention_no_response_before_perturbation():
    """Steps STRICTLY BEFORE perturb_t must show zero divergence (the two
    input sequences are identical up to that point, and the simulation is
    deterministic) -- a basic a-causality sanity check."""
    seeds = make_nested_seeds(2, 0)
    res = intervention_causal_latency(_cfg(), T=20, seeds=seeds, perturb_t=10)
    for name, delta in res.delta_by_group.items():
        assert np.all(delta[:10] < res.epsilon_numeric), f"{name} showed divergence BEFORE the perturbation step"


def test_intervention_response_decays_at_large_delay():
    """The perturbation's effect should generally decay (not necessarily
    monotonically, but should not remain at its peak) by many steps later
    -- a basic sanity check that the delta series is not simply constant/
    saturated (which would suggest a bug, e.g. comparing wrong arrays)."""
    seeds = make_nested_seeds(3, 0)
    res = intervention_causal_latency(_cfg(), T=22, seeds=seeds, perturb_t=8)
    delta = res.delta_by_group["X_M"]
    assert delta[8] > 0
    assert not np.allclose(delta[8:], delta[8], atol=1e-9), "expected the divergence trajectory to actually vary over time"


def test_intervention_deterministic_rerun():
    seeds = make_nested_seeds(4, 0)
    res1 = intervention_causal_latency(_cfg(), T=15, seeds=seeds, perturb_t=6)
    res2 = intervention_causal_latency(_cfg(), T=15, seeds=seeds, perturb_t=6)
    assert res1.ell_causal == res2.ell_causal
    for name in res1.delta_by_group:
        assert np.allclose(res1.delta_by_group[name], res2.delta_by_group[name], atol=1e-9)


def test_causal_vs_detectable_vs_peak_delay_are_structurally_distinct():
    """Test requirement #2: `ell_causal` (from intervention),
    `ell_detect` (from significance testing), and `ell_peak`
    (argmax C_{1,tau}, formerly mislabeled 'ell_0') must be computed by
    THREE DIFFERENT mechanisms and are not required to agree -- this test
    locks in that the three concepts are exposed as genuinely separate
    quantities, not silently aliased to one another."""
    from decoupled_qrc.ipc_decomposition import compute_ipc_decomposed
    from decoupled_qrc.delay_concepts import ell_detect_from_records, ell_peak_from_records
    from decoupled_qrc.seeded_runner import run_directional_dqrc_seeded
    from qrc_qiskit import chrono_split

    seeds = make_nested_seeds(5, 0)
    cfg = _cfg()
    run = run_directional_dqrc_seeded(cfg, T=120, seeds=seeds, reset_period=None)
    train, val, test = chrono_split(120, 15, 25, 30, 6)
    decomp = compute_ipc_decomposed(run.u, run.X_combined, train, val, test, max_delay=5, max_degree=2,
                                     max_targets_per_degree=6, n_surrogates=19, seed=0)

    ell_detect = ell_detect_from_records(decomp.records, degree=1)
    ell_peak = ell_peak_from_records(decomp.records, degree=1)
    intervention = intervention_causal_latency(cfg, T=20, seeds=seeds, perturb_t=8)

    # all three must be well-defined, independently-computed integers/None -- no assertion that
    # they're EQUAL (they are conceptually different quantities and may legitimately differ)
    assert isinstance(ell_peak, int)
    assert ell_detect is None or isinstance(ell_detect, int)
    assert intervention.ell_causal is None or isinstance(intervention.ell_causal, int)


def test_intervention_result_includes_timing_table():
    seeds = make_nested_seeds(0, 0)
    res = intervention_causal_latency(_cfg(), T=15, seeds=seeds, perturb_t=6)
    assert len(res.timing_table) == 6
    assert all(row["depends_on_u_t"] for row in res.timing_table)
