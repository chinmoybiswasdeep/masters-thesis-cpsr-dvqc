"""
test_backaction_decomposition.py -- V2.2 Phase 13 / test requirement #20:
back-action control decomposition (full / memory-export-disabled /
processor-reception-disabled, each vs a common both-disabled baseline).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.backaction_decomposition import decompose_backaction  # noqa: E402
from decoupled_qrc.directional_dqrc import DirectionalConfig  # noqa: E402
from decoupled_qrc.validation_utils import make_nested_seeds  # noqa: E402


def _cfg():
    return DirectionalConfig(memory_variant="protected_integrable", N_M=2, N_P=5, g_processor=0.5,
                              J_processor=0.33, epsilon_M=0.5, theta=0.2, phi=0.8, ap_kind="xy")


def test_memory_export_disabled_shows_near_zero_disturbance():
    """With theta=0 (no memory-ancilla coupling), the memory register's
    reduced state cannot differ from the fully-disabled baseline AT ALL
    (the ancilla never receives anything from the memory) -- the
    disturbance must be exactly ~0, a strong physical sanity check."""
    seeds = make_nested_seeds(0, 0)
    res = decompose_backaction(_cfg(), T=8, seeds=seeds)
    assert res.memory_export_disabled.max_trace_distance < 1e-9


def test_full_and_processor_reception_disabled_both_show_nonzero_disturbance():
    seeds = make_nested_seeds(1, 0)
    res = decompose_backaction(_cfg(), T=8, seeds=seeds)
    assert res.full.mean_trace_distance > 1e-6
    assert res.processor_reception_disabled.mean_trace_distance > 1e-6


def test_three_conditions_share_common_random_numbers():
    """All three comparisons use the SAME seeds object -- rerunning must be
    fully deterministic."""
    seeds = make_nested_seeds(2, 0)
    res1 = decompose_backaction(_cfg(), T=8, seeds=seeds)
    res2 = decompose_backaction(_cfg(), T=8, seeds=seeds)
    assert res1.full.trace_distance == res2.full.trace_distance
    assert res1.memory_export_disabled.trace_distance == res2.memory_export_disabled.trace_distance
