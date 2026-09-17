"""
test_memory_backaction.py -- Part 5/6/14's back-action metrics.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.directional_dqrc import DirectionalConfig  # noqa: E402
from decoupled_qrc.directional_diagnostics import (  # noqa: E402
    memory_back_action, q_transfer, full_channel_qnd_check,
)


def test_zero_coupling_gives_zero_back_action():
    """theta=phi=0 compared against itself must show EXACTLY zero
    disturbance -- the trivial but necessary sanity check."""
    cfg = DirectionalConfig(memory_variant="shift", N_M=2, N_P=4, theta=0.0, phi=0.0)
    result = memory_back_action(cfg, T_small=10, master_seed=0)
    assert result["trace_distance"] == pytest.approx(0.0, abs=1e-9)
    assert result["fidelity"] == pytest.approx(1.0, abs=1e-6)


def test_nonzero_coupling_gives_measurable_back_action():
    cfg = DirectionalConfig(memory_variant="shift", N_M=2, N_P=4, theta=0.6, phi=0.6)
    result = memory_back_action(cfg, T_small=10, master_seed=0)
    assert result["trace_distance"] > 0.0
    assert result["fidelity"] < 1.0


def test_q_transfer_ratio():
    assert q_transfer(nl_delta=1.0, D_M=0.5) == pytest.approx(2.0, rel=1e-3)
    assert q_transfer(nl_delta=0.0, D_M=0.0) == pytest.approx(0.0, abs=1e-3)


def test_full_channel_qnd_check_zero_disturbance_for_pure_zz_coupling():
    """The full-channel check (not just the bare commutator) should ALSO
    show zero disturbance to <Z_tap> for this architecture's ZZ-type M-A
    coupling -- a stronger, trajectory-level confirmation of the QND
    property (Part 14's explicit 'the first commutator alone is
    insufficient' requirement)."""
    cfg = DirectionalConfig(memory_variant="shift", N_M=2, N_P=4, theta=0.5, phi=0.6)
    result = full_channel_qnd_check(cfg, T_small=12, master_seed=0)
    assert result["z_tap_disturbance"] < 1e-9, result
