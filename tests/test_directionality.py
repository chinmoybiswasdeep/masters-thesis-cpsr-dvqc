"""
test_directionality.py -- Part 5's central demand: do NOT assume the
channel is directional just because an ancilla exists -- measure it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.directional_dqrc import DirectionalConfig, run_directional_dqrc  # noqa: E402
from decoupled_qrc import diagnostics as diag  # noqa: E402
from decoupled_qrc import utils  # noqa: E402


def test_phi_zero_gives_no_processor_information_gain_even_if_theta_nonzero():
    """If the A-P stage never runs (phi=0), the processor cannot have
    gained any information about u_t no matter how strongly M couples to A
    -- this is the literal, mechanistic proof that information flows
    strictly M -> A -> P (not some other path), because cutting the SECOND
    link alone must fully block transfer."""
    CFG = utils.active_config()
    cfg_on = DirectionalConfig(memory_variant="shift", N_M=2, N_P=4, theta=0.8, phi=0.0)
    run_on = run_directional_dqrc(cfg_on, T=CFG.T_ipc, master_seed=0)
    M, NL, total, _ = diag.ipc_MN(run_on.u, run_on.X_proc, CFG, seed=0)
    assert total < 0.5, f"expected near-zero processor information with phi=0, got total={total}"


def test_theta_zero_gives_no_processor_information_gain_even_if_phi_nonzero():
    """Symmetric check: if the FIRST link never runs (theta=0), the ancilla
    never learns anything about memory, so no amount of A-P coupling can
    inject real temporal information into the processor (the ancilla is
    just relaying its own fixed |0> state)."""
    CFG = utils.active_config()
    cfg_off = DirectionalConfig(memory_variant="shift", N_M=2, N_P=4, theta=0.0, phi=0.8)
    run_off = run_directional_dqrc(cfg_off, T=CFG.T_ipc, master_seed=0)
    M, NL, total, _ = diag.ipc_MN(run_off.u, run_off.X_proc, CFG, seed=0)
    assert total < 0.5, f"expected near-zero processor information with theta=0, got total={total}"


def test_both_nonzero_gives_real_processor_information_gain():
    """The positive control: with BOTH theta and phi nonzero, the
    processor's own features MUST show real, above-noise information about
    the input -- otherwise the whole collision-channel construction would
    be pointless regardless of directionality."""
    CFG = utils.active_config()
    cfg = DirectionalConfig(memory_variant="shift", N_M=2, N_P=5, theta=0.5, phi=0.6)
    run = run_directional_dqrc(cfg, T=CFG.T_ipc, master_seed=0)
    M, NL, total, _ = diag.ipc_MN(run.u, run.X_proc, CFG, seed=0)
    assert total > 0.5, f"expected real processor information with theta,phi>0, got total={total}"
