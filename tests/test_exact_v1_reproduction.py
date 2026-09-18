"""
test_exact_v1_reproduction.py -- V2.1 Defect 8 / test requirement #6:
exact V1 protocol reconstruction. This test actually runs a real
(FAST_CONFIG-scale, T=250) circuit -- slower than most unit tests, kept
because reproducing V1's saved numbers exactly is the single most
important regression check in this whole V2.1 pass (if it ever silently
drifts, EVERY downstream V2.1 comparison against V1 becomes meaningless).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.exact_v1_reproduction import (run_exact_v1_reproduction, V1_REFERENCE,  # noqa: E402
                                                  UNRECONSTRUCTABLE_FROM_V1)


def test_exact_v1_reproduction_matches_saved_reference_numbers():
    result = run_exact_v1_reproduction()
    assert result.M_matches_reference, f"M={result.M} does not match V1's saved reference {V1_REFERENCE['M']}"
    assert result.NL_matches_reference, f"NL={result.NL} does not match V1's saved reference {V1_REFERENCE['NL']}"


def test_exact_v1_reproduction_uses_v1s_own_config_not_v2_defaults():
    result = run_exact_v1_reproduction()
    assert result.config_used["T_ipc"] == 250          # FAST_CONFIG.T_ipc, not V2's T=140
    assert result.config_used["max_delay_ipc"] == 6     # FAST_CONFIG, not V2's max_delay_ipc=5
    assert result.config_used["n_surrogates"] == 5      # FAST_CONFIG, not V2's n_surrogates=4
    assert result.config_used["seed_scheme"] == "make_seed_bundle(0)"  # the OLD seed system, not NestedSeeds


def test_unreconstructable_from_v1_is_nonempty_and_explicit():
    """Defect 8's explicit rule: 'if exact V1 configuration information is
    unavailable, explicitly list what cannot be reconstructed' -- this
    list must exist and be non-empty (there genuinely are gaps: V1 never
    recorded package/git provenance)."""
    assert len(UNRECONSTRUCTABLE_FROM_V1) > 0
    assert all(isinstance(s, str) and len(s) > 20 for s in UNRECONSTRUCTABLE_FROM_V1)
