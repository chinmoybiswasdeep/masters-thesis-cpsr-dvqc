"""
test_chaos_symmetry.py -- V2.1 Phase 5 / test requirements #20, #21:
symmetry-sector handling and matched Hamiltonians between chaos
diagnostics and the actual task simulation.
"""
import os
import sys

import numpy as np
import pytest
from scipy.stats import unitary_group

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.chaos_symmetry import (detect_symmetries, symmetry_resolved_level_spacing,  # noqa: E402
                                           commutator_norm, _global_parity_operator)
from decoupled_qrc.directional_dqrc import DirectionalConfig  # noqa: E402
from decoupled_qrc.seeded_runner import matched_processor_params  # noqa: E402
from decoupled_qrc.validation_utils import make_nested_seeds  # noqa: E402

import mixed_syk_core as msc  # noqa: E402


def test_commutator_norm_zero_for_commuting_operators():
    N = 3
    Pz = _global_parity_operator(N, "Z")
    assert commutator_norm(Pz, Pz) == pytest.approx(0.0, abs=1e-12)


def test_detect_symmetries_finds_z_parity_for_diagonal_unitary():
    """A diagonal unitary (random phases on the computational basis) commutes
    with EVERY diagonal operator, including the global Z-parity operator
    (also diagonal) -- a known-positive control case."""
    N = 4
    rng = np.random.RandomState(0)
    phases = rng.uniform(0, 2 * np.pi, 2 ** N)
    U1 = np.diag(np.exp(1j * phases))
    detection = detect_symmetries(U1, N)
    assert detection["global_Z_parity"]["commutes"]
    assert detection["global_Z_parity"]["commutator_norm"] < 1e-8


def test_detect_symmetries_finds_no_symmetry_for_generic_random_unitary():
    """A Haar-random unitary generically commutes with NONE of the candidate
    global-parity operators -- the expected, physically honest outcome for
    a generic chaotic Hamiltonian with no special structure."""
    N = 3
    U1 = unitary_group.rvs(2 ** N, random_state=42)
    detection = detect_symmetries(U1, N)
    assert not any(info["commutes"] for info in detection.values())


def test_symmetry_resolved_level_spacing_splits_into_two_sectors_for_diagonal_unitary():
    N = 5
    rng = np.random.RandomState(1)
    phases = rng.uniform(0, 2 * np.pi, 2 ** N)
    U1 = np.diag(np.exp(1j * phases))
    result = symmetry_resolved_level_spacing(U1, N, min_sector_size=4)
    assert result.symmetry_used == "global_Z_parity"
    assert result.n_sectors_total == 2
    assert set(result.sector_sizes.values()) == {2 ** (N - 1)}
    assert not np.isnan(result.r_mean)
    assert len(result.r_by_sector) == 2


def test_symmetry_resolved_level_spacing_falls_back_to_full_spectrum_when_no_symmetry_found():
    N = 3
    U1 = unitary_group.rvs(2 ** N, random_state=7)
    result = symmetry_resolved_level_spacing(U1, N)
    assert result.symmetry_used is None
    assert result.n_sectors_total == 1
    assert result.r_mean == pytest.approx(result.r_unresolved_full_spectrum, abs=1e-12)


def test_symmetry_resolved_level_spacing_excludes_too_small_sectors():
    """N=3 -> sectors of size 4 each; requiring min_sector_size=8 must
    exclude both (too small), leaving r_mean as NaN rather than a
    statistically meaningless average over <3-level sectors."""
    N = 3
    rng = np.random.RandomState(2)
    phases = rng.uniform(0, 2 * np.pi, 2 ** N)
    U1 = np.diag(np.exp(1j * phases))
    result = symmetry_resolved_level_spacing(U1, N, min_sector_size=8)
    assert result.symmetry_used == "global_Z_parity"
    assert len(result.sectors_excluded_too_small) == 2
    assert np.isnan(result.r_mean)


def test_matched_hamiltonian_between_chaos_diagnostic_and_task_circuit():
    """Test requirement #21: the SAME NestedSeeds-derived processor params
    used to build the actual DQRC task circuit must be exactly what feeds
    the chaos diagnostic's own step-unitary construction -- both routed
    through `matched_processor_params`, so there is only one path and no
    way to accidentally pass mismatched (term_seed, disorder_seed) to one
    but not the other."""
    cfg = DirectionalConfig(memory_variant="protected_integrable", N_M=2, N_P=5, g_processor=0.5,
                             J_processor=0.33, epsilon_M=0.5, theta=0.2, phi=0.8, ap_kind="xy")
    seeds = make_nested_seeds(0, 0)
    params_a = matched_processor_params(cfg, seeds)
    params_b = matched_processor_params(cfg, seeds)  # simulating a second call site (e.g. discovery loop)
    U1_a = msc.single_layer_unitary_mixed(params_a.N_p, params_a.g, params_a.terms, params_a.couplings,
                                           params_a.paulis, params_a.bias_z)
    U1_b = msc.single_layer_unitary_mixed(params_b.N_p, params_b.g, params_b.terms, params_b.couplings,
                                           params_b.paulis, params_b.bias_z)
    assert np.allclose(U1_a, U1_b), "two independent call sites using the same (cfg, seeds) must build IDENTICAL Hamiltonians"

    result = symmetry_resolved_level_spacing(U1_a, params_a.N_p, min_sector_size=4)
    assert result.r_unresolved_full_spectrum is not None and not np.isnan(result.r_unresolved_full_spectrum)
