"""
standalone_match.py -- V2.2 Phase 6: compares the standalone EOC
processor against the full DQRC using IDENTICAL T, input sequence,
Hamiltonian/disorder seeds (the SAME `term_seed=seeds.reservoir_seed+1,
disorder_seed=seeds.reservoir_seed+2` convention
`directional_dqrc.build_directional_circuit` itself uses for the
processor sub-Hamiltonian -- see `seeded_runner.matched_processor_params`),
target set, delay set, and train/val/test split -- so any difference in
NL_tau between the two is attributable to embedding the processor in the
directional architecture, not to a confound in the comparison itself.

Defines the retained nonlinear capacity:
    eta_NL(tau) = NL_tau^DQRC / (NL_tau^standalone + eps)
using the CLIPPED (non-negative) capacity for both numerator and
denominator (Phase 4's `_bc`/`bc` field) -- the SIGNED metric is the
correct one for DERIVATIVES, but a ratio of two possibly-negative signed
quantities is not a meaningful "fraction retained"; both signed and
clipped values are still reported for transparency alongside eta_NL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import processor as procmod
from .directional_dqrc import DirectionalConfig
from .ipc_decomposition import nl_tensor_by_fixed_delay
from .validation_utils import NestedSeeds
from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

from qrc_qiskit import random_input  # noqa: E402


def run_matched_standalone(cfg: DirectionalConfig, T: int, seeds: NestedSeeds):
    """Standalone EOC processor, using the EXACT SAME (term_seed,
    disorder_seed, input_seed) that the corresponding DQRC circuit's own
    processor sub-Hamiltonian and input encoding would use for this
    `seeds` object -- see `seeded_runner.matched_processor_params` (same
    `+1`/`+2` offset convention)."""
    labels, X, u, info = procmod.run_processor_standalone_gJ(
        cfg.N_P, g=cfg.g_processor, J=cfg.J_processor, T=T, reps=cfg.reps_processor,
        max_weight=cfg.max_weight_proc, term_seed=seeds.reservoir_seed + 1,
        disorder_seed=seeds.reservoir_seed + 2, input_seed=seeds.input_seed)
    return labels, X, u, info


@dataclass
class RetainedNLResult:
    eta_NL_by_delay: dict          # {tau: float}
    nl_dqrc_signed_by_delay: dict
    nl_standalone_signed_by_delay: dict
    nl_dqrc_bc_by_delay: dict
    nl_standalone_bc_by_delay: dict
    suppression_flagged: bool      # eta_NL(0) < 0.1 (Phase 6's preregistered diagnostic threshold)


def retained_nl_ratio(records_dqrc: list, records_standalone: list, max_delay: int,
                       degree_min: int = 2, eps: float = 1e-9,
                       suppression_threshold: float = 0.1) -> RetainedNLResult:
    """eta_NL(tau) for every fixed delay 0..max_delay, using CLIPPED
    (non-negative) capacity in both numerator and denominator. Flags
    `suppression_flagged` if eta_NL(0) falls below `suppression_threshold`
    -- reported as a DIAGNOSTIC (Phase 6's own explicit instruction: never
    used to claim quantum advantage or a fundamental impossibility)."""
    tensor_dqrc = nl_tensor_by_fixed_delay(records_dqrc, max_delay, degree_min=degree_min)
    tensor_standalone = nl_tensor_by_fixed_delay(records_standalone, max_delay, degree_min=degree_min)

    eta = {}
    signed_d, signed_s, bc_d, bc_s = {}, {}, {}, {}
    for tau in range(max_delay + 1):
        d_bc = tensor_dqrc[tau]["bc"]
        s_bc = tensor_standalone[tau]["bc"]
        eta[tau] = float(d_bc / (s_bc + eps))
        signed_d[tau] = tensor_dqrc[tau]["signed"]
        signed_s[tau] = tensor_standalone[tau]["signed"]
        bc_d[tau] = d_bc
        bc_s[tau] = s_bc

    suppression_flagged = eta.get(0, float("nan")) < suppression_threshold

    return RetainedNLResult(eta_NL_by_delay=eta, nl_dqrc_signed_by_delay=signed_d,
                             nl_standalone_signed_by_delay=signed_s, nl_dqrc_bc_by_delay=bc_d,
                             nl_standalone_bc_by_delay=bc_s, suppression_flagged=suppression_flagged)
