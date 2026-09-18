"""
run_stages.py -- V2.1 Defect 2 fix: "FULL mode is only nominal". V2's
`RUN_MODE="FULL"` branch changed a config dict's numbers but nothing in
the notebook actually LOOPED over 5 reservoir seeds x 3 input seeds, ran
held-out confirmation, or evaluated a neighborhood -- every real execution
path only ever touched 1 seed.

This module is the actual staged control flow (discovery -> candidate
selection -> held-out confirmation -> neighborhood robustness), as plain,
independently-testable Python functions that take a PLUGGABLE
`evaluate_fn(m, g, J, seeds) -> result` callable -- so
`tests/test_run_stages.py` can verify the real seed-loop counts and
discovery/confirmation separation using a cheap monkeypatched fake,
without paying for real quantum circuit simulation in every CI run. The
notebook wires `evaluate_fn = candidate_eval.evaluate_candidate_cached`
(partially applied over the shared config) for actual execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .discovery_sampling import (discovery_seeds, confirmation_seeds, sobol_3d_samples, to_raw_candidates)
from .validation_utils import ControlRange, stencil_offsets_5point


@dataclass
class DiscoveryOutcome:
    all_results: list                 # every sampled point's result, valid or not (Defect 3: "store every point")
    n_sampled: int
    n_valid: int
    reservoir_idxs_used: set = field(default_factory=set)


def discovery_stage(n_points: int, m_range: ControlRange, g_range: ControlRange, J_range: ControlRange,
                     evaluate_fn: Callable, sampling_seed: int = 0, input_idx: int = 0) -> DiscoveryOutcome:
    """Visits `n_points` Sobol-sampled (m,g,J) candidates, each with its
    OWN discovery-only seed bundle (`discovery_seeds(k, input_idx)` --
    always a negative `reservoir_idx`), and calls `evaluate_fn(m, g, J,
    seeds)` for each. Every result is kept (valid or not), per Defect 3's
    'save accepted and rejected points, including explicit rejection
    reasons' -- filtering/ranking happens downstream, not here."""
    dimless = sobol_3d_samples(n_points, seed=sampling_seed)
    candidates = to_raw_candidates(dimless, m_range, g_range, J_range)
    results, idxs_used = [], set()
    for k, cand in enumerate(candidates):
        seeds = discovery_seeds(k, input_idx)
        idxs_used.add(seeds.reservoir_idx)
        result = evaluate_fn(cand["m"], cand["g"], cand["J"], seeds)
        results.append(result)
    n_valid = sum(1 for r in results if getattr(r, "valid", False))
    return DiscoveryOutcome(all_results=results, n_sampled=len(results), n_valid=n_valid,
                             reservoir_idxs_used=idxs_used)


@dataclass
class ConfirmationOutcome:
    candidate: dict                   # the frozen (m,g,J) center point -- must not change during confirmation
    per_seed_results: list            # flat list of (reservoir_idx, input_idx, result)
    n_reservoir_seeds: int
    n_input_seeds: int
    reservoir_idxs_used: set = field(default_factory=set)


def confirmation_stage(candidate: dict, n_reservoir_seeds: int, n_input_seeds: int, evaluate_fn: Callable,
                        discovery_reservoir_idxs: set = frozenset()) -> ConfirmationOutcome:
    """Visits EVERY (reservoir_idx, input_idx) combination in the
    `n_reservoir_seeds x n_input_seeds` confirmation grid, using
    `confirmation_seeds` (always non-negative `reservoir_idx`). Raises if
    any confirmation index was also used during discovery (the seed-
    separation guarantee is enforced BY CONSTRUCTION via the negative/
    non-negative split, but this is an explicit, testable belt-and-braces
    check in case a caller passes a stale/hand-built discovery index set).
    `candidate` (m,g,J) is fixed for the entire call -- confirmation must
    never alter candidate points, per Phase 11."""
    per_seed, idxs_used = [], set()
    for r_idx in range(n_reservoir_seeds):
        if r_idx in discovery_reservoir_idxs:
            raise ValueError(f"confirmation reservoir_idx={r_idx} was also used during discovery -- "
                              f"held-out confirmation must use disjoint seeds.")
        idxs_used.add(r_idx)
        for i_idx in range(n_input_seeds):
            seeds = confirmation_seeds(r_idx, i_idx)
            result = evaluate_fn(candidate["m"], candidate["g"], candidate["J"], seeds)
            per_seed.append((r_idx, i_idx, result))
    return ConfirmationOutcome(candidate=dict(candidate), per_seed_results=per_seed,
                                n_reservoir_seeds=n_reservoir_seeds, n_input_seeds=n_input_seeds,
                                reservoir_idxs_used=idxs_used)


@dataclass
class NeighborhoodOutcome:
    candidate: dict
    offsets: list                     # [(axis, offset_tilde), ...]
    results: list                     # parallel list of results, one per offset (plus the center)


def neighborhood_stage(candidate: dict, m_range: ControlRange, g_range: ControlRange, J_range: ControlRange,
                        h_tilde: float, evaluate_fn: Callable, seeds) -> NeighborhoodOutcome:
    """Evaluates the candidate center PLUS a small neighborhood around it
    (one axis perturbed at a time, using the SAME 5-point stencil offsets
    `stencil_offsets_5point` already builds for local derivatives) on the
    SAME confirmation `seeds` object (common random numbers across the
    neighborhood) -- Gate I's 'holds over a finite region, not one point'
    evidence."""
    offsets = stencil_offsets_5point(h_tilde)
    m0_t, g0_t, J0_t = m_range.to_dimensionless(candidate["m"]), g_range.to_dimensionless(candidate["g"]), \
        J_range.to_dimensionless(candidate["J"])
    entries, results = [("center", 0.0)], [evaluate_fn(candidate["m"], candidate["g"], candidate["J"], seeds)]
    for axis, rng, base_t in (("m", m_range, m0_t), ("g", g_range, g0_t), ("J", J_range, J0_t)):
        for off in offsets:
            if off == 0.0:
                continue
            p_t = base_t + off
            if not (0.0 <= p_t <= 1.0):
                continue
            p_raw = rng.from_dimensionless(p_t)
            m_eval = p_raw if axis == "m" else candidate["m"]
            g_eval = p_raw if axis == "g" else candidate["g"]
            J_eval = p_raw if axis == "J" else candidate["J"]
            entries.append((axis, off))
            results.append(evaluate_fn(m_eval, g_eval, J_eval, seeds))
    return NeighborhoodOutcome(candidate=dict(candidate), offsets=entries, results=results)
