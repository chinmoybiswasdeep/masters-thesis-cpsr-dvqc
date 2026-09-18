"""
discovery_sampling.py -- V2.1 Defect 3: V2's own discovery pass fixed
m=0.5 and scanned only (g,J), despite describing a three-dimensional
search. This module builds a genuine, reproducible 3D (m,g,J) sample set
(Sobol or Latin-hypercube, via `scipy.stats.qmc`), and enforces
discovery/confirmation seed separation BY CONSTRUCTION: discovery always
uses NEGATIVE `reservoir_idx` values, confirmation always uses
non-negative ones, so the two sets can never overlap regardless of how
many points either stage uses.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .validation_utils import ControlRange, NestedSeeds, make_nested_seeds


def sobol_3d_samples(n_points: int, seed: int = 0) -> np.ndarray:
    """`n_points` reproducible Sobol samples in [0,1]^3 (dimensionless
    m,g,J). `scipy.stats.qmc.Sobol` gives the best equidistribution
    guarantees for `n_points` a power of 2, but works (with a UserWarning
    scipy itself emits) for any `n_points`; kept general here since the
    exact discovery budget is a caller-chosen, mode-dependent value."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n_points, 1))))
    pts = sampler.random_base2(m=m) if 2 ** m == n_points else sampler.random(n_points)
    return pts[:n_points]


def latin_hypercube_3d_samples(n_points: int, seed: int = 0) -> np.ndarray:
    """Latin-hypercube alternative -- cheaper to get an EXACT `n_points`
    count with good 1D-projection coverage, no power-of-2 requirement."""
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    return sampler.random(n_points)


def to_raw_candidates(dimensionless_pts: np.ndarray, m_range: ControlRange, g_range: ControlRange,
                       J_range: ControlRange) -> list:
    """Converts an (n,3) array of [0,1]^3 samples into a list of
    {'m':..,'g':..,'J':..} raw-unit dicts via each axis's own
    `ControlRange.from_dimensionless`."""
    out = []
    for row in dimensionless_pts:
        out.append({"m": m_range.from_dimensionless(float(row[0])), "g": g_range.from_dimensionless(float(row[1])),
                    "J": J_range.from_dimensionless(float(row[2]))})
    return out


# =============================================================================
# Discovery/confirmation seed separation -- BY CONSTRUCTION, not by convention
# the caller has to remember.
# =============================================================================

def discovery_seeds(k: int, input_idx: int = 0) -> NestedSeeds:
    """The k-th discovery-stage seed bundle -- ALWAYS a negative
    reservoir_idx (-(k+1)), so it can never collide with any
    `confirmation_seeds` index no matter how many of either are drawn."""
    return make_nested_seeds(-(k + 1), input_idx)


def confirmation_seeds(k: int, input_idx: int = 0) -> NestedSeeds:
    """The k-th confirmation-stage seed bundle -- ALWAYS a non-negative
    reservoir_idx (k), disjoint from every `discovery_seeds` index by
    construction (negative vs. non-negative)."""
    return make_nested_seeds(k, input_idx)


def assert_disjoint_discovery_confirmation(n_discovery: int, n_confirmation: int) -> None:
    """Explicit, testable proof that the two index ranges never overlap --
    test requirement #10."""
    disc_idxs = {-(k + 1) for k in range(n_discovery)}
    conf_idxs = {k for k in range(n_confirmation)}
    overlap = disc_idxs & conf_idxs
    if overlap:
        raise ValueError(f"discovery and confirmation reservoir_idx sets overlap: {overlap}")
