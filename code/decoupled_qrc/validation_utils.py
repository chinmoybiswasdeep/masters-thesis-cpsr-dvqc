"""
validation_utils.py -- Parts 6/7/10 of the V2 validation spec: independent
nested seed streams, dimensionless control coordinates, and local
polynomial response-surface fitting (an alternative to bare finite
differences, for cross-checking derivative stability).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# =============================================================================
# Part 6 -- independent, nested seed streams
# =============================================================================

@dataclass(frozen=True)
class NestedSeeds:
    """Every source of randomness this validation touches, each an
    INDEPENDENT stream derived from (reservoir_idx, input_idx) so that,
    e.g., re-running with a different bootstrap_seed can never accidentally
    also change which input sequence or which disorder realization was
    used (Part 6's explicit common-random-numbers requirement)."""
    reservoir_idx: int
    input_idx: int
    reservoir_seed: int
    disorder_seed: int
    input_seed: int
    split_seed: int
    projection_seed: int
    bootstrap_seed: int
    shot_seed: int


def make_nested_seeds(reservoir_idx: int, input_idx: int = 0) -> NestedSeeds:
    """SHA256-derived child seeds -- same pattern `utils.make_seed_bundle`
    already established in this project, extended to the full nested
    reservoir x input design Part 6 requires, plus split/projection/
    bootstrap/shot streams that bundle didn't need."""
    def child(tag: str) -> int:
        h = hashlib.sha256(f"dqrc_v2:{reservoir_idx}:{input_idx}:{tag}".encode()).digest()
        return int.from_bytes(h[:4], "big")

    return NestedSeeds(
        reservoir_idx=reservoir_idx, input_idx=input_idx,
        reservoir_seed=child("reservoir"), disorder_seed=child("disorder"),
        input_seed=child("input"), split_seed=child("split"), projection_seed=child("projection"),
        bootstrap_seed=child("bootstrap"), shot_seed=child("shot"),
    )


# =============================================================================
# Part 7 -- dimensionless control coordinates
# =============================================================================

@dataclass(frozen=True)
class ControlRange:
    name: str
    p_min: float
    p_max: float

    def to_dimensionless(self, p: float) -> float:
        return (p - self.p_min) / (self.p_max - self.p_min)

    def from_dimensionless(self, p_tilde: float) -> float:
        return self.p_min + p_tilde * (self.p_max - self.p_min)


def stencil_margin_ok(p_tilde: float, h_tilde: float, k: int = 2) -> bool:
    """V2.1 Defect 1: a candidate point at dimensionless coordinate
    `p_tilde` (in [0,1]) can only support a `k`-step-wide symmetric stencil
    (default k=2, i.e. offsets {-2h,-h,0,h,2h}) if EVERY offset stays
    in-domain: `k*h_tilde <= p_tilde <= 1 - k*h_tilde`. V2's own candidate
    selection picked J*=0.6 (the declared range's own upper bound) and then
    evaluated J*+h_J=0.65, outside [0.05,0.6] -- this is the check that
    catches that mistake before any circuit is ever simulated."""
    return (k * h_tilde) <= p_tilde <= (1.0 - k * h_tilde)


def assert_in_domain(value: float, p_min: float, p_max: float, name: str) -> None:
    """Raise if `value` (a RAW, not dimensionless, parameter value) falls
    outside its declared [p_min, p_max] domain. Every raw-unit stencil
    evaluation in V2.1's response estimation calls this before running a
    single circuit -- an out-of-domain evaluation is a hard error, not a
    warning, per Defect 1's 'never use a symmetric stencil at a point
    without sufficient margin from every boundary' rule."""
    if not (p_min <= value <= p_max):
        raise ValueError(f"{name}={value!r} is outside its declared domain [{p_min}, {p_max}] -- "
                          f"refusing to evaluate a parameter point outside its own preregistered range.")


def robust_output_scale(values: Sequence[float]) -> float:
    """Interquartile range of a discovery-scan output column -- the 'robust
    output scale' Part 7 asks response-vector axes to be normalized by
    (more outlier-resistant than std for a scan that may include a few
    ceiling-contaminated or degenerate points)."""
    values = np.asarray(values, dtype=float)
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    return float(iqr) if iqr > 1e-9 else float(np.std(values) + 1e-9)


# =============================================================================
# Part 10 -- local polynomial response-surface fitting
# =============================================================================

@dataclass
class ResponseSurfaceFit:
    coeffs: dict          # {'const','m','g','J','mg','mJ','gJ','mm','gg','JJ'} -> float
    residual_rms: float
    n_points: int

    def gradient_at(self, m_tilde: float = 0.0, g_tilde: float = 0.0, J_tilde: float = 0.0) -> tuple:
        """Analytic gradient of the fitted quadratic surface at the given
        dimensionless point (default: the surface's own center, m=g=J=0)."""
        c = self.coeffs
        dY_dm = c["m"] + c["mg"] * g_tilde + c["mJ"] * J_tilde + 2 * c["mm"] * m_tilde
        dY_dg = c["g"] + c["mg"] * m_tilde + c["gJ"] * J_tilde + 2 * c["gg"] * g_tilde
        dY_dJ = c["J"] + c["mJ"] * m_tilde + c["gJ"] * g_tilde + 2 * c["JJ"] * J_tilde
        return dY_dm, dY_dg, dY_dJ


def fit_response_surface(m_tilde: np.ndarray, g_tilde: np.ndarray, J_tilde: np.ndarray,
                          y: np.ndarray) -> ResponseSurfaceFit:
    """Least-squares fit of
        Y = b0 + bm*m + bg*g + bJ*J + bmg*mg + bmJ*mJ + bgJ*gJ + bmm*m^2 + bgg*g^2 + bJJ*J^2
    via `np.linalg.lstsq` (never an explicit normal-equation matrix
    inverse). `m_tilde,g_tilde,J_tilde` are coordinates RELATIVE TO the
    candidate point (i.e. already centered, so the fitted gradient at the
    origin is directly the derivative AT the candidate point)."""
    m_tilde, g_tilde, J_tilde, y = map(np.asarray, (m_tilde, g_tilde, J_tilde, y))
    A = np.column_stack([
        np.ones_like(m_tilde), m_tilde, g_tilde, J_tilde,
        m_tilde * g_tilde, m_tilde * J_tilde, g_tilde * J_tilde,
        m_tilde ** 2, g_tilde ** 2, J_tilde ** 2,
    ])
    beta, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    rms = float(np.sqrt(np.mean((pred - y) ** 2)))
    names = ["const", "m", "g", "J", "mg", "mJ", "gJ", "mm", "gg", "JJ"]
    coeffs = {name: float(val) for name, val in zip(names, beta)}
    return ResponseSurfaceFit(coeffs=coeffs, residual_rms=rms, n_points=len(y))


def stencil_offsets_5point(h: float) -> list:
    """Symmetric 5-point stencil offsets {-2h,-h,0,h,2h} for ONE control
    axis, Part 10's own example -- used to build the (m,g,J) neighborhood
    sample set (one axis varied at a time around the candidate point, a
    cheaper design than a full 5^3 cube while still giving 2 independent
    step-size central differences PLUS enough points for a linear/
    quadratic local fit along each axis)."""
    return [-2 * h, -h, 0.0, h, 2 * h]


# =============================================================================
# Derivative stability flagging (Part 8/Gate B), shared by every derivative
# estimate this validation computes.
# =============================================================================

def is_unstable(estimates: Sequence[float], spread_threshold: float = 0.3) -> dict:
    """Flag a derivative as unstable if its sign changes across the given
    `estimates` (e.g. different step sizes / different estimation methods)
    OR its relative spread exceeds `spread_threshold` -- Part 8's exact
    rule, used everywhere this validation reports a derivative."""
    estimates = np.asarray(estimates, dtype=float)
    sign_change = bool(np.any(estimates > 0) and np.any(estimates < 0))
    mean = np.mean(estimates)
    spread = float(np.std(estimates) / (abs(mean) + 1e-9))
    return {"estimates": estimates.tolist(), "mean": float(mean), "relative_spread": spread,
            "sign_change": sign_change, "unstable": bool(sign_change or spread > spread_threshold)}
