"""
safe_ratio.py -- V3 fix for the V2.2 defect where a retained-NL ratio
exploded to ~3.75e7 because the standalone bias-corrected denominator had
crossed zero (verified in
`results/dqrc_gj_validation_v2_2/validation_v2_2_results.json`:
`phase6_retained_nl.eta_NL_by_delay['5'] = 37547771.17`).

A ratio is returned ONLY when its denominator is positive, exceeds a
preregistered absolute floor, AND stands clear of its own null
uncertainty. Otherwise the result is `NOT EVALUABLE` with an explicit
rejection reason -- never an epsilon-rescued enormous number.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class SafeRatio:
    numerator: float
    denominator: float
    denominator_uncertainty: float
    ratio: float                 # float('nan') when not evaluable
    evaluable: bool
    rejection_reason: str
    floor: float
    uncertainty_multiple: float

    def as_dict(self) -> dict:
        return asdict(self)


def safe_ratio(numerator: float, denominator: float, denominator_uncertainty: float = 0.0,
                floor: float = 0.05, uncertainty_multiple: float = 2.0) -> SafeRatio:
    """`floor` and `uncertainty_multiple` are preregistered thresholds:
    the denominator must be positive, at least `floor` in magnitude, and
    at least `uncertainty_multiple` times its own uncertainty."""
    def _reject(reason):
        return SafeRatio(numerator=float(numerator), denominator=float(denominator),
                          denominator_uncertainty=float(denominator_uncertainty), ratio=float("nan"),
                          evaluable=False, rejection_reason=reason, floor=floor,
                          uncertainty_multiple=uncertainty_multiple)

    if not _is_finite(denominator) or not _is_finite(numerator):
        return _reject("non_finite_input")
    if denominator <= 0.0:
        return _reject(f"denominator_not_positive ({denominator:.4g})")
    if denominator < floor:
        return _reject(f"denominator_below_floor ({denominator:.4g} < {floor:.4g})")
    if denominator_uncertainty > 0.0 and denominator < uncertainty_multiple * denominator_uncertainty:
        return _reject(f"denominator_within_{uncertainty_multiple:g}x_uncertainty "
                        f"({denominator:.4g} < {uncertainty_multiple * denominator_uncertainty:.4g})")

    return SafeRatio(numerator=float(numerator), denominator=float(denominator),
                      denominator_uncertainty=float(denominator_uncertainty),
                      ratio=float(numerator / denominator), evaluable=True, rejection_reason="",
                      floor=floor, uncertainty_multiple=uncertainty_multiple)


def _is_finite(x) -> bool:
    try:
        return float(x) == float(x) and abs(float(x)) != float("inf")
    except (TypeError, ValueError):
        return False


def retained_nl_safe(nl_architecture: float, nl_standalone: float,
                      standalone_uncertainty: float = 0.0, floor: float = 0.05) -> SafeRatio:
    """eta_NL = NL_architecture / NL_standalone, with the V3 safety rules.
    Replaces V2.2's `retained_nl_ratio`'s bare `x / (y + eps)`."""
    return safe_ratio(nl_architecture, nl_standalone, standalone_uncertainty, floor=floor)
