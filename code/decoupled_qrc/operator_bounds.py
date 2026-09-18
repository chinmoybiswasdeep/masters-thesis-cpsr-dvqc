"""
operator_bounds.py -- V3 fix for the invalid chaos-diagnostic
normalisation V2.2 exposed: the saved V2.2 result
`phase14_eoc_finalist` reports `operator_entanglement = 1.7537` while the
repository's own printed "max possible" was `log(2^(N_P//2)) = 1.386` --
a reported value ABOVE its own stated bound.

The stated bound was the one appropriate for a pure STATE's entanglement
entropy across a bipartition. The operator entanglement of a unitary is
the entanglement entropy of its Choi/operator-Schmidt vector on the
DOUBLED Hilbert space, whose correct maximum is

    S_op_max = 2 * log(min(d_A, d_B))

(for a bipartition into dimensions d_A, d_B). For N = 5 with
d_A = 2^2 = 4, d_B = 2^3 = 8 this is 2*log(4) = log(16) = 2.7726, so
1.7537 is in fact perfectly legal -- the VALUE was fine, the BOUND was
wrong.

This module computes the correct bound, validates any reported value
against it, and FAILS LOUDLY on a genuine violation. Known reference
cases (identity, product unitary, SWAP, Haar-random) are checked in
`tests/test_operator_bounds.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def operator_entanglement_max(n_qubits: int, n_a: int = None) -> float:
    """Correct maximum operator entanglement (in nats) for a unitary on
    `n_qubits`, bipartitioned into `n_a` and `n_qubits - n_a` qubits:
        S_max = 2 * log(min(d_A, d_B)).
    Defaults to the balanced split `n_a = n_qubits // 2`, matching this
    repository's own `mixed_syk_core.operator_entanglement`."""
    n_a = n_qubits // 2 if n_a is None else n_a
    d_a, d_b = 2 ** n_a, 2 ** (n_qubits - n_a)
    return float(2.0 * np.log(min(d_a, d_b)))


def operator_entanglement(u: np.ndarray, n_qubits: int, n_a: int = None) -> float:
    """Operator (Schmidt) entanglement entropy of a unitary, in nats --
    same construction as `mixed_syk_core.operator_entanglement` (reshape
    to the operator-Schmidt matrix, singular values, Shannon entropy of
    the normalised squared spectrum)."""
    n_a = n_qubits // 2 if n_a is None else n_a
    n_b = n_qubits - n_a
    d_a, d_b = 2 ** n_a, 2 ** n_b
    mat = u.reshape(d_a, d_b, d_a, d_b).transpose(0, 2, 1, 3).reshape(d_a * d_a, d_b * d_b)
    s = np.linalg.svd(mat, compute_uv=False)
    p = s ** 2
    total = p.sum()
    if total <= 1e-14:
        return 0.0
    p = p / total
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


@dataclass
class BoundCheck:
    value: float
    maximum: float
    normalized: float           # value / maximum, in [0, 1] when valid
    valid: bool
    message: str


def validate_operator_entanglement(value: float, n_qubits: int, n_a: int = None,
                                    tol: float = 1e-8) -> BoundCheck:
    """Validates a reported operator entanglement against the CORRECT
    bound. A value above the bound (beyond `tol`) is a hard error
    condition -- `valid=False` with an explanatory message, so the caller
    can mark the diagnostic NOT EVALUABLE rather than quietly reporting an
    impossible number."""
    maximum = operator_entanglement_max(n_qubits, n_a)
    if not np.isfinite(value):
        return BoundCheck(value=float(value), maximum=maximum, normalized=float("nan"), valid=False,
                           message="non-finite operator entanglement")
    if value < -tol:
        return BoundCheck(value=float(value), maximum=maximum, normalized=float("nan"), valid=False,
                           message=f"negative operator entanglement ({value:.6g})")
    if value > maximum + tol:
        return BoundCheck(value=float(value), maximum=maximum, normalized=float("nan"), valid=False,
                           message=(f"operator entanglement {value:.6g} EXCEEDS its mathematical maximum "
                                     f"{maximum:.6g} for n_qubits={n_qubits} -- diagnostic is invalid"))
    return BoundCheck(value=float(value), maximum=maximum, normalized=float(value / maximum),
                       valid=True, message="")


def audit_v22_operator_entanglement(value: float = 1.7537322258856058, n_qubits: int = 5) -> dict:
    """Re-audits the exact V2.2 saved value against both the WRONG bound
    the repository printed and the CORRECT bound, for the V3 report."""
    wrong_bound = float(np.log(2 ** (n_qubits // 2)))
    correct = validate_operator_entanglement(value, n_qubits)
    return {"value": value, "n_qubits": n_qubits, "repository_printed_bound": wrong_bound,
            "exceeds_repository_printed_bound": bool(value > wrong_bound),
            "correct_bound": correct.maximum, "valid_under_correct_bound": correct.valid,
            "normalized_under_correct_bound": correct.normalized}
