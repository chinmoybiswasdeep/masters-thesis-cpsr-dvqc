"""
gate_logic.py -- V2.2's explicit rule (Phase 10 / acceptance-gates
section): "If Gate B is not evaluable, Gates C-E must also be
NOT EVALUABLE." A derivative-dependent ratio (R_M, R_NL, angle) computed
from an unstable/not-evaluable derivative is not a meaningful FAIL --
reporting it as FAIL would overstate what the data actually show. This
module is the single, explicit, testable place that rule lives, so no
notebook cell can silently compute R_M/R_NL/angle as PASS/FAIL without
checking derivative stability first.
"""
from __future__ import annotations

VALID_RESULTS = ("PASS", "FAIL", "NOT EVALUABLE")


def apply_not_evaluable_propagation(gate_b_result: str, raw_gate_results: dict) -> dict:
    """`raw_gate_results` = {'C': 'PASS'|'FAIL'|'NOT EVALUABLE', 'D': ..., 'E': ...}
    (the NAIVE numeric-threshold verdicts, computed WITHOUT regard to
    derivative stability). If `gate_b_result != 'PASS'`, every one of C/D/E
    is forced to 'NOT EVALUABLE' regardless of its naive verdict --
    a decisive derivative that happens to be unstable must never be
    reported as a formal FAIL (or PASS)."""
    if gate_b_result not in VALID_RESULTS:
        raise ValueError(f"gate_b_result must be one of {VALID_RESULTS}, got {gate_b_result!r}")
    for k, v in raw_gate_results.items():
        if v not in VALID_RESULTS:
            raise ValueError(f"raw_gate_results[{k!r}] must be one of {VALID_RESULTS}, got {v!r}")

    if gate_b_result == "PASS":
        return dict(raw_gate_results)
    return {k: "NOT EVALUABLE" for k in raw_gate_results}


def summarize_gates(gates: dict) -> dict:
    """{'n_pass':.., 'n_fail':.., 'n_not_evaluable':..} over every gate's
    `result` field."""
    n_pass = sum(1 for g in gates.values() if g.get("result") == "PASS")
    n_fail = sum(1 for g in gates.values() if g.get("result") == "FAIL")
    n_ne = sum(1 for g in gates.values() if g.get("result") == "NOT EVALUABLE")
    return {"n_pass": n_pass, "n_fail": n_fail, "n_not_evaluable": n_ne, "n_total": len(gates)}
