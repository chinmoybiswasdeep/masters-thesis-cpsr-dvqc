"""
test_gate_logic.py -- V2.2 Phase 10 / test requirement #13: conditional
propagation of NOT EVALUABLE from Gate B to Gates C/D/E.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.gate_logic import apply_not_evaluable_propagation, summarize_gates  # noqa: E402


def test_gate_b_pass_leaves_cde_unchanged():
    raw = {"C": "PASS", "D": "FAIL", "E": "PASS"}
    result = apply_not_evaluable_propagation("PASS", raw)
    assert result == raw


def test_gate_b_not_evaluable_forces_cde_not_evaluable():
    raw = {"C": "PASS", "D": "FAIL", "E": "PASS"}
    result = apply_not_evaluable_propagation("NOT EVALUABLE", raw)
    assert result == {"C": "NOT EVALUABLE", "D": "NOT EVALUABLE", "E": "NOT EVALUABLE"}


def test_gate_b_fail_also_forces_cde_not_evaluable():
    """An UNSTABLE derivative (Gate B = FAIL, not just NOT EVALUABLE) must
    still block C/D/E from being reported as a formal PASS/FAIL -- the
    rule is 'if B has not PASSED', not narrowly 'if B is exactly
    NOT EVALUABLE'."""
    raw = {"C": "FAIL", "D": "FAIL", "E": "FAIL"}
    result = apply_not_evaluable_propagation("FAIL", raw)
    assert result == {"C": "NOT EVALUABLE", "D": "NOT EVALUABLE", "E": "NOT EVALUABLE"}


def test_invalid_result_string_rejected():
    with pytest.raises(ValueError):
        apply_not_evaluable_propagation("MAYBE", {"C": "PASS"})
    with pytest.raises(ValueError):
        apply_not_evaluable_propagation("PASS", {"C": "MAYBE"})


def test_summarize_gates_counts_correctly():
    gates = {"A": {"result": "PASS"}, "B": {"result": "NOT EVALUABLE"}, "C": {"result": "FAIL"},
             "D": {"result": "NOT EVALUABLE"}}
    summary = summarize_gates(gates)
    assert summary == {"n_pass": 1, "n_fail": 1, "n_not_evaluable": 2, "n_total": 4}
