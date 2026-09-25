import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from decoupled_qrc.v7_gates import hh_gate
from decoupled_qrc.v7_protocol import load_protocol


def test_hh_gate_rejects_amplitude_or_ordering_failure():
    protocol = load_protocol()
    assert hh_gate({"LL": 0.0, "LH": 0.1, "HL": 0.1, "HH": 0.2}, protocol)["passed"]
    assert not hh_gate({"LL": 0.0, "LH": 0.19, "HL": 0.1, "HH": 0.2}, protocol)["passed"]

