import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from decoupled_qrc.v8_evidence import _largest_absolute


def test_profile_collapse_keeps_worst_readout_signed_difference():
    assert _largest_absolute([0.01, -0.04, 0.02, 0.03]) == -0.04
    assert _largest_absolute([-0.01, 0.04, -0.02]) == 0.04


def test_no_numpy_quantum_evolution_symbols_in_evidence_module():
    source = (Path(__file__).resolve().parents[1] / "code/decoupled_qrc/v8_evidence.py").read_text()
    forbidden = ("Statevector", "DensityMatrix", "binomial", "from_instruction")
    assert not any(name in source for name in forbidden)
