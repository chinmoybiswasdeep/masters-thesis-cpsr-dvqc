import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from qiskit_aer import AerSimulator
from decoupled_qrc.v7_qiskit_runner import run_sequence


def test_exact_and_shots_invoke_aer_run():
    original = AerSimulator.run
    with patch.object(AerSimulator, "run", autospec=True, side_effect=original) as called:
        exact = run_sequence([-0.5, 0.25], 1.0, 1.0, mode="exact")
        shots = run_sequence([-0.5, 0.25], 1.0, 1.0, mode="shots", shots=100, seed_simulator=4)
    assert called.call_count >= 2
    assert len(exact.rows) == len(shots.rows) == 2
    assert exact.circuit_fingerprints and shots.circuit_fingerprints

