import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from decoupled_qrc.v7_qiskit_runner import SETTINGS, run_sequence


def test_shots_replay_every_prefix_and_exact_does_not_measure():
    values = [-0.8, 0.0, 0.8]
    exact = run_sequence(values, 1.0, 1.0, mode="exact")
    shots = run_sequence(values, 1.0, 1.0, mode="shots", shots=20, seed_simulator=2)
    assert exact.resources.circuit_executions == len(SETTINGS)
    assert shots.resources.circuit_executions == len(values) * len(SETTINGS)
    assert shots.resources.total_shots == 20 * len(values) * len(SETTINGS)
    assert max(shots.resources.replay_depths) > min(shots.resources.replay_depths)

