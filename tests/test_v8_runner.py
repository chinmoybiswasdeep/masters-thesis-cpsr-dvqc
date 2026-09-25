import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from decoupled_qrc.v8_measurements import expectation_from_counts, parity
from decoupled_qrc.v8_parallel_runner import run_parallel_delay


def test_scientific_features_invoke_real_aer_run_and_account_routes():
    from qiskit_aer import AerSimulator

    original = AerSimulator.run
    calls = []

    def instrumented(self, circuits, *args, **kwargs):
        calls.append(circuits)
        return original(self, circuits, *args, **kwargs)

    with patch.object(AerSimulator, "run", instrumented):
        result = run_parallel_delay((-0.4, 0.2, 0.7), 1.0, 1.0)
    assert calls and result["resources"]["aer_jobs"] == 1
    assert result["resources"]["distinct_reservoir_circuits"] == 202
    assert len(result["features"][0]) == 202


def test_temporal_delay_and_exact_cross_invariance():
    values = (-0.8, -0.2, 0.3, 0.9)
    low_g = run_parallel_delay(values, 1.0, 0.0)["features"]
    high_g = run_parallel_delay(values, 1.0, 1.0)["features"]
    low_m = run_parallel_delay(values, 0.0, 1.0)["features"]
    for timestep in range(1, len(values)):
        assert abs(low_g[timestep]["M:d1"] - values[timestep - 1]) < 1e-12
    memory = [name for name in low_g[0] if name.startswith("M:")]
    nonlinear = [name for name in low_g[0] if name.startswith("N:")]
    assert max(abs(low_g[t][name] - high_g[t][name]) for t in range(4) for name in memory) < 1e-12
    assert max(abs(low_m[t][name] - high_g[t][name]) for t in range(4) for name in nonlinear) < 1e-12


def test_same_bitstring_parity_and_qiskit_bit_ordering():
    counts = {"00": 40, "01": 10, "10": 20, "11": 30}
    assert parity("01", (0,)) == -1
    assert parity("10", (0,)) == 1
    assert expectation_from_counts(counts, (0, 1)) == 0.4


def test_finite_shots_use_measurement_counts_and_prefix_replay():
    result = run_parallel_delay((-0.5, 0.25), 1.0, 1.0, mode="shots", shots=100, seed_simulator=91)
    assert result["resources"]["circuit_executions"] == 202 * 2
    assert result["resources"]["state_preparations"] == 202 * 2
    assert result["resources"]["total_shots"] == 202 * 2 * 100
    assert all(-1.0 <= value <= 1.0 for row in result["features"] for value in row.values())
