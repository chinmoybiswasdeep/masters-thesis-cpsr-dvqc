import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from decoupled_qrc.v8_qiskit_runner import ROUTES, _run_exact_route, circuit_fingerprint, run_fifo


def test_physical_fifo_invokes_aer_for_every_route_and_replays_time():
    from qiskit_aer import AerSimulator

    original, calls = AerSimulator.run, []

    def instrumented(self, circuits, *args, **kwargs):
        calls.append(circuits)
        return original(self, circuits, *args, **kwargs)

    values = (-0.8, -0.2, 0.3, 0.9)
    with patch.object(AerSimulator, "run", instrumented):
        result = run_fifo(values, 1.0, 1.0)
    assert len(calls) == len(ROUTES)
    assert result["resources"]["peak_qubits"] == 15
    assert result["resources"]["distinct_reservoir_circuits"] == len(ROUTES)
    assert result["resources"]["dynamical_settings"] == len(ROUTES)
    assert result["resources"]["measurement_bases"] == 1
    assert abs(result["features"][3]["M:d1"] - values[2]) < 1e-11
    assert abs(result["features"][3]["M:d2"] - values[1]) < 1e-11
    expected_cubic = 4 * values[2] ** 3 - 3 * values[2]
    assert abs(result["features"][3]["J:delay_p3:d1"] - expected_cubic) < 1e-11
    mixed = (2 * values[3] ** 2 - 1) * (4 * values[2] ** 3 - 3 * values[2])
    assert abs(result["features"][3]["J:mix_p2_p3:d1"] - mixed) < 1e-11


def test_physical_fifo_exact_cross_invariance_and_shot_prefix_count():
    values = (-0.7, 0.1, 0.8)
    low_g = run_fifo(values, 1.0, 0.0)["features"]
    high_g = run_fifo(values, 1.0, 1.0)["features"]
    memory = [name for name in low_g[0] if name.startswith("M:")]
    assert max(abs(low_g[t][name] - high_g[t][name]) for t in range(3) for name in memory) < 1e-11
    shot = run_fifo(values[:2], 1.0, 1.0, mode="shots", shots=100, seed_simulator=17)
    assert shot["resources"]["circuit_executions"] == len(ROUTES) * 2
    assert shot["resources"]["total_shots"] == len(ROUTES) * 2 * 100


def test_qpy_fingerprint_is_deterministic_and_configuration_sensitive():
    inputs = (-0.4, 0.2, 0.9)
    first = circuit_fingerprint("memory", inputs, 1.0, 1.0)
    assert first == circuit_fingerprint("memory", inputs, 1.0, 1.0)
    assert first != circuit_fingerprint("memory", inputs, 0.5, 1.0)


def test_final_runner_has_no_analytical_or_numpy_quantum_fallback():
    root = Path(__file__).resolve().parents[1]
    source = (root / "code/decoupled_qrc/v8_qiskit_runner.py").read_text()
    architecture = (root / "code/decoupled_qrc/v8_architectures/fifo.py").read_text()
    forbidden = ("import numpy", "Statevector", "DensityMatrix", "binomial", "legendre", "target(")
    assert not any(token in source for token in forbidden)
    assert "target(" not in architecture


def test_every_exact_route_is_seed_invariant_without_entangled_resets():
    values = (-0.7, 0.1, 0.8)
    for route in ROUTES:
        left = _run_exact_route(route, values, 1.0, 1.0, 11, "double")[1]
        right = _run_exact_route(route, values, 1.0, 1.0, 991, "double")[1]
        assert max(
            abs(left[timestep][name] - right[timestep][name])
            for timestep in range(len(values)) for name in left[timestep]
        ) < 1e-12
