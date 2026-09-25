"""Circuit-level V8 execution using an IBM fake-backend noise model."""

from __future__ import annotations

from time import perf_counter

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

from .v8_qiskit_runner import ROUTES, _append, _extract_counts, _new_circuit


FAKE_BACKEND_NAME = "FakeGuadalupeV2"


def run_fifo_noisy(inputs, m, g, *, shots, seed_simulator, seed_transpiler):
    """Replay and measure every prefix after transpilation to the fake target."""
    values = tuple(float(value) for value in inputs)
    fake_backend = FakeGuadalupeV2()
    noise_model = NoiseModel.from_backend(fake_backend)
    simulator = AerSimulator(method="matrix_product_state", noise_model=noise_model)
    rows = [dict() for _ in values]
    depths = []
    aggregate_counts = {}
    layouts = []
    started = perf_counter()

    for route in ROUTES:
        circuits = []
        for stop in range(1, len(values) + 1):
            circuit = _new_circuit(route)
            for value in values[:stop]:
                _append(circuit, route, value, m, g)
            circuit.measure_all()
            circuits.append(circuit)
        physical = transpile(
            circuits,
            backend=fake_backend,
            optimization_level=1,
            seed_transpiler=seed_transpiler,
        )
        result = simulator.run(
            physical,
            shots=shots,
            seed_simulator=seed_simulator,
        ).result()
        if not result.success:
            raise RuntimeError(result.status)
        for timestep, circuit in enumerate(physical):
            rows[timestep].update(_extract_counts(route, result.get_counts(timestep)))
            depths.append(circuit.depth())
            for gate, count in circuit.count_ops().items():
                aggregate_counts[gate] = aggregate_counts.get(gate, 0) + int(count)
        final_layout = getattr(physical[-1], "layout", None)
        layouts.append(str(final_layout) if final_layout is not None else None)

    return {
        "features": tuple(rows),
        "backend": {
            "name": "AerSimulator",
            "method": "matrix_product_state",
            "noise_model_from": FAKE_BACKEND_NAME,
            "fake_backend_qubits": fake_backend.num_qubits,
            "seed_simulator": seed_simulator,
            "seed_transpiler": seed_transpiler,
        },
        "resources": {
            "distinct_reservoir_circuits": len(ROUTES),
            "logical_qubits_by_circuit": {route: _new_circuit(route).num_qubits for route in ROUTES},
            "peak_logical_qubits": max(_new_circuit(route).num_qubits for route in ROUTES),
            "transpiled_physical_qubits": fake_backend.num_qubits,
            "measurement_bases": ["Z"],
            "dynamical_settings": len(ROUTES),
            "aer_jobs": len(ROUTES),
            "physical_circuit_executions": len(ROUTES) * len(values),
            "state_preparations": len(ROUTES) * len(values),
            "input_encodings": sum(range(1, len(values) + 1)) * 23,
            "shots_per_setting": shots,
            "total_shots": len(ROUTES) * len(values) * shots,
            "transpiled_depths": depths,
            "gate_counts": aggregate_counts,
            "swap_count": aggregate_counts.get("swap", 0),
            "layouts": layouts,
            "wall_seconds": perf_counter() - started,
        },
    }
