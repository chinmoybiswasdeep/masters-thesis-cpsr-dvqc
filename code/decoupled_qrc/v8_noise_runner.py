"""Circuit-level V8 execution using an IBM fake-backend noise model."""

from __future__ import annotations

from time import perf_counter

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

from .v8_qiskit_runner import PREFIX_BATCH_SIZE, ROUTES, _append, _extract_counts, _new_circuit


FAKE_BACKEND_NAME = "FakeGuadalupeV2"
PHYSICAL_SUBGRAPH = tuple(range(16))


def run_fifo_noisy(inputs, m, g, *, shots, seed_simulator, seed_transpiler):
    """Replay and measure every prefix after transpilation to the fake target."""
    values = tuple(float(value) for value in inputs)
    fake_backend = FakeGuadalupeV2()
    noise_model = NoiseModel.from_backend(fake_backend)
    simulator = AerSimulator(method="matrix_product_state", noise_model=noise_model)
    rows = [dict() for _ in values]
    depths = []
    aggregate_counts = {}
    logical_swap_count = 0
    one_qubit_gates = 0
    two_qubit_gates = 0
    layouts = []
    aer_jobs = 0
    started = perf_counter()

    for route in ROUTES:
        final_layout = None
        for first in range(0, len(values), PREFIX_BATCH_SIZE):
            stops = range(first + 1, min(first + PREFIX_BATCH_SIZE, len(values)) + 1)
            circuits = []
            for stop in stops:
                circuit = _new_circuit(route)
                for value in values[:stop]:
                    _append(circuit, route, value, m, g)
                circuit.measure_all()
                circuits.append(circuit)
                logical_swap_count += int(circuit.count_ops().get("swap", 0))
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
            aer_jobs += 1
            if not result.success:
                raise RuntimeError(result.status)
            for index, (stop, circuit) in enumerate(zip(stops, physical)):
                rows[stop - 1].update(_extract_counts(route, result.get_counts(index)))
                depths.append(circuit.depth())
                for gate, count in circuit.count_ops().items():
                    aggregate_counts[gate] = aggregate_counts.get(gate, 0) + int(count)
                for instruction in circuit.data:
                    if instruction.operation.num_qubits == 1 and instruction.operation.name not in {"measure", "reset", "barrier"}:
                        one_qubit_gates += 1
                    elif instruction.operation.num_qubits == 2:
                        two_qubit_gates += 1
            final_layout = getattr(physical[-1], "layout", None)
        layouts.append(str(final_layout) if final_layout is not None else None)

    return {
        "features": tuple(rows),
        "backend": {
            "name": "AerSimulator",
            "method": "matrix_product_state",
            "noise_model_from": FAKE_BACKEND_NAME,
            "fake_backend_qubits": fake_backend.num_qubits,
            "physical_subgraph": PHYSICAL_SUBGRAPH,
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
            "aer_jobs": aer_jobs,
            "physical_circuit_executions": len(ROUTES) * len(values),
            "state_preparations": len(ROUTES) * len(values),
            "input_encodings": sum(range(1, len(values) + 1)) * 23,
            "shots_per_setting": shots,
            "total_shots": len(ROUTES) * len(values) * shots,
            "transpiled_depths": depths,
            "gate_counts": aggregate_counts,
            "one_qubit_gates": one_qubit_gates,
            "two_qubit_gates": two_qubit_gates,
            "swap_count": aggregate_counts.get("swap", 0),
            "logical_swap_count_before_basis_translation": logical_swap_count,
            "layouts": layouts,
            "wall_seconds": perf_counter() - started,
        },
    }
