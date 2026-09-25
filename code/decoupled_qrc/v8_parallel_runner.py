"""Aer execution for V8 parallel polynomial delay-register circuits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from qiskit_aer import AerSimulator

from .v8_architectures.parallel_delay import (
    build_observable_batches,
    build_observable_circuit,
    build_serial_observable_circuit,
    observable_width,
    specifications,
)
from .v8_measurements import expectation_from_probabilities


@dataclass(frozen=True)
class ParallelResources:
    distinct_reservoir_circuits: int
    logical_qubits_by_circuit: dict
    peak_qubits: int
    measurement_bases: int
    aer_jobs: int
    circuit_executions: int
    state_preparations: int
    input_encodings: int
    shots_per_setting: int
    total_shots: int
    feature_count: int
    wall_seconds: float


def run_parallel_delay(inputs, m, g, *, mode="exact", shots=1000, seed_simulator=0, precision="double"):
    if mode not in {"exact", "shots"}:
        raise ValueError("mode must be exact or shots")
    values = tuple(float(value) for value in inputs)
    specs = specifications(values, m, g)
    method = "statevector" if mode == "exact" else "matrix_product_state"
    backend = AerSimulator(method=method, precision=precision)
    started = perf_counter()
    if mode == "exact":
        circuits = [build_serial_observable_circuit(specs)]
        result = backend.run(circuits, shots=None, seed_simulator=seed_simulator).result()
        if not result.success:
            raise RuntimeError(result.status)
        rows = [dict() for _ in values]
        data = result.data(0)
        for spec in specs:
            for timestep in range(len(values)):
                rows[timestep][spec.name] = float(data[f"{timestep}:{spec.name}"])
        executions = preparations = len(circuits)
        encodings = sum(len(spec.values) for spec in specs)
        total_shots = 0
    else:
        circuits, identities = [], []
        for spec in specs:
            width = observable_width(spec)
            for timestep in range(len(values)):
                circuit = build_observable_circuit(
                    type(spec)(spec.name, spec.values[: (timestep + 1) * width])
                )
                circuit.measure_all()
                circuits.append(circuit)
                identities.append((spec.name, timestep, width))
        result = backend.run(circuits, shots=shots, seed_simulator=seed_simulator).result()
        if not result.success:
            raise RuntimeError(result.status)
        rows = [dict() for _ in values]
        for index, (name, timestep, width) in enumerate(identities):
            counts = result.get_counts(index)
            rows[timestep][name] = sum(
                (-1 if bits.replace(" ", "").count("1") % 2 else 1) * count
                for bits, count in counts.items()
            ) / shots
        executions = preparations = len(circuits)
        encodings = sum(
            len(spec.values) // observable_width(spec)
            * (len(spec.values) + observable_width(spec)) // 2
            for spec in specs
        )
        total_shots = executions * shots
    resources = ParallelResources(
        distinct_reservoir_circuits=len(specs),
        logical_qubits_by_circuit={str(index): circuit.num_qubits for index, circuit in enumerate(circuits)},
        peak_qubits=max(circuit.num_qubits for circuit in circuits),
        measurement_bases=1,
        aer_jobs=1,
        circuit_executions=executions,
        state_preparations=preparations,
        input_encodings=encodings,
        shots_per_setting=0 if mode == "exact" else shots,
        total_shots=total_shots,
        feature_count=len(specs),
        wall_seconds=perf_counter() - started,
    )
    return {
        "features": tuple(rows),
        "resources": asdict(resources),
        "backend": {"name": "AerSimulator", "method": method, "precision": precision},
    }
