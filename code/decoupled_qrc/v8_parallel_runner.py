"""Aer execution for V8 parallel polynomial delay-register circuits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .v8_architectures.parallel_delay import (
    append_measurement_timestep,
    build_observable_circuit,
    measurement_batches,
    observable_width,
    specifications,
)


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
        circuits = [build_observable_circuit(spec) for spec in specs]
        result = backend.run(circuits, shots=None, seed_simulator=seed_simulator).result()
        if not result.success:
            raise RuntimeError(result.status)
        rows = [dict() for _ in values]
        for index, spec in enumerate(specs):
            data = result.data(index)
            for timestep in range(len(values)):
                rows[timestep][spec.name] = float(data[str(timestep)])
        executions = preparations = len(circuits)
        encodings = sum(len(spec.values) for spec in specs)
        total_shots = 0
    else:
        circuits, identities = [], []
        batches = measurement_batches(specs)
        for batch_index, batch in enumerate(batches):
            width = sum(item[2] for item in batch)
            circuit = QuantumCircuit(width)
            for timestep in range(len(values)):
                append_measurement_timestep(circuit, batch, timestep)
                measured = circuit.copy()
                measured.measure_all()
                circuits.append(measured)
                identities.append((batch_index, timestep))
        result = backend.run(circuits, shots=shots, seed_simulator=seed_simulator).result()
        if not result.success:
            raise RuntimeError(result.status)
        rows = [dict() for _ in values]
        for index, (batch_index, timestep) in enumerate(identities):
            counts = result.get_counts(index)
            for spec, offset, width in batches[batch_index]:
                mask = tuple(range(offset, offset + width))
                rows[timestep][spec.name] = sum(
                    (-1 if sum((int(bits.replace(" ", ""), 2) >> bit) & 1 for bit in mask) % 2 else 1) * count
                    for bits, count in counts.items()
                ) / shots
        executions = preparations = len(circuits)
        encodings = sum(len(spec.values) for spec in specs) * (len(values) + 1) // 2
        total_shots = executions * shots
    logical = ({"single_route": 1, "joint_route": 2} if mode == "exact" else
               {f"batch_{index}": sum(item[2] for item in batch) for index, batch in enumerate(batches)})
    resources = ParallelResources(
        distinct_reservoir_circuits=len(specs),
        logical_qubits_by_circuit=logical,
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
