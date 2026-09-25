"""Aer-only modular execution for V8 FIFO reservoirs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .v8_architectures.fifo import _angle, _write_fifo, transport_depth
from .v8_measurements import expectation_from_counts, local_z_product


TAPS = 13
ROUTES = (
    "memory", "processor", "current1", "current2", "current3", "current4",
    "delayed2", "delayed3", "delayed4", "pair",
)


@dataclass(frozen=True)
class Resources:
    distinct_reservoir_circuits: int
    logical_qubits_by_circuit: dict
    peak_qubits: int
    dynamical_settings: int
    measurement_bases: int
    aer_jobs: int
    circuit_executions: int
    state_preparations: int
    input_encodings: int
    shots_per_setting: int
    total_shots: int
    replay_depths: tuple[int, ...]
    wall_seconds: float


def _append(circuit, route, value, m, g):
    theta, depth = _angle(value), transport_depth(m)
    if route == "memory":
        _write_fifo(circuit, range(TAPS), TAPS, theta, depth)
    elif route == "processor":
        for degree, qubit in zip((1, 2, 3, 4), range(4)):
            circuit.reset(qubit)
            circuit.ry((1.0 + (degree - 1) * g) * theta, qubit)
    elif route.startswith("current"):
        degree = int(route[-1])
        _write_fifo(circuit, range(TAPS), TAPS, theta, depth)
        circuit.reset(TAPS + 1)
        circuit.ry((1.0 + (degree - 1) * g) * theta, TAPS + 1)
    elif route.startswith("delayed"):
        degree = int(route[-1])
        _write_fifo(circuit, range(TAPS), 2 * TAPS, theta, depth)
        _write_fifo(circuit, range(TAPS, 2 * TAPS), 2 * TAPS, (1.0 + (degree - 1) * g) * theta, depth)
    else:
        _write_fifo(circuit, range(TAPS), TAPS, g * theta, depth)


def _new_circuit(route):
    return QuantumCircuit({"memory": 14, "processor": 4, "pair": 14}.get(route, 15 if route.startswith("current") else 27))


def _save(circuit, route, timestep):
    if route == "memory":
        for delay in range(1, 13):
            circuit.save_expectation_value(local_z_product(1), [delay], label=f"{timestep}:M:d{delay}")
    elif route == "processor":
        for degree, qubit in zip((1, 2, 3, 4), range(4)):
            circuit.save_expectation_value(local_z_product(1), [qubit], label=f"{timestep}:N:p{degree}")
    elif route.startswith("current"):
        degree = int(route[-1])
        for delay in range(1, 13):
            name = f"J:current_p{degree}_x_linear:d{delay}"
            circuit.save_expectation_value(local_z_product(2), [TAPS + 1, delay], label=f"{timestep}:{name}")
    elif route.startswith("delayed"):
        degree = int(route[-1])
        for delay in range(1, 13):
            nonlinear = TAPS + delay
            name = f"J:delay_p{degree}:d{delay}"
            circuit.save_expectation_value(local_z_product(1), [nonlinear], label=f"{timestep}:{name}")
            name = f"J:past_p{degree}_x_linear:d{delay}"
            circuit.save_expectation_value(local_z_product(2), [nonlinear, delay], label=f"{timestep}:{name}")
    else:
        for left in range(1, 13):
            for right in range(left + 1, 13):
                name = f"J:linear_pair:d{left}:d{right}"
                circuit.save_expectation_value(local_z_product(2), [left, right], label=f"{timestep}:{name}")


def _extract_exact(result, length):
    rows = [dict() for _ in range(length)]
    for route_index, route in enumerate(ROUTES):
        data = result.data(route_index)
        for timestep in range(length):
            prefix = f"{timestep}:"
            rows[timestep].update({key[len(prefix):]: float(value) for key, value in data.items() if key.startswith(prefix)})
    return tuple(rows)


def _extract_counts(route, counts):
    row = {}
    if route == "memory":
        for delay in range(1, 13):
            row[f"M:d{delay}"] = expectation_from_counts(counts, (delay,))
    elif route == "processor":
        for degree, qubit in zip((1, 2, 3, 4), range(4)):
            row[f"N:p{degree}"] = expectation_from_counts(counts, (qubit,))
    elif route.startswith("current"):
        degree = int(route[-1])
        for delay in range(1, 13):
            row[f"J:current_p{degree}_x_linear:d{delay}"] = expectation_from_counts(counts, (TAPS + 1, delay))
    elif route.startswith("delayed"):
        degree = int(route[-1])
        for delay in range(1, 13):
            row[f"J:delay_p{degree}:d{delay}"] = expectation_from_counts(counts, (TAPS + delay,))
            row[f"J:past_p{degree}_x_linear:d{delay}"] = expectation_from_counts(counts, (TAPS + delay, delay))
    else:
        for left in range(1, 13):
            for right in range(left + 1, 13):
                row[f"J:linear_pair:d{left}:d{right}"] = expectation_from_counts(counts, (left, right))
    return row


def run_fifo(inputs, m, g, *, mode="exact", shots=1000, seed_simulator=0, precision="double"):
    values = tuple(float(value) for value in inputs)
    if mode not in {"exact", "shots"}:
        raise ValueError("mode must be exact or shots")
    backend = AerSimulator(method="matrix_product_state", precision=precision)
    started, depths = perf_counter(), []
    if mode == "exact":
        circuits = []
        for route in ROUTES:
            circuit = _new_circuit(route)
            for timestep, value in enumerate(values):
                _append(circuit, route, value, m, g)
                depths.append(circuit.depth())
                _save(circuit, route, timestep)
            circuits.append(circuit)
        result = backend.run(circuits, shots=None, seed_simulator=seed_simulator).result()
        if not result.success:
            raise RuntimeError(result.status)
        rows = _extract_exact(result, len(values))
        executions, preparations, encodings, total_shots = len(ROUTES), len(ROUTES), len(values) * 20, 0
    else:
        circuits, route_order = [], []
        for route in ROUTES:
            for stop in range(1, len(values) + 1):
                circuit = _new_circuit(route)
                for value in values[:stop]:
                    _append(circuit, route, value, m, g)
                circuit.measure_all()
                circuits.append(circuit)
                route_order.append(route)
        depths = [circuit.depth() for circuit in circuits]
        result = backend.run(circuits, shots=shots, seed_simulator=seed_simulator).result()
        if not result.success:
            raise RuntimeError(result.status)
        rows = [dict() for _ in values]
        for index, route in enumerate(route_order):
            rows[index % len(values)].update(_extract_counts(route, result.get_counts(index)))
        rows = tuple(rows)
        executions = preparations = len(circuits)
        encodings = sum(range(1, len(values) + 1)) * 20
        total_shots = executions * shots
    resources = Resources(len(ROUTES), {route: _new_circuit(route).num_qubits for route in ROUTES}, 27, len(ROUTES), 1, 1, executions, preparations, encodings, 0 if mode == "exact" else shots, total_shots, tuple(depths), perf_counter() - started)
    return {"features": rows, "resources": asdict(resources), "backend": {"name": "AerSimulator", "method": "matrix_product_state", "precision": precision}}
