"""Aer-only modular execution for V8 FIFO reservoirs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from time import perf_counter

from qiskit import QuantumCircuit, qpy
from qiskit_aer import AerSimulator

from .v8_architectures.fifo import _angle, _write_fifo, transport_depth
from .v8_measurements import PAIR_DELAYS, expectation_from_counts, local_z_product


TAPS = 13
PREFIX_BATCH_SIZE = 32
ROUTES = (
    "memory", "processor", "current1", "current2", "current3", "current4",
    "delayed2", "delayed3", "delayed4", "pair", "mix12", "mix23", "mix32",
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
    gate_counts: dict
    one_qubit_gates: int
    two_qubit_gates: int
    swap_count: int
    measurement_basis_labels: tuple[str, ...]
    feature_count: int
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
        _write_fifo(circuit, range(TAPS), TAPS, (1.0 + (degree - 1) * g) * theta, depth)
    elif route.startswith("mix"):
        current_degree, delayed_degree = int(route[-2]), int(route[-1])
        _write_fifo(
            circuit, range(TAPS), TAPS,
            (1.0 + (delayed_degree - 1) * g) * theta, depth,
        )
        circuit.reset(TAPS + 1)
        circuit.ry((1.0 + (current_degree - 1) * g) * theta, TAPS + 1)
    else:
        _write_fifo(circuit, range(TAPS), TAPS, g * theta, depth)


def _new_circuit(route):
    return QuantumCircuit(
        {"memory": 14, "processor": 4, "pair": 14}.get(
            route, 15 if route.startswith(("current", "mix")) else 14
        ),
        name=f"v8_{route}",
    )


def circuit_fingerprint(route, inputs, m, g, *, measure=False):
    circuit = _new_circuit(route)
    for value in inputs:
        _append(circuit, route, float(value), m, g)
    if measure:
        circuit.measure_all()
    payload = BytesIO()
    qpy.dump(circuit, payload)
    return sha256(payload.getvalue()).hexdigest()


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
            name = f"J:delay_p{degree}:d{delay}"
            circuit.save_expectation_value(local_z_product(1), [delay], label=f"{timestep}:{name}")
    elif route.startswith("mix"):
        current_degree, delayed_degree = int(route[-2]), int(route[-1])
        for delay in range(1, 13):
            name = f"J:mix_p{current_degree}_p{delayed_degree}:d{delay}"
            circuit.save_expectation_value(
                local_z_product(2), [TAPS + 1, delay], label=f"{timestep}:{name}"
            )
    else:
        for left, right in PAIR_DELAYS:
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


def _run_exact_route(route, values, m, g, seed_simulator, precision):
    circuit = _new_circuit(route)
    depths = []
    for timestep, value in enumerate(values):
        _append(circuit, route, value, m, g)
        depths.append(circuit.depth())
        _save(circuit, route, timestep)
    result = AerSimulator(method="matrix_product_state", precision=precision).run(
        circuit, shots=None, seed_simulator=seed_simulator
    ).result()
    if not result.success:
        raise RuntimeError(result.status)
    rows = [dict() for _ in values]
    data = result.data(0)
    for timestep in range(len(values)):
        prefix = f"{timestep}:"
        rows[timestep].update({key[len(prefix):]: float(value) for key, value in data.items() if key.startswith(prefix)})
    return route, rows, depths, {name: int(count) for name, count in circuit.count_ops().items()}


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
            row[f"J:delay_p{degree}:d{delay}"] = expectation_from_counts(counts, (delay,))
    elif route.startswith("mix"):
        current_degree, delayed_degree = int(route[-2]), int(route[-1])
        for delay in range(1, 13):
            row[f"J:mix_p{current_degree}_p{delayed_degree}:d{delay}"] = expectation_from_counts(
                counts, (TAPS + 1, delay)
            )
    else:
        for left, right in PAIR_DELAYS:
            row[f"J:linear_pair:d{left}:d{right}"] = expectation_from_counts(counts, (left, right))
    return row


def run_fifo(inputs, m, g, *, mode="exact", shots=1000, seed_simulator=0, precision="double"):
    values = tuple(float(value) for value in inputs)
    if mode not in {"exact", "shots"}:
        raise ValueError("mode must be exact or shots")
    started, depths, gate_counts = perf_counter(), [], {}
    if mode == "exact":
        route_results = [
            _run_exact_route(route, values, m, g, seed_simulator, precision) for route in ROUTES
        ]
        rows = [dict() for _ in values]
        for _, route_rows, route_depths, route_counts in route_results:
            for timestep, row in enumerate(route_rows):
                rows[timestep].update(row)
            depths.extend(route_depths)
            for gate, count in route_counts.items():
                gate_counts[gate] = gate_counts.get(gate, 0) + count
        rows = tuple(rows)
        executions, preparations, encodings, total_shots = len(ROUTES), len(ROUTES), len(values) * 23, 0
        aer_jobs = len(ROUTES)
    else:
        backend = AerSimulator(method="matrix_product_state", precision=precision)
        rows = [dict() for _ in values]
        aer_jobs = 0
        for route in ROUTES:
            for first in range(0, len(values), PREFIX_BATCH_SIZE):
                stops = range(first + 1, min(first + PREFIX_BATCH_SIZE, len(values)) + 1)
                circuits = []
                for stop in stops:
                    circuit = _new_circuit(route)
                    for value in values[:stop]:
                        _append(circuit, route, value, m, g)
                    circuit.measure_all()
                    circuits.append(circuit)
                depths.extend(circuit.depth() for circuit in circuits)
                for circuit in circuits:
                    for gate, count in circuit.count_ops().items():
                        gate_counts[gate] = gate_counts.get(gate, 0) + int(count)
                result = backend.run(circuits, shots=shots, seed_simulator=seed_simulator).result()
                aer_jobs += 1
                if not result.success:
                    raise RuntimeError(result.status)
                for index, stop in enumerate(stops):
                    rows[stop - 1].update(_extract_counts(route, result.get_counts(index)))
        rows = tuple(rows)
        executions = preparations = len(ROUTES) * len(values)
        encodings = sum(range(1, len(values) + 1)) * 23
        total_shots = executions * shots
    resources = Resources(
        distinct_reservoir_circuits=len(ROUTES),
        logical_qubits_by_circuit={route: _new_circuit(route).num_qubits for route in ROUTES},
        peak_qubits=15,
        dynamical_settings=len(ROUTES),
        measurement_bases=1,
        aer_jobs=aer_jobs,
        circuit_executions=executions,
        state_preparations=preparations,
        input_encodings=encodings,
        shots_per_setting=0 if mode == "exact" else shots,
        total_shots=total_shots,
        gate_counts=gate_counts,
        one_qubit_gates=gate_counts.get("ry", 0),
        two_qubit_gates=gate_counts.get("swap", 0) + gate_counts.get("cx", 0),
        swap_count=gate_counts.get("swap", 0),
        measurement_basis_labels=("Z",),
        feature_count=len(rows[0]) if rows else 0,
        replay_depths=tuple(depths),
        wall_seconds=perf_counter() - started,
    )
    return {"features": rows, "resources": asdict(resources), "backend": {"name": "AerSimulator", "method": "matrix_product_state", "precision": precision}}
