"""Small Qiskit circuits for a generic parallel polynomial delay bank."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .fifo import transport_depth


@dataclass(frozen=True)
class Observable:
    name: str
    values: tuple[float, ...]


def _theta(value: float) -> float:
    if not -1.0 <= value <= 1.0:
        raise ValueError("input outside [-1,1]")
    return acos(value)


def _available(m: float, delay: int) -> bool:
    return delay < transport_depth(m)


def specifications(inputs, m: float, g: float) -> tuple[Observable, ...]:
    """Return gate angles only; no target or feature value is computed here."""
    values = tuple(float(value) for value in inputs)
    specs = []
    for delay in range(1, 13):
        specs.append(Observable(f"M:d{delay}", tuple(
            _theta(values[t - delay]) if t >= delay and _available(m, delay) else pi / 2
            for t in range(len(values))
        )))
    for degree in (1, 2, 3, 4):
        specs.append(Observable(f"N:p{degree}", tuple(
            (1 + (degree - 1) * g) * _theta(value) for value in values
        )))
    for degree in (1, 2, 3, 4):
        for delay in range(1, 13):
            specs.append(Observable(f"J:current_p{degree}_x_linear:d{delay}", tuple(
                angle
                for t in range(len(values))
                for angle in (
                    (1 + (degree - 1) * g) * _theta(values[t]),
                    _theta(values[t - delay]) if t >= delay and _available(m, delay) else pi / 2,
                )
            )))
    for degree in (2, 3, 4):
        for delay in range(1, 13):
            delayed = tuple(
                (1 + (degree - 1) * g) * _theta(values[t - delay])
                if t >= delay and _available(m, delay) else pi / 2
                for t in range(len(values))
            )
            specs.append(Observable(f"J:delay_p{degree}:d{delay}", delayed))
            specs.append(Observable(f"J:past_p{degree}_x_linear:d{delay}", tuple(
                angle
                for t, nonlinear in enumerate(delayed)
                for angle in (
                    nonlinear,
                    _theta(values[t - delay]) if t >= delay and _available(m, delay) else pi / 2,
                )
            )))
    for left in range(1, 13):
        for right in range(left + 1, 13):
            specs.append(Observable(f"J:linear_pair:d{left}:d{right}", tuple(
                angle
                for t in range(len(values))
                for delay in (left, right)
                for angle in ((1 - g) * pi / 2 + g * _theta(values[t - delay])
                              if t >= delay and _available(m, delay) else pi / 2,)
            )))
    return tuple(specs)


def build_observable_circuit(specification: Observable) -> QuantumCircuit:
    paired = specification.name.startswith("J:") and not specification.name.startswith("J:delay_p")
    width = 2 if paired else 1
    if len(specification.values) % width:
        raise ValueError("invalid angle stream")
    circuit = QuantumCircuit(width)
    for timestep in range(len(specification.values) // width):
        for qubit in range(width):
            circuit.reset(qubit)
            circuit.ry(specification.values[timestep * width + qubit], qubit)
        circuit.save_expectation_value(
            SparsePauliOp("Z" * width), range(width), label=str(timestep)
        )
    circuit.metadata = {"family": "parallel_polynomial_delay", "feature": specification.name}
    return circuit


def observable_width(specification: Observable) -> int:
    return 2 if specification.name.startswith("J:") and not specification.name.startswith("J:delay_p") else 1


def build_observable_batches(specs, maximum_qubits: int = 8):
    """Pack independent feature routes without changing their quantum states."""
    batches, current, used = [], [], 0
    for spec in specs:
        width = observable_width(spec)
        if current and used + width > maximum_qubits:
            batches.append(tuple(current))
            current, used = [], 0
        current.append((spec, used, width))
        used += width
    if current:
        batches.append(tuple(current))

    circuits = []
    for batch in batches:
        circuit = QuantumCircuit(sum(item[2] for item in batch))
        length = len(batch[0][0].values) // batch[0][2]
        for timestep in range(length):
            for spec, offset, width in batch:
                for local in range(width):
                    circuit.reset(offset + local)
                    circuit.ry(spec.values[timestep * width + local], offset + local)
            circuit.save_probabilities(range(circuit.num_qubits), label=str(timestep))
        circuit.metadata = {
            "family": "parallel_polynomial_delay",
            "features": [item[0].name for item in batch],
        }
        circuits.append(circuit)
    return tuple(circuits), tuple(batches)


def build_serial_observable_circuit(specs):
    """Serialize independent routes with resets to avoid Aer experiment overhead."""
    circuit = QuantumCircuit(2)
    for spec in specs:
        width = observable_width(spec)
        for timestep in range(len(spec.values) // width):
            for qubit in range(width):
                circuit.reset(qubit)
                circuit.ry(spec.values[timestep * width + qubit], qubit)
            circuit.save_expectation_value(
                SparsePauliOp("Z" * width), range(width), label=f"{timestep}:{spec.name}"
            )
    circuit.metadata = {
        "family": "parallel_polynomial_delay",
        "dynamical_settings": len(specs),
    }
    return circuit
