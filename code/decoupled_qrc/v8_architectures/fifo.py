"""Generic 12-delay quantum FIFO with parallel polynomial registers."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class FIFOLayout:
    taps: int = 13

    @property
    def registers(self):
        return tuple(tuple(range(order * self.taps, (order + 1) * self.taps)) for order in range(4))

    @property
    def input_ancilla(self):
        return 4 * self.taps

    @property
    def processors(self):
        return (self.input_ancilla + 1, self.input_ancilla + 2, self.input_ancilla + 3)

    @property
    def n_qubits(self):
        return self.input_ancilla + 4


LAYOUT = FIFOLayout()


def transport_depth(m: float) -> int:
    mapping = {0.0: 0, 0.25: 3, 0.5: 6, 0.75: 10, 1.0: 13}
    try:
        return mapping[round(float(m), 2)]
    except KeyError as error:
        raise ValueError("FIFO m must be a preregistered grid level") from error


def _angle(value: float) -> float:
    if not -1.0 <= value <= 1.0:
        raise ValueError("input outside [-1,1]")
    return acos(value)


def _write_fifo(circuit: QuantumCircuit, register, ancilla: int, angle: float, depth: int):
    if depth == 0:
        circuit.reset(ancilla)
        circuit.ry(angle, ancilla)
        return
    for index in range(depth - 2, -1, -1):
        circuit.swap(register[index], register[index + 1])
    circuit.reset(ancilla)
    circuit.ry(angle, ancilla)
    circuit.swap(ancilla, register[0])


def append_fifo_timestep(circuit: QuantumCircuit, value: float, m: float, g: float):
    if not 0.0 <= g <= 1.0:
        raise ValueError("g outside [0,1]")
    theta, depth = _angle(value), transport_depth(m)
    for harmonic, register in enumerate(LAYOUT.registers, start=1):
        _write_fifo(circuit, register, LAYOUT.input_ancilla, (1.0 + (harmonic - 1) * g) * theta, depth)
    for harmonic, qubit in enumerate(LAYOUT.processors, start=2):
        circuit.reset(qubit)
        circuit.ry((1.0 + (harmonic - 1) * g) * theta, qubit)


def build_fifo_circuit(inputs, m: float, g: float):
    circuit = QuantumCircuit(LAYOUT.n_qubits)
    for value in inputs:
        append_fifo_timestep(circuit, float(value), m, g)
    circuit.metadata = {"family": "quantum_fifo", "timesteps": len(inputs), "m": m, "g": g}
    return circuit

