"""Small, genuinely distinct Qiskit architecture-family screening circuits."""

from __future__ import annotations

from math import acos

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import amplitude_damping_error


FAMILIES = (
    "quantum_fifo",
    "collision_multiscale",
    "ancilla_shift",
    "coherent_recurrent",
    "dissipative_kraus",
    "parallel_polynomial_delay",
    "interaction_network",
    "hybrid_modular",
)


def _full_z(*qubits):
    label = ["I"] * 7
    for qubit in qubits:
        label[6 - qubit] = "Z"
    return SparsePauliOp("".join(label))


def build_screen_circuit(family, inputs, m, g):
    if family not in FAMILIES:
        raise ValueError(f"unknown architecture family {family}")
    circuit = QuantumCircuit(7)
    damping = amplitude_damping_error(0.35 * (1 - m)).to_instruction()
    for timestep, value in enumerate(inputs):
        theta = acos(float(value))
        if family in {"quantum_fifo", "ancilla_shift", "hybrid_modular"}:
            for index in range(2, -1, -1):
                if m >= (index + 1) / 4:
                    circuit.swap(index, index + 1)
            circuit.reset(4)
            circuit.ry(theta, 4)
            if m > 0:
                circuit.swap(4, 0)
        elif family == "collision_multiscale":
            circuit.reset(4)
            circuit.ry(theta, 4)
            for qubit, scale in enumerate((1.0, 0.55, 0.25, 0.1)):
                circuit.rxx(m * scale, 4, qubit)
                circuit.ryy(m * scale, 4, qubit)
        elif family == "coherent_recurrent":
            circuit.reset(0)
            circuit.ry(theta, 0)
            for qubit in range(3):
                circuit.rx(0.37 * m, qubit + 1)
                circuit.rzz(0.61 * m, qubit, qubit + 1)
        elif family == "dissipative_kraus":
            circuit.reset(4)
            circuit.ry(theta, 4)
            circuit.rxx(0.8 * m, 4, 0)
            for qubit in range(4):
                circuit.append(damping, [qubit])
        elif family == "parallel_polynomial_delay":
            circuit.reset(3)
            past = float(inputs[timestep - 3]) if timestep >= 3 and m > 0 else 0.0
            circuit.ry(acos(past), 3)
        else:
            circuit.reset(4)
            circuit.ry(theta, 4)
            circuit.rxx(0.7 * m, 4, 0)
            circuit.rzx(0.9 * g, 0, 1)
        circuit.reset(5)
        circuit.ry((1 + g) * theta, 5)
        if family in {"interaction_network", "hybrid_modular"}:
            circuit.rzx(0.7 * g, 3, 5)
        circuit.save_expectation_value(_full_z(3), range(7), label=f"{timestep}:memory")
        circuit.save_expectation_value(_full_z(5), range(7), label=f"{timestep}:nonlinear")
        circuit.save_expectation_value(_full_z(3, 5), range(7), label=f"{timestep}:joint")
    circuit.metadata = {"family": family, "m": m, "g": g}
    return circuit
