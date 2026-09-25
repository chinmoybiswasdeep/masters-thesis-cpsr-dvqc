"""Generic Pauli observables and joint-bitstring extraction for V8."""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp


MEMORY_FEATURES = tuple(f"M:d{delay}" for delay in range(1, 13))
NONLINEAR_FEATURES = tuple(f"N:p{degree}" for degree in (1, 2, 3, 4))
JOINT_FEATURES = tuple(
    [f"J:current_p{degree}_x_linear:d{delay}" for degree in (1, 2, 3, 4) for delay in range(1, 13)]
    + [f"J:delay_p{degree}:d{delay}" for degree in (2, 3, 4) for delay in range(1, 13)]
    + [f"J:past_p{degree}_x_linear:d{delay}" for degree in (2, 3, 4) for delay in range(1, 13)]
    + [f"J:linear_pair:d{left}:d{right}" for left in range(1, 13) for right in range(left + 1, 13)]
)


def z_pauli(n_qubits: int, *qubits: int) -> SparsePauliOp:
    label = ["I"] * n_qubits
    for qubit in qubits:
        label[n_qubits - 1 - qubit] = "Z"
    return SparsePauliOp("".join(label))


def local_z_product(size: int) -> SparsePauliOp:
    """Return a local Z product for an equally sized ``qargs`` list."""
    if size < 1:
        raise ValueError("a Pauli product needs at least one qubit")
    return SparsePauliOp("Z" * size)


def parity(bits: str, qubits) -> int:
    value = int(bits.replace(" ", ""), 2)
    return -1 if sum((value >> qubit) & 1 for qubit in qubits) % 2 else 1


def expectation_from_counts(counts, qubits) -> float:
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("empty Aer counts")
    return sum(parity(bits, qubits) * count for bits, count in counts.items()) / shots


def expectation_from_probabilities(probabilities, bits) -> float:
    """Extract a Z parity from an Aer probability snapshot."""
    return float(sum(
        (-1 if sum((basis >> bit) & 1 for bit in bits) % 2 else 1) * probability
        for basis, probability in enumerate(probabilities)
    ))
