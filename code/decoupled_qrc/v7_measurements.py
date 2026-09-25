"""V7 observable settings and extraction from shared Aer bitstrings."""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from .v7_qiskit_architecture import LAYOUT


BASE_OBSERVABLES = {
    "m0": (0,), "m1": (1,), "m2": (2,), "q": (3,), "p": (4,), "c": (5,),
    "m0m1": (0, 1), "m0m2": (0, 2), "m1m2": (1, 2),
    "m0p": (0, 4), "m1p": (1, 4), "m2p": (2, 4),
    "m0c": (0, 5), "m1c": (1, 5), "m2c": (2, 5),
}
MEMORY_FEATURES = tuple(f"b{bank}m{mode}" for bank in range(4) for mode in range(3))
PROCESSOR_FEATURES = ("p2", "p3", "p4")
INTERACTION_FEATURES = tuple(
    name
    for bank in range(4)
    for name in (f"b{bank}c", f"b{bank}m0c", f"b{bank}m1c", f"b{bank}m2c")
)
CURRENT_MEMORY_FEATURES = tuple(f"b{bank}m{mode}p2" for bank in range(4) for mode in range(3))
ALL_FEATURES = (
    MEMORY_FEATURES + ("q2", "q3", "q4", "p2", "p3", "p4")
    + CURRENT_MEMORY_FEATURES + INTERACTION_FEATURES
)


def pauli_for(qubits: tuple[int, ...]) -> SparsePauliOp:
    label = ["I"] * LAYOUT.n_qubits
    for qubit in qubits:
        label[LAYOUT.n_qubits - 1 - qubit] = "Z"
    return SparsePauliOp("".join(label))


def parity(bitstring: str, qubits: tuple[int, ...]) -> int:
    value = int(bitstring.replace(" ", ""), 2)
    return -1 if sum((value >> qubit) & 1 for qubit in qubits) % 2 else 1


def setting_from_counts(counts: dict[str, int]) -> dict[str, float]:
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("Aer returned no measurement shots")
    return {
        name: sum(parity(bits, qubits) * count for bits, count in counts.items()) / shots
        for name, qubits in BASE_OBSERVABLES.items()
    }


def combine_settings(*settings: dict) -> dict[str, float]:
    if len(settings) != 7:
        raise ValueError("V7.6 requires four interaction banks and three processor settings")
    banks, processors = settings[:4], settings[4:]
    row = {
        f"b{bank}m{mode}": data[f"m{mode}"]
        for bank, data in enumerate(banks)
        for mode in range(3)
    }
    for bank, data in enumerate(banks):
        row[f"b{bank}c"] = data["c"]
        for mode in range(3):
            row[f"b{bank}m{mode}p2"] = data[f"m{mode}p"]
            row[f"b{bank}m{mode}c"] = data[f"m{mode}c"]
    for order, data in enumerate(processors, start=2):
        row[f"q{order}"] = data["q"]
        row[f"p{order}"] = data["p"]
    return row
