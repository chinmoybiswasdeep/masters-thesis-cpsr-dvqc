"""V7 circuit construction. This module intentionally has no NumPy path."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi, sqrt

from qiskit import QuantumCircuit
from qiskit_aer.noise import phase_damping_error


@dataclass(frozen=True)
class V7Layout:
    memory: tuple[int, ...] = (0, 1, 2)
    nonlinear_memory: tuple[int, ...] = (3,)
    processor: tuple[int, ...] = (4,)
    combined_probe: int = 5
    memory_input: int = 6
    nonlinear_input: int = 6

    @property
    def n_qubits(self) -> int:
        return 7

    @property
    def feature_qubits(self) -> tuple[int, ...]:
        return self.memory + self.nonlinear_memory + self.processor + (self.combined_probe,)


LAYOUT = V7Layout()
FULL_DEPHASING = phase_damping_error(1.0).to_instruction()
RETENTION_BANKS = (
    (-0.97, -0.88, -0.75),
    (-0.58, -0.38, -0.13),
    (0.13, 0.38, 0.58),
    (0.75, 0.88, 0.97),
)


def _theta(value: float) -> float:
    if not -1.0 <= value <= 1.0:
        raise ValueError("inputs must lie in [-1, 1]")
    return acos(value)


def _partial_iswap(circuit: QuantumCircuit, source: int, target: int, angle: float) -> None:
    """Exchange excitation through a genuine two-qubit XY interaction."""
    circuit.rxx(angle, source, target)
    circuit.ryy(angle, source, target)


def _prepare(circuit: QuantumCircuit, qubit: int, angle: float) -> None:
    circuit.reset(qubit)
    circuit.ry(angle, qubit)


def _prepare_diagonal(circuit: QuantumCircuit, qubit: int, angle: float) -> None:
    _prepare(circuit, qubit, angle)
    circuit.append(FULL_DEPHASING, (qubit,))


def _controlled_ry(circuit: QuantumCircuit, angle: float, control: int, target: int) -> None:
    """Native-basis decomposition avoids per-prefix transpilation overhead."""
    circuit.ry(angle / 2.0, target)
    circuit.cx(control, target)
    circuit.ry(-angle / 2.0, target)
    circuit.cx(control, target)


def initialize_reservoir(circuit: QuantumCircuit) -> None:
    """Start persistent channels at I/2, eliminating a long |0> transient."""
    for qubit in LAYOUT.memory + LAYOUT.nonlinear_memory:
        circuit.h(qubit)
        circuit.append(FULL_DEPHASING, (qubit,))


def append_timestep(
    circuit: QuantumCircuit,
    value: float,
    m: float,
    g: float,
    *,
    interaction: bool = True,
    reset_memory: bool = False,
    disconnect_m: bool = False,
    disconnect_g: bool = False,
    nonlinear_order: int = 1,
    processor_order: int | None = None,
    memory_bank: int = 0,
) -> None:
    """Append one physical reservoir update; measurements are never appended here."""
    if not 0.0 <= m <= 1.0 or not 0.0 <= g <= 1.0:
        raise ValueError("m and g must lie in [0, 1]")
    if nonlinear_order not in (1, 2, 3):
        raise ValueError("nonlinear_order must be 1, 2, or 3")
    if processor_order is None:
        processor_order = nonlinear_order
    if processor_order not in (1, 2, 3) or memory_bank not in range(len(RETENTION_BANKS)):
        raise ValueError("invalid processor order or memory bank")
    if reset_memory:
        for qubit in LAYOUT.memory + LAYOUT.nonlinear_memory:
            circuit.reset(qubit)

    theta = _theta(value)
    memory_control = 0.5 if disconnect_m else m
    nonlinear_control = 0.5 if disconnect_g else g

    # A dephased collision gives the exact channel
    # z' = retention*z + (1-retention)*u. Thus m changes timescale, not scale.
    for mode, max_retention in zip(LAYOUT.memory, RETENTION_BANKS[memory_bank]):
        _prepare_diagonal(circuit, LAYOUT.memory_input, theta)
        _partial_iswap(
            circuit,
            LAYOUT.memory_input,
            mode,
            acos(sqrt(abs(max_retention) * memory_control)),
        )
        if max_retention < 0.0:
            circuit.x(mode)

    # The second channel stores a re-uploaded input. At g=0 it is linear; at
    # g=1 its Z expectation is the second Chebyshev polynomial.
    for mode in LAYOUT.nonlinear_memory:
        nonlinear_theta = (1.0 + nonlinear_order * nonlinear_control) * theta
        _prepare_diagonal(circuit, LAYOUT.nonlinear_input, nonlinear_theta)
        _partial_iswap(
            circuit,
            LAYOUT.nonlinear_input,
            mode,
            acos(sqrt(0.97 * memory_control)),
        )

    # Current-input harmonics. Varying g changes their normalized span, not
    # merely their amplitude: at g=1 these are T2, T3, and T4.
    for qubit in LAYOUT.processor:
        _prepare(circuit, qubit, (1.0 + processor_order * nonlinear_control) * theta)

    _prepare(circuit, LAYOUT.combined_probe, 0.0)
    if interaction:
        # Memory is the control, so its Z population—and hence the registered
        # linear-memory curve—is invariant. The probe dynamics changes with g.
        _controlled_ry(circuit, pi * nonlinear_control, LAYOUT.memory[0], LAYOUT.combined_probe)
        _controlled_ry(circuit, 0.7 * pi * nonlinear_control, LAYOUT.memory[1], LAYOUT.combined_probe)
        _controlled_ry(circuit, 0.4 * pi * nonlinear_control, LAYOUT.memory[2], LAYOUT.combined_probe)


def build_temporal_circuit(
    inputs: list[float] | tuple[float, ...],
    m: float,
    g: float,
    nonlinear_order: int = 1,
    processor_order: int | None = None,
    memory_bank: int = 0,
    **controls: bool,
) -> QuantumCircuit:
    """Replay an input prefix from |0...0>; callers append save/measure ops."""
    circuit = QuantumCircuit(LAYOUT.n_qubits)
    initialize_reservoir(circuit)
    for value in inputs:
        append_timestep(
            circuit, float(value), m, g, nonlinear_order=nonlinear_order,
            processor_order=processor_order, memory_bank=memory_bank, **controls
        )
    circuit.metadata = {
        "v7": True,
        "timesteps": len(inputs),
        "m": m,
        "g": g,
        "nonlinear_order": nonlinear_order,
        "processor_order": processor_order,
        "memory_bank": memory_bank,
        "input_encodings": len(inputs) * 7,
        "state_preparations": 1,
    }
    return circuit
