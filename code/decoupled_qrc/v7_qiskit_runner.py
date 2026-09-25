"""Only supported V7 simulator path: real circuits executed by Qiskit Aer."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .v7_measurements import BASE_OBSERVABLES, combine_settings, pauli_for, setting_from_counts
from .v7_qiskit_architecture import LAYOUT, append_timestep, build_temporal_circuit, initialize_reservoir


@dataclass(frozen=True)
class RunResources:
    mode: str
    timesteps: int
    circuit_executions: int
    measurement_settings: int
    shots_per_setting: int
    total_shots: int
    input_encodings: int
    state_preparations: int
    replay_depths: tuple[int, ...]
    wall_seconds: float


@dataclass(frozen=True)
class FeatureRun:
    rows: tuple[dict[str, float], ...]
    resources: RunResources
    circuit_fingerprints: tuple[str, ...]


def _fingerprint(circuit) -> str:
    from io import BytesIO
    from hashlib import sha256
    from qiskit import qpy

    stream = BytesIO()
    qpy.dump(circuit, stream)
    return sha256(stream.getvalue()).hexdigest()


SETTINGS = (
    (1, 1, 0, True), (1, 1, 1, True), (1, 1, 2, True), (1, 1, 3, True),
    (1, 1, 0, False), (2, 2, 0, False), (3, 3, 0, False),
)


def _shot_circuit(inputs, m, g, nonlinear_order, processor_order, memory_bank, interaction, controls):
    circuit = build_temporal_circuit(
        inputs, m, g, nonlinear_order=nonlinear_order, processor_order=processor_order,
        memory_bank=memory_bank, interaction=interaction, **controls
    )
    circuit.measure_all()
    return circuit


def run_sequence(
    inputs: Iterable[float],
    m: float,
    g: float,
    *,
    mode: str = "exact",
    shots: int = 1000,
    seed_simulator: int = 0,
    precision: str = "double",
    backend=None,
    seed_transpiler: int = 0,
    **controls: bool,
) -> FeatureRun:
    """Execute every prefix by physical replay; there is no fallback path."""
    values = tuple(float(value) for value in inputs)
    if mode not in {"exact", "shots"}:
        raise ValueError("mode must be 'exact' or 'shots'")
    if mode == "shots" and shots <= 0:
        raise ValueError("shots must be positive")

    simulator = AerSimulator(
        method="density_matrix" if mode == "exact" else "matrix_product_state",
        precision=precision,
        noise_model=None if backend is None else __import__("qiskit_aer.noise", fromlist=["NoiseModel"]).NoiseModel.from_backend(backend),
    )
    rows, depths, fingerprints = [], [], []
    controls = dict(controls)
    interaction_enabled = controls.pop("interaction", True)
    started = perf_counter()
    if mode == "exact":
        # Save operations are non-destructive. Four settings provide three
        # re-upload orders and one interacting probe using persistent circuits.
        circuits = []
        setting_depths = []
        for nonlinear_order, processor_order, memory_bank, interacting in SETTINGS:
            circuit = QuantumCircuit(LAYOUT.n_qubits)
            initialize_reservoir(circuit)
            local_depths = []
            for timestep, value in enumerate(values):
                append_timestep(
                    circuit, value, m, g, nonlinear_order=nonlinear_order,
                    processor_order=processor_order, memory_bank=memory_bank,
                    interaction=interacting and interaction_enabled, **controls
                )
                local_depths.append(circuit.depth())
                for name, qubits in BASE_OBSERVABLES.items():
                    circuit.save_expectation_value(
                        pauli_for(qubits), range(LAYOUT.n_qubits), label=f"{timestep}:{name}"
                    )
            circuits.append(circuit)
            setting_depths.append(local_depths)
        fingerprints = [_fingerprint(circuit) for circuit in circuits]
        run_circuits = circuits if backend is None else transpile(
            circuits, backend=backend, seed_transpiler=seed_transpiler, optimization_level=1
        )
        result = simulator.run(run_circuits, shots=None, seed_simulator=seed_simulator).result()
        if not result.success:
            raise RuntimeError(result.status)
        settings = []
        for setting_index in range(len(SETTINGS)):
            data = result.data(setting_index)
            settings.append([
                {name: float(data[f"{timestep}:{name}"]) for name in BASE_OBSERVABLES}
                for timestep in range(len(values))
            ])
        rows = [combine_settings(*(setting[timestep] for setting in settings)) for timestep in range(len(values))]
        depths = [depth for timestep in range(len(values)) for depth in (setting[timestep] for setting in setting_depths)]
    else:
        # Measurements are destructive, so each measured timestep is a replay
        # of its complete history. Circuits share one Aer job, not one state.
        circuits = [
            _shot_circuit(
                values[:stop], m, g, nonlinear_order, processor_order, memory_bank,
                interacting and interaction_enabled, controls
            )
            for nonlinear_order, processor_order, memory_bank, interacting in SETTINGS
            for stop in range(1, len(values) + 1)
        ]
        fingerprints = [_fingerprint(circuit) for circuit in circuits]
        run_circuits = circuits if backend is None else transpile(
            circuits, backend=backend, seed_transpiler=seed_transpiler, optimization_level=1
        )
        depths = [circuit.depth() for circuit in run_circuits]
        result = simulator.run(run_circuits, shots=shots, seed_simulator=seed_simulator).result()
        if not result.success:
            raise RuntimeError(result.status)
        settings = [
            [setting_from_counts(result.get_counts(setting_index * len(values) + timestep)) for timestep in range(len(values))]
            for setting_index in range(len(SETTINGS))
        ]
        rows = [combine_settings(*(setting[timestep] for setting in settings)) for timestep in range(len(values))]
    elapsed = perf_counter() - started
    executions = len(SETTINGS) if mode == "exact" else len(values) * len(SETTINGS)
    return FeatureRun(
        tuple(rows),
        RunResources(
            mode=mode,
            timesteps=len(values),
            circuit_executions=executions,
            measurement_settings=len(SETTINGS),
            shots_per_setting=0 if mode == "exact" else shots,
            total_shots=0 if mode == "exact" else executions * shots,
            input_encodings=(len(values) * len(SETTINGS) if mode == "exact" else sum(range(1, len(values) + 1)) * len(SETTINGS)) * 7,
            state_preparations=executions,
            replay_depths=tuple(depths),
            wall_seconds=elapsed,
        ),
        tuple(fingerprints),
    )
