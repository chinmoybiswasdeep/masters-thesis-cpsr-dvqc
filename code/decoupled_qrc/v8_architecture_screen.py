"""Execute early Qiskit-only architecture-family screens."""

from __future__ import annotations

from time import perf_counter

from qiskit_aer import AerSimulator

from .v8_architectures.screening import FAMILIES, build_screen_circuit


def run_architecture_screen(inputs, seed_simulator=8080):
    circuits = [
        build_screen_circuit(family, inputs, m, g)
        for family in FAMILIES
        for m, g in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    ]
    backend = AerSimulator(method="matrix_product_state")
    started = perf_counter()
    result = backend.run(circuits, shots=None, seed_simulator=seed_simulator).result()
    if not result.success:
        raise RuntimeError(result.status)
    rows = []
    for index, circuit in enumerate(circuits):
        data = result.data(index)
        rows.append({
            "family": circuit.metadata["family"],
            "m": circuit.metadata["m"],
            "g": circuit.metadata["g"],
            "memory": [float(data[f"{t}:memory"]) for t in range(len(inputs))],
            "nonlinear": [float(data[f"{t}:nonlinear"]) for t in range(len(inputs))],
            "joint": [float(data[f"{t}:joint"]) for t in range(len(inputs))],
        })
    return {
        "backend": {"name": "AerSimulator", "method": "matrix_product_state"},
        "families": list(FAMILIES),
        "circuits": rows,
        "resource": {
            "aer_jobs": 1,
            "physical_circuit_executions": len(circuits),
            "distinct_architecture_families": len(FAMILIES),
            "peak_qubits": max(circuit.num_qubits for circuit in circuits),
            "wall_seconds": perf_counter() - started,
        },
    }
