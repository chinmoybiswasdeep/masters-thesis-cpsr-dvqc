"""Aggregate persisted V8 execution resources without rerunning circuits."""

from __future__ import annotations

import json
from pathlib import Path


def aggregate_resources(paths):
    totals = {
        "artifacts": 0, "aer_jobs": 0, "physical_circuit_executions": 0,
        "state_preparations": 0, "input_encodings": 0, "total_shots": 0,
        "wall_seconds": 0.0, "peak_logical_qubits": 0, "feature_count": 0,
        "maximum_replay_or_transpiled_depth": 0, "logical_swap_count": 0,
        "one_qubit_gates": 0, "two_qubit_gates": 0,
    }
    for path in paths:
        payload = json.loads(Path(path).read_text())
        resource = payload.get("resources")
        if not resource:
            continue
        totals["artifacts"] += 1
        totals["aer_jobs"] += int(resource.get("aer_jobs", 0))
        totals["physical_circuit_executions"] += int(
            resource.get("physical_circuit_executions", resource.get("circuit_executions", 0))
        )
        for key in ("state_preparations", "input_encodings", "total_shots"):
            totals[key] += int(resource.get(key, 0))
        totals["wall_seconds"] += float(resource.get("wall_seconds", 0.0))
        totals["peak_logical_qubits"] = max(
            totals["peak_logical_qubits"], int(resource.get("peak_logical_qubits", resource.get("peak_qubits", 0)))
        )
        totals["feature_count"] = max(totals["feature_count"], int(resource.get("feature_count", 0)))
        depths = resource.get("replay_depths", resource.get("transpiled_depths", ()))
        totals["maximum_replay_or_transpiled_depth"] = max(
            totals["maximum_replay_or_transpiled_depth"], max(depths, default=0)
        )
        totals["logical_swap_count"] += int(
            resource.get("logical_swap_count_before_basis_translation", resource.get("swap_count", 0))
        )
        totals["one_qubit_gates"] += int(resource.get("one_qubit_gates", resource.get("gate_counts", {}).get("ry", 0)))
        totals["two_qubit_gates"] += int(resource.get("two_qubit_gates", resource.get("swap_count", 0)))
    totals["estimated_qpu_burden"] = {
        "circuit_submissions": totals["physical_circuit_executions"],
        "shots": totals["total_shots"],
        "note": "Simulator execution only; this is the equivalent direct QPU sampling burden before queue/calibration overhead.",
    }
    return totals


def raw_artifacts(root):
    metric_suffix = "_metrics.json"
    return sorted(
        path for path in Path(root).rglob("*.json")
        if not path.name.endswith(metric_suffix)
        and path.name not in {"development_evidence.json", "development_gates.json", "validation_evidence.json", "validation_gates.json"}
    )
