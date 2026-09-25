"""Resumable V7 Qiskit-Aer experiment driver."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoupled_qrc.v7_measurements import ALL_FEATURES
from decoupled_qrc.v7_metrics import evaluate_features, geometry
from decoupled_qrc.v7_protocol import ROOT, atomic_json, load_protocol, load_seed_bank
from decoupled_qrc.v7_qiskit_runner import run_sequence


READOUTS = ("raw_ols", "standardized_ols", "ridge_cv", "whitened_ols")


def inputs_for(seed: int, length: int) -> list[float]:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, length).tolist()


def run_point(version: str, bank_name: str, seed: dict, m: float, g: float, mode: str, shots: int) -> dict:
    protocol = load_protocol()
    output = ROOT / "results" / "v7" / version / bank_name / mode
    key = f"i{seed['input']}_m{m:g}_g{g:g}_s{shots}.json"
    path = output / key
    if path.exists():
        return json.loads(path.read_text())
    values = inputs_for(seed["input"], protocol["data"]["sequence_length"])
    run = run_sequence(values, m, g, mode=mode, shots=shots, seed_simulator=seed["simulator"], seed_transpiler=seed["transpiler"])
    record = {
        "version": version,
        "bank": bank_name,
        "seed": seed,
        "m": m,
        "g": g,
        "mode": mode,
        "shots": 0 if mode == "exact" else shots,
        "inputs": values,
        "feature_names": list(ALL_FEATURES),
        "features": list(run.rows),
        "resources": run.resources.__dict__,
        "circuit_fingerprints": list(run.circuit_fingerprints),
        "metrics": {name: evaluate_features(run.rows, values, protocol, name) for name in READOUTS},
        "geometry": geometry(run.rows),
        "environment": {"python": platform.python_version(), "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__},
    }
    atomic_json(path, record)
    return record


def run_bank(version: str, bank_name: str, *, corners_only: bool = False, modes=("exact", "shots")) -> list[dict]:
    protocol = load_protocol()
    bank = load_seed_bank(bank_name)
    levels_m = [0.0, 1.0] if corners_only else protocol["data"]["m_levels"]
    levels_g = [0.0, 1.0] if corners_only else protocol["data"]["g_levels"]
    records = []
    for seed in bank["seeds"]:
        for mode in modes:
            shots = protocol["simulators"]["grid_shots"]
            for m in levels_m:
                for g in levels_g:
                    print(f"{version} {bank_name} input={seed['input']} {mode} m={m:g} g={g:g}", flush=True)
                    records.append(run_point(version, bank_name, seed, m, g, mode, shots))
    atomic_json(ROOT / "results" / "v7" / version / bank_name / "index.json", [
        {key: row[key] for key in ("version", "bank", "seed", "m", "g", "mode", "shots")} for row in records
    ])
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("SMOKE", "DEVELOP", "VALIDATE"))
    parser.add_argument("--version", default="V7.0")
    args = parser.parse_args()
    if args.stage == "SMOKE":
        bank = load_seed_bank("development")
        original = bank["seeds"]
        bank["seeds"] = original[:1]
        # Smoke is deliberately explicit so it never mutates the frozen bank.
        protocol = load_protocol()
        records = []
        for mode in ("exact", "shots"):
            for m in (0.0, 1.0):
                for g in (0.0, 1.0):
                    records.append(run_point(args.version, "smoke", original[0], m, g, mode, protocol["simulators"]["grid_shots"]))
        atomic_json(ROOT / "results" / "v7" / args.version / "smoke" / "summary.json", records)
    elif args.stage == "DEVELOP":
        run_bank(args.version, "development")
    else:
        run_bank(args.version, "internal_validation")


if __name__ == "__main__":
    main()
