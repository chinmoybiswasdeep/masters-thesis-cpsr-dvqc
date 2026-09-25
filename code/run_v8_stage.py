"""Resumable command-line stages for DQRC V8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decoupled_qrc.v8_architecture_screen import run_architecture_screen
from decoupled_qrc.v8_baselines import evaluate_classical_baselines
from decoupled_qrc.v8_confirmation import reveal_confirmation
from decoupled_qrc.v8_controls import evaluate_negative_controls
from decoupled_qrc.v8_evidence import (
    analyze_conditioning_and_learning, analyze_response_surface, build_corner_evidence,
    analyze_noise_robustness, compare_precision,
)
from decoupled_qrc.v8_freeze import freeze_candidate, verify_manifest
from decoupled_qrc.v8_gates import evaluate_gates
from decoupled_qrc.v8_metrics import evaluate_point
from decoupled_qrc.v8_noise_runner import run_fifo_noisy
from decoupled_qrc.v8_parallel_runner import run_parallel_delay
from decoupled_qrc.v8_protocol import ROOT, RESULTS, atomic_json, load_open_bank, load_protocol
from decoupled_qrc.v8_qiskit_runner import run_fifo


STAGES = (
    "PREREGISTER", "ARCHITECTURE_SCREEN", "DEVELOP_EXACT", "DEVELOP_SHOTS",
    "NEGATIVE_CONTROLS", "ROBUSTNESS", "VALIDATE", "FREEZE", "CONFIRM",
    "REPORT", "VERIFY",
)

FINITE_SHOT_CORE_GATES = (
    "memory_main", "nonlinearity_main", "memory_to_nonlinearity_equivalence",
    "nonlinearity_to_memory_equivalence", "degree_profile_preservation",
    "memory_curve_preservation", "scale_invariant_separation", "nonlinear_geometry",
    "per_delay_requirements", "combined_hh", "heldout_targets", "seed_completeness",
    "target_completeness", "train_test_leakage",
)


def _seed(bank, index):
    if bank == "confirmation":
        revealed = RESULTS / "confirmation_seeds_revealed.json"
        if not revealed.exists():
            raise PermissionError("confirmation bank has not been revealed after freeze")
        verify_manifest(RESULTS / "frozen_manifest.json")
        return json.loads(revealed.read_text())["seeds"][index]
    return load_open_bank(bank)["seeds"][index]


def _inputs(seed, length):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, length).tolist()


def _point_name(seed, m, g, mode, shots=0, precision="double", simulator_seed_offset=0):
    suffix = f"_{shots}shots" if mode == "shots" else ""
    if precision != "double":
        suffix += f"_{precision}"
    if simulator_seed_offset:
        suffix += f"_simplus{simulator_seed_offset}"
    return f"seed_{seed}_m{m:g}_g{g:g}_{mode}{suffix}.json"


def execute_point(args):
    protocol = load_protocol()
    seed = _seed(args.bank, args.seed_index)
    inputs = _inputs(seed["input"], protocol["data"]["sequence_length"])
    if args.bank == "confirmation":
        output = RESULTS / ("confirmation_shots" if args.mode == "shots" else "confirmation_exact")
    elif args.bank == "internal_validation":
        output = RESULTS / ("validation_shots" if args.mode == "shots" else "validation")
    else:
        output = RESULTS / ("development_shots" if args.mode == "shots" else "development_exact")
    output.mkdir(parents=True, exist_ok=True)
    target = output / _point_name(
        seed["input"], args.m, args.g, args.mode, args.shots,
        args.precision, args.simulator_seed_offset,
    )
    if target.exists() and not args.force:
        print(f"checkpoint exists: {target}")
        return
    result = run_fifo(
        inputs, args.m, args.g, mode=args.mode, shots=args.shots,
        seed_simulator=seed["simulator"] + args.simulator_seed_offset, precision=args.precision,
    )
    atomic_json(target, {
        "candidate": "V8.6", "bank": args.bank, "seed": seed,
        "m": args.m, "g": args.g, "inputs": inputs, **result,
    })
    metrics = evaluate_point(inputs, result["features"], protocol)
    atomic_json(target.with_name(target.stem + "_metrics.json"), metrics)
    print(target)
    resource = {key: value for key, value in result["resources"].items() if key != "replay_depths"}
    resource["replay_depth_count"] = len(result["resources"]["replay_depths"])
    resource["maximum_replay_depth"] = max(result["resources"]["replay_depths"], default=0)
    print(json.dumps(resource, sort_keys=True))


def execute_noisy(args):
    protocol = load_protocol()
    seed = _seed("development", args.seed_index)
    inputs = _inputs(seed["input"], protocol["data"]["sequence_length"])
    output = RESULTS / "robustness" / "fake_guadalupe_v2"
    output.mkdir(parents=True, exist_ok=True)
    simulator_seed = seed["simulator"] + args.simulator_seed_offset
    transpiler_seed = seed["transpiler"] + args.transpiler_seed_offset
    target = output / (
        f"seed_{seed['input']}_m{args.m:g}_g{args.g:g}_sim{simulator_seed}_"
        f"trans{transpiler_seed}_{args.shots}shots.json"
    )
    if target.exists() and not args.force:
        print(f"checkpoint exists: {target}")
        return
    result = run_fifo_noisy(
        inputs, args.m, args.g, shots=args.shots,
        seed_simulator=simulator_seed, seed_transpiler=transpiler_seed,
    )
    atomic_json(target, {
        "candidate": "V8.6", "seed": seed, "m": args.m, "g": args.g,
        "inputs": inputs, **result,
    })
    atomic_json(target.with_name(target.stem + "_metrics.json"), evaluate_point(inputs, result["features"], protocol))
    print(target)


def execute_negative_controls(args):
    protocol = load_protocol()
    seed = _seed("development", args.seed_index)
    directory = RESULTS / "development_exact"
    points = {}
    inputs = None
    for label, (m, g) in {"LL": (0, 0), "HL": (1, 0), "LH": (0, 1), "HH": (1, 1)}.items():
        raw = json.loads((directory / _point_name(seed["input"], m, g, "exact")).read_text())
        inputs = raw["inputs"]
        points[label] = raw["features"]
    output = RESULTS / "negative_controls"
    permutation = np.random.default_rng(seed["input"] + 8801).permutation(len(inputs))
    permuted_path = output / f"seed_{seed['input']}_time_permuted_input.json"
    if permuted_path.exists() and not args.force:
        permuted = json.loads(permuted_path.read_text())
    else:
        result = run_fifo(
            np.asarray(inputs)[permutation].tolist(), 1.0, 1.0,
            seed_simulator=seed["simulator"], precision="double",
        )
        permuted = {
            "candidate": "V8.6", "seed": seed, "permutation": permutation.tolist(),
            "features": result["features"], "resources": result["resources"], "backend": result["backend"],
        }
        atomic_json(permuted_path, permuted)
    controls = evaluate_negative_controls(
        inputs, points, protocol, permuted_input_rows=permuted["features"], seed=seed["input"] + 8800,
    )
    atomic_json(output / f"seed_{seed['input']}_controls.json", controls)
    baselines = evaluate_classical_baselines(inputs, points["HH"], protocol, seed=seed["input"] + 8700)
    atomic_json(output / f"seed_{seed['input']}_baselines.json", baselines)
    print(json.dumps({"controls": controls, "baseline_complete": baselines["baseline_complete"]}, indent=2))


def finite_shot_matrix_status():
    protocol = load_protocol()
    directory = RESULTS / "development_shots"
    seed = _seed("development", 0)
    details = {}
    passed = True
    for shots in (1000, 10000):
        evidence = build_corner_evidence(directory, "development", mode="shots", shots=shots)
        gates = evaluate_gates(evidence, protocol)
        core = all(gates.get(name, {}).get("passed", False) for name in FINITE_SHOT_CORE_GATES)
        details[f"primary_{shots}"] = core
        passed &= core
        atomic_json(RESULTS / f"development_{shots}shots_evidence.json", evidence)
        atomic_json(RESULTS / f"development_{shots}shots_gates.json", gates)

    checks = ((1000, 100000), (10000, 100000), (100000, 0))
    for shots, simulator_offset in checks:
        points = {}
        for label, (m, g) in {"LL": (0, 0), "HL": (1, 0), "LH": (0, 1), "HH": (1, 1)}.items():
            path = directory / _point_name(
                seed["input"], m, g, "shots", shots,
                simulator_seed_offset=simulator_offset,
            )
            metric_path = path.with_name(path.stem + "_metrics.json")
            if not metric_path.exists():
                raise FileNotFoundError(metric_path)
            points[label] = json.loads(metric_path.read_text())
        ordered = True
        for family in protocol["task_families"][4:]:
            for readout in protocol["readouts"]:
                for delay in range(8, 12):
                    hh = points["HH"]["capacities"][readout][family][delay]
                    other = max(points[label]["capacities"][readout][family][delay] for label in ("LL", "HL", "LH"))
                    ordered &= hh > other
        label = f"representative_{shots}_simplus{simulator_offset}"
        details[label] = bool(ordered)
        passed &= ordered
    for shots in (1000, 10000):
        center = directory / _point_name(seed["input"], 0.5, 0.5, "shots", shots)
        details[f"center_{shots}"] = center.exists()
        passed &= center.exists()
    return {"finite_shot_pass": bool(passed), "details": details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--bank", choices=("development", "internal_validation", "confirmation"), default="development")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--mode", choices=("exact", "shots"), default="exact")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--simulator-seed-offset", type=int, choices=(0, 100000), default=0)
    parser.add_argument("--transpiler-seed-offset", type=int, choices=(0, 100000), default=0)
    parser.add_argument("--supplemental")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest", default=str(RESULTS / "frozen_manifest.json"))
    parser.add_argument("--key")
    args = parser.parse_args()

    if args.stage == "PREREGISTER":
        print(json.dumps(load_protocol(), indent=2, sort_keys=True))
    elif args.stage == "ARCHITECTURE_SCREEN":
        seed = _seed("development", args.seed_index)
        inputs = _inputs(seed["input"], 64)
        result = run_architecture_screen(inputs, seed["simulator"])
        atomic_json(RESULTS / "architecture_screen" / f"raw_screen_seed_{seed['input']}.json", {
            **result, "inputs": inputs, "input_seed": seed["input"], "simulator_seed": seed["simulator"],
        })
    elif args.stage in {"DEVELOP_EXACT", "DEVELOP_SHOTS", "VALIDATE"}:
        if args.stage == "DEVELOP_SHOTS":
            args.mode = "shots"
        if args.stage == "VALIDATE":
            args.bank = "internal_validation"
        execute_point(args)
    elif args.stage == "FREEZE":
        files = [
            "code/decoupled_qrc/v8_architectures/fifo.py",
            "code/decoupled_qrc/v8_qiskit_runner.py",
            "code/decoupled_qrc/v8_noise_runner.py",
            "code/decoupled_qrc/v8_baselines.py",
            "code/decoupled_qrc/v8_controls.py",
            "code/decoupled_qrc/v8_measurements.py",
            "code/decoupled_qrc/v8_metrics.py",
            "code/decoupled_qrc/v8_evidence.py",
            "code/decoupled_qrc/v8_statistics.py",
            "code/decoupled_qrc/v8_gates.py",
            "code/decoupled_qrc/v8_protocol.py",
            "code/decoupled_qrc/v8_freeze.py",
            "code/decoupled_qrc/v8_confirmation.py",
            "code/run_v8_stage.py",
            "code/run_v8_matrix.py",
            "results/v8/preregistered_v8_protocol.json",
            "results/v8/amendment_001.json",
            "results/v8/amendment_002.json",
            "results/v8/amendment_003.json",
            "results/v8/amendment_004.json",
            "results/v8/candidate_registry.json",
            "docs/V8_DEVELOPMENT_LOG.md",
            "requirements-lock.txt",
        ]
        files.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").glob("test_v8_*.py"))
        artifact_roots = (
            "architecture_screen", "development_exact", "development_shots",
            "validation", "validation_shots", "negative_controls", "robustness",
        )
        for directory in artifact_roots:
            files.extend(
                path.relative_to(ROOT).as_posix()
                for path in (RESULTS / directory).rglob("*") if path.is_file()
            )
        files.extend(
            path.relative_to(ROOT).as_posix()
            for path in RESULTS.glob("*_evidence.json")
        )
        files.extend(
            path.relative_to(ROOT).as_posix()
            for path in RESULTS.glob("*_gates.json")
        )
        gates = [RESULTS / "development_gates.json", RESULTS / "validation_gates.json"]
        for gate_file in gates:
            decision = json.loads(gate_file.read_text())
            if not decision.get("all_passed"):
                raise SystemExit(f"cannot freeze: failed or incomplete {gate_file}")
        print(freeze_candidate(
            "V8.6", files, "2ee6baf7bc5cb167810164dedd2b0635310da170",
            args.manifest, require_committed=True,
        ))
    elif args.stage == "CONFIRM":
        revealed = RESULTS / "confirmation_seeds_revealed.json"
        if not revealed.exists():
            if not args.key:
                raise SystemExit("--key is required for the one-time reveal")
            print(reveal_confirmation(args.manifest, args.key.encode()))
        args.bank = "confirmation"
        execute_point(args)
    elif args.stage == "VERIFY":
        print(verify_manifest(args.manifest))
    elif args.stage == "ROBUSTNESS":
        execute_noisy(args)
    elif args.stage == "NEGATIVE_CONTROLS":
        execute_negative_controls(args)
    elif args.stage == "REPORT":
        if args.bank == "confirmation":
            directory = RESULTS / ("confirmation_shots" if args.mode == "shots" else "confirmation_exact")
        elif args.bank == "internal_validation":
            directory = RESULTS / ("validation_shots" if args.mode == "shots" else "validation")
        else:
            directory = RESULTS / ("development_shots" if args.mode == "shots" else "development_exact")
        supplemental = json.loads(Path(args.supplemental).read_text()) if args.supplemental else {}
        if args.bank == "development" and args.mode == "exact" and args.precision == "double":
            control_files = [
                RESULTS / "negative_controls" / f"seed_{row['input']}_controls.json"
                for row in load_open_bank("development")["seeds"]
            ]
            if all(path.exists() for path in control_files):
                controls = [json.loads(path.read_text()) for path in control_files]
                supplemental["negative_controls"] = {
                    name: max(row[name] for row in controls) for name in controls[0]
                }
                supplemental["encoder_leakage_max"] = supplemental["negative_controls"]["future"]
            baseline_files = [
                RESULTS / "negative_controls" / f"seed_{row['input']}_baselines.json"
                for row in load_open_bank("development")["seeds"]
            ]
            if all(path.exists() for path in baseline_files):
                baselines = [json.loads(path.read_text()) for path in baseline_files]
                supplemental["baseline_complete"] = all(row["baseline_complete"] for row in baselines)
                supplemental["quantum_advantage"] = all(row["quantum_advantage"] for row in baselines)
            try:
                surface = analyze_response_surface(directory, _seed("development", 0)["input"])
            except FileNotFoundError:
                surface = None
            if surface:
                supplemental["saturated_fraction"] = surface["saturated_fraction"]
                supplemental["response_surface"] = surface
            hh = directory / _point_name(_seed("development", 0)["input"], 1, 1, "exact")
            if hh.exists():
                stability = analyze_conditioning_and_learning(hh)
                supplemental.update({
                    "conditioning_pass": stability["conditioning_pass"],
                    "learning_curve_pass": stability["learning_curve_pass"],
                    "stability": stability,
                })
            try:
                precision = compare_precision(directory, _seed("development", 0)["input"])
            except FileNotFoundError:
                precision = None
            if precision:
                supplemental["precision_pass"] = precision["precision_pass"]
                supplemental["precision"] = precision
            try:
                noise = analyze_noise_robustness(
                    RESULTS / "robustness" / "fake_guadalupe_v2", directory,
                    _seed("development", 0), shots=1000,
                )
            except FileNotFoundError:
                noise = None
            if noise:
                supplemental.update({
                    "noise_effect_retention": noise["noise_effect_retention"],
                    "noise_hh_ordered": noise["noise_hh_ordered"],
                    "noise": noise,
                })
            try:
                finite_shots = finite_shot_matrix_status()
            except (FileNotFoundError, ValueError):
                finite_shots = None
            if finite_shots:
                supplemental["finite_shot_pass"] = finite_shots["finite_shot_pass"]
                supplemental["finite_shots"] = finite_shots
        evidence = build_corner_evidence(
            directory, args.bank, supplemental, mode=args.mode, shots=args.shots,
            precision=args.precision, simulator_seed_offset=args.simulator_seed_offset,
        )
        gates = evaluate_gates(evidence, load_protocol())
        prefix = {
            "internal_validation": "validation",
            "confirmation": "confirmation",
        }.get(args.bank, "development")
        if args.mode == "shots":
            prefix += f"_{args.shots}shots"
        if args.precision != "double":
            prefix += f"_{args.precision}"
        if args.simulator_seed_offset:
            prefix += f"_simplus{args.simulator_seed_offset}"
        atomic_json(RESULTS / f"{prefix}_evidence.json", evidence)
        atomic_json(RESULTS / f"{prefix}_gates.json", gates)
        print(json.dumps(gates, indent=2, sort_keys=True))
    else:
        raise SystemExit(f"{args.stage} requires its stage-specific implementation before execution")


if __name__ == "__main__":
    main()
