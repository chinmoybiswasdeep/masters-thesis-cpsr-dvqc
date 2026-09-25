"""Small resumable sharding wrapper around ``run_v8_stage.py``."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CORNERS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


def indices(text):
    if "-" in text:
        start, stop = map(int, text.split("-", 1))
        return range(start, stop + 1)
    return [int(value) for value in text.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("DEVELOP_EXACT", "DEVELOP_SHOTS", "VALIDATE", "NEGATIVE_CONTROLS", "ROBUSTNESS", "CONFIRM"))
    parser.add_argument("--bank", choices=("development", "internal_validation", "confirmation"), default="development")
    parser.add_argument("--seed-indices", default="0")
    parser.add_argument("--points", choices=("corners", "grid", "interior", "center", "hh"), default="corners")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--mode", choices=("exact", "shots"), default="exact")
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--simulator-seed-offset", type=int, choices=(0, 100000), default=0)
    parser.add_argument("--transpiler-seed-offset", type=int, choices=(0, 100000), default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    if args.points == "corners":
        points = CORNERS
    elif args.points == "grid":
        points = tuple((m, g) for m in GRID for g in GRID)
    elif args.points == "interior":
        points = tuple((m, g) for m in GRID for g in GRID if (m, g) not in CORNERS)
    elif args.points == "center":
        points = ((0.5, 0.5),)
    else:
        points = ((1.0, 1.0),)
    jobs = [(seed, point) for seed in indices(args.seed_indices) for point in points]
    if args.stage == "NEGATIVE_CONTROLS":
        jobs = [(seed, None) for seed in indices(args.seed_indices)]
    jobs = [job for index, job in enumerate(jobs) if index % args.shard_count == args.shard_index]
    stage_script = Path(__file__).with_name("run_v8_stage.py")
    for seed, point in jobs:
        command = [
            sys.executable, str(stage_script), args.stage,
            "--bank", args.bank, "--seed-index", str(seed),
            "--mode", args.mode, "--shots", str(args.shots), "--precision", args.precision,
            "--simulator-seed-offset", str(args.simulator_seed_offset),
            "--transpiler-seed-offset", str(args.transpiler_seed_offset),
        ]
        if point is not None:
            command.extend(("--m", str(point[0]), "--g", str(point[1])))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
