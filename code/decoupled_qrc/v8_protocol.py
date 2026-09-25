"""V8 immutable protocol and seed-bank verification."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "v8"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, target)


def load_protocol():
    protocol = RESULTS / "preregistered_v8_protocol.json"
    expected = (RESULTS / "preregistered_v8_protocol.sha256").read_text().split()[0]
    if sha256_file(protocol) != expected:
        raise RuntimeError("V8 protocol hash mismatch")
    return json.loads(protocol.read_text())


def load_open_bank(name):
    if name not in {"development", "internal_validation"}:
        raise PermissionError("confirmation bank requires frozen-manifest verification and escrow key")
    protocol = load_protocol()
    path = RESULTS / f"{name}_seeds.json"
    if sha256_file(path) != protocol["seed_banks"][name]["sha256"]:
        raise RuntimeError("seed-bank hash mismatch")
    return json.loads(path.read_text())


def assert_disjoint(*banks):
    seen = set()
    for bank in banks:
        for row in bank["seeds"]:
            values = set(row.values())
            if len(values) != len(row) or seen & values:
                raise AssertionError("seed banks overlap")
            seen |= values


def environment_lock():
    packages = ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy", "scipy", "scikit-learn", "cryptography")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "requirements_lock_sha256": sha256_file(ROOT / "requirements-lock.txt"),
    }
