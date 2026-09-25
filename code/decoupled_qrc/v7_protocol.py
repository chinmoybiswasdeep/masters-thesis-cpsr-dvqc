"""Immutable protocol, seed isolation, hashing, and atomic JSON helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "v7"
PROTOCOL = RESULTS / "preregistered_v7_protocol.json"


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict:
    expected = (RESULTS / "preregistered_v7_protocol.sha256").read_text().split()[0]
    if sha256_file(PROTOCOL) != expected:
        raise RuntimeError("V7 preregistration hash mismatch")
    return json.loads(PROTOCOL.read_text())


def load_seed_bank(stage: str) -> dict:
    if stage not in {"development", "internal_validation"}:
        raise PermissionError("confirmation seeds are inaccessible before freeze")
    name = "development_seeds.json" if stage == "development" else "validation_seeds.json"
    bank = RESULTS / name
    expected = load_protocol()["seed_banks"][stage]["sha256"]
    if sha256_file(bank) != expected:
        raise RuntimeError(f"{stage} seed-bank hash mismatch")
    return json.loads(bank.read_text())


def reveal_confirmation(frozen_manifest: str | Path) -> dict:
    manifest_path = Path(frozen_manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "frozen" or not manifest.get("source_hashes"):
        raise PermissionError("a valid frozen candidate is required")
    source = ROOT / ".git" / "v7_confirmation_seeds.json"
    expected = load_protocol()["seed_banks"]["confirmation"]["salted_sha256_commitment"]
    if sha256_file(source) != expected:
        raise RuntimeError("confirmation seed commitment mismatch")
    bank = json.loads(source.read_text())
    atomic_json(RESULTS / "confirmation_seeds_revealed.json", bank)
    return bank


def assert_seed_disjointness(*banks: dict) -> None:
    seen: set[int] = set()
    for bank in banks:
        for triple in bank["seeds"]:
            values = set(triple.values())
            if len(values) != len(triple) or seen.intersection(values):
                raise AssertionError("seed banks are not pairwise disjoint")
            seen.update(values)


def atomic_json(path: str | Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, target)

