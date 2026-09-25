"""Candidate freezing and source-hash verification."""

from __future__ import annotations

import json
from pathlib import Path

from .v8_protocol import ROOT, atomic_json, sha256_file


def freeze_candidate(version: str, files: list[str], preregistration_commit: str, output: str) -> dict:
    manifest = {
        "status": "frozen",
        "version": version,
        "preregistration_commit": preregistration_commit,
        "source_hashes": {name: sha256_file(ROOT / name) for name in sorted(files)},
    }
    atomic_json(output, manifest)
    return manifest


def verify_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("status") != "frozen" or not manifest.get("source_hashes"):
        raise RuntimeError("invalid frozen manifest")
    for name, expected in manifest["source_hashes"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"frozen source changed: {name}")
    return manifest

