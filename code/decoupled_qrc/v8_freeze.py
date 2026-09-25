"""Candidate freezing and source-hash verification."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .v8_protocol import ROOT, atomic_json, sha256_file


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def freeze_candidate(
    version: str, files: list[str], preregistration_commit: str, output: str,
    *, require_committed=False,
) -> dict:
    source_commit = _git("rev-parse", "HEAD") if require_committed else None
    if require_committed:
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *files], cwd=ROOT, check=True)
        for name in files:
            _git("ls-files", "--error-unmatch", name)
    manifest = {
        "status": "frozen",
        "version": version,
        "preregistration_commit": preregistration_commit,
        "source_commit": source_commit,
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


def verify_committed_manifest(path: str | Path) -> dict:
    """Reject a locally-created freeze that was not committed before reveal."""
    manifest = verify_manifest(path)
    target = Path(path).resolve()
    relative = target.relative_to(ROOT).as_posix()
    _git("ls-files", "--error-unmatch", relative)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT, check=True)
    if not manifest.get("source_commit"):
        raise RuntimeError("manifest lacks committed source revision")
    _git("merge-base", "--is-ancestor", manifest["source_commit"], "HEAD")
    return manifest
