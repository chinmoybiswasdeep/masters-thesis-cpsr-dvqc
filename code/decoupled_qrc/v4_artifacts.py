"""
v4_artifacts.py -- freezing, hashing, checkpointing and claim enforcement.

The one sentence this module exists to protect:

    MEMORY-NONLINEARITY SEPARATION DEMONSTRATED

`ClaimGuard.render` is the ONLY place that sentence can be produced, and it
refuses unless every mandatory gate passed on a confirmation run whose frozen
config hash matches the artifact that was frozen before the data existed. A
caller cannot pass a flag to override it.

FREEZING. `freeze` serialises the architecture, the grid, the metric
definitions, the target libraries, the thresholds, the preprocessing, the
observable sets, the resource budgets and the statistical plan into one
payload, hashes it, and writes it read-only-by-convention. `load_frozen`
re-hashes on read, so a payload edited after freezing fails to load rather
than silently taking effect.

CHECKPOINTING. Expensive grid rows and search candidates append to JSONL one
row at a time, so an interrupted run resumes having lost at most one row.
Completeness is DERIVED from an expected-row-key check, never asserted.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class FrozenViolation(RuntimeError):
    pass


class ClaimViolation(RuntimeError):
    pass


def canonical_json(payload) -> str:
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if hasattr(o, "as_dict"):
        return _jsonable(o.as_dict())
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def sha256_of(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def environment() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = "unknown"
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy_v,
            "platform": platform.platform(), "git_commit": commit,
            "precision": "complex128", "cpu_count": os.cpu_count()}


# =============================================================================
# Freezing
# =============================================================================
@dataclass
class FrozenConfig:
    payload: dict
    sha256: str
    created_utc: str
    path: Path = None

    def as_dict(self) -> dict:
        return {"sha256": self.sha256, "created_utc": self.created_utc,
                "path": str(self.path) if self.path else None,
                "components": sorted(self.payload)}


REQUIRED_FROZEN_COMPONENTS = (
    "architecture", "grid", "metrics", "target_libraries", "thresholds",
    "preprocessing", "observables", "resources", "statistics", "seed_bank",
)


def freeze(payload: dict, path) -> FrozenConfig:
    """Freeze the complete experimental definition and hash it.

    Every component in `REQUIRED_FROZEN_COMPONENTS` must be present. A missing
    component means something was left free to move between freezing and
    confirmation, which is the failure mode freezing exists to prevent.
    """
    missing = [k for k in REQUIRED_FROZEN_COMPONENTS if k not in payload]
    if missing:
        raise FrozenViolation(
            f"refusing to freeze: missing component(s) {missing}. Anything not frozen "
            f"is free to move between freezing and confirmation.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_of(payload)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    blob = {"sha256": digest, "created_utc": created, "environment": environment(),
            "payload": _jsonable(payload)}
    path.write_text(json.dumps(blob, indent=2, sort_keys=True), encoding="utf-8")
    return FrozenConfig(payload=_jsonable(payload), sha256=digest,
                        created_utc=created, path=path)


def load_frozen(path) -> FrozenConfig:
    """Load and RE-HASH. A payload edited after freezing fails here."""
    path = Path(path)
    if not path.exists():
        raise FrozenViolation(f"no frozen artifact at {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    recomputed = sha256_of(blob["payload"])
    if recomputed != blob["sha256"]:
        raise FrozenViolation(
            f"frozen artifact at {path} has been MODIFIED since freezing "
            f"(stored {blob['sha256'][:16]}..., recomputed {recomputed[:16]}...)")
    return FrozenConfig(payload=blob["payload"], sha256=blob["sha256"],
                        created_utc=blob["created_utc"], path=path)


def assert_matches_frozen(frozen: FrozenConfig, current: dict, component: str) -> None:
    want = sha256_of(frozen.payload.get(component))
    got = sha256_of(_jsonable(current))
    if want != got:
        raise FrozenViolation(
            f"component {component!r} differs from the frozen artifact "
            f"(frozen {want[:16]}..., current {got[:16]}...). Confirmation must run the "
            f"configuration that was frozen, not a variant of it.")


# =============================================================================
# Checkpointing
# =============================================================================
class Checkpoint:
    """Append-only JSONL row store; an interrupted run loses at most one row."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def done_keys(self) -> set:
        if not self.path.exists():
            return set()
        keys = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(str(json.loads(line)["row_key"]))
            except Exception:
                continue                    # tolerate a torn final line
        return keys

    def append(self, row_key: str, row: dict) -> None:
        payload = dict(_jsonable(row))
        payload["row_key"] = str(row_key)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def rows(self) -> list:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out


def verify_complete(rows, expected_keys, *, required_fields=(),
                    config_hash: str = None) -> dict:
    """DERIVE completeness. Never assign it."""
    import math
    rows = list(rows)
    expected = list(dict.fromkeys(str(k) for k in expected_keys))
    seen = [str(r.get("row_key")) for r in rows]
    dup = sorted({k for k in seen if seen.count(k) > 1})
    missing = sorted(set(expected) - set(seen))
    unexpected = sorted(set(seen) - set(expected))
    bad_hash, non_finite = [], []
    for r in rows:
        if config_hash is not None and str(r.get("config_hash")) != str(config_hash):
            bad_hash.append(r.get("row_key"))
        for f in required_fields:
            v = r.get(f)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                non_finite.append([r.get("row_key"), f])
    complete = bool(expected and not missing and not dup and not unexpected
                    and not bad_hash and not non_finite)
    return {"complete": complete, "n_expected": len(expected), "n_rows": len(rows),
            "missing": missing, "duplicates": dup, "unexpected": unexpected,
            "wrong_config_hash": bad_hash, "non_finite": non_finite}


# =============================================================================
# Claim enforcement
# =============================================================================
CLAIM_SENTENCE = "MEMORY-NONLINEARITY SEPARATION DEMONSTRATED"


@dataclass
class ClaimGuard:
    """The only route to the claim sentence."""

    gates: dict = field(default_factory=dict)
    confirmation_complete: bool = False
    frozen_hash: str = None
    run_config_hash: str = None
    sequential_valid: bool = True

    def blockers(self) -> list:
        out = []
        failed = sorted(k for k, v in self.gates.items()
                        if not (isinstance(v, dict) and v.get("passed")))
        if failed:
            out.append(f"gate(s) not passed: {failed}")
        if not self.gates:
            out.append("no gates were evaluated")
        if not self.confirmation_complete:
            out.append("confirmation run is not complete")
        if self.frozen_hash is None or self.run_config_hash is None:
            out.append("missing frozen-config provenance")
        elif self.frozen_hash != self.run_config_hash:
            out.append("confirmation config does not match the frozen artifact")
        if not self.sequential_valid:
            out.append("sequential error control is exhausted or invalid")
        return out

    def render(self) -> str:
        """Return the claim sentence, or a refusal naming every blocker."""
        bl = self.blockers()
        if bl:
            return ("SEPARATION NOT DEMONSTRATED -- " + "; ".join(bl))
        return CLAIM_SENTENCE

    def as_dict(self) -> dict:
        return {"claim": self.render(), "blockers": self.blockers(),
                "n_gates": len(self.gates),
                "n_passed": sum(1 for v in self.gates.values()
                                if isinstance(v, dict) and v.get("passed")),
                "confirmation_complete": self.confirmation_complete,
                "frozen_hash": self.frozen_hash,
                "run_config_hash": self.run_config_hash,
                "sequential_valid": self.sequential_valid}
