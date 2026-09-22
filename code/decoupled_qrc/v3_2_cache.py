"""
v3_2_cache.py -- the V3.2 cache namespace.

A NEW schema version and a NEW directory, so no V1/V2/V2.1/V2.2/V3/V3.1 entry
can ever satisfy a V3.2 lookup: they hash a different key structure AND live
in a different namespace.

WHAT V3'S KEY WAS MISSING. `v3_cache.canonical_key` accepts arbitrary
`**fields` and, as actually called from `candidate_eval.py`, carried
(m, g, J), the seed bundle, N_M/N_P/theta/phi/ap_kind, the control ranges,
the stencil steps, the IPC settings and the thresholds -- but NOT the
processor's Hamiltonian term set, the input encoder, the observable set, dt,
the layer count or the layer ordering. Under V3.2 those are exactly the
fields that change between ablations, so an incomplete key could serve an
`encoder_only` result for a `full` request. `processor_key_fields` and
`architecture_key_fields` enumerate them explicitly, and
`tests/test_v3_2_cache.py` includes an INTENTIONAL-ALIAS test that constructs
two configurations differing only in a Hamiltonian field and asserts their
keys differ.

THE SHOT BUDGET AND SHOT SEED ARE PART OF THE KEY. The primary metric is
defined at a finite shot budget, so a cached result computed at a different
budget is a different measurement.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

CACHE_SCHEMA_VERSION = "v3.2.0"

_CACHE_DIR = Path(os.environ.get(
    "DQRC_V3_2_CACHE_DIR",
    Path(__file__).resolve().parent.parent / "_dqrc_cache_v3_2"))


def _jsonable(obj):
    """Deterministic, recursive conversion to JSON-safe primitives."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "as_dict"):
        return _jsonable(obj.as_dict())
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    return str(obj)


def environment_fingerprint() -> dict:
    """Everything about the run environment that could change a number."""
    try:
        import numpy as np
        numpy_v = np.__version__
    except Exception:                                   # pragma: no cover
        numpy_v = "unknown"
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:                                   # pragma: no cover
        scipy_v = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                                   # pragma: no cover
        commit = "unknown"
    return {"schema": CACHE_SCHEMA_VERSION, "python": sys.version.split()[0],
            "numpy": numpy_v, "scipy": scipy_v, "platform": platform.platform(),
            "git_commit": commit, "precision": "complex128"}


def processor_key_fields(spec, ablation: str, g: float, J: float,
                          hamiltonian_seed: int, disorder_seed: int) -> dict:
    """Every processor field that can change a feature value."""
    d = spec.as_dict()
    return {"processor.N_P": d["N_P"], "processor.dt": d["dt"], "processor.R": d["R"],
            "processor.layer_order": d["layer_order"],
            "processor.readout": d["readout"],
            "processor.chain_periodic": d["chain_periodic"],
            "processor.omega_lo": d["omega_lo"], "processor.omega_hi": d["omega_hi"],
            "processor.field_amp": d["field_amp"],
            "processor.hamiltonian_terms": ["H_mix(Xi,Yi,Zi disordered)",
                                            "H_2=sum_<ij> ZiZj",
                                            "H_4=sum_quartets ZZZZ"],
            "processor.ablation": str(ablation),
            "processor.encoder": "linear_product_z:(I+sZ)/2",
            "processor.g": float(g), "processor.J": float(J),
            "processor.hamiltonian_seed": int(hamiltonian_seed),
            "processor.disorder_seed": int(disorder_seed)}


def memory_key_fields(spec, m: float, disorder_seed: int) -> dict:
    d = spec.as_dict()
    return {"memory.mechanism": d["mechanism"], "memory.L": d["L"],
            "memory.max_weight": d["max_weight"], "memory.hop": d["hop"],
            "memory.tau_mix": d["tau_mix"], "memory.field_amp": d["field_amp"],
            "memory.control_range": d["control_range"],
            "memory.encoder": "linear_product_z:(I+sZ)/2",
            "memory.m": float(m), "memory.disorder_seed": int(disorder_seed)}


def architecture_key_fields(arch_spec, *, m, g, J, ablation, hamiltonian_seed,
                            disorder_seed) -> dict:
    d = arch_spec.as_dict()
    out = {"architecture": d["architecture"], "lambda": d["lambda"],
           "mem_tap": d["mem_tap"], "proc_entry": d["proc_entry"],
           "coupling": d["coupling"]}
    if arch_spec.has_memory:
        out.update(memory_key_fields(arch_spec.memory, m, disorder_seed))
    if arch_spec.has_processor:
        out.update(processor_key_fields(arch_spec.processor, ablation, g, J,
                                        hamiltonian_seed, disorder_seed))
    return out


def canonical_key(**fields) -> str:
    """SHA256 over a canonical JSON serialisation of every supplied field
    plus the environment fingerprint and schema version."""
    payload = {"_schema": CACHE_SCHEMA_VERSION,
               "_env": environment_fingerprint(),
               "fields": _jsonable(fields)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def cached_v3_2(key_fn):
    """Disk-memoise a function. `key_fn(*args, **kwargs) -> dict` must return
    EVERY scientifically relevant field; anything it omits is a potential
    alias."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fields = dict(key_fn(*args, **kwargs))
            fields["_fn"] = fn.__qualname__
            key = canonical_key(**fields)
            path = cache_dir() / f"{key}.pkl"
            if path.exists():
                try:
                    with open(path, "rb") as fh:
                        return pickle.load(fh)
                except Exception:                        # pragma: no cover
                    path.unlink(missing_ok=True)         # corrupt entry: recompute
            result = fn(*args, **kwargs)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(result, fh)
            tmp.replace(path)                            # atomic: no half-written entry
            return result
        wrapper.cache_key = lambda *a, **k: canonical_key(**{**dict(key_fn(*a, **k)),
                                                            "_fn": fn.__qualname__})
        wrapper.uncached = fn
        return wrapper
    return decorator
