"""
v3_cache.py -- V3's cache with a NEW schema version, so no V1/V2/V2.1/V2.2
cache entry can ever silently satisfy a V3 lookup (they live in different
namespaces AND hash a different key structure).

The key is a canonical JSON serialisation of every scientifically
relevant field the V3 spec enumerates: architecture, full memory and
processor configs, (m, g, J, lambda), qubit counts, tap depth, interface
controls, reset channel, microstep count, input encoding, feature and
target definitions, split lengths, ALL eight seeds, precision, simulator
backend, package versions, git commit and schema version.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path

CACHE_SCHEMA_VERSION = "v3.0.0"
_CACHE_DIR = Path(os.environ.get("DQRC_V3_CACHE_DIR",
                                 Path(__file__).resolve().parent.parent / "_dqrc_cache_v3"))


def _jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if hasattr(obj, "as_dict") and callable(obj.as_dict):
        return _jsonable(obj.as_dict())
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def environment_fingerprint() -> dict:
    """Package versions, git commit, precision and backend -- all part of
    the cache key, so a V3 result computed under a different environment
    is never silently reused."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    try:
        import qiskit
        import qiskit_aer
        qiskit_v, aer_v = qiskit.__version__, qiskit_aer.__version__
    except Exception:
        qiskit_v, aer_v = "unknown", "unknown"
    try:
        import sklearn
        sk_v = sklearn.__version__
    except Exception:
        sk_v = "unknown"
    return {"schema_version": CACHE_SCHEMA_VERSION, "git_commit": commit, "qiskit": qiskit_v,
            "qiskit_aer": aer_v, "sklearn": sk_v, "python": platform.python_version(),
            "precision": "complex128", "simulator": "aer_density_matrix"}


def canonical_key(**fields) -> str:
    """Deterministic SHA256 over a canonical JSON serialisation. Any
    change to ANY field changes the key."""
    payload = _jsonable(dict(fields))
    payload["_schema"] = CACHE_SCHEMA_VERSION
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_path(namespace: str, key: str) -> Path:
    return _CACHE_DIR / namespace / f"{key}.pkl"


def load_or_none(namespace: str, key: str):
    path = cache_path(namespace, key)
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def store(namespace: str, key: str, value) -> Path:
    path = cache_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f)
    return path


def cached_v3(namespace: str, force_recompute: bool = False):
    """Decorator caching on `canonical_key(**kwargs)` -- so callers MUST
    pass every scientifically relevant field as an explicit keyword, never
    read one from an enclosing notebook global (the V2.1 Defect-10
    lesson)."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(**kwargs):
            key = canonical_key(fn=fn.__qualname__, **kwargs)
            if not force_recompute:
                hit = load_or_none(namespace, key)
                if hit is not None:
                    return hit
            result = fn(**kwargs)
            store(namespace, key, result)
            return result
        wrapper.cache_namespace = namespace
        wrapper.cache_dir = _CACHE_DIR / namespace
        return wrapper
    return decorator


def set_cache_root(path) -> Path:
    """Repoints the cache root (used to place checkpoints on Google Drive
    so a Colab disconnection does not lose completed work)."""
    global _CACHE_DIR
    _CACHE_DIR = Path(path)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def get_cache_root() -> Path:
    return _CACHE_DIR
