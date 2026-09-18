"""
colab_support.py -- environment detection, path resolution, runtime
estimation and checkpointing for the V3 Colab notebook. Kept here (not in
notebook cells) so it is unit-testable off-Colab, per the V3 rule that
scientific and infrastructure logic must not live only in the notebook.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def gpu_available() -> bool:
    """Reports whether Aer actually exposes a GPU device. Never assumes
    `qiskit-aer-gpu` is installed; a missing GPU is not an error."""
    try:
        from qiskit_aer import AerSimulator
        return "GPU" in AerSimulator().available_devices()
    except Exception:
        return False


@dataclass
class Paths:
    repo_root: Path
    code_dir: Path
    results_root: Path
    cache_root: Path
    checkpoint_root: Path
    persistent: bool

    def as_dict(self) -> dict:
        d = {k: str(v) for k, v in asdict(self).items() if k != "persistent"}
        d["persistent"] = self.persistent
        return d


def resolve_paths(repo_root=None, drive_results_root=None, use_drive: bool = False,
                   local_results_subdir: str = "results/dqrc_dual_route_v3") -> Paths:
    """Resolves every path the notebook needs, on Colab or locally.
    When `use_drive` is True AND the Drive mount point exists, results and
    checkpoints go to Drive so a Colab disconnection does not lose work;
    otherwise they stay inside the repository checkout."""
    if repo_root is None:
        here = Path(__file__).resolve()
        repo_root = here.parent.parent.parent
    repo_root = Path(repo_root)
    code_dir = repo_root / "code"

    drive_ok = False
    if use_drive and drive_results_root is not None:
        drive_root = Path(drive_results_root)
        try:
            drive_root.mkdir(parents=True, exist_ok=True)
            drive_ok = drive_root.parent.exists()
        except Exception:
            drive_ok = False

    if drive_ok:
        results_root = Path(drive_results_root)
        cache_root = results_root / "_cache"
        checkpoint_root = results_root / "_checkpoints"
    else:
        results_root = repo_root / local_results_subdir
        cache_root = code_dir / "_dqrc_cache_v3"
        checkpoint_root = results_root / "_checkpoints"

    for p in (results_root, cache_root, checkpoint_root):
        p.mkdir(parents=True, exist_ok=True)

    return Paths(repo_root=repo_root, code_dir=code_dir, results_root=results_root,
                  cache_root=cache_root, checkpoint_root=checkpoint_root, persistent=drive_ok)


def runtime_metadata() -> dict:
    meta = {"in_colab": in_colab(), "python": platform.python_version(), "platform": platform.platform(),
            "gpu_available": gpu_available(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for mod in ("qiskit", "qiskit_aer", "numpy", "scipy", "sklearn", "matplotlib"):
        try:
            meta[f"{mod}_version"] = __import__(mod).__version__
        except Exception:
            meta[f"{mod}_version"] = "not installed"
    try:
        import psutil
        meta["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        meta["ram_gb"] = None
    return meta


@dataclass
class RuntimeEstimate:
    n_simulations: int
    seconds_per_simulation: float
    estimated_seconds: float
    estimated_minutes: float
    estimated_peak_mb: float
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"{self.n_simulations} simulations x ~{self.seconds_per_simulation:.2f}s "
                f"= ~{self.estimated_minutes:.1f} min, peak ~{self.estimated_peak_mb:.0f} MB")


def estimate_runtime(n_simulations: int, seconds_per_simulation: float, n_qubits: int,
                      detail: dict = None) -> RuntimeEstimate:
    """Density-matrix memory is 2^(2n) complex128 = 16 * 4^n bytes, plus
    working copies; a factor of 4 covers transpilation and Aer overhead."""
    bytes_per_dm = 16.0 * (4.0 ** n_qubits)
    peak_mb = 4.0 * bytes_per_dm / 1e6
    total = n_simulations * seconds_per_simulation
    return RuntimeEstimate(n_simulations=n_simulations, seconds_per_simulation=seconds_per_simulation,
                            estimated_seconds=total, estimated_minutes=total / 60.0,
                            estimated_peak_mb=peak_mb, detail=detail or {})


class Checkpointer:
    """Stage/architecture/seed-granular checkpointing. A checkpoint is
    reused ONLY when its stored cache key matches the current one, so a
    resumed run can never silently mix results from different configs."""

    def __init__(self, root, force_recompute: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.force_recompute = force_recompute

    def _path(self, stage: str, name: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}.json"

    def save(self, stage: str, name: str, key: str, payload) -> Path:
        path = self._path(stage, name)
        with open(path, "w") as f:
            json.dump({"key": key, "payload": payload,
                       "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f,
                      indent=2, default=str)
        return path

    def load(self, stage: str, name: str, key: str):
        if self.force_recompute:
            return None
        path = self._path(stage, name)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception:
            return None
        if blob.get("key") != key:
            return None     # config changed -- never reuse
        return blob.get("payload")

    def list_stage(self, stage: str) -> list:
        d = self.root / stage
        return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []
