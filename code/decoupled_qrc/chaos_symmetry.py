"""
chaos_symmetry.py -- V2.1 Phase 5: the existing level-spacing ratio
(`mixed_syk_core.level_spacing_ratio`) is computed on the FULL Hilbert
space, without first checking whether the processor's step unitary
commutes with any exact symmetry. If it does, mixing eigenphases from
different symmetry sectors artificially skews <r> toward Poisson
statistics even for a genuinely chaotic Hamiltonian (Berry/Tabor-style
sector-mixing artifact) -- so <r> must be verified sector-by-sector, not
assumed to need no resolution AND not blindly resolved against an
assumed symmetry that may not actually be present in this specific random
realization.

This module NUMERICALLY DETECTS which (if any) of a small library of
candidate global-parity symmetries actually commutes with a given
processor step unitary, then computes <r> within each resolved sector
(excluding sectors too small for meaningful statistics), falling back
honestly to the unresolved full-spectrum statistic when no candidate
symmetry is found to commute (which is the EXPECTED outcome for a random
mixed-Pauli SYK4 Hamiltonian with generic couplings -- verified, not
assumed, either way).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

import mixed_syk_core as msc  # noqa: E402

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_LOCAL_PAULI = {"X": _X, "Y": _Y, "Z": _Z}


def _global_parity_operator(N: int, pauli: str) -> np.ndarray:
    """Kron of the given single-qubit Pauli on every one of N qubits --
    Qiskit's little-endian convention is irrelevant here since the SAME
    Pauli is placed on every qubit (the tensor product is symmetric under
    qubit relabeling for this specific operator)."""
    P = _LOCAL_PAULI[pauli]
    op = np.array([[1.0]], dtype=complex)
    for _ in range(N):
        op = np.kron(op, P)
    return op


def commutator_norm(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A @ B - B @ A))


def detect_symmetries(U1: np.ndarray, N: int, tol: float = 1e-8) -> dict:
    """Numerically checks whether each candidate global-parity operator
    (Pi_X = kron(X,X,...,X), Pi_Y, Pi_Z) commutes with the given
    single-layer processor unitary. Returns {name: (commutes: bool,
    commutator_norm: float)}."""
    out = {}
    for name in ("X", "Y", "Z"):
        S = _global_parity_operator(N, name)
        norm = commutator_norm(S, U1)
        out[f"global_{name}_parity"] = {"commutes": norm < tol, "commutator_norm": norm}
    return out


@dataclass
class SymmetryResolvedChaos:
    symmetry_detection: dict          # detect_symmetries()'s own output, for full transparency
    symmetry_used: str                # name of the symmetry block-diagonalized on, or None
    n_sectors_total: int
    sector_sizes: dict                # {sector_label: dim}
    sectors_excluded_too_small: list  # [sector_label, ...]
    r_by_sector: dict                 # {sector_label: <r>}
    r_mean: float                     # mean of r_by_sector (excluding excluded sectors); NaN if none usable
    r_unresolved_full_spectrum: float  # the naive, sector-UNAWARE <r> (mixed_syk_core.level_spacing_ratio),
                                       # always reported alongside, for direct before/after comparison


def _level_spacing_ratio_from_phases(phases: np.ndarray) -> float:
    if len(phases) < 4:
        return float("nan")
    phases = np.sort(phases)
    gaps = np.diff(np.concatenate([phases, [phases[0] + 2 * np.pi]]))
    gaps = gaps[gaps > 1e-13]
    if len(gaps) < 3:
        return float("nan")
    r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
    return float(np.mean(r))


def symmetry_resolved_level_spacing(U1: np.ndarray, N: int, min_sector_size: int = 8,
                                     tol: float = 1e-8) -> SymmetryResolvedChaos:
    """Phase 5's corrected chaos diagnostic. If a candidate global-parity
    symmetry is found to commute with `U1`, block-diagonalizes into its
    +1/-1 eigenspaces and computes <r> WITHIN each sector big enough for
    meaningful statistics (>= min_sector_size levels), reporting per-sector
    values and their mean. If no candidate symmetry commutes (the
    physically expected case for a random mixed-Pauli SYK4 Hamiltonian),
    falls back to the honest unresolved full-spectrum <r> -- always
    computed and reported alongside, regardless, for direct comparison."""
    detection = detect_symmetries(U1, N, tol=tol)
    r_full = msc.level_spacing_ratio(U1)
    commuting = [name for name, info in detection.items() if info["commutes"]]

    if not commuting:
        return SymmetryResolvedChaos(
            symmetry_detection=detection, symmetry_used=None, n_sectors_total=1,
            sector_sizes={"full_spectrum": 2 ** N}, sectors_excluded_too_small=[],
            r_by_sector={"full_spectrum": r_full}, r_mean=r_full, r_unresolved_full_spectrum=r_full,
        )

    name = commuting[0]
    pauli = name.split("_")[1]
    S = _global_parity_operator(N, pauli)
    eigvals_S, eigvecs_S = np.linalg.eigh(S)

    sectors = {}
    for val, vec in zip(eigvals_S, eigvecs_S.T):
        key = f"{pauli}_parity={int(round(val.real)):+d}"
        sectors.setdefault(key, []).append(vec)

    sector_sizes = {k: len(v) for k, v in sectors.items()}
    r_by_sector, excluded = {}, []
    for key, vecs in sectors.items():
        V = np.array(vecs).T
        if V.shape[1] < min_sector_size:
            excluded.append(key)
            continue
        U_block = V.conj().T @ U1 @ V
        eigvals = np.linalg.eigvals(U_block)
        phases = np.angle(eigvals)
        r = _level_spacing_ratio_from_phases(phases)
        if not np.isnan(r):
            r_by_sector[key] = r
        else:
            excluded.append(key)

    r_mean = float(np.mean(list(r_by_sector.values()))) if r_by_sector else float("nan")
    return SymmetryResolvedChaos(
        symmetry_detection=detection, symmetry_used=name, n_sectors_total=len(sectors),
        sector_sizes=sector_sizes, sectors_excluded_too_small=excluded, r_by_sector=r_by_sector,
        r_mean=r_mean, r_unresolved_full_spectrum=r_full,
    )
