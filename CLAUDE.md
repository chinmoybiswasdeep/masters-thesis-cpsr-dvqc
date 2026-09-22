# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Master's thesis research code (INRS-EMT) on quantum reservoir computing / QELM in Qiskit. It is a *scientific* repo: notebooks and `docs/*.md` reports are the deliverables, the Python modules exist to make them reproducible.

## Commands

```bash
.venv/Scripts/python.exe -m pytest tests/ -q          # 369 tests; no config file, tests self-insert code/ on sys.path
.venv/Scripts/python.exe -m pytest tests/test_ipc.py::test_name -q     # single test
cd code && python _build_notebook_<name>.py           # regenerate a notebook from its builder
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 code/<Notebook>.ipynb
```

There is no lint/build step, no `pyproject.toml`, no installed package — imports work via `sys.path.insert(..., "code")`.

Env vars: `DQRC_CACHE_DIR`, `DQRC_V3_CACHE_DIR` (disk caches, gitignored), `IBM_QUANTUM_TOKEN` / `IBM_QUANTUM_INSTANCE` (real-hardware path only; never hard-code credentials).

## Notebooks are generated, not hand-edited

Every `code/DQRC_*.ipynb` and `code/5_*.ipynb` is emitted by its `code/_build_notebook_*.py` generator. **Edit the builder and re-run it**; direct notebook edits are lost. Scientific logic belongs in `code/decoupled_qrc/` (importable, tested), not in cell source — builders should mostly wire modules together.

Versioned lineages (V1 → V2 → V2.1 → V2.2 → V3 → V3.1) are append-only: a new pass gets a *new* builder, notebook, results dir and doc. Never modify a prior version's artifacts; the docs cross-reference them as evidence.

## Architecture

Two largely independent lineages share one base module:

- **`code/qrc_qiskit.py`** — the base recurrent QRC (fixed chain topology, persistent state `|psi_t> = U(u_t)|psi_{t-1}>`, only q0 reset+`Ry(pi*u_t)`-encoded each step). Also the shared utilities everything reuses: `ReservoirConfig`, `chrono_split` (leakage-guarded train/val/test), `nrmse`, `select_and_eval_ridge`, `memory_capacity`, task generators. Import these; do not re-derive them.
- **Mixed-SYK QELM lineage** — `mixed_syk_core.py` (SYK2(g)/SYK4(J) layer, `kappa_to_gJ`, EOC diagnostics; copied verbatim from notebook 4 and regression-tested against it), `eoc_config.py` (loads notebook 4's *already-computed* EOC point, never re-derives kappa), `shadow_measurements.py` (reservoir-agnostic local-Pauli shadows), `jerbi_shadow.py` (Choi-flipped shadow readout built on those two).
- **DQRC lineage** — `code/decoupled_qrc/`, a ~9.8k-line package testing whether memory and nonlinearity can be controlled independently. `utils.py` is the dependency root (RunConfig/`FAST_MODE` vs `PUBLICATION_MODE`, separated RNG streams, disk cache). Subsystems: `memory.py`/`memory_bank.py`/`directional_memory.py` (M), `processor.py`/`nonlinear_processor.py` (P, EOC), and the coupling — `interface.py`, `collision_interface.py`, `dual_route.py`, `directional_dqrc.py`. Measurement: `ipc.py`, `metrics.py`, `diagnostics.py`, `isolation.py`. Orchestration: `run_stages.py`, `v3_1_workflow.py`, `seeded_runner.py`, `candidate_eval.py`, `v3_cache.py`.

Non-obvious invariants, each learned from a real defect:

- **Memory/interface**: `rzz` and `cp` interfaces are diagonal in the processor qubit's own Z basis and inject *nothing* when the processor conserves Z-magnetization (always true for `N_P < 4`). Prefer `'zx'`; if using a diagonal interface, verify with `diagnostics.info_theoretic_diagnostics` that entanglement is actually created.
- **Simulator**: reset-heavy trajectory circuits must use `method='density_matrix'`. `AerSimulator(method='statevector')` silently produces wrong results (see `code/test_aer_statevector_reset_bug.py`); `run_memory_register` refuses it outright.
- **Structural isolation** (`dual_route_current`, `parallel_fixed_taps`): M and P are simulated as *separate circuits* so cross-dependence is impossible by construction. `isolation.py` proves the implementation really does this and is able to fail.
- **Caches** are keyed on a canonical JSON of every scientifically relevant field plus a schema version, so a V2 entry can never satisfy a V3 lookup. Bump `CACHE_SCHEMA_VERSION` when a key-relevant field changes.

## Scientific conventions

- Discovery and confirmation seeds come from disjoint families (`v3_seeds.py`, `SeedBroker`); a confirmation stage that touches a discovery seed is a bug, and there are tests asserting this.
- Never tune an operating point (kappa, EOC g) on data later used for benchmarking — load the frozen value from its source of truth.
- Single-seed results are not evidence here; dynamic-range numbers have repeatedly evaporated across seeds. Report medians across seeds and say which RunConfig (`FAST_MODE` / `PUBLICATION_MODE`) produced a figure.
- Results docs use an explicit claim ladder (Level 1–5) and preregistered gates. State the highest *supported* level and report negative/inconclusive outcomes as such rather than repackaging them.
