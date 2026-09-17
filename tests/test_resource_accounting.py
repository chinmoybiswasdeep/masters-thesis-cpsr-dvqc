"""
test_resource_accounting.py -- qubit/ancilla/depth counts reported in
`ResourceUsage` must match what the actual built circuit uses (Part 9's
fixed-resource comparisons are only honest if the reported numbers are real).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.experiments import DQRCConfig, build_dqrc_circuit, run_dqrc, run_ablation  # noqa: E402
from decoupled_qrc.utils import make_seed_bundle  # noqa: E402
from qrc_qiskit import random_input  # noqa: E402


def test_dqrc_config_qubit_total_matches_built_circuit():
    cfg = DQRCConfig(N_M=3, N_P=4, stitching_blocks=None)
    seeds = make_seed_bundle(0)
    u = random_input(10, seed=0)
    qc, labels, mem_qubits, proc_qubits, ancilla_qubits = build_dqrc_circuit(cfg, u, seeds)
    assert qc.num_qubits == cfg.n_qubits_total
    assert qc.num_qubits == 1 + len(mem_qubits) + len(proc_qubits) + len(ancilla_qubits)


def test_dqrc_config_qubit_total_matches_built_circuit_with_stitching():
    cfg = DQRCConfig(N_M=2, N_P=4, stitching_blocks=((2, 1.0), (2, 1.0)))
    seeds = make_seed_bundle(0)
    u = random_input(10, seed=0)
    qc, labels, mem_qubits, proc_qubits, ancilla_qubits = build_dqrc_circuit(cfg, u, seeds)
    assert qc.num_qubits == cfg.n_qubits_total
    assert len(ancilla_qubits) == 1  # 2 blocks -> 1 boundary -> 1 ancilla


def test_run_dqrc_resources_match_config():
    cfg = DQRCConfig(N_M=2, N_P=3, reps_processor=1, max_weight_readout=2)
    run = run_dqrc(cfg, T=10, master_seed=0)
    assert run.resources.n_qubits_physical == cfg.n_qubits_total
    assert run.resources.n_ancilla == 0
    assert run.resources.circuit_depth == run.circuit_depth
    assert run.resources.n_features == run.X.shape[1]


def test_run_dqrc_resources_match_config_with_stitching():
    cfg = DQRCConfig(N_M=2, N_P=4, reps_processor=1, max_weight_readout=2,
                      stitching_blocks=((2, 1.0), (2, 1.0)))
    run = run_dqrc(cfg, T=10, master_seed=0)
    assert run.resources.n_ancilla == 1
    assert run.resources.n_qubits_physical == 1 + 2 + 4 + 1


def test_fixed_qubit_comparison_is_actually_matched():
    """A Part 9 fixed-qubit comparison (e.g. N_total=8) must report the SAME
    n_qubits_physical for every ablation drawing from that budget -- this is
    the literal 'don't give the new architecture extra hidden quantum
    resources' check."""
    cfg = DQRCConfig(N_M=3, N_P=4)  # 1 + 3 + 4 = 8
    assert cfg.n_qubits_total == 8
    _, _, _, res_mono = run_ablation("monolithic_qrc", T=10, master_seed=0, N_total=8)
    _, _, _, res_dqrc = run_ablation("dqrc_no_stitch", T=10, master_seed=0, dqrc_cfg=cfg, N_total=8)
    assert res_mono.n_qubits_physical == res_dqrc.n_qubits_physical == 8
