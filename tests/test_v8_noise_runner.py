import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from decoupled_qrc.v8_noise_runner import FAKE_BACKEND_NAME, PHYSICAL_SUBGRAPH


def test_noise_backend_is_frozen_and_not_a_density_matrix_expansion():
    source = (Path(__file__).resolve().parents[1] / "code/decoupled_qrc/v8_noise_runner.py").read_text()
    assert FAKE_BACKEND_NAME == "FakeGuadalupeV2"
    assert PHYSICAL_SUBGRAPH == tuple(range(16))
    assert "NoiseModel.from_backend" in source
    assert 'method="matrix_product_state"' in source
    assert "density_matrix" not in source


def test_frozen_fake_backend_subgraph_is_connected():
    from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

    adjacency = {qubit: set() for qubit in PHYSICAL_SUBGRAPH}
    for left, right in FakeGuadalupeV2().coupling_map.get_edges():
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    reached, pending = set(), [PHYSICAL_SUBGRAPH[0]]
    while pending:
        node = pending.pop()
        if node not in reached:
            reached.add(node)
            pending.extend(adjacency[node] - reached)
    assert reached == set(PHYSICAL_SUBGRAPH)
