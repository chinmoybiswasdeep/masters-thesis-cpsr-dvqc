import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from decoupled_qrc.v8_resources import aggregate_resources


def test_resource_aggregation_counts_physical_work(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps({"resources": {"aer_jobs": 2, "circuit_executions": 3, "total_shots": 30, "peak_qubits": 5, "replay_depths": [2, 7]}}))
    second.write_text(json.dumps({"resources": {"aer_jobs": 4, "physical_circuit_executions": 6, "total_shots": 60, "peak_logical_qubits": 8, "transpiled_depths": [9]}}))
    result = aggregate_resources((first, second))
    assert result["aer_jobs"] == 6
    assert result["physical_circuit_executions"] == 9
    assert result["total_shots"] == 90
    assert result["peak_logical_qubits"] == 8
    assert result["maximum_replay_or_transpiled_depth"] == 9
