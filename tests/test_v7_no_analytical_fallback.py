import ast
from pathlib import Path


def test_architecture_and_runner_have_no_numpy_or_fallback():
    root = Path(__file__).resolve().parents[1] / "code" / "decoupled_qrc"
    for name in ("v7_qiskit_architecture.py", "v7_qiskit_runner.py"):
        source = (root / name).read_text()
        tree = ast.parse(source)
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        assert "numpy" not in imports
        assert "binomial" not in source
        assert "analytical" not in source.lower()

