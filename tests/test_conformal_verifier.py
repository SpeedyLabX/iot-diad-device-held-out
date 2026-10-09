"""Independent-verifier edge cases; no production conformal calculations imported."""

import importlib.util
import math
from pathlib import Path
from typing import Any

import pytest


def verifier() -> Any:
    path = Path(__file__).parents[1] / "scripts/verify_conformal.py"
    spec = importlib.util.spec_from_file_location("independent_conformal_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_boundary_and_tied_atom() -> None:
    module = verifier()
    counts = {str(i): 1 for i in range(9)}
    below = {str(i): int(i < 8) for i in range(9)}
    evidence = module.check_quantile(counts, below, counts, 0.5, "hierarchical")
    assert evidence == {"cdf_strictly_below": "4/5", "cdf_at_threshold": "9/10"}
    with pytest.raises(ValueError, match="unnecessary infinite"):
        module.check_quantile(counts, counts, counts, math.inf, "hierarchical")
    with pytest.raises(ValueError, match="CDF bracket"):
        module.check_quantile(counts, counts, counts, 0.6, "hierarchical")


def test_unsupported_class_and_device_infinity() -> None:
    module = verifier()
    for counts in [{}, {str(i): 1000 for i in range(8)}]:
        evidence = module.check_quantile(counts, counts, counts, math.inf, "hierarchical")
        assert evidence["cdf_at_threshold"] == "1"
    with pytest.raises(ValueError, match="CDF bracket"):
        module.check_quantile({"a": 1000}, {"a": 0}, {"a": 1000}, 1, "hierarchical")


def test_equal_row_and_hierarchical_weights_are_distinct() -> None:
    module = verifier()
    counts = {str(i): 1 for i in range(9)} | {"large": 1000}
    through = {str(i): 1 for i in range(9)} | {"large": 0}
    below = {str(i): 0 for i in range(9)} | {"large": 0}
    with pytest.raises(ValueError, match="CDF bracket"):
        module.check_quantile(counts, below, through, 0.5, "hierarchical")
    through["large"] = 900
    assert (
        module.check_quantile(counts, below, through, 0.5, "hierarchical")["cdf_at_threshold"]
        == "9/10"
    )
    assert (
        module.check_quantile(counts, below, through, 0.5, "equal_row_diagnostic")[
            "cdf_at_threshold"
        ]
        == "9/10"
    )
    through["large"] = 1000
    through["8"] = 0
    module.check_quantile(counts, below, through, 0.5, "equal_row_diagnostic")
    with pytest.raises(ValueError, match="CDF bracket"):
        module.check_quantile(counts, below, through, 0.5, "hierarchical")
    through["8"] = 1
    through["large"] = 899
    with pytest.raises(ValueError, match="CDF bracket"):
        module.check_quantile(counts, below, through, 0.5, "equal_row_diagnostic")


def test_aggregate_uses_device_balance_and_retains_empty_sets() -> None:
    module = verifier()
    metrics = module.metrics_from_counts(
        {("small", "Benign"): [1, 0], ("large", "Benign"): [9, 9]}, [10, 12, 6, 2]
    )
    assert metrics["classes"][0]["device_balanced_coverage"] == 0.5
    assert metrics["classes"][0]["row_weighted_coverage"] == 0.9
    assert metrics["mean_set_size"] == 1.2
    assert metrics["singleton_rate"] == 0.6
    assert metrics["empty_set_rate"] == 0.2
    assert metrics["worst_device_class_coverage"] == 0
    assert metrics["classes"][1]["device_balanced_coverage"] is None
    with pytest.raises(ValueError, match="numeric mismatch"):
        module.equal({"coverage": 0.91}, {"coverage": 0.9}, "corrupt metric")
