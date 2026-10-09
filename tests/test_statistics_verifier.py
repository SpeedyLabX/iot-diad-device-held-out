"""Independent STAT-001 verifier calculation tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def verifier() -> Any:
    path = Path(__file__).parents[1] / "scripts/verify_statistics.py"
    spec = importlib.util.spec_from_file_location("independent_statistics_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_signed_rank_handles_ties_and_zeros() -> None:
    statistic, p_value, nonzero, zero = verifier().signed_rank(np.asarray([0.0, 1.0, -1.0]))
    assert statistic == 1.5
    assert p_value == 1
    assert nonzero == 2
    assert zero == 1


def test_independent_adjustments() -> None:
    module = verifier()
    assert module.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert module.bh([0.01, 0.04, 0.03]) == [0.03, 0.04, 0.04]
