"""Edge cases for the frozen STAT-001 calculations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iot_diad.modeling.statistics import (
    average_ranks,
    bca_interval,
    benjamini_hochberg,
    exact_signed_rank,
    friedman,
    holm,
)


def test_average_ranks_use_exact_tie_midranks() -> None:
    actual = average_ranks(np.asarray([4.0, 1.0, 1.0, 3.0]))
    assert np.array_equal(actual, np.asarray([4.0, 1.5, 1.5, 3.0]))


def test_exact_signed_rank_enumerates_signs_and_discards_zeros() -> None:
    ordinary = exact_signed_rank(np.asarray([1.0, 2.0]))
    assert ordinary["statistic"] == 0
    assert ordinary["raw_p_value"] == 0.5
    assert ordinary["enumerations"] == 4

    tied = exact_signed_rank(np.asarray([0.0, 1.0, -1.0]))
    assert tied["statistic"] == 1.5
    assert tied["raw_p_value"] == 1
    assert tied["zero_differences"] == 1
    assert tied["absolute_rank_tie_groups"] == 1


def test_all_zero_signed_rank_is_retained_honestly() -> None:
    result = exact_signed_rank(np.zeros(4))
    assert result["status"] == "all_zero_differences"
    assert result["raw_p_value"] == 1
    assert result["nonzero_differences"] == 0


def test_multiplicity_adjustments_preserve_their_own_family() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_friedman_uses_device_blocks_and_tie_correction() -> None:
    values = np.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    result = friedman(values)
    assert result["devices"] == 3
    assert result["degrees_of_freedom"] == 2
    assert 0 <= result["raw_p_value"] <= 1
    assert result["tie_correction"] < 1


def test_bca_reports_nonfinite_acceleration_for_constant_jackknife() -> None:
    low, high, _, acceleration, status = bca_interval(0.5, np.full(100, 0.5), np.full(14, 0.5))
    assert low is None and high is None and acceleration is None
    assert status == "unavailable_nonfinite_acceleration"


def test_bca_is_finite_for_varying_device_estimates() -> None:
    vector = np.linspace(0.1, 0.9, 14)
    samples = np.random.default_rng(2027).integers(0, 14, size=(2_000, 14))
    bootstrap = vector[samples].mean(axis=1)
    jackknife = np.asarray([np.delete(vector, index).mean() for index in range(14)])
    low, high, bias, acceleration, status = bca_interval(float(vector.mean()), bootstrap, jackknife)
    assert status == "available"
    assert low is not None and high is not None and low < high
    assert bias is not None and math.isfinite(bias)
    assert acceleration is not None and math.isfinite(acceleration)
