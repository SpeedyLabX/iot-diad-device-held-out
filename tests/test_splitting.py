import numpy as np

from iot_diad.splitting import SplitConfig, optimize_assignments, split_score


def test_split_optimizer_is_deterministic_and_preserves_partition_sizes() -> None:
    matrix = np.asarray(
        [
            [100, 10],
            [80, 20],
            [50, 50],
            [20, 80],
            [10, 100],
            [40, 40],
        ],
        dtype=np.float64,
    )
    config = SplitConfig(
        seed=2027,
        random_candidates=100,
        partition_sizes=(4, 1, 1),
        minimum_support=(1, 1, 1),
        total_rows_weight=2.0,
        class_rows_weight=1.0,
        class_support_weight=0.25,
    )
    first, first_score = optimize_assignments(matrix, config)
    second, second_score = optimize_assignments(matrix, config)

    assert np.array_equal(first, second)
    assert first_score == second_score == split_score(matrix, first, config)
    assert [int((first == index).sum()) for index in range(3)] == [4, 1, 1]
