import math

import numpy as np
import pytest

from vlashift.stats import (
    PairedCounts,
    contrast_bootstrap,
    mcnemar_exact,
    paired_counts,
    per_task_differences,
    task_bootstrap,
    wilson,
)


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (9, 10, 0.5958, 0.9821),
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (50, 100, 0.4038, 0.5962),
        (1, 1, 0.2065, 1.0),
    ],
)
def test_wilson_matches_reference_values(k, n, low, high):
    got_low, got_high = wilson(k, n)
    assert got_low == pytest.approx(low, abs=1e-3)
    assert got_high == pytest.approx(high, abs=1e-3)


def test_wilson_bounds_contain_the_rate_and_stay_in_unit_interval():
    for n in (1, 7, 100):
        for k in range(n + 1):
            low, high = wilson(k, n)
            assert 0.0 <= low <= k / n <= high <= 1.0


def test_wilson_interval_narrows_with_more_trials():
    low10, high10 = wilson(5, 10)
    low400, high400 = wilson(200, 400)
    assert high400 - low400 < high10 - low10


def test_wilson_confidence_level_changes_the_width():
    narrow = wilson(30, 100, confidence=0.80)
    wide = wilson(30, 100, confidence=0.99)
    assert wide[0] < narrow[0] < narrow[1] < wide[1]


def test_wilson_without_trials_is_nan_and_bad_counts_raise():
    assert all(math.isnan(x) for x in wilson(0, 0))
    for k, n in [(-1, 5), (6, 5), (1, -1)]:
        with pytest.raises(ValueError, match="impossible"):
            wilson(k, n)


@pytest.mark.parametrize(
    ("a", "b", "p"),
    [
        (0, 0, 1.0),
        (5, 5, 1.0),
        (1, 0, 1.0),
        (10, 0, 2 / 1024),
        (0, 10, 2 / 1024),
        (3, 9, 598 / 4096),
        (9, 3, 598 / 4096),
        (2, 8, 0.109375),
    ],
)
def test_mcnemar_exact_values(a, b, p):
    assert mcnemar_exact(a, b) == pytest.approx(p)


def test_mcnemar_is_symmetric_and_a_probability():
    for a in range(0, 12):
        for b in range(0, 12):
            p = mcnemar_exact(a, b)
            assert p == mcnemar_exact(b, a)
            assert 0.0 < p <= 1.0


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError, match="negative"):
        mcnemar_exact(-1, 2)


def test_paired_counts_use_only_shared_episodes():
    a = {(0, 0): True, (0, 1): True, (0, 2): False, (1, 0): False, (9, 9): True}
    b = {(0, 0): True, (0, 1): False, (0, 2): True, (1, 0): False, (7, 7): True}
    counts = paired_counts(a, b)
    assert counts == PairedCounts(both=1, only_a=1, only_b=1, neither=1)
    assert counts.pairs == 4


def test_paired_counts_of_disjoint_runs_is_empty():
    assert paired_counts({(0, 0): True}, {(1, 1): True}).pairs == 0


def test_per_task_differences_are_success_rate_a_minus_b():
    a = {(0, 0): True, (0, 1): True, (1, 0): True, (1, 1): False}
    b = {(0, 0): True, (0, 1): False, (1, 0): False, (1, 1): False}
    assert per_task_differences(a, b) == [0.5, 0.5]
    assert per_task_differences(b, a) == [-0.5, -0.5]


def test_per_task_differences_ignore_unshared_episodes():
    a = {(0, 0): True, (0, 1): True, (2, 0): True}
    b = {(0, 0): False}
    assert per_task_differences(a, b) == [1.0]


def test_task_bootstrap_of_equal_differences_is_a_point():
    mean, low, high = task_bootstrap([0.3] * 10)
    assert mean == pytest.approx(0.3)
    assert low == pytest.approx(0.3)
    assert high == pytest.approx(0.3)


def test_task_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    diffs = [0.0, 0.1, 0.4, 0.2, 0.3, 0.0, 0.5, 0.2, 0.1, 0.4]
    mean, low, high = task_bootstrap(diffs, n_boot=2000, seed=1)
    assert low < mean < high
    assert (mean, low, high) == task_bootstrap(diffs, n_boot=2000, seed=1)
    assert mean == pytest.approx(np.mean(diffs))


def test_task_bootstrap_interval_is_wider_for_more_spread():
    tight = task_bootstrap([0.2, 0.21, 0.19, 0.2, 0.22, 0.18], seed=2)
    loose = task_bootstrap([0.9, -0.5, 0.4, -0.3, 0.8, -0.2], seed=2)
    assert loose[2] - loose[1] > tight[2] - tight[1]


def test_task_bootstrap_needs_a_task():
    with pytest.raises(ValueError, match="at least one task"):
        task_bootstrap([])


def test_contrast_bootstrap_is_the_difference_of_the_group_means():
    a = [0.5, 0.6, 0.7, 0.4, 0.5]
    b = [0.0, 0.1, 0.0, 0.1, 0.0]
    mean, low, high = contrast_bootstrap(a, b, n_boot=3000, seed=0)
    assert mean == pytest.approx(0.5)
    assert low < mean < high
    assert low > 0.3
    swapped = contrast_bootstrap(b, a, n_boot=3000, seed=0)
    assert swapped[0] == pytest.approx(-0.5)


def test_contrast_bootstrap_needs_both_groups():
    with pytest.raises(ValueError, match="each group"):
        contrast_bootstrap([0.1], [])


def test_per_task_differences_group_pooled_suites_by_suite_and_task():
    a = {
        ("s1", 0, 0): True,
        ("s1", 0, 1): True,
        ("s1", 1, 0): False,
        ("s1", 1, 1): False,
        ("s2", 0, 0): True,
        ("s2", 0, 1): False,
    }
    b = dict.fromkeys(a, False)
    assert per_task_differences(a, b) == [1.0, 0.0, 0.5]
