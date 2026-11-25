"""Wilson intervals, exact McNemar test and task-level bootstrap for paired success data."""

import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


def wilson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval as proportions; both bounds are nan when there is no trial."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"impossible counts: {successes} successes out of {trials} trials")
    if trials == 0:
        return (math.nan, math.nan)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value from the two discordant counts of a paired comparison."""
    if only_a < 0 or only_b < 0:
        raise ValueError("discordant counts must not be negative")
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**discordant)


@dataclass(frozen=True)
class PairedCounts:
    both: int
    only_a: int
    only_b: int
    neither: int

    @property
    def pairs(self) -> int:
        return self.both + self.only_a + self.only_b + self.neither


def paired_counts(a: Mapping[Hashable, bool], b: Mapping[Hashable, bool]) -> PairedCounts:
    """Outcome pairs of the episodes present in both mappings."""
    shared = a.keys() & b.keys()
    both = sum(1 for k in shared if a[k] and b[k])
    only_a = sum(1 for k in shared if a[k] and not b[k])
    only_b = sum(1 for k in shared if b[k] and not a[k])
    return PairedCounts(both, only_a, only_b, len(shared) - both - only_a - only_b)


def per_task_differences(a: Mapping[tuple, bool], b: Mapping[tuple, bool]) -> list[float]:
    """Success rate of a minus that of b on the shared episodes, one value per task.

    A key is a tuple whose last element is the episode index; the rest names the task.
    """
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for key in a.keys() & b.keys():
        groups[key[:-1]].append(key)
    return [
        sum(a[k] for k in groups[task]) / len(groups[task])
        - sum(b[k] for k in groups[task]) / len(groups[task])
        for task in sorted(groups)
    ]


def _interval(means: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def task_bootstrap(
    diffs: Sequence[float], n_boot: int = 10_000, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Mean of per-task differences with a percentile interval from resampling whole tasks."""
    values = np.asarray(diffs, dtype=float)
    if values.size == 0:
        raise ValueError("need at least one task")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    low, high = _interval(means, confidence)
    return float(values.mean()), low, high


def contrast_bootstrap(
    diffs_a: Sequence[float],
    diffs_b: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Mean(a) minus mean(b) for two independent task groups, resampling each group's tasks."""
    a = np.asarray(diffs_a, dtype=float)
    b = np.asarray(diffs_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("need at least one task in each group")
    rng = np.random.default_rng(seed)
    means_a = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    means_b = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    low, high = _interval(means_a - means_b, confidence)
    return float(a.mean() - b.mean()), low, high
