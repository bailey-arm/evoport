from __future__ import annotations

import numpy as np
import pandas as pd


def lorenz_dominant(
    df: pd.DataFrame,
    cols: list[str],
    higher_is_better: bool | list[bool] = True,
) -> pd.DataFrame:
    """
    Lorenz-domination: favours solutions that are *balanced* across objectives.
    Row j Lorenz-dominates row i if, when both rows' objective values are
    sorted in ascending order, every prefix sum of j's sorted values is >= i's
    (and at least one is strictly >).

    This keeps portfolios where no single objective is sacrificed too heavily,
    producing a frontier that is a subset of the Pareto front biased toward
    equity across objectives.
    """
    vals = df[cols].to_numpy(dtype=float)

    if isinstance(higher_is_better, bool):
        higher_is_better = [higher_is_better] * len(cols)
    signs = np.array([1 if h else -1 for h in higher_is_better])
    vals = vals * signs

    # Normalise to [0, 1] so objectives are comparable
    mins = vals.min(axis=0)
    maxs = vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (vals - mins) / ranges

    # For each row compute sorted values and their prefix sums
    sorted_vals = np.sort(normed, axis=1)  # ascending
    prefix_sums = np.cumsum(sorted_vals, axis=1)

    n = len(prefix_sums)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        diff = prefix_sums - prefix_sums[i]
        weakly_better = (diff >= -1e-12).all(axis=1)
        strictly_better = (diff > 1e-12).any(axis=1)
        dominated_by = weakly_better & strictly_better
        dominated_by[i] = False
        if dominated_by.any():
            is_dominated[i] = True

    return df[~is_dominated].copy()
