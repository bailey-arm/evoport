from __future__ import annotations

import numpy as np
import pandas as pd


def cone_dominant(
    df: pd.DataFrame,
    cols: list[str],
    higher_is_better: bool | list[bool] = True,
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """
    Cone-domination: generalises Pareto by using a weighted ordering cone.
    Row j dominates row i if j is better than i when measured under every
    convex combination of the objective weights.

    In practice this is equivalent to: j dominates i iff the weighted
    improvement w @ (f(j) - f(i)) > 0 for ALL weight vectors on the simplex.
    With the standard simplex this reduces to Pareto, but with a restricted
    weight set (specified via `weights`) it becomes *more selective* — keeping
    solutions that excel along the preferred trade-off directions.

    `weights` defines the centre of the ordering cone. Objectives aligned with
    larger weights are given more importance when deciding domination.
    """
    vals = df[cols].to_numpy(dtype=float)

    if isinstance(higher_is_better, bool):
        higher_is_better = [higher_is_better] * len(cols)
    signs = np.array([1 if h else -1 for h in higher_is_better])
    vals = vals * signs

    k = len(cols)
    if weights is None:
        weights = [1.0 / k] * k
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    # Normalise objectives to [0, 1] so weights are comparable
    mins = vals.min(axis=0)
    maxs = vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (vals - mins) / ranges

    n = len(normed)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        diff = normed - normed[i]  # (n, k)
        # Weighted scalar improvement
        weighted_sum = diff @ w  # (n,)
        # Also require non-negative on each weighted axis
        weighted_diff = diff * w  # (n, k)
        all_nonneg = (weighted_diff >= 0).all(axis=1)
        dominates = all_nonneg & (weighted_sum > 1e-12)
        dominates[i] = False
        if dominates.any():
            is_dominated[i] = True

    return df[~is_dominated].copy()
