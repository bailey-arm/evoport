from __future__ import annotations

import numpy as np
import pandas as pd


def epsilon_dominant(
    df: pd.DataFrame,
    cols: list[str],
    higher_is_better: bool | list[bool] = True,
    epsilon: float | list[float] = 0.05,
) -> pd.DataFrame:
    """
    Epsilon-domination: a row is dominated if another row beats it by at least
    epsilon on every objective. This produces a *sparser* frontier than Pareto
    by collapsing near-equivalent solutions.

    epsilon can be a single float (applied to all cols) or per-column.
    Values are fractions of each column's range.
    """
    vals = df[cols].to_numpy(dtype=float)

    if isinstance(higher_is_better, bool):
        higher_is_better = [higher_is_better] * len(cols)
    signs = np.array([1 if h else -1 for h in higher_is_better])
    vals = vals * signs

    if isinstance(epsilon, (int, float)):
        epsilon = [float(epsilon)] * len(cols)
    eps = np.array(epsilon)

    # Convert fractional epsilon to absolute using column ranges
    col_ranges = vals.max(axis=0) - vals.min(axis=0)
    col_ranges[col_ranges == 0] = 1.0  # avoid division issues
    abs_eps = eps * col_ranges

    n = len(vals)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        diff = vals - vals[i]
        # j eps-dominates i if j >= i + eps on ALL objectives
        eps_better = (diff >= abs_eps).all(axis=1)
        eps_better[i] = False
        if eps_better.any():
            is_dominated[i] = True

    return df[~is_dominated].copy()
