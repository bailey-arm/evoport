from __future__ import annotations

import numpy as np
import pandas as pd

def pareto_dominant(df: pd.DataFrame, cols: list[str], higher_is_better: bool | list[bool] = True) -> pd.DataFrame:
    """
    Returns rows on the Pareto frontier — not dominated by any other row.
    A row is dominated if another row is >= on all cols and > on at least one.
    """
    vals = df[cols].to_numpy(dtype=float)
    
    if isinstance(higher_is_better, bool):
        higher_is_better = [higher_is_better] * len(cols)
    
    # Flip cols where lower is better so we always maximise
    signs = np.array([1 if h else -1 for h in higher_is_better])
    vals = vals * signs
    
    n = len(vals)
    is_dominated = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if is_dominated[i]:
            continue
        # Check if any other row dominates row i
        # dominates[j] = True if row j dominates row i
        diff = vals - vals[i]           # shape (n, k)
        weakly_better = (diff >= 0).all(axis=1)   # >= on all cols
        strictly_better = (diff > 0).any(axis=1)  # > on at least one col
        dominated_by = weakly_better & strictly_better
        dominated_by[i] = False         # exclude self
        if dominated_by.any():
            is_dominated[i] = True
    
    return df[~is_dominated].copy()