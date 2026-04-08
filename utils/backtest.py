from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.enums.data import Data
from utils.loss.template import ListLoss
from utils.evolution import Evolution, EvolutionResult


@dataclass
class BacktestResult:
    portfolio_returns: np.ndarray
    rebalance_points: list[int]
    chosen_weights: list[np.ndarray]
    evolution_results: list[EvolutionResult]

    @property
    def cumulative_returns(self) -> np.ndarray:
        return np.cumprod(1 + self.portfolio_returns) - 1

    @property
    def total_return(self) -> float:
        return float((1 + self.portfolio_returns).prod() - 1)


class Backtest:
    def __init__(
        self,
        population_cls,
        population_kwargs: dict,
        list_loss: ListLoss,
        domination_fn: callable,
        domination_kwargs: dict | None = None,
        n_generations: int = 10,
        criterion: str | callable = 'sharpe',
    ):
        self.population_cls = population_cls
        self.population_kwargs = population_kwargs
        self.list_loss = list_loss
        self.domination_fn = domination_fn
        self.domination_kwargs = domination_kwargs or {}
        self.n_generations = n_generations
        self.criterion = criterion

    def select(self, result: EvolutionResult) -> np.ndarray:
        if callable(self.criterion):
            return self.criterion(result)

        frontier_obj = result.frontier_objectives
        frontier_w = result.frontier_weights

        if self.criterion == 'aggressive':
            # Highest mean return (sum of mean_ret * w proxied by equal
            # weighting towards the highest-return end of the frontier).
            # Pick the frontier portfolio whose weight-weighted mean return
            # is largest — use the first objective column that is maximised.
            col = frontier_obj.columns[0]
            best_idx = frontier_obj[col].idxmax()

        elif self.criterion == 'defensive':
            # Lowest portfolio variance: w' Σ w
            # Computed directly from frontier weights and the return matrix
            # stored in the last evolution's data.  Falls back to the
            # objective with the smallest spread if returns aren't available.
            best_idx = self._lowest_variance(result)

        elif self.criterion == 'sharpe':
            # Best Sharpe ratio — look for a column whose name contains
            # 'sharpe' (case-insensitive), else fall back to first column.
            col = self._find_column(frontier_obj, 'sharpe')
            idx = frontier_obj.columns.tolist().index(col)
            if self.list_loss.minimax_structure[idx]:
                best_idx = frontier_obj[col].idxmin()
            else:
                best_idx = frontier_obj[col].idxmax()

        else:
            # Treat criterion as a column name (or substring) and maximise
            # it, respecting the minimax structure.
            col = self._find_column(frontier_obj, self.criterion)
            idx = frontier_obj.columns.tolist().index(col)
            if self.list_loss.minimax_structure[idx]:
                best_idx = frontier_obj[col].idxmin()
            else:
                best_idx = frontier_obj[col].idxmax()

        return result.weights[best_idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_column(df: pd.DataFrame, hint: str) -> str:
        hint_lower = hint.lower()
        for c in df.columns:
            if hint_lower in c.lower():
                return c
        raise KeyError(
            f'No column matching "{hint}" in frontier objectives: '
            f'{df.columns.tolist()}'
        )

    @staticmethod
    def _lowest_variance(result: EvolutionResult) -> int:
        frontier_w = result.frontier_weights
        if result.returns is not None:
            cov = np.cov(result.returns, rowvar=False)
            port_var = (frontier_w @ cov * frontier_w).sum(axis=1)
        else:
            # Fallback: lowest weight concentration (HHI proxy)
            port_var = (frontier_w ** 2).sum(axis=1)
        return result.frontier_idx[np.argmin(port_var)]

    @staticmethod
    def _build_data_package(returns_slice: np.ndarray) -> dict:
        return {
            Data.MeanRet: returns_slice.mean(axis=0),
            Data.Returns: returns_slice,
        }

    def run(
        self,
        returns: np.ndarray,
        train_window: int,
        step_size: int,
        expanding: bool = True,
        verbose: bool = False,
    ) -> BacktestResult:
        n_total = len(returns)
        all_oos_returns = []
        rebalance_points = []
        chosen_weights = []
        evolution_results = []

        t = train_window
        step = 0
        while t < n_total:
            if expanding:
                train_data = returns[:t]
            else:
                train_data = returns[t - train_window : t]

            data_package = self._build_data_package(train_data)

            population = self.population_cls(**self.population_kwargs)
            evo = Evolution(
                population=population,
                list_loss=self.list_loss,
                domination_fn=self.domination_fn,
                domination_kwargs=self.domination_kwargs,
            )
            result = evo.run(data_package, self.n_generations)
            evolution_results.append(result)

            weights = self.select(result)
            chosen_weights.append(weights)
            rebalance_points.append(t)

            oos_end = min(t + step_size, n_total)
            oos_returns = returns[t:oos_end] @ weights
            all_oos_returns.append(oos_returns)

            if verbose:
                cum_ret = (1 + np.concatenate(all_oos_returns)).prod() - 1
                print(
                    f'Step {step:>3d}  |  train [{t - len(train_data)}:{t}]  |  '
                    f'OOS [{t}:{oos_end}]  |  cumulative: {cum_ret:+.4f}'
                )

            t += step_size
            step += 1

        return BacktestResult(
            portfolio_returns=np.concatenate(all_oos_returns),
            rebalance_points=rebalance_points,
            chosen_weights=chosen_weights,
            evolution_results=evolution_results,
        )
