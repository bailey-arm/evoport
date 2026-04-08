from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.enums.data import Data
from utils.populations.population_template import Population
from utils.loss.template import ListLoss


@dataclass
class EvolutionResult:
    weights: np.ndarray
    objectives: pd.DataFrame
    frontier_idx: np.ndarray
    history: list[dict]
    returns: np.ndarray | None = None

    @property
    def frontier_weights(self) -> np.ndarray:
        return self.weights[self.frontier_idx]

    @property
    def frontier_objectives(self) -> pd.DataFrame:
        return self.objectives.iloc[self.frontier_idx]


class Evolution:
    def __init__(
        self,
        population: Population,
        list_loss: ListLoss,
        domination_fn: callable,
        domination_kwargs: dict | None = None,
    ):
        self.population = population
        self.list_loss = list_loss
        self.domination_fn = domination_fn
        self.domination_kwargs = domination_kwargs or {}
        self.higher_is_better = [not m for m in list_loss.minimax_structure]

    def run(
        self,
        data_package: dict,
        n_generations: int,
        verbose: bool = False,
    ) -> EvolutionResult:
        pop = self.population.initial_weights.copy()
        cols = [loss.name for loss in self.list_loss.losses]
        history = []

        for gen in range(n_generations):
            res = self.list_loss.evaluate(pop, data_package)
            df_gen = pd.DataFrame(res)
            df_front = self.domination_fn(
                df=df_gen,
                cols=cols,
                higher_is_better=self.higher_is_better,
                **self.domination_kwargs,
            )

            history.append({
                'gen': gen,
                'objectives': df_gen,
                'frontier_idx': df_front.index.to_numpy(),
                'frontier_size': len(df_front),
            })

            if verbose:
                medians = ' | '.join(
                    f'{c}: {df_gen[c].median():.4f}' for c in cols
                )
                print(
                    f'Gen {gen:>2d}  |  frontier: {len(df_front):>6,}  |  {medians}'
                )

            survivors = pop[df_front.index]
            n_children = len(pop) - len(survivors)
            children = self.population.breed(survivors, n_children)
            pop = np.vstack([survivors, children])

        # Final evaluation
        res = self.list_loss.evaluate(pop, data_package)
        df_final = pd.DataFrame(res)
        df_front_final = self.domination_fn(
            df=df_final,
            cols=cols,
            higher_is_better=self.higher_is_better,
            **self.domination_kwargs,
        )

        return EvolutionResult(
            weights=pop,
            objectives=df_final,
            frontier_idx=df_front_final.index.to_numpy(),
            history=history,
            returns=data_package.get(Data.Returns),
        )
