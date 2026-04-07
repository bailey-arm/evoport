import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

DEFAULT_POPULATION_SIZE = 1_000

class Population(ABC):
    def __init__(self, problem_dimension: int, population_size: int = DEFAULT_POPULATION_SIZE):
        self.problem_dimension = problem_dimension
        self.population_size = population_size

    @abstractmethod
    def initialise(self):
        pass

    @abstractmethod
    def project(self, weights: np.ndarray) -> np.ndarray:
        """Project arbitrary weights back onto the feasible set."""
        pass

    def breed(self, parents: np.ndarray, n_children: int, noise_scale: float = 0.05) -> np.ndarray:
        """Spawn n_children by perturbing randomly selected parents, then projecting to feasibility."""
        indices = np.random.randint(0, len(parents), size=n_children)
        children = parents[indices].copy()
        noise = np.random.randn(n_children, self.problem_dimension) * noise_scale
        children += noise
        return self.project(children)
