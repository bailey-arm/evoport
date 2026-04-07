import numpy as np

from utils.populations.population_template import Population, DEFAULT_POPULATION_SIZE

class LongOnlyBC(Population):
    def __init__(self, problem_dimension: int, population_size: int = DEFAULT_POPULATION_SIZE, budget: float = 1):
        super().__init__(problem_dimension, population_size)
        self.budget = budget
        self.initialise()
    
    def initialise(self):
        w = np.random.random(size = (self.population_size, self.problem_dimension))
        w /= w.sum(axis=1, keepdims=True)
        w *= self.budget
        self.initial_weights = w

class LongOnlyBR(Population):
    def __init__(self, problem_dimension: int, population_size: int = DEFAULT_POPULATION_SIZE, budget_max: float = 1):
        super().__init__(problem_dimension, population_size)
        self.budget_max = budget_max
        self.initialise()
    
    def initialise(self):
        budgets = np.random.random(size = (self.population_size)) * self.budget_max
        w = np.random.random(size = (self.population_size, self.problem_dimension))
        w /= w.sum(axis=1, keepdims=True)
        for dim in range(self.problem_dimension):
            w[:,dim] = np.multiply(w[:,dim], budgets)
        self.initial_weights = w