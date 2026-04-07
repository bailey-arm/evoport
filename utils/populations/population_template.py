import numpy as np 
import pandas as pd 
from abc import ABC, abstractmethod

DEFAULT_POPULATION_SIZE = 1_000

class Population:
    def __init__(self, problem_dimension: int, population_size: int = DEFAULT_POPULATION_SIZE):
        self.problem_dimension = problem_dimension
        self.population_size = population_size

    def initialise(self):
        pass 
    