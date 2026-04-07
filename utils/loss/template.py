import numpy as np 
import pandas as pd 
from abc import ABC, abstractmethod

class Loss(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def evaluate(self, weights: np.array, data_package: dict):
        pass
    