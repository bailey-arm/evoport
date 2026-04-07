import numpy as np 
import pandas as pd 
from abc import ABC, abstractmethod

class Loss(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def evaluate(self, weights: np.array, data_package: dict):
        pass

    @abstractmethod
    def data_needed(self) -> list:
        pass
    
    @abstractmethod
    def to_minimise(self) -> bool:
        pass 

class ListLoss:
    def __init__(self, losses: list[Loss]) -> None:
        self.losses = losses
        self.load_minimax_structure()

    def evaluate(self, weights: np.array, data_package: dict) -> dict:
        loss_values = {}
        for loss in self.losses:
            loss_values[loss.name] = loss.evaluate(weights, data_package)
        return loss_values  
    
    def data_needed(self) -> list:
        data_needed = []
        for loss in self.losses:
            data_needed += loss.data_needed()
        return list(set(data_needed))
    
    def load_minimax_structure(self) -> None:
        losses = self.losses
        structure = []
        for loss in losses:
            structure.append(loss.to_minimise)
        self.minimax_structure = structure