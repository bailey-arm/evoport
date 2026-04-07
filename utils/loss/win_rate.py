import numpy as np

from utils.loss.template import Loss
from utils.enums.data import Data

class WinRate(Loss):
    def __init__(self, name, to_minimise = False):
        super().__init__(name, to_minimise)

    def evaluate(self, weights: np.array, data_package: dict):
        returns = data_package[Data.Returns]
        port_returns = returns @ weights.T
        return (port_returns > 0).mean(axis=0)

    def data_needed(self) -> list:
        return [Data.Returns]
