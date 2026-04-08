import numpy as np

from utils.loss.template import Loss
from utils.enums.data import Data

class MaxDD(Loss):
    def __init__(self, name, to_minimise: bool = False): #False as we want it to be as close to 0 as possible (DD <= 0) 
        super().__init__(name, to_minimise)

    def evaluate(self, weights: np.array, data_package: dict):
        returns = data_package[Data.Returns]
        port_returns = returns @ weights.T
        cum_returns = (1 + port_returns).cumprod(axis=0)
        running_max = np.maximum.accumulate(cum_returns, axis=0)
        drawdowns = (cum_returns - running_max) / running_max
        return drawdowns.min(axis=0)

    def data_needed(self) -> list:
        return [Data.Returns]
