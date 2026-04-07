import numpy as np

from utils.loss.template import Loss
from utils.enums.data import Data

class CVaR(Loss):

    def __init__(self, name: str, alpha: float = 0.05, to_minimize: bool = True):
        super().__init__(
            name = name, 
            to_minimise = self.to_minimise
            )
        self.alpha = alpha

    def evaluate(self, weights: np.array, data_package: dict):
        returns = data_package[Data.Returns]
        port_returns = returns @ weights.T
        var_threshold = np.percentile(port_returns, self.alpha * 100, axis=0)
        mask = port_returns <= var_threshold
        results = np.where(
            mask.sum(axis=0) > 0,
            np.where(mask, port_returns, 0).sum(axis=0) / np.maximum(mask.sum(axis=0), 1),
            var_threshold
        )
        return results

    def data_needed(self) -> list:
        return [Data.Returns]
