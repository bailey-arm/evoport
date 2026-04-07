import numpy as np

from utils.loss.template import Loss
from utils.enums.data import Data

class VaR(Loss):

    def __init__(self, name: str, alpha: float = 0.05):
        super().__init__(name)
        self.alpha = alpha

    def evaluate(self, weights: np.array, data_package: dict):
        returns = data_package[Data.Returns]
        port_returns = returns @ weights.T
        return np.percentile(port_returns, self.alpha * 100, axis=0)

    def data_needed(self) -> list:
        return [Data.Returns]
