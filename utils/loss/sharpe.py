import numpy as np

from utils.loss.template import Loss
from utils.enums.data import Data

class Sharpe(Loss):

    def evaluate(self, weights: np.array, data_package: dict):
        returns = data_package[Data.Returns]
        port_returns = returns @ weights.T
        return port_returns.mean(axis=0) / port_returns.std(axis=0)

    def data_needed(self) -> list:
        return [Data.Returns]
