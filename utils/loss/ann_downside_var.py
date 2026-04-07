import numpy as np

from utils.loss.template import Loss
from utils.enums.data import Data

class AnnDownsideVar(Loss):

    def __init__(self, name, to_minimise = True):
        super().__init__(name, to_minimise)

    def evaluate(self, weights: np.array, data_package: dict):
        returns = data_package[Data.Returns]
        port_returns = returns @ weights.T
        downside = np.minimum(port_returns, 0)
        return (downside ** 2).mean(axis=0) * 252

    def data_needed(self) -> list:
        return [Data.Returns]
    
if __name__ == "__main__":
    loss = AnnDownsideVar('DownsideVarTest')
    print(loss.to_minimise)