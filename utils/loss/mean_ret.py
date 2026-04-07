import numpy as np
import pandas as pd 

from utils.loss.template import Loss
from utils.enums.data import Data

class MeanRet(Loss):

    def evaluate(self, weights: np.array, data_package: dict):
        mean_ret = data_package[Data.MeanRet]
        return (mean_ret * weights).sum(axis = 1)

    def data_needed(self) -> list:
        return [Data.MeanRet]

if __name__ == "__main__":
    data = pd.read_csv('/Users/baileyarm/evoport/data/prices.csv')
