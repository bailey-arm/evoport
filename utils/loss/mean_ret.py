import numpy as np
import pandas as pd 

from utils.loss.template import Loss

class MeanRet(Loss):

    def evaluate_single(self, weights: np.array, data_package: dict):

        pass

    def data_needed(self) -> list:
        
        pass 

if __name__ == "__main__":
    data = pd.read_csv('/Users/baileyarm/evoport/data/prices.csv')
