import numpy as np

from utils.loss.template import Loss
from utils.enums.data import Data

class WinLoss(Loss):

    def evaluate(self, weights: np.array, data_package: dict):
        returns = data_package[Data.Returns]
        port_returns = returns @ weights.T
        wins = np.where(port_returns > 0, port_returns, 0)
        losses = np.where(port_returns < 0, port_returns, 0)
        n_wins = (port_returns > 0).sum(axis=0)
        n_losses = (port_returns < 0).sum(axis=0)
        mean_win = wins.sum(axis=0) / np.maximum(n_wins, 1)
        mean_loss = np.abs(losses.sum(axis=0)) / np.maximum(n_losses, 1)
        return np.where(mean_loss == 0, np.inf, mean_win / mean_loss)

    def data_needed(self) -> list:
        return [Data.Returns]
