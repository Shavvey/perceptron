import numpy as np
import numpy.typing as npt
import pandas as pd


class Perceptron:
    bias: float = 0.00
    num_weights: int = 0
    weights: npt.NDArray[np.float64]

    def __init__(self, bias, weights: npt.NDArray[np.float64]):
        self.num_weights = len(weights)
        self.bias = bias

    @staticmethod
    def init_random(
        num_weights: int, low: float = 0, high: float = 5.00
    ) -> "Perceptron":
        # init with random rng
        rng = np.random.default_rng()
        # sample from a uniform distribution
        rand_values = rng.normal(low - high / 2, scale=2, size=(num_weights + 1,))
        bias = rand_values[0]
        weights = rand_values[1:]
        return Perceptron(bias, weights)

    def train(self, points: pd.DataFrame):
        pass
