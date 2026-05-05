import numpy as np
import numpy.typing as npt
import pandas as pd


class Perceptron:
    bias: float = 0.00
    num_weights: int = 0
    weights: npt.NDArray[np.float64]

    def __init__(self, num_weights: int = 2):
        self.num_weights = num_weights
        self.weights = np.array([0 for _ in range(self.num_weights)], dtype=np.float64)
        self.bias = 0

    def __str__(self) -> str:
        return f"[Bias: {self.bias}, Weights: {self.weights}]"

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
        p = Perceptron()
        p.bias = bias
        p.weights = weights
        return p

    def activation(self) -> float:
        # Compute the weighted activation value of the percetron, z
        z = 0
        for i in self.weights:
            z += i
        z += self.bias
        return z

    # Compute one iteration of the training algorithm
    def train(self, points: npt.NDArray):
        for point in points:
            # Compute activation
            activation = self.activation()
            y = point[-1]
            if y * activation <= 0:
                # update weights and bias
                for i, x in enumerate(point[:-1]):
                    self.weights[i] += y * x
                    self.bias += y

    def test(self, point: npt.NDArray) -> list[int]:
        SUCCESS, FAILURE = 1, -1
        results: list[int] = []
        a = self.bias
        # Compute acitvation from spatial x_i spatial component
        for i, x in enumerate(point):
            a = x * self.weights[i]
        results.append(SUCCESS) if a > 0.00 else results.append(FAILURE)
        return results
