class Perceptron:
    bias: float = 0.00
    num_weights: int = 0
    weights: list[float] = []

    def __init__(self, bias, weights: list[float]):
        self.num_weights = len(weights)
        self.bias = bias
