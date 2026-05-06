from perceptron import Perceptron
import pandas as pd
import numpy.typing as npt
import math as m


class Layer:
    num_neurons = 0
    neurons: list[Perceptron]

    def __init__(self, neurons: list[Perceptron]):
        self.num_neurons = len(neurons)
        self.neurons = neurons


def softmax(outputs: list[float]) -> list[float]:
    sum: float = 0
    for out in outputs:
        sum += m.exp(out)
    return [m.exp(out) / sum for out in outputs]


# MultiLayer Perceptron tailored to predict IRIS data
class MultiLayerPerceptron:
    hidden_layer: Layer
    output: list[float] = []
    features: list[str] = []

    def __init__(self, layer: Layer):
        self.hidden_layer = layer

    def train(self, data: pd.DataFrame, features: list[str], feature_col: str):
        for i, neuron in enumerate(self.hidden_layer.neurons):
            dataset = data.copy()
            dataset[dataset[feature_col] == features[i]] = 1
            dataset[dataset[feature_col] != features[i]] = -1
            neuron.train(dataset.to_numpy())

    def test(
        self, data: pd.DataFrame, features: list[str], feature_col: str
    ) -> list[int]:
        preds: list[int] = []
        for i, _ in enumerate(features):
            dataset = data.copy()
            dataset[dataset[feature_col] == features[i]] = 1
            dataset[dataset[feature_col] != features[i]] = -1
            for row in dataset.iterrows():
                activations = []
                for neuron in self.hidden_layer.neurons:
                    a = neuron.bias
                    for j, x in enumerate(row[:-1]):
                        a += neuron.weights[j] * x
                    activations.append(a)
                outputs = softmax(activations)
                max_idx = outputs.index((max(outputs)))
                preds.append(-1) if max_idx != i else preds.append(1)
        return preds
