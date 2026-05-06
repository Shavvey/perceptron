from perceptron import Perceptron
import pandas as pd
import numpy.typing as npt
import math as m
from collections import defaultdict


class Layer:
    num_neurons = 0
    neurons: list[Perceptron]

    def __init__(self, neurons: list[Perceptron]):
        self.num_neurons = len(neurons)
        self.neurons = neurons


def softmax(outputs: list[float]) -> list[float]:
    sum: float = 0
    for out in outputs:
        print(m.exp(out))
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
        for i, feature in enumerate(features):
            dataset = data.copy()
            mapping = defaultdict(lambda: -1, {feature: 1})
            dataset[feature_col] = dataset[feature_col].map(mapping)
            self.hidden_layer.neurons[i].train(dataset.to_numpy())

    def __str__(self) -> str:
        s = ""
        for neuron in self.hidden_layer.neurons:
            s += "{ " + str(neuron) + " }\n"
        return s

    def test(self, data: pd.DataFrame, idx_feature_map: dict) -> list[int]:
        preds: list[int] = []
        for _, row in data.iterrows():
            activations = []
            for neuron in self.hidden_layer.neurons:
                a = neuron.bias
                for j, x in enumerate(row[:-1]):
                    a += neuron.weights[j] * x
                activations.append(a)
            max_idx = activations.index((max(activations)))
            preds.append(-1) if max_idx != idx_feature_map[row[-1]] else preds.append(1)
        return preds
