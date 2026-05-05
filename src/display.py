import matplotlib.pyplot as plt
from perceptron import Perceptron
from pandas import DataFrame


def display(p: Perceptron, points: DataFrame):
    fig, axes = plt.subplots(1, 1)
    xs: list[float] = []
    ys: list[float] = []
    labels: list[int] = []
    for x1, x2, label, in zip(points["x1"], points["x2"], points["label"]):
        xs.append(x1)
        ys.append(x2)
        labels.append(label)
    axes.scatter(xs, ys, c=labels)
    plt.show()
