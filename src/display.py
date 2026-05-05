import matplotlib.pyplot as plt
from perceptron import Perceptron
from pandas import DataFrame
import numpy as np


def display(p: Perceptron, points: DataFrame):
    fig, axes = plt.subplots(1, 1)
    xs: list[float] = []
    ys: list[float] = []
    labels: list[int] = []
    x1_min = 1 << 32
    x1_max = -1 << 32
    for (
        x1,
        x2,
        label,
    ) in zip(points["x1"], points["x2"], points["label"]):
        x1_min = x1 if x1 < x1_min else x1_min
        x1_max = x1 if x1 > x1_max else x1_max
        xs.append(x1)
        ys.append(x2)
        labels.append(label)
    axes.scatter(xs, ys, c=labels)

    # graph perceptron line, here we have to convert to standard to point slope (y=mx+b) form
    # NOTE: We should probably compute x interval here...
    xss = np.linspace(x1_min, x1_max+1, 100)
    yss = -p.weights[0] / p.weights[1] * xss - (p.bias / p.weights[1])

    axes.plot(xss, yss)
    plt.show()
